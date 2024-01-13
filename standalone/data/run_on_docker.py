import argparse
import pathlib
import shutil

import docker

cwd_path = pathlib.Path.cwd()
# Setup argument parser
parser = argparse.ArgumentParser(
    description="Copy files to/from a Docker container and execute python script inside it."
)
parser.add_argument("--input_parameter_file", required=True)
parser.add_argument("--input_code_file", required=True)
parser.add_argument("--src_dir", required=False, help="Path to the src directory", default=cwd_path.joinpath("src"))
parser.add_argument(
    "--out_dir", required=False, help="Path to the host output directory", default=cwd_path.joinpath("out")
)
parser.add_argument("--container_name", required=False, help="Name of the FeniCS docker container", default="dolfinx")
parser.add_argument("--working_dir_container", required=False, help="Base directory in container", default="/home/")
parser.add_argument(
    "--is_complex", action="store_true", required=False, help="Toggle for complex valued execution", default=True
)
parser.add_argument(
    "--cleanup_docker",
    required=False,
    help="Toggle for cleaning up all working directories on the docker container",
    default=False,
)

# Parse arguments
args = parser.parse_args()

# Assign variables from arguments
#   host
description_file = cwd_path.joinpath(args.input_parameter_file)
py_file = cwd_path.joinpath(args.input_code_file)
host_src_dir = cwd_path.joinpath(args.src_dir)
host_output_dir = cwd_path.joinpath(args.out_dir)
#   target
container_name = args.container_name
container_working_dir = pathlib.Path(args.working_dir_container)
container_src_dir = "src"
container_out_dir = "out"
is_complex = args.is_complex
cleanup = args.cleanup_docker

# Complex build toggle
builds = {True: "/usr/local/bin/dolfinx-complex-mode", False: "/usr/local/bin/dolfinx-real-mode"}

client = docker.from_env()


def copy_to_container(container, host_entry: pathlib.Path, container_dir: pathlib.Path) -> None:
    """Copies host dir to container dir

    :param container:
    :param host_entry:
    :param container_dir:
    :return:
    """
    if host_entry.is_dir():
        tar_stream = shutil.make_archive("tmp", "tar", host_entry)
    elif host_entry.is_file():
        tar_stream = shutil.make_archive("tmp", "tar", host_entry.parent, host_entry.name)
    else:
        raise ValueError(f"Unknown host file system entry: {host_entry}!")
    with open(tar_stream, "rb") as file_obj:
        container.put_archive(container_dir, file_obj)
    cwd_path.joinpath("tmp.tar").unlink()


def execute_script_in_container(container, script_path: pathlib.Path) -> None:
    """Executes script in container

    :param container: container
    :param script_path:
    :return:
    """
    ec, out = container.exec_run(
        f"bash -c 'source {builds[is_complex]}; "
        f"python3 {script_path.name} --input_file {description_file.name} --output_dir {container_out_dir}'",
        workdir=str(container_working_dir),
    )
    print(out.decode())
    if ec != 0:
        raise ValueError(f"File execution failed with code {ec}!")


def cleanup_container(container, clean_dir) -> None:
    """Deletes files in container base dir.

    :param clean_dir:
    :param container:
    :return:
    """
    ec, _ = container.exec_run(f"rm -rf {clean_dir}/*")
    if ec != 0:
        raise Exception(f"Directory cleanup failed with exit code {ec}!")


def setup_container(container, working_dir: pathlib.Path) -> None:
    """

    :param container:
    :param working_dir:
    :return:
    """
    container.exec_run(f"mkdir -p {working_dir.joinpath(container_out_dir)}")
    container.exec_run(f"mkdir -p {working_dir.joinpath(container_src_dir)}")


def copy_from_container(container, container_dir: pathlib.Path, host_dir: pathlib.Path) -> None:
    """Copies output files from container to output directory on host.

    :param container:
    :param container_dir:
    :param host_dir:
    :return:
    """
    stream, stat = container.get_archive(str(container_dir))
    with open("output.tar", "wb") as out_file:
        for chunk in stream:
            out_file.write(chunk)
    shutil.unpack_archive("output.tar", host_dir)
    cwd_path.joinpath("output.tar").unlink()


try:
    # Create or get container
    container_ = client.containers.get(container_name)

    # setup container
    if cleanup:
        cleanup_container(container_, container_working_dir)
    setup_container(container_, container_working_dir)

    # Copy relevant files to container
    copy_to_container(container_, host_src_dir, container_working_dir.joinpath(container_src_dir))
    copy_to_container(container_, description_file, container_working_dir)
    copy_to_container(container_, py_file, container_working_dir)

    # Set environment variables and execute the script
    execute_script_in_container(container_, py_file)

    # ensure proper output dir exists for host
    host_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy output files to host
    copy_from_container(container_, container_working_dir.joinpath(container_out_dir), host_output_dir)

finally:
    client.close()
