#!/bin/bash

# Define paths and container name
local_directory="../../"
container_directory="/home/fenics/shared"
container_name="fenicsx_complex"
executable_path="/usr/local/bin/dolfinx-complex-mode"
python_script="file_name.py"
local_output_directory="../../../out"
container_output_directory="/home/fenics/out"
tmp_dir="./tmp/"


# Check if the specified container is running
RUNNING=$(docker inspect --format="{{ .State.Running }}" $container_name 2> /dev/null)

# start container if it is not running yet
if [ "$RUNNING" == "false" ]; then
  docker start $$container_name
fi

# set python root dir
export PYTHONPATH="${PYTHONPATH}:$container_directory"

# Ensure the temporary directory exists and is empty
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

# Copy the directory excluding venv directories
rsync -av --exclude 'venv/' "$local_directory/" "$tmp_dir/"

# Copy directory to container
docker cp "$tmp_dir/." "$container_name":"$container_directory"

# Clean up temporary directory
rm -rf "$tmp_dir"

# Execute script inside the container
docker exec -it "$container_name" /bin/bash -c "source $executable_path; python3 $container_directory/$python_script"

# Copy output directory back to host
docker cp "$container_name":"$container_output_directory" "$local_output_directory"
