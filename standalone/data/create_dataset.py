import argparse
import pathlib

from src.data.helmholtz import Helmholtz

# Setup argument parser
parser = argparse.ArgumentParser(description="Creates a dataset for given input file.")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--input_file", required=True)

# Parse arguments
args = parser.parse_args()

# Assign variables from arguments
in_file = pathlib.Path.cwd().joinpath(args.input_file)
out_dir = pathlib.Path.cwd().joinpath(args.output_dir)


def create_dataset(description_file: pathlib.Path, output_dir: pathlib.Path):
    data_generator = Helmholtz(description_file, output_dir)
    data_generator.run()


if __name__ == "__main__":
    create_dataset(in_file, out_dir)
