# Benchmarks Guide

Welcome to our benchmarks documentation! Here, you'll find all the necessary information to run benchmarks for
evaluating performance. This guide is structured to help you get started quickly and efficiently.

## Prerequisites

Before you can run any benchmarks, there are a few steps you need to complete. These steps ensure that your environment
is correctly set up with all the required dependencies.

1. **Install Dependencies**: To install the necessary dependencies, use the following command. This will set up your
   environment with everything you need to run the benchmarks:

 ```bash
 poetry install --with=benchmark
 ```

## Running Benchmarks

The benchmarks are designed to be straightforward to run, allowing you to evaluate performance with minimal setup.

### Location

All benchmark scripts are located in the `benchmarks` directory. Within this directory, the `main.py` script is your
primary tool for running benchmarks.

### Running All Benchmarks

To execute all available default benchmarks, navigate to the `benchmarks` directory and use the following command:

```bash
python main.py
```

This command will systematically run through all default benchmarks and output the results.

### Running Specific Benchmarks

If you wish to run a specific benchmark, perhaps to focus on a particular module or functionality, you can do so by
specifying the benchmark module. For example, to run benchmarks specifically for the BelNet operator, use the following
command:

```bash
python main.py -m operator=belnet
```

This flexibility allows you to isolate and test specific components of your project efficiently.

## Additional Tips

- Benchmark Options: For more advanced usage and additional command-line options, you can use the -h or --help flag with
  the main.py script to display all available options.
- Custom Benchmarks: If you're interested in contributing or creating custom benchmarks, you need to follow these steps:
    - Define a benchmark instance for your dataset and add your desired metrics,
    - Add a custom yaml file to the `benchmark/configs/benchmark` dir,
    - Integrate this file with the top level configuration in `benchmark/config.yaml`.
