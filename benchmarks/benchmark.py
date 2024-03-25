import json
import pathlib
from collections import (
    defaultdict,
)

import hydra
from hydra.core.hydra_config import (
    HydraConfig,
)
from omegaconf import (
    DictConfig,
    OmegaConf,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    """Main entry point for running benchmarks.

    Example usage:
        ```
        python run.py
        ```

    Args:
        cfg: Configuration.
    """
    # ----- SETUP BENCHMARK -----
    benchmark = hydra.utils.instantiate(cfg.benchmark)
    operator = hydra.utils.instantiate(cfg.operator, shapes=benchmark.train_set.shapes, _convert_="object")
    optimizer = hydra.utils.instantiate(cfg.optim, params=operator.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        operator=operator,
        max_epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
    )

    # ----- BUILD OUT DIR -----
    sweep_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)

    models_dir = sweep_dir.joinpath("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    with open(models_dir.joinpath("choices.json"), "w") as file_handle:
        json.dump(choices, file_handle)

    # ----- TRAIN OPERATOR -----
    operator = trainer(benchmark.train_set)

    # ----- RESULTS -----
    benchmark_name = choices["benchmark"]
    operator_name = choices["operator"]
    benchmark_dir = sweep_dir.parent
    json_file = benchmark_dir.joinpath("results.json")

    # Evaluate on train/test set
    results = defaultdict(lambda: defaultdict(dict))
    for mt in benchmark.metrics:
        train_result = mt(operator, benchmark.train_set)
        test_result = mt(operator, benchmark.test_set)
        result = {"train": train_result, "test": test_result}
        results[benchmark_name][operator_name][str(mt)] = result

    # Update and save results
    old_results = load_benchmark_data(json_file)
    results = update_benchmark_data(old_results, results)
    save_benchmark_data(json_file, results)


def load_benchmark_data(file_path: pathlib.Path) -> dict:
    """Load benchmark data from a JSON file."""
    if file_path.exists():
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        return {}  # Return an empty dict if file doesn't exist


def recursive_dict_merge(dict_a: dict, dict_b: dict) -> dict:
    """Recursively merges two dictionaries.

    Recursively merges two dictionaries. If one value in the dictionary is a leaf of the tree it combines both values
    into a list with the value of dict_a first.

    Args:
        dict_a: Dictionary to merge with dict_b.
        dict_b: Dictionary to merge with dict_a.

    Returns:
        Merged dictionary with values from both dictionaries.
    """
    for key, value in dict_b.items():
        if key not in dict_a:
            dict_a[key] = value
            continue

        # key in dict
        if not isinstance(value, dict) or not isinstance(dict_a[key], dict):
            # one dict cannot be traversed recursively anymore
            dict_a[key] = [dict_a[key], value]
            continue

        # value in both dicts is still a dict
        dict_a[key] = recursive_dict_merge(dict_a[key], value)

    return dict_a


def update_benchmark_data(existing_data: dict, new_data: dict) -> dict:
    """Update the existing benchmark data with new data."""
    return recursive_dict_merge(existing_data, new_data)


def save_benchmark_data(file_path: pathlib.Path, data: dict):
    """Save the updated benchmark data to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    run()
