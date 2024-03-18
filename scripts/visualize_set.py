import pathlib
from typing import (
    List,
    Tuple,
)

import torch.nn as nn
from continuity.data import (
    OperatorDataset,
)

from nos.data import (
    TLDatasetCompact,
)
from nos.operators import (
    DeepDotOperator,
    MeanStackNeuralOperator,
    deserialize,
)
from nos.plots import (
    visualize_worst_mean_median_best,
)


def visualize_multirun(run_dir: pathlib.Path, datasets: List[Tuple[str, OperatorDataset]]):
    for name, dataset in datasets:
        for run_path in run_dir.glob("*"):
            if run_path.is_file():
                continue
            models_dir = run_path.joinpath("models")
            for model_path in models_dir.glob("*"):
                plot_dir = model_path.joinpath("plots", name)
                plot_dir.mkdir(parents=True, exist_ok=True)

                if "MeanStackNeuralOperator" in model_path.name:
                    base_class = MeanStackNeuralOperator
                elif "DeepDotOperator" in model_path.name:
                    base_class = DeepDotOperator
                else:
                    raise ValueError("Unknown base class")

                operator = deserialize(model_dir=model_path, model_base_class=base_class)
                visualize_worst_mean_median_best(
                    operator=operator, dataset=dataset, criterion=nn.MSELoss(), out_dir=plot_dir
                )


def visualize_single():
    # operator
    model_name = "deep_dot_small"
    model_path = pathlib.Path.cwd().joinpath("models", model_name)
    operator = deserialize(model_dir=model_path, model_base_class=DeepDotOperator)

    # dataset
    data_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss")
    dataset = TLDatasetCompact(data_path)

    # out dir
    out_dir = pathlib.Path.cwd().joinpath("out", model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    visualize_worst_mean_median_best(operator=operator, dataset=dataset, criterion=nn.MSELoss(), out_dir=out_dir)


if __name__ == "__main__":
    run_dir = pathlib.Path.cwd().joinpath("multirun", "2024-03-18", "08-27-35")

    test_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss")
    test_set = TLDatasetCompact(test_path)

    train_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss")
    train_set = TLDatasetCompact(train_path)

    visualize_multirun(run_dir=run_dir, datasets=[("test", test_set), ("train", train_set)])
