import json
import pathlib
from collections import (
    defaultdict,
)
from typing import (
    Tuple,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn
import torch.nn as nn
from continuity.data import (
    OperatorDataset,
)
from torch.utils.data import (
    DataLoader,
)
from tqdm import (
    tqdm,
)

from nos.operators import (
    DeepDotOperator,
    MeanStackNeuralOperator,
)
from nos.operators.utils import (
    deserialize,
)
from nos.plots import (
    MultiRunData,
)

from .utils import (
    eval_operator,
)


def plot_multirun_metrics(multirun: MultiRunData, dataset: OperatorDataset, out_dir: pathlib.Path):
    out_dir = out_dir.joinpath("metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    pbar = tqdm(multirun.runs, leave=False, position=1)
    for run in pbar:
        pbar.set_postfix_str("processing metrics ...")
        pbar2 = tqdm(run.models, leave=False, position=2)
        for model in pbar2:
            df_operator = eval_operator(model.operator, dataset, [torch.nn.MSELoss(), torch.nn.L1Loss()])
            df_operator["Architecture"] = run.name
            df_operator["Checkpoint"] = model.name

            df = pd.concat([df, df_operator])

    fig, ax = plt.subplots()
    sns.boxplot(df, x="Architecture", y="MSELoss", hue="Checkpoint", ax=ax)
    ax.set_yscale("log")
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("mse.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(df, x="Architecture", y="L1Loss", hue="Checkpoint", ax=ax)
    ax.set_yscale("log")
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("l1.png"))
    plt.close(fig)


def plot_all_metrics(run_dir: pathlib.Path, data_tuple: Tuple[str, OperatorDataset]):
    data_name, dataset = data_tuple
    # initialize all operators
    operators = defaultdict(dict)
    for run_path in run_dir.glob("*"):
        if run_path.is_file():
            continue
        if "plot" in run_path.name:
            continue
        model_dir = run_path.joinpath("models")

        with open(model_dir.joinpath("choices.json"), "r") as file_handle:
            choices = json.load(file_handle)

        operator_name = choices["operator"]

        for operator_dir in model_dir.glob("*_*_*"):
            if "DeepDot" in operator_dir.name:
                operator_base_class = DeepDotOperator
            elif "MeanStack" in operator_dir.name:
                operator_base_class = MeanStackNeuralOperator
            else:
                raise ValueError(f"Unknown operator in {operator_dir.name}")

            with open(operator_dir.joinpath("checkpoint.json"), "r") as file_handle:
                checkpoint = json.load(file_handle)

            epoch = checkpoint["epoch"]

            operator = deserialize(operator_dir, operator_base_class)
            operators[operator_name][epoch] = operator

    eval_loader = DataLoader(dataset, batch_size=1)
    l1_err = nn.L1Loss()
    ms_err = nn.MSELoss()

    df = pd.DataFrame()
    for operator_name, operator_snap in operators.items():
        local_performance = defaultdict(list)
        for epoch, operator in operator_snap.items():
            operator.eval()
            for x, u, y, v in eval_loader:
                out = operator(x, u, y)

                l1 = l1_err(out, v).item()
                mse = ms_err(out, v).item()

                local_performance["Architecture"].append(operator_name)
                local_performance["Epoch"].append(epoch)
                local_performance["l1"].append(l1)
                local_performance["mse"].append(mse)
        local_df = pd.DataFrame(local_performance)
        df = pd.concat([df, local_df])

    out_dir = run_dir.joinpath("plots", data_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    sns.boxplot(df, x="Architecture", y="mse", hue="Epoch", ax=ax)
    ax.set_yscale("log")
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("mse.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(df, x="Architecture", y="l1", hue="Epoch", ax=ax)
    ax.set_yscale("log")
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("l1.png"))
    plt.close(fig)
