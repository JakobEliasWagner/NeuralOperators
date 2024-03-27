import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn
from continuity.data import (
    OperatorDataset,
)
from tqdm import (
    tqdm,
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

    df["size"] = df["Architecture"].apply(lambda s: s.split("_")[-1])
    df["Architecture"] = df["Architecture"].apply(lambda s: "-".join(s.split("_")[:-1]))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(df, x="Architecture", y="MSELoss", hue="size", ax=ax)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("mse.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(df, x="Architecture", y="L1Loss", hue="Checkpoint", ax=ax)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("l1.png"))
    plt.close(fig)
