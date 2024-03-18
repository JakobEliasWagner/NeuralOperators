import pathlib
from typing import (
    List,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from continuity.data import (
    OperatorDataset,
)
from tqdm import (
    tqdm,
)

from nos.operators import (
    NeuralOperator,
)

from .data import (
    ModelData,
    MultiRunData,
    RunData,
)
from .utils import (
    eval_operator,
)


def plot_multirun_transmission_loss(multi_run: MultiRunData, dataset: OperatorDataset, out_dir: pathlib.Path):
    out_dir = out_dir.joinpath("comparison")

    pbar = tqdm(multi_run.runs, leave=False, position=1)
    for run in pbar:
        pbar.set_postfix_str(f"... processing transmission loss for {run.name} ...")
        run_out = out_dir.joinpath(run.name)
        plot_run_transmission_loss(run, dataset, run_out)


def plot_run_transmission_loss(run: RunData, dataset: OperatorDataset, out_dir: pathlib.Path):
    run_out = out_dir.joinpath(run.name)
    pbar = tqdm(run.models, leave=False, position=2)
    for model in pbar:
        pbar.set_postfix_str(f"... processing transmission loss for {model.name} ...")
        model_out = run_out.joinpath(model.name)
        plot_model_transmission_loss(run, model, dataset, model_out)


def plot_model_transmission_loss(run: RunData, model: ModelData, dataset: OperatorDataset, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    op = model.operator

    df = eval_operator(op, dataset, [torch.nn.MSELoss()])

    # set axis names
    df["Radius [mm]"] = df["u"].apply(lambda x: x[0] * 1e3)
    df["Inner Radius [mm]"] = df["u"].apply(lambda x: x[1] * 1e3)
    df["Gap Width [mm]"] = df["u"].apply(lambda x: x[2] * 1e3)

    plot_transmission_loss_best(df, op, dataset, out_dir)
    plot_transmission_loss_mean(df, op, dataset, out_dir)
    plot_transmission_loss_median(df, op, dataset, out_dir)
    plot_transmission_loss_worst(df, op, dataset, out_dir)

    plot_transmission_loss_hist(df, out_dir)
    plot_transmission_loss_qtile(df, op, dataset, out_dir)

    if run.training_config["val_size"] + run.training_config["train_size"] == len(dataset):
        val_indices = run.training_config["val_indices"]
    else:
        val_indices = None
    plot_transmission_loss_tile(df, out_dir, val_indices)


def _get_single_df(operator: NeuralOperator, dataset: OperatorDataset, idx: int) -> pd.DataFrame:
    x, u, y, v = dataset[idx]
    v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))

    data = {"y": y, "v": v, "v_out": v_out}
    for idf, tensor in data.items():
        if idf not in dataset.transform:
            continue
        data[idf] = dataset.transform[idf].undo(tensor)
    if "v" in dataset.transform:
        data["v_out"] = dataset.transform["v"].undo(data["v_out"])

    for idf, tensor in data.items():
        data[idf] = tensor.squeeze().detach().numpy()

    return pd.DataFrame(data)


def plot_transmission_loss_best(
    df: pd.DataFrame, operator: NeuralOperator, dataset: OperatorDataset, out_dir: pathlib.Path
):
    out_file = out_dir.joinpath("best.png")
    best_idx = df["MSELoss"].idxmin()
    df_single = _get_single_df(operator, dataset, best_idx)
    plot_single(df_single, out_file)


def plot_transmission_loss_median(
    df: pd.DataFrame, operator: NeuralOperator, dataset: OperatorDataset, out_dir: pathlib.Path
):
    median_idx = df[df["MSELoss"] == df["MSELoss"].quantile(interpolation="nearest")].index[0]
    out_file = out_dir.joinpath("median.png")
    df_single = _get_single_df(operator, dataset, median_idx)
    plot_single(df_single, out_file)


def plot_transmission_loss_mean(
    df: pd.DataFrame, operator: NeuralOperator, dataset: OperatorDataset, out_dir: pathlib.Path
):
    mean_idx = df.iloc[(df["MSELoss"] - df["MSELoss"].mean()).abs().argsort()[:1]].index[0]
    out_file = out_dir.joinpath("mean.png")
    df_single = _get_single_df(operator, dataset, mean_idx)
    plot_single(df_single, out_file)


def plot_transmission_loss_worst(
    df: pd.DataFrame, operator: NeuralOperator, dataset: OperatorDataset, out_dir: pathlib.Path
):
    worst_idx = df["MSELoss"].idxmax()
    out_file = out_dir.joinpath("worst.png")
    df_single = _get_single_df(operator, dataset, worst_idx)
    plot_single(df_single, out_file)


def _get_param_string(u) -> str:
    u_corr = u.squeeze().detach().numpy() * 1e3
    return f"R={u_corr[0]:.2f}mm, R_i={u_corr[1]:.2f}mm, b={u_corr[2]:.2f}mm"


def plot_transmission_loss_hist(df: pd.DataFrame, out_dir: pathlib.Path):
    out_file = out_dir.joinpath("hist.png")

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="MSELoss", ax=ax, bins=31)
    ax.axvline(x=df["MSELoss"].median(), c="k", ls="-", lw=2.5, label="median")
    ax.axvline(x=df["MSELoss"].mean(), c="orange", ls="--", lw=2.5, label="mean")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def plot_transmission_loss_qtile(
    df: pd.DataFrame, operator: NeuralOperator, dataset: OperatorDataset, out_dir: pathlib.Path
):
    out_file = out_dir.joinpath("qtile.png")
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), sharex=True, sharey=True)
    quantiles = torch.linspace(0.0, 1.0, 9)
    for i, quantile in zip(range(9), quantiles):
        row, col = i // 3, i % 3
        q_idx = df[df["MSELoss"] == df["MSELoss"].quantile(q=quantile, interpolation="nearest")].index[0]
        df_single = _get_single_df(operator, dataset, q_idx)
        plot_ax(ax=axs[row, col], df=df_single)
        axs[row, col].set_title(f"{i}/8 quantile")
        axs[row, col].set_ylabel("Frequency [Hz]")
        axs[row, col].set_xlabel("TL [dB]")

    # only use a single set of labels
    lines_labels = [ax.get_legend_handles_labels() for ax in axs.flatten()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    by_label = dict(zip(labels, lines))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def plot_transmission_loss_tile(df: pd.DataFrame, out_dir: pathlib.Path, val_indices: List[int] = None):
    out_file = out_dir.joinpath("tile.png")

    # create quantiles
    df["quantile"] = (df["MSELoss"] - min(df["MSELoss"])) / (max(df["MSELoss"]) - min(df["MSELoss"]))
    df["quantile"] = (df["quantile"] * 10).round() / 10

    # plot properties
    config = {
        "data": df,
        "vars": ["MSELoss", "Radius [mm]", "Inner Radius [mm]", "Gap Width [mm]"],
        "hue": "quantile",
        "palette": "coolwarm",
        "diag_kind": "kde",
        "diag_kws": {"multiple": "stack"},
    }
    if val_indices is not None:
        df["Is Validation"] = df.index.isin(val_indices)
        config["vars"].append("Is Validation")
        config["plot_kws"] = {
            "alpha": 0.5,
            "style": df["Is Validation"],
            "markers": {True: "D", False: "o"},
        }

    sns.pairplot(**config)
    plt.savefig(out_file)
    plt.close()


def plot_single(df: pd.DataFrame, out_file: pathlib.Path):
    fig, ax = plt.subplots()
    plot_ax(df, ax)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def plot_ax(df: pd.DataFrame, ax):
    sns.lineplot(data=df, x="v_out", y="y", ax=ax, label="prediction", orient="y")
    sns.lineplot(data=df, x="v", y="y", ax=ax, label="FEM", alpha=0.5, orient="y")
