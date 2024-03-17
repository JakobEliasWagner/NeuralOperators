import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from continuity.data import (
    OperatorDataset,
)
from loguru import (
    logger,
)
from torch.utils.data import (
    DataLoader,
)

from nos.data import (
    TLDatasetCompact,
)
from nos.operators import (
    DeepDotOperator,
    NeuralOperator,
    deserialize,
)


def create_df(dataset: OperatorDataset, y, v, v_out):
    y, v, v_out = dataset.transform["y"].undo(y), dataset.transform["v"].undo(v), dataset.transform["v"].undo(v_out)
    df = pd.DataFrame(
        {
            "y": y.squeeze().detach().numpy(),
            "v": v.squeeze().detach().numpy(),
            "v_out": v_out.squeeze().detach().numpy(),
        }
    )

    return df


def get_param_string(u, dataset: OperatorDataset) -> str:
    u_corr = dataset.transform["u"].undo(u)
    u_corr = u_corr.squeeze().detach().numpy() * 1e3
    return f"R={u_corr[0]:.2f}mm, R_i={u_corr[1]:.2f}mm, b={u_corr[2]:.2f}mm"


def visualize_worst_mean_median_best(
    operator: NeuralOperator, dataset: OperatorDataset, criterion: nn.Module, out_dir: pathlib.Path
):
    out_dir = out_dir.joinpath("worst-mean-median")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 50)
    logger.info("Start Visualizing worst, mean, median, and best plots.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    operator.eval()

    losses = []
    radii = []
    inner_radii = []
    gap_widths = []
    with torch.no_grad():
        for x, u, y, v in loader:
            out = operator(x, u, y)
            loss = criterion(out, v)
            losses.append(loss.item())
            radii.append(x.squeeze().detach().numpy()[0])
            inner_radii.append(x.squeeze().detach().numpy()[1])
            gap_widths.append(x.squeeze().detach().numpy()[2])
    df = pd.DataFrame({"Loss": losses, "radius": radii, "inner_radius": inner_radii, "gap_width": gap_widths})

    logger.info(f"Best loss: {min(losses)}")
    logger.info(f"Worst loss: {max(losses)}")
    logger.info(f"Mean loss: {torch.mean(torch.tensor(losses))}")

    # ----- hist loss -----
    logger.info("Starting hist plot...")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Loss", ax=ax, bins=31)
    ax.axvline(x=df["Loss"].median(), c="k", ls="-", lw=2.5, label="median")
    ax.axvline(x=df["Loss"].mean(), c="orange", ls="--", lw=2.5, label="mean")
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("hist.png"))
    fig.clear()

    # ----- hist tile -----
    logger.info("Starting hist tile plot...")
    sns.pairplot(data=df, vars=["Loss", "radius", "inner_radius", "gap_width"])
    plt.savefig(out_dir.joinpath("hist_tile.png"))

    # ----- best -----
    logger.info("Starting best plot...")
    best_idx = df["Loss"].idxmin()
    x, u, y, v = dataset[best_idx]
    v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))
    df_single = create_df(dataset, y, v, v_out)
    fig, ax = plt.subplots()
    plot_single(ax=ax, df=df_single)
    ax.legend()
    fig.suptitle(get_param_string(u, dataset))
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("best.png"))
    fig.clear()

    # ----- median -----
    logger.info("Starting median plot...")
    median_idx = df[df["Loss"] == df["Loss"].quantile(interpolation="nearest")].index[0]
    x, u, y, v = dataset[median_idx]
    v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))
    df_single = create_df(dataset, y, v, v_out)
    fig, ax = plt.subplots()
    plot_single(ax=ax, df=df_single)
    ax.legend()
    fig.suptitle(get_param_string(u, dataset))
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("median.png"))
    fig.clear()

    # ----- mean -----
    logger.info("Starting mean plot...")
    mean_idx = df.iloc[(df["Loss"] - df["Loss"].mean()).abs().argsort()[:1]].index[0]
    x, u, y, v = dataset[mean_idx]
    v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))
    df_single = create_df(dataset, y, v, v_out)
    fig, ax = plt.subplots()
    plot_single(ax=ax, df=df_single)
    ax.legend()
    fig.suptitle(get_param_string(u, dataset))
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("mean.png"))
    fig.clear()

    # ----- worst -----
    logger.info("Starting worst plot...")
    worst_idx = df["Loss"].idxmax()
    x, u, y, v = dataset[worst_idx]
    v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))
    df_single = create_df(dataset, y, v, v_out)
    fig, ax = plt.subplots()
    plot_single(ax=ax, df=df_single)
    ax.legend()
    fig.suptitle(get_param_string(u, dataset))
    fig.tight_layout()
    plt.savefig(out_dir.joinpath("worst.png"))
    fig.clear()

    # ----- tile -----
    logger.info("Starting tile plot...")
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), sharex=True, sharey=True)
    quantiles = torch.linspace(0.0, 1.0, 9)
    for i, quantile in zip(range(9), quantiles):
        row, col = i // 3, i % 3
        q_idx = df[df["Loss"] == df["Loss"].quantile(q=quantile, interpolation="nearest")].index[0]
        x, u, y, v = dataset[q_idx]
        v_out = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0))
        df_single = create_df(dataset, y, v, v_out)
        plot_single(ax=axs[row, col], df=df_single)
        axs[row, col].set_title(f"{i}/9 quantile")
        axs[row, col].set_ylabel("Frequency [Hz]")
        axs[row, col].set_xlabel("TL [dB]")
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
    plt.savefig(out_dir.joinpath("tile.png"))
    fig.clear()


def plot_single(ax, df):
    sns.lineplot(data=df, x="v_out", y="y", ax=ax, label="prediction", orient="y")
    sns.lineplot(data=df, x="v", y="y", ax=ax, label="FEM", alpha=0.5, orient="y")


if __name__ == "__main__":
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
