import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import (
    tqdm,
)

from .data import (
    MultiRunData,
    RunData,
)


def plot_multirun_curves(multi_run: MultiRunData, out_dir: pathlib.Path):
    out_dir = out_dir.joinpath("training_curves")

    df = pd.DataFrame()
    pbar = tqdm(multi_run.runs, leave=False, position=1)
    for run in pbar:
        run_out = out_dir.joinpath(run.name)
        run_out.mkdir(parents=True, exist_ok=True)
        plot_run_curves(run, run_out)

        run_df = run.training
        run_df["Architecture"] = run.name
        df = pd.concat([df, run_df])

    out_file = out_dir.joinpath("epoch_vs_mse.png")
    plot_xyz_vs_error(df, out_file, "Epochs")
    out_file = out_dir.joinpath("epoch_vs_mse.png")
    plot_xyz_vs_error(df, out_file, "time")


def plot_run_curves(run: RunData, out_dir: pathlib.Path):
    run_df = run.training
    run_df["Architecture"] = run.name
    out_file = out_dir.joinpath("epoch_vs_mse.png")
    plot_xyz_vs_error(run_df, out_file, "Epochs")
    out_file = out_dir.joinpath("epoch_vs_mse.png")
    plot_xyz_vs_error(run_df, out_file, "time")


def plot_xyz_vs_error(df_plot: pd.DataFrame, out_file: pathlib.Path, x: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(df_plot, x=x, y="Val_loss", ax=ax, linestyle="--", hue="Architecture")
    sns.lineplot(df_plot, x=x, y="Train_loss", ax=ax, hue="Architecture")
    ax.set_yscale("log")
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    by_label = dict(zip(labels, lines))
    ax.legend(
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
