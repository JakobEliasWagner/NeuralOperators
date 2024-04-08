import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import (
    logger,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)


def plot_boxplot(data: pd.DataFrame, out_dir: pathlib.Path):
    tl_dir = out_dir.joinpath("transmission_loss")
    tl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting transmission box-plots")
    for size in ["small", "medium", "large"]:
        logger.info(f"\tSize: {size}")

        size_data = data[data["size"] == size]
        size_data = size_data[size_data["phase"] == "TEST"]
        size_data = size_data[size_data["dataset"] == "SMOOTH"]

        if len(size_data) == 0:
            continue

        fig, ax = plt.subplots()
        sns.boxplot(
            size_data, x="architecture", y="MSE", ax=ax, linecolor="black", color="lightgray", showfliers=False
        )
        ax.set_xlabel("Architecture")
        ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"mse_{size}_box.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.boxplot(size_data, x="architecture", y="L1", ax=ax, linecolor="black", color="lightgray", showfliers=False)
        ax.set_xlabel("Architecture")
        ax.set_ylabel(r"L1 Error $[\text{dB}]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"l1_{size}_box.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.violinplot(
            size_data, x="architecture", y="MSE", ax=ax, linecolor="black", color="lightgray", log_scale=True
        )
        ax.set_xlabel("Architecture")
        ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"mse_{size}_vio.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.violinplot(
            size_data, x="architecture", y="L1", ax=ax, linecolor="black", color="lightgray", log_scale=True
        )
        ax.set_xlabel("Architecture")
        ax.set_ylabel(r"L1 Error $[\text{dB}]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"l1_{size}_vio.pdf"))
        plt.close(fig)
