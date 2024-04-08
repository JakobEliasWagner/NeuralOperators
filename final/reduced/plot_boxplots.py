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
    red_dir = out_dir.joinpath("reduced")
    red_dir.mkdir(parents=True, exist_ok=True)

    different_n_obs = data["n_observations"].drop_duplicates()

    logger.info("Plotting transmission box-plots")
    for n_observation in different_n_obs:
        logger.info(f"\tSize: {n_observation}")

        size_data = data[data["n_observations"] == n_observation]
        size_data = size_data[size_data["phase"] == "TEST"]
        size_data = size_data[size_data["dataset"] == "SMOOTH"]

        if len(size_data) == 0:
            continue

        fig, ax = plt.subplots()
        sns.boxplot(
            size_data, x="n_evaluations", y="MSE", ax=ax, linecolor="black", color="lightgray", showfliers=False
        )
        ax.set_xlabel("Number of Evaluations [1]")
        ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(red_dir.joinpath(f"mse_{n_observation}_box.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.boxplot(
            size_data, x="n_evaluations", y="L1", ax=ax, linecolor="black", color="lightgray", showfliers=False
        )
        ax.set_xlabel("Number of Evaluations [1]")
        ax.set_ylabel(r"L1 Error $[\text{dB}]$")
        ax.set_yscale("log")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(red_dir.joinpath(f"l1_{n_observation}_box.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.violinplot(size_data, x="n_evaluations", y="MSE", ax=ax, linecolor="black", color="lightgray")
        ax.set_xlabel("Number of Evaluations [1]")
        ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(red_dir.joinpath(f"mse_{n_observation}_vio.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.violinplot(size_data, x="n_evaluations", y="L1", ax=ax, linecolor="black", color="lightgray")
        ax.set_xlabel("Number of Evaluations [1]")
        ax.set_ylabel(r"L1 Error $[\text{dB}]$")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(red_dir.joinpath(f"l1_{n_observation}_vio.pdf"))
        plt.close(fig)
