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


def plot_avail_vs_mse(dataset: pd.DataFrame, tl_dir: pathlib.Path):
    fig, ax = plt.subplots()

    sns.lineplot(
        dataset, x="n_points", y="MSE", ax=ax, style="n_observations", errorbar=None, estimator="mean", color="black"
    )
    ax.set_xlabel("Number of Training Samples [1]")
    ax.set_ylabel(r"Mean Squared Error $[\text{dB}^2]$")
    ax.legend(title="Varying Number of Evaluations")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath("mse_vs_obs.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.lineplot(
        dataset, x="n_points", y="MSE", ax=ax, style="n_evaluations", errorbar=None, estimator="mean", color="black"
    )
    ax.set_xlabel("Number of Training Samples [1]")
    ax.set_ylabel(r"Mean Squared Error $[\text{dB}^2]$")
    ax.legend(title="Varying Number of Evaluations")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath("mse_vs_eval.pdf"))
    plt.close(fig)


def plot_avail_vs_l1(dataset: pd.DataFrame, tl_dir: pathlib.Path):
    fig, ax = plt.subplots()

    sns.lineplot(
        dataset, x="n_points", y="L1", ax=ax, style="n_observations", errorbar=None, estimator="mean", color="black"
    )
    ax.set_xlabel("Number of Training Samples [1]")
    ax.set_ylabel(r"Mean L1 Error $[\text{dB}]$")
    ax.legend(title="Varying Number of Observations")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath("l1_vs_obs.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.lineplot(
        dataset, x="n_points", y="L1", ax=ax, style="n_evaluations", errorbar=None, estimator="mean", color="black"
    )
    ax.set_xlabel("Number of Training Samples [1]")
    ax.set_ylabel(r"Mean L1 Error $[\text{dB}]$")
    ax.legend(title="Varying Number of Observations")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath("l1_vs_eval.pdf"))
    plt.close(fig)


def plot_lines(data: pd.DataFrame, out_dir: pathlib.Path):
    tl_dir = out_dir.joinpath("reduced")
    tl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting reduced lines")

    plot_avail_vs_mse(data, tl_dir)
    logger.info("\t\t reduced vs. mse")
    plot_avail_vs_l1(data, tl_dir)
    logger.info("\t\t reduced vs. l1")
