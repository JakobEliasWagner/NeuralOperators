import pathlib

import matplotlib.pyplot as plt
import numpy as np
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


def plot_frequency_vs_mse(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    fig, ax = plt.subplots()
    sns.lineplot(dataset, x="frequency", y="MSE", ax=ax, hue="architecture", errorbar=None, estimator="median")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
    ax.set_yscale("log")
    ax.legend(title="Architecture")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath(f"mse_{size}_freq.pdf"))
    plt.close(fig)


def plot_frequency_vs_l1(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    fig, ax = plt.subplots()
    sns.lineplot(dataset, x="frequency", y="L1", ax=ax, hue="architecture", errorbar=None, estimator="median")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"L1 Error $[\text{dB}]$")
    ax.set_yscale("log")
    ax.legend(title="Architecture")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath(f"l1_{size}_freq.pdf"))
    plt.close(fig)


def plot_param_vs_l1(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    for param in ["radius", "inner_radius", "gap_widths"]:
        param_nice = " ".join(param.split("_"))
        dataset = quantize(dataset, param)
        fig, ax = plt.subplots()
        sns.lineplot(
            dataset,
            x=f"{param}",
            y="L1",
            ax=ax,
            hue="architecture",
            errorbar=None,
            estimator="median",
            marker=".",
            markers=True,
        )
        ax.set_xlabel(f"{param_nice.title()} [mm]")
        ax.set_ylabel(r"L1 Error $[\text{dB}]$")
        ax.set_yscale("log")
        ax.legend(title="Architecture")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"l1_{size}_{param}.pdf"))
        plt.close(fig)


def plot_param_vs_mse(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    for param in ["radius", "inner_radius", "gap_widths"]:
        param_nice = " ".join(param.split("_"))
        dataset = quantize(dataset, param)
        fig, ax = plt.subplots()
        sns.lineplot(
            dataset,
            x=f"{param}_quantized",
            y="L1",
            ax=ax,
            hue="architecture",
            errorbar=None,
            estimator="median",
            marker=".",
            markers=True,
        )
        ax.set_xlabel(f"{param_nice.title()} [mm]")
        ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
        ax.set_yscale("log")
        ax.legend(title="Architecture")
        ax.yaxis.grid(which="major")
        fig.tight_layout()
        plt.savefig(tl_dir.joinpath(f"mse_{size}_{param}.pdf"))
        plt.close(fig)


def quantize(dataset: pd.DataFrame, col: str, n_q: int = 100) -> pd.DataFrame:
    if f"{col}_quantized" in dataset:
        return dataset

    max_val = dataset[col].max()
    min_val = dataset[col].min()

    intervals = pd.cut(dataset[f"{col}"], bins=np.linspace(min_val, max_val, n_q))
    center = intervals.apply(lambda x: (x.right + x.left) / 2)
    dataset[f"{col}_quantized"] = center
    return dataset


def plot_tl_vs_mse(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    dataset = quantize(dataset, "transmission_loss")

    fig, ax = plt.subplots()
    sns.lineplot(
        dataset, x="transmission_loss_quantized", y="MSE", ax=ax, hue="architecture", errorbar=None, estimator="median"
    )
    ax.set_xlabel("Transmission Loss [dB]")
    ax.set_ylabel(r"Squared Error $[\text{dB}^2]$")
    ax.set_yscale("log")
    ax.legend(title="Architecture")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath(f"mse_{size}_tl.pdf"))
    plt.close(fig)


def plot_tl_vs_l1(dataset: pd.DataFrame, size: str, tl_dir: pathlib.Path):
    dataset = quantize(dataset, "transmission_loss")
    fig, ax = plt.subplots()
    sns.lineplot(
        dataset, x="transmission_loss_quantized", y="L1", ax=ax, hue="architecture", errorbar=None, estimator="median"
    )
    ax.set_xlabel("Transmission Loss [dB]")
    ax.set_ylabel(r"L1 Error $[\text{dB}]$")
    ax.set_yscale("log")
    ax.legend(title="Architecture")
    ax.yaxis.grid(which="major")
    fig.tight_layout()
    plt.savefig(tl_dir.joinpath(f"l1_{size}_tl.pdf"))
    plt.close(fig)


def plot_lines(data: pd.DataFrame, out_dir: pathlib.Path):
    tl_dir = out_dir.joinpath("transmission_loss")
    tl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting transmission lines")
    for size in ["small", "medium", "large"]:
        logger.info(f"\tSize: {size}")

        size_data = data[data["size"] == size]
        size_data = size_data[size_data["phase"] == "TEST"]
        size_data = size_data[size_data["dataset"] == "SMOOTH"]

        if len(size_data) == 0:
            continue

        plot_frequency_vs_mse(size_data, size, tl_dir)
        logger.info("\t\t f vs. mse")
        plot_frequency_vs_l1(size_data, size, tl_dir)
        logger.info("\t\t f vs. l1")
        plot_tl_vs_mse(size_data, size, tl_dir)
        logger.info("\t\t tl vs. mse")
        plot_tl_vs_l1(size_data, size, tl_dir)
        logger.info("\t\t tl vs. l1")
        plot_param_vs_l1(size_data, size, tl_dir)
        logger.info("\t\t param vs. l1")
        plot_param_vs_mse(size_data, size, tl_dir)
        logger.info("\t\t param vs. mse")
