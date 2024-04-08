import pathlib

import numpy as np
import pandas as pd
from process_operators import (
    process_all,
)
from reduced.plot_boxplots import (
    plot_boxplot as plot_boxplots_reduced,
)
from reduced.plot_lines import (
    plot_lines as plot_lines_reduced,
)
from tabulate import (
    tabulate,
)
from transmission_loss.plot_boxplots import (
    plot_boxplot,
)
from transmission_loss.plot_lines import (
    plot_lines,
)

ARCH_MAPS = {
    "deep_dot_operator": "DDO",
    "deep_neural_operator": "DNO",
    "deep_o_net": "DON",
    "fourier_neural_operator": "FNO",
    "transformer_neural_operator": "TNO",
}

FLOAT_COLUMNS = [
    "MSE",
    "L1",
    "frequency",
    "transmission_loss",
    "radius",
    "inner_radius",
    "gap_widths",
    "eval_time",
]


def preprocess_tl_data(data: pd.DataFrame) -> pd.DataFrame:
    data["architecture"] = data["architecture"].map(ARCH_MAPS)
    for col in data.columns:
        if col in FLOAT_COLUMNS:
            data[col] = data[col].astype(np.float64)
    return data


def print_performances(data: pd.DataFrame) -> None:
    operators = data[["architecture", "size", "dataset", "phase"]].drop_duplicates()
    results = []
    for _, operator in operators.iterrows():
        op_df = data[
            (data["architecture"] == operator["architecture"])
            & (data["size"] == operator["size"])
            & (data["dataset"] == operator["dataset"])
            & (data["phase"] == operator["phase"])
        ]
        means = op_df[["L1", "MSE"]].mean()
        results.append(
            (
                operator["architecture"],
                operator["size"],
                operator["dataset"],
                operator["phase"],
                means["L1"],
                means["MSE"],
            )
        )
    print(tabulate(results, headers=["architecture", "size", "dataset", "phase", "L1", "MSE"]))


def plot_finished(data_file: pathlib.Path, out_dir: pathlib.Path):
    data = pd.read_csv(data_file)
    data = preprocess_tl_data(data)
    print_performances(data)
    plot_boxplot(data, out_dir)
    plot_lines(data, out_dir)


def preprocess_reduced(df: pd.DataFrame) -> pd.DataFrame:
    df[["n_observations", "n_evaluations"]] = df["size"].str.extract("(\d+)\D+(\d+)")

    df[["n_observations", "n_evaluations"]] = df[["n_observations", "n_evaluations"]].astype(int)
    df["n_points"] = df["n_observations"] * df["n_evaluations"]

    return df


def plot_reduced(data_file: pathlib.Path, out_dir: pathlib):
    data = pd.read_csv(data_file)
    data = preprocess_reduced(data)
    plot_lines_reduced(data, out_dir)
    plot_boxplots_reduced(data, out_dir)


def main():
    out_path = pathlib.Path.cwd().joinpath("out", "final")

    data_dir = pathlib.Path.cwd().joinpath("finished_models")
    process_all(data_dir)
    data_path = data_dir.joinpath("results.csv")
    plot_finished(data_path, out_path)

    data_dir = pathlib.Path.cwd().joinpath("finished_reduced")
    process_all(data_dir)
    data_path = data_dir.joinpath("results.csv")
    plot_reduced(data_path, out_path)


if __name__ == "__main__":
    main()
