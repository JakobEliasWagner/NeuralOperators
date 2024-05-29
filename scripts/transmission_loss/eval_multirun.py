import pathlib
import time
from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn
from continuity.data import (
    OperatorDataset,
)
from torch.utils.data import (
    DataLoader,
)

from nos.data import (
    TLDatasetCompact,
    TLDatasetCompactWave,
)
from nos.plots import (
    MultiRunData,
)


def evaluate_multirun(
    multirun_path: pathlib.Path, datasets: List[Tuple[str, OperatorDataset]], out_file: pathlib.Path
):
    multirun = MultiRunData.from_dir(multirun_path)
    mse_crit = torch.nn.MSELoss()
    l1_crit = torch.nn.L1Loss()

    if multirun_path.is_file():
        df = pd.read_csv(out_file)
    else:
        df = pd.DataFrame()

    for dataset_name, dataset in datasets:
        loader = DataLoader(dataset, batch_size=1)
        for run in multirun.runs:
            for model in run.models:
                model.operator.eval()

                start = time.time_ns()
                for x, u, y, v in loader:
                    v_pred = model.operator(x, u, y)
                end = time.time_ns()

                mse_losses = []
                l1_losses = []
                for x, u, y, v in loader:
                    v_pred = model.operator(x, u, y)
                    mse_loss = mse_crit(v, v_pred)
                    mse_losses.append(mse_loss.item())

                    l1_loss = l1_crit(v, v_pred)
                    l1_losses.append(l1_loss.item())

                total_params = sum(p.numel() for p in model.operator.parameters() if p.requires_grad)

                model_df = pd.DataFrame(
                    {
                        "model": model.name,
                        "run": run.name,
                        "multirun": multirun.name,
                        "dataset": dataset_name,
                        "MSE": mse_losses,
                        "L1": l1_losses,
                        "eval_time": (end - start) * 1e-9 / len(dataset),
                        "n_parameters": total_params,
                    }
                )
                df = pd.concat([df, model_df], ignore_index=True)

    df = df.drop_duplicates(ignore_index=True)
    df.to_csv(out_file, index=False)


def eval_wave_multirun(out_file: pathlib.Path):
    run_id = ["2024-03-27", "09-31-37"]
    multirun_path = pathlib.Path.cwd().joinpath("multirun", *run_id)

    test_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_smooth")
    test_set = TLDatasetCompactWave(test_path)

    non_smooth_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_lin")
    non_smooth_set = TLDatasetCompactWave(non_smooth_path)

    train_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth")
    train_set = TLDatasetCompactWave(train_path)
    test_set.transform = train_set.transform
    non_smooth_set.transform = train_set.transform

    evaluate_multirun(
        multirun_path=multirun_path,
        datasets=[("transmission_loss_smooth", test_set), ("transmission_loss_lin", non_smooth_set)],
        out_file=out_file,
    )


def eval_compact_multirun(out_file: pathlib.Path):
    run_id = ["2024-03-26", "22-45-12"]
    multirun_path = pathlib.Path.cwd().joinpath("multirun", *run_id)

    test_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_smooth")
    test_set = TLDatasetCompact(test_path)

    non_smooth_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_lin")
    non_smooth_set = TLDatasetCompact(non_smooth_path)

    train_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth")
    train_set = TLDatasetCompact(train_path)
    test_set.transform = train_set.transform
    non_smooth_set.transform = train_set.transform

    evaluate_multirun(
        multirun_path=multirun_path,
        datasets=[("transmission_loss_smooth", test_set), ("transmission_loss_lin", non_smooth_set)],
        out_file=out_file,
    )


if __name__ == "__main__":
    out_file = pathlib.Path.cwd().joinpath("out", "performance.csv")
    eval_wave_multirun(out_file)
    eval_compact_multirun(out_file)

    df = pd.read_csv(out_file, index_col=False)
    pp = sns.pairplot(df, hue="run")
    log_columns = ["MSE", "L1", "eval_time"]
    for ax in pp.axes.flat:
        if ax.get_xlabel() in log_columns:
            ax.set(xscale="log")
        if ax.get_ylabel() in log_columns:
            ax.set(yscale="log")

    plt.show()
