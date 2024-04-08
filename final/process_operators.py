import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from continuity.data import (
    OperatorDataset,
)
from continuity.operators import (
    Operator,
)
from loguru import (
    logger,
)
from torch.utils.data import (
    DataLoader,
)

from nos.data import (
    TLDatasetCompact,
    TLDatasetCompactWave,
)
from nos.operators import (
    FourierNeuralOperator,
    deserialize,
)

CWD = pathlib.Path.cwd()
DATA_DIR = CWD.joinpath("data")
DATASETS = {
    "SMOOTH": {
        "TRAIN": DATA_DIR.joinpath("train", "transmission_loss_smooth"),
        "TEST": DATA_DIR.joinpath("test", "transmission_loss_smooth"),
    },
}


def process_operator(operator: Operator, datasets: dict) -> pd.DataFrame:
    operator.eval()

    l1 = nn.L1Loss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    df = pd.DataFrame()
    for dset_type, phases in datasets.items():
        for phase, dataset in phases.items():
            eval_loader = DataLoader(dataset, batch_size=1)

            # losses
            l1_losses = []
            mse_losses = []
            frequencies = []
            tls = []
            radii = []
            inner_radii = []
            gap_widths = []
            for x, u, y, v in eval_loader:
                v_pred = operator(x, u, y)

                v_unscaled = dataset.transform["v"].undo(v)
                v_pred_unscaled = dataset.transform["v"].undo(v_pred)

                l1_loss = l1(v_pred_unscaled, v_unscaled)
                l1_losses.extend(l1_loss.squeeze().tolist())
                mse_loss = mse(v_pred_unscaled, v_unscaled)
                mse_losses.extend(mse_loss.squeeze().tolist())

                y_unscaled = dataset.transform["y"].undo(y)
                frequencies.extend(y_unscaled.squeeze().tolist())

                v_unscaled = dataset.transform["v"].undo(v)
                tls.extend(v_unscaled.squeeze().tolist())

                u_unscaled = dataset.transform["u"].undo(u)
                if u_unscaled.size(1) != 1:
                    # wave dataset
                    f_space = np.linspace(-1, 1, u_unscaled.size(1))
                    min_idx = np.argmin(u_unscaled[0, :, :], axis=0)
                    max_idx = np.argmax(u_unscaled[0, :, :], axis=0)

                    h_wave_length = np.abs(f_space[max_idx] - f_space[min_idx])
                    params = np.pi / h_wave_length
                    params = params / np.pi - 1

                    smallest_vals = np.array([0.0033, 0.0030, 0.0010])
                    biggest_vals = np.array([0.0100, 0.0096, 0.0080])
                    deltas = biggest_vals - smallest_vals

                    params = params * deltas + smallest_vals
                    params = list(params)
                else:
                    # compact wave dataset
                    params = u_unscaled.squeeze().tolist()

                radii.extend([params[0]] * y.nelement())
                inner_radii.extend([params[1]] * y.nelement())
                gap_widths.extend([params[2]] * y.nelement())

            logger.info(f"\tLosses: {torch.mean(torch.tensor(mse_losses))}")

            # n param
            n_param = sum(p.numel() for p in operator.parameters() if p.requires_grad)
            logger.info(f"\t#Parameters: {n_param}")

            # eval speed
            start = time.time()
            end = time.time()
            n_rot = 0
            while end - start < 1:
                for x, u, y, v in eval_loader:
                    _ = operator(x, u, y)
                n_rot += 1
                end = time.time()
            eval_time = (end - start) / n_rot / len(eval_loader)
            logger.info(f"\tTotal eval time: {(end - start)}, Single: {eval_time}")

            tmp_df = pd.DataFrame(
                {
                    "MSE": mse_losses,
                    "L1": l1_losses,
                    "frequency": frequencies,
                    "transmission_loss": tls,
                    "radius": radii,
                    "inner_radius": inner_radii,
                    "gap_widths": gap_widths,
                }
            )
            tmp_df["eval_time"] = eval_time
            tmp_df["n_param"] = n_param
            tmp_df["phase"] = phase
            tmp_df["dataset"] = dset_type

            df = pd.concat([df, tmp_df])

    return df


def get_all_datasets() -> dict:
    return {
        "CompactWave": get_datasets(TLDatasetCompactWave),
        "Compact": get_datasets(TLDatasetCompact),
    }


def get_datasets(base_class: type(OperatorDataset)) -> dict:
    datasets = {}
    for d_id, d_all in DATASETS.items():
        dataset = {}
        for phase, d_path in d_all.items():
            dataset[phase] = base_class(d_path)
        dataset["TEST"].transform = dataset["TRAIN"].transform
        del dataset["TRAIN"]
        datasets[d_id] = dataset
    return datasets


def process_operators(operators: list, operator_dir: pathlib.Path):
    datasets = get_all_datasets()

    df_file = operator_dir.joinpath("results.csv")
    if df_file.exists():
        df = pd.read_csv(df_file)
        processed_operators = df[["architecture", "size"]].drop_duplicates(ignore_index=True)
    else:
        df = pd.DataFrame()
        processed_operators = pd.DataFrame({"architecture": [], "size": []})
    for arch, size, operator in operators:
        logger.info(f"Processing {arch} of size {size}.")

        if ((processed_operators["architecture"] == arch) & (processed_operators["size"] == size)).any():
            logger.info(f"Skipping {arch} of size {size} because it already exists.")
            continue

        key = "CompactWave" if isinstance(operator, FourierNeuralOperator) else "Compact"
        op_df = process_operator(operator, datasets[key])
        op_df["architecture"] = arch
        op_df["size"] = size
        df = pd.concat([df, op_df], ignore_index=True)

    df.to_csv(df_file, index=False)


def load_operators(finished_models: pathlib.Path) -> list:
    operators = []
    for operator_meta_dir in finished_models.glob("*"):
        if operator_meta_dir.is_file():
            continue
        for size_dir in operator_meta_dir.glob("*"):
            if size_dir.is_file():
                continue
            operator_pts = size_dir.rglob("*.pt")
            for operator_pt in operator_pts:
                operator_dir = operator_pt.parent
                operator = deserialize(operator_dir)
                operators.append((operator_meta_dir.name, size_dir.name, operator))

    return operators


def process_all(finished_models: pathlib.Path):
    operators = load_operators(finished_models)
    logger.info(f"Loaded {len(operators)} operators.")
    process_operators(operators, finished_models)


if __name__ == "__main__":
    finished_models_path = CWD.joinpath("finished_models")
    process_all(finished_models_path)
