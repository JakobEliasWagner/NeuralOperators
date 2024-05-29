import pathlib
import shutil

import mlflow
import torch
import torch.optim.lr_scheduler as sched
from torch.utils.data import (
    DataLoader,
    random_split,
)
from tqdm import (
    tqdm,
)

from nos.data import (
    TLDatasetCompact,
    TLDatasetCompactWave,
)
from nos.operators import (
    DeepDotOperator,
    DeepNeuralOperator,
    DeepONet,
    FourierNeuralOperator,
    serialize,
)

N_EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 12
N_SAMPLES = 3

OPERATORS = {
    "DeepONet": {
        "class": DeepONet,
        "small": {
            "branch_width": 32,
            "branch_depth": 4,
            "trunk_depth": 5,
            "trunk_width": 32,
            "stride": 1,
            "basis_functions": 4,
        },
        "medium": {
            "branch_width": 42,
            "branch_depth": 24,
            "trunk_depth": 28,
            "trunk_width": 44,
            "stride": 4,
            "basis_functions": 48,
        },
        "big": {
            "branch_width": 64,
            "branch_depth": 96,
            "trunk_depth": 64,
            "trunk_width": 96,
            "stride": 8,
            "basis_functions": 48,
        },
    },
    "DeepNeuralOperator": {
        "class": DeepNeuralOperator,
        "small": {
            "depth": 8,
            "width": 36,
            "stride": 2,
        },
        "medium": {
            "depth": 32,
            "width": 56,
            "stride": 4,
        },
        "big": {
            "depth": 64,
            "width": 125,
            "stride": 8,
        },
    },
    "FourierNeuralOperator": {
        "class": FourierNeuralOperator,
        "small": {"width": 4, "depth": 5},
        "medium": {"width": 10, "depth": 8},
        "big": {"width": 22, "depth": 16},
    },
    "DeepDotOperator": {
        "class": DeepDotOperator,
        "small": {
            "branch_width": 28,
            "branch_depth": 4,
            "trunk_depth": 4,
            "trunk_width": 32,
            "dot_depth": 4,
            "dot_width": 32,
            "stride": 1,
        },
        "medium": {
            "branch_width": 32,
            "branch_depth": 16,
            "trunk_depth": 16,
            "trunk_width": 32,
            "dot_depth": 32,
            "dot_width": 46,
            "stride": 4,
        },
        "big": {
            "branch_width": 64,
            "branch_depth": 48,
            "trunk_depth": 48,
            "trunk_width": 64,
            "dot_depth": 48,
            "dot_width": 112,
            "stride": 8,
        },
    },
}


def run_single(operator_class, operator_config, dataset_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize dataset
    if operator_class == FourierNeuralOperator:
        dataset_class = TLDatasetCompactWave
    else:
        dataset_class = TLDatasetCompact
    dataset = dataset_class(
        path=dataset_path,
    )
    for trf in dataset.transform.keys():
        dataset.transform[trf].to(device)
    dataset.x = dataset.x.to(device)
    dataset.u = dataset.u.to(device)
    dataset.y = dataset.y.to(device)
    dataset.v = dataset.v.to(device)

    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # initialize operator
    operator = operator_class(dataset.shapes, **operator_config)

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-4)
    data_criterion = torch.nn.MSELoss().to(device)

    pbar = tqdm(total=N_EPOCHS)

    operator = operator.to(device)

    best_val_loss = torch.inf
    best_val_75_loss = torch.inf
    with mlflow.start_run(run_name=f"benchmark-{operator_class.__name__}"):
        for epoch in range(N_EPOCHS):
            operator.train()
            train_losses = []
            val_losses = []
            for x, u, y, v in train_loader:
                # data loss
                data_output = operator(x, u, y)
                loss = data_criterion(data_output, v)
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update metrics
                train_losses.append(loss.item())
            operator.eval()
            for x, u, y, v in val_loader:
                # data loss
                data_output = operator(x, u, y)
                loss = data_criterion(data_output, v)
                val_losses.append(loss.item())

            pbar.update()
            scheduler.step(epoch=epoch)

            mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
            mlflow.log_metric("Val loss", mean_val_loss, step=epoch)

            mean_train_loss = torch.mean(torch.tensor(train_losses)).item()
            mlflow.log_metric("Train loss", mean_train_loss, step=epoch)

            mlflow.log_metric("LR", scheduler.optimizer.param_groups[0]["lr"], step=epoch)

            if epoch % 100 == 0 and epoch != N_EPOCHS - 1:
                serialize(operator, out_dir=out_dir)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_dir = out_dir.joinpath("best_mean")
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                serialize(operator, out_dir=best_dir)

            q_75_val_loss = torch.quantile(torch.tensor(val_losses), 0.75).item()
            if q_75_val_loss < best_val_75_loss:
                best_val_75_loss = q_75_val_loss
                best_dir = out_dir.joinpath("best_75")
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                serialize(operator, out_dir=best_dir)

    serialize(operator, out_dir=out_dir)


def run():
    out_dir = pathlib.Path.cwd().joinpath("out_models", "benchmark2")
    data_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth")

    for size in ["small", "medium", "big"]:
        for model in OPERATORS.keys():
            for n_run in range(N_SAMPLES):
                run_dir = out_dir.joinpath(f"{model}_{size}_{n_run}")
                if run_dir.exists():
                    print(f"Skipping {run_dir.name}")
                    continue
                run_single(
                    operator_class=OPERATORS[model]["class"],
                    operator_config=OPERATORS[model][size],
                    dataset_path=data_path,
                    out_dir=run_dir,
                )


if __name__ == "__main__":
    run()
