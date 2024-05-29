import json
import pathlib

import mlflow
import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)
from torch.utils.data import (
    DataLoader,
    random_split,
)
from tqdm import (
    tqdm,
)

from nos.data import (
    PulsatingSphere,
)
from nos.operators import (
    DeepDotOperator,
    serialize,
)
from nos.utils import (
    UniqueId,
)

N_EPOCHS = 173
LR = 5e-3
BATCH_SIZE = 64


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    uid = UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("out_models", str(uid))

    torch.manual_seed(42)

    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"),
        n_samples=-1,
    )
    del dataset.transform["v"]

    logger.info("dataset loaded.")
    train_set, val_set = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=32,
        branch_depth=12,
        trunk_depth=12,
        trunk_width=32,
        dot_depth=12,
        dot_width=32,
        stride=4,
    )
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-4)
    # scheduler = sched.ConstantLR(optimizer)
    data_criterion = torch.nn.MSELoss().to(device)

    for trf in dataset.transform.keys():
        dataset.transform[trf].to(device)

    dataset.x = dataset.x.to(device)
    dataset.u = dataset.u.to(device)
    dataset.y = dataset.y.to(device)
    dataset.v = dataset.v.to(device)

    logger.info("optimizer, scheduler, criterion initialized.")
    logger.info("Start training")

    pbar = tqdm(total=N_EPOCHS)

    operator = operator.to(device)

    best_val_data_loss = 1e-2

    with mlflow.start_run(run_name="DATA-DDO"):
        operator.eval()
        val_losses = []
        val_data_losses = []
        for x, u, y, v in val_loader:
            # scale v
            max_v, _ = torch.max(torch.abs(v), dim=1, keepdim=True)
            v_scaled = v / max_v
            v_scaled = v_scaled.to(device)

            # data loss
            data_output = operator(x, u, y)
            data_loss = data_criterion(data_output, v_scaled)
            # overall loss
            loss = data_loss
            val_losses.append(loss.item())
            val_data_losses.append(data_loss.item())
        mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses)).item(), step=-1)
        mlflow.log_metric("Val DATA loss", torch.mean(torch.tensor(val_data_losses)).item(), step=-1)

        for epoch in range(N_EPOCHS):
            operator.train()
            train_losses = []
            train_data_losses = []
            for x, u, y, v in train_loader:
                # scale v
                max_v, _ = torch.max(torch.abs(v), dim=1, keepdim=True)
                v_scaled = v / max_v
                v_scaled = v_scaled.to(device)

                # data loss
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v_scaled)
                # overall loss
                loss = data_loss
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update metrics
                train_data_losses.append(data_loss.item())
                train_losses.append(loss.item())

            operator.eval()
            val_losses = []
            val_data_losses = []
            for x, u, y, v in val_loader:
                # scale v
                max_v, _ = torch.max(torch.abs(v), dim=1, keepdim=True)
                v_scaled = v / max_v
                v_scaled = v_scaled.to(device)

                # data loss
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v_scaled)
                # overall loss
                loss = data_loss
                val_losses.append(loss.item())
                val_data_losses.append(data_loss.item())

            pbar.update()
            scheduler.step(epoch=epoch)

            mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses)).item(), step=epoch)
            mean_val_data_loss = torch.mean(torch.tensor(val_data_losses)).item()
            mlflow.log_metric("Val DATA loss", mean_val_data_loss, step=epoch)

            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_losses)).item(), step=epoch)
            mlflow.log_metric("Train DATA loss", torch.mean(torch.tensor(train_data_losses)).item(), step=epoch)

            if mean_val_data_loss < best_val_data_loss:
                best_val_data_loss = mean_val_data_loss
                model_out_dir = serialize(operator, out_dir=out_dir)
                with open(model_out_dir.joinpath("epoch.json"), "w") as file_handle:
                    file_handle.write(json.dumps({"epoch": epoch, "loss": mean_val_data_loss}, indent=4))

    serialize(operator, out_dir=out_dir)


if __name__ == "__main__":
    run()
