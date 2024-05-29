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
from nos.networks import (
    ResNet,
)

N_EPOCHS = 500
LR = 5e-3
BATCH_SIZE = 64


def run():
    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"),
        n_samples=-1,
    )

    # clear transformation
    del dataset.transform["v"]

    logger.info("dataset loaded.")
    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    scale_net = torch.nn.Sequential(
        torch.nn.Linear(5, 16),
        torch.nn.Tanh(),
        ResNet(width=16, depth=8, stride=2, act=torch.nn.Tanh()),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 2),
    )

    logger.info("Scale net initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    optimizer = torch.optim.Adam(scale_net.parameters(), lr=LR)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-4)
    scale_net.to(device)

    criterion = torch.nn.MSELoss().to(device)

    for trf in dataset.transform.keys():
        dataset.transform[trf].to(device)

    dataset.x = dataset.x.to(device)
    dataset.u = dataset.u.to(device)
    dataset.y = dataset.y.to(device)
    dataset.v = dataset.v.to(device)

    logger.info("optimizer, scheduler, criterion initialized.")
    logger.info("Start training")

    pbar = tqdm(total=N_EPOCHS)

    max_v, _ = torch.max(torch.abs(dataset.v), dim=1, keepdim=True)
    max_v = torch.log10(max_v)
    lower_v, _ = torch.min(max_v, dim=0, keepdim=True)
    upper_v, _ = torch.max(max_v, dim=0, keepdim=True)

    out_dir = pathlib.Path().cwd().joinpath("scale_net", "ps_500")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = torch.inf

    with mlflow.start_run(run_name="scale-net"):
        scale_net.eval()
        val_loss = []
        for x, u, y, v in val_loader:
            # v is unscaled
            max_vals, _ = torch.max(torch.abs(v), dim=1)
            max_vals = max_vals.squeeze()
            max_vals = torch.log10(max_vals)

            max_vals = (max_vals - lower_v) * (upper_v - lower_v)

            # data part
            out = scale_net(u).squeeze()
            loss = criterion(out, max_vals)
            val_loss.append(loss.item())
        mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_loss)).item(), step=-1)

        for epoch in range(N_EPOCHS):
            train_loss = []
            val_loss = []
            scale_net.train()
            for x, u, y, v in train_loader:
                # v is unscaled
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.squeeze()
                max_vals = torch.log10(max_vals)

                max_vals = (max_vals - lower_v) * (upper_v - lower_v)

                # data part
                out = scale_net(u).squeeze()
                loss = criterion(out, max_vals)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            scale_net.eval()
            for x, u, y, v in val_loader:
                # v is unscaled
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.squeeze()
                max_vals = torch.log10(max_vals)

                max_vals = (max_vals - lower_v) * (upper_v - lower_v)

                # data part
                out = scale_net(u).squeeze()
                loss = criterion(out, max_vals)
                val_loss.append(loss.item())

            pbar.update()

            scheduler.step(epoch=epoch)

            mean_val = torch.mean(torch.tensor(val_loss)).item()
            mlflow.log_metric("Val loss", mean_val, step=epoch)
            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_loss)).item(), step=epoch)
            mlflow.log_metric("LR", scheduler.optimizer.param_groups[0]["lr"], step=epoch)

            if mean_val < best_val:
                best_val = mean_val
                torch.save(scale_net.state_dict(), out_dir.joinpath("best.pt"))


if __name__ == "__main__":
    run()
