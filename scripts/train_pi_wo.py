import pathlib

import matplotlib.pyplot as plt
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

N_EPOCHS = 1000
LR = 5e-3
BATCH_SIZE = 64


def run():
    uid = UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("out_models", str(uid))

    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)  # , weight_decay=1e-4)
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
    train_losses = []
    train_data_losses = []
    val_losses = []
    val_data_losses = []

    operator = operator.to(device)

    loss_weights_initial = torch.tensor([1.0, 0.0]).to(device)
    loss_weights = loss_weights_initial
    loss_weights.to(device)

    with mlflow.start_run(run_name="PI-DDO"):
        for epoch in range(N_EPOCHS):
            operator.train()
            for x, u, y, v in train_loader:
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.view(-1, 1, v.size(-1))

                # data loss
                data_output = operator(x, u, y)
                v_scaled = v / max_vals
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
            for x, u, y, v in val_loader:
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.view(-1, 1, v.size(-1))

                # data loss
                data_output = operator(x, u, y)
                v_scaled = v / max_vals
                data_loss = data_criterion(data_output, v_scaled)
                # overall loss
                loss = data_loss
                val_losses.append(loss.item())
                val_data_losses.append(data_loss.item())

            pbar.update()
            scheduler.step(epoch=epoch)

            mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Val DATA loss", torch.mean(torch.tensor(val_data_losses[-10:])).item(), step=epoch)

            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Train DATA loss", torch.mean(torch.tensor(train_data_losses[-10:])).item(), step=epoch)

            if epoch % 50 == 0:
                serialize(operator, out_dir=out_dir)
    serialize(operator, out_dir=out_dir)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(train_losses)), train_losses, label="train")
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(val_losses)), val_losses, label="val")
    axs[0].set_yscale("log")

    for ax in axs.flatten():
        ax.legend()

    plt.show()


if __name__ == "__main__":
    run()
