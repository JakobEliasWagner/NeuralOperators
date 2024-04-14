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
from nos.physics import (
    HelmholtzDomainMSE,
)

N_EPOCHS = 100
N_EPOCHS_SCALING = 10000
LR = 5e-3
BATCH_SIZE = 64
N_COL_SAMPLES = 2**9


def sample_collocations(n_samples: int):
    collocations = []
    while len(collocations) < n_samples:
        sample = torch.rand(2)
        if torch.isclose(sample, torch.zeros(2)).any():
            continue
        if torch.isclose(sample, torch.ones(2)).any():
            continue
        if torch.sqrt(torch.sum(sample**2)).item() <= 0.2:
            continue
        collocations.append(sample)
    collocations = torch.stack(collocations)
    return collocations.unsqueeze(0)


def run():
    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"),
        n_samples=-1,
    )

    # clear transformation
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

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-4)

    data_criterion = torch.nn.MSELoss().to(device)
    pde_criterion = HelmholtzDomainMSE().to(device)

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
    train_pde_losses = []
    train_data_losses = []
    val_losses = []
    val_pde_losses = []
    val_data_losses = []

    operator.to(device)

    cp = sample_collocations(n_samples=N_COL_SAMPLES).to(device)
    cp = cp.to(device)
    cp.requires_grad = True

    loss_weights_initial = torch.tensor([1.0, 0.0]).to(device)
    loss_weights = loss_weights_initial
    loss_weights.to(device)

    with mlflow.start_run(run_name="PI-DDO"):
        for epoch in range(N_EPOCHS):
            operator.train()
            for x, u, y, v in train_loader:
                # v is unscaled
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.view(-1, 1, v.size(-1))

                # data part
                data_output = operator(x, u, y)
                gt_data_scaled = v / max_vals
                data_loss = data_criterion(data_output, gt_data_scaled)

                # prepare pde part
                tmp_cp = cp.expand(x.size(0), -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                pde_parameters = dataset.transform["u"].undo(u).to(device)
                f = pde_parameters[:, :, -1]
                wl = 343.0 / f
                k = 2 * torch.pi / wl

                # pde loss
                pde_output = operator(x, u, tmp_scp)
                pde_loss = pde_criterion(tmp_cp.to(device), pde_output, k.to(device))
                # overall loss
                loss = torch.stack([data_loss, pde_loss]).to(device)
                loss = loss_weights @ loss
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update metrics
                train_data_losses.append(data_loss.item())
                train_pde_losses.append(pde_loss.item())
                train_losses.append(loss.item())

            operator.eval()
            for x, u, y, v in val_loader:
                # v is unscaled
                max_vals, _ = torch.max(torch.abs(v), dim=1)
                max_vals = max_vals.view(-1, 1, v.size(-1))

                # data part
                data_output = operator(x, u, y)
                gt_data_scaled = v / max_vals
                data_loss = data_criterion(data_output, gt_data_scaled)

                # prepare pde part
                tmp_cp = cp.expand(x.size(0), -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                pde_parameters = dataset.transform["u"].undo(u).to(device)
                f = pde_parameters[:, :, -1]
                wl = 343.0 / f
                k = 2 * torch.pi / wl

                # pde loss
                pde_output = operator(x, u, tmp_scp)
                pde_loss = pde_criterion(tmp_cp.to(device), pde_output, k.to(device))

                # overall loss
                loss = torch.stack([data_loss, pde_loss]).to(device)
                loss = loss_weights @ loss

                # update metrics
                val_data_losses.append(data_loss.item())
                val_pde_losses.append(pde_loss.item())
                val_losses.append(loss.item())

            pbar.update()

            if epoch % 10 == 5 and epoch > 20:
                mean_loss = torch.mean(torch.tensor(val_losses[-10:]))
                mean_pde_loss = torch.mean(torch.tensor(val_pde_losses[-10:]))

                if torch.isnan(mean_loss) or torch.isnan(mean_pde_loss):
                    mean_loss = torch.mean(torch.tensor(train_losses[-10:]))
                    mean_pde_loss = torch.mean(torch.tensor(train_pde_losses[-10:]))

                pde_weight = 0.1 * mean_loss / mean_pde_loss
                loss_weights = torch.tensor([1.0, pde_weight]).to(device)

            if epoch % 10 == 0 and epoch > 0:
                cp = sample_collocations(n_samples=N_COL_SAMPLES).to(device)
                cp = cp.to(device)
                cp.requires_grad = True

            scheduler.step(epoch=epoch)

            mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Val PDE loss", torch.mean(torch.tensor(val_pde_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Val DATA loss", torch.mean(torch.tensor(val_data_losses[-10:])).item(), step=epoch)

            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Train PDE loss", torch.mean(torch.tensor(train_pde_losses[-10:])).item(), step=epoch)
            mlflow.log_metric("Train DATA loss", torch.mean(torch.tensor(train_data_losses[-10:])).item(), step=epoch)

            mlflow.log_metric("DATA weight", loss_weights[0].item(), step=epoch)
            mlflow.log_metric("PDE weight", loss_weights[1].item(), step=epoch)

            mlflow.log_metric("LR", scheduler.optimizer.param_groups[0]["lr"], step=epoch)

            if epoch % 10 == 0:
                serialize(operator)

    serialize(operator)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(train_losses)), train_losses, label="train")
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(val_losses)), val_losses, label="val")
    axs[0].set_yscale("log")

    for ax in axs.flatten():
        ax.legend()

    plt.show()


if __name__ == "__main__":
    run()
