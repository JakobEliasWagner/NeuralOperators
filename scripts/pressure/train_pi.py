import pathlib
import time

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
    DeepNeuralOperator,
    serialize,
)
from nos.physics import (
    HelmholtzDomainMSE,
)

N_EPOCHS = 10000
LR = 5e-3
BATCH_SIZE = 128
N_COL_SAMPLES = 2**11


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
    del dataset.transform["v"]

    logger.info("dataset loaded.")
    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    operator = DeepNeuralOperator(
        dataset.shapes,
        depth=8,
        width=32,
        stride=2,
    )
    logger.info("Operator initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-4)
    # scheduler = sched.ConstantLR(optimizer)
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

    pbar = tqdm(total=N_EPOCHS * (len(train_loader) + len(val_loader)))
    train_losses = []
    train_pde_losses = []
    train_data_losses = []
    val_losses = []
    val_pde_losses = []
    val_data_losses = []

    load_times = []
    load_pde_times = []
    forward_times = []
    backward_times = []

    operator = operator.to(device)

    cp = sample_collocations(n_samples=N_COL_SAMPLES).to(device)
    cp = cp.to(device)
    cp.requires_grad = True

    loss_weights_initial = torch.tensor([1.0, 0.0]).to(device)
    loss_weights = loss_weights_initial
    loss_weights.to(device)

    with mlflow.start_run(run_name="PI-DDO"):
        for epoch in range(N_EPOCHS):
            operator.train()
            end = time.time()
            for x, u, y, v in train_loader:
                # prepare data part
                t_loading_done = time.time()

                # prepare pde part
                tmp_cp = cp.expand(x.size(0), -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                pde_parameters = dataset.transform["u"].undo(u).to(device)
                f = pde_parameters[:, :, -1]
                wl = 343.0 / f
                k = 2 * torch.pi / wl

                t_process_pi_param = time.time()

                # data loss
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v)

                # pde loss
                pde_output = operator(x, u, tmp_scp)
                pde_true_output = dataset.transform["v"].undo(pde_output).to(device)
                pde_loss = pde_criterion(tmp_cp.to(device), pde_true_output, k.to(device))

                # overall loss
                loss = torch.stack([data_loss, pde_loss]).to(device)
                loss = loss_weights @ loss

                t_forward_done = time.time()
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_backward_done = time.time()

                # update metrics
                train_data_losses.append(data_loss.item())
                train_pde_losses.append(pde_loss.item())
                train_losses.append(loss.item())
                load_times.append(t_loading_done - end)
                load_pde_times.append(t_process_pi_param - t_loading_done)
                forward_times.append(t_forward_done - t_loading_done)
                backward_times.append(t_backward_done - t_forward_done)

                pbar.update()
                end = time.time()

            operator.eval()
            for x, u, y, v in val_loader:
                x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

                # data loss
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v)

                # pde loss
                tmp_cp = cp.expand(x.size(0), -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                pde_parameters = dataset.transform["u"].undo(u).to(device)
                f = pde_parameters[:, :, -1]
                wl = 343.0 / f
                k = 2 * torch.pi / wl
                pde_output = operator(x, u, tmp_scp)
                pde_true_output = dataset.transform["v"].undo(pde_output).to(device)
                pde_loss = pde_criterion(tmp_cp.to(device), pde_true_output, k.to(device))

                # overall loss
                loss = torch.stack([data_loss, pde_loss]).to(device)
                loss = loss_weights @ loss

                val_losses.append(loss.item())
                val_data_losses.append(data_loss.item())
                val_pde_losses.append(pde_loss.item())

                pbar.update()

            if epoch % 1000 == 500:
                mean_loss = torch.mean(torch.tensor(val_losses[-10:]))
                mean_pde_loss = torch.mean(torch.tensor(val_pde_losses[-10:]))

                if torch.isnan(mean_loss) or torch.isnan(mean_pde_loss):
                    mean_loss = torch.mean(torch.tensor(train_losses[-10:]))
                    mean_pde_loss = torch.mean(torch.tensor(train_pde_losses[-10:]))

                pde_weight = 1 * mean_loss / mean_pde_loss
                loss_weights = torch.tensor([1.0, pde_weight]).to(device)

            if epoch % 1000 == 0 and epoch > 0:
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

            mlflow.log_metric("Load", torch.mean(torch.tensor(load_times[-10:])).item(), step=epoch)
            mlflow.log_metric("Load PDE", torch.mean(torch.tensor(load_pde_times[-10:])).item(), step=epoch)
            mlflow.log_metric("Forward", torch.mean(torch.tensor(forward_times[-10:])).item(), step=epoch)
            mlflow.log_metric("Backward", torch.mean(torch.tensor(backward_times[-10:])).item(), step=epoch)

            if epoch % 1000 == 0:
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
