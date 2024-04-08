import pathlib
import time

import matplotlib.pyplot as plt
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
    WeightSchedulerLinear,
)

FREQUENCY = 500.0
LMBDA = 343.0 / FREQUENCY
K0 = 2 * torch.pi / LMBDA
N_EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 128
N_COL_SAMPLES = 1024
PDE_LOSS_START = N_EPOCHS // 10
PDE_LOSS_FULL = PDE_LOSS_START + N_EPOCHS // 10


def sample_collocations(n_samples: int):
    collocations = []
    while len(collocations) < n_samples:
        sample = torch.rand(2)
        if torch.isclose(sample, torch.zeros(2)).any():
            continue
        if torch.sqrt(torch.sum(sample**2)).item() <= 0.2:
            continue
        collocations.append(sample)
    collocations = torch.stack(collocations)
    collocations = torch.cat([collocations, torch.zeros(collocations.size(0), 1)], dim=-1)
    return collocations.unsqueeze(0)


def run():
    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"),
        n_samples=-1,
    )
    logger.info("dataset loaded.")
    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=28,
        branch_depth=4,
        trunk_depth=4,
        trunk_width=32,
        dot_depth=4,
        dot_width=32,
        stride=2,
    )
    logger.info("Operator initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    for dim in dataset.transform.keys():
        dataset.transform[dim] = dataset.transform[dim].to(device)

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS)
    data_criterion = torch.nn.MSELoss().to(device)
    pde_criterion = HelmholtzDomainMSE().to(device)

    logger.info("optimizer, scheduler, criterion initialized.")
    logger.info("Start training")

    pbar = tqdm(total=N_EPOCHS * (len(train_loader) + len(val_loader)))
    train_losses = []
    mean_train_loss = torch.inf
    val_losses = []
    mean_val_loss = torch.inf

    load_times = []
    forward_times = []
    backward_times = []

    operator = operator.to(device)

    cp = sample_collocations(n_samples=N_COL_SAMPLES)
    cp.requires_grad = True
    cp = cp.to(device)

    weight_scheduler = WeightSchedulerLinear(PDE_LOSS_START, PDE_LOSS_FULL)
    weight_scheduler.to(device)

    for epoch in range(N_EPOCHS):
        loss_weights = weight_scheduler(epoch).to(device)
        operator.train()
        end = time.time()
        for x, u, y, v in train_loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
            t_loading_done = time.time()

            # data loss
            data_output = operator(x, u, y)
            data_loss = data_criterion(data_output, v)

            # pde loss
            tmp_cp = cp.expand(x.size(0), -1, -1)
            tmp_cp = tmp_cp.to(device)
            tmp_scp = dataset.transform["y"](tmp_cp)
            tmp_scp = tmp_scp.to(device)
            pde_output = operator(x, u, tmp_scp)
            pde_true_output = dataset.transform["v"].undo(pde_output).to(device)
            pde_parameters = dataset.transform["u"].undo(u).to(device)
            f = pde_parameters[:, :, -1]
            wl = 343.0 / f
            k = 2 * torch.pi / wl
            pde_loss = 1e-4 * pde_criterion(tmp_cp, pde_true_output, k)

            # overall loss
            loss = torch.stack([data_loss, pde_loss]).to(device)
            loss = loss_weights @ loss

            t_forward_done = time.time()
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            t_backward_done = time.time()

            # update times
            load_times.append(t_loading_done - end)
            forward_times.append(t_forward_done - t_loading_done)
            backward_times.append(t_backward_done - t_forward_done)

            pbar.update()
            mean_train_loss = torch.mean(torch.tensor(train_losses[-10:]))
            mean_load_time = torch.mean(torch.tensor(load_times[-10:]))
            mean_forward_time = torch.mean(torch.tensor(forward_times[-10:]))
            mean_backward_time = torch.mean(torch.tensor(backward_times[-10:]))
            pbar.set_description(
                f"Epoch: {epoch}/{N_EPOCHS},\t"
                f"Train-Loss: {mean_train_loss: .5f},\t"
                f"Val-Loss: {mean_val_loss: .5f},\t"
                f"O(load): {torch.log10(mean_load_time).item():1.2f}, "
                f"O(pass): {torch.log10(mean_forward_time).item():1.2f}, "
                f"O(back): {torch.log10(mean_backward_time).item():1.2f}"
            )

            end = time.time()

        operator.eval()
        for x, u, y, v in val_loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

            # data loss
            data_output = operator(x, u, y)
            data_loss = data_criterion(data_output, v)

            # pde loss
            tmp_cp = cp.expand(x.size(0), -1, -1)
            tmp_cp = tmp_cp.to(device)
            tmp_scp = dataset.transform["y"](tmp_cp)
            tmp_scp = tmp_scp.to(device)
            pde_output = operator(x, u, tmp_scp)
            pde_true_output = dataset.transform["v"].undo(pde_output).to(device)
            pde_parameters = dataset.transform["u"].undo(u).to(device)
            f = pde_parameters[:, :, -1]
            wl = 343.0 / f
            k = 2 * torch.pi / wl
            pde_loss = 1e-4 * pde_criterion(tmp_cp, pde_true_output, k)

            # overall loss
            loss = torch.stack([data_loss, pde_loss]).to(device)
            loss = loss_weights @ loss

            val_losses.append(loss.item())
            mean_val_loss = torch.mean(torch.tensor(train_losses[-10:]))
            pbar.update()
            pbar.set_description(
                f"Epoch: {epoch}/{N_EPOCHS},\t"
                f"Train-Loss: {mean_train_loss: .5f},\t"
                f"Val-Loss: {mean_val_loss: .5f},\t"
                f"O(load): {torch.log(mean_load_time).item():1.2f}, "
                f"O(pass): {torch.log(mean_forward_time).item():1.2f}, "
                f"O(back): {torch.log(mean_backward_time).item():1.2f}"
            )
        scheduler.step(epoch=epoch)
    serialize(operator)

    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(train_losses)), train_losses, label="train")
    axs[0].plot(torch.linspace(0, N_EPOCHS, len(val_losses)), val_losses, label="val")
    axs[0].set_yscale("log")

    axs[1].plot(torch.arange(0, N_EPOCHS), weight_scheduler(torch.arange(0, N_EPOCHS)), label="weights")

    for ax in axs.flatten():
        ax.legend()

    plt.show()


if __name__ == "__main__":
    run()
