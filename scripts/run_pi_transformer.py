import pathlib
from typing import (
    Union,
)

import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as sched
from continuity.operators import (
    Operator,
)
from continuity.pde import (
    Grad,
)
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
    PressureBoundaryDataset,
    SelfSupervisedDataset,
)
from nos.operators import (
    AttentionOperator,
    serialize,
)
from nos.transforms import (
    QuantileScaler,
)

FREQUENCY = 500.0
LMBDA = 343.0 / FREQUENCY
K0 = 2 * torch.pi / LMBDA
N_EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 128
N_COL_SAMPLES = 1024
PDE_LOSS_START = N_EPOCHS // 10
PDE_LOSS_FULL = PDE_LOSS_START + N_EPOCHS // 10


class Laplace(Operator):
    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        gradients = Grad()(x, u)
        return torch.sum(Grad()(x, gradients), dim=-1, keepdim=True)


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


def helmholtz_domain_residual(_, u, y, v):
    residual = Laplace()(y, v) + K0**2 * v
    residual = residual**2
    return torch.mean(residual)


def weight_scheduler(epoch: Union[float, torch.tensor]) -> torch.tensor:
    if isinstance(epoch, int):
        epoch = epoch * torch.ones(1)

    fraction = epoch / N_EPOCHS

    def data_weight(frac: torch.tensor) -> torch.tensor:
        return torch.ones(frac.shape)

    def pde_weight(frac: torch.tensor) -> torch.tensor:
        start_frac = PDE_LOSS_START / torch.tensor([N_EPOCHS])
        end_frac = PDE_LOSS_FULL / torch.tensor([N_EPOCHS])
        out = (frac - start_frac) / (end_frac - start_frac)
        out[out < 0.0] = 0.0
        out[out > 1] = 1.0
        return out * 1e-3

    return torch.stack([data_weight(fraction), pde_weight(fraction)], dim=1)


def run():
    logger.info("Start run().")
    src_dataset = PressureBoundaryDataset(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"),
        n_samples=-1,
        n_observations=2,
    )
    logger.info("Src dataset loaded.")
    u_transform = QuantileScaler(src=src_dataset.u.detach())
    logger.info("Quantile U transformation initialized.")
    dataset = SelfSupervisedDataset(
        x=src_dataset.x.detach(),
        u=src_dataset.u.detach(),
        x_transform=src_dataset.transform["x"],
        u_transform=u_transform,
        n_input=128,
        n_combinations=1,
    )
    logger.info("dataset loaded.")
    train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    operator = AttentionOperator(dataset.shapes, encoding_dim=32, n_heads=4, n_layers=1, dropout=0.25)
    logger.info("Operator initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS)
    criterion = torch.nn.MSELoss()
    logger.info("optimizer, scheduler, criterion initialized.")
    logger.info("Start training")

    pbar = tqdm(total=N_EPOCHS * (len(train_loader) + len(val_loader)))
    train_losses = []
    mean_train_loss = torch.inf
    val_losses = []
    mean_val_loss = torch.inf

    operator.to(device)

    cp = sample_collocations(n_samples=N_COL_SAMPLES)
    cp.requires_grad = True
    cp = cp.to(device)

    for epoch in range(N_EPOCHS):
        loss_weights = weight_scheduler(epoch)
        operator.train()
        for x, u, y, v in train_loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
            # compute output
            data_output = operator(x, u, y)
            data_loss = criterion(data_output, v)

            tmp_cp = cp.expand(x.size(0), -1, -1)
            tmp_cp = tmp_cp.to(device)
            tmp_scp = dataset.transform["x"](tmp_cp)
            tmp_scp = tmp_scp.to(device)

            pde_output = operator(x, u, tmp_scp)
            pde_true_output = dataset.transform["u"].undo(pde_output)
            pde_loss = helmholtz_domain_residual(None, None, tmp_cp, pde_true_output)

            loss = torch.stack([data_loss, pde_loss])
            loss = loss_weights @ loss

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            pbar.update()
            mean_train_loss = torch.mean(torch.tensor(train_losses[-10:]))
            pbar.set_description(
                f"Epoch: {epoch}/{N_EPOCHS},\t Train-Loss: {mean_train_loss: .5f},\t Val-Loss: {mean_val_loss: .5f}"
            )

        operator.eval()
        for x, u, y, v in val_loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
            # compute output
            data_output = operator(x, u, y)
            data_loss = criterion(data_output, v)

            tmp_cp = cp.expand(x.size(0), -1, -1)
            tmp_cp = tmp_cp.to(device)
            tmp_scp = dataset.transform["x"](tmp_cp)
            tmp_scp = tmp_scp.to(device)

            pde_output = operator(x, u, tmp_scp)
            pde_true_output = dataset.transform["u"].undo(pde_output)
            pde_loss = helmholtz_domain_residual(None, None, tmp_cp, pde_true_output)

            loss = torch.stack([data_loss, pde_loss])
            loss = loss_weights @ loss

            val_losses.append(loss.item())
            mean_val_loss = torch.mean(torch.tensor(train_losses[-10:]))
            pbar.update()
            pbar.set_description(
                f"Epoch: {epoch}/{N_EPOCHS},\t Train-Loss: {mean_train_loss: .5f},\t Val-Loss: {mean_val_loss: .5f}"
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
