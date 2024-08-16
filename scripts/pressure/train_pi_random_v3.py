import pathlib
import shutil

import mlflow
import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)
from torch.utils.data import (
    DataLoader,
)
from tqdm import (
    tqdm,
)

from nos.data import (
    PulsatingSphere,
)
from nos.operators import (
    deserialize,
    serialize,
)
from nos.physics import (
    HelmholtzDomainMSE,
)
from nos.utils import (
    UniqueId,
)

N_EPOCHS = 10000
LR = 5e-3
BATCH_SIZE = 1
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uid = UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("out_models", str(uid))

    torch.manual_seed(42)

    logger.info("Start run().")
    dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_paper"),
        n_samples=-1,
    )

    val_dataset = PulsatingSphere(
        data_dir=pathlib.Path.cwd().joinpath("data", "test", "pulsating_sphere_paper"),
        n_samples=-1,
    )
    val_dataset.transform = dataset.transform
    for trf in dataset.transform.keys():
        val_dataset.transform[trf].to(device)
    val_dataset.x = val_dataset.x.to(device)
    val_dataset.u = val_dataset.u.to(device)
    val_dataset.y = val_dataset.y.to(device)
    val_dataset.v = val_dataset.v.to(device)

    logger.info("dataset loaded.")
    # train_set, val_set = random_split(dataset, [0.9, 0.1])
    train_set = dataset
    val_set = val_dataset

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info("train/val test split performed.")

    """operator = DeepDotOperator(
        dataset.shapes,
        branch_width=32,
        branch_depth=12,
        trunk_depth=12,
        trunk_width=32,
        dot_depth=12,
        dot_width=32,
        stride=4,
    )"""
    operator = deserialize(
        pathlib.Path.cwd().joinpath("finished_pressure", "paper", "ddo_wo", "DeepDotOperator_2024_05_07_12_53_38-3")
    )
    logger.info("Operator initialized.")

    logger.info(f"Running on device: {device}")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)
    # scheduler = sched.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)
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

    operator.to(device)

    min_us, _ = torch.min(dataset.u.view(-1, 5), dim=0)
    max_us, _ = torch.max(dataset.u.view(-1, 5), dim=0)
    delta_us = max_us - min_us
    min_us = min_us.to(device)
    delta_us = delta_us.to(device)

    cp = sample_collocations(n_samples=N_COL_SAMPLES).to(device)
    cp = cp.to(device)
    cp.requires_grad = True

    loss_weights_initial = torch.tensor([1.0, 1e-3]).to(device)
    loss_weights = loss_weights_initial
    loss_weights.to(device)

    best_val_data_loss = torch.inf
    best_val_pde_loss = torch.inf
    best_val_75_loss = torch.inf

    with mlflow.start_run(run_name="PI-DDO"):
        operator.eval()
        val_losses = []
        val_pde_losses = []
        val_data_losses = []
        for x, u, y, v in val_loader:
            # data part
            data_output = operator(x, u, y)
            data_loss = data_criterion(data_output, v)

            # prepare pde part
            tmp_cp = cp.expand(BATCH_SIZE, -1, -1).to(device)
            tmp_scp = dataset.transform["y"](tmp_cp).to(device)
            u_pde = torch.rand(BATCH_SIZE, 1, 5).to(device) * delta_us + min_us
            u_pde.requires_grad = True
            f = u_pde[:, :, -1]
            k = 2 * torch.pi * f / 343.0
            u_pde_in = dataset.transform["u"](u_pde).to(device)

            # pde loss
            pde_output = operator(u_pde_in, u_pde_in, tmp_scp)
            pde_loss = pde_criterion(tmp_cp.to(device), pde_output, k.to(device))

            # overall loss
            loss = torch.stack([data_loss, pde_loss]).to(device)
            loss = loss_weights @ loss

            # update metrics
            val_data_losses.append(data_loss.item())
            val_pde_losses.append(pde_loss.item())
            val_losses.append(loss.item())
        mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses)).item(), step=-1)
        mlflow.log_metric("Val PDE loss", torch.mean(torch.tensor(val_pde_losses)).item(), step=-1)
        mean_val_data_loss = torch.mean(torch.tensor(val_data_losses)).item()
        mlflow.log_metric("Val DATA loss", mean_val_data_loss, step=-1)

        for epoch in range(N_EPOCHS):
            train_losses = []
            train_pde_losses = []
            train_data_losses = []
            operator.train()
            for x, u, y, v in train_loader:
                # data part
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v)

                # prepare pde part
                tmp_cp = cp.expand(BATCH_SIZE, -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                u_pde = torch.rand(BATCH_SIZE, 1, 5).to(device) * delta_us + min_us
                u_pde.requires_grad = True
                f = u_pde[:, :, -1]
                k = 2 * torch.pi * f / 343.0

                u_pde_in = dataset.transform["u"](u_pde).to(device)

                # pde loss
                pde_output = operator(u_pde_in, u_pde_in, tmp_scp)
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
            val_losses = []
            val_pde_losses = []
            val_data_losses = []
            for x, u, y, v in val_loader:
                # data part
                data_output = operator(x, u, y)
                data_loss = data_criterion(data_output, v)

                # prepare pde part
                tmp_cp = cp.expand(BATCH_SIZE, -1, -1).to(device)
                tmp_scp = dataset.transform["y"](tmp_cp).to(device)
                u_pde = torch.rand(BATCH_SIZE, 1, 5).to(device) * delta_us + min_us
                u_pde.requires_grad = True
                f = u_pde[:, :, -1]
                k = 2 * torch.pi * f / 343.0
                u_pde_in = dataset.transform["u"](u_pde).to(device)

                # pde loss
                pde_output = operator(u_pde_in, u_pde_in, tmp_scp)
                pde_loss = pde_criterion(tmp_cp.to(device), pde_output, k.to(device))

                # overall loss
                loss = torch.stack([data_loss, pde_loss]).to(device)
                loss = loss_weights @ loss

                # update metrics
                val_data_losses.append(data_loss.item())
                val_pde_losses.append(pde_loss.item())
                val_losses.append(loss.item())

            pbar.update()

            if epoch % 1000 == 0 and epoch > 0:
                cp = sample_collocations(n_samples=N_COL_SAMPLES).to(device)
                cp = cp.to(device)
                cp.requires_grad = True

            mlflow.log_metric("Val loss", torch.mean(torch.tensor(val_losses)).item(), step=epoch)
            mean_val_pde_loss = torch.mean(torch.tensor(val_pde_losses)).item()
            mlflow.log_metric("Val PDE loss", mean_val_pde_loss, step=epoch)
            mean_val_data_loss = torch.mean(torch.tensor(val_data_losses)).item()
            mlflow.log_metric("Val DATA loss", mean_val_data_loss, step=epoch)

            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_losses)).item(), step=epoch)
            mlflow.log_metric("Train PDE loss", torch.mean(torch.tensor(train_pde_losses)).item(), step=epoch)
            mlflow.log_metric("Train DATA loss", torch.mean(torch.tensor(train_data_losses)).item(), step=epoch)

            mlflow.log_metric("DATA weight", loss_weights[0].item(), step=epoch)
            mlflow.log_metric("PDE weight", loss_weights[1].item(), step=epoch)

            mlflow.log_metric("LR", scheduler.optimizer.param_groups[0]["lr"], step=epoch)

            scheduler.step(epoch)

            if epoch % (N_EPOCHS // 10) == 0:
                serialize(operator, out_dir=out_dir)

            if mean_val_data_loss < best_val_data_loss:
                best_val_data_loss = mean_val_data_loss
                best_dir = out_dir.joinpath("best_mean")
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                serialize(operator, out_dir=best_dir)

            if mean_val_pde_loss < best_val_pde_loss:
                best_val_pde_loss = mean_val_pde_loss
                best_dir = out_dir.joinpath("best_pde_mean")
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                serialize(operator, out_dir=best_dir)

            q_75_val_loss = torch.quantile(torch.tensor(val_data_losses), 0.75).item()
            if q_75_val_loss < best_val_75_loss:
                best_val_75_loss = q_75_val_loss
                best_dir = out_dir.joinpath("best_75")
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(parents=True, exist_ok=True)
                serialize(operator, out_dir=best_dir)

    serialize(operator, out_dir=out_dir)


if __name__ == "__main__":
    run()