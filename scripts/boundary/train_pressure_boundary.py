import pathlib
import shutil
from datetime import (
    datetime,
)

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
    PressureBoundaryDataset,
)
from nos.operators import (
    DeepDotOperator,
    serialize,
)
from nos.utils import (
    UniqueId,
)

BATCH_SIZE = 128
N_EPOCHS = 200
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PressureBoundaryDataset(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"),
        n_samples=-1,
    )

    for trf in dataset.transform.keys():
        dataset.transform[trf].to(device)
    dataset.x = dataset.x.to(device)
    dataset.u = dataset.u.to(device)
    dataset.y = dataset.y.to(device)
    dataset.v = dataset.v.to(device)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    criterion = torch.nn.MSELoss()
    criterion.to(device)

    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=64,
        branch_depth=16,
        trunk_depth=4,
        trunk_width=64,
        dot_depth=4,
        dot_width=64,
        stride=2,
    )
    operator.to(device)
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)

    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=5e-5)

    run_name = f"pb_{operator.__class__.__name__}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    uid = UniqueId()
    out_dir = pathlib.Path.cwd().joinpath("out_models", str(uid))

    best_val_loss = torch.inf

    with mlflow.start_run():
        if run_name is not None:
            mlflow.set_tag("mlflow.runName", run_name)
        for epoch in tqdm(range(N_EPOCHS)):
            train_losses = []
            val_losses = []

            operator.train()
            for x, u, y, v in train_loader:
                output = operator(x, u, y)
                loss = criterion(output, v)

                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update metrics
                train_losses.append(loss.item())

            operator.eval()
            for x, u, y, v in val_loader:
                output = operator(x, u, y)
                loss = criterion(output, v)

                # update metrics
                val_losses.append(loss.item())

            scheduler.step(epoch)

            mean_val_loss = torch.mean(torch.tensor(val_losses)).item()
            mlflow.log_metric("Val loss", mean_val_loss, step=epoch)
            mlflow.log_metric("Train loss", torch.mean(torch.tensor(train_losses)).item(), step=epoch)
            mlflow.log_metric("LR", optimizer.param_groups[0]["lr"], step=epoch)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_dir = out_dir.joinpath("best")
                if best_dir.is_dir():
                    shutil.rmtree(best_dir)
                best_dir.mkdir(exist_ok=True, parents=True)

                serialize(operator, out_dir=best_dir)
        serialize(operator)


if __name__ == "__main__":
    main()
