import pathlib

import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)

from nos.data import (
    PressureBoundaryDataset,
)
from nos.operators import (
    DeepONet,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 64
N_EPOCHS = 10000
LR = 1e-3


def main():
    dataset = PressureBoundaryDataset(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_450"), n_samples=-1, n_observations=10
    )

    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepONet(dataset.shapes)
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS)

    trainer = Trainer(
        operator=operator,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        max_epochs=N_EPOCHS,
        lr_scheduler=scheduler,
    )
    logger.info("Trainer initialized.")
    trainer(
        dataset,
    )
    logger.info("Finished all jobs.")


if __name__ == "__main__":
    main()
