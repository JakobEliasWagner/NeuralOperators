import pathlib

import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)

from nos.data import (
    TLDatasetCompactWave,
)
from nos.operators import (
    FourierNeuralOperator,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 16
N_EPOCHS = 1000
LR = 1e-3


def main():
    dataset = TLDatasetCompactWave(pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"))

    logger.info(f"Dataset size: {len(dataset)}")
    operator = FourierNeuralOperator(dataset.shapes, width=8, depth=4)
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = sched.CosineAnnealingWarmRestarts(optimizer, 16, 2)

    trainer = Trainer(
        operator=operator, criterion=torch.nn.MSELoss(), optimizer=optimizer, max_epochs=1000, lr_scheduler=scheduler
    )
    logger.info("Trainer initialized.")
    trainer(
        dataset,
    )
    logger.info("Finished all jobs.")


if __name__ == "__main__":
    main()
