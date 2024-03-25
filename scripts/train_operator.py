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
    FastTrainer,
)

BATCH_SIZE = 16
N_EPOCHS = 1000
LR = 1e-3


def main():
    dataset = TLDatasetCompactWave(pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"))

    logger.info(f"Dataset size: {len(dataset)}")
    operator = FourierNeuralOperator(dataset.shapes, width=8, depth=4)
    logger.info("Operator initialized.")

    trainer = FastTrainer(
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(operator.parameters(), lr=LR),
    )
    logger.info("Trainer initialized.")
    trainer(
        operator,
        dataset,
        max_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        scheduler=sched.CosineAnnealingLR(trainer.optimizer, T_max=N_EPOCHS),
    )
    logger.info("Finished all jobs.")


if __name__ == "__main__":
    main()
