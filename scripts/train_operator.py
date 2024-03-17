import pathlib

import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)

from nos.data import (
    TLDatasetCompact,
)
from nos.operators import (
    DeepDotOperator,
)
from nos.trainers import (
    FastTrainer,
)

BATCH_SIZE = 4
N_EPOCHS = 1000
LR = 1e-3


def main():
    dataset_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss")
    dataset = TLDatasetCompact(path=dataset_path)

    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=32,
        branch_depth=32,
        trunk_width=32,
        trunk_depth=32,
        dot_width=32,
        dot_depth=32,
        stride=4,
    )
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
