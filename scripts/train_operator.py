import pathlib

import torch
import torch.optim.lr_scheduler as sched
from continuity.discrete import (
    UniformBoxSampler,
)
from loguru import (
    logger,
)

from nos.data import (
    IndicatorTLDataset,
)
from nos.operators import (
    DeepOBranchFNO,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 461
N_EPOCHS = 1000
LR = 1e-2


def main():
    dataset = IndicatorTLDataset(
        path=pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"),
        sampler=UniformBoxSampler([0.0, 0.0], [22e-3, 22e-3]),
        n_box_samples=2**10,
        n_samples=-1,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepOBranchFNO(dataset.shapes, trunk_width=64, trunk_depth=16, branch_width=8, branch_depth=8)
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=5e-3)
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
