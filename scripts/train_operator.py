import pathlib

import torch
import torch.optim.lr_scheduler as sched
from loguru import (
    logger,
)

from nos.data import (
    PressureBoundaryDataset,
    SelfSupervisedDataset,
)
from nos.operators import (
    AttentionOperator,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 2**10
N_EPOCHS = 100
LR = 1e-3


def main():
    src_dataset = PressureBoundaryDataset(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"),
        n_samples=-1,
        n_observations=2,
        do_normalize=True,
    )
    dataset = SelfSupervisedDataset(
        x=src_dataset.x,
        u=src_dataset.u,
        x_transform=src_dataset.transform["x"],
        u_transform=src_dataset.transform["u"],
        input_ratio=0.099,
        n_combinations=2**5,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    operator = AttentionOperator(dataset.shapes, width=64, n_heads=16, n_layers=4, dropout=0.5)
    # operator = DeepONet(dataset.shapes)
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
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
