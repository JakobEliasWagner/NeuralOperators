import pathlib
from datetime import (
    datetime,
)

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
    DeepDotOperator,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 128
N_EPOCHS = 100
LR = 1e-3


def main():
    src_dataset = PressureBoundaryDataset(
        data_dir=pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"),
        n_samples=-1,
    )

    dataset = SelfSupervisedDataset(
        x=src_dataset.x,
        u=src_dataset.u,
        x_transform=src_dataset.transform["x"],
        u_transform=src_dataset.transform["u"],
        n_input=32,
        n_combinations=1,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=32,
        branch_depth=8,
        trunk_depth=2,
        trunk_width=16,
        dot_depth=8,
        dot_width=32,
        stride=2,
    )
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=5e-5)
    # scheduler = sched.ConstantLR(optimizer=optimizer)

    trainer = Trainer(
        operator=operator,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        max_epochs=N_EPOCHS,
        lr_scheduler=scheduler,
        batch_size=BATCH_SIZE,
    )

    run_name = f"pb_{operator.__class__.__name__}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    logger.info("Trainer initialized.")
    trainer(dataset, run_name=run_name)
    logger.info("Finished all jobs.")


if __name__ == "__main__":
    main()
