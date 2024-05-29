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
    TLDatasetCompact,
    TLDatasetCompactWave,
)
from nos.operators import (
    DeepDotOperator,
)
from nos.trainers import (
    Trainer,
)

BATCH_SIZE = 12
N_EPOCHS = 1000
LR = 1e-3
IS_FNO = False


def main():
    if IS_FNO:
        dataset_class = TLDatasetCompactWave
    else:
        dataset_class = TLDatasetCompact

    dataset = dataset_class(
        path=pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"),
        n_samples=-1,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    operator = DeepDotOperator(
        dataset.shapes,
        branch_width=32,
        branch_depth=16,
        trunk_width=32,
        trunk_depth=16,
        dot_width=46,
        dot_depth=32,
        stride=4,
    )
    logger.info("Operator initialized.")

    optimizer = torch.optim.Adam(operator.parameters(), lr=LR)
    # scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=5e-5)
    scheduler = sched.ConstantLR(optimizer=optimizer)

    trainer = Trainer(
        operator=operator,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        max_epochs=N_EPOCHS,
        lr_scheduler=scheduler,
        batch_size=BATCH_SIZE,
    )

    run_name = f"{operator.__class__.__name__}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    logger.info("Trainer initialized.")
    trainer(dataset, run_name=run_name)
    logger.info("Finished all jobs.")


if __name__ == "__main__":
    main()
