import pathlib
from datetime import (
    datetime,
)
from itertools import (
    product,
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

BATCH_SIZE = 16
N_EPOCHS = 1000
LR = 1e-3
IS_FNO = False
N_RUNS = 5


def main():
    if IS_FNO:
        dataset_class = TLDatasetCompactWave
    else:
        dataset_class = TLDatasetCompact

    dataset_sizes = [
        35,
    ]
    freq_sizes = [
        172,
    ]

    confs = list(product(dataset_sizes, freq_sizes))
    out_dir = pathlib.Path.cwd().joinpath("run", "smaller_data")

    for run in range(N_RUNS):
        for i, (dataset_size, freq_size) in enumerate(confs):
            run_dir = out_dir.joinpath(f"run_{run}_dset_{dataset_size}_freq_{freq_size}_run_{i}")
            if run_dir.is_dir():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)

            dataset = dataset_class(
                path=pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"),
                n_samples=-1,
            )

            # apply size transformation
            perm = torch.randperm(len(dataset))
            indices = perm[:dataset_size]
            dataset.x = dataset.x[indices]
            dataset.u = dataset.u[indices]
            dataset.y = dataset.y[indices]
            dataset.v = dataset.v[indices]

            # apply frequency transformation
            perms = []
            for _ in range(dataset_size):
                perm = torch.randperm(dataset.shapes.y.num)
                indices = perm[:freq_size]
                perms.append(indices)
            perms = torch.stack(perms)
            dataset.y = torch.gather(dataset.y, 1, perms[:, :, None].expand(-1, -1, dataset.shapes.y.dim))
            dataset.v = torch.gather(dataset.v, 1, perms[:, :, None].expand(-1, -1, dataset.shapes.v.dim))

            # fix shapes
            dataset.shapes.y.num = freq_size
            dataset.shapes.v.num = freq_size

            logger.info(f"Starting run {run}.{i}, n_observations: {dataset_size}, n_evaluations: {freq_size}")
            logger.info(f"Dataset size: {len(dataset)}")
            operator = DeepDotOperator(
                dataset.shapes,
                branch_width=32,
                branch_depth=16,
                trunk_depth=16,
                trunk_width=32,
                dot_depth=32,
                dot_width=46,
                stride=4,
            )
            logger.info("Operator initialized.")

            optimizer = torch.optim.Adam(operator.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = sched.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=5e-5)

            trainer = Trainer(
                operator=operator,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                max_epochs=N_EPOCHS,
                lr_scheduler=scheduler,
                out_dir=run_dir,
                batch_size=BATCH_SIZE,
                max_n_logs=100,
            )

            run_name = f"{operator.__class__.__name__}_ds{dataset_size}_f{freq_size}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

            trainer(dataset, run_name=run_name)


if __name__ == "__main__":
    main()
