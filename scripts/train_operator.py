import json
import pathlib
from datetime import datetime

import torch
from loguru import logger

from continuity.operators import DeepONet
from nos.data import TLDatasetCompact
from nos.trainer import Trainer

DATA_DIR = pathlib.Path.cwd().joinpath("data", "transmission_loss_const_gap")
TRAIN_PATH = DATA_DIR.joinpath("dset.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = TLDatasetCompact(TRAIN_PATH)

    logger.info(f"Loaded dataset from {TRAIN_PATH}.")
    logger.info(f"Dataset size: {len(dataset)}")
    operator_config = {
        "branch_width": 256,
        "branch_depth": 16,
        "trunk_width": 256,
        "trunk_depth": 4,
        "basis_functions": 32,
    }
    operator = DeepONet(dataset.shapes, **operator_config)
    logger.info("DeepONet initialized.")

    time_stamp = datetime.now()
    name = f"{operator.__class__.__name__}_{time_stamp.strftime('%Y_%m_%d_%H_%M_%S')}"
    out_dir = pathlib.Path.cwd().joinpath("models", name)
    out_dir.mkdir(parents=True, exist_ok=False)

    logger.info(f"Output directory: {out_dir}.")

    trainer = Trainer(
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(operator.parameters(), lr=1e-3),
    )
    logger.info("Trainer initialized.")
    logger.info("Starting training...")
    optimized_don = trainer(operator, dataset, max_epochs=1000, batch_size=1)
    logger.info("...Finished training")

    logger.info("Starting to write model.")
    torch.save(optimized_don.state_dict(), out_dir.joinpath("model.pt"))
    with open(out_dir.joinpath("model_parameters.json"), "w") as file_handle:
        json.dump(operator_config, file_handle)
    logger.info("Finished all jobs.")
