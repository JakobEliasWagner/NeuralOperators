import json
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from loguru import logger

from continuity.operators import DeepONet
from nos.data import TLDatasetCompactExp
from nos.trainer import Trainer

DATA_DIR = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = TLDatasetCompactExp(DATA_DIR, n_samples=1)

    logger.info(f"Loaded dataset from {DATA_DIR}.")
    logger.info(f"Dataset size: {len(dataset)}")
    operator_config = {
        "branch_width": 1,
        "branch_depth": 1,
        "trunk_width": 32,
        "trunk_depth": 32,
        "basis_functions": 1,
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
        optimizer=torch.optim.Adam(operator.parameters(), lr=3e-5),
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

    for i, (x, u, y, v) in enumerate(dataset):
        x, u, y, v = x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0), v.unsqueeze(0)
        logger.info(f"Starting plot {i + 1} of {len(dataset)}.")
        fig, ax = plt.subplots()
        # plot reference
        idx = torch.argsort(y.squeeze())
        v_plot, y_plot = v[0, idx], y[0, idx]
        v_plot, y_plot = dataset.transform["v"].undo(v_plot), dataset.transform["y"].undo(y_plot)
        ax.plot(v_plot, y_plot, alpha=0.5, label="Reference")

        # generate prediction
        v_eval = operator(x, u, y)
        v_plot = dataset.transform["v"].undo(v_eval)
        ax.plot(v_plot.squeeze().detach().numpy(), y_plot.squeeze().detach().numpy(), ".", label="Prediction")

        #
        ax.legend()
        fig.tight_layout()
        plt.show()
