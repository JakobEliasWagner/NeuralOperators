import json
import pathlib

import matplotlib.pyplot as plt
import torch
from continuity.operators import (
    DeepONet,
)
from loguru import (
    logger,
)

from nos.data import (
    TLDatasetCompact,
)

TEST_PATH = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss")
TRAIN_PATH = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = pathlib.Path.cwd().joinpath("out", "transmission_loss")
MODEL_PATH = pathlib.Path.cwd().joinpath("models", "DeepONet_2024_03_11_15_50_09")


def visualize_all_sets():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
        data_set = TLDatasetCompact(path)
        local_out_dir = OUT_DIR.joinpath(name)
        local_out_dir.mkdir(parents=True, exist_ok=True)
        visualize_set(data_set, local_out_dir)


def visualize_set(data_set, out_dir):
    # operator
    with open(MODEL_PATH.joinpath("model_parameters.json"), "r") as fp:
        model_parameters = json.load(fp)
    operator = DeepONet(data_set.shapes, **model_parameters)
    logger.info(f"Initialized {operator.__class__.__name__}.")
    operator.load_state_dict(torch.load(MODEL_PATH.joinpath("model.pt")))
    logger.info(f"Loaded parameters from {MODEL_PATH} for {operator.__class__.__name__}.")
    operator.eval()

    # out dir

    y_plot = torch.linspace(2000, 20000, 2**11)
    y_eval = data_set.transform["y"](y_plot).reshape(1, -1, 1)
    for i, (x, u, y, v) in enumerate(data_set):
        logger.info(f"Starting plot {i + 1} of {len(data_set)}.")
        fig, ax = plt.subplots()
        # plot reference
        idx = torch.argsort(y.squeeze())
        v, y = v[idx], y[idx]
        v, y = data_set.transform["v"].undo(v), data_set.transform["y"].undo(y)
        ax.plot(v, y, alpha=0.5, label="Reference")

        # generate prediction
        v_eval = operator(x, u, y_eval)
        v_plot = data_set.transform["v"].undo(v_eval)
        ax.plot(v_plot.squeeze().detach(), y_plot, label="Prediction")

        #
        ax.legend()
        fig.tight_layout()
        plt.savefig(out_dir.joinpath(f"test_plot_{i}.png"))
        plt.close(fig)

    logger.info(f"Finished {len(data_set)} plots and saved them to {out_dir}.")


if __name__ == "__main__":
    visualize_all_sets()
