import pathlib

import matplotlib.pyplot as plt
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.operators import (
    Operator,
)
from tabulate import (
    tabulate,
)
from torch.utils.data import (
    DataLoader,
)

from nos.data import (
    PressureBoundaryDataset,
)
from nos.operators import (
    deserialize,
)

# plt.rcParams["text.usetex"] = True


def plot_three_tile_plot(dataset: OperatorDataset, operator: Operator, out_dir: pathlib.Path):
    n_plot = 5
    out_dir = out_dir.joinpath("three_tile")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loader = DataLoader(dataset, batch_size=1)
    operator.eval()
    for i, (x, u, y, v) in enumerate(plot_loader):
        if i >= n_plot:
            break
        v_pred = operator(x, u, y)

        x_plot = dataset.transform["x"].undo(x).squeeze().detach()
        u_plot = dataset.transform["u"].undo(u).squeeze().detach()
        y_plot = dataset.transform["y"].undo(y).squeeze().detach()
        v_plot = dataset.transform["v"].undo(v).squeeze().detach()
        v_pred_plot = dataset.transform["v"].undo(v_pred).squeeze().detach()

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        v = torch.max(torch.abs(u_plot[:, 0]))
        im = axs[1, 0].tricontourf(
            x_plot[:, 0], x_plot[:, 1], u_plot[:, 0], levels=256, cmap="seismic", vmin=-v, vmax=v
        )
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[1, 0].add_patch(circ1)
        axs[1, 0].add_patch(circ2)
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[1, 0])
        axs[1, 0].title.set_text("Real pressure field")

        v = torch.max(torch.abs(u_plot[:, 1]))
        im = axs[0, 1].tricontourf(
            x_plot[:, 0], x_plot[:, 1], u_plot[:, 1], levels=256, cmap="seismic", vmin=-v, vmax=v
        )
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[0, 1].add_patch(circ1)
        axs[0, 1].add_patch(circ2)
        axs[0, 1].set_xlim(0, 1)
        axs[0, 1].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[0, 1])
        axs[0, 1].title.set_text("Imag pressure field")

        axs[0, 0].plot(y_plot, torch.abs(v_plot[:, 0]), "r--", label="$\Re(Y_{gt})(x)$")
        axs[0, 0].plot(y_plot, torch.abs(v_plot[:, 1]), "g--", label="$-\Im(Y_{gt})(x)$")
        axs[0, 0].plot(y_plot, torch.abs(v_pred_plot[:, 0]), "r-", label="$\Re(Y_{pred})(x)$")
        axs[0, 0].plot(y_plot, torch.abs(v_pred_plot[:, 1]), "g-", label="$-\Im(Y_{pred})(x)$")
        axs[0, 0].set_xlabel("x [m]")
        axs[0, 0].set_ylabel("Y [1]")
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].title.set_text("Admittance on top boundary")
        axs[0, 0].legend()

        axs[1, 1].plot(torch.abs(v_plot[:, 2]), y_plot, "r--", label="$\Re(Y_{gt})(y)$")
        axs[1, 1].plot(torch.abs(v_plot[:, 3]), y_plot, "g--", label="$-\Im(Y_{gt})(y)$")
        axs[1, 1].plot(torch.abs(v_pred_plot[:, 2]), y_plot, "r-", label="$\Re(Y_{pred})(y)$")
        axs[1, 1].plot(torch.abs(v_pred_plot[:, 3]), y_plot, "g-", label="$-\Im(Y_{pred})(y)$")
        axs[1, 1].set_xlabel("Y [1]")
        axs[1, 1].set_ylabel("y [m]")
        axs[1, 1].legend()
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].title.set_text("Admittance on right boundary")

        fig.suptitle("")
        fig.tight_layout()

        plt_path = out_dir.joinpath(f"plot_{i}.png")
        plt.savefig(plt_path)
        plt.close(fig)


def print_performance(dataset: OperatorDataset, operator: Operator):
    eval_loader = DataLoader(dataset, batch_size=1)
    operator.eval()
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()

    mses = []
    l1s = []

    for i, (x, u, y, v) in enumerate(eval_loader):
        v_pred = operator(x, u, y)

        v_unscaled = dataset.transform["v"].undo(v)
        v_pred_unscaled = dataset.transform["v"].undo(v_pred)

        mses.append(mse(v_unscaled, v_pred_unscaled).item())
        l1s.append(l1(v_unscaled, v_pred_unscaled).item())

    tab = tabulate(
        [
            (
                torch.mean(torch.tensor(mses)),
                torch.mean(torch.tensor(l1s)),
                torch.median(torch.tensor(mses)),
                torch.median(torch.tensor(l1s)),
            )
        ],
        headers=("Mean SE", "Mean L1", "Median SE", "Median L1"),
    )
    print(tab)


def main():
    # load operator
    operator_dir = pathlib.Path.cwd().joinpath(
        "finished_pressure",
        "parf",
        "ddo",
        "2024_05_10_01_01_30-c79b7adc-4191-4900-a7d1-2bf5cad69e09",
        "best_mean",
        "DeepDotOperator_2024_05_10_01_24_38",
    )
    operator = deserialize(operator_dir)

    # load dataset
    data_path = pathlib.Path.cwd().joinpath("data", "test", "pulsating_sphere_narrow")
    scale_path = pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow")
    dataset_scale = PressureBoundaryDataset(data_dir=scale_path, n_samples=-1)
    dataset = PressureBoundaryDataset(data_dir=data_path, n_samples=-1)
    dataset.transform = dataset_scale.transform

    # out dir
    out_dir = pathlib.Path.cwd().joinpath("out", operator_dir.name)
    # print performance on test set
    print_performance(dataset, operator)

    # plotting
    plot_three_tile_plot(dataset, operator, out_dir)


if __name__ == "__main__":
    main()
