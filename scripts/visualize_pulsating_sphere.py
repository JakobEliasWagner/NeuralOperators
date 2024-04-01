import pathlib

import matplotlib.pyplot as plt
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.operators import (
    Operator,
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
    out_dir = out_dir.joinpath("three_tile")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loader = DataLoader(dataset, batch_size=1)
    operator.eval()
    for i, (x, u, y, v) in enumerate(plot_loader):
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


def main():
    # load operator
    operator_dir = pathlib.Path.cwd().joinpath(
        "run", "2024_03_31_18_35_37-7bb54957-0781-4c0b-b4c1-099c25102fc6", "DeepONet_2024_03_31_18_56_24"
    )
    operator = deserialize(operator_dir)

    # load dataset
    data_path = pathlib.Path.cwd().joinpath("data", "test", "pulsating_sphere_narrow")
    scale_path = pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow")
    dataset_scale = PressureBoundaryDataset(data_dir=scale_path, n_samples=-1, do_normalize=True)
    dataset = PressureBoundaryDataset(data_dir=data_path, n_samples=5, do_normalize=True)
    dataset.transform = dataset_scale.transform

    # out dir
    out_dir = pathlib.Path.cwd().joinpath("out", operator_dir.name)

    # plotting
    plot_three_tile_plot(dataset, operator, out_dir)


if __name__ == "__main__":
    main()
