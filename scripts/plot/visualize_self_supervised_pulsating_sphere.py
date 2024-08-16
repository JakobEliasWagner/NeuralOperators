import pathlib

import matplotlib.pyplot as plt
import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.operators import (
    Operator,
)

from nos.data import (
    PressureBoundaryDataset,
)
from nos.operators import (
    deserialize,
)


def plot_super_res(dataset: OperatorDataset, operator: Operator, out_dir: pathlib.Path, n: int = 4, n_in: int = 16):
    out_dir.mkdir(parents=True, exist_ok=True)

    x, u, _, _ = dataset[:n]

    x_tmp = []
    u_tmp = []
    for xi, ui in zip(x, u):
        perm = torch.randperm(ui.size(0))
        in_indices = perm[:n_in]
        x_tmp.append(xi[in_indices])
        u_tmp.append(ui[in_indices])
    x_in = torch.stack(x_tmp)
    u_in = torch.stack(u_tmp)

    operator.eval()
    v_pred = operator(x_in, u_in, x)

    x_plot = dataset.transform["x"].undo(x).squeeze().detach()
    x_in_plot = dataset.transform["x"].undo(x_in).squeeze().detach()

    u_plot = dataset.transform["u"].undo(u).squeeze().detach()
    v_plot = dataset.transform["v"].undo(v_pred).squeeze().detach()

    for i, (xi, ui, vi, xi_in) in enumerate(zip(x_plot, u_plot, v_plot, x_in_plot)):
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))

        # real gt
        v_real = torch.max(torch.abs(ui[:, 0]))
        im = axs[0, 0].tricontourf(xi[:, 0], xi[:, 1], ui[:, 0], levels=256, cmap="seismic", vmin=-v_real, vmax=v_real)
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[0, 0].add_patch(circ1)
        axs[0, 0].add_patch(circ2)
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[0, 0])
        axs[0, 0].title.set_text("Real GT")
        axs[0, 0].plot(xi_in[:, 0], xi_in[:, 1], "kx")

        # imag gt
        v_imag = torch.max(torch.abs(ui[:, 1]))
        im = axs[1, 0].tricontourf(xi[:, 0], xi[:, 1], ui[:, 1], levels=256, cmap="seismic", vmin=-v_imag, vmax=v_imag)
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[1, 0].add_patch(circ1)
        axs[1, 0].add_patch(circ2)
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[1, 0])
        axs[1, 0].title.set_text("Imag GT")
        axs[1, 0].plot(xi_in[:, 0], xi_in[:, 1], "kx")

        # real pred
        im = axs[0, 1].tricontourf(xi[:, 0], xi[:, 1], vi[:, 0], levels=256, cmap="seismic", vmin=-v_real, vmax=v_real)
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[0, 1].add_patch(circ1)
        axs[0, 1].add_patch(circ2)
        axs[0, 1].set_xlim(0, 1)
        axs[0, 1].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[0, 1])
        axs[0, 1].title.set_text("Real prediction")

        # imag pred
        im = axs[1, 1].tricontourf(xi[:, 0], xi[:, 1], vi[:, 1], levels=256, cmap="seismic", vmin=-v_imag, vmax=v_imag)
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[1, 1].add_patch(circ1)
        axs[1, 1].add_patch(circ2)
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[1, 1])
        axs[1, 1].title.set_text("Imag prediction")

        # real diff
        vi_real = vi[:, 0] - ui[:, 0]
        im = axs[0, 2].tricontourf(xi[:, 0], xi[:, 1], vi_real, levels=256, cmap="seismic", vmin=-v_real, vmax=v_real)
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[0, 2].add_patch(circ1)
        axs[0, 2].add_patch(circ2)
        axs[0, 2].set_xlim(0, 1)
        axs[0, 2].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[0, 2])
        axs[0, 2].title.set_text("Real Diff")

        # imag diff
        im = axs[1, 2].tricontourf(
            xi[:, 0], xi[:, 1], vi[:, 1] - ui[:, 1], levels=256, cmap="seismic", vmin=-v_imag, vmax=v_imag
        )
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        axs[1, 2].add_patch(circ1)
        axs[1, 2].add_patch(circ2)
        axs[1, 2].set_xlim(0, 1)
        axs[1, 2].set_ylim(0, 1)
        fig.colorbar(im, ax=axs[1, 2])
        axs[1, 2].title.set_text("Imag Diff")

        for ax in axs.flatten():
            ax.set_aspect("equal")

        fig.suptitle("")
        fig.tight_layout()

        plt_path = out_dir.joinpath(f"super_res_{i}.png")
        plt.savefig(plt_path)
        plt.close(fig)


def main():
    # load operator
    operator_dir = pathlib.Path.cwd().joinpath(
        "run", "2024_04_01_15_16_53-685b8b62-458d-4dc1-9b9f-530a8cb6ad03", "AttentionOperator_2024_04_01_20_06_56"
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
    plot_super_res(dataset, operator, out_dir, n_in=32)


if __name__ == "__main__":
    main()
