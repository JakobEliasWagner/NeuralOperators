import pathlib

import matplotlib.pyplot as plt
import torch
from continuity.pde import (
    Grad,
)

from nos.data import (
    PulsatingSphere,
)
from nos.operators import (
    deserialize,
)
from nos.physics import (
    HelmholtzDomainResidual,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)


def sample_collocations(n_samples: int):
    collocations = []
    while len(collocations) < n_samples:
        sample = torch.rand(2)
        if torch.isclose(sample, torch.zeros(2)).any():
            continue
        if torch.isclose(sample, torch.ones(2)).any():
            continue
        if torch.sqrt(torch.sum(sample**2)).item() <= 0.2:
            continue
        collocations.append(sample)
    collocations = torch.stack(collocations)
    return collocations.unsqueeze(0)


operators = [
    ("3000", deserialize(pathlib.Path.cwd().joinpath("out_models", "DeepNeuralOperator_2024_04_12_14_42_44"))),
]

train_set = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_paper"))
test_set = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_paper"))
test_set.transform = train_set.transform

residual = HelmholtzDomainResidual()

diffs = []
for name, operator in operators:
    fig, axs = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(20, 16))

    x, u, y, v = test_set[:]

    y_plot = test_set.transform["y"].undo(y)
    y = test_set.transform["y"](y_plot)
    v_plot = test_set.transform["v"].undo(v)

    v_p = operator(x, u, y)

    v_p_plot = test_set.transform["v"].undo(v_p)

    v_abs_max = torch.max(v_plot[0, :, 0])
    axs[0, 0].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_plot[0, :, 0].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[0, 0].title.set_text("Ground Truth")
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    axs[0, 1].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_p_plot[0, :, 0].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[0, 1].title.set_text("Prediction")
    axs[0, 2].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_plot[0, :, 0].detach() - v_p_plot[0, :, 0].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[0, 2].title.set_text("Difference")

    v_abs_max = torch.max(v_plot[0, :, 1])
    axs[1, 0].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_plot[0, :, 1].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[1, 0].title.set_text("Ground Truth")
    axs[1, 1].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_p_plot[0, :, 1].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[1, 1].title.set_text("Prediction")
    axs[1, 2].tricontourf(
        y_plot[0, :, 0].detach(),
        y_plot[0, :, 1].detach(),
        v_plot[0, :, 1].detach() - v_p_plot[0, :, 1].detach(),
        cmap="seismic",
        vmin=-v_abs_max,
        vmax=v_abs_max,
        levels=256,
    )
    axs[1, 2].title.set_text("Difference")

    print(name, torch.mean((v_plot - v_p_plot) ** 2).item())

    pde_parameters = test_set.transform["u"].undo(u)
    f = pde_parameters[:, :, -1]
    wl = 343.0 / f
    k = 2 * torch.pi / wl

    y_cp = sample_collocations(2**12).repeat(2, 1, 1)
    y_cp.requires_grad = True
    y_scp = test_set.transform["y"](y_cp)
    v_cp = operator(x, u, y_scp)
    v_cp_plot = test_set.transform["v"].undo(v_cp)
    res = residual(y_cp, v_cp_plot, k)

    im = axs[0, 3].tricontourf(
        y_cp[0, :, 0].detach(), y_cp[0, :, 1].detach(), res[0, :, 0].detach() ** 2, cmap="gist_stern_r", levels=256
    )
    fig.colorbar(im, ax=axs[0, 3])
    axs[0, 3].title.set_text("Squared Residual")
    im = axs[1, 3].tricontourf(
        y_cp[0, :, 0].detach(), y_cp[0, :, 1].detach(), res[0, :, 1].detach() ** 2, cmap="gist_stern_r", levels=256
    )
    fig.colorbar(im, ax=axs[1, 3])
    axs[1, 3].title.set_text("Squared Residual")

    fig.suptitle(name)

    diffs.append(v_plot.detach() - v_p_plot.detach())

    for ax in axs.flatten():
        ax.set_aspect("equal")
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.show()

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    im = axs[0, 0].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), v_p_plot[0, :, 0].detach(), cmap="seismic", levels=256
    )
    axs[0, 0].title.set_text("$\Re(p)$")
    fig.colorbar(im, ax=axs[0, 0])

    d_f_real = Grad()(y_plot, v_p_plot[:, :, 0])
    im = axs[0, 1].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), d_f_real[0, :, 0].detach(), cmap="seismic", levels=256
    )
    fig.colorbar(im, ax=axs[0, 1])
    axs[0, 1].title.set_text("$\Re(\partial p / \partial x)$")
    im = axs[0, 2].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), d_f_real[0, :, 1].detach(), cmap="seismic", levels=256
    )
    axs[0, 2].title.set_text("$\Re(\partial p / \partial y)$")
    fig.colorbar(im, ax=axs[0, 2])
    im = axs[1, 0].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), d_f_real[0, :, 0].detach(), cmap="seismic", levels=256
    )
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].title.set_text("$\Re(\partial p / \partial x)$")
    im = axs[2, 0].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), d_f_real[0, :, 1].detach(), cmap="seismic", levels=256
    )
    axs[2, 0].title.set_text("$\Re(\partial p / \partial y)$")
    fig.colorbar(im, ax=axs[2, 0])

    dd_f_real_x = Grad()(y_plot, d_f_real[:, :, 0])
    dd_f_real_y = Grad()(y_plot, d_f_real[:, :, 1])
    im = axs[1, 1].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), dd_f_real_x[0, :, 0].detach(), cmap="seismic", levels=256
    )
    fig.colorbar(im, ax=axs[1, 1])
    axs[1, 1].title.set_text("$\Re(\partial^2 p / \partial x^2)$")
    im = axs[1, 2].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), dd_f_real_x[0, :, 1].detach(), cmap="seismic", levels=256
    )
    axs[1, 2].title.set_text("$\Re(\partial^2 p / \partial x\partial y)$")
    fig.colorbar(im, ax=axs[1, 2])
    im = axs[2, 1].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), dd_f_real_y[0, :, 0].detach(), cmap="seismic", levels=256
    )
    fig.colorbar(im, ax=axs[2, 1])
    axs[2, 1].title.set_text("$\Re(\partial^2 p / \partial y \partial x)$")
    im = axs[2, 2].tricontourf(
        y_plot[0, :, 0].detach(), y_plot[0, :, 1].detach(), dd_f_real_y[0, :, 1].detach(), cmap="seismic", levels=256
    )
    axs[2, 2].title.set_text("$\Re(\partial^2 p / \partial y^2)$")
    fig.colorbar(im, ax=axs[2, 2])

    for ax in axs.flatten():
        ax.set_aspect("equal")
        circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
        circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()
