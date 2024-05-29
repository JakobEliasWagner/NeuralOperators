import pathlib

import matplotlib.pyplot as plt
import torch
from torch.utils.data import (
    DataLoader,
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


def sample_collocations(n_samples: int):
    collocations = []
    while len(collocations) < n_samples:
        sample = torch.rand(2)
        if torch.isclose(sample, torch.zeros(2)).any():
            continue
        if torch.sqrt(torch.sum(sample**2)).item() <= 0.2:
            continue
        collocations.append(sample)
    collocations = torch.stack(collocations)
    collocations = torch.cat([collocations, torch.zeros(collocations.size(0), 1)], dim=-1)
    return collocations.unsqueeze(0)


train_dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"))
test_dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "test", "pulsating_sphere_500"))
test_dataset.transform = train_dataset.transform

operator = deserialize(pathlib.Path.cwd().joinpath("out_models", "DeepDotOperator_2024_04_08_14_21_47"))

plot_loader = DataLoader(test_dataset, batch_size=1)

pde_res = HelmholtzDomainResidual()
cp = sample_collocations(n_samples=2**13)
cp.requires_grad = True

for i, (x, u, y, v) in enumerate(plot_loader):
    if i >= 5:
        break

    y.require_grad = True

    v_pred = operator(x, u, y)

    y_plot = test_dataset.transform["y"].undo(y)
    v_plot = test_dataset.transform["v"].undo(v)
    v_pred_plot = test_dataset.transform["v"].undo(v_pred)

    tmp_cp = cp.expand(x.size(0), -1, -1)
    tmp_scp = test_dataset.transform["y"](tmp_cp)
    pde_parameters = test_dataset.transform["u"].undo(u)
    f = pde_parameters[:, :, -1]
    wl = 343.0 / f
    k = 2 * torch.pi / wl
    pde_output = operator(x, u, tmp_scp)
    pde_true_output = test_dataset.transform["v"].undo(pde_output)
    residual = pde_res(tmp_cp, pde_true_output, k)

    fig, ax = plt.subplots(ncols=3)
    ax[0].tricontourf(
        y_plot[0, :, 0].squeeze(),
        y_plot[0, :, 1].squeeze(),
        v_plot[0, :, 0].squeeze().detach(),
        levels=256,
        cmap="seismic",
    )
    ax[1].tricontourf(
        y_plot[0, :, 0].squeeze(),
        y_plot[0, :, 1].squeeze(),
        v_pred_plot[0, :, 0].squeeze().detach(),
        levels=256,
        cmap="seismic",
    )
    im = ax[2].tricontourf(
        tmp_cp[0, :, 0].squeeze().detach(),
        tmp_cp[0, :, 1].squeeze().detach(),
        residual[0, :, 0].squeeze().detach(),
        levels=256,
        cmap="seismic",
    )
    fig.colorbar(im, ax=ax[2])

    for axi in ax.flatten():
        axi.set_aspect("equal")
    fig.tight_layout()
    plt.show()
