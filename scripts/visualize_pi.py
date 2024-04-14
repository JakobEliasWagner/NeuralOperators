import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

train_dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"))
test_dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "test", "pulsating_sphere_narrow"))
test_dataset.transform = train_dataset.transform
del test_dataset.transform["v"]

"""operator = deserialize(pathlib.Path.cwd().joinpath("finished_pi",
                                                   "DeepDotOperator-narrow-data",
                                                   "DeepDotOperator_2024_04_13_21_59_24"))"""
operator = deserialize(
    pathlib.Path.cwd().joinpath(
        "out_models", "2024_04_14_12_46_52-28bf3ff1-9e2f-4ebd-9e01-7fc095d7bac8", "DeepDotOperator_2024_04_14_13_01_54"
    )
)

plot_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for i, (x, u, y, v) in enumerate(plot_loader):
    if i >= 5:
        break

    max_vals, _ = torch.max(torch.abs(v), dim=1)
    max_vals = max_vals.view(-1, 1, v.size(-1))

    v_pred = operator(x, u, y)

    y_plot = test_dataset.transform["y"].undo(y)
    v_plot = v / max_vals
    # v_plot = test_dataset.transform["v"].undo(v)
    v_pred_plot = v_pred
    # v_pred_plot = test_dataset.transform["v"].undo(v_pred)
    u_true = test_dataset.transform["u"].undo(u)
    print(u_true.squeeze().tolist())

    fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True, figsize=(12, 10))

    # REAL
    v_real_max = torch.max(torch.abs(v_plot[0, :, 0]))
    ax[0, 0].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        v_plot[0, :, 0].squeeze().detach(),
        vmin=-v_real_max,
        vmax=v_real_max,
        levels=256,
        cmap="RdBu",
    )
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[0, 0].add_patch(circ1)
    ax[0, 0].add_patch(circ2)
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].title.set_text("Real GT")

    ax[0, 1].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        v_pred_plot[0, :, 0].squeeze().detach(),
        vmin=-v_real_max,
        vmax=v_real_max,
        levels=256,
        cmap="RdBu",
    )
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[0, 1].add_patch(circ1)
    ax[0, 1].add_patch(circ2)
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].title.set_text("Real Prediction")

    im = ax[0, 2].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        (v_pred_plot[0, :, 0].squeeze().detach() - v_plot[0, :, 0].squeeze().detach()) ** 2,
        levels=256,
        cmap="magma_r",
    )
    fig.colorbar(im, ax=ax[0, 2])
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[0, 2].add_patch(circ1)
    ax[0, 2].add_patch(circ2)
    ax[0, 2].set_xlim(0, 1)
    ax[0, 2].set_ylim(0, 1)
    ax[0, 2].title.set_text("Real Squared Error")

    # IMAG
    v_imag_max = torch.max(torch.abs(v_plot[0, :, 1]))
    ax[1, 0].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        v_plot[0, :, 1].squeeze().detach(),
        vmin=-v_imag_max,
        vmax=v_imag_max,
        levels=256,
        cmap="RdBu",
    )
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[1, 0].add_patch(circ1)
    ax[1, 0].add_patch(circ2)
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].title.set_text("Imag GT")

    ax[1, 1].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        v_pred_plot[0, :, 1].squeeze().detach(),
        vmin=-v_imag_max,
        vmax=v_imag_max,
        levels=256,
        cmap="RdBu",
    )
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[1, 1].add_patch(circ1)
    ax[1, 1].add_patch(circ2)
    ax[1, 1].set_xlim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].title.set_text("Imag Prediction")

    im = ax[1, 2].tricontourf(
        y_plot[0, :, 0].squeeze().detach(),
        y_plot[0, :, 1].squeeze().detach(),
        (v_pred_plot[0, :, 1].squeeze().detach() - v_plot[0, :, 1].squeeze().detach()) ** 2,
        levels=256,
        cmap="magma_r",
    )
    fig.colorbar(im, ax=ax[1, 2])
    circ1 = plt.Circle((0.0, 0.0), 0.2, color="white")
    circ2 = plt.Circle((0.0, 0.0), 0.2, color="black", fill=False)
    ax[1, 2].add_patch(circ1)
    ax[1, 2].add_patch(circ2)
    ax[1, 2].set_xlim(0, 1)
    ax[1, 2].set_ylim(0, 1)
    ax[1, 2].title.set_text("Imag Squared Error")

    for axi in ax.flatten():
        axi.set_aspect("equal")

    plt.show()


df = pd.DataFrame()

mse_crit = torch.nn.MSELoss()

for i, (x, u, y, v) in enumerate(plot_loader):
    max_vals, _ = torch.max(torch.abs(v), dim=1)
    max_vals = max_vals.view(-1, 1, v.size(-1))

    v_pred = operator(x, u, y)
    v_scaled = v / max_vals
    u_unscaled = test_dataset.transform["u"].undo(u)

    loss = mse_crit(v_scaled, v_pred)
    tmp_df = pd.DataFrame(
        {
            "MSE": loss.item(),
            "Y1_real": [u_unscaled[0, 0, 0].item()],
            "Y1_imag": [u_unscaled[0, 0, 1].item()],
            "Y2_real": [u_unscaled[0, 0, 2].item()],
            "Y2_imag": [u_unscaled[0, 0, 3].item()],
            "frequency": [u_unscaled[0, 0, 4].item()],
        }
    )
    df = pd.concat([df, tmp_df], ignore_index=True)


fig, ax = plt.subplots()
sns.scatterplot(df, x="frequency", y="MSE", ax=ax)
ax.set_yscale("log")
plt.show()
