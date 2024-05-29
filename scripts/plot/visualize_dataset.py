import pathlib

import matplotlib.pyplot as plt
import torch

from nos.data import (
    PulsatingSphere,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)

# dataset = PressureBoundaryDataset(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"))
dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"))

real_pressure, _ = torch.max(torch.abs(dataset.v[:, :, 0]), axis=1)
imag_pressure, _ = torch.max(torch.abs(dataset.v[:, :, 1]), axis=1)
frequencies = dataset.u[:, 0, 4]

"""sns.histplot(
    x=real_pressure,
    bins=torch.logspace(torch.log10(min(real_pressure)), torch.log10(max(real_pressure)), 31),
    weights=torch.ones(len(real_pressure)) / len(real_pressure),
    ax=axs[0],
    fill=False,
    color='black'
)
sns.histplot(
    x=imag_pressure,
    bins=torch.logspace(torch.log10(min(imag_pressure)), torch.log10(max(imag_pressure)), 31),
    weights=torch.ones(len(imag_pressure)) / len(imag_pressure),
    ax=axs[1],
    fill=False,
    color='black'
)
axs[0].set_xlabel(r'$\lvert \Re\{p\} \rvert$')
axs[1].set_xlabel(r'$\lvert \Im\{p\} \rvert$')

axs[0].set_ylabel(r'Fraction of all observations')
for ax in axs.flatten():
    ax.set_xscale('log')
    ax.grid()"""

fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

axs[0].plot(frequencies, real_pressure, "kx")
axs[0].set_ylabel(r"$\lvert \Re\{p\} \rvert$")
axs[1].plot(frequencies, imag_pressure, "kx")
axs[1].set_ylabel(r"$\lvert \Im\{p\} \rvert$")

for ax in axs.flatten():
    ax.set_yscale("log")
    ax.grid()
    ax.set_xlabel(r"Frequency [Hz]")


fig.tight_layout()
plt.savefig("narrow_data_freq_vs_magnitude.pdf")
