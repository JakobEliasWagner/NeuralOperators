import pathlib
import re
from itertools import combinations
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import CloughTocher2DInterpolator

from nos.data.helmholtz import HelmholtzDataset

from .report import Report

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams["pgf.texsystem"] = "pdflatex"


class PressureReport(Report):
    @staticmethod
    def run(out_dir: pathlib.Path, data_sets: Dict[str, HelmholtzDataset]) -> None:
        pr_out = out_dir.joinpath("pressure")
        pr_out.mkdir()

        PressureReport.plot_difference(pr_out, data_sets)

    @staticmethod
    def plot_difference(out_dir: pathlib.Path, data_sets: Dict[str, HelmholtzDataset]) -> None:
        studies = combinations(data_sets.items(), 2)
        for d1, d2 in studies:
            name1, dset1 = d1
            name2, dset2 = d2
            study_id = "_".join([re.sub(r"[^\w\-_\. ]", "_", name) for name in [name1, name2]])

            bbox1 = dset1.description.crystal_box
            relevant_values1 = np.where(bbox1.distance(dset1.x.T) < 1e-8)[0]
            bbox2 = dset2.description.crystal_box
            relevant_values2 = np.where(bbox2.distance(dset2.x.T) < 1e-8)[0]

            n_points = 50**2
            ratio = bbox1.size[1] / bbox1.size[0]
            x_plot = np.linspace(bbox1.x_min, bbox1.x_max, int(n_points * (1 - ratio)))
            y_plot = np.linspace(bbox1.y_min, bbox1.y_max, int(n_points * ratio))
            xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)

            # comparable values
            f_comp = []
            for i, freq in enumerate(dset1.frequencies):
                if freq in dset2.frequencies:
                    f_comp.append((i, dset2.frequencies.tolist().index(freq)))

            # interpolate functions
            differences = np.empty((len(f_comp), len(y_plot), len(x_plot)))
            v_max = 0.0
            for i, (idx1, idx2) in enumerate(f_comp):
                freq = dset1.frequencies[idx1]
                # set 1
                p1 = dset1.p[idx1, relevant_values1]
                interp1 = CloughTocher2DInterpolator(dset1.x[relevant_values1, :2], p1, fill_value=0.0)
                vals1 = np.real(interp1(xx_plot, yy_plot))
                # set 2
                p2 = dset2.p[idx2, relevant_values2]
                interp2 = CloughTocher2DInterpolator(dset2.x[relevant_values2, :2], p2, fill_value=0.0)
                vals2 = np.real(interp2(xx_plot, yy_plot))

                differences[i, :, :] = vals1 - vals2

            v_max = np.max(np.abs(differences))

            fig, ax = plt.subplots()
            maxes = np.max(np.abs(differences), axis=(1, 2))
            plt.plot([dset1.frequencies[idx1] for idx1, _ in f_comp], maxes)
            plt.savefig(out_dir.joinpath(f"{study_id}_real_diff_over_f.pdf"))
            plt.clf()

            for i, (idx1, idx2) in enumerate(f_comp):
                freq = dset1.frequencies[idx1]
                file = out_dir.joinpath(f"{study_id}_real_difference_{freq:.0f}.png")

                fig, ax = plt.subplots(figsize=(20 * (1 - ratio), 20 * ratio))
                sns.heatmap(
                    data=differences[i, :, :],
                    ax=ax,
                    vmax=v_max,
                    vmin=-v_max,
                    xticklabels=False,
                    yticklabels=False,
                    cmap=sns.diverging_palette(220, 20, as_cmap=True),
                )
                plt.title(f"Difference {name1} and {name2} ({freq:.0f}Hz)")
                plt.tight_layout()
                plt.savefig(file)
