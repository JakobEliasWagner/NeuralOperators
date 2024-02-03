import pathlib
import re
from itertools import combinations
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d

from nos.data.helmholtz import HelmholtzDataset

from .report import Report
from .transmission_loss import transmission_loss

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams["pgf.texsystem"] = "pdflatex"


class GapReport(Report):
    @staticmethod
    def run(out_dir: pathlib.Path, data_sets: Dict[str, HelmholtzDataset]) -> None:
        gr_out = out_dir.joinpath("tl_gaps")
        gr_out.mkdir()

        tl_sets = {name: (dset, transmission_loss(dset, 1.0)) for name, dset in data_sets.items()}

        GapReport.individual_plots(gr_out, tl_sets)
        GapReport.relational_plots(gr_out, tl_sets)

    @staticmethod
    def relational_plots(out_dir: pathlib.Path, data_sets: Dict[str, Tuple[HelmholtzDataset, np.array]]) -> None:
        if len(data_sets) < 2:
            return
        studies = combinations(data_sets.items(), 2)
        for d1, d2 in studies:
            name1, _ = d1
            name2, _ = d2
            # filename
            study_id = "_".join([re.sub(r"[^\w\-_\. ]", "_", name) for name in [name1, name2]])
            GapReport.difference_plot(out_dir.joinpath(f"tl_{study_id}_difference.pdf"), d1, d2)
            GapReport.comparative_plot(out_dir.joinpath(f"tl_{study_id}_comparison.pdf"), d1, d2)

    @staticmethod
    def comparative_plot(out_file: pathlib.Path, study1: Tuple, study2: Tuple) -> None:
        name1, (dset1, tl1) = study1
        name2, (dset2, tl2) = study2

        f_min = min([min(dset1.frequencies), min(dset2.frequencies)])
        f_max = max([max(dset1.frequencies), max(dset2.frequencies)])
        f_delta = f_max - f_min
        space = 0.01  # five percent on top and bottom
        y_lim = (f_min - space * f_delta, f_max + space * f_delta)

        fig, ax = plt.subplots()
        sns.lineplot(x=tl1, y=dset1.frequencies, ax=ax, orient="y", label=name1)
        sns.lineplot(x=tl2, y=dset2.frequencies, ax=ax, orient="y", linestyle="--", label=name2, alpha=0.7)
        plt.title("Transmission loss comparison")
        plt.xlabel("TL [dB]")
        plt.ylabel("Frequency [Hz]")
        # plt.xlim((-100, 10))
        plt.ylim(y_lim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file)
        plt.clf()

    @staticmethod
    def difference_plot(out_file: pathlib.Path, study1: Tuple, study2: Tuple) -> None:
        name1, (dset1, tl1) = study1
        name2, (dset2, tl2) = study2

        tl1i = interp1d(dset1.frequencies, tl1)
        tl2i = interp1d(dset2.frequencies, tl2)

        f_min = min([min(dset1.frequencies), min(dset2.frequencies)])
        f_max = max([max(dset1.frequencies), max(dset2.frequencies)])
        f_delta = f_max - f_min
        space = 0.01  # five percent on top and bottom
        y_lim = (f_min - space * f_delta, f_max + space * f_delta)

        # valid range
        f_min = max([min(dset1.frequencies), min(dset2.frequencies)])
        f_max = min([max(dset1.frequencies), max(dset2.frequencies)])
        f = np.linspace(f_min, f_max, 300)
        delta = np.abs(tl2i(f) - tl1i(f))

        # plot
        fig, ax = plt.subplots()
        sns.lineplot(x=delta, y=f, ax=ax, orient="y")
        plt.title("Absolute Transmission Loss Difference")
        plt.xlabel("TL [dB]")
        plt.ylabel("Frequency [Hz]")
        plt.xscale("log")
        plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(out_file)
        plt.clf()

    @staticmethod
    def individual_plots(out_dir: pathlib.Path, data_sets: Dict[str, Tuple[HelmholtzDataset, np.array]]) -> None:
        for name, (dset, tl) in data_sets.items():
            path_name = re.sub(r"[^\w\-_\. ]", "_", name)
            file = out_dir.joinpath(f"tl_{path_name}.pdf")

            fig, ax = plt.subplots()
            sns.lineplot(x=tl, y=dset.frequencies, ax=ax, orient="y")
            plt.title(f"Transmission loss {name}")
            plt.xlabel("TL [dB]")
            plt.ylabel("Frequency [Hz]")
            # plt.xlim((-100, 10))
            plt.tight_layout()
            plt.savefig(file)
            plt.clf()
