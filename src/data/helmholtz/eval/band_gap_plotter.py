import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from band_gaps import get_band_gaps


class PlotDescriptionStrategy(ABC):
    def __init__(self, descriptions: List[dict]):
        self.descriptions = descriptions
        self.description_ids = [desc["unique_id"] for desc in descriptions]

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_dset_str(self, idx_or_id: int | str) -> str:
        pass

    def get_dset_description(self, attr: int | str) -> dict:
        if isinstance(attr, int):
            des_strategy = self.get_description_by_idx
        else:
            des_strategy = self.get_description_by_id
        return des_strategy(attr)

    def get_dset_idx(self, dset_id: str) -> int:
        return self.description_ids.index(dset_id)

    def get_description_by_id(self, dset_id: str) -> dict:
        index = self.get_dset_idx(dset_id)
        return self.get_description_by_id(index)

    def get_description_by_idx(self, dset_idx: int) -> dict:
        return self.descriptions[dset_idx]


class CrystalDescriptionStrategy(PlotDescriptionStrategy):
    def __init__(self, descriptions: List[dict]):
        super().__init__(descriptions)
        self.crystal_descriptions = [des["crystal_description"] for des in descriptions]
        self.keys = set().union(*(d.keys() for d in self.crystal_descriptions))
        self.sampled_keys = self.get_sampled_crystal_keys()
        self.unsampled_keys = self.keys - self.sampled_keys

    def get_sampled_crystal_keys(self) -> Set[str]:
        """

        :return:
        """
        key_counter = defaultdict(set)
        for desc in self.crystal_descriptions:
            for key, value in desc.items():
                key_counter[key].add(value)

        return {key for key, value in key_counter.items() if len(value) > 1}

    def get_dset_str(self, idx_or_id: int | str) -> str:
        dset = self.get_dset_description(idx_or_id)
        crystal_dset = dset["crystal_description"]
        keys = set(crystal_dset.keys())
        relevant_keys = keys.intersection(self.sampled_keys)
        return ", ".join([f"{key}:{crystal_dset[key]}" for key in relevant_keys])

    def __str__(self):
        """Overall description of the domain with constant crystal properties over all descriptions"""
        return ", ".join([f"{key}:{self.crystal_descriptions[0][key]}" for key in self.unsampled_keys])


class BandGapPlotter:
    def __init__(
        self,
        data_dir: pathlib.Path,
        out_dir: pathlib.Path,
        fig_size: Tuple[int, int] = None,
        describer: Type[PlotDescriptionStrategy] = CrystalDescriptionStrategy,
        theme: str = "whitegrid",
    ):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.fig_size = fig_size

        self.descriptions, self.frequencies, self.band_gaps = get_band_gaps(self.data_dir)
        self.plot_descriptions = describer(self.descriptions)

        self.size = len(self.descriptions)

        sns.set_theme(style=theme)

    def plot_all(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # plot

        self.plot_grid()
        self.plot_one()
        self.plot_surf()
        self.plot_individuals()

    def plot_grid(self):
        rows = int(np.round(np.sqrt(self.size)))
        cols = int(np.ceil(self.size / rows))

        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=self.fig_size)

        for i, (freq, gaps) in enumerate(zip(self.frequencies, self.band_gaps)):
            row = i // cols
            col = i % cols
            sns.lineplot(ax=axs[row, col], x=gaps, y=freq, orient="y")
            axs[row, col].title.set_text(self.plot_descriptions.get_dset_str(i))
        fig.savefig(self.out_dir.joinpath("grid.png"))

    def plot_surf(self):
        pass

    def plot_individuals(self):
        for i, (freq, gaps) in enumerate(zip(self.frequencies, self.band_gaps)):
            fig, ax = plt.subplots(figsize=self.fig_size)
            fig.suptitle(self.plot_descriptions.get_dset_str(i))
            sns.lineplot(ax=ax, x=gaps, y=freq, orient="y")
            ax.set_xlabel("Reduced Sound Pressure Level [dB]")
            ax.set_ylabel("Frequency [Hz]")
            fig.savefig(self.out_dir.joinpath(f"individual_{i}.png"))

    def plot_one(self):
        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.suptitle(self.plot_descriptions)

        for i, (freq, gaps) in enumerate(zip(self.frequencies, self.band_gaps)):
            sns.lineplot(ax=ax, x=gaps, y=freq, orient="y", label=self.plot_descriptions.get_dset_str(i))
        ax.set_xlabel("Reduced Sound Pressure Level [dB]")
        ax.set_ylabel("Frequency [Hz]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        fig.tight_layout()
        fig.savefig(self.out_dir.joinpath("one.png"))


if __name__ == "__main__":
    data_dir = pathlib.Path.cwd().joinpath("out", "20240116150544-3c7ab99e-6939-4dbb-ac05-3bace2b8813c")
    descriptions, frequencies, band_gaps = get_band_gaps(data_dir)
    plot_descriptions = CrystalDescriptionStrategy(descriptions)

    plotter = BandGapPlotter(data_dir, data_dir.joinpath("plots"), fig_size=(10, 10))

    plotter.plot_all()
