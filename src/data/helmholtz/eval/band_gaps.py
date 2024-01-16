import pathlib

import matplotlib.pyplot as plt
import numpy as np

from src.data.utility import xdmf_to_numpy


def plot_bandgaps(data_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(15, 15))
    for file in data_dir.glob("*.xdmf"):
        data = xdmf_to_numpy(file)
        x = data["Geometry"]
        values = data["Values"]
        frequencies = data["Frequencies"]

        # get band gaps
        p0 = 0.05  # 343 ** 2 * 1.25
        in_eval_area = (x[:, 0] > 0.21999999999999997) * (x[:, 0] < 0.21999999999999997 + 0.04716250000000001)

        db_levels = []
        for val in values:
            db_level = 20 * np.log10(np.max(np.abs(val.squeeze()[in_eval_area])) / p0)
            db_levels.append(db_level)

        ax.plot(db_levels, frequencies, "-")

    plt.show()


def get_out_sound_pressure_level(data_file):
    # gets json
    # reads correct area
    # returns sound pressure levels and frequencies
    pass


if __name__ == "__main__":
    data_dir = pathlib.Path.cwd().joinpath("out", "20240115190654-c48e3827-98fa-4803-9a89-73feac8ac3ae")

    plot_bandgaps(data_dir)
