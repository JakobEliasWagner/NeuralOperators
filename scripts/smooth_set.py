import pathlib

import pandas as pd

from nos.data import (
    TLDatasetCompact,
)
from nos.preprocessing import (
    LowPassFilter1D,
)


def smooth_set(in_set: pathlib.Path, out_file: pathlib.Path):
    dataset = TLDatasetCompact(path=in_set)
    lpf = LowPassFilter1D(15, 1e-1, 10)

    x, y, v = dataset.x, dataset.y, dataset.v

    data = {"radius": [], "inner_radius": [], "gap_width": [], "frequency": [], "transmission_loss": []}

    for xi, yi, vi in zip(x, y, v):
        v_transformed = lpf(vi.squeeze()).reshape(vi.shape)
        # Assuming xi contains repeated values for all elements in a batch
        data["radius"].extend([float(xi[0][0])] * len(yi))
        data["inner_radius"].extend([float(xi[0][1])] * len(yi))
        data["gap_width"].extend([float(xi[0][2])] * len(yi))
        data["frequency"].extend(yi.squeeze().tolist())
        data["transmission_loss"].extend(v_transformed.squeeze().tolist())

    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    in_set = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_lin")
    out_set = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_smooth", "transmission_loss.csv")
    out_set.parent.mkdir(parents=True, exist_ok=True)

    smooth_set(in_set, out_set)
