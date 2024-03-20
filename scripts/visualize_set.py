import pathlib
from typing import (
    List,
    Tuple,
)

from continuity.data import (
    OperatorDataset,
)
from tqdm import (
    tqdm,
)

from nos.data import (
    TLDatasetCompact,
)
from nos.plots import (
    MultiRunData,
    plot_multirun_curves,
    plot_multirun_metrics,
    plot_multirun_transmission_loss,
)


def visualize_multirun(
    multirun_path: pathlib.Path, datasets: List[Tuple[str, OperatorDataset]], out_dir: pathlib.Path
):
    multirun = MultiRunData.from_dir(multirun_path)

    plot_multirun_curves(multirun, out_dir)

    pbar = tqdm(datasets, position=0)
    for name, dataset in datasets:
        pbar.set_postfix_str(f"... processing {name} ...")
        dataset_out = out_dir.joinpath(name)

        plot_multirun_transmission_loss(multirun, dataset, dataset_out)
        plot_multirun_metrics(multirun, dataset, dataset_out)


if __name__ == "__main__":
    multirun_path = pathlib.Path.cwd().joinpath("multirun", "2024-03-20", "10-36-27")

    test_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_lin")
    test_set = TLDatasetCompact(test_path)

    train_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_lin")
    train_set = TLDatasetCompact(train_path)

    visualize_multirun(
        multirun_path=multirun_path,
        datasets=[("test", test_set), ("train", train_set)],
        out_dir=pathlib.Path.cwd().joinpath("out"),
    )
