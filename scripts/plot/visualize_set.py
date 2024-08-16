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
    TLDatasetCompactWave,
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


def visualize_single_operator():
    pass


if __name__ == "__main__":
    run_id = ["2024-04-06", "09-09-59"]
    multirun_path = pathlib.Path.cwd().joinpath("multirun", *run_id)

    is_fno = False
    if is_fno:
        dataset_class = TLDatasetCompactWave
    else:
        dataset_class = TLDatasetCompact

    test_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_smooth")
    test_set = dataset_class(test_path)

    non_smooth_path = pathlib.Path.cwd().joinpath("data", "test", "transmission_loss_lin")
    non_smooth_set = dataset_class(non_smooth_path)

    train_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth")
    train_set = dataset_class(train_path)
    test_set.transform = train_set.transform
    non_smooth_set.transform = train_set.transform

    visualize_multirun(
        multirun_path=multirun_path,
        datasets=[("smooth", test_set), ("not_smooth", non_smooth_set)],
        out_dir=pathlib.Path.cwd().joinpath("out", *run_id),
    )
