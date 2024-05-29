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
    RunData,
    plot_run_curves,
    plot_run_transmission_loss,
)


def visualize_run(run_path: pathlib.Path, datasets: List[Tuple[str, OperatorDataset]], out_dir: pathlib.Path):
    run = RunData.from_dir(run_path)

    plot_run_curves(run, out_dir)

    pbar = tqdm(datasets, position=0)
    for name, dataset in datasets:
        pbar.set_postfix_str(f"... processing {name} ...")
        dataset_out = out_dir.joinpath(name)

        plot_run_transmission_loss(run, dataset, dataset_out)


if __name__ == "__main__":
    run_id = ["deep_dot_operator", "medium", "models", "DeepDotOperator_2024_04_07_15_11_37"]
    run_path = pathlib.Path.cwd().joinpath("finished_models", *run_id)

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

    visualize_run(
        run_path=run_path,
        datasets=[("smooth", test_set), ("not_smooth", non_smooth_set)],
        out_dir=pathlib.Path.cwd().joinpath("out", *run_id),
    )
