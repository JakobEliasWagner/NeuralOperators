import pathlib
import tempfile

from nos.plots import (
    plot_multirun_metrics,
)


def can_run(exemplar_multirun_data, tl_compact_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        plot_multirun_metrics(exemplar_multirun_data, tl_compact_dataset, tmp_path)
