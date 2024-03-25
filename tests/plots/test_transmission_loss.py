import pathlib
import tempfile

from nos.plots import (
    plot_multirun_transmission_loss,
)


def can_run(exemplar_multirun_data, tl_compact_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        plot_multirun_transmission_loss(exemplar_multirun_data, tl_compact_dataset, tmp_path)
