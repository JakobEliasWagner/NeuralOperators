import pathlib
import tempfile

from nos.plots import (
    plot_multirun_curves,
)


def can_run(exemplar_multirun_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        plot_multirun_curves(exemplar_multirun_data, tmp_path)
