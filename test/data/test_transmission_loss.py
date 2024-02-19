import pathlib

import pytest

from nos.data import TLDataset, TLDatasetCompact


@pytest.fixture(scope="module")
def tl_csv_file():
    data_dir = pathlib.Path.cwd().joinpath("data", "transmission_loss")
    return next(data_dir.glob("*.csv"))


@pytest.fixture(scope="module")
def tl_dset(tl_csv_file):
    return TLDataset(tl_csv_file)


@pytest.fixture(scope="module")
def tl_dset_compact(tl_csv_file):
    return TLDatasetCompact(tl_csv_file)


def test_tl_dataset_shape(tl_dset, tl_dset_compact):
    for dset in [tl_dset, tl_dset_compact]:
        assert dset.x.size(0) == dset.y.size(0) == dset.u.size(0) == dset.v.size(0)


def test_tl_dataset_compact(tl_dset, tl_dset_compact):
    unique_crystals = tl_dset.x.unique(dim=0)
    assert tl_dset_compact.y.size(1) == tl_dset.x.size(0) // unique_crystals.size(0)
