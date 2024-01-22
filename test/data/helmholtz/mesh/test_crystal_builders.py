import pathlib

import pytest

from nos.data.helmholtz.domain_properties import read_config
from nos.data.helmholtz.mesh import CrystalBuilder

TEMPLATES_DIR = pathlib.Path.cwd().joinpath("templates")


@pytest.fixture
def descriptions():
    files = TEMPLATES_DIR.glob("*.ini")
    descriptions = []
    for file in files:
        descriptions.extend(read_config(file))
    return descriptions


def test_base_crystal_builder_empty(descriptions):
    for description in descriptions:
        cb = CrystalBuilder(description)
        assert len(cb.build()) == 0
