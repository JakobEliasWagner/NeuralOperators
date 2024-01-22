import pathlib

import pytest

from nos.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, read_config

TEMPLATES_DIR = pathlib.Path.cwd().joinpath("templates")


@pytest.fixture
def descriptions():
    files = TEMPLATES_DIR.glob("*.ini")
    descriptions = []
    for file in files:
        descriptions.extend(read_config(file))
    return descriptions


@pytest.fixture
def cylindrical_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal_description, CylinderDescription)]


@pytest.fixture
def c_shaped_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal_description, CShapeDescription)]


@pytest.fixture
def none_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal_description, CShapeDescription)]
