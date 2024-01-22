import pathlib

import gmsh
import pytest

from nos.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, read_config
from nos.data.helmholtz.mesh import CrystalBuilder, CShapedCrystalBuilder, CylindricalCrystalBuilder

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


def test_base_crystal_builder_empty(descriptions):
    for description in descriptions:
        cb = CrystalBuilder(description)
        assert len(cb.build()) == 0


def test_cylindrical_builder(cylindrical_descriptions):
    description = cylindrical_descriptions[0]
    cb = CylindricalCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_cylindrical_builder")
    n_cylinders = description.crystal_description.n_x * description.crystal_description.n_y
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()


def test_c_shaped_builder(c_shaped_descriptions):
    description = c_shaped_descriptions[0]
    cb = CShapedCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_c_shaped_builder")
    n_cylinders = description.crystal_description.n_x * description.crystal_description.n_y
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()
