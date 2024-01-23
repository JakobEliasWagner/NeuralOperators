import pathlib

import gmsh
import pytest

from nos.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, NoneDescription, read_config
from nos.data.helmholtz.mesh import CrystalBuilder, CShapedCrystalBuilder, CylindricalCrystalBuilder


@pytest.fixture
def descriptions():
    templates = pathlib.Path.cwd().joinpath("templates")
    files = templates.glob("*.ini")
    descriptions = []
    for file in files:
        descriptions.extend(read_config(file))
    return descriptions


@pytest.fixture
def cylindrical_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal, CylinderDescription)]


@pytest.fixture
def c_shaped_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal, CShapeDescription)]


@pytest.fixture
def none_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal, NoneDescription)]


def test_base_crystal_builder_empty(descriptions):
    for description in descriptions:
        cb = CrystalBuilder(description)
        assert len(cb.build()) == 0


def test_cylindrical_builder(cylindrical_descriptions):
    description = cylindrical_descriptions[0]
    cb = CylindricalCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_cylindrical_builder")
    n_cylinders = description.crystal.n
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()


def test_c_shaped_builder(c_shaped_descriptions):
    description = c_shaped_descriptions[0]
    cb = CShapedCrystalBuilder(description)
    gmsh.initialize()
    gmsh.model.add("test_c_shaped_builder")
    n_cylinders = description.crystal.n
    assert len(cb.build()) == n_cylinders
    gmsh.finalize()
