import pathlib

import gmsh
import pytest

from nos.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, NoneDescription, read_config
from nos.data.helmholtz.mesh import CrystalDomainBuilder, CShapedCrystalDomainBuilder, CylindricalCrystalDomainBuilder


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


def test_cylindrical_crystal_domain_builder(cylindrical_descriptions):
    description = cylindrical_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test cylindrical domain")
    b = CylindricalCrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()


def test_c_shaped_crystal_domain_builder(c_shaped_descriptions):
    description = c_shaped_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test c-shaped domain")
    b = CShapedCrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()


def test_none_crystal_domain_builder(none_descriptions):
    description = none_descriptions[0]
    gmsh.initialize()
    gmsh.model.add("test none domain")
    b = CrystalDomainBuilder(description)
    assert len(b.build()) == 1
    gmsh.finalize()
