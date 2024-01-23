import dataclasses
import pathlib
import tempfile
import warnings

import numpy as np
import pytest

from nos.data.helmholtz.domain_properties import (
    AdiabaticAbsorberDescription,
    CrystalDescription,
    CShapeDescription,
    CylinderDescription,
    Description,
    NoneDescription,
)
from nos.data.helmholtz.mesh import MeshBuilder


@pytest.fixture
def cylindrical_description():
    des = Description(
        frequencies=np.arange(4000, 4001),
        rho=1.25,
        c=343,
        lambda_left_width=0.55,
        lambda_right_width=0.75,
        elements_per_lambda=6.1,
        absorber=AdiabaticAbsorberDescription(3.5),
        crystal=CylinderDescription(22e-3, 2, 6e-3),
    )
    return des


@pytest.fixture
def c_shaped_description(cylindrical_description):
    des = dataclasses.replace(cylindrical_description)
    des.crystal_description = CShapeDescription(22e-3, 2, 6.5e-3, 0.9, 0.2)
    return des


@pytest.fixture
def none_description(cylindrical_description):
    des = dataclasses.replace(cylindrical_description)
    des.crystal_description = NoneDescription(22e-3, 2)
    return des


@pytest.fixture
def wrong_description(cylindrical_description):
    class WrongCrystal(CrystalDescription):
        """wrong crystal class."""

        pass

    des = dataclasses.replace(cylindrical_description)
    des.crystal_description = WrongCrystal(22e-3, 2)
    return des


def test_mesh_builder_none(none_description, cylindrical_description, c_shaped_description):
    domain_descriptions = [none_description, cylindrical_description, c_shaped_description]
    for description in domain_descriptions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir).joinpath("mesh.msh")
            cb = MeshBuilder(description, mesh_path)

            cb.build()

            assert mesh_path.is_file()


def test_unknown_crystal_builder(wrong_description):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir).joinpath("mesh.msh")
            cb = MeshBuilder(wrong_description, mesh_path)

            cb.build()
    assert True
