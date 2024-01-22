import pathlib
import random
import tempfile

import pytest

from nos.data.helmholtz.domain_properties import CShapeDescription, CylinderDescription, NoneDescription, read_config
from nos.data.helmholtz.mesh import MeshBuilder


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
    return [des for des in descriptions if isinstance(des.crystal_description, CylinderDescription)]


@pytest.fixture
def c_shaped_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal_description, CShapeDescription)]


@pytest.fixture
def none_descriptions(descriptions):
    return [des for des in descriptions if isinstance(des.crystal_description, NoneDescription)]


@pytest.mark.slow
def test_mesh_builder_none(none_descriptions, cylindrical_descriptions, c_shaped_descriptions):
    domain_descriptions = [
        random.choice(none_descriptions),
        random.choice(cylindrical_descriptions),
        random.choice(c_shaped_descriptions),
    ]
    for description in domain_descriptions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir).joinpath("mesh.msh")
            cb = MeshBuilder(description, mesh_path)

            cb.build()

            assert mesh_path.is_file()
