import pathlib
import tempfile

import numpy as np
import pytest

import nos.data.helmholtz.domain_properties as d


@pytest.fixture
def crystal_descriptions():
    return [
        d.CylinderDescription(0.22, 19, 42.42),
        d.CShapeDescription(123.23, 12, 1.22, 0.9, 0.5),
        d.NoneDescription(0.22, 12),
    ]


@pytest.fixture
def absorber_description():
    return d.AdiabaticAbsorberDescription(0.22, 123.23, 123)


@pytest.fixture
def descriptions(absorber_description, crystal_descriptions):
    return [
        d.Description(
            frequencies=np.arange(1, 42),
            rho=1.25,
            c=343.0,
            lambda_left_width=0.55,
            lambda_right_width=0.55,
            elements_per_lambda=6.2,
            crystal=c_des,
            absorber=absorber_description,
        )
        for c_des in crystal_descriptions
    ]


def test_create_description_instance(descriptions):
    for des in descriptions:
        assert des.height == des.crystal.grid_size
        assert des.domain_width == des.crystal.n * des.crystal.grid_size
        assert np.allclose(des.wave_lengths, des.c / des.frequencies)
        assert np.allclose(des.ks, des.frequencies / des.c * 2.0 * np.pi)


def test_crystal_descriptions_correct(descriptions, crystal_descriptions):
    for des, c_des in zip(descriptions, crystal_descriptions):
        assert des.crystal == c_des


def test_serialize_description(descriptions):
    ds = [des.serialize() for des in descriptions]
    for o in ds:
        assert isinstance(o, dict)


def test_can_save_to_json(descriptions):
    with tempfile.TemporaryDirectory() as fp:
        fp_path = pathlib.Path(fp)

        for des in descriptions:
            des.save_to_json(fp_path)

        files = list(fp_path.glob("*.json"))

        assert len(files) == len(descriptions)
