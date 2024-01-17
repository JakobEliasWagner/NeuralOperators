import numpy as np
import pytest

import src.data.helmholtz.domain_properties.description as d


@pytest.mark.incremental
class TestDescription:
    def __init__(self):
        self.crystal_descriptions = []
        self.descriptions = []

    def test_create_crystal_d_instance(self):
        self.crystal_descriptions.append(d.CrystalDescription("crystal", 42.42, 2, 42))
        assert True

    def test_create_cylinder_d_instance(self):
        self.crystal_descriptions.append(d.CylinderDescription("Cylinder", 0.22, 19, 2, 42.42))
        assert True

    def test_create_c_shape_d_instance(self):
        self.crystal_descriptions.append(d.CShapeDescription("C-shaped", 123.23, 12, 23, 1.22, 0.9, 0.5))
        assert True

    def test_create_none_d_instance(self):
        self.crystal_descriptions.append(d.NoneDescription("None", 0.22, 12, 2))
        assert True

    def test_create_description_instance(self):
        self.descriptions = [
            d.Description(
                np.arange(10),
                1.25,
                343.0,
                0.2,
                12.4,
                900,
                901,
                899,
                1.2,
                1e-6,
                {"positive_x": True, "foo": False},
                3,
                877,
                c_des,
            )
            for c_des in self.crystal_descriptions
        ]
        assert True

    def test_crystal_descriptions_correct(self):
        for des, c_des in zip(self.descriptions, self.crystal_descriptions):
            assert des.crystal_description == c_des

    def test_serialize_description(self):
        ds = [des.serialize() for des in self.descriptions]
        for o in ds:
            assert isinstance(dict, o)

    def test_can_save_to_json(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as fp:
            fp_path = pathlib.Path(fp.name)

            for des in self.descriptions:
                des.save_to_json(fp_path)

            files = list(fp_path.glob("*.json"))

            assert len(files) == len(self.descriptions)
