import pathlib
import tempfile

import numpy as np

import nos.data.helmholtz.domain_properties as d


class TestDescription:
    crystal_descriptions = [
        d.CylinderDescription("Cylinder", 0.22, 19, 2, 42.42),
        d.CShapeDescription("C-shaped", 123.23, 12, 23, 1.22, 0.9, 0.5),
        d.NoneDescription("None", 0.22, 12, 2),
    ]
    descriptions = []

    def test_create_description_instance(self):
        self.descriptions = [
            d.Description(
                np.arange(1, 42),
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
        for des in self.descriptions:
            assert des.height == des.crystal_description.n_y * des.crystal_description.grid_size
            assert des.width == des.crystal_description.n_x * des.crystal_description.grid_size
            assert des.absorber_depth == np.max(des.wave_lengths) * des.depth
            assert np.allclose(des.wave_lengths, des.c / des.frequencies)
            assert np.allclose(des.ks, des.frequencies / des.c * 2.0 * np.pi)

    def test_crystal_descriptions_correct(self):
        for des, c_des in zip(self.descriptions, self.crystal_descriptions):
            assert des.crystal_description == c_des

    def test_serialize_description(self):
        ds = [des.serialize() for des in self.descriptions]
        for o in ds:
            assert isinstance(dict, o)

    def test_can_save_to_json(self):
        with tempfile.TemporaryDirectory() as fp:
            fp_path = pathlib.Path(fp)

            for des in self.descriptions:
                des.save_to_json(fp_path)

            files = list(fp_path.glob("*.json"))

            assert len(files) == len(self.descriptions)
