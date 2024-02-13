import pathlib

import numpy as np
import pytest

import nos.data.helmholtz.domain_properties as d

templates_path = pathlib.Path.cwd().joinpath("templates")


def test_file_input_domain_general():
    for file in ["domain_c_shaped.ini", "domain_cylindrical.ini", "domain_none.ini"]:
        template = templates_path.joinpath(file)
        des = d.read_config(template)[0]

        assert np.allclose(des.frequencies, np.linspace(4000, 20000, 300))
        assert des.rho == 1.25
        assert des.c == 343.0
        c = des.crystal
        assert c.grid_size == 22e-3
        assert c.n == 10


def test_file_input_domain_c_shaped():
    template = templates_path.joinpath("domain_c_shaped.ini")
    descriptions = d.read_config(template)
    crystals = [des.crystal for des in descriptions]

    outer_rs = [c.radius for c in crystals]
    inner_rs = [c.inner_radius for c in crystals]
    gap_ws = [c.gap_width for c in crystals]

    assert len(outer_rs) == len(inner_rs)
    assert len(outer_rs) == len(gap_ws)

    assert len(set(outer_rs)) == 5
    assert len(set(inner_rs)) == 6
    assert len(set(gap_ws)) == 7

    assert set(outer_rs) == set(np.linspace(6.5e-3, 0.75, 5))
    assert set(gap_ws) == set(np.linspace(0.61538461538, 1.0, 7))


def test_file_input_domain_cylindrical():
    template = templates_path.joinpath("domain_cylindrical.ini")
    descriptions = d.read_config(template)
    crystals = [des.crystal for des in descriptions]

    rs = np.array([c.radius for c in crystals])

    assert len(set(rs)) == 5

    rs.sort()
    assert np.allclose(rs, np.linspace(6.5e-3, 0.75, 5))


def test_file_input_domain_none():
    template = templates_path.joinpath("domain_none.ini")
    descriptions = d.read_config(template)
    crystals = [des.crystal for des in descriptions]

    assert len(crystals) == 1


def test_unknown_crystal_type():
    template = pathlib.Path(__file__).parent.joinpath("domain_wrong_crystal_type.ini")
    with pytest.raises(TypeError):
        d.read_config(template)
