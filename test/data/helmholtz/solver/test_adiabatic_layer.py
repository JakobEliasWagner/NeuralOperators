import numpy as np
import pytest

from nos.data.helmholtz.domain_properties import Description, NoneDescription
from nos.data.helmholtz.solver import AdiabaticAbsorber


@pytest.fixture
def all_sides_absorber():
    """Returns an absorber with absorber-free dimensions lower left = (0, 0), upper right = (3, 2).

    Returns:
        adiabatic absorber
    """
    des = Description(
        frequencies=np.array([1]),
        rho=1.2,
        c=1.0,
        left_space=0.5,
        right_space=0.5,
        elements=42,
        depth=1,
        round_trip=10,
        directions={
            "top": True,
            "left": True,
            "bottom": True,
            "right": True,
        },
        crystal_description=NoneDescription("None", grid_size=1, n_x=2, n_y=2),
    )
    absorber = AdiabaticAbsorber(des, des.round_trip)

    assert absorber.bbox.x_max == 3.0
    assert absorber.bbox.y_max == 2.0

    return absorber


def test_inside_zero(all_sides_absorber):
    x = np.random.random((1000, 2)) @ np.array([[3.0, 0.0], [0.0, 2.0]])  # scale to match all inside
    assert np.allclose(all_sides_absorber.eval(x), 0)


def test_outside_not_zero(all_sides_absorber):
    # all outside
    x = np.stack(
        [
            np.concatenate([np.random.random((500,)) - 1.1, np.random.random((500,)) + 3.1]),
            np.concatenate([np.random.random((500,)) - 1.1, np.random.random((500,)) + 2.1]),
        ],
        axis=1,
    )
    assert all(all_sides_absorber.eval(x) != 0)


def test_correct_value(all_sides_absorber):
    x = np.array([[4.0, 1.0], [-1, -1]])

    sigma_0 = -3 * np.log(10) / 2.0
    sol = sigma_0 * 1j * np.array([1.0, np.sqrt(2)]) ** 2

    assert np.allclose(all_sides_absorber.eval(x), sol)
