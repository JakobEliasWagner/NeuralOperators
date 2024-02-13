import numpy as np
import pytest

from nos.data.helmholtz.domain_properties import AdiabaticAbsorberDescription, Description, NoneDescription
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
        n_left=0.5,
        n_right=0.5,
        elements_per_lambda=6.5,
        absorber=AdiabaticAbsorberDescription(1.0, 10.0, 2),
        crystal=NoneDescription(grid_size=1, n=2),
    )
    absorber = AdiabaticAbsorber(des)

    assert absorber.bbox.x_max == 2.0
    assert absorber.bbox.x_min == -1.0
    assert absorber.bbox.y_max == 0.5
    assert absorber.bbox.y_min == -0.5

    return absorber


def test_inside_zero(all_sides_absorber):
    x = np.array([[3.0, 0.0], [0.0, 1.0]]) @ np.random.random((2, 1000)) - np.array([[1.0, 0.5]]).T  # all inside
    assert np.allclose(all_sides_absorber.eval(x), 0)


def test_outside_not_zero(all_sides_absorber):
    # all outside
    x = np.stack(
        [
            np.concatenate([np.random.random((500,)) - 2.1, np.random.random((500,)) + 2.1]),
            np.concatenate([np.random.random((500,)) - 1.6, np.random.random((500,)) + 0.6]),
        ],
        axis=1,
    ).T
    assert all(all_sides_absorber.eval(x) != 0)


def test_correct_value(all_sides_absorber):
    x = np.array([[4.0, 1.0], [-1, -1]]).T

    sigma_0 = -3 * np.log(10) / 4.0
    sol = sigma_0 * np.array([np.sqrt(2.0**2 + 0.5**2), 0.5]) ** 2

    assert np.allclose(all_sides_absorber.eval(x), sol)
