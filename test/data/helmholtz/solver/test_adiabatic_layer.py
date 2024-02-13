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
        lambda_left_width=0.5,
        lambda_right_width=0.5,
        elements_per_lambda=6.5,
        absorber=AdiabaticAbsorberDescription(1.0, 10.0, 2),
        crystal=NoneDescription(grid_size=1, n=2),
    )
    absorber = AdiabaticAbsorber(des)

    assert absorber.bbox.x_max == 2.5
    assert absorber.bbox.x_min == -0.5
    assert absorber.bbox.y_max == 1.0
    assert absorber.bbox.y_min == 0.0

    return absorber


def test_inside_zero(all_sides_absorber):
    x = np.array([[3.0, 0.0], [0.0, 1.0]]) @ np.random.random((2, 1000)) - np.array([[0.5, 0]]).T  # all inside
    assert np.allclose(all_sides_absorber.eval(x), 0)


def test_outside_not_zero(all_sides_absorber):
    # all outside
    x = np.stack(
        [
            np.concatenate([np.random.random((500,)) - 1.1, np.random.random((500,)) + 3.1]),
            np.concatenate([np.random.random((500,)) - 1.1, np.random.random((500,)) + 2.1]),
        ],
        axis=1,
    ).T
    assert all(all_sides_absorber.eval(x) != 0)


def test_correct_value(all_sides_absorber):
    x = np.array([[4.0, 1.0], [-1, -1]]).T

    sigma_0 = -3 * np.log(10) / 2.0
    sol = sigma_0 * 1j * np.array([1.5, np.sqrt(1.25)]) ** 2

    assert np.allclose(all_sides_absorber.eval(x), sol)
