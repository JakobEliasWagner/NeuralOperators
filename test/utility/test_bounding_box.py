import numpy as np

from nos.utility import BoundingBox2D


def test_inside_with_points_inside():
    bbox = BoundingBox2D(x_min=0, y_min=0, x_max=10, y_max=10)
    points = np.array([[1, 1], [5, 5], [9, 9]])
    inside_indices = bbox.inside(points)
    assert np.array_equal(inside_indices, np.array([0, 1, 2]))


def test_inside_with_points_outside():
    bbox = BoundingBox2D(x_min=0, y_min=0, x_max=10, y_max=10)
    points = np.array([[-1, -1], [11, 11], [20, 20]])
    inside_indices = bbox.inside(points)
    assert len(inside_indices) == 0


def test_inside_with_points_on_edges():
    bbox = BoundingBox2D(x_min=0, y_min=0, x_max=10, y_max=10)
    points = np.array([[0, 0], [10, 10], [0, 10], [10, 0]])
    inside_indices = bbox.inside(points)
    assert np.array_equal(inside_indices, np.array([0, 1, 2, 3]))
