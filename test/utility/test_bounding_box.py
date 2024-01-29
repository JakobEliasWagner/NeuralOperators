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


def test_distance_greater_zero():
    """Norms are greater or equal to zero."""
    test_size = 10
    sample_size = 1000
    for _ in range(test_size):
        dims = 2 * np.random.random((4,)) - 1  # [-1, 1)
        dims.sort()
        bbox = BoundingBox2D(*list(dims))

        x = 2 * np.random.random((sample_size, 2)) - 1

        dist = bbox.distance(x)
        assert all(dist >= 0)


def test_distance_inside():
    """All points inside -> distance needs to be always zero"""
    test_size = 10
    sample_size = 1000
    for _ in range(test_size):
        x = 2 * np.random.random((sample_size, 2))
        max_vals = np.max(x, axis=0)
        min_vals = np.min(x, axis=0)
        bbox = BoundingBox2D(*list(min_vals), *list(max_vals))

        dist = bbox.distance(x)
        assert np.allclose(dist, 0.0)


def test_distance_outside():
    """All points outside the bounding box -> no distance can be zero because of half-open interval."""
    test_size = 10
    sample_size = 1000
    for _ in range(test_size):
        dims = np.random.random((4,))  # [0, 1)
        dims.sort()
        bbox = BoundingBox2D(*list(dims))

        x = np.random.random((sample_size, 2)) - 2.0

        dist = bbox.distance(x)
        assert all(dist > 0)


def test_distance_array():
    bbox = BoundingBox2D(-1, -2, 2, 2)

    x = np.array(
        [
            [1, 1],
            [-0.5, 1],
            [-0.5, -1],
            [1, -1],
            [3, 1],
            [3, 3],
            [1, 4],
            [-3, 4],
            [-2, 1],
            [-2, -3],
            [0, -4],
            [3, 5],
        ]
    )
    correct = np.array([0, 0, 0, 0, 1, np.sqrt(2), 2, np.sqrt(8), 1, np.sqrt(2), 2, np.sqrt(10)])
    print(bbox.distance(x))
    assert np.allclose(bbox.distance(x), correct)


def test_distance_single():
    bbox = BoundingBox2D(-1, -2, 2, 2)
    x = np.array([1, 2, 3])
    bbox.distance(x)
    assert True
