import dataclasses
from typing import Tuple

import numpy as np


@dataclasses.dataclass
class BoundingBox2D:
    """2D bounding box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    min: np.array = dataclasses.field(init=False)
    max: np.array = dataclasses.field(init=False)

    def __post_init__(self):
        self.min = np.array([self.x_min, self.y_min, 0.0])  # 3d to be able to handle 3d values
        self.max = np.array([self.x_max, self.y_max, 0.0])

    def check_input_array(self, x: np.array) -> Tuple[int, np.array]:
        """Checks that the dimensions align with the underling implementations.

        Args:
            x: input array

        Returns:
            dimensions of the input array and the correctly shaped array.
        """
        try:
            n_dim = x.shape[1]
        except IndexError:
            n_dim = x.shape[0]
            x = x.reshape((-1, n_dim))
        return n_dim, x

    def inside(self, x: np.array) -> np.ndarray:
        """Method to tell which points from the input array are inside the bounding box.

        Args:
            x: Array that should be probed.

        Returns:
            indices to all values in the array which are inside the bounding box.
        """
        n_dim, x = self.check_input_array(x)

        x_inside = (x[:, 0] >= self.x_min) & (x[:, 0] <= self.x_max)
        y_inside = (x[:, 1] >= self.y_min) & (x[:, 1] <= self.y_max)

        return np.where(x_inside & y_inside)[0]

    def distance(self, x: np.array) -> np.array:
        """Calculates the distance between points in x and the bounding box.

        The distance is calculated to the closest point on the box. Points inside the box will be returned as zero as
        their distance to the box is zero. The Frobenius norm is used to calculate the distance.

        Args:
            x: (n, n_dim) array of coordinates.

        Returns:
            array of distances.
        """
        n_dim, x = self.check_input_array(x)

        clamped = np.maximum(self.min[:n_dim], np.minimum(x, self.max[:n_dim]))

        # Compute the distance from the clamped points to the original points
        dist = np.linalg.norm(clamped - x, axis=1)

        return dist
