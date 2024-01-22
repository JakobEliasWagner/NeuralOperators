import dataclasses

import numpy as np


@dataclasses.dataclass
class BoundingBox2D:
    """2D bounding box."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def inside(self, x: np.array) -> np.ndarray:
        """Method to tell which points from the input array are inside the bounding box.

        Args:
            x: Array that should be probed.

        Returns:
            indices to all values in the array which are inside the bounding box.
        """
        x_inside = (x[:, 0] >= self.x_min) & (x[:, 0] <= self.x_max)
        y_inside = (x[:, 1] >= self.y_min) & (x[:, 1] <= self.y_max)

        return np.where(x_inside & y_inside)[0]
