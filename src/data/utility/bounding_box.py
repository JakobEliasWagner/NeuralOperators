import numpy as np


class BoundingBox2D:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def inside(self, x: np.array) -> np.ndarray:
        """Returns array of points indices located inside the box

        :param x:
        :return:
        """
        x_inside = (x[:, 0] >= self.x_min) & (x[:, 0] <= self.x_max)
        y_inside = (x[:, 1] >= self.y_min) & (x[:, 1] <= self.y_max)

        return np.where(x_inside & y_inside)[0]
