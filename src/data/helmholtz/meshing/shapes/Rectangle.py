import gmsh
import numpy as np

from src.utility import run_once

from .Shape import Shape


class Rectangle(Shape):
    """2D Rectangle

    Attributes:

    """

    def __init__(self, x: float, y: float, z: float, dx: float, dy: float):
        """

        Args:
            x: lower left corner coordinate
            y: lower left corner coordinate
            z: lower left corner coordinate
            dx: x-size
            dy: y-size
        """
        super().__init__(x, y, z, dx, dy)

        # boundary tags ar not always assigned in the same order
        self.boundaries = {
            "top": None,
            "left": None,
            "bottom": None,
            "right": None,
        }
        a_tol = min(dx, dy) / 1e5
        for boundary in self.elements[1]:
            coordinates = gmsh.model.getBoundingBox(1, boundary)
            delta_x = coordinates[3] - coordinates[0]

            if np.isclose(delta_x, dx, atol=a_tol):
                # top or bottom
                if np.isclose(coordinates[1], y, atol=a_tol):
                    self.boundaries["bottom"] = boundary
                    continue
                self.boundaries["top"] = boundary
                continue
            # left or right
            if np.isclose(coordinates[0], x, atol=a_tol):
                self.boundaries["left"] = boundary
                continue
            self.boundaries["right"] = boundary

    @run_once
    def generate(self, x: float, y: float, z: float, dx: float, dy: float):
        """

        Args:
            x: lower left corner coordinate
            y: lower left corner coordinate
            z: lower left corner coordinate
            dx: x-size
            dy: y-size
        """
        self.factory.addRectangle(x, y, z, dx, dy)
