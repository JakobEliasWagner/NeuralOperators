import torch

from .indicator import (
    Indicator,
)


class Rectangle(Indicator):
    def __init__(self, center: torch.Tensor, width: torch.Tensor, height: torch.Tensor, normal_outside: bool = True):
        """
        Initializes the Rectangle class.

        Parameters:
        - center (torch.Tensor): A tensor of shape (2,) representing the center of the rectangle.
        - width (float): The width of the rectangle.
        - height (float): The height of the rectangle.
        """
        self.center = center
        self.width = width
        self.height = height
        self.normal_outside = normal_outside

    def __call__(self, x):
        """
        Calculates the distance from each point in x to the rectangle.

        Parameters:
        - x (torch.Tensor): A tensor of shape (n, 2) representing n points in 2D space.

        Returns:
        - torch.Tensor: A tensor of shape (n,) containing the distance of each point from the rectangle.
        """
        # Calculate half dimensions
        half_width = self.width / 2
        half_height = self.height / 2

        # Calculate distances to the rectangle's sides from the center
        dx = torch.abs(x[:, 0] - self.center[0]) - half_width
        dy = torch.abs(x[:, 1] - self.center[1]) - half_height

        # Euclidean distance for points outside the rectangle, and 0 for points inside
        mask_x = torch.less_equal(dx, 0)
        mask_y = torch.less_equal(dy, 0)
        mask = mask_x & mask_y

        distance_to_rectangle = torch.sqrt(dx**2 + dy**2)
        distance_to_rectangle[mask_y] = torch.sqrt(dx[mask_y] ** 2)
        distance_to_rectangle[mask_x] = torch.sqrt(dy[mask_x] ** 2)
        distance_to_rectangle[mask] = torch.maximum(dx[mask], dy[mask])

        if not self.normal_outside:
            distance_to_rectangle *= -1

        return distance_to_rectangle
