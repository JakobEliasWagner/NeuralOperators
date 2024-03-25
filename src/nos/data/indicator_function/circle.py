import torch

from .indicator import (
    Indicator,
)


class Circle(Indicator):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor, normal_outside: bool = True):
        self.n_dim = center.ndim
        self.center = center
        self.radius = radius
        self.normal_outside = normal_outside

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._distance(x)

    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        distance_to_center = torch.sqrt(torch.sum((x - self.center) ** 2, dim=1))
        distance_to_circle = distance_to_center - self.radius
        if not self.normal_outside:
            distance_to_circle *= -1

        return distance_to_circle
