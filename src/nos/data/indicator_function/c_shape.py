import torch

from .indicator import (
    Indicator,
)


class CShape(Indicator):
    def __init__(
        self,
        outer_radius: torch.tensor = torch.tensor([6.5e-3]),
        inner_radius: torch.tensor = torch.tensor([5e-3]),
        gap_width: torch.tensor = torch.tensor([4e-3]),
        grid_size: torch.tensor = torch.tensor([22e-3]),
        n_circle_samples: int = 300,
        n_gap_samples: int = 100,
    ):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.gap_width = gap_width
        self.center = (grid_size / 2) * torch.ones(
            2,
        )

        # Compute angle for gap boundaries
        theta_gap_outer = torch.asin(self.gap_width / 2 / self.outer_radius)
        theta_gap_inner = torch.asin(self.gap_width / 2 / self.inner_radius)

        # Define ranges for circle sampling excluding the gap
        theta_outer = torch.linspace(theta_gap_outer.item(), 2 * torch.pi - theta_gap_outer.item(), n_circle_samples)
        theta_inner = torch.linspace(theta_gap_inner.item(), 2 * torch.pi - theta_gap_inner.item(), n_circle_samples)

        # Sample points on the outer and inner circles
        outer_circle_x = self.outer_radius * torch.cos(theta_outer)
        outer_circle_y = self.outer_radius * torch.sin(theta_outer)
        outer_circle = torch.stack([outer_circle_x, outer_circle_y], dim=1)
        outer_circle = self.center - outer_circle

        inner_circle_x = self.inner_radius * torch.cos(theta_inner)
        inner_circle_y = self.inner_radius * torch.sin(theta_inner)
        inner_circle = torch.stack([inner_circle_x, inner_circle_y], dim=1)
        inner_circle = self.center - inner_circle

        # Sample points on the gap edges
        gap_x = torch.linspace(
            self.center[0] - self.outer_radius.item() * torch.cos(theta_gap_outer).item(),
            self.center[0] - self.inner_radius.item() * torch.cos(theta_gap_inner).item(),
            n_gap_samples // 2,
        )
        gap_y = self.gap_width / 2

        # Concatenate all samples
        self.samples = torch.cat(
            [
                outer_circle,
                inner_circle,
                torch.stack([gap_x, torch.ones(gap_x.shape) * self.center[1] + gap_y], dim=1),
                torch.stack([gap_x, torch.ones(gap_x.shape) * self.center[1] - gap_y], dim=1),
            ],
            dim=0,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(x, self.samples)
        distance, _ = torch.min(distance, dim=1)

        dist_from_center = torch.cdist(x, self.center.unsqueeze(0)).squeeze()
        is_in_outer = torch.less_equal(dist_from_center, self.outer_radius)
        is_outside_inner = torch.greater_equal(dist_from_center, self.inner_radius)

        is_right = torch.greater_equal(x[:, 0], self.center[0])
        is_gap_height = torch.greater_equal(torch.abs(x[:, 1] - self.center[1]), self.gap_width / 2)
        is_outside_gap = is_right | is_gap_height

        mask = is_in_outer & is_outside_inner & is_outside_gap
        distance[mask] *= -1

        return distance
