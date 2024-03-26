from typing import (
    Tuple,
)

import torch
from continuity.discrete.box_sampler import (
    BoxSampler,
)

from .c_shape import (
    CShape,
)


def params_to_indicator(
    u: torch.Tensor, sampler: BoxSampler, n_samples: int = 2**10
) -> (Tuple)[torch.Tensor, torch.Tensor]:
    n_observations = u.size(0)

    x = torch.stack([sampler(n_samples) for _ in range(n_observations)], dim=0)
    indicator = torch.empty((n_observations, n_samples, 1))

    for i, (xi, ui) in enumerate(zip(x, u)):
        shape = CShape(outer_radius=ui[:, 0], inner_radius=ui[:, 1], gap_width=ui[:, 2])
        indicator[i] = shape(xi)

    return x, indicator
