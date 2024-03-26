from typing import (
    Optional,
)

import torch
import torch.nn as nn
from continuity.operators import (
    Operator,
    OperatorShapes,
)

from nos.networks import (
    ResNet,
)

from .operator import (
    NeuralOperator,
)


class DeepNeuralOperator(Operator, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        width: int = 32,
        depth: int = 3,
        stride: int = 1,
        act: Optional[torch.nn.Module] = None,
    ):
        if act is None:
            act = torch.nn.Tanh()
        super().__init__()
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={"width": width, "depth": depth, "stride": stride, "act": act.__class__.__name__},
        )

        self.lift = nn.Linear(shapes.x.dim * shapes.x.num + shapes.u.dim * shapes.u.num + shapes.y.dim, width)
        self.hidden = ResNet(width=width, depth=depth, stride=stride, act=act)
        self.project = nn.Linear(width, shapes.v.dim)

        self.net = nn.Sequential(self.lift, self.hidden, self.project)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_repeated = u.flatten(1, 2).unsqueeze(1).expand(-1, y.size(1), -1)
        x_repeated = x.flatten(1, 2).unsqueeze(1).expand(-1, y.size(1), -1)

        net_input = torch.cat([x_repeated, u_repeated, y], dim=-1)

        return self.net(net_input)
