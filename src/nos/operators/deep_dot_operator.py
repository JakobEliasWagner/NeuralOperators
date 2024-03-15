import torch
import torch.nn as nn
from continuity.operators.shape import (
    OperatorShapes,
)

from nos.networks import (
    ResNet,
)

from .operator import (
    NosOperator,
)


class DeepDotOperator(NosOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 16,
        branch_depth: int = 2,
        trunk_width: int = 16,
        trunk_depth: int = 2,
        dot_width: int = 16,
        dot_depth: int = 2,
        act: nn.Module = nn.Tanh,
        stride: int = 1,
    ):
        super().__init__(
            properties={
                "branch_width": branch_width,
                "branch_depth": branch_depth,
                "trunk_width": trunk_width,
                "trunk_depth": trunk_depth,
                "dot_width": dot_width,
                "dot_depth": dot_depth,
                "act": act.__name__,
                "stride": stride,
            },
            shapes=shapes,
        )

        # branch network
        dot_branch_width = dot_width // 2 + dot_depth % 2
        self.branch_lift = nn.Linear(shapes.u.num * shapes.u.dim, branch_width)
        self.branch_hidden = ResNet(width=branch_width, depth=branch_depth, act=act, stride=stride)
        self.branch_project = nn.Linear(branch_width, dot_branch_width)
        self.branch = nn.Sequential(self.branch_lift, self.branch_hidden, self.branch_project)

        # root branch convolution
        self.root_branch_conv = nn.Conv1d(2, 1, 1)

        # trunk network
        dot_trunk_width = dot_width // 2
        self.trunk_lift = nn.Linear(shapes.y.dim, trunk_width)
        self.trunk_hidden = ResNet(width=trunk_width, depth=trunk_depth, act=act, stride=stride)
        self.trunk_project = nn.Linear(trunk_width, dot_trunk_width)
        self.trunk = nn.Sequential(self.trunk_lift, self.trunk_hidden, self.trunk_project)

        # deep dot
        self.deep_dot_hidden = ResNet(width=dot_width, depth=dot_depth, act=act, stride=stride)
        self.deep_dot_project = nn.Linear(dot_width, shapes.v.dim)
        self.deep_dot = nn.Sequential(self.deep_dot_hidden, self.deep_dot_project)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(u.flatten(-2, -1)).unsqueeze(1).expand(-1, y.size(1), -1)
        trunk_out = self.trunk(y)

        dot_cat = torch.cat([branch_out, trunk_out], dim=-1)
        return self.deep_dot(dot_cat)
