from dataclasses import (
    replace,
)

import torch
import torch.nn as nn
from continuity.operators import (
    FourierNeuralOperator,
    Operator,
    OperatorShapes,
)

from nos.networks import (
    ResNet,
)

from .operator import (
    NeuralOperator,
)


class DeepOBranchFNO(Operator, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 4,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
        stride: int = 1,
        act: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "branch_width": branch_width,
                "branch_depth": branch_depth,
                "trunk_width": trunk_width,
                "trunk_depth": trunk_depth,
                "stride": stride,
                "basis_functions": basis_functions,
                "act": act.__class__.__name__,
            },
        )

        self.basis_functions = basis_functions
        self.dot_dim = shapes.v.dim * basis_functions

        # branch network
        branch_shapes = replace(shapes)
        branch_shapes.y = branch_shapes.x
        branch_shapes.v = branch_shapes.u
        self.branch_hidden = FourierNeuralOperator(shapes=branch_shapes, width=branch_width, depth=branch_depth)
        self.branch_flatten = nn.Flatten()
        self.branch_project = nn.Linear(branch_shapes.v.num * branch_shapes.v.dim, self.dot_dim)
        self.branch = nn.Sequential(self.branch_hidden, self.branch_flatten, self.branch_project)

        # trunk network
        self.trunk_lift = nn.Linear(shapes.y.dim, trunk_width)
        self.trunk_hidden = ResNet(width=trunk_width, depth=trunk_depth, act=act, stride=stride)
        self.trunk_project = nn.Linear(trunk_width, self.dot_dim)
        self.trunk = nn.Sequential(self.trunk_lift, self.trunk_hidden, self.trunk_project)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = self.branch_hidden(x, u, x)
        b = self.branch_flatten(b)
        b = self.branch_project(b)

        y = y.flatten(0, 1)
        t = self.trunk(y)

        # dot product
        b = b.reshape(-1, self.shapes.v.dim, self.basis_functions)
        t = t.reshape(
            b.size(0),
            -1,
            self.shapes.v.dim,
            self.basis_functions,
        )

        return torch.einsum("abcd,acd->abc", t, b)
