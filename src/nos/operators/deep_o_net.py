import torch
import torch.nn as nn
from continuity.operators import (
    DeepONet as DON,
)
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


class ConDeepONet(DON, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
    ):
        super().__init__(
            shapes=shapes,
            branch_width=branch_width,
            branch_depth=branch_depth,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            basis_functions=basis_functions,
        )
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "branch_width": branch_width,
                "branch_depth": branch_depth,
                "trunk_width": trunk_width,
                "trunk_depth": trunk_depth,
                "basis_functions": basis_functions,
            },
        )


class DeepONet(Operator, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        stride: int = 1,
        dropout_p: float = 0.0,
        batch_norm: bool = True,
        basis_functions: int = 8,
        act: nn.Module = None,
    ):
        super().__init__()
        if act is None:
            act = nn.Tanh()
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
        self.branch_lift = nn.Linear(shapes.u.num * shapes.u.dim, branch_width)
        self.branch_hidden = ResNet(
            width=branch_width, depth=branch_depth, act=act, stride=stride, dropout_p=dropout_p, batch_norm=batch_norm
        )
        self.branch_project = nn.Linear(branch_width, self.dot_dim)
        self.branch = nn.Sequential(self.branch_lift, self.branch_hidden, self.branch_project)

        # trunk network
        self.trunk_lift = nn.Linear(shapes.y.dim, trunk_width)
        self.trunk_hidden = ResNet(
            width=branch_width, depth=branch_depth, act=act, stride=stride, dropout_p=dropout_p, batch_norm=batch_norm
        )
        self.trunk_project = nn.Linear(trunk_width, self.dot_dim)
        self.trunk = nn.Sequential(self.trunk_lift, self.trunk_hidden, self.trunk_project)

    def forward(self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # flatten inputs for both trunk and branch network
        u = u.flatten(1, -1)

        y = y.flatten(0, 1)

        # Pass through branch and trunk networks
        b = self.branch(u)
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
