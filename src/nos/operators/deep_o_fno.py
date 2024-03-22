from dataclasses import (
    replace,
)

import torch
import torch.nn as nn
from continuity.operators import (
    DeepONet,
    FourierNeuralOperator,
    Operator,
    OperatorShapes,
)

from .operator import (
    NeuralOperator,
)


class DeepOFNO(Operator, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
        depth: int = 3,
        width: int = 3,
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
                "basis_functions": basis_functions,
                "depth": depth,
                "width": width,
                "act": act.__class__.__name__,
            },
        )
        self.deep_o_net = DeepONet(
            shapes=shapes,
            branch_width=branch_width,
            branch_depth=branch_depth,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            basis_functions=basis_functions,
        )
        fno_shapes = replace(shapes)
        fno_shapes.x = fno_shapes.y
        fno_shapes.u = shapes.v
        self.fno = FourierNeuralOperator(shapes=fno_shapes, width=width, depth=depth, act=act)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.deep_o_net(x, u, y)
        return self.fno(x, out, y)
