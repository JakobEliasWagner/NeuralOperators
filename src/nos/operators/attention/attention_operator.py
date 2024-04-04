from typing import (
    Callable,
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

from ..operator import (
    NeuralOperator,
)
from .function_encoder import (
    FunctionEncoder,
)


class AttentionOperator(Operator, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        encoding_dim: int = 16,
        n_layers: int = 1,
        n_heads: int = 1,
        dropout: float = 0,
        act: nn.Module = nn.Tanh(),
        attention: Callable = nn.functional.scaled_dot_product_attention,
    ):
        super().__init__()

        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "encoding_dim": encoding_dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dropout": dropout,
                "act": act.__class__.__name__,
                "attention": attention.__name__,
            },
        )

        self.shapes = shapes

        # cross attention
        self.query_encoder = nn.Sequential(nn.Linear(shapes.y.dim, encoding_dim), ResNet(encoding_dim, 2, act=act))
        self.value_encoder = nn.Sequential(
            nn.Linear(shapes.u.dim + shapes.x.dim, encoding_dim), ResNet(encoding_dim, 2, act=act)
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(shapes.u.dim + shapes.x.dim, encoding_dim), ResNet(encoding_dim, 2, act=act)
        )
        self.attn = attention
        self.ffn = ResNet(encoding_dim, 4, act=act, stride=2)

        # self attention blocks
        self.function_layers = nn.Sequential()
        for i in range(n_layers):
            self.function_layers.add_module(f"function_encoder_{i}", FunctionEncoder(encoding_dim, n_head=n_heads))

        # projection
        self.projection = nn.Linear(encoding_dim, shapes.v.dim)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        queries = self.query_encoder(y)
        xu = torch.cat((x, u), dim=-1)
        vals = self.value_encoder(xu)
        keys = self.key_encoder(xu)

        out = self.attn(queries, keys, vals)
        out = out + queries
        out = self.ffn(out)

        for mod in self.function_layers:
            out = mod(out)

        return self.projection(out)
