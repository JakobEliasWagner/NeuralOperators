from typing import (
    Optional,
)

import torch
from continuity.operators import (
    BelNet as BN,
)
from continuity.operators import (
    OperatorShapes,
)

from .operator import (
    NeuralOperator,
)


class BelNet(BN, NeuralOperator):
    def __init__(
        self,
        shapes: OperatorShapes,
        K: int = 8,
        N_1: int = 32,
        D_1: int = 3,
        N_2: int = 32,
        D_2: int = 3,
        a_x: Optional[torch.nn.Module] = torch.nn.Tanh(),
        a_u: Optional[torch.nn.Module] = torch.nn.Tanh(),
        a_y: Optional[torch.nn.Module] = torch.nn.Tanh(),
    ):
        super().__init__(shapes=shapes, K=K, N_1=N_1, D_1=D_1, N_2=N_2, D_2=D_2, a_x=a_x, a_u=a_u, a_y=a_y)
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "K": K,
                "N_1": N_1,
                "D_1": D_1,
                "N_2": N_2,
                "D_2": D_2,
                "a_x": a_x.__class__.__name__,
                "a_u": a_u.__class__.__name__,
                "a_y": a_y.__class__.__name__,
            },
        )
