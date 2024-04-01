from typing import (
    Optional,
)

import torch
import torch.nn as nn
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
        act_x: Optional[torch.nn.Module] = nn.Tanh(),
        act_u: Optional[torch.nn.Module] = nn.Tanh(),
        act_y: Optional[torch.nn.Module] = nn.Tanh(),
    ):
        super().__init__(shapes=shapes, K=K, N_1=N_1, D_1=D_1, N_2=N_2, D_2=D_2, a_x=act_x, a_u=act_u, a_y=act_y)
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "K": K,
                "N_1": N_1,
                "D_1": D_1,
                "N_2": N_2,
                "D_2": D_2,
                "act_x": act_x.__class__.__name__,
                "act_u": act_u.__class__.__name__,
                "act_y": act_y.__class__.__name__,
            },
        )
