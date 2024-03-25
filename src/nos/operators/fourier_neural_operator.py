from typing import (
    Optional,
)

import torch.nn as nn
from continuity.operators import (
    FourierNeuralOperator as FNO,
)
from continuity.operators import (
    OperatorShapes,
)

from .operator import (
    NeuralOperator,
)


class FourierNeuralOperator(FNO, NeuralOperator):
    def __init__(self, shapes: OperatorShapes, depth: int = 3, width: int = 3, act: Optional[nn.Module] = None):
        super().__init__(shapes=shapes, depth=depth, width=width, act=act)
        NeuralOperator.__init__(
            self,
            shapes=shapes,
            properties={
                "depth": depth,
                "width": width,
                "act": act.__class__.__name__,
            },
        )
