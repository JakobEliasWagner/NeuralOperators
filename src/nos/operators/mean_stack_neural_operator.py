"""
`continuity.operators.deep_neural_operator`

The Deep Neural Operator architecture.
"""

import torch
import torch.nn as nn
from continuity.operators.shape import (
    OperatorShapes,
)

from nos.networks import (
    ResNet,
)

from .operator import (
    NeuralOperator,
)


class MeanStackNeuralOperator(NeuralOperator):
    """
    The `MeanStackNeuralOperator` class integrates a deep residual network within a neural operator framework. It uses all
    scalar values of the input locations, input functions, and individual evaluation points as inputs for a deep
    residual network.

    Args:
        shapes: An instance of `DatasetShapes`.
        width: The width of the Deep Residual Network, defining the number of neurons in each hidden layer.
        depth: The depth of the Deep Residual Network, indicating the number of hidden layers in the network.

    """

    def __init__(
        self, shapes: OperatorShapes, width: int = 32, depth: int = 3, act: nn.Module = nn.Tanh(), stride: int = 1
    ):
        super().__init__(properties={"width": width, "depth": depth, "act": act.__class__.__name__}, shapes=shapes)

        self.width = width
        self.depth = depth

        self.lift = nn.Linear(shapes.x.dim + shapes.y.dim + shapes.u.dim, width)
        self.hidden = ResNet(width=width, depth=depth, act=act, stride=stride)
        self.project = nn.Linear(width, shapes.v.dim)
        self.net = nn.Sequential(self.lift, self.hidden, self.project)

    def forward(self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the operator.

        Performs the forward pass through the operator, processing the input function values `u` and input function
        probe locations `x` by flattening them. They are then expanded to match the dimensions of the evaluation
        coordinates y. The preprocessed x, preprocessed u, and y are stacked and passed through a deep residual network.


        Args:
            x: Input coordinates of shape (batch_size, #sensors, x_dim), representing the points in space at
                which the input function values are probed.
            u: Input function values of shape (batch_size, #sensors, u_dim), representing the values of the input
                functions at different sensor locations.
            y: Evaluation coordinates of shape (batch_size, #evaluations, y_dim), representing the points in space at
                which the output function values are to be computed.

        Returns:
            The output of the operator, of shape (batch_size, #evaluations, v_dim), representing the computed function
                values at the specified evaluation coordinates.
        """
        x_repeated = x.unsqueeze(1).expand(-1, y.size(1), -1, -1)
        u_repeated = u.unsqueeze(1).expand(-1, y.size(1), -1, -1)
        y_repeated = y.unsqueeze(2).expand(-1, -1, x.size(1), -1)

        net_input = torch.cat([x_repeated, u_repeated, y_repeated], dim=-1)

        output = self.net(net_input)

        return torch.mean(output, dim=-2)
