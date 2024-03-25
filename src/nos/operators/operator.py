import json
from abc import (
    ABC,
)

from continuity.operators import (
    OperatorShapes,
)


class NeuralOperator(ABC):
    """

    Args:
        properties: properties to fully describe the operator (used for serialization).
    """

    def __init__(self, properties: dict, shapes: OperatorShapes = None):
        self.properties = properties
        self.shapes = shapes

    def info(self):
        return json.dumps(self.properties, sort_keys=True, indent=4)
