import json
from abc import (
    ABC,
)

from continuity.operators import (
    Operator,
    OperatorShapes,
)


class NosOperator(Operator, ABC):
    """

    Args:
        properties: properties to fully describe the operator (used for serialization).
    """

    def __init__(self, properties: dict, shapes: OperatorShapes = None):
        super().__init__()
        self.properties = properties
        self.shapes = shapes

    def __str__(self):
        return json.dumps(self.properties, sort_keys=True, indent=4)
