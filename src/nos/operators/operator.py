import json
from abc import (
    ABC,
)

from continuity.operators import (
    Operator,
)


class NosOperator(Operator, ABC):
    """

    Args:
        properties: properties to fully describe the operator (used for serialization).
    """

    def __init__(self, properties: dict):
        super().__init__()
        self.properties = properties

    def __str__(self):
        return json.dumps(self.properties, sort_keys=True, indent=4)
