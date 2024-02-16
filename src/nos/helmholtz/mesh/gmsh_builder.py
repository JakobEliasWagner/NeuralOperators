from abc import ABC, abstractmethod
from typing import List

import gmsh

from nos.data.helmholtz.domain_properties import Description


class GmshBuilder(ABC):
    """Abstract builder for all builders using the gmsh library."""

    def __init__(self, description: Description):
        self.factory = gmsh.model.occ
        self.description = description

    @abstractmethod
    def build(self) -> List[int]:
        """Builds structures or mesh.

        Returns:
            indices to all net elements created by this builder.
        """
        pass
