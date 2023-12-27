import pathlib
from abc import ABC, abstractmethod
import dolfinx

from .Domain import BoxDomain


class MeshStrategy(ABC):
    @abstractmethod
    def load_mesh(self) -> dolfinx.mesh.Mesh:
        pass


class MeshFromFile(MeshStrategy):
    def __init__(self, data_file: pathlib.Path):
        super().__init__()

    def load_mesh(self) -> dolfinx.mesh.Mesh:
        pass


class MeshFromBox(MeshStrategy):
    """creates a mesh within a box like dolfinx.mesh.Mesh structure

    """

    def __init__(self, domain: BoxDomain):
        self.domain = domain

    def load_mesh(self) -> dolfinx.mesh.Mesh:
        pass
