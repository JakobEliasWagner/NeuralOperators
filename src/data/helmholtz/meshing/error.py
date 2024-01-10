class MeshConstructionError(Exception):
    """Exception raised for errors in the construction of a mesh."""

    def __init__(self, message="Mesh construction failed"):
        self.message = message
        super().__init__(self.message)
