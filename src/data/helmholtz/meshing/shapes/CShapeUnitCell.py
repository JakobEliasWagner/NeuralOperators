from src.utility import run_once

from .CShape import CShape
from .Rectangle import Rectangle
from .Shape import Shape


class CShapeUnitCell(Shape):
    """Unit Cell or crystal with a c-shape inside

    Attributes:

    """

    def __init__(
        self, x: float, y: float, z: float, dx: float, dy: float, r_outer: float, r_inner: float, b, cut: bool = False
    ):
        """Unit cell with c-shape placed at the center

        Args:
            x: coordinate
            y: coordinate
            z: coordinate
            dx: x-dimension
            dy: y-dimension
            r_outer: c-shape outer radius
            r_inner: c-shape inner radius
            b: slit width of c-shape
            cut: c-shape should be cut from cell mesh
        """
        raise DeprecationWarning("DO not use")
        super().__init__(x, y, z, dx, dy, r_outer, r_inner, b, cut)

    @run_once
    def generate(
        self, x: float, y: float, z: float, dx: float, dy: float, r_outer: float, r_inner: float, b, cut: bool = True
    ) -> None:
        """

        Args:
            x: coordinate
            y: coordinate
            z: coordinate
            dx: x-dimension
            dy: y-dimension
            r_outer: c-shape outer radius
            r_inner: c-shape inner radius
            b: slit width of c-shape
            cut: c-shape should be cut from cell mesh

        Returns: None

        """
        center_x = x + dx / 2
        center_y = y + dy / 2

        domain = Rectangle(x, y, z, dx, dy)
        c_shape = CShape(center_x, center_y, z, r_outer, r_inner, b)

        # cut or fragment for final domain
        self.children[domain.__class__.__name__] = domain
        if cut:
            self.factory.cut(domain(), c_shape())
            return
        self.factory.fragment(domain(), c_shape())
        self.children[c_shape.__class__.__name__] = c_shape
