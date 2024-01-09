from operator import add

from src.utility import run_once

from .Shape import Shape


class CShape(Shape):
    """2D C-shape

    C-shape is a specific kind of sonic crystal. They are usually arranged in a periodic fashion. A grid of these
    c-shapes can exhibit sub-frequency behavior, specific to meta-materials. An overview over meta-materials can be
    found in "Acoustic meta-materials: From local resonances to broad horizons" by Ma et al.

    Attributes:

    """

    def __init__(self, x: float, y: float, z: float, r_outer: float, r_inner: float, b: float):
        """

        Args:
           x: x-coordinate of center (defined as center of both arcs)
           y: y-coordinate of center (defined as center of both arcs)
           z: z-coordinate of center (defined as center of both arcs)
           r_outer: outer radius
           r_inner: inner radius
           b: width of the slit
        """
        super().__init__(x, y, z, r_outer, r_inner, b)

    @run_once
    def generate(self, x: float, y: float, z: float, r_outer: float, r_inner: float, b: float):
        """
        Args:
            x: x-coordinate of center (defined as center of both arcs)
            y: y-coordinate of center (defined as center of both arcs)
            z: z-coordinate of center (defined as center of both arcs)
            r_outer: outer radius
            r_inner: inner radius
            b: width of the slit
        """
        coordinates = [x, y, z]
        r = sorted([r_inner, r_outer])
        rect_offset = [0.0, -b / 2.0, 0.0]

        # basic shapes
        outer_disk = self.factory.addDisk(*coordinates, r[1], r[1])
        inner_disk = self.factory.addDisk(*coordinates, r[0], r[0])
        rectangle_coordinates = list(map(add, coordinates, rect_offset))
        slit = self.factory.addRectangle(*rectangle_coordinates, -r[1], b)

        # cut inner circle to get pipe and slit pipe to obtain C-shape
        self.factory.cut([(2, outer_disk)], [(2, inner_disk), (2, slit)])
