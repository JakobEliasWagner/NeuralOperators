from abc import ABC, abstractmethod

import dolfinx.fem
import numpy as np

from src.data.helmholtz.domain_properties import Description


class WaveNumberModifier(ABC):
    @abstractmethod
    def eval(self, function_space: dolfinx.fem.FunctionSpace, ct: dolfinx.mesh.MeshTags) -> dolfinx.fem.Function:
        pass


class AdiabaticLayer(WaveNumberModifier):
    """Adiabatic layer to truncate a domain without reflections.

    Shown in Oskooi et al. "The failure of perfectly matched layers, and towards their redemption by adiabatic
    absorbers" Adiabatic layers asymptotically approach an adiabatic limit of zero reflections. This is done by
    gradually increasing absorption, which results in a truncated domain with only trivial reflections.
    This implementation assumes that the simulation domain without truncation is a rectangle.
    """

    def __init__(self, pd: Description, degree: int = 2):
        self.box_min = np.array([0.0, 0.0, 0.0])
        self.box_max = np.array([pd.width + pd.right_width, pd.height, 0.0])
        self.depth = pd.absorber_depth
        self.degree = degree
        rt = pd.round_trip
        self.sigma_0 = -(self.degree + 1) * np.log(rt) / (2.0 * self.depth)

    def eval(self, function_space: dolfinx.fem.FunctionSpace, ct: dolfinx.mesh.MeshTags) -> dolfinx.fem.Function:
        """Returns modification to the wave number caused by the adiabatic layer.

        Inside an absorbing layer, the wave number is modified with k = k0 + 1j * sigma, where sigma is a scaled shape
        function.

        Args:
            ct: not used here
            function_space:

        Returns: modification to the wave number.

        """

        def wave_mod(x: np.array) -> np.array:
            # Clamp point coordinates to the range defined by the box
            clamped = np.maximum(self.box_min[:, np.newaxis], np.minimum(x, self.box_max[:, np.newaxis]))

            # Compute the distance from the clamped points to the original points
            dist = np.linalg.norm(clamped - x, axis=0)

            # Relative distance inside absorber
            dist = dist / self.depth

            return self.sigma_0 * 1j * dist**self.degree

        f = dolfinx.fem.Function(function_space)
        f.interpolate(wave_mod)
        return f


class Crystals(WaveNumberModifier):
    def __init__(self, pd: Description):
        self.ref_index = pd.crystal_description.ref_index
        self.crystal_index = pd.crystal_index
        self.cut = pd.crystal_description.cut

    def eval(self, function_space: dolfinx.fem.FunctionSpace, ct: dolfinx.mesh.MeshTags) -> dolfinx.fem.Function:
        f = dolfinx.fem.Function(function_space)
        if self.cut:
            f.interpolate(lambda x: 0 * x[0])
        else:
            crystal_cells = ct.find(self.crystal_index)
            f.x.array[crystal_cells] = np.full_like(
                crystal_cells, (self.ref_index - 1), dtype=dolfinx.default_scalar_type
            )
        return f
