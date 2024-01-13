import pathlib

from .domain_properties import read_config
from .solver import HelmholtzSolver


class Helmholtz:
    """Facade for generating datasets for different descriptions of sonic crystals in the acoustic domain.

    The Helmholtz equation is a partial differential equation derived from applying the Laplace operator to a function
    and equating it to the negative of the function multiplied by a constant. It describes the behavior of wave
    phenomena, such as sound waves, electromagnetic waves, or quantum mechanical waves, in a given medium. This equation
    is fundamental in various fields of physics and engineering, where it's used to understand wave propagation and
    vibration in different contexts. In this context, only applications to acoustic problems are researched, though this
    code could be extended to be agnostic to the type of underlying problem.
    Acoustic sonic crystals are artificial structures made of periodic arrangements of materials with different acoustic
    properties, designed to affect the propagation of sound waves. They manipulate sound waves through phenomena like
    diffraction and interference, leading to unique effects like sound filtering, sound guiding, and the creation of
    acoustic band gaps. These structures are used in various applications, including noise reduction, sound control in
    buildings, and the development of new acoustic devices.
    """

    def __init__(self, problem_description_file: pathlib.Path, out_dir: pathlib.Path):
        self.descriptions = read_config(problem_description_file)
        self.out_dir = out_dir

    def run(self):
        solver = HelmholtzSolver(self.out_dir, ("Lagrange", 3))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for description in self.descriptions:
            description.save_to_json(self.out_dir)
            solver(description)
