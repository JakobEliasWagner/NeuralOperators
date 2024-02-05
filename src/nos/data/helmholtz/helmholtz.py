import multiprocessing as mp
import pathlib
import time

from nos.utility import ProgressMonitor

from .domain_properties import read_config
from .solver import HelmholtzSolver


class Helmholtz:
    """Facade for generating datasets for different descriptions of sonic crystals in the acoustic domain.

    $$\

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

    def run(self, n_threads: int = 1):
        """Generates the dataset for the current description in parallel.

        :param n_threads:
        :return:
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # setup multi-processing
        n_threads_actual = min([n_threads, len(self.descriptions), mp.cpu_count()])
        pool = mp.Pool(processes=n_threads_actual)
        manager = mp.Manager()
        queue = manager.Queue()

        start = time.time_ns()

        args = [(i, description, queue) for i, description in enumerate(self.descriptions)]
        result = pool.map_async(self.run_single_description, args)

        # monitoring loop
        ProgressMonitor.monitor_pool(
            result,
            queue,
            len(self.descriptions),
            prefix="Helmholtz: ",
            suffix=f"Running on {n_threads_actual} threads.",
        )

        end = time.time_ns()
        print(f"\nTotal runtime: {(end - start) / 1e+9:.4f}")

    def run_single_description(self, args):
        i, description, queue = args
        description.save_to_json(self.out_dir)
        solver = HelmholtzSolver(self.out_dir, ("CG", 2))
        solver(description)

        # update queue
        queue.put(i)
