import pathlib

from nos.data import (
    PulsatingSphere,
)
from nos.operators import (
    deserialize,
)

operator = deserialize(pathlib.Path.cwd().joinpath("finished_pi", "DeepDotOperator_2024_04_09_16_51_07"))
dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500"))
