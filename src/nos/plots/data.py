import json
import pathlib
from dataclasses import (
    dataclass,
)
from typing import (
    List,
    Self,
)

import pandas as pd
import yaml

from nos.operators import (
    NeuralOperator,
    deserialize,
)


@dataclass
class ModelData:
    path: pathlib.Path
    name: str
    operator: NeuralOperator
    checkpoint: dict

    @classmethod
    def from_dir(cls, path: pathlib.Path) -> Self:
        operator = deserialize(path)

        checkpoint_path = path.joinpath("checkpoint.json")
        with open(checkpoint_path, "r") as fh:
            checkpoint = json.load(fh)

        return cls(path, path.name, operator, checkpoint)


@dataclass
class RunData:
    path: pathlib.Path
    name: str
    log: str
    training: pd.DataFrame
    choices: dict
    training_config: dict
    models: List[ModelData]

    @classmethod
    def from_dir(cls, path: pathlib.Path) -> Self:
        models = []
        models_dir = path.joinpath("models")
        for model_dir in models_dir.glob("*_*_*_*_*_*"):
            models.append(ModelData.from_dir(model_dir))

        log_file = path.joinpath("benchmark.log")
        with open(log_file, "r") as fh:
            log = "\n".join(fh.readlines())

        choices_file = models_dir.joinpath("choices.json")
        with open(choices_file, "r") as fh:
            choices = json.load(fh)

        name = choices["operator"]

        training_file = models_dir.joinpath("training.csv")
        training = pd.read_csv(training_file)

        training_config_file = models_dir.joinpath("training_config.json")
        with open(training_config_file, "r") as fh:
            training_config = json.load(fh)

        return cls(path, name, log, training, choices, training_config, models)


@dataclass
class MultiRunData:
    path: pathlib.Path
    name: str
    runs: List[RunData]
    results: dict
    config: dict

    @classmethod
    def from_dir(cls, path: pathlib.Path) -> Self:
        runs = []
        for element in path.glob("*"):
            if element.is_file():
                continue
            if "plots" in element.name:
                continue
            runs.append(RunData.from_dir(element))

        results_file = path.joinpath("results.json")
        with open(results_file, "r") as fh:
            results = json.load(fh)

        config_file = path.joinpath("multirun.yaml")
        with open(config_file, "r") as fh:
            config = yaml.load(fh, Loader=yaml.FullLoader)

        return cls(path, path.name, runs, results, config)
