import json
import pathlib
from datetime import datetime
from typing import Dict, List

from continuity.data import OperatorDataset
from continuity.operators import Operator
from nos.metric import Metric


class Benchmark:
    def __init__(
        self,
        train_set: OperatorDataset,
        test_set: OperatorDataset,
        operators: List[Operator],
        metrics: List[Metric],
        trainer=None,
        out_dir: pathlib.Path = None,
    ):
        # core benchmark setup
        self.train_set = train_set
        self.test_set = test_set
        self.operators = operators
        self.metrics = metrics

        # actors
        self.trainer = trainer

        # benchmark utilities
        self.time_stamp = datetime.now()
        self.name = f"{self.__class__.__name__}_{self.time_stamp.strftime('%Y_%m_%d_%H_%M_%S')}"
        self.out_dir = out_dir.joinpath(self.name)
        self.out_dir.mkdir(parents=True, exist_ok=False)

    def run(self):
        for operator in self.operators:
            # Train the operator
            trained_operator = self.train_operator(operator, self.train_set)

            # Calculate and log metrics
            trained_operator.eval()
            out = {}
            for metric in self.metrics:
                out[str(metric)] = metric.calculate(operator=operator, dataset=self.test_set)

            # best parameters to json
            out = {operator.__class__.__name__: out}
            self.write_results_to_json(out)

    def train_operator(self, operator: Operator, data_set: OperatorDataset) -> Operator:
        return self.trainer(operator, data_set, 20)

    def write_results_to_json(self, out: Dict):
        json_file = self.out_dir.joinpath("metrics.json")
        # load and update current file contents
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
            run_id = next(iter(out))
            while run_id in data:
                run_id += "i"
            out[run_id] = out[next(iter(out))]
            out.update(data)

        # write total data to file
        with open(json_file, "w") as f:
            json.dump(out, f)
