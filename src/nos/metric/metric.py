from abc import ABC


class Metric(ABC):
    def __init__(self, name: str, dataset_name: str):
        self.name = name
        self.dataset_name = dataset_name

    def on_train_begin(self):
        self.create_log_entry("begin", 0.0)

    def on_train_end(self):
        self.create_log_entry("end", 0.0)

    def on_epoch(self, epoch: int):
        self.create_log_entry(f"epoch-{epoch}", 0.0)

    def create_log_entry(self, stage: str, value: float):
        log_entry = {"stage": stage, "dataset": self.dataset_name, "metric": self.name, "value": value}
        return log_entry
