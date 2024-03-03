class Benchmark:
    def __init__(self, datasets, metrics):
        self.datasets = datasets  # A dictionary of datasets, e.g., {'train': ..., 'test': ...}
        self.metrics = {metric_name: metric() for metric_name, metric in metrics.items()}
        self.logs = []

    def evaluate(self, model, stage="post"):
        for dataset_name, dataset in self.datasets.items():
            for metric_name, metric in self.metrics.items():
                metric.reset()
                for data in dataset:  # Assuming dataset yields (inputs, targets)
                    predictions = model.predict(data[0])
                    metric.update(predictions, data[1])
                self.log_metric(stage, dataset_name, metric_name, metric.get_value())

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch(self, epoch: int):
        pass

    def log_metric(self, stage, dataset_name, metric_name, value):
        self.logs.append((stage, dataset_name, metric_name, value))

    def get_logs(self):
        return self.logs
