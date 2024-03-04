class AverageMetric:
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0.0
        self.n_items = 0
        self.avg = 0.0

    def __call__(self):
        return self.avg

    def reset(self):
        self.val = 0.0
        self.n_items = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.n_items += n
        self.val += val

        self.avg = self.val / self.n_items

    def __str__(self):
        return f"{self.name}: {self.avg: {self.fmt}}"

    def to_dict(self):
        return {"name": self.name, "val": self.avg}
