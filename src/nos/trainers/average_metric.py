class AverageMetric:  # noqa: D100
    """Metric that can be updated continously."""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """Initialize.

        Args:
            name (str): Name.
            fmt (_type_, optional): Formatting of the metric in a string. Defaults to ":f".

        """
        self.name = name
        self.fmt = fmt
        self.val = 0.0
        self.n_items = 0
        self.avg = 0.0

    def __call__(self) -> float:
        """Return metric average."""
        return self.avg

    def reset(self) -> None:
        """Reset all values in metric to zero."""
        self.val = 0.0
        self.n_items = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Update metric with val (consisting of n observations)."""
        self.n_items += n
        self.val += val

        self.avg = self.val / self.n_items

    def __str__(self) -> str:
        """Format metric as string."""
        return f"{self.name}: {self.avg: {self.fmt}}"

    def to_dict(self) -> dict:
        """Return metric as dict."""
        return {"name": self.name, "val": self.avg}
