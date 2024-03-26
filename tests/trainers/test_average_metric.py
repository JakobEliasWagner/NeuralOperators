import numpy as np

from nos.trainers.average_metric import (
    AverageMetric,
)


def test_can_initialize():
    some_metric = AverageMetric("hello", ":.4f")
    assert isinstance(some_metric, AverageMetric)


def test_average_correct():
    some_metric = AverageMetric("hello", ".2f")

    for i in range(50):
        some_metric.update(2.0, 2)
    assert np.isclose(some_metric(), 1.0)

    assert some_metric.to_dict()["name"] == "hello"
    assert np.isclose(some_metric.to_dict()["val"], 1.0)

    some_metric.reset()
    assert np.isclose(some_metric(), 0.0)

    print(some_metric)
    assert True
