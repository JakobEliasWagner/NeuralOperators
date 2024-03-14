from nos.benchmarks import transmission_loss_const_gap
from nos.data import TLDatasetCompact


def test_can_initialize():
    assert isinstance(transmission_loss_const_gap.train_set, TLDatasetCompact)
    assert isinstance(transmission_loss_const_gap.test_set, TLDatasetCompact)
