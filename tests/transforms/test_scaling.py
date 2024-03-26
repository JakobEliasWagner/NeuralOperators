import torch

from nos.transforms import (
    MinMaxScale,
)


def test_can_initialize():
    tf = MinMaxScale(torch.zeros(3, 3), torch.ones(3, 3))
    assert isinstance(tf, MinMaxScale)


def test_scale_in_bounds():
    vec = torch.rand(100)
    tf = MinMaxScale(torch.min(vec), torch.max(vec))
    out = tf(vec)

    assert torch.all(torch.greater_equal(out, -torch.ones(vec.shape)))
    assert torch.all(torch.less_equal(out, torch.ones(vec.shape)))
    assert torch.min(out) == -1
    assert torch.max(out) == 1


def test_can_undo():
    vec = torch.rand(100)
    tf = MinMaxScale(torch.min(vec), torch.max(vec))
    out = tf(vec)
    vec_undone = tf.undo(out)

    assert torch.allclose(vec, vec_undone)
