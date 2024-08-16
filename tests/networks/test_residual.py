import torch
import torch.nn as nn

from nos.networks import (
    ResBlock,
    ResNet,
)


def test_block_can_initialize():
    block = ResBlock(32, 4, nn.Tanh())
    assert isinstance(block, ResBlock)


def test_block_forward():
    vec = torch.rand(
        32,
    )
    block = ResBlock(32, 4, nn.Tanh())
    out = block(vec)
    assert out.shape == vec.shape


def test_can_initialize():
    net = ResNet(width=32, depth=4, act=nn.Tanh(), stride=2)
    assert isinstance(net, ResNet)


def test_can_initialize_without_transformations():
    net = ResNet(16, 6, nn.Tanh(), 2)
    assert isinstance(net, ResNet)


def test_forward():
    vec = torch.rand(10, 32)
    net = ResNet(width=32, depth=4, act=nn.Tanh(), stride=2)
    out = net(vec)
    assert out.shape == vec.shape
