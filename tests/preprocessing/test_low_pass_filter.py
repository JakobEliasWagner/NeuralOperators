import torch

from nos.preprocessing import (
    LowPassFilter1D,
)


class TestLowPassFilter1D:
    def test_can_initialize(self):
        kernel_size = 5
        cutoff_frequency = 10
        sampling_rate = 100

        f = LowPassFilter1D(kernel_size, cutoff_frequency, sampling_rate)

        # Kernel size increases by 1 if it's initially even
        assert f.kernel_size == kernel_size
        assert len(f.filter_kernel) == kernel_size

        # Check that the filter kernel sum is approximately 1 for normalization
        assert torch.isclose(torch.sum(f.filter_kernel), torch.tensor(1.0), atol=1e-5)

    def test_forward_impulse_response(self):
        kernel_size = 5
        f = LowPassFilter1D(kernel_size, 10, 100)
        x = torch.zeros(10)
        x[len(x) // 2] = 1
        output = f.forward(x)

        assert len(output) == len(x)
        # The sum of the output should be close to 1 due to the filter normalization
        assert torch.isclose(torch.sum(output), torch.tensor(1.0), atol=1e-5)
