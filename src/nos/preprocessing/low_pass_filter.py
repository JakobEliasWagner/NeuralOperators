import torch


class LowPassFilter1D(torch.nn.Module):
    def __init__(self, kernel_size, cutoff_frequency, sampling_rate):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        nyquist_rate = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_rate

        # Generate the filter kernel
        ideal_response = torch.zeros(kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            if i == center:
                ideal_response[i] = 2 * normalized_cutoff
            else:
                ideal_response[i] = torch.sin(torch.tensor(2 * torch.pi * normalized_cutoff * (i - center))) / (
                    torch.pi * (i - center)
                )

        window = torch.hann_window(kernel_size)
        self.filter_kernel = ideal_response * window
        self.filter_kernel /= torch.sum(self.filter_kernel)

    def forward(self, x):
        output_length = len(x)
        output = torch.empty(output_length)

        x_padded = torch.zeros(output_length + self.kernel_size - 1)
        x_padded[self.kernel_size // 2 : -self.kernel_size // 2 + 1] = x

        for i in range(output_length):
            j = i + self.kernel_size // 2
            window = x_padded[j - self.kernel_size // 2 : j + self.kernel_size // 2 + 1]

            output[i] = torch.dot(window, self.filter_kernel)

        return output
