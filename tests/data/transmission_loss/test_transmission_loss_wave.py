import pytest

from nos.data import (
    TLDatasetCompactWave,
)
from nos.data.transmission_loss.transmission_loss_wave import (
    gaussian_modulated_sine_encoding,
    simple_sine_encoding,
)


@pytest.mark.slow
@pytest.mark.parametrize("encoding_function", [simple_sine_encoding, gaussian_modulated_sine_encoding])
def test_shape_correct(tl_paths, tl_dataset_sizes, encoding_function):
    for path in tl_paths:
        for size in tl_dataset_sizes:
            dataset = TLDatasetCompactWave(path=path, n_samples=size, wave_encoding=encoding_function)

            assert dataset.x.size(1) == dataset.u.size(1) == dataset.y.size(1) == dataset.v.size(1)
            assert dataset.x.size(2) == dataset.y.size(2)
