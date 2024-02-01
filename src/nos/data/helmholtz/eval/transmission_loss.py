from typing import Optional

import numpy as np
from scipy.integrate import simpson

from nos.data.helmholtz import HelmholtzDataset


def transmission_loss(data_set: HelmholtzDataset, p_0: Optional[float] = None) -> np.array:
    """

    Args:
        data_set:
        p_0:

    Returns:
        array containing the transmission loss.
    """

    # extract values (x, y, pin, pout)
    # in
    idx_in = np.where(np.isclose(data_set.x[:, 0], data_set.description.crystal_box.x_min))[0]
    y_in = data_set.x[idx_in, 1]
    if p_0:
        p_in = np.ones(data_set.p[:, idx_in].shape)
    else:
        p_in = data_set.p[:, idx_in]

    # out
    idx_out = np.where(np.isclose(data_set.x[:, 0], data_set.description.crystal_box.x_max))[0]
    y_out = data_set.x[idx_out, 1]
    p_out = data_set.p[:, idx_out]

    # pre-process
    p_out = np.real(p_out) ** 2 + np.imag(p_out) ** 2
    p_in = np.real(p_in) ** 2 + np.imag(p_in) ** 2

    # sort
    o_in = np.argsort(y_in)
    y_in = y_in[o_in]
    p_in = p_in[:, o_in]

    o_out = np.argsort(y_out)
    y_out = y_out[o_out]
    p_out = p_out[:, o_out]

    # integrate with some scheme
    mock_power_in = simpson(y=p_in, x=y_in, even="first")
    mock_power_out = simpson(y=p_out, x=y_out, even="first")

    # mock_power_in = np.mean(p_in, axis=1)
    # mock_power_out = np.mean(p_out, axis=1)

    return 10 * np.log10(mock_power_out / mock_power_in)
