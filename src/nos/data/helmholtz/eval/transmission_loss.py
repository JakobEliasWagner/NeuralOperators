import pathlib
from typing import Optional

import numpy as np

from nos.data.helmholtz.domain_properties import read_from_json
from nos.data.utility import xdmf_to_numpy


def transmission_loss(description_file: pathlib.Path, p_0: Optional[float] = None) -> np.array:
    """

    Args:
        p_0: reference incomming
        description_file: path to the description json.

    Returns:
        array containing frequencies and transmission loss.
    """
    # id
    uid = description_file.name.split("_")[0]
    data_file = description_file.parent.joinpath(f"{uid}_solution.xdmf")

    # get description
    description = read_from_json(description_file)

    # get the correct data
    data = xdmf_to_numpy(data_file)

    # extract values (x, y, pin, pout)
    # in
    idx_in = np.where(np.isclose(data["Geometry"][:, 0], description.crystal_box.x_min))[0]
    y_in = data["Geometry"][idx_in, 1]
    p_in = data["Values"][:, idx_in]

    # out
    idx_out = np.where(np.isclose(data["Geometry"][:, 0], description.crystal_box.x_max))[0]
    y_out = data["Geometry"][idx_out, 1]
    p_out = data["Values"][:, idx_out]

    # pre-process
    p_out = np.real(p_out) ** 2 + np.imag(p_out) ** 2
    if p_0:
        p_in = np.ones(p_in.shape)
    else:
        p_in = np.real(p_in) ** 2 + np.imag(p_in) ** 2

    # sort
    o_in = np.argsort(y_in)
    y_in = y_in[o_in]
    p_in = p_in[:, o_in]

    o_out = np.argsort(y_out)
    y_out = y_out[o_out]
    p_out = p_out[:, o_out]

    # integrate with some scheme
    # mock_power_in = simpson(y=p_in, x=y_in, even='first')
    # mock_power_out = simpson(y=p_out, x=y_out, even='first')

    mock_power_in = np.mean(p_in, axis=1)
    mock_power_out = np.mean(p_out, axis=1)

    tl = 10 * np.log10(mock_power_out / mock_power_in)

    return np.stack([data["Frequencies"], tl])
