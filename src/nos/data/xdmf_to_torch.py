import pathlib  # noqa: D100

import h5py
import numpy as np
import torch
from defusedxml import ElementTree


class DataItemNotFoundError(Exception):
    """DataItem not found or does not exist error."""


def get_array(data_dir: pathlib.Path, element: ElementTree.Element) -> torch.Tensor:
    """Get the first DataItem from a given element.

    Args:
        data_dir: directory in which both the xdmf and h5 files are located
        element: parent

    Returns:
        array of all elements stored in the first DataItem for the given element.

    """
    data = element.find(".//DataItem")
    txt: list[str]
    if isinstance(data, ElementTree.Element) and data.text is not None:
        txt = data.text.split(":")
    else:
        raise DataItemNotFoundError

    h5_file = txt[0]
    path = txt[1]

    with h5py.File(data_dir.joinpath(h5_file), "r") as file:
        d_set = file[path]
        out_arr = np.empty(d_set.shape)
        d_set.read_direct(out_arr)
    return torch.tensor(out_arr, dtype=torch.get_default_dtype())


def xdmf_to_torch(file: pathlib.Path) -> dict:
    """Convert xdmf file to a dictionary with all values stored within it.

    Args:
        file: path to the xdmf file (h5 file needs to be located in the same dir).

    Returns:
        Dictionary containing topology, geometry, frequencies, and complex values.

    """
    tree = ElementTree.parse(file)
    root = tree.getroot()

    # topo and geometry
    geometry = get_array(file.parent, root.find(".//Geometry"))

    # values
    fs = root.find('.//Grid[@GridType="Collection"]')
    values = []
    encodings = []
    for f in fs.findall(".//Grid"):
        encodings.append(float(f.find("Time").attrib["Value"]))
        real_f = f.find('.//Attribute[@Name="real_f"]')
        imag_f = f.find('.//Attribute[@Name="imag_f"]')
        values.append(torch.stack([get_array(file.parent, real_f), get_array(file.parent, imag_f)], dim=1).squeeze())
    value_tensor = torch.stack(values, dim=0)
    encoding_tensor = torch.tensor(encodings, dtype=torch.get_default_dtype())

    return {
        "Geometry": geometry,
        "Values": value_tensor,
        "Encoding": encoding_tensor,
    }
