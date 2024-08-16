import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch


def get_array(data_dir: pathlib.Path, element: ET.Element) -> torch.tensor:
    """Gets the first DataItem from a given element.

    Args:
        data_dir: directory in which both the xdmf and h5 files are located
        element: parent

    Returns:
        array of all elements stored in the first DataItem for the given element.
    """
    data = element.find(".//DataItem")
    txt = data.text.split(":")

    h5_file = txt[0]
    path = txt[1]

    with h5py.File(data_dir.joinpath(h5_file), "r") as file:
        d_set = file[path]
        out_arr = np.empty(d_set.shape)
        d_set.read_direct(out_arr)
    out_arr = torch.tensor(out_arr)

    return out_arr


def xdmf_to_torch(file: pathlib.Path) -> dict:
    """Converts a xdmf file to a dictionary with all values stored within it.

    Args:
        file: path to the xdmf file (h5 file needs to be located in the same dir).

    Returns:
        Dictionary containing topology, geometry, frequencies, and complex values.
    """
    tree = ET.parse(file)
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
    values = torch.stack(values, dim=0)
    encodings = torch.tensor(encodings)

    return {
        "Geometry": geometry,
        "Values": values,
        "Encoding": encodings,
    }
