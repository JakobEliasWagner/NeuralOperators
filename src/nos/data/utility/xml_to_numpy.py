import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def get_array(data_dir: pathlib.Path, element: ET.Element) -> np.array:
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

    return out_arr


def xdmf_to_numpy(file: pathlib.Path) -> dict:
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
    frequencies = []
    for f in fs.findall(".//Grid"):
        frequencies.append(float(f.find("Time").attrib["Value"]))
        real_f = f.find('.//Attribute[@Name="real_f"]')
        imag_f = f.find('.//Attribute[@Name="imag_f"]')
        values.append(get_array(file.parent, real_f) + 1j * get_array(file.parent, imag_f))

    return {
        "Geometry": geometry.squeeze(),
        "Values": np.array(values).squeeze(),
        "Frequencies": np.array(frequencies).squeeze(),
    }
