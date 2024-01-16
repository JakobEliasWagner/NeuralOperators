import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def get_array(data_dir: pathlib.Path, element: ET.Element):
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
    tree = ET.parse(file)
    root = tree.getroot()

    # topo and geometry
    topology = get_array(file.parent, root.find(".//Topology"))
    geometry = get_array(file.parent, root.find(".//Geometry"))

    # values
    fs = root.find('.//Grid[@GridType="Collection"]')
    values = []
    frequencies = []
    for f in fs.findall(".//Grid"):
        frequencies.append(float(f.find("Time").attrib["Value"]))
        values.append(get_array(file.parent, f))

    return {
        "Topology": topology.squeeze(),
        "Geometry": geometry.squeeze(),
        "Values": np.array(values).squeeze(),
        "Frequencies": np.array(frequencies).squeeze(),
    }
