from io import BytesIO
from typing import Dict

import h5py as h5
import numpy as np
from plyfile import PlyData, PlyElement


def construct_list_of_attributes(data: Dict):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(data["features_dc"].shape[1] * data["features_dc"].shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range(data["features_rest"].shape[1] * data["features_rest"].shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(data["scale"].shape[1]):
        l.append("scale_{}".format(i))
    for i in range(data["rotation"].shape[1]):
        l.append("rot_{}".format(i))
    return l


def _get_dataset(group: h5.Group, dataset_name: str):
    data = group.get(dataset_name)
    return np.array(data, dtype=data.dtype)


def load_point_cloud_from_h5(data: BytesIO):
    file = h5.File(data, mode="r")
    points = _get_dataset(file, "points")
    normals = _get_dataset(file, "normals")
    features_dc = _get_dataset(file, "features_dc")
    features_rest = _get_dataset(file, "features_rest")
    opacities = _get_dataset(file, "opacities")
    scale = _get_dataset(file, "scale")
    rotation = _get_dataset(file, "rotation")
    sh_degree = int(_get_dataset(file, "sh_degree")[0])
    file.close()

    data_dict = {}
    data_dict["points"] = points
    data_dict["normals"] = normals
    data_dict["features_dc"] = features_dc
    data_dict["features_rest"] = features_rest
    data_dict["opacities"] = opacities
    data_dict["scale"] = scale
    data_dict["rotation"] = rotation
    data_dict["sh_degree"] = sh_degree

    return data_dict


def convert_h5_to_ply(data_in: BytesIO, data_out: BytesIO):
    data_dict = load_point_cloud_from_h5(data_in)
    xyz = data_dict["points"]
    normals = data_dict["normals"]
    f_dc = np.squeeze(data_dict["features_dc"])
    d0, _, d2 = data_dict["features_rest"].shape
    # f_rest = data_dict["features_rest"].reshape(d0, d2)
    opacities = data_dict["opacities"]
    scale = data_dict["scale"]
    rotation = data_dict["rotation"]
    # rotation = data_dict["sh_degree"] = sh_degree

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(data_dict)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(data_out)


if __name__ == "__main__":
    h5file_path = "/home/ruben/Documents/Atlas/tiger_pcl.h5"
    in_buffer: BytesIO = BytesIO()
    out_buffer: BytesIO = BytesIO()
    with open(h5file_path, "rb") as f:
        in_buffer = BytesIO(f.read())

    with open("/home/ruben/Documents/Atlas/tiger3_pcl.ply", "wb") as f:
        convert_h5_to_ply(in_buffer, out_buffer)
        out_buffer.seek(0)
        f.write(out_buffer.getbuffer())
