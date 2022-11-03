import open3d as o3d
import json
import os
import numpy as np
from multiprocessing import Process
import argparse
import math


def load_json(file):
    """Read a json file.

    Parameters
    ----------
    file : str
        Path to a json file.

    Returns
    -------
    dict
        Contents of the json file.
    """
    with open(file) as f:
        dict = json.load(f)
    return dict


def load_seg_groups(filepath, filename):
    """Read the segmentation groups.

    Parameters
    ----------
    filepath : str
        Path to a json file.
    filename : str
        Name of the json file.

    Returns
    -------
    dict
        Segmentation groups.
    """
    file = filepath + filename
    dict = load_json(file)
    seg_groups = dict["segGroups"]
    return seg_groups


def load_seg_indices(filepath, filename):
    """Read the segmentation indices.

    Parameters
    ----------
    filepath : str
        Path to a json file.
    filename : str
        Name of the json file.

    Returns
    -------
    dict
        Segmentation indices.
    """
    file = filepath + filename
    dict = load_json(file)
    seg_indices = dict["segIndices"]
    seg_indices = np.array(seg_indices, dtype=np.int32)
    return seg_indices


def load_mesh(filepath, filename):
    """Load a triangle mesh file from disk.

    Parameters
    ----------
    filepath : str
        Path to a ply file.
    filename : str
        Name of the ply file.

    Returns
    -------
    o3d.geometry.Mesh
        Mesh.
    """
    file = filepath + filename
    mesh = o3d.io.read_triangle_mesh(file)
    return mesh


def get_object_idxs(seg_group, seg_indices):
    """Get the vertex indices which represent an object.

    Parameters
    ----------
    seg_group : list(int)
        Superpoint values.
    seg_indices : list(int)
        Indices of the superpoints.

    Returns
    -------
    o3d.geometry.Mesh
        Mesh.
    """
    superpoints = seg_group["segments"]
    superpoints = np.array(superpoints, dtype=np.int32)
    idxs = []
    for i in range(superpoints.shape[0]):
        superpoint = superpoints[i]
        idxs_i = np.where(seg_indices == superpoint)[0]
        idxs_i = idxs_i.reshape(idxs_i.shape[0], 1)
        idxs.append(idxs_i)
    idxs = np.vstack(idxs)
    idxs = idxs.reshape(idxs.shape[0], )
    return idxs


def get_ground_truth(scannet_dir, scene):
    filepath = scannet_dir + "/" + scene + "/"
    seg_indices_file = filepath + scene + "_vh_clean_2.0.010000.segs.json"
    mesh_file = filepath + scene + "_vh_clean_2.ply"

    seg_indices = load_seg_indices(filepath, scene + "_vh_clean_2.0.010000.segs.json")
    mesh = load_mesh(filepath, scene + "_vh_clean_2.ply")
    seg_groups = load_seg_groups(filepath, scene + ".aggregation.json")
    n = len(seg_groups)
    n_V = np.asarray(mesh.vertices).shape[0]

    partition_vec = np.zeros((n_V, ), dtype=np.int32)
    O = 1
    
    for i in range(n):
        seg_group = seg_groups[i]
        #label = seg_group["label"]
        O_idxs = get_object_idxs(seg_group, seg_indices)
        partition_vec[O_idxs] = O
        O += 1
    return mesh, partition_vec, mesh_file


def get_scannet_dir():
    return os.environ["SCANNET_DIR"] + "/scans"


def get_scenes(blacklist=[]):
    scannet_dir = os.environ["SCANNET_DIR"] + "/scans"
    scenes = os.listdir(scannet_dir)
    scene_dict = {}
    s_len = len("scene") + 4
    uni_scenes = []
    for i in range(len(scenes)):
        scene = scenes[i]
        key = scene[:s_len]
        if key in scene_dict:
            continue
        scene_dict[key] = key
        if scene in blacklist:
            continue
        uni_scenes.append(scene)
    return uni_scenes, scenes, scannet_dir
