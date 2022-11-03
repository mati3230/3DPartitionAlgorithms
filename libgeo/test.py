import numpy as np
import open3d as o3d
import sys
import argparse
if sys.platform == "win32":
    sys.path.append("./build/Release")
else: # linux, macos
    sys.path.append("./build")
import libgeo
sys.path.append("../python_utils")
import visu_utils
import graph_utils
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../sn000000.ply", type=str, help="Path to Open3D radable file (ply, obj, ...)")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.file)
    
    mesh.compute_adjacency_list()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    P = np.hstack((xyz, rgb))

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.

    n_points = P.shape[0]
    print("Point cloud has {0} points".format(n_points))

    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=mesh.adjacency_list)
    if uni_verts.shape[0] != xyz.shape[0]:
        print("Error: There are points which are target only")
        return
    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)

    source_idx = 12
    target_idx = 1000
    exclude_closest = True

    # uncomment the blocks to test the functions of the libgeo!
    """
    t1 = time.time()
    s_dists = libgeo.one_to_all(source_idx, source, target, distances, uni_verts.shape[0])
    t2 = time.time()
    distance = s_dists[target_idx]
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for one to all paths from {1} to {2} with distance {3:.3f}".format(cpp_time, source_idx, target_idx, distance))  
    #print(uni_verts.shape[0] * k)
    """

    """ really slow!
    source_idx = 0
    target_idx = 1000
    k = 30
    t1 = time.time()
    a_dists = libgeo.all_to_all_k(source, target, distances, uni_verts.shape[0], k)
    t2 = time.time()
    distance = a_dists[source_idx][target_idx]
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for all to all paths from {1} to {2} with distance {3:.3f}".format(cpp_time, source_idx, target_idx, distance))  
    #print(uni_verts.shape[0] * k)
    """

    """
    t1 = time.time()
    distance = libgeo.geodesic_path(source_idx, target_idx, uni_verts, direct_neigh_idxs, n_edges, target, distances)
    t2 = time.time()
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for the geodesic path from {1} to {2} with distance {3:.3f}".format(cpp_time, source_idx, target_idx, distance))  
    #print(uni_verts.shape[0] * k)
    """

    """
    k = 15
    t1 = time.time()
    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
    t2 = time.time()
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for the geodesic knn search (k={1})".format(cpp_time, k))  
    #print(uni_verts.shape[0] * k)
    """

    #"""
    depth = 3
    t1 = time.time()
    source_, target_, distances_, ok = libgeo.geodesic_neighbours(uni_verts, direct_neigh_idxs, n_edges, target, distances, depth, uni_verts.shape[0], exclude_closest)
    t2 = time.time()
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for the geodesic neighbourhood search (depth={1})".format(cpp_time, depth))  
    if(not ok[0]):
        print("Some error happened while calculating geodesic neighbours")
    #print(uni_verts.shape[0] * k)
    #"""

    """
    radius = 0.05
    exclude_closest=False
    t1 = time.time()
    source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, exclude_closest)
    t2 = time.time()
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds for the geodesic radius nn search (radius r={1})".format(cpp_time, radius))  
    print(source_.shape, target_.shape, distances_.shape)
    #print(distances_[:100])
    """
    """
    t1 = time.time()
    mask = libgeo.unidirectional(uni_verts, direct_neigh_idxs, n_edges, target)
    mask = mask.astype(np.bool)
    #print(mask.shape, mask.dtype)
    source_ = source[mask]
    target_ = target[mask]
    distances_ = distances[mask]
    t2 = time.time()
    cpp_time = t2 - t1
    print("C++ interface: {0:.3f} seconds".format(cpp_time))
    
    t1 = time.time()
    d_mesh = {
        "source": source,
        "target": target,
        "distances": distances
    }
    d_mesh = graph_utils.clean_edges_threads(d_mesh=d_mesh, verbose=False)
    #d_mesh = graph_utils.clean_edges(d_mesh=d_mesh, verbose=False)
    t2 = time.time()
    py_time = t2 - t1
    print("Python interface: {0:.3f} seconds".format(py_time))
    print("C++ is {0:.3f} times slower/faster than Python".format(py_time / cpp_time))

    #print(source_.shape, target_.shape, distances_.shape)
    #print(d_mesh["c_source"].shape, d_mesh["c_target"].shape, d_mesh["c_distances"].shape)
    assert(source_.shape[0] == d_mesh["c_source"].shape[0])
    """
if __name__ == "__main__":
    main()