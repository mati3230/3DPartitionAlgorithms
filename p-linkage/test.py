import numpy as np
import open3d as o3d
import argparse
import sys
if sys.platform == "win32":
    sys.path.append("./build/Release")
    sys.path.append("../libgeo/build/Release")
elif sys.platform.startswith("linux"): # linux
    sys.path.append("./build")
    sys.path.append("../libgeo/build")
else: # apple
    sys.path.append("./build")
    sys.path.append("../libgeo/build")
import libplink
import libgeo
sys.path.append("../python_utils")
import visu_utils
import graph_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../sn000000.ply", type=str, help="Path to Open3D radable file (ply, obj, ...)")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.file)

    mesh.compute_adjacency_list()
    mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    normals = np.asarray(mesh.vertex_normals)
    P = np.hstack((xyz, rgb))

    P = P.astype(np.float32)
    normals = normals.astype(np.float32)

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.

    n_points = P.shape[0]
    print("Point cloud has {0} points".format(n_points))

    angle = 90
    k = 100
    min_cluster_size = 10
    angle_dev = 10.0

    point_idxs, p_vec, duration = libplink.plinkage(P, k=k, angle=angle, min_cluster_size=min_cluster_size,angle_dev=angle_dev)
    if sys.platform == "darwin": # apple
        p_vec = np.array(p_vec, dtype=np.int32)

    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("PLinkage: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)

    point_idxs, p_vec = graph_utils.refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=mesh.adjacency_list)


    p_idxs_of_first_sp = point_idxs[0]
    sp = P[p_idxs_of_first_sp]

    for i in range(len(point_idxs)):
        sp_idxs = point_idxs[i]
        sp = P[sp_idxs]

    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("PLinkage: {0} superpoints, {1:.2f}% outliers".format(len(point_idxs), outliers))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)

    #"""
    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=mesh.adjacency_list)

    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)

    exclude_closest = False
    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
    use_normals = False
    point_idxs, p_vec, duration = libplink.plinkage_geo(P, target_, normals, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev, use_normals=use_normals)
    if sys.platform == "darwin": # apple
        p_vec = np.array(p_vec, dtype=np.int32)

    outliers = 100 * np.sum(p_vec == -1) / n_points

    print("PLinkage Geo: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)

    point_idxs, p_vec = graph_utils.refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=mesh.adjacency_list)
    
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("PLinkage Geo: {0} superpoints, {1:.2f}% outliers".format(len(point_idxs), outliers))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
    #"""
if __name__ == "__main__":
    main()