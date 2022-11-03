import numpy as np
import open3d as o3d
import argparse
import sys
if sys.platform == "win32":
    sys.path.append("./build/Release")
    sys.path.append("../libgeo/build/Release")
else: # linux, macos
    sys.path.append("./build")
    sys.path.append("../libgeo/build")
import libvccs
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
    P = np.hstack((xyz, rgb, normals))
    P = P.astype(np.float32)

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.
    n_points = P.shape[0]
    print("Point cloud has {0} points".format(n_points))
    
    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=mesh.adjacency_list)
    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)

    """
    # this parameter set causes longer computational duration but produces less superpoints
    voxel_resolution=0.1
    seed_resolution=0.5
    color_importance=0.2
    spatial_importance=0.4
    normal_importance=0.4
    refinementIter=0
    """
    #"""
    voxel_resolution=0.5
    seed_resolution=0.5
    color_importance=0.3
    spatial_importance=0.3
    normal_importance=0.3
    refinementIter=0
    r_search_gain = 0.5
    #"""

    point_idxs, p_vec, duration = libvccs.vccs(P, voxel_resolution=voxel_resolution, seed_resolution=seed_resolution,
        color_importance=color_importance, spatial_importance=spatial_importance, normal_importance=normal_importance,
        refinementIter=refinementIter)
    
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("VCCS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=True)

    precalc = True
    radius = r_search_gain * seed_resolution
    exclude_closest=False
    print("geo rnn")
    source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, exclude_closest)
    print("done")
    source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
    source_ = source_.astype(np.uint32)
    target_ = target_.astype(np.uint32)
    distances_ = distances_.astype(np.float32)

    uni_verts_ = uni_verts_.astype(np.uint32)
    direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
    n_edges_ = n_edges_.astype(np.uint32)
    print("Calculated neighbours for radius {0}".format(radius))

    point_idxs, p_vec, duration = libvccs.vccs_mesh(P, uni_verts_, direct_neigh_idxs_, n_edges_, source_, target_, distances_, 
        voxel_resolution=voxel_resolution, seed_resolution=seed_resolution, color_importance=color_importance,
        spatial_importance=spatial_importance, normal_importance=normal_importance, refinementIter=refinementIter,
        r_search_gain=r_search_gain, precalc=precalc)
    
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("VCCS Mesh: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=True)

if __name__ == "__main__":
    main()