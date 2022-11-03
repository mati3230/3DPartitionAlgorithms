import numpy as np
import open3d as o3d
import sys
import argparse
if sys.platform == "win32":
    sys.path.append("./cut-pursuit/build/src/Release")
    sys.path.append("./ply_c/build/Release")
    sys.path.append("../libgeo/build/Release")
else: # linux, macos
    sys.path.append("./cut-pursuit/build/src")
    sys.path.append("./ply_c/build")
    sys.path.append("../libgeo/build")
import libgeo
import libply_c
import libcp
sys.path.append("../python_utils")
import visu_utils
import graph_utils


def unidirectional(graph_nn=None):
    source = graph_nn["source"]
    target = graph_nn["target"]
    distances = graph_nn["distances"]
    uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    mask = libgeo.unidirectional(uni_verts, direct_neigh_idxs, n_edges, target)
    mask = mask.astype(np.bool)
    #print(mask.shape, mask.dtype)
    c_source = np.array(source[mask], copy=True)
    c_target = np.array(target[mask], copy=True)
    c_distances = np.array(distances[mask], copy=True)
    return {
        "source": source,
        "target": target,
        "c_source": c_source,
        "c_target": c_target,
        "distances": distances,
        "c_distances": c_distances
    }


def get_geodesic_knns(target, distances, k, uni_verts=None, direct_neigh_idxs=None, n_edges=None, exclude_closest=True, source=None):
    if uni_verts is None or direct_neigh_idxs is None or n_edges is None:
        if source is None:
            raise Exception("source need to be inserted!")
        uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
    return {
        "source": source_,
        "target": target_,
        "distances": distances_,
    }


def apply_cp(xyz, rgb, k_nn_adj=15, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.07, d_se_max=0, 
        nn_fdir=None, nn_fname=None, mesh=False, uni_verts=None, direct_neigh_idxs=None, n_edges=None,
        source=None, target=None, distances=None, exclude_closest=True, graph_nn=None):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    try:
        graph_nn = graph_utils.load_nn_file(fdir=nn_fdir, fname=nn_fname, verbose=False)
    except:
        if graph_nn is None:
            if mesh:
                if source is None or target is None or distances is None:
                    raise Exception("Missing input arguments")
                graph_nn = get_geodesic_knns(uni_verts=uni_verts, direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges,
                    target=target, distances=distances, k=k_nn_adj, exclude_closest=exclude_closest)
            else:
                graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof, verbose=False)
        graph_nn = unidirectional(
            graph_nn=graph_nn)
        if nn_fdir is not None and nn_fname is not None:
            graph_utils.save_nn_file(fdir=nn_fdir, fname=nn_fname, d_mesh=graph_nn)
    geof = libply_c.compute_geof(xyz, graph_nn["target"], k_nn_adj, False).astype(np.float32)
    features = np.hstack((geof, rgb)).astype("float32")# add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] # increase importance of verticality (heuristic)
    
    verbosity_level = 0.0
    speed = 2.0
    store_bin_labels = 0
    cutoff = 0 
    spatial = 0 
    weight_decay = 1
    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["c_distances"] / np.mean(graph_nn["c_distances"])), dtype = "float32")

    point_idxs, p_vec, stats, duration = libcp.cutpursuit(features, graph_nn["c_source"], graph_nn["c_target"], 
        graph_nn["edge_weight"], reg_strength, cutoff, spatial, weight_decay, verbosity_level, speed, store_bin_labels)
    return point_idxs, p_vec, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="../sn000000.ply", type=str, help="Path to Open3D radable file (ply, obj, ...)")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.file)
    #mesh = o3d.io.read_triangle_mesh("../../../S3DIS_A1_CR1.ply")
    mesh.compute_adjacency_list()
    mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    ########################################################################
    #xyz = np.random.rand(1000,3)
    rgb = np.asarray(mesh.vertex_colors)
    #rgb = np.ones(xyz.shape)
    normals = np.asarray(mesh.vertex_normals)
    P = np.hstack((xyz, rgb))

    P = P.astype(np.float32)
    n_points = P.shape[0]
    normals = normals.astype(np.float32)

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.
    k = 15
    lambda_edge_weight=1
    reg_strength=0.3
    d_se_max=0

    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)

    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=mesh.adjacency_list)
    uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)
    point_idxs, p_vec, duration = apply_cp(xyz=xyz, rgb=rgb, k_nn_adj=k, k_nn_geof=k,
        lambda_edge_weight=lambda_edge_weight, reg_strength=reg_strength, d_se_max=d_se_max, nn_fdir="./", nn_fname="nn", mesh=True,
        uni_verts=uni_verts, direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges,
        source=source, target=target, distances=distances)
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("CP: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
if __name__ == "__main__":
    main()