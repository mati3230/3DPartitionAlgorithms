import sys
if sys.platform == "win32":
    sys.path.append("./cp/ply_c/build/Release")
    sys.path.append("./cp/cut-pursuit/build/src/Release")
    sys.path.append("./p-linkage/build/Release")
    sys.path.append("./vgs-svgs/build/Release")
    sys.path.append("./vccs/build/Release")
    sys.path.append("./libgeo/build/Release")
else:
    sys.path.append("./cp/ply_c/build")
    sys.path.append("./cp/cut-pursuit/build/src")
    sys.path.append("./p-linkage/build")
    sys.path.append("./vgs-svgs/build")
    sys.path.append("./vccs/build")
    sys.path.append("./libgeo/build")
import libply_c
import libcp
import libplink
import libvgs
import libvccs
import libgeo

import argparse
import open3d as o3d
import numpy as np
import os
from random import shuffle, seed
from tqdm import tqdm
import pandas as pd
import time


from python_utils.scannet_utils import get_scenes, \
    get_ground_truth, get_scannet_dir

from python_utils.graph_utils import compute_graph_nn_2,\
    save_nn_file,\
    load_nn_file,\
    refine,\
    get_edges,\
    sort_graph

from python_utils.partition.partition import Partition
from python_utils.partition.density_utils import densities_np

#from python_utils.visu_utils import render_partition_o3d, load_colors


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def write_csv(filedir, filename, csv):
    if filedir[-1] != "/":
        filedir += "/"
    if not filename.endswith(".csv"):
        filename += ".csv"
    out_filename = filedir + filename
    file = open(out_filename, "w")
    file.write(csv)
    file.close()


def save_csv(res, csv_dir, csv_name, data_header):
    # write results of the cut pursuit calculations as csv
    csv = ""
    for header in data_header:
        csv += header + ","
    csv = csv[:-1] + "\n"

    for tup in res:
        if tup is None:
            continue
        for elem in tup:
            if type(elem) == list:
                csv += str(elem)[1:-1].replace(" ", "") + ","
            else:
                csv += str(elem) + ","
        csv = csv[:-1] + "\n"
    write_csv(filedir=csv_dir, filename=csv_name, csv=csv)


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


def reduce_knns(source, target, distances, k_new):
    _ , uni_index, uni_counts = np.unique(source, return_index=True, return_counts=True)
    k_old = uni_counts[0]
    if k_new >= k_old:
        raise Exception("Error k_new too large")
    n_verts = uni_counts.shape[0]
    n_edges_new = n_verts * k_new
    source_ = np.zeros((n_edges_new, ), dtype=np.uint32)
    target_ = np.zeros((n_edges_new, ), dtype=np.uint32)
    distances_ = np.zeros((n_edges_new, ), dtype=np.float32)
    for i in range(uni_counts.shape[0]):
        start = uni_index[i]
        stop = start + uni_counts[i]
        start_ = i * k_new
        stop_ = i * k_new + k_new
        source_[start_:stop_] = source[start:stop][:k_new]
        target_[start_:stop_] = target[start:stop][:k_new]
        distances_[start_:stop_] = distances[start:stop][:k_new]
    return source_, target_, distances_


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


def ooa(par_v_gt, par_v, partition_gt=None, sortation=None):
    precalc = partition_gt is not None and sortation is not None
    if precalc:
        par_v = par_v[sortation]
        partition_A = Partition(partition=par_v)
    else:
        sortation = np.argsort(par_v_gt)
        par_v_gt = par_v_gt[sortation]
        par_v = par_v[sortation]
        
        ugt, ugt_idxs, ugt_counts = np.unique(par_v_gt, return_index=True, return_counts=True)

        partition_gt = Partition(partition=par_v_gt, uni=ugt, idxs=ugt_idxs, counts=ugt_counts)
        partition_A = Partition(partition=par_v)
        
    max_density = partition_gt.partition.shape[0]
    ooa_A, _ = partition_gt.overall_obj_acc(max_density=max_density, partition_B=partition_A, density_function=densities_np)
    return ooa_A, partition_gt, sortation


def apply_cp(xyz, rgb, k_nn_adj=15, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.07, d_se_max=0, 
        nn_fdir=None, nn_fname=None, mesh=False, uni_verts=None, direct_neigh_idxs=None, n_edges=None,
        source=None, target=None, distances=None, exclude_closest=True, graph_nn=None):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    try:
        graph_nn = load_nn_file(fdir=nn_fdir, fname=nn_fname, verbose=False)
    except:
        if graph_nn is None:
            if mesh:
                if source is None or target is None or distances is None:
                    raise Exception("Missing input arguments")
                graph_nn = get_geodesic_knns(uni_verts=uni_verts, direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges,
                    target=target, distances=distances, k=k_nn_adj, exclude_closest=exclude_closest)
            else:
                graph_nn, target_fea = graph_utils.compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof, verbose=False)
        graph_nn = unidirectional(
            graph_nn=graph_nn)
        if nn_fdir is not None and nn_fname is not None:
            save_nn_file(fdir=nn_fdir, fname=nn_fname, d_mesh=graph_nn)
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


def get_csv_header(algorithms=["cp", "vccs", "plink", "vgs", "svgs"]):
    header = [
        "ID",
        "Name",
        "|V|"
    ]
    algo_stats = [
        "OOA",
        "|S|",
        "Duration"
    ]

    for algo in algorithms:
        for algs in algo_stats:
            header.append(algs + "_" + algo)
    return header


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--csv_dir", default="./csvs", type=str, help="Directory where we save the csv.")
    #parser.add_argument("--csv_name", default="baseline", type=str, help="filename of the csv.")
    #parser.add_argument("--n_proc", default=1, type=int, help="Number of processes that will be used.")
    parser.add_argument("--n_scenes", default=100, type=int, help="Number of scenes that will be used for the evaluation.")
    parser.add_argument("--pkg_size", default=5, type=int, help="Number of packages to save a csv")
    #parser.add_argument("--offset", default=0, type=int, help="Offset for the name of the csv file.")
    parser.add_argument("--verbose", default=False, type=bool, help="Enable/disable prints.")
    parser.add_argument("--mesh", default=False, type=bool, help="Enable/disable usage of mesh features such as normals and triangles.")
    
    parser.add_argument("--scenes", default="", type=str, help="Path to file where the scenes will be loaded")
    parser.add_argument("--exclude", default="", type=str, help="Exclude algorithms as comma seperated list")
    parser.add_argument("--n_points", default=-1, type=int, help="Max number of points")
    args = parser.parse_args()
    csv_name = "baseline"
    csv_dir = "./csvs_baseline"
    if args.mesh:
        csv_name = "mesh"
        csv_dir = "./csvs_mesh"
    mkdir(csv_dir)
    seed(42)

    #colors = load_colors(cpath="./python_utils/colors.npz")
    #colors = colors/255.

    header = get_csv_header()
    if args.mesh:
        header = get_csv_header(algorithms=["cp", "vccs", "plink", "svgs"])

    algorithms = {
        "cp": {
            "knn": 15,
            "reg_strength": 0.07,
            "exclude_closest": True
        },
        "plinkage": {
            "angle": 90,
            "k": 30,
            "min_cluster_size": 10,
            "angle_dev": 10.0,
            "exclude_closest": True,
            "use_normals": False
        },
        "vccs": {
            "voxel_resolution": 0.05,
            "seed_resolution": 0.1,
            "color_importance": 0.2,
            "spatial_importance": 0.2,
            "normal_importance": 0.6,
            "refinementIter": 3,
            "r_search_gain": 0.5
        },
        "vgs": {
            "voxel_size": 0.15,
            "graph_size": 0.3,
            "sig_p": 0.2,
            "sig_n": 0.2,
            "sig_o": 0.2,
            "sig_e": 0.2,
            "sig_c": 0.2,
            "sig_w": 2.0,
            "cut_thred": 0.3,
            "points_min": 0,
            "adjacency_min": 0,
            "voxels_min": 0
        },
        "svgs": {
            "voxel_size": 0.05,
            "seed_size": 0.1,
            "graph_size": 0.1,
            "sig_p": 0.2,
            "sig_n": 0.2,
            "sig_o": 0.2,
            "sig_f": 0.2,
            "sig_e": 0.2,
            "sig_w": 1.0,
            "sig_a": 0.0,
            "sig_b": 0.25,
            "sig_c": 0.75,
            "cut_thred": 0.3,
            "points_min": 0,
            "adjacency_min": 0,
            "voxels_min": 0
        }
    }

    exclude = []
    if args.exclude != "":
        exclude = args.exclude.replace(" ", "")
        exclude = exclude.split(",")
    print("The following algorithms will be excluded:", exclude)

    if os.path.isfile(args.scenes):
        scenes = pd.read_csv(args.scenes, sep=",")
        scenes = scenes["scenes"].to_list()
        scannet_dir = get_scannet_dir()
        print("Will process {0} scenes from '{1}' directory".format(len(scenes), scannet_dir))
    else:
        blacklist = []
        scenes, _, scannet_dir = get_scenes(blacklist=blacklist)


    scenes_ids = list(zip(scenes, list(range(len(scenes)))))


    shuffle(scenes_ids)
    scenes_ids = scenes_ids[:args.n_scenes]
    n_scenes = len(scenes_ids)
    
    verbose = args.verbose
    data = []
    desc = "Baseline"
    skipped = 0
    if args.mesh:
        desc = "Mesh"
    processed = 0
    #remain = n_scenes % args.pkg_size
    #print(remain, n_scenes)
    #i_start = n_scenes - remain
    for i in tqdm(range(n_scenes), desc=desc, disable=verbose):
        (scene_name, scene_id) = scenes_ids[i]
        if verbose:
            print("process", i, scene_name, scene_id)
        #mesh = o3d.io.read_triangle_mesh("/home/username/Projects/Datasets/sn000000.ply")
        mesh, p_vec_gt, file_gt = get_ground_truth(scannet_dir=scannet_dir, scene=scene_name)
        mesh.compute_adjacency_list()
        xyz = np.asarray(mesh.vertices)
        xyz = xyz.astype(np.float32)
        rgb = np.asarray(mesh.vertex_colors)
        rgb = rgb.astype(np.float32)

        source = None
        target = None
        distances = None
        uni_verts = None
        direct_neigh_idxs = None
        n_edges = None
        
        partition_gt = None

        ooa_cp=0; size_cp=0; dur_cp=0
        ooa_vccs=0; size_vccs=0; dur_vccs=0
        ooa_plink=0; size_plink=0; dur_plink=0
        ooa_vgs=0; size_vgs=0; dur_vgs=0
        ooa_svgs=0; size_svgs=0; dur_svgs=0


        if args.mesh:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            P = np.hstack((xyz, rgb, normals))
            source, target, distances, uni_verts, direct_neigh_idxs, n_edges = get_edges(
                mesh_vertices=xyz, adj_list=mesh.adjacency_list)
            if uni_verts.shape[0] != xyz.shape[0]:
                if verbose:
                    print("Error: There are points which are target only, uni_verts: {0}, |P|: {1}".format(uni_verts.shape[0], xyz.shape[0]))
                skipped += 1
                continue
            source = source.astype(np.uint32)
            target = target.astype(np.uint32)
            uni_verts = uni_verts.astype(np.uint32)
            direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
            n_edges = n_edges.astype(np.uint32)
            distances = distances.astype(np.float32)

            calc_plink_knn = algorithms["plinkage"]["k"] >= algorithms["cp"]["knn"]
            max_r = max(algorithms["svgs"]["graph_size"], (algorithms["vccs"]["r_search_gain"] * algorithms["vccs"]["seed_resolution"]))
            if calc_plink_knn:
                max_knn = algorithms["plinkage"]["k"]
            else:
                max_knn = algorithms["cp"]["knn"]
            same_knn = algorithms["plinkage"]["exclude_closest"] == algorithms["cp"]["exclude_closest"]
        else:
            P = np.hstack((xyz, rgb))

        n_points = P.shape[0]
        if verbose:
            print("Point cloud has {0} points".format(n_points))

        if args.n_points != -1:
            if n_points > args.n_points:
                continue
        
        #"""
        #######################################PLINK######################################
        angle = algorithms["plinkage"]["angle"]
        k = algorithms["plinkage"]["k"]
        min_cluster_size = algorithms["plinkage"]["min_cluster_size"]
        angle_dev = algorithms["plinkage"]["angle_dev"]
        use_normals = algorithms["plinkage"]["use_normals"]
        if args.mesh:
            exclude_closest = algorithms["plinkage"]["exclude_closest"]
            if calc_plink_knn:
                #print("geo knn")
                if args.verbose:
                    print("Apply geodesic KNN")
                t1 = time.time()
                source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
                t2 = time.time()
                if args.verbose:
                    print("Calculate geodesic KNN in {0:.2f} seconds".format(t2-t1))
                # TODO check if sizes are appropriate
                target_size = uni_verts.shape[0] * k
                if source_.shape[0] != target_size:
                    print("Did not found enough neighbours - have {0} and expected {1}".format(source_.shape[0], target_size))
                    skipped += 1
                    continue
                #print("geo plink")
                if "plink" not in exclude:
                    if args.verbose:
                        print("Apply P-Linkage")
                    point_idxs, p_vec, duration = libplink.plinkage_geo(P, target_, normals, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev, use_normals=use_normals)
                #print("done")
            else:
                if same_knn:
                    if args.verbose:
                        print("Apply geodesic KNN")
                    t1 = time.time()
                    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, max_knn, exclude_closest)
                    t2 = time.time()
                    if args.verbose:
                        print("Calculate geodesic KNN in {0:.2f} seconds".format(t2-t1))
                    target_size = uni_verts.shape[0] * max_knn
                    if source_.shape[0] != target_size:
                        print("Did not found enough neighbours - have {0} and expected {1}".format(source_.shape[0], target_size))
                        skipped += 1
                        continue
                    source__, target__, distances__ = reduce_knns(source=source_, target=target_, distances=distances_, k_new=k)
                    if "plink" not in exclude:
                        if args.verbose:
                            print("Apply P-Linkage")
                        point_idxs, p_vec, duration = libplink.plinkage_geo(P, target__, normals, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev, use_normals=use_normals)
                else:
                    if args.verbose:
                        print("Apply geodesic KNN")
                    t1 = time.time()
                    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
                    t2 = time.time()
                    if args.verbose:
                        print("Calculate geodesic KNN in {0:.2f} seconds".format(t2-t1))
                    target_size = uni_verts.shape[0] * k
                    if source_.shape[0] != target_size:
                        print("Did not found enough neighbours - have {0} and expected {1}".format(source_.shape[0], target_size))
                        skipped += 1
                        continue

                    if "plink" not in exclude:
                        if args.verbose:
                            print("Apply P-Linkage")
                        point_idxs, p_vec, duration = libplink.plinkage_geo(P, target_, normals, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev, use_normals=use_normals)
        else:
            if "plink" not in exclude:
                point_idxs, p_vec, duration = libplink.plinkage(P, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev)
        if "plink" not in exclude:
            point_idxs, p_vec = refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=mesh.adjacency_list)
            #ooa_plink, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec, partition_gt=partition_gt, sortation=sortation)
            ooa_plink, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec)
            size_plink = len(point_idxs)
            dur_plink = duration
            if verbose:
                outliers = 100 * np.sum(p_vec == -1) / n_points
                print("P-Linkage: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds, OOA: {3:.4f}".format(len(point_idxs), outliers, duration, ooa_plink))

        #"""
        ########################################CP########################################
        if "cp" not in exclude:
            graph_nn = None
            if args.mesh:
                if calc_plink_knn:
                    source__, target__, distances__ = reduce_knns(source=source_, target=target_, distances=distances_, k_new=algorithms["cp"]["knn"])
                    graph_nn = {
                        "source": source__,
                        "target": target__,
                        "distances": distances__
                    }
                else:
                    if same_knn:
                        graph_nn = {
                            "source": source_,
                            "target": target_,
                            "distances": distances_
                        }
            if args.verbose:
                print("Apply CP")

            point_idxs, p_vec, duration = apply_cp(xyz=P[:, :3], rgb=P[:, 3:6], k_nn_adj=algorithms["cp"]["knn"], 
                reg_strength=algorithms["cp"]["reg_strength"], mesh=args.mesh, uni_verts=uni_verts, 
                direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges, source=source, target=target,
                distances=distances, exclude_closest=algorithms["cp"]["exclude_closest"], graph_nn=graph_nn)
            
            if partition_gt is None:
                ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec)
            else:
                ooa_cp, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec, partition_gt=partition_gt, sortation=sortation)
            size_cp = len(point_idxs)
            dur_cp = duration
            if verbose:
                outliers = 100 * np.sum(p_vec == -1) / n_points
                print("CP: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds, OOA: {3:.4f}".format(len(point_idxs), outliers, duration, ooa_cp))
        
        #######################################VCCS#######################################
        if args.mesh:
            if ("vccs" not in exclude) or ("svgs" not in exclude):
                if args.verbose:
                    print("Apply geodesic RNN")
                t1 = time.time()
                source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, max_r, False)
                source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = sort_graph(source=source_, target=target_, distances=distances_)
                t2 = time.time()
                if args.verbose:
                    print("Geodesic RNN in {0:.2f} seconds".format(t2-t1))
                source_ = source_.astype(np.uint32)
                target_ = target_.astype(np.uint32)
                distances_ = distances_.astype(np.float32)
                uni_verts_ = uni_verts_.astype(np.uint32)
                direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
                n_edges_ = n_edges_.astype(np.uint32)
        if "vccs" not in exclude:
            voxel_resolution=algorithms["vccs"]["voxel_resolution"]
            seed_resolution=algorithms["vccs"]["seed_resolution"]
            color_importance=algorithms["vccs"]["color_importance"]
            spatial_importance=algorithms["vccs"]["spatial_importance"]
            normal_importance=algorithms["vccs"]["normal_importance"]
            refinementIter=algorithms["vccs"]["refinementIter"]
            
            if args.verbose:
                print("Apply VCCS")
            if args.mesh:    
                r_search_gain = algorithms["vccs"]["r_search_gain"]

                point_idxs, p_vec, duration = libvccs.vccs_mesh(P, uni_verts_, direct_neigh_idxs_, n_edges_, source_, target_, distances_, 
                    voxel_resolution=voxel_resolution, seed_resolution=seed_resolution, color_importance=color_importance,
                    spatial_importance=spatial_importance, normal_importance=normal_importance, refinementIter=refinementIter,
                    r_search_gain=r_search_gain, precalc=True)
            else:
                point_idxs, p_vec, duration = libvccs.vccs(P, voxel_resolution=voxel_resolution, seed_resolution=seed_resolution,
                    color_importance=color_importance, spatial_importance=spatial_importance, normal_importance=normal_importance,
                    refinementIter=refinementIter)
            if partition_gt is None:
                ooa_vccs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec)
            else:
                ooa_vccs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec, partition_gt=partition_gt, sortation=sortation)
            size_vccs = len(point_idxs)
            dur_vccs = duration
            if verbose:
                outliers = 100 * np.sum(p_vec == -1) / n_points
                print("VCCS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds, OOA: {3:.4f}".format(len(point_idxs), outliers, duration, ooa_vccs))
        #"""

        #######################################VGS########################################
        if "vgs" not in exclude:
            if not args.mesh:
                if args.verbose:
                    print("Apply VGS")
                voxel_size=algorithms["vgs"]["voxel_size"]
                graph_size=algorithms["vgs"]["graph_size"]
                sig_p=algorithms["vgs"]["sig_p"]
                sig_n=algorithms["vgs"]["sig_n"]
                sig_o=algorithms["vgs"]["sig_o"]
                sig_e=algorithms["vgs"]["sig_e"]
                sig_c=algorithms["vgs"]["sig_c"]
                sig_w=algorithms["vgs"]["sig_w"]
                cut_thred=algorithms["vgs"]["cut_thred"]
                points_min=algorithms["vgs"]["points_min"]
                adjacency_min=algorithms["vgs"]["adjacency_min"]
                voxels_min=algorithms["vgs"]["voxels_min"]
                point_idxs, p_vec, duration = libvgs.vgs(P, voxel_size=voxel_size, graph_size=graph_size, sig_p=sig_p,
                    sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
                    points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
                point_idxs, p_vec = refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=mesh.adjacency_list)
                if partition_gt is None:
                    ooa_vgs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec)
                else:
                    ooa_vgs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec, partition_gt=partition_gt, sortation=sortation)
                size_vgs = len(point_idxs)
                dur_vgs = duration
                if verbose:
                    outliers = 100 * np.sum(p_vec == -1) / n_points
                    print("VGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds, OOA: {3:.4f}".format(len(point_idxs), outliers, duration, ooa_vgs))

        #######################################SVGS#######################################
        if "svgs" not in exclude:
            if args.verbose:
                print("Apply SVGS")
            voxel_size=algorithms["svgs"]["voxel_size"]
            seed_size=algorithms["svgs"]["seed_size"]
            graph_size=algorithms["svgs"]["graph_size"]
            sig_p=algorithms["svgs"]["sig_p"]
            sig_n=algorithms["svgs"]["sig_n"]
            sig_o=algorithms["svgs"]["sig_o"]
            sig_f=algorithms["svgs"]["sig_f"]
            sig_e=algorithms["svgs"]["sig_e"]
            sig_w=algorithms["svgs"]["sig_w"]
            sig_a=algorithms["svgs"]["sig_a"]
            sig_b=algorithms["svgs"]["sig_b"]
            sig_c=algorithms["svgs"]["sig_c"]
            cut_thred=algorithms["svgs"]["cut_thred"]
            points_min=algorithms["svgs"]["points_min"]
            adjacency_min=algorithms["svgs"]["adjacency_min"]
            voxels_min=algorithms["svgs"]["voxels_min"]
            if args.mesh:
                point_idxs, p_vec, duration = libvgs.svgs_mesh(P, source_, target_, uni_verts_, direct_neigh_idxs_, n_edges_, distances_, 
                    voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size,
                    sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
                    sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
                    points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min,
                    r_search_gain=r_search_gain, precalc=True)
            else:
                point_idxs, p_vec, duration = libvgs.svgs(P, voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size, 
                    sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
                    sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
                    points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
            point_idxs, p_vec = refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=mesh.adjacency_list)
            if partition_gt is None:
                ooa_svgs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec)
            else:
                ooa_svgs, partition_gt, sortation = ooa(par_v_gt=p_vec_gt, par_v=p_vec, partition_gt=partition_gt, sortation=sortation)
            size_svgs = len(point_idxs)
            dur_svgs = duration
            if verbose:
                outliers = 100 * np.sum(p_vec == -1) / n_points
                print("SVGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds, OOA: {3:.4f}".format(len(point_idxs), outliers, duration, ooa_svgs))
        #"""
        if args.mesh:
            data.append(
                (
                    scene_id, scene_name, n_points,
                    ooa_cp, size_cp, dur_cp,
                    ooa_vccs, size_vccs, dur_vccs,
                    ooa_plink, size_plink, dur_plink,
                    #ooa_vgs, size_vgs, dur_vgs,
                    ooa_svgs, size_svgs, dur_svgs
                )
            )
        else:
            data.append(
                (
                    scene_id, scene_name, n_points,
                    ooa_cp, size_cp, dur_cp,
                    ooa_vccs, size_vccs, dur_vccs,
                    ooa_plink, size_plink, dur_plink,
                    ooa_vgs, size_vgs, dur_vgs,
                    ooa_svgs, size_svgs, dur_svgs
                )
            )
        if processed % args.pkg_size == 0 or i == (n_scenes - 1):
            save_csv(res=data, csv_dir=csv_dir, csv_name=csv_name + "_" + str(i), data_header=header)
            data = []
        processed += 1
    print("{0} scenes skipped".format(skipped))


if __name__ == "__main__":
    main()