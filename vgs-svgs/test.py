import numpy as np
import open3d as o3d
import sys
if sys.platform == "win32":
    sys.path.append("./build/Release")
    sys.path.append("../libgeo/build/Release")
else: # linux
    sys.path.append("./build")
    sys.path.append("../libgeo/build")
import libvgs
import libgeo
sys.path.append("../python_utils")
import visu_utils
import graph_utils


def main():
    mesh = o3d.io.read_triangle_mesh("../sn000000.ply")
    mesh.compute_adjacency_list()
    mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    normals = np.asarray(mesh.vertex_normals)
    P = np.hstack((xyz, rgb, normals))
    P = P.astype(np.float32)
    normals = normals.astype(np.float32)

    #visu_utils.render_partition_o3d(mesh=mesh, colors=colors, w_co=False)
    #visu_utils.render_o3d(mesh, w_co=False)

    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([cloud])
    """

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.
    #print(P[10:])
    #print("")
    #P = P[:1000]
    #"""
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
    #"""
    """

    voxel_size=0.5
    graph_size=1.5
    sig_p=0.2 # spatial distance
    sig_n=0.2 # angle 
    sig_o=0.2 # stair
    sig_e=0.2 # eigen
    sig_c=0.2 # convex
    sig_w=2.0 # similarity weight
    cut_thred=0.3
    points_min=0
    adjacency_min=0
    voxels_min=0 # minimum number of points in a voxel
    
    point_idxs, p_vec, duration = libvgs.vgs(P, voxel_size=voxel_size, graph_size=graph_size, sig_p=sig_p,
        sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
        points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
    
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("VGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
    """
    
    #"""
    voxel_size=0.08
    graph_size=0.12
    """
    make sig parameters smaller in order to increase the total distance
    $d = sqrt{underset{j}{sum} (d_j / sig_j)^2}$
    """
    sig_p=0.2 # spatial distance
    sig_n=0.01 # angle distance
    sig_o=0.01 # stair distance
    sig_e=0.2 # eigen distance
    sig_c=0.01 # convex distance
    sig_w=2.0 # similarity weight distance
    """
    Increase cut_thred in order to have more merges and a smaller partition
    merge_thred approx 1 - cut_thred
    if similarity(voxel_i, voxel_j) > merge_thred -> merge
    """
    
    """
    cut_thred=0.9
    points_min=0
    adjacency_min=0
    voxels_min=0 # minimum number of points in a voxel
    exclude_closest=False
    source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, graph_size, exclude_closest)
    source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
    source_ = source_.astype(np.uint32)
    target_ = target_.astype(np.uint32)
    distances_ = distances_.astype(np.float32)

    uni_verts_ = uni_verts_.astype(np.uint32)
    direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
    n_edges_ = n_edges_.astype(np.uint32)

    print("Calculated neighbours for radius {0}".format(graph_size))
    point_idxs, p_vec, duration = libvgs.vgs_mesh(P, target_, direct_neigh_idxs_, n_edges_, distances_, normals, voxel_size=voxel_size, 
        sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
        points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min, use_normals=True)
    
    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("VGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
    """

    #"""
    voxel_size=0.2
    seed_size=0.3
    graph_size=0.3
    sig_p=0.2
    sig_n=0.2
    sig_o=0.2
    sig_f=0.2
    sig_e=0.2
    sig_w=1.0
    sig_a=0.0
    sig_b=0.25
    sig_c=0.75
    cut_thred=0.3
    points_min=0
    adjacency_min=0
    voxels_min=0 # minimum size of a cluster
    r_search_gain = 0.5
    """
    source_adj, target_adj, distances_adj, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, 15)

    source_nei, target_nei, distances_nei, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, 30)
    """
    """
    point_idxs, p_vec, duration = libvgs.svgs(P, voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size, 
        sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
        sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
        points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)

    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("SVGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))

    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
    """

    #"""
    precal=False


    vccs_search_radius = r_search_gain * seed_size
    radius = max(graph_size, vccs_search_radius)
    exclude_closest=False
    source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, exclude_closest)
    source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
    source_ = source_.astype(np.uint32)
    target_ = target_.astype(np.uint32)
    uni_verts_ = uni_verts_.astype(np.uint32)
    direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
    n_edges_ = n_edges_.astype(np.uint32)
    distances_ = distances_.astype(np.float32)
    precalc=True
    print("radius search done")

    # memory exception can occur if graph has too many edges!!!
    point_idxs, p_vec, duration = libvgs.svgs_mesh(P, source_, target_, uni_verts_, direct_neigh_idxs_, n_edges_, distances_, 
        voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size,
        sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
        sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
        points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min, r_search_gain=r_search_gain, precalc=precalc)

    outliers = 100 * np.sum(p_vec == -1) / n_points
    print("SVGS: {0} superpoints, {1:.2f}% outliers, {2:.2f} seconds".format(len(point_idxs), outliers, duration))
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)
    #"""
if __name__ == "__main__":
    main()