import numpy as np
import open3d as o3d
import sys
if sys.platform == "win32":
    sys.path.append("./build/Release")
else: # linux
    sys.path.append("./build")
import libgeo

from scipy.spatial import KDTree

sys.path.append("../python_utils")
import visu_utils
import graph_utils


def interp(d, c_far=np.array([0, 1, 0]), c_close=np.array([1, 0, 0])):
    d_ = d / np.max(d)
    d_ = d_.reshape(d_.shape[0], 1)
    c_close = c_close.reshape(1, c_close.shape[0])
    c_far = c_far.reshape(1, c_far.shape[0])
    C = np.matmul(d_, c_close) + np.matmul((1-d_), c_far)
    return C 


def colorize(mesh, C, p_idx, nn_distances, nn_targets):
    C_ = np.array(C, copy=True)
    C_[p_idx] = np.array([0, 1, 0])
    C_interp = interp(d=nn_distances)
    C_[nn_targets] = C_interp
    mesh.vertex_colors = o3d.utility.Vector3dVector(C_)


def main():    
    mesh = o3d.io.read_triangle_mesh("../sn000000.ply")
    mesh.compute_adjacency_list()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    P = np.hstack((xyz, rgb))

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.
    n_points = P.shape[0]
    print("Point cloud has {0} points".format(n_points))

    tree = KDTree(data=xyz)
    
    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=mesh.adjacency_list)
    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)

    C = np.array(rgb, copy=True)

    render_euclid = False
    knn_search = False
    visu_voxel = False
    voxel_resolution = 0.5

    while True:
        #"""
        if knn_search:
            try:
                k = int(input("k>=2 [-1: exit]: "))
                if k == -1:
                    return
                if k <= 1:
                    raise Exception("Please choose k >= 2")
            except Exception as e:
                continue
            p_idxs = visu_utils.pick_points_o3d(pcd=mesh, is_mesh=True)
            try:
                p_idx = p_idxs[0]
            except Exception as e:
                continue
        else:
            try:
                radius = float(input("radius>0 [-1: exit]: "))
                if radius == -1:
                    return
                if radius <= 0:
                    raise Exception("Please choose radius > 0")
            except Exception as e:
                continue
            p_idxs = visu_utils.pick_points_o3d(pcd=mesh, is_mesh=True)
            try:
                p_idx = p_idxs[0]
            except Exception as e:
                continue

        #"""
        """ debugging
        k = 5
        radius = 0.3
        p_idx = 36705
        """
        if knn_search:
            source_, target_, distances_, ok = libgeo.geodesic_knn_single(
                p_idx, uni_verts, direct_neigh_idxs, n_edges, target, distances, k, False)
            if(not ok[0]):
                print("Some error happened while calculating geodesic neighbours")
            """ visualize depth based search
            sources = np.array([p_idx], dtype=np.uint32)
            source_, target_, distances_, ok = libgeo.geodesic_neighbours(
                sources, direct_neigh_idxs, n_edges, target, distances, 3, uni_verts.shape[0], False)
            if(not ok[0]):
                print("Some error happened while calculating geodesic neighbours")
            """
        else:
            source_, target_, distances_, ok = libgeo.geodesic_radiusnn_single(
                p_idx, uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, False)
            if(not ok[0]):
                print("Some error happened while calculating geodesic neighbours")
        colorize(mesh=mesh, C=C, p_idx=p_idx, nn_distances=distances_, nn_targets=target_)
        
        if visu_voxel:
            voxel = o3d.geometry.TriangleMesh.create_box(width=voxel_resolution, height=voxel_resolution, depth=voxel_resolution)
            pos = xyz[p_idx]
            pos -= (voxel_resolution / 2)
            voxel.translate(pos)
            visu_utils.render_o3d(x=[mesh, voxel], w_co=True)
        else:
            visu_utils.render_o3d(x=mesh, w_co=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)

        if render_euclid:
            v = xyz[p_idx]
            distances_, target_ = tree.query(x=v, k=k+1)
            target_ = target_[1:].astype(np.uint32)
            distances_ = distances_[1:]

            colorize(mesh=mesh, C=C, p_idx=p_idx, nn_distances=distances_, nn_targets=target_)
            visu_utils.render_o3d(x=mesh, w_co=True)
            mesh.vertex_colors = o3d.utility.Vector3dVector(C)

if __name__ == "__main__":
    main()