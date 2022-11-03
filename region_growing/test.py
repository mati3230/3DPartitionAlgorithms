import numpy as np
import open3d as o3d
import sys
sys.path.append("../python_utils")
import visu_utils
import graph_utils

from sklearn.neighbors import NearestNeighbors

from region_growing import rg_normals

def main():
    mesh = o3d.io.read_triangle_mesh("../sn000000.ply")
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
    normals = normals.astype(np.float32)

    colors = visu_utils.load_colors(cpath="../python_utils/colors.npz")
    colors = colors/255.
    #print(P[10:])
    #print("")
    #P = P[:1000]
    #"""
    n_points = P.shape[0]
    print("Point cloud has {0} points".format(n_points))

    knn = 15
    nn = NearestNeighbors(n_neighbors=knn+1, algorithm="kd_tree").fit(xyz)
    # get the k nearest neighbours of every point in the point cloud.
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    print(neighbors.shape, normals.shape)

    picked_point_idx = 0
    visited, region = rg_normals(seed_idx=picked_point_idx, search_idx=picked_point_idx,
        visited=[], neighbors=neighbors, normals=normals, angle_thres=10, region=[])

    inliers = np.array(region, dtype=np.uint32)
    remaining = np.arange(n_points, dtype=np.uint32)
    remaining = np.delete(remaining, inliers)

    inliers = inliers[inliers != picked_point_idx]
    
    point_idxs = [np.array([picked_point_idx], dtype=np.uint32), inliers, remaining]
    #print(point_idxs)
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)

if __name__ == "__main__":
    main()