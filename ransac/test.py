import numpy as np
import open3d as o3d
import sys
sys.path.append("../python_utils")
import visu_utils
import graph_utils

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

    point_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)

    inliers = np.array(inliers, dtype=np.uint32)
    remaining = np.arange(n_points, dtype=np.uint32)
    remaining = np.delete(remaining, inliers)
    point_idxs = [inliers, remaining]
    #print(point_idxs)
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=point_idxs, colors=colors, w_co=False)

if __name__ == "__main__":
    main()