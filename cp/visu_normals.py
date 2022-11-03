import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import sys
if sys.platform == "win32":
    sys.path.append("./ply_c/build/Release")
elif sys.platform.startswith("linux"): # linux
    sys.path.append("./ply_c/build")
else: # apple
    sys.path.append("./build")
import libply_c
sys.path.append("../python_utils")
import visu_utils


def render_point_cloud(xyz, normals):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    visu_utils.render_o3d(x=cloud, w_co=False)


def compute_normals_cloud(xyz, n_points, k=5):
    tree = KDTree(xyz)
    distances, neighbors = tree.query(xyz, k=k+1)
    neighbors = neighbors[:, 1:]
    target = np.reshape(neighbors, (n_points * k, ))
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    target = target.astype(np.uint32)
    
    geof = libply_c.compute_all_geof(xyz, target, k, False)
    normals_cloud = geof[:, -3:]

    normal_norm = np.linalg.norm(normals_cloud, axis=1)
    normal_norm = np.hstack((normal_norm[:, None], normal_norm[:, None], normal_norm[:, None]))
    normals_cloud = normals_cloud / normal_norm
    return normals_cloud


def main2():
    xyz = np.random.rand(1000, 3)
    n_points = xyz.shape[0]

    def visu_noise(noise=0, k=5):
        xyz[:, 2] = noise * np.random.rand(1000, )
        normals_cloud = compute_normals_cloud(xyz=xyz, n_points=n_points, k=k)
        render_point_cloud(xyz=xyz, normals=normals_cloud)

    visu_noise(noise=0)
    visu_noise(noise=0.01)
    visu_noise(noise=0.1)

    visu_noise(noise=0, k=15)
    visu_noise(noise=0.01, k=15)
    visu_noise(noise=0.1, k=15)


def main():
    mesh = o3d.io.read_triangle_mesh("../sn000000.ply")
    mesh.compute_adjacency_list()
    mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    xyz = xyz[:1000]
    normals_mesh = np.asarray(mesh.vertex_normals)
    n_points = xyz.shape[0]
    normals_mesh = normals_mesh.astype(np.float32)
    normals_mesh = normals_mesh[:1000]
    render_point_cloud(xyz=xyz, normals=normals_mesh)
    
    normals_cloud = compute_normals_cloud(xyz=xyz, n_points=n_points, k=5)
    
    render_point_cloud(xyz=xyz, normals=normals_cloud)

if __name__ == "__main__":
    main2()