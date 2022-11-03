import numpy as np
import pptk
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


def render_pptk_feat(P, point_size=-1, v=None, feats=None):
    if P is None:
        return None
    if feats is None:
        return None
    xyz = P[:, :3]
    rgb = np.array(P[:, 3:], copy=True)
    max_c = np.max(rgb, axis=0)
    max_c = max_c[max_c > 1]
    if max_c.shape[0] > 0:
        #print("Normalize colors.")
        rgb /= 255.
    if v is None:
        v = pptk.viewer(xyz)
    else:
        #persp = get_perspective(viewer=v)
        v.clear()
        v.load(xyz)
    if point_size > 0:
        v.set(point_size=point_size)
    #feats.append(rgb)
    v.attributes(rgb, *feats)
    print("Press Return in the 3D windows to continue.")
    #set_perspective(viewer=v, p=persp)
    v.wait()
    return v


def main():
    mesh = o3d.io.read_triangle_mesh("../sn000000.ply")
    mesh.compute_adjacency_list()
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    n_points = xyz.shape[0]
    #print(n_points)

    k = 5
    tree = KDTree(xyz)
    distances, neighbors = tree.query(xyz, k=k+1)
    neighbors = neighbors[:, 1:]
    target = np.reshape(neighbors, (n_points * k, ))
    
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    target = target.astype(np.uint32)
    
    geof = libply_c.compute_all_geof(xyz, target, k, False)
    #print(geof.shape)
    feats = [geof[:, 0], geof[:, 1], geof[:, 2]]
    P = np.hstack((xyz, rgb))
    v = render_pptk_feat(P=P, feats=feats, point_size=0.01)
    v.close()


if __name__ == "__main__":
    main()