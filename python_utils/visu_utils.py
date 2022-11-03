import numpy as np
import open3d as o3d


def load_colors(cpath):
    #print("Load colors...")
    data = np.load(cpath)
    colors = data["colors"]
    #print("Done")
    return colors


def render_partition_o3d(mesh, sp_idxs, colors, w_co=False):
    vertices = np.asarray(mesh.vertices)
    n_vert = vertices.shape[0]
    rgb = np.zeros((n_vert, 3), dtype=np.float32)
    for i in range(len(sp_idxs)):
        sp = sp_idxs[i]
        color = colors[i]
        #if i == 0:
        #    color = np.array([1, 0, 0])
        rgb[sp] = color
    pmesh = o3d.geometry.TriangleMesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles)
    pmesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    render_o3d(pmesh, w_co=w_co)
    return pmesh


def coordinate_system():
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def render_o3d(x, w_co=False):
    if type(x) == list:
        if w_co:
            x.append(coordinate_system())
        o3d.visualization.draw_geometries(x)
        if w_co:
            x.pop(-1)
        return
    if w_co:
        o3d.visualization.draw_geometries([x, coordinate_system()])
    else:
        o3d.visualization.draw_geometries([x])


def pick_points_o3d(pcd, is_mesh=False):
    print("")
    print("1) Pick points by using [shift + left click]")
    print("Press [shift + right click] to undo a selection")
    print("2) After picking, press 'Q' to close the window")
    if is_mesh:
        vis = o3d.visualization.VisualizerWithVertexSelection()
    else:
        vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    pp = vis.get_picked_points()
    picked_points = len(pp)*[None]
    for i in range(len(pp)):
        picked_points[i] = pp[i].index
    return picked_points