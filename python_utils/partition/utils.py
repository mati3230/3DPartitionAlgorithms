import numpy as np
import open3d as o3d
import os


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def coordinate_system():
    """Returns a coordinate system.

    Returns
    -------
    o3d.geometry.LineSet
        The lines of a coordinate system that are colored in red, green
        and blue.

    """
    line_set = o3d.geometry.LineSet()
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3]]).astype(int)
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def render_all_segments(P, partition_vec, animate=True):
    """Renders every superpoint.

    Parameters
    ----------
    P : np.ndarray
        Nx3 or Nx6 matrix. N is the number of points. A point should have at
        least 3 spatial coordinates and can have optionally 3 color values.
    partition_vec : np.ndarray
        The partition of the point cloud.
    animate : boolean
        If True, the point cloud will be rotated with x_speed and y_speed. A
        simulation of dragging the mouse in standard rendering mode is
        simulated.
    """
    uni_segs, uni_idxs, uni_counts = np.unique(
        partition_vec, return_index=True, return_counts=True)
    for i in range(uni_segs.shape[0]):
        idx = uni_idxs[i]
        count = uni_counts[i]
        P_ = P[idx:idx+count, :3]
        partition_vec_ = np.zeros((count, ), dtype=np.int32)
        partition_vec_[:] = uni_segs[i]
        render_point_cloud(
            P=P_, partition_vec=partition_vec_, animate=animate)


def render_point_cloud(
        P, partition_vec=None, classification=None, gt_partition=None, animate=False, x_speed=2.5, y_speed=0.0, width=1920, left=0, colors_dir=None):
    """Displays a point cloud.

    Parameters
    ----------
    P : np.ndarray
        Nx3 or Nx6 matrix. N is the number of points. A point should have at
        least 3 spatial coordinates and can have optionally 3 color values.
    partition_vec : np.ndarray
        The partition of the point cloud.
    classification : np.ndarray
        Match classification matrix.
    gt_partition : np.ndarray
        Ground truth partition.
    animate : boolean
        If True, the point cloud will be rotated with x_speed and y_speed. A
        simulation of dragging the mouse in standard rendering mode is
        simulated.
    x_speed : float
        Used if point cloud will be animated. Strength if the horizontal mouse
        drag.
    y_speed : float
        Used if point cloud will be animated. Strength if the vertical mouse
        drag.
    width : int
        Set the width of the open3D plot
    left : int
        How much should the open3D plot be dragged to the right. 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    if partition_vec is not None:
        col_mat = np.zeros((P.shape[0], 3))
        colors_path = "colors.npz"
        if colors_dir is not None:
            colors_path = colors_dir + "/" + colors_path
        data = np.load(colors_path)
        colors = data["colors"]
        superpoints = np.unique(partition_vec)
        n_superpoints = superpoints.shape[0]
        if classification is not None and gt_partition is not None:
            segment_val_to_color_idx = {}
            if n_superpoints != classification.shape[0]:
                raise Exception("Mismatch of number of superpoints in the classification and the partition.")
            for i in range(n_superpoints):
                superpoint_value = superpoints[i]
                idxs = np.where(classification[i, :] != 0)[0]
                if idxs.shape[0] > 1:
                    raise Exception("One-to-many assignment in classification")
                if idxs.shape[0] == 1:
                    color_idx = idxs[0]
                    segment_val_to_color_idx[superpoint_value] = color_idx
            color_idx_offset = classification.shape[1]
            for i in range(n_superpoints):
                superpoint_value = superpoints[i]
                idx = np.where(partition_vec == superpoint_value)[0]
                col_idx = color_idx_offset + i
                if superpoint_value in segment_val_to_color_idx:
                    col_idx = segment_val_to_color_idx[superpoint_value]
                col_mat[idx, :] = colors[col_idx, :] / 255
        else:
            for i in range(n_superpoints):
                superpoint_value = superpoints[i]
                idx = np.where(partition_vec == superpoint_value)[0]
                color = colors[i, :] / 255
                col_mat[idx, :] = color
        pcd.colors = o3d.utility.Vector3dVector(col_mat)
    else:
        try:
            # print(P[:5, 3:6] / 255.0)
            pcd.colors = o3d.utility.Vector3dVector(P[:, 3:6] / 255.0)
        except Exception as e:
            print(e)
    render_pc(pcd=pcd, animate=animate, x_speed=x_speed, y_speed=y_speed, width=width, left=left)
    return pcd


def render_unify_suggestion(
        P,
        main_sp_P_idxs,
        neighbour_sp_P_idxs,
        main_sp_color=np.array([1, 0, 0]),
        neighbour_sp_color=np.array([0, 0, 1])):
    """Visualize a superpoint and a neighbour superpoint.

    Parameters
    ----------
    P : np.ndarray
        The point cloud as NxM matrix. N represents the number of points. A
        point should have at least 3 spatial coordinates.
    main_sp_P_idxs : np.ndarray
        Indices of the superpoint.
    neighbour_sp_P_idxs : np.ndarray
        Indices of the neighbour superpoint.
    main_sp_color : np.ndarray
        Color which will be applied to the main superpoint.
    neighbour_sp_color : np.ndarray
        Color which will be applied to the neighbour superpoints.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    col_mat = np.zeros((P.shape[0], 3))
    col_mat[main_sp_P_idxs] = main_sp_color
    col_mat[neighbour_sp_P_idxs] = neighbour_sp_color
    pcd.colors = o3d.utility.Vector3dVector(col_mat)
    render_pc(pcd=pcd)


def get_P_idxs(partition, seg_idx):
    """Get the point indices of a certain element of a partiton.

    Parameters
    ----------
    partition : Partition
        A partition object.
    seg_idx : int
        Index of the superpoint/subset.

    Returns
    -------
    np.ndarray
        Indices of the points that are assdigned with the seg_idx.

    """
    val = partition.uni[seg_idx]
    P_idxs = np.where(partition.partition == val)[0]
    return P_idxs


def vary_object_color(col_mat, densities, S, O, partition, P_idxs, color):
    """An object is colored with a certain input color. The points of a
    superpoint where the density is >0 will be colored with a different brightness
    of that color. The partition of the superpoints should not be the ground
    truth partition.

    Parameters
    ----------
    col_mat : np.ndarray
        Color matrix of the points.
    densities : np.ndarray
        densities matrix of the match classification.
    S : int
        Index of a superpoint that should not be considered in this
        colorization process.
    O : int
        Object index of which the brightness should be varied.
    partition : Partition
        Superpoint partition.
    P_idxs : np.ndarray
        Point indices of the object.
    color : np.ndarray
        Specific color that should be varied by changing the brightness.
    """
    sp_in_O = np.where(densities[:, O] > 0)[0]
    for S_j in sp_in_O:
        if S_j == S:
            continue
        P_S_j_idxs = get_P_idxs(partition=partition, seg_idx=S_j)
        intersection = np.intersect1d(P_idxs, P_S_j_idxs)
        f = S_j / partition.n_uni
        col_mat[intersection, :] = f * color


def render_pc(pcd, animate=False, x_speed=2.5, y_speed=0.0, width=1920, left=0):
    """Render a point cloud.

    Parameters
    ----------
    pcd : o3d.PointCloud
        Open3D point cloud.
    animate : boolean
        If True, the point cloud will be rotated with x_speed and y_speed. A
        simulation of dragging the mouse in standard rendering mode is
        simulated.
    x_speed : float
        Used if point cloud will be animated. Strength if the horizontal mouse
        drag.
    y_speed : float
        Used if point cloud will be animated. Strength if the vertical mouse
        drag.
    width : int
        Set the width of the open3D plot
    left : int
        How much should the open3D plot be dragged to the right. 

    """
    if animate:
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(x_speed, y_speed)
            return False
        o3d.visualization.draw_geometries_with_animation_callback(
            [pcd, coordinate_system()], rotate_view, width=width, left=left)
    else:
        o3d.visualization.draw_geometries([pcd, coordinate_system()], width=width, left=left)


def render_normals(P, knn=30):
    """Render normals of a point cloud.

    Parameters
    ----------
    P : np.ndarray
        Point cloud.
    knn : int
        How many neighbours of point should be considered to estimate a normal.

    Returns
    -------
    o3d.geometry.PointCloud
        Open3D point cloud

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3])
    try:
        pcd.colors = o3d.utility.Vector3dVector(P[:, 3:6])
    except:
        print("no colors available")
    tree = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    pcd.estimate_normals(search_param=tree)
    render_pc(pcd=pcd)
    return pcd


def render_matches(
        P,
        partition_a,
        partition_gt,
        free,
        classification,
        densities,
        match_color=np.array([1, 0, 0]),
        non_match_color=np.array([0, 0, 1]),
        remaining_color=np.array([1, 1, 1]),
        animate=False,
        x_speed=2.5,
        y_speed=0.0):
    """Method to visualize the matches.

    Parameters
    ----------
    P : np.ndarray
        Point cloud.
    partition_a : Partition
        Segmentation of the agent.
    partition_gt : Partition
        Ground truth partition.
    free : int
        Segmentation value for free points.
    classification : np.ndarray
        Match classification.
    densities : np.ndarray
        Density matrix that is used for the pair classification.
    match_color : np.ndarray
        Color to visualize the segmented points within the ground truth object.
    non_match_color : np.ndarray
        Color to visualize the segmented points out of the ground truth object.
    remaining_color : np.ndarray
        There is a match between an object and a superpoint. It is the color of
        the points of an object. However, the points are not in the matched
        superpoint. Hence the points where not recognized as points of the
        object.
    animate : boolean
        If True, the point cloud will be rotated with x_speed and y_speed. A
        simulation of dragging the mouse in standard rendering mode is
        simulated.
    x_speed : float
        Used if point cloud will be animated. Strength if the horizontal mouse
        drag.
    y_speed : float
        Used if point cloud will be animated. Strength if the vertical mouse
        drag.
    """
    # Maybe visualize object borders? Make classification more explainable
    data = np.load("colors.npz")
    colors = data["colors"]
    if densities.shape[0] != classification.shape[0]:
        raise Exception("Dimension mismatch of densities and classification matrix.")
    if densities.shape[1] != classification.shape[1]:
        raise Exception("Dimension mismatch of densities and classification matrix.")
    for S in range(partition_a.n_uni):
        # Determine if the superpoint has a match
        sp_i_matches = classification[S, :]
        O_idxs = np.where(sp_i_matches != 0)[0]
        if O_idxs.shape[0] > 1:
            raise Exception("One-to-many assignment in classification")
        if O_idxs.shape[0] == 0:
            continue
        # The superpoint has a match - setup the colors
        col_mat = np.zeros((P.shape[0], 3))
        O = O_idxs[0]

        # colorize the ground truth points
        gt_P_idxs = get_P_idxs(partition=partition_gt, seg_idx=O)
        vary_object_color(
            col_mat=col_mat,
            densities=densities,
            S=S,
            O=O,
            partition=partition_a,
            P_idxs=gt_P_idxs,
            color=remaining_color)

        # colorize superpoint inliers
        main_sp_P_idxs = get_P_idxs(partition=partition_a, seg_idx=S)
        match_idxs, _, comm_main_sp = np.intersect1d(
            gt_P_idxs, main_sp_P_idxs, return_indices=True)
        col_mat[match_idxs, :] = match_color

        # identify the indices of the outliers
        non_match_idxs = np.delete(main_sp_P_idxs, comm_main_sp)
        # colorize the other objects and the outliers
        for O_j in range(partition_gt.n_uni):
            if O_j == O:
                continue
            """
            show ground truth object and vary brightness to visualize the
            agent partition within that object - variations of the ground
            truth color
            """
            col = colors[O_j, :] / 255
            gt_idxs = get_P_idxs(partition=partition_gt, seg_idx=O_j)
            col_mat[gt_idxs, :] = col
            vary_object_color(
                col_mat=col_mat,
                densities=densities,
                S=S,
                O=O_j,
                partition=partition_a,
                P_idxs=gt_idxs,
                color=col)

            """
            vary brightness to visualize the outlier that intersects other
            objects - variations of the non-match color
            """
            intersection = np.intersect1d(gt_idxs, non_match_idxs)
            f = O_j / partition_gt.n_uni
            col_mat[intersection, :] = f * non_match_color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(col_mat)

        print("Inliers:", match_idxs.shape[0])
        print("Remaining:", gt_P_idxs.shape[0] - match_idxs.shape[0])
        print("Outliers:", non_match_idxs.shape[0])
        print("Match Type:", classification[S, O])
        print("-----------------")
        render_pc(pcd=pcd, animate=animate, x_speed=x_speed, y_speed=y_speed)


def get_remaining_cloud(P, segment_indxs):
    """Filter points of the point cloud P by inserting the indices that should
    be filtered.

    Parameters
    ----------
    P : np.ndarray
        Point cloud P.
    segment_indxs : np.ndarray
        Indices of points that should be filtered.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The remaining point cloud indexes by subtracting the segmented indexes.
        The remaining points.

    """
    indxs = np.arange(P.shape[0])
    indxs = np.delete(indxs, segment_indxs, axis=0)
    return indxs, P[indxs]


def get_interval(orig_seg_idx, orig_indices, gt_object_counts):
    """Interval of a certain ground truth superpoint value.

    Parameters
    ----------
    orig_seg_idx : int
        Index of a certain ground truth superpoint.
    orig_indices : np.ndarray
        Start indices of a sorted ground truth superpoints array. The indices can
        result from a np.uniquee operation.
    gt_object_counts : np.ndarray
        Counts of the ground truth superpoint values. The counts can result from a
        np.uniquee operation.

    Returns
    -------
    tuple(int, int, int)
        Start index, length and stop index of a certain ground truth superpoint.

    """
    start = orig_indices[orig_seg_idx]
    length = gt_object_counts[orig_seg_idx]
    stop = start + length
    return start, stop, length
