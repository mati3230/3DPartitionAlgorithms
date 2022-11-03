import numpy as np

def rg_normals(seed_idx, search_idx, visited, neighbors, normals, angle_thres, region):
    if search_idx in visited:
        return visited, region
    region.append(search_idx)
    visited.append(search_idx)
    ref_normal = normals[seed_idx]
    angle_thres_rad = np.pi * angle_thres / 180
    for neigh in neighbors[search_idx]:
        normal = normals[neigh]
        angle = np.dot(ref_normal, normal)
        if angle < angle_thres_rad:
            visited, region = rg_normals(seed_idx=seed_idx, search_idx=neigh, visited=visited,
                neighbors=neighbors, normals=normals, angle_thres=angle_thres, region=region)
    return visited, region