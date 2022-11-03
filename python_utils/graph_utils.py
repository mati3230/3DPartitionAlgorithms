import numpy as np
import numpy.matlib
from multiprocessing.pool import ThreadPool
import time
#from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
from tqdm import tqdm
import h5py
from multiprocessing import Pool
import sys


def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
 
        mid = (high + low) // 2
 
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return -1


def clean_edges_helper(i, uni_source, uni_index, uni_counts, target):
    start = uni_index[i]
    stop = start + uni_counts[i]
    u_s = uni_source[i]
    res = []
    for j in range(start, stop):
        #assert(u_s == source[j])
        u_t = target[j]
        res.append([u_s, u_t, j, -1])
        if u_t > u_s: # search right
            # search index of u_t in the uni source array
            t_idx = binary_search(arr=uni_source, low=i, high=uni_source.shape[0], x=u_t)
        else: # search left
            t_idx = binary_search(arr=uni_source, low=0, high=i, x=u_t)
        if t_idx == -1:
            raise Exception("Index not found")

        t_start = uni_index[t_idx]
        t_stop = t_start + uni_counts[t_idx]
        r_idx = -1
        for r_i in range(t_start, t_stop): # iterate through source/target idxs of u_t
            if target[r_i] == u_s: # find the target u_s, i.e. the reverse edge (u_t, u_s) according to the edge (u_s, u_t)
                r_idx = r_i # we do not need this edge anymore
                break
        if r_idx == -1:
            continue
        res[-1][-1] = r_idx
    return res


def clean_edges_threads(d_mesh, verbose=False):
    t1 = time.time()
    source = d_mesh["source"]
    target = d_mesh["target"]
    distances = d_mesh["distances"]

    uni_source, uni_index, uni_counts = np.unique(source, return_index=True, return_counts=True)
    mask = np.ones((source.shape[0], ), dtype=np.bool)
    checked = np.zeros((source.shape[0], ), dtype=np.bool)
    if verbose:
        print("start threads")
        print("process {0} unique edges".format(uni_source.shape[0]))
    #"""
    tpool = ThreadPool(1000)
    w_args = []
    #for i in range(1000):
    for i in range(uni_source.shape[0]):
        #w_args.append((i, np.array(uni_source, copy=True), np.array(uni_index, copy=True), np.array(uni_counts, copy=True), np.array(target, copy=True)))
        ui = uni_source[i]
        w_args.append((ui, uni_source, uni_index, uni_counts, target))

    results = tpool.starmap(clean_edges_helper, w_args)
    #"""
    t2 = time.time()
    if verbose:
        print("threads finished in {0:.3f} seconds".format(t2-t1))
    #print(len(results), len(results[0]))
    t1 = time.time()
    nrpl = 0
    for i in range(len(results)):
        result = results[i]
        for j in range(len(result)):
            res = result[j]
            ui = res[0] # source idx in uni_source
            uj = res[1] # target idx in source, target array
            s_idx = res[2]
            checked[s_idx] = True
            t_idx = res[3] # the reverse index in the source, target array
            if t_idx == -1:
                continue
            else:
                nrpl += 1
            if checked[t_idx]:
                continue
            #print(ui, uj, s_idx, t_idx)
            checked[t_idx] = True
            mask[t_idx] = False
    t2 = time.time()
    print("nrpl:", nrpl)
    c_source = np.array(source[mask], copy=True)
    c_target = np.array(target[mask], copy=True)
    c_distances = np.array(distances[mask], copy=True)
    if verbose:
        print("post processed edges in {0:.3f} seconds ".format(t2-t1))
    return {
        "source": source,
        "target": target,
        "c_source": c_source,
        "c_target": c_target,
        "distances": distances,
        "c_distances": c_distances
    }


def clean_edges(d_mesh, verbose=False):
    source = d_mesh["source"]
    target = d_mesh["target"]
    distances = d_mesh["distances"]

    uni_source, uni_index, uni_counts = np.unique(source, return_index=True, return_counts=True)
    mask = np.ones((source.shape[0], ), dtype=np.bool)
    checked = np.zeros((uni_source.shape[0], ), dtype=np.bool)

    for i in tqdm(range(uni_source.shape[0]), desc="Remove bidirectional edges", disable=not verbose):
        start = uni_index[i]
        stop = start + uni_counts[i]
        u_s = uni_source[i]
        for j in range(start, stop):
            u_t = target[j]
            t_idx = np.where(uni_source == u_t)[0][0]
            if checked[t_idx]:
                #print("check")
                continue
            t_start = uni_index[t_idx]
            t_stop = t_start + uni_counts[t_idx]
            #print(t_idx, t_start, t_stop)
            reverse_idxs = np.where(target[t_start:t_stop] == u_s)[0]
            if reverse_idxs.shape[0] == 0:
                continue
            reverse_idxs += t_start
            mask[reverse_idxs] = False
        checked[i] = True
        
    #print("Delete {0} reverse edges".format(np.sum(mask)))
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


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0, verbose=True):
    if verbose:
        print("Compute Graph NN")
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    #assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    #nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    tree = KDTree(xyz)
    distances, neighbors = tree.query(xyz, k=k_nn2+1)
    # get the k nearest neighbours of every point in the point cloud.
    #distances, neighbors = nn.kneighbors(xyz)
    #del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    # row wise flattening with shape: (n_ver * k_nn1, )
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        # --- We do not use the voronoi functionality yet ---
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]
        
        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))
        
        edg_id = graph["source"] + n_ver * graph["target"]
        
        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]
       
        graph["distances"] = graph["distances"][keep_edges]
    else:
        # matrix n_ver X k_nn1
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]

        graph["source"] = np.matlib.repmat(range(0, n_ver), k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    if verbose:
        print("Done")
    sortation = np.argsort(graph["source"])
    graph["source"] = graph["source"][sortation]
    graph["target"] = graph["target"][sortation]
    graph["distances"] = graph["distances"][sortation]
    return graph, target2


def refine(point_idxs, p_vec, adjacency_list):
    # BFS to erase unlabelled points
    n_points = p_vec.shape[0]
    p2process = np.arange(n_points, dtype=np.int32)
    todo = np.zeros((n_points, ), dtype=np.bool)
    todo[p_vec == -1] = True
    tmp_p2process = p2process[todo == True]
    last_amount = tmp_p2process.shape[0]
    while tmp_p2process.shape[0] > 0:
        #print(tmp_p2process.shape[0])
        for i in range(tmp_p2process.shape[0]):
            idx = tmp_p2process[i]
            label = p_vec[idx]
            unlabelled = label == -1
            todo[idx] = unlabelled # we only have to analyse unlabelled points
            if unlabelled:
                N = list(adjacency_list[idx]) # one hop neighbourhood of N
                labels_N = p_vec[N] # get the labels of the neighbourhood
                uni_labels_N, counts_N = np.unique(labels_N, return_counts=True) # get histogram
                if uni_labels_N.shape[0] == 1:
                    n_unlabelled = uni_labels_N[0] == -1
                    if n_unlabelled:
                        continue
                    p_vec[idx] = uni_labels_N[0]
                    todo[idx] = False
                elif uni_labels_N.shape[0] > 1:
                    counts_N = counts_N[uni_labels_N != -1]
                    uni_labels_N = uni_labels_N[uni_labels_N != -1]
                    max_idx = np.argmax(counts_N)
                    p_vec[idx] = uni_labels_N[max_idx]
                    todo[idx] = False
        tmp_p2process = p2process[todo == True]
        amount = tmp_p2process.shape[0]
        if last_amount == amount:
            #print("Warning: No improvement")
            break
        last_amount = amount
    max_label = np.max(p_vec)
    point_idxs = []
    for i in range(max_label + 1):
        cluster = np.where(p_vec == i)[0]
        if cluster.shape[0] == 0:
            continue
        point_idxs.append(cluster)
    idxs = np.where(p_vec == -1)[0]
    if idxs.shape[0] > 0:
        max_sp = np.max(p_vec)
        p_vec[idxs] = max_sp + 1
        point_idxs.append(idxs)
    return point_idxs, p_vec


def load_nn_file(fdir, fname, verbose=True, use_c=True):
    if verbose:
        print("Load nearest neighbours")
    hf = h5py.File("{0}/nn_{1}.h5".format(fdir, fname), "r")

    distances = np.array(hf["distances"], copy=True)
    #edge_weight = np.array(hf["edge_weight"], copy=True)
    source = np.array(hf["source"], copy=True)
    target = np.array(hf["target"], copy=True)
    if use_c:
        c_distances = np.array(hf["c_distances"], copy=True)
        #edge_weight = np.array(hf["edge_weight"], copy=True)
        c_source = np.array(hf["c_source"], copy=True)
        c_target = np.array(hf["c_target"], copy=True)

        d_mesh = {
            "distances": distances,
            #"edge_weight": edge_weight,
            "source": source,
            "target": target,
            "c_distances": c_distances,
            "c_source": c_source,
            "c_target": c_target
        }
    else:
        d_mesh = {
            "distances": distances,
            #"edge_weight": edge_weight,
            "source": source,
            "target": target
        }

    if verbose:
        print("Done")
    return d_mesh


def save_nn_file(fdir, fname, d_mesh, verbose=True, use_c=True):
    if verbose:
        print("Save nearest neighbours")
    hf = h5py.File("{0}/nn_{1}.h5".format(fdir, fname), "w")
    hf.create_dataset("source", data=d_mesh["source"])
    hf.create_dataset("target", data=d_mesh["target"])
    hf.create_dataset("distances", data=d_mesh["distances"])
    if use_c:
        hf.create_dataset("c_source", data=d_mesh["c_source"])
        hf.create_dataset("c_target", data=d_mesh["c_target"])
        hf.create_dataset("c_distances", data=d_mesh["c_distances"])
    #hf.create_dataset("edge_weight", data=d_mesh["edge_weight"])
    hf.close()
    if verbose:
        print("Done")


def get_neigh(v, dv, edges, direct_neigh_idxs, n_edges, distances):
    """Query the direct neighbours of the vertex v. The distance dv to the vertex v
    will be added to the distances of the direct neigbours. 

    Parameters
    ----------
    v : int
        A vertex index
    dv : float
        Distances to the vertex v
    edges : np.ndarray
        Edges in the graph in a source target format
    direct_neigh_idxs : np.ndarray
        Array with the direct neighbours of the vertices in the graph.
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    distances : np.ndarray
        Array that containes the direct neigbours of a vertex

    Returns
    -------
    neighs : np.ndarray
        Direct neighbours (adjacent vertices) of the vertex v
    dists : np.ndarray
        The distances to the direct neighbourhood.
    """
    start = direct_neigh_idxs[v]
    stop = start + n_edges[v]
    neighs = edges[1, start:stop]
    dists = dv + distances[start:stop]
    return neighs, dists


def search_bfs_distance(vi, edges, distances, direct_neigh_idxs, n_edges, max_distance):
    """Search k nearest neigbours of a vertex with index vi with a BFS.

    Parameters
    ----------
    vi : int
        A vertex index
    edges : np.ndarray
        The edges of the mesh stored as 2xN array where N is the number of edges.
        The first row characterizes
    distances : np.ndarray
        distances of the edges in the graph.
    direct_neigh_idxs : np.ndarray
        Array that containes the direct neigbours of a vertex
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    k : int
        Number of neighbours that should be found

    Returns
    -------
    fedges : np.ndarray
        An array of size 3xk.
        The first two rows are the neighbourhood connections in a source,
        target format. The last row stores the distances between the
        nearest neighbours.

    """
    source = []
    target = []
    distances = []

    # a list of tuples where each tuple consist of a path and its length
    #shortest_paths = []
    paths_to_check = [(vi, 0)]
    # all paths that we observe
    #paths = []
    # bound to consider a path as nearest neighbour
    bound = max_distance
    # does the shortest paths contain k neighbours?, i.e. len(sortest_paths) == k
    #k_reached = False
    # dictionary containing all target vertices with the path length as value
    all_p_lens = {vi:0}
    # outer while loop
    while len(paths_to_check) > 0:
        #print("iter")
        # ---------BFS--------------
        tmp_paths_to_check = {}
        # we empty all paths to check at each iteration and fill them up at the end of the outer while loop
        while len(paths_to_check) > 0:
            target, path_distance = paths_to_check.pop(0)
            # if path is too long, we do not need to consider it anymore
            #if path_distance >= bound:
            #    continue
            # get the adjacent vertices of the target (last) vertex of this path 
            ns, ds = get_neigh(
                v=target,
                dv=path_distance,
                edges=edges,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                distances=distances)
            for z in range(ns.shape[0]):
                vn = int(ns[z])
                ds_z = ds[z]
                """
                ensure that you always save the shortest path to a target
                and that this shortest path is considered for future iterations
                """
                if vn in all_p_lens:
                    p_d = all_p_lens[vn]
                    if ds_z >= p_d:
                        continue
                all_p_lens[vn] = ds_z
                # new path that to be considered in the next iteration
                #tmp_paths_to_check.append((vn, ds_z))
                tmp_paths_to_check[vn] = ds_z
        # end inner while loop
        # sort the paths according to the distances in ascending order
        all_p_lens = dict(sorted(all_p_lens.items(), key=lambda x: x[1], reverse=False))
        for vert, dist in tmp_paths_to_check.items():
            if dist >= bound:
                continue
            paths_to_check.append((vert, dist))

    # end outer while loop
    # finally, return thek nearest targets and distances
    for key, value in all_p_lens.items():
        if key == vi:
            continue
        source.append(vi)
        target.append(key)
        distances.append(value)
    n_found = len(source)
    fedges = np.zeros((3, n_found), dtype=np.float32)
    fedges[0, :] = source
    fedges[1, :] = target
    fedges[2, :] = distances
    return fedges


def search_bfs_depth(vi, edges, distances, direct_neigh_idxs, n_edges, depth):
    """Search k nearest neigbours of a vertex with index vi with a BFS.

    Parameters
    ----------
    vi : int
        A vertex index
    edges : np.ndarray
        The edges of the mesh stored as 2xN array where N is the number of edges.
        The first row characterizes
    distances : np.ndarray
        distances of the edges in the graph.
    direct_neigh_idxs : np.ndarray
        Array that containes the direct neigbours of a vertex
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    k : int
        Number of neighbours that should be found

    Returns
    -------
    fedges : np.ndarray
        An array of size 3xk.
        The first two rows are the neighbourhood connections in a source,
        target format. The last row stores the distances between the
        nearest neighbours.

    """
    # output structure (source, target, distance)
    out_source = []
    out_target = []
    out_distances = []

    # a list of tuples where each tuple consist of a path and its length
    #shortest_paths = []
    paths_to_check = [(vi, 0)]
    # all paths that we observe
    #paths = []
    # does the shortest paths contain k neighbours?, i.e. len(sortest_paths) == k
    #k_reached = False
    # dictionary containing all target vertices with the path length as value
    all_p_lens = {vi:0}
    # outer while loop
    for j in range(depth):
        #print("iter")
        # ---------BFS--------------
        tmp_paths_to_check = {}
        # we empty all paths to check at each iteration and fill them up at the end of the outer while loop
        while len(paths_to_check) > 0:
            target, path_distance = paths_to_check.pop(0)
            # if path is too long, we do not need to consider it anymore
            #if path_distance >= bound:
            #    continue
            # get the adjacent vertices of the target (last) vertex of this path 
            ns, ds = get_neigh(
                v=target,
                dv=path_distance,
                edges=edges,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                distances=distances)
            for z in range(ns.shape[0]):
                vn = int(ns[z])
                ds_z = ds[z]
                """
                ensure that you always save the shortest path to a target
                and that this shortest path is considered for future iterations
                """
                if vn in all_p_lens:
                    p_d = all_p_lens[vn]
                    if ds_z >= p_d:
                        continue
                all_p_lens[vn] = ds_z
                # new path that to be considered in the next iteration
                #tmp_paths_to_check.append((vn, ds_z))
                tmp_paths_to_check[vn] = ds_z
        # end inner while loop
        # sort the paths according to the distances in ascending order
        #all_p_lens = dict(sorted(all_p_lens.items(), key=lambda x: x[1], reverse=False))
        
        # throw paths away that have a larger distance than the bound
        """for j in range(len(tmp_paths_to_check)):
            target, path_distance = tmp_paths_to_check[j]
            if path_distance >= bound:
                continue
            paths_to_check.append((target, path_distance))"""
        for vert, dist in tmp_paths_to_check.items():
            paths_to_check.append((vert, dist))

    # end outer while loop
    # finally, return thek nearest targets and distances
    for key, value in all_p_lens.items():
        out_source.append(vi)
        out_target.append(key)
        out_distances.append(value)
    return np.array(out_source, dtype=np.uint32), np.array(out_target, dtype=np.uint32), np.array(out_distances, dtype=np.float32)


def search_bfs(vi, edges, distances, direct_neigh_idxs, n_edges, k):
    """Search k nearest neigbours of a vertex with index vi with a BFS.

    Parameters
    ----------
    vi : int
        A vertex index
    edges : np.ndarray
        The edges of the mesh stored as 2xN array where N is the number of edges.
        The first row characterizes
    distances : np.ndarray
        distances of the edges in the graph.
    direct_neigh_idxs : np.ndarray
        Array that containes the direct neigbours of a vertex
    n_edges : np.ndarray
        Number of adjacent vertices per vertex
    k : int
        Number of neighbours that should be found

    Returns
    -------
    fedges : np.ndarray
        An array of size 3xk.
        The first two rows are the neighbourhood connections in a source,
        target format. The last row stores the distances between the
        nearest neighbours.

    """
    # output structure (source, target, distance)
    fedges = np.zeros((3, k), dtype=np.float32)
    fedges[0, :] = vi

    # a list of tuples where each tuple consist of a path and its length
    #shortest_paths = []
    paths_to_check = [(vi, 0)]
    # all paths that we observe
    #paths = []
    # bound to consider a path as nearest neighbour
    bound = sys.maxsize
    # does the shortest paths contain k neighbours?, i.e. len(sortest_paths) == k
    #k_reached = False
    # dictionary containing all target vertices with the path length as value
    all_p_lens = {vi:0}
    # outer while loop
    while len(paths_to_check) > 0:
        #print("iter")
        # ---------BFS--------------
        tmp_paths_to_check = {}
        # we empty all paths to check at each iteration and fill them up at the end of the outer while loop
        while len(paths_to_check) > 0:
            target, path_distance = paths_to_check.pop(0)
            # if path is too long, we do not need to consider it anymore
            #if path_distance >= bound:
            #    continue
            # get the adjacent vertices of the target (last) vertex of this path 
            ns, ds = get_neigh(
                v=target,
                dv=path_distance,
                edges=edges,
                direct_neigh_idxs=direct_neigh_idxs,
                n_edges=n_edges,
                distances=distances)
            for z in range(ns.shape[0]):
                vn = int(ns[z])
                ds_z = ds[z]
                """
                ensure that you always save the shortest path to a target
                and that this shortest path is considered for future iterations
                """
                if vn in all_p_lens:
                    p_d = all_p_lens[vn]
                    if ds_z >= p_d:
                        continue
                all_p_lens[vn] = ds_z
                # new path that to be considered in the next iteration
                #tmp_paths_to_check.append((vn, ds_z))
                tmp_paths_to_check[vn] = ds_z
        # end inner while loop
        # sort the paths according to the distances in ascending order
        all_p_lens = dict(sorted(all_p_lens.items(), key=lambda x: x[1], reverse=False))
        # update the bound
        if len(all_p_lens) >= k+1:
            #old_bound = bound
            bound = list(all_p_lens.values())[k]
            #if bound > old_bound:
            #    raise Exception("Bound Error: {0}, {1}".format(bound, old_bound))
        
        # throw paths away that have a larger distance than the bound
        """for j in range(len(tmp_paths_to_check)):
            target, path_distance = tmp_paths_to_check[j]
            if path_distance >= bound:
                continue
            paths_to_check.append((target, path_distance))"""
        for vert, dist in tmp_paths_to_check.items():
            if dist >= bound:
                continue
            paths_to_check.append((vert, dist))

    # end outer while loop
    # finally, return thek nearest targets and distances
    added = 0
    for key, value in all_p_lens.items():
        if key == vi:
            continue
        if added == k:
            break
        fedges[1, added] = key
        fedges[2, added] = value
        added += 1
    if added != k:
        raise Exception("Vertex {2}: Only found {0}/{1} neighbours".format(added, k, vi))
    return fedges


def get_edges(mesh_vertices, adj_list):
    # extract the neighbourhood of each point
    neighbours = []
    v_idxs = []
    for i in range(mesh_vertices.shape[0]):
        al = adj_list[i]
        v_idxs.extend(len(al) * [i])
        neighbours.extend(list(al))
    # edges as 2Xn array
    edges = np.zeros((2, len(neighbours)), dtype=np.uint32)
    edges[0, :] = v_idxs
    edges[1, :] = neighbours
    edges = np.unique(edges, axis=1)
    
    # sort the source vertices
    sortation = np.argsort(edges[0])
    edges = edges[:, sortation]
    #print(edges[:, :100])

    # edges in source, target layout
    source = edges[0]
    target = edges[1]

    # distances of the edges from a vertex v_i to v_j 
    distances = np.sqrt(np.sum((mesh_vertices[source] - mesh_vertices[target])**2, axis=1))

    # unique vertices with the begin of each vertex (which is stored in 'direct_neigh_idxs') plus the number of neighbours 'n_edges'
    uni_verts, direct_neigh_idxs, n_edges = np.unique(edges[0, :], return_index=True, return_counts=True)
    #print(uni_verts.shape[0], mesh_vertices.shape[0])
    return source, target, distances, uni_verts, direct_neigh_idxs, n_edges


def sort_graph(source, target, distances):
    sortation = np.argsort(source)
    source = source[sortation]
    target = target[sortation]
    distances = distances[sortation]
    uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    return source, target, distances, uni_verts, direct_neigh_idxs, n_edges


def geodesic_knn(mesh_vertices, adj_list, knn, n_proc=1, verbose=False):
    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = get_edges(mesh_vertices=mesh_vertices, adj_list=adj_list)

    if n_proc > 1:
        args = [(vidx, edges, distances, direct_neigh_idxs, n_edges, knn) for vidx in range(uni_verts.shape[0])]
        t1 = time.time()
        if verbose:
            print("Use {0} processes".format(n_proc))
        with Pool(processes=n_proc) as p:
            res = p.starmap(search_bfs, args)
        t2 = time.time()
        medges = np.hstack(res)
        f_edges = medges[:2, :].astype(np.uint32)
        f_distances = medges[2, :]
        if verbose:
            print("Done in {0:.3f} seconds".format(t2-t1))
    else:
        f_edges = np.zeros((2, uni_verts.shape[0]*knn), dtype=np.uint32)
        f_distances = np.zeros((uni_verts.shape[0]*knn, ), dtype=np.float32)
        arr_idx = 0
        for v_idx in tqdm(range(uni_verts.shape[0]), desc="Searching", disable=not verbose):
            fedges = search_bfs(vi=v_idx, edges=edges, distances=distances,
                direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges, k=knn)
            f_edges[0, arr_idx:arr_idx+knn] = v_idx
            f_edges[1, arr_idx:arr_idx+knn] = fedges[1, :]
            f_distances[arr_idx:arr_idx+knn] = fedges[2, :]
            arr_idx += knn
    source = f_edges[0]
    target = f_edges[1]

    return {"source": source, "target": target, "distances": f_distances}