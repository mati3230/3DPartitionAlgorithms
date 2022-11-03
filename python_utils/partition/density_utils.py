import numpy as np
import tensorflow as tf


def densities_np_osize(uni_a, uni_b, p_a, p_b, indices_a, counts_a):
    """ Compute the densities between a partition A and B with a numpy function. 
    The object sizes will be considered.


    Parameters
    ----------
    uni_a : np.ndarray
        Unique values of the partition A.
    uni_b : np.ndarray
        Unique values of the partition B.
    p_a : np.ndarray
        Partition Vector of A.
    p_b : np.ndarray
        Partition Vector of B.
    indices_a : np.ndarray
        The indices according to the unique values of the Partition A.
    counts_a : np.ndarray
        The counts according to the unique values of the Partition A.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    n_uni_b = uni_b.shape[0]
    n_uni_a = uni_a.shape[0]
    densities = np.zeros((n_uni_b, n_uni_a), dtype=np.float32)
    #print("compute densities", n_uni_a, n_uni_b, len(p_a), len(p_b))

    for i in range(n_uni_a):
        idx_a = indices_a[i]
        count_a = counts_a[i]
        # O = uni_a[i]
        p_b_sub = p_b[idx_a:idx_a+count_a]
        #print(idx_a, count_a, p_b_sub)
        for j in range(uni_b.shape[0]):
            S = uni_b[j]
            idxs_b = np.where(p_b_sub == S)[0]
            density_SO = idxs_b.shape[0]
            #print(density_SO)
            densities[j, i] = density_SO / count_a
    return densities


def densities_np(uni_a, uni_b, p_a, p_b, indices_a, counts_a):
    """ Compute the densities between a partition A and B with a numpy function. 
    The object sizes will not be considered.


    Parameters
    ----------
    uni_a : np.ndarray
        Unique values of the partition A.
    uni_b : np.ndarray
        Unique values of the partition B.
    p_a : np.ndarray
        Partition Vector of A.
    p_b : np.ndarray
        Partition Vector of B.
    indices_a : np.ndarray
        The indices according to the unique values of the Partition A.
    counts_a : np.ndarray
        The counts according to the unique values of the Partition A.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    n_uni_b = uni_b.shape[0]
    n_uni_a = uni_a.shape[0]
    densities = np.zeros((n_uni_b, n_uni_a), dtype=np.float32)
    #print("compute densities", n_uni_a, n_uni_b)
    for i in range(n_uni_a):
        idx_a = indices_a[i]
        count_a = counts_a[i]
        # O = uni_a[i]
        p_b_sub = p_b[idx_a:idx_a+count_a]
        for j in range(n_uni_b):
            S = uni_b[j]
            idxs_b = np.where(p_b_sub == S)[0]
            density_SO = idxs_b.shape[0]
            densities[j, i] = density_SO
    return densities


def densities_np_inters_osize(uni_a, uni_b, p_a, p_b):
    """ Compute the densities between a partition A and B with a numpy function. 
    The object sizes will be considered. The intersection operation will be 
    executed to calculate the densities.


    Parameters
    ----------
    uni_a : np.ndarray
        Unique values of the partition A.
    uni_b : np.ndarray
        Unique values of the partition B.
    p_a : np.ndarray
        Partition Vector of A.
    p_b : np.ndarray
        Partition Vector of B.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    n_uni_b = uni_b.shape[0]
    n_uni_a = uni_a.shape[0]
    densities = np.zeros((n_uni_b, n_uni_a), dtype=np.float32)
    for i in range(n_uni_b):
        S = uni_b[i]
        for j in range(n_uni_a):
            O = uni_a[j]

            idxs_A = np.where(p_a == O)[0]
            idxs_B = np.where(p_b == S)[0]
            intersection = np.intersect1d(idxs_A, idxs_B)
            density = intersection.shape[0] / idxs_A.shape[0]

            densities[i, j] = density
    return densities


def densities_np_inters(uni_a, uni_b, p_a, p_b):
    """ Compute the densities between a partition A and B with a numpy function. 
    The object sizes will not be considered. The intersection operation will be 
    executed to calculate the densities.


    Parameters
    ----------
    uni_a : np.ndarray
        Unique values of the partition A.
    uni_b : np.ndarray
        Unique values of the partition B.
    p_a : np.ndarray
        Partition Vector of A.
    p_b : np.ndarray
        Partition Vector of B.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    n_uni_b = uni_b.shape[0]
    n_uni_a = uni_a.shape[0]
    densities = np.zeros((n_uni_b, n_uni_a), dtype=np.float32)
    for i in range(n_uni_b):
        S = uni_b[i]
        for j in range(n_uni_a):
            O = uni_a[j]

            idxs_A = np.where(p_a == O)[0]
            idxs_B = np.where(p_b == S)[0]
            intersection = np.intersect1d(idxs_A, idxs_B)
            density = intersection.shape[0]

            densities[i, j] = density
    return densities


def compute_densities_np(density_function, **kwargs):
    """ Template function to compute the densities.


    Parameters
    ----------
    density_function : function
        Function that computes the density values
    kwargs : dict
        Dictionary that contains information of the partitions.

    Returns
    -------
    np.ndarray
        The density matrix.
    """
    densities = density_function(**kwargs)
    return densities