import numpy as np
import math
from .density_utils import compute_densities_np


def get_rel_matches(classification, match_type_nr):
    """Get the number of matches in relation to the number of objects.

    Parameters
    ----------
    classification : np.ndarray
        Pair classification matrix.
    match_type_nr : int
        Order of match (first, second, third).

    Returns
    -------
    float
        Number of matches in relation to the number of objects.
    """
    n_matches = np.where(classification == match_type_nr)[0].shape[0]
    n_objects = classification.shape[1]
    return n_matches / n_objects


class Partition:
    """This class enables to calcute the match classification.

    Parameters
    ----------
    partition : np.ndarray
        A partition.
    uni : np.ndarray
        The unique superpoint values.
    idxs : np.ndarray
        The indices according to the unique values.
    counts : np.ndarray
        Counts of the unique superpoint values.

    Attributes
    ----------
    uni : np.ndarray
        The unique superpoint values.
    idxs : np.ndarray
        The indices according to the unique values.
    counts : np.ndarray
        Counts of the unique superpoint values.
    n_uni : int
        The number of superpoints.
    first_order_class : int
        Value of the first order match.
    second_order_class : int
        Value of the second order match.
    third_order_class : int
        Value of the third order match.
    fourth_order_class : int
        Value of the fourth order match.
    partition : np.ndarray
        A partition.
    """
    def __init__(self, partition, uni=None, idxs=None, counts=None):
        """Constructor.

        Parameters
        ----------
        partition : np.ndarray
            A partition.
        uni : np.ndarray
            The unique superpoint values.
        idxs : np.ndarray
            The indices according to the unique values.
        counts : np.ndarray
            Counts of the unique superpoint values.
        """
        self.partition = partition
        if uni is None:
            self.uni, self.idxs, self.counts = np.unique(self.partition, return_index=True, return_counts=True)
        else:
            self.uni = uni
            self.idxs = idxs
            self.counts = counts
        self.n_uni = self.uni.shape[0]
        self.first_order_class = 3
        self.second_order_class = 2
        self.third_order_class = 1
        self.fourth_order_class = 0

    def compute_densities(self, partition_B, density_function):
        """Computes the densities between the elements of two partitions. Calculate
        the densities. Create a density matrix and initialize all densities to 0. The
        density matrix has the size of n_unique_superpoints X n_unique_objects.

        Parameters
        ----------
        partition_B : Partition
            Another paritition to classify the matches.
        density_function : function(elem_A, elem_B, partition_A, partition_B)
            A function where as subset from partition A (elem_A) and
            partition B (elem_B) and
            the partitions itself (partition_A, partition_B) can be inserted.

        Returns
        -------
        np.ndarray
            2D matrix of the densities.

        """
        args = {}
        args["uni_a"] = self.uni
        args["uni_b"] = partition_B.uni
        args["p_a"] = self.partition
        args["p_b"] = partition_B.partition
        args["indices_a"] = self.idxs
        args["counts_a"] = self.counts
        #print("compute densities")
        densities = compute_densities_np(density_function, **args)
        #print("done")
        return densities

    def alpha(self, densities):
        """Best matching objects: argmax in a row

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.

        Returns
        -------
        np.ndarray
            Vector where each element represents the best matching object of a
            superpoint.
        """
        alpha = np.argmax(densities, axis=1)
        return alpha

    def beta(self, densities):
        """Majorities of superpoints: argmax in a column

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.

        Returns
        -------
        np.ndarray
            Vector where each element represents a superpoint that has the
            majority in an object. We assign the value of -1 if the density
            between all superoints and an object is 0.
        """
        beta = np.argmax(densities, axis=0)
        beta_max = np.amax(densities, axis=0)
        zero_idxs = np.where(beta_max == 0)[0]
        beta[zero_idxs] = -1
        return beta

    def alpha_beta(self, densities, alpha, beta):
        """Computation of the best majority objects.

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        alpha : np.ndarray
            Vector with best matching objects.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.

        Returns
        -------
        np.ndarray
            Vector where each element represents the best majority object of a
            superpoint.
        """
        alpha_beta = -np.ones(alpha.shape, dtype=np.int32)
        O = np.where(beta != -1)[0]
        S = beta[O]
        S_betas = np.unique(S)
        for S_beta in S_betas:
            Os = np.where(beta == S_beta)[0]
            fcs = densities[S_beta, Os]
            if fcs.shape[0] > 1:
                max_idx = np.argmax(fcs)
                alpha_beta[S_beta] = Os[max_idx]
            else:
                alpha_beta[S_beta] = Os[0]
        return alpha_beta

    def gamma(self, densities, beta, alpha_beta):
        """Computation of gamma(O).

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        beta : np.ndarray
            Vector of the superpoints that have a majority an object.
        alpha_beta : np.ndarray
            Vector of the best majority objects of each superpoint.

        Returns
        -------
        np.ndarray
            Superpoint for each object that has a potential third order
            relationship.
        """
        gamma = -np.ones(beta.shape[0], dtype=np.int32)

        for O in range(beta.shape[0]):
            S_ = beta[O]
            candidate = -1
            candidate_force = -1
            for S in range(alpha_beta.shape[0]):
                # superpoint should not have a majority
                if S == S_:
                    continue
                # density of the gamma candidate
                density = densities[S, O]
                if density == 0:
                    continue
                O_ = alpha_beta[S]
                # density with the best majority object
                force_ = 0
                if O_ != -1:
                    force_ = densities[S, O_]
                '''
                density of gamma candidate should be smaller than density with the
                best majority object
                '''
                if force_ >= density:
                    continue
                # potential candidate
                if density > candidate_force:
                    candidate_force = density
                    candidate = S
            gamma[O] = candidate
        return gamma

    def M(self, alpha, beta):
        """Computes the number of majorities of each superpoint.

        Parameters
        ----------
        alpha : np.ndarray
            Vector with best matching objects.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.

        Returns
        -------
        np.ndarray
            Number of majorities of each superpoint.
        """
        M = np.zeros(alpha.shape, dtype=np.int32)
        beta_uni, beta_counts = np.unique(beta, return_counts=True)
        M[beta_uni] = beta_counts
        return M

    #def ord3(self, alpha, beta, gamma, M):
    def ord3(self, alpha, beta, gamma):
        """Determine if there is a third order relationship between the
        superpoints and objects.

        Parameters
        ----------
        alpha : np.ndarray
            Vector with best matching objects.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.
        gamma : np.ndarray
            Vector of superpoints where each superpoint has a potential third
            order relationship with an object.
        M : np.ndarray
            Vector of the number of majoritities of each superpoint.

        Returns
        -------
        np.ndarray
            2D boolean matrix with the size of n_unique_superpoints X n_unique_objects.
            Each True entry indicates a third order relationship.
        """
        ord3 = np.zeros((alpha.shape[0], beta.shape[0]), dtype=np.bool)
        for O in range(beta.shape[0]):
            S_ = beta[O]
            if S_ != -1:
                O_ = alpha[S_]
                if O == O_:
                    continue
            else: # no density between any superpoint and object O
                continue
            #if M[S_] < 2:
            #    continue
            S_gamma = gamma[O]
            if S_gamma == -1:
                continue
            ord3[S_gamma, O] = True
        return ord3

    def b(self, densities, ord3):
        """Compute the best third order superpoint of each object.

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        ord3 : np.ndarray
            2D boolean matrix with the size of n_unique_superpoints X n_unique_objects.
            Each True entry indicates a third order relationship.

        Returns
        -------
        np.ndarray
            Vector where each element represents a superpoint that has the
            best third order majority in an object. We assign the value of -1
            if an object has no third order relationship with a superpoint.
        """
        forces_ = np.array(densities, copy=True)
        forces_[np.invert(ord3)] = 0
        b = self.beta(forces_)
        return b

    def a(self, ord3, densities):
        """Computation of the n-th best third order objects.

        Parameters
        ----------
        ord3 : np.ndarray
            2D boolean matrix with the size of n_unique_superpoints X n_unique_objects.
            Each True entry indicates a third order relationship.
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.

        Returns
        -------
        np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.
        """
        forces_ = np.array(densities, copy=True)
        forces_[np.invert(ord3)] = -1
        a = np.argsort(forces_, axis=1)
        a = a[:, ::-1]
        forces_sort = np.sort(forces_, axis=1)
        forces_sort = forces_sort[:, ::-1]
        a[forces_sort == -1] = -1
        return a

    def delta(self, a, b):
        """Compute the best third order index for each superpoint.

        Parameters
        ----------
        a : np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.
        b : np.ndarray
            Vector where each element represents a superpoint that has the
            best third order majority in an object. We assign the value of -1
            if an object has no third order relationship with a superpoint.

        Returns
        -------
        np.ndarray
            Vector with the best third order index for each superpoint.
        """
        delta = -np.ones((a.shape[0], ), dtype=np.int32)
        for S in range(a.shape[0]):
            for n in range(self.n_uni):
                O = a[S, n]
                if O == -1:
                    break
                S_ = b[O]
                if S != S_:
                    continue
                delta[S] = n
                break
        return delta

    def force2(self, S, densities, alpha_beta):
        """Determine the density between a superpoint S and its best majority
        object.

        Parameters
        ----------
        S : int
            Superpoint index.
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        alpha_beta : np.ndarray
            Vector of the best majority objects of each superpoint.

        Returns
        -------
        (float, int)
            density between a superpoint S and its best majority object. The best
            majority object.
        """
        O_alpha_beta = alpha_beta[S]
        force2 = 0

        # invalid superpoint
        if S == -1:
            return force2, O_alpha_beta

        # no best majority object
        if O_alpha_beta == -1:
            return force2, O_alpha_beta

        force2 = densities[S, O_alpha_beta]
        return force2, O_alpha_beta

    def force3(self, S, densities, delta, a):
        """Determine the density between a superpoint S and its best third order
        object.

        Parameters
        ----------
        S : int
            Superpoint index.
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        delta : np.ndarray
            Vector with the best third order index for each superpoint.
        a : np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.

        Returns
        -------
        (float, int, int)
            density between a superpoint S and its best third order object. The
            best third order index delta(S). The best third order object of S.
        """
        force3 = 0
        O_a = -1
        n = -1

        # invalid superpoint
        if S == -1:
            return force3, n, O_a

        n = delta[S]
        # no third order match
        if n == -1:
            return force3, n, O_a

        O_a = a[S, n]
        force3 = densities[S, O_a]
        return force3, n, O_a

    def first_order(self, S, O, alpha, beta):
        """Check if the superpoint S and object O have a 1st order relationship.

        Parameters
        ----------
        S : int
            Superpoint index.
        O : int
            Object index.
        alpha : np.ndarray
            Vector with best matching objects.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.

        Returns
        -------
        (boolean, int, int)
            True, if there is a 1st order relationship. The best matching
            object of S. The Superpoint that has the maajority in O.

        """
        O_alpha = alpha[S]
        S_beta = beta[O]
        if S == S_beta and O == O_alpha:
            return True, O_alpha, S_beta
        return False, O_alpha, S_beta

    def second_order(
            self, S, O, S_beta, beta, O_alpha, densities, alpha_beta, delta, a):
        """Check if the superpoint S and object O have a 2nd order relationship.

        Parameters
        ----------
        S : int
            Superpoint index.
        O : int
            Object index.
        S_beta : int
            The Superpoint that has the maajority in O.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.
        O_alpha : int
            The best matching object of S.
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        alpha_beta : np.ndarray
            Vector of the best majority objects of each superpoint.
        delta : np.ndarray
            Vector with the best third order index for each superpoint.
        a : np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.

        Returns
        -------
        (boolean, int, float, int, float, int, int)
            True, if there is a 2nd order relationship. Superpoint
            S_beta_O_alpha that has the majority in the best matching object of
            S. 2nd order density force2. Best matching object O_alpha_beta of the
            superpoint that has the majority in O. 3rd order density force3. Best
            3rd order index n and the best 3rd order object O_a.
        """
        S_beta_O_alpha = beta[O_alpha]
        force2, O_alpha_beta = self.force2(S, densities, alpha_beta)
        force3, n, O_a = self.force3(S, densities, delta, a)
        if S == S_beta and S != S_beta_O_alpha and O == O_alpha_beta and force3 <= force2:
            return True, S_beta_O_alpha, force2, O_alpha_beta, force3, n, O_a
        return False, S_beta_O_alpha, force2, O_alpha_beta, force3, n, O_a

    def third_order(
            self, O, O_a, force3, force2, S_beta, densities, alpha_beta, delta, a):
        """Check if the superpoint S and object O have a 3rd order relationship.

        Parameters
        ----------
        O : int
            Object index.
        O_a : int
            The best 3rd order object.
        force3 : float
            3rd order density.
        force2 : float
            2nd order density.
        S_beta : int
            The Superpoint that has the maajority in O.
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        alpha_beta : np.ndarray
            Vector of the best majority objects of each superpoint.
        delta : np.ndarray
            Vector with the best third order index for each superpoint.
        a : np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.

        Returns
        -------
        boolean
            True, if there is a 3rd order relationship.

        """
        if force3 > force2 and O == O_a:
            force2, O_alpha_beta = self.force2(S_beta, densities, alpha_beta)
            force3, n, O_a = self.force3(S_beta, densities, delta, a)

            # check if the beta(O) has not a higher order relationship with O
            if not (force3 <= force2 and O == O_alpha_beta):
                return True
        return False

    def apply_classification(
            self, densities, alpha, beta, alpha_beta, delta, a):
        """Applies the pair classification of the superpoints and the objects.

        Parameters
        ----------
        densities : np.ndarray
            density matrix with the size of n_unique_superpoints X n_unique_objects.
        alpha : np.ndarray
            Vector with best matching objects.
        beta : np.ndarray
            Vector with superpoints that have a majority an object.
        alpha_beta : np.ndarray
            Vector of the best majority objects of each superpoint.
        delta : np.ndarray
            Vector with the best third order index for each superpoint.
        a : np.ndarray
            2D Matrix with the size of n_unique_superpoints X n_unique_objects.
            Each Superpoint is assigned by an order of superpoints. The order
            depends on the densities between the superpoints and the objects where
            a 3 order relationship exists. The objects for each superpoints are
            sorted in descending order. Invalid relationships are assigned by a
            value of -1.

        Returns
        -------
        np.ndarray
            Order of the relationships between the superpoints and objects.

        """
        classification = np.zeros(densities.shape, dtype=np.int32)
        for S in range(alpha.shape[0]):
            for O in range(beta.shape[0]):
                result, O_alpha, S_beta = self.first_order(S, O, alpha, beta)
                if result:
                    classification[S, O] = self.first_order_class

                result, S_beta_O_alpha, force2, O_alpha_beta, force3, n, O_a =\
                    self.second_order(
                        S,
                        O,
                        S_beta,
                        beta,
                        O_alpha,
                        densities,
                        alpha_beta,
                        delta,
                        a)
                if result:
                    classification[S, O] = self.second_order_class

                result =\
                    self.third_order(
                        O,
                        O_a,
                        force3,
                        force2,
                        S_beta,
                        densities,
                        alpha_beta,
                        delta,
                        a)
                if result:
                    classification[S, O] = self.third_order_class
        return classification

    def get_index(self, S):
        """ Get the index of the unique value S which is in the partition vector.

        Parameters
        ----------
        S : int
            Unique value of the partition.

        Returns
        -------
        int
            Index of the unique value S which is in the partition vector.
        """
        idxs = np.where(self.uni == S)[0]
        return idxs[0]

    def precision_feedback(self, S_i, S_j, densities=None, partition_B=None, density_function=None, pos_val=1, neg_val=0):
        """ Expert function which maximises the sum of the precision values

        Parameters
        ----------
        action : int
            Should the superpoints be unified?

        Returns
        -------
        int
            The expert action.
        """
        if densities is None:
            if partition_B is None:
                raise Exception("Missing partition B")
            if density_function is None:
                raise Exception("Missing partition density_function")
            densities = self.compute_densities(partition_B, density_function)
        alpha = self.alpha(densities)
        idx_S_i = self.get_index(S_i)
        idx_S_j = self.get_index(S_j)
        #print("S_i:", idx_S_i, "S_j:", idx_S_j)
        #print("a S_i:", alpha[idx_S_i], "a S_j:", alpha[idx_S_j])
        #print("densities", densities)
        if alpha[idx_S_i] == alpha[idx_S_j]:
            return pos_val
        return neg_val

    def classify(self, partition_B, density_function, verbose=False):
        """Classify the matches between partition A and B.

        Parameters
        ----------
        partition_B : Partition
            Another paritition to classify the matches.
        density_function : function(elem_A, elem_B, partition_A, partition_B)
            A function where as subset from partition A (elem_A) and
            partition B (elem_B) and
            the partitions itself (partition_A, partition_B) can be inserted.
        verbose : boolean
            If True, results of the helper functions will be printed.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            2D Matrix where the entries are the classes.

        """
        densities = self.compute_densities(partition_B, density_function)
        '''
        We will consider the unique partition labels as indices due to the
        argmax opertations. For instance, the unique partition labels [1,2,3,4]
        will be mapped to the indices [0,1,2,3].
        '''
        alpha = self.alpha(densities)
        beta = self.beta(densities)
        alpha_beta = self.alpha_beta(densities, alpha, beta)
        gamma = self.gamma(densities, beta, alpha_beta)
        #M = self.M(alpha, beta)
        #ord3 = self.ord3(alpha, beta, gamma, M)
        ord3 = self.ord3(alpha, beta, gamma)
        b = self.b(densities, ord3)
        a = self.a(ord3, densities)
        delta = self.delta(a, b)
        if verbose:
            print("densities:\n", densities)
            print("alpha:\n", alpha)
            print("beta:\n", beta)
            print("alpha_beta:\n", alpha_beta)
            print("gamma:\n", gamma)
            #print("M:\n", M)
            print("ord3:\n", ord3)
            print("b:\n", b)
            print("a:\n", a)
            print("delta:\n", delta)
        classification = self.apply_classification(
            densities, alpha, beta, alpha_beta, delta, a)
        return classification, densities

    def overall_obj_acc(self, max_density, partition_B=None, density_function=None, densities=None):
        """Computes the overall object accuracy.
        See https://arxiv.org/abs/1904.02113v1 for more information.

        Parameters
        ----------
        max_density: int
            Maximum sum of densities such as the size of a point cloud |P|.
        partition_B : Partition
            Another paritition to classify the matches.
        density_function : function(elem_A, elem_B, partition_A, partition_B)
            A function where as subset from partition A (elem_A) and
            partition B (elem_B) and
            the partitions itself (partition_A, partition_B) can be inserted.
        densities : np.ndarray
            If provided, time can be saved as there is no need to compute them.

        Returns
        -------
        float
            Overall object accuracy.

        """
        if densities is None and partition_B is None and density_function is None:
            raise Exception("Cannot calculate OOA")
        if densities is None:
            densities = self.compute_densities(partition_B, density_function)
        alpha = self.alpha(densities)
        density_sum = 0
        for S in range(densities.shape[0]):
            O = alpha[S]
            density = densities[S, O]
            density_sum += density
        ooa = density_sum / max_density
        return ooa, densities
