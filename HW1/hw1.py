'''
hw1.py
Author: Pengcheng Xu (pengcheng.xu@tufts.edu)

Tufts CS 135 Intro ML

'''

from itertools import permutations
import numpy as np
import math 

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
    if random_state is None:
        random_state = np.random
    ## TODO fixme
    if type(random_state) == int:
        rng = np.random.RandomState(random_state)
    if type(random_state) == np.random.mtrand.RandomState:
        rng = random_state


    input_size = len(x_all_LF)

    input_index_list = list(range(input_size))

    test_size = int(math.ceil(input_size * frac_test))
    permutated_test_index_list = rng.permutation(input_index_list)[: test_size]

    test_data = np.array( [x_all_LF[idx] for idx in input_index_list if idx in permutated_test_index_list] )
    train_data = np.array( [x_all_LF[idx] for idx in input_index_list if idx not in permutated_test_index_list] )
    

    return train_data, test_data

def euclidean_dis(v1, v2) :
    return np.sum(np.square(v1 - v2))  # we don't need to calculate sqrt cuz the bigger square, the bigger sqrt

def vector_k_nearest_neighbours(v, data_NF, K):
        dis_list = [ euclidean_dis(x, v) for x in data_NF ]
        dis_idx_list = np.argsort(dis_list);
        res_idx_list = dis_idx_list[: K];
        return data_NF[res_idx_list]

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    # TODO fixme
    res = [ vector_k_nearest_neighbours(vec, data_NF, K) for vec in query_QF]

    res= np.array(res)
    return res


# x_LF = np.eye(10)

# train_MF, test_NF = split_into_train_and_test(x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
                                
# print(train_MF)
# print(test_NF)
# xcopy_LF = x_LF.copy()
# print(np.allclose(x_LF, xcopy_LF))

# test for func2

# data = np.eye(10)
# query = data[[1, 3, 5]]
# # print(query)
# print(calc_k_nearest_neighbors(data, query, K=2))