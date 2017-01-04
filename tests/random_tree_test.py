import numpy as np
from numba import autojit
import random as rand
import time

import adaptive_jit as adept


def unit_tests():
    #n_samples = 10
    #elements = np.array([2.9, 4., 6.,10., 1.2, 8., 2., 9., 2.5, 7.])

    # Test 1 - propagate up
    elements = np.array([3.,5.,1.])
    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)

    assert np.array_equal(tree_array, np.array([9, 6, 3.,5.,1.]))

    elements = np.array([4.,8.,20.,15,-5.,-4])
    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)

    assert np.array_equal(tree_array, np.array([ 38. , 26. , 12.,  35.,  -9.,   4. ,  8. , 20.,  15.,  -5.,  -4.]))




    # Test 2 - single propagate up
    new_value = -100
    row_index = 2

    adept.single_propagate_up(new_value, row_index, tree_array , n_leaves, n_inner_nodes)
    assert np.array_equal(tree_array, np.array([ -82.,  -94.,   12.,  -85. ,  -9.,    4. ,   8. ,-100. ,  15. ,  -5.,-4.]))


    new_value = 100
    row_index = 1
    adept.single_propagate_up(new_value, row_index, tree_array , n_leaves, n_inner_nodes)
    assert np.array_equal(tree_array, np.array([ 10.,  -94.,   104.,  -85. ,  -9.,    4. ,  100. ,-100. ,  15. ,  -5.,-4.]))

    # Test 3 - batch propagate up
    elements = np.array([4.,8.,20.,15,-5.,-4])
    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)

    new_values = np.array([-100., 100])
    row_indices = np.array([2, 1]).astype(np.int64)

    adept.batch_propagate_up(new_values, row_indices, tree_array , n_leaves, n_inner_nodes)
    assert np.array_equal(tree_array, np.array([ 10.,  -94., 104., -85. , -9., 4. , 100. ,-100. , 15. , -5.,-4.]))

    n_samples = 10000
    elements_1 = np.random.rand(n_samples)

    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements_1)


    n_samples = 10000
    elements_2 = np.random.rand(n_samples)

    tree_array_2, n_leaves_2, n_inner_nodes_2 = adept.construct_tree(elements_2)

    adept.batch_propagate_up(elements_2, np.arange(n_samples).astype(np.int64), tree_array , n_leaves, n_inner_nodes)

    assert  abs(np.linalg.norm(tree_array - tree_array_2)) < 1e-5

    # Test 4 - test propagate down
    elements = np.array([4.,4.,4.,4.,4.,1,1,1,1.,1])
    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)

    left, right = 0,0
    for i in range(100000):
        k = adept.propagate_down(tree_array, n_leaves, n_inner_nodes)
        assert k >= 0
        assert k < n_leaves

        if k < n_leaves / 2:
            left += 1
        else:
            right +=1
    assert abs(float(left)/right - 4) < 0.4

    elements = np.array([5.,5.,5.,5.,5.,10,10,10,10.,10])
    tree_array, n_leaves, n_inner_nodes = adept.construct_tree(elements)

    left, right = 0,0
    for i in range(100000):
        k = adept.propagate_down(tree_array, n_leaves, n_inner_nodes)
        assert k >= 0
        assert k < n_leaves

        if k < n_leaves / 2:
            left += 1
        else:
            right +=1

    assert abs(float(right)/left - 2) < 0.4

s = time.time()
unit_tests()
print time.time() - s

