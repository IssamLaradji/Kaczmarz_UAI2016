import numpy as np
from numba import autojit
import random as rand
import time
from numba import float64, int32

def construct_tree(elements):
    n_leaves = elements.size
    n_inner_nodes = get_inner_nodes(n_leaves)

    tree_array =  np.zeros(n_inner_nodes + n_leaves)
    tree_array[-n_leaves:] = elements

    propagate_up(tree_array, n_leaves, n_inner_nodes)

    return tree_array, n_leaves, n_inner_nodes

def get_inner_nodes(n_leaves):
    """Compute the number of nodes in the tree """

    # Compute tree height
    height = int(np.ceil(np.log(n_leaves) / np.log(2)) + 1)

    # Compute number of nodes in the tree
    n_nodes_before_last_level = 2**(height - 2)
    n_inner_nodes_before_last_level = n_leaves - n_nodes_before_last_level

    n_nodes = n_inner_nodes_before_last_level + 2**(height - 2) - 1 + n_leaves

    return n_nodes - n_leaves

@autojit("float64[:], int32[:], float64[:], int64, int64", locals=dict(delta=float64, index=int32),
         nopython=True)
def batch_propagate_up(new_values, index_list, tree_array , n_leaves, n_inner_nodes):
    for new_value, row_index in zip(new_values, index_list):
        index = n_inner_nodes + row_index
        delta = new_value - tree_array[index]

        if delta == 0:
            continue

        while index >= 0:
            tree_array[index] +=  delta
            index = ((index + 1)/2) - 1

@autojit("float64[:], int64, int64", nopython=True)
def propagate_up(tree_array , n_leaves, n_inner_nodes):
    parent_index = n_inner_nodes - 1
    # Please no one sample datasets
    while parent_index >= 0:
        # Parent node = child node 1 + child node 2
        tree_array[parent_index] = tree_array[2*parent_index + 1] + tree_array[2*parent_index + 2]

        # Get next parent
        parent_index -= 1

@autojit("float64, int64, float64[:], int64, int64", nopython=True)
def single_propagate_up(new_value, row_index, tree_array , n_leaves, n_inner_nodes):
    index = n_inner_nodes + row_index
    delta = new_value - tree_array[index]

    while index >= 0:
        tree_array[index] +=  delta
        index = ((index + 1)/2) - 1



@autojit("float64[:], int64, int64", nopython=True)
def propagate_down(tree_array, n_leaves, n_inner_nodes):
    parent_index = 0

    while(parent_index < n_inner_nodes):
        value = tree_array[parent_index]

        random_number =  rand.random() * value

        child_1 = 2*parent_index + 1
        child_2 = child_1 + 1

        if tree_array[child_1] >= random_number:
            parent_index = child_1
        else:
            parent_index = child_2

    return parent_index - n_inner_nodes


