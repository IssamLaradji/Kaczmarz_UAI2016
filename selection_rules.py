import numpy as np
from bisect import bisect_left         
from trees import random_tree as adapt
import safe_sparse

def select_row(rule, x, A, b, args):
    n_samples, n_features = A.shape
    epoch = args["epoch"]

    if rule == "C":
       # Cyclic 
       row_index = epoch % n_samples

    elif rule == "RP":
        # Random permutations
        if epoch == 0:
            permutation = np.arange(n_samples)

        elif epoch % n_samples == 0:
            np.random.shuffle(permutation)

        row_index = permutation[epoch % n_samples]

    elif rule == "U":
       # Uniform
       row_index = int(np.ceil(np.random.random(1)[0] * n_samples)-1)

    elif rule == "NU":
        # Non-uniform
        if epoch == 0:
            norm_sum = np.sum(args["norm_list"])
            args["norm_cumsum"] = np.cumsum(args["norm_list"] / norm_sum)

        i = float(np.random.random(1).flat[0])
        row_index = bisect_left(args["norm_cumsum"], i)

    elif rule == "A_U" or rule == "A_NU": 
        if epoch == 0:
            if rule == "A_U":
                # Adaptive uniform
                elements = np.ones(n_samples)

            elif rule == "A_NU":
                # Adaptive non-uniform
                elements = args["norm_list"]

            tree = adapt.construct_tree(elements)
            tree_array, n_leaves, n_inner_nodes = tree

            args["tree_array"] = tree_array
            args["n_leaves"] = n_leaves
            args["n_inner_nodes"] = n_inner_nodes

            neighbors, n_neighbors, max_neighbors = get_neighbors(A)
            args["neighbors"] = neighbors
            args["n_neighbors"] = n_neighbors
            args["max_neighbors"] = max_neighbors

            args["elements"] = elements

        row_index = adapt.propagate_down(args["tree_array"], 
                                         args["n_leaves"], 
                                         args["n_inner_nodes"])

    elif rule == "MR":
        # Greedy residual
        residue = np.absolute(A.dot(x.T) -  b)
        row_index = np.argmax(residue)

    elif rule == "hybrid-switch":
        if epoch % 2 == 0:
            residue = np.absolute(A.dot(x.T) -  b)
            row_index = np.argmax(residue)
        else:
            row_index, args = select_row("greedy_dis", x, A, b, args)

    elif rule == "MD":
        # Greedy Distance
        if args["epoch"] == 0:
            args["AA"] = safe_sparse.compute_AA(A)

        distance = (A.dot(x.T) -  b) ** 2

        non_zeros_indices = np.diag(args["AA"]) != 0
        zero_indices = np.diag(args["AA"]) == 0
        distance[non_zeros_indices] /= np.diag(args["AA"])[non_zeros_indices]

        assert  np.sum(np.diag(args["AA"]) < 0) == 0
        assert np.array_equal(distance[zero_indices], np.zeros(np.sum(zero_indices)))

        row_index = np.argmax(distance)
        #print np.max(distance)

    elif rule == "heap":
        neighbors, n_neighbors, max_neighbors = get_neighbors(A)
        args["neighbors"] = neighbors
        args["n_neighbors"] = n_neighbors
        args["max_neighbors"] = max_neighbors

    else:
      print "selection rule %s doesn't exist" % rule
      raise

    return row_index, args

def get_neighbors(A):
  AA = safe_sparse.compute_AA(A)

  n_rows = AA.shape[0]
  neighbors_bool = AA > 0
  max_neighbors = np.max(neighbors_bool.sum(axis=1))

  neighbors = np.ones((n_rows, max_neighbors), "int32") * -1
  n_neighbors = np.zeros(n_rows + 1, "int32")
  for i in range(n_rows):
    row_neighbors = np.where(neighbors_bool[i] > 0)[0]

    n_neighbors[i] = row_neighbors.size
    neighbors[i, :n_neighbors[i]] = row_neighbors
  
  return neighbors, n_neighbors, max_neighbors