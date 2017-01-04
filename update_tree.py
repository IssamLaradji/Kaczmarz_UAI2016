from trees import random_tree as adapt

def update_tree(rule, x, A, b, args, row_index, update_value):
    if rule == "A_NU" or rule == "A_U":
        # update random tree
        neighbours = args["neighbors"][row_index]
        new_values = args["elements"][neighbours]

        adapt.batch_propagate_up(new_values, neighbours, 
                                 args["tree_array"], 
                                 args["n_leaves"], 
                                 args["n_inner_nodes"])

        adapt.single_propagate_up(0, row_index, 
                                  args["tree_array"], 
                                  args["n_leaves"], 
                                  args["n_inner_nodes"])

    elif rule == "min_heap":
        neighbours = neighbors_list[neighbors_index[sample_index]:neighbors_index[sample_index+1]]
        #n_neighbors = neighbours.size

        a_dot_x[neighbours] += (update_value * AA_transpose[sample_index, neighbours])

        new_residuals = abs(a_dot_x[neighbours] - b[neighbours])

        if self.algorithm == "heap_rule_distance":
            new_residuals = (new_residuals**2) / diag_AA[neighbours]
            #nn = new_residuals.copy()
            #np.testing.assert_equal(numba_square_divide(new_residuals, diag_AA[neighbours], n_neighbors), (nn**2) / diag_AA[neighbours])
            #new_residuals = numba_square_divide(new_residuals, diag_AA[neighbours], n_neighbors)
        hp.batch_update(new_residuals, neighbours, heap, heap_to_list_index, list_to_heap_index, n_samples)

