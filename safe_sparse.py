import numpy as np

import scipy.sparse as sps

def update_x(x, A, b, args, row_index):
    row_norm = args["norm_list"][row_index]**2

    if row_norm > 1e-11:
        if sps.issparse(A):
            a_i = A.data[A.indptr[row_index]:A.indptr[row_index+1]]
            non_zeros = A.indices[A.indptr[row_index]:A.indptr[row_index+1]]

            b_i = b[row_index]
            
            update_value = (b_i - a_i.dot(x[non_zeros].T)) / row_norm
            x[non_zeros] += (a_i * update_value)

        else:
            a_i = A[row_index]
            b_i = b[row_index]
            
            update_value = (b_i - a_i.dot(x.T)) / row_norm
            x += (a_i * update_value)

    else:
        update_value = 0

    return x, update_value
    
def compute_norm_list(A):
    if sps.issparse(A):
        return np.linalg.norm(A.toarray(), axis=1)
    else:
        return np.linalg.norm(A, axis=1)

def compute_AA(A):
    if sps.issparse(A):
        return A.dot(A.T).toarray()
    else:
        return A.dot(A.T)
    
