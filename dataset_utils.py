from scipy.io import loadmat
from scipy.sparse import csr_matrix
import numpy as np

def load_dataset(name):
    if name == "exp1":
        data = loadmat("datasets/exp1.mat")
        A, b = data['X'], data['y'].ravel()

        A, b = get_leastsq_formulation(A,b)
        n_samples, n_features = A.shape
        A = csr_matrix(A)


    elif name == "very_sparse":
        data = loadmat("datasets/sparse_consistent_data.mat")

        A, b = data['A_sparse_consistent'], data['b_sparse_consistent'].ravel()

        n_samples, n_features = A.shape
        A = csr_matrix(A)
        


    elif name == "label_prop":
        A = np.load("datasets/A_labelProp.npy")
        b = np.load("datasets/b_labelProp.npy")

        n_samples, n_features = A.shape
        A = csr_matrix(A)
        name = "label prop data"

    elif name == "exp3":
        n_samples = 2500
        n_features = 2500

        A = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            A[i, i] = np.random.randn()

            if i >= 1:
                A[i, i - 1] = np.random.randn()
            if i + 1 < n_samples:
                A[i, i + 1] = np.random.randn()

            if i >= 50:
                A[i, i - 50] = np.random.randn()

            if i + 50 < n_samples:
                A[i, i + 50] = np.random.randn()

        w = np.random.randn(n_features)
        b = A.dot(w) + np.random.randn(n_samples)

        A = csr_matrix(A)


    return {"A": A, "b": b}

def get_leastsq_formulation(X,y):
    n, m = X.shape
    I = np.eye(n)
    Z = np.zeros((m,m))
    A = np.vstack([np.hstack([X,-I]),np.hstack([Z,X.T])])
    z = np.zeros(m)
    b = np.hstack([y,z])
    return A, b

def isConsistent(A, b):
    A_B = np.hstack([A.toarray(), b[:,np.newaxis]])
    r1 = np.linalg.matrix_rank(A.toarray())
    r2 = np.linalg.matrix_rank(A_B)

    if r1 == r2:
        print "System is Consistent!"
        return True
    else:
        print "System is not Consistent!"
        return False