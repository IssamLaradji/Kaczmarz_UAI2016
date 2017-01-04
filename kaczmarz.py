import numpy as np

import selection_rules as sr
import update_tree as ut
import safe_sparse

from sklearn.base import BaseEstimator, RegressorMixin

effective_zero = 1E-11

def objective_rules(rule, x, A, b, args):
    if rule == "squared_loss":
        return np.linalg.norm(A.dot(x.T) - b)**2

    elif rule == "distance":
        if args["epoch"] == 0:
            # Get x_true
            pass

        return np.linalg.norm(x - args["x_true"])**2

class Kaczmarz(BaseEstimator):
    def __init__(self, n_iters=10, selection_rule="uniform", 
                 objective="squared_loss",
                 dataset_name=None,
                 verbose=True):
        self.selection_rule = selection_rule
        self.n_iters = n_iters
        self.objective = objective
        self.dataset_name = dataset_name
        self.verbose=verbose

        self.args = {"epoch":0}


    def fit(self, A, b):
        n_samples, n_features = A.shape
        self.x = np.ones(n_features)

        # Compute norm for each sample
        self.args["norm_list"] = safe_sparse.compute_norm_list(A)
        row_index = None
        
        s_rule = self.selection_rule
        s_func = lambda x, args : sr.select_row(s_rule, x, A, b, args)
        f_func = lambda x, args : objective_rules(self.objective, x, A, b, args)

        self.results = np.zeros(self.n_iters + 1)
        for epoch in range(self.n_iters + 1):
            self.args["epoch"] = epoch

            # Compute loss
            loss = f_func(self.x, self.args)  
            self.results[epoch] = loss
            print "%d - %s: %.3f" % (epoch, self.objective, loss)

            # Select row index
            row_index, self.args = s_func(self.x, self.args)
            
            # Update 'x'
            self.x, update_value = safe_sparse.update_x(self.x, A, b, self.args, row_index)
               
            # Update Tree if needed
            ut.update_tree(s_rule, self.x, A, b, self.args, row_index, update_value)

class KZRegressor(Kaczmarz, RegressorMixin):
  def __init__(self, n_iters=10, selection_rule="uniform", 
                 objective="squared_loss",
                 dataset_name=None,
                 verbose=True):
    super(KZRegressor, self).__init__(n_iters=n_iters,
                                   selection_rule=selection_rule,
                                   objective=objective,
                                   dataset_name=dataset_name,
                                   verbose=verbose)
  def predict(self, X):
    y_scores = self._decision_scores(X)

    return y_scores

  def _decision_scores(self, A):
    """Predict"""
    n_samples, n_features = A.shape

    return A.dot(self.x)