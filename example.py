from sklearn import datasets
from kaczmarz import KZRegressor

import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    n_samples, n_features = 1000, 500

    # Regression
    X, y = datasets.make_regression(n_samples, n_features)

    reg = KZRegressor(selection_rule='MD', n_iters=300, verbose=True)

    # Train regressor
    reg.fit(X, y)

    print "\nRegression training score: %.3f" % reg.score(X,y)

    # Regression training score:  0.999999798676
