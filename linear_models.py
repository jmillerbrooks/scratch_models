import numpy as np


def descend_step(thetas, gradient, step_size):
    """Descend one step in the direction of gradient, multiplied by
    scalar step_size, by adding step_size*gradient to the coefficient vector
    thetas"""
    step = step_size * gradient
    return thetas - step


def gradient_vector_thetas(X, y, thetas):
    residuals = y - X.dot(thetas)
    gradient = 2 * X.T.dot(residuals)
    return gradient


def add_intercept(X):
    """helper function to add bias/intercept column to a feature matrix X,
    returns X with a leading column of ones added"""
    X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
    return X_with_intercept


def batch_gradient_descent(X, y, eta0=.01, max_iter=10, thetas=None):
    X_bias = add_intercept(X)
    if thetas is None:
        thetas = np.random.randn((X.shape[1], 1))
    return thetas


class LinRegScratch():
    def __init__(self):
        pass

    def fit(self, X, y):
        # add an intercept column to X
        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        # make this accessible after fitting
        self.X_intercept = X_with_intercept

        self.thetas = np.linalg.inv(X_with_intercept.T.dot(
            X_with_intercept)).dot(X_with_intercept.T).dot(y)
        return self

    def predict(self, X):
        thetas = self.thetas
        X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
        self.predictions = X_predictor.dot(thetas)
        return self.predictions


class RidgeRegScratch():
    def __init__(self, alpha=1.0, solver='closed'):
        self.alpha = alpha
        self.solver = solver

    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        self.X_intercept = X_with_intercept
        if self.solver == 'closed':
            # number of columns in matrix of X including intercept
            dimension = X_with_intercept.shape[1]
            # Identity matrix of dimension compatible with our X_intercept Matrix
            A = np.identity(dimension)
            # set first 1 on the diagonal to zero so as not to include a bias term for
            # the intercept
            A[0, 0] = 0
            # We create a bias term corresponding to alpha for each column of X not
            # including the intercept
            A_biased = self.alpha * A
            thetas = np.linalg.inv(X_with_intercept.T.dot(
                X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
        self.thetas = thetas
        return self

    def predict(self, X):
        thetas = self.thetas
        X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
        self.predictions = X_predictor.dot(thetas)
        return self.predictions
