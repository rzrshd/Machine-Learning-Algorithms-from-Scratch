import math
from sklearn import datasets
import numpy as np


def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return math.sqrt(distance)

def calculate_covariance_matrix(X, Y=None):
    if Y is None:
        Y = X

    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


class GaussianMixtureModel():
    def __init__(self, k=2, max_iterations=2000, tolerance=1e-8):
        self.k = k
        self.parameters = []
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.responsibilities = []
        self.sample_assignments = None
        self.responsibility = None

    def _init_random_gaussians(self, X):
        n_samples = np.shape(X)[0]
        self.priors = (1 / self.k) * np.ones(self.k)
        for i in range(self.k):
            params = {}
            params['mean'] = X[np.random.choice(range(n_samples))]
            params['cov'] = calculate_covariance_matrix(X)
            self.parameters.append(params)

    def multivariate_gaussian(self, X, params):
        n_features = np.shape(X)[1]
        mean = params['mean']
        covar = params['cov']
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])
        for i, sample in enumerate(X):
            d = n_features  # dimension
            coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2) * math.sqrt(determinant)))
            exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

    def _get_likelihoods(self, X):
        n_samples = np.shape(X)[0]
        likelihoods = np.zeros((n_samples, self.k))
        for i in range(self.k):
            likelihoods[:, i] = self.multivariate_gaussian(X, self.parameters[i])

        return likelihoods

    def _expectation(self, X):
        weighted_likelihoods = self._get_likelihoods(X) * self.priors
        sum_likelihoods = np.expand_dims(np.sum(weighted_likelihoods, axis=1), axis=1)
        self.responsibility = weighted_likelihoods / sum_likelihoods
        self.sample_assignments = self.responsibility.argmax(axis=1)
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    def _maximization(self, X):
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = (resp * X).sum(axis=0) / resp.sum()
            covariance = (X - mean).T.dot((X - mean) * resp) / resp.sum()
            self.parameters[i]['mean'], self.parameters[
                i]['cov'] = mean, covariance

        n_samples = np.shape(X)[0]
        self.priors = self.responsibility.sum(axis=0) / n_samples

    def _converged(self, X):
        if len(self.responsibilities) < 2:
            return False

        diff = np.linalg.norm(self.responsibilities[-1] - self.responsibilities[-2])
        return diff <= self.tolerance

    def predict(self, X):
        self._init_random_gaussians(X)

        # Run EM until convergence or for max iterations
        for _ in range(self.max_iterations):
            self._expectation(X) # E-step
            self._maximization(X) # M-step

            if self._converged(X):
                break

        self._expectation(X)
        return self.sample_assignments
