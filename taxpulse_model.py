import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ManualGaussianNB(ClassifierMixin, BaseEstimator):
    """Gaussian Naive Bayes implemented with NumPy.

    Kept pickle-compatible with the original project: attribute names
    (classes_, class_prior_, theta_, var_) are unchanged.
    """

    def __init__(self, var_smoothing=1e-7):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        epsilon = self.var_smoothing * max(np.var(X, axis=0).max(), 1.0)

        counts, priors, means, variances = [], [], [], []
        for class_value in self.classes_:
            X_class = X[y == class_value]
            counts.append(len(X_class))
            priors.append(len(X_class) / len(X))
            means.append(np.mean(X_class, axis=0))
            variances.append(np.var(X_class, axis=0) + epsilon)

        self.class_count_ = np.asarray(counts)
        self.class_prior_ = np.asarray(priors)
        self.theta_ = np.asarray(means)
        self.var_ = np.asarray(variances)
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        scores = []
        for i in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[i])
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[i])
                + ((X - self.theta_[i]) ** 2 / self.var_[i]),
                axis=1,
            )
            scores.append(log_prior + log_likelihood)
        return np.column_stack(scores)

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
