import numpy as np
import pandas as pd


class Node:
    def __init__(self, x, y, indices, min_leaf=5, depth=10):
        self.x = x
        self.y = y
        self.indices = indices
        self.depth = depth
        self.min_leaf = min_leaf
        self.row_count = len(indices)
        self.col_count = x.shape[1]
        self.val = self.compute_gamma(y[self.indices])
        self.score = float('-inf')
        self.find_varsplit()

    def find_varsplit(self):
        for c in range(self.col_count):
            self.find_greedy_split(c)

        if self.is_leaf:
            return

        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.indices[lhs], self.min_leaf, depth=self.depth - 1)
        self.rhs = Node(self.x, self.y, self.indices[rhs], self.min_leaf, depth=self.depth - 1)

    def find_greedy_split(self, var_idx):
        x = self.x.values[self.indices, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf:
                continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def gain(self, lhs, rhs):
        gradient = self.y[self.indices]

        lhs_gradient = gradient[lhs].sum()
        lhs_n_intances = len(gradient[lhs])

        rhs_gradient = gradient[rhs].sum()
        rhs_n_intances = len(gradient[rhs])

        gain = ((lhs_gradient ** 2 / (lhs_n_intances)) + (rhs_gradient ** 2 / (rhs_n_intances)) - ((lhs_gradient + rhs_gradient) ** 2 / (lhs_n_intances + rhs_n_intances)))
        return gain

    @staticmethod
    def compute_gamma(gradient):
        return np.sum(gradient) / len(gradient)

    @property
    def split_col(self):
        return self.x.values[self.indices, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)


class DecisionTreeRegressor:
    def fit(self, X, y, min_leaf=5, depth=5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf, depth)
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)


class GradientBoostingClassification:
    def __init__(self):
        self.estimators = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def negativeDerivitiveLogloss(self, y, log_odds):
        p = self.sigmoid(log_odds)
        return(y - p)

    @staticmethod
    def log_odds(column):
        if isinstance(column, pd.Series):
            binary_yes = np.count_nonzero(column.values == 1)
            binary_no  = np.count_nonzero(column.values == 0)
        elif isinstance(column, list):
            column = np.array(column)
            binary_yes = np.count_nonzero(column == 1)
            binary_no  = np.count_nonzero(column == 0)
        else:
            binary_yes = np.count_nonzero(column == 1)
            binary_no  = np.count_nonzero(column == 0)

        value = np.log(binary_yes / binary_no)
        return np.full((len(column), 1), value).flatten()

    def fit(self, X, y, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=5):
        self.learning_rate = learning_rate
        self.base_pred = self.log_odds(y)

        for booster in range(boosting_rounds):
            pseudo_residuals = self.negativeDerivitiveLogloss(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X=X, y=pseudo_residuals, depth=5, min_leaf=5)
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        return self.log_odds(y) + pred


class GradientBoostingRegressor:
    def __init__(self, classification=False):
        self.estimators = []

    @staticmethod
    def MeanSquaredError(y, y_pred):
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def negativeMeanSquaredErrorDerivitive(y, y_pred):
        return 2 * (y - y_pred)

    def fit(self, X, y, depth=5, min_leaf=5, learning_rate=0.1, boosting_rounds=5):
        self.learning_rate = learning_rate
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten()

        for booster in range(boosting_rounds):
            pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
            boosting_tree = DecisionTreeRegressor().fit(X=X, y=pseudo_residuals, depth=5, min_leaf=5)
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        return np.full((X.shape[0], 1), np.mean(y)).flatten() + pred
