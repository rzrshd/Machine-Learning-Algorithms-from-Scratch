from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Greedily select the best split according to information gain:
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)

        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_index = None
        split_threshold = None
        for feature_index in feature_indices:
            feature = X[:, feature_index]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                gain = self._information_gain(y, feature, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _information_gain(self, y, feature, split_threshold):
        # Parent loss:
        parent_entropy = entropy(y)

        # Generate split:
        left_indices, right_indices = self._split(feature, split_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Compute the weighted average of the loss for children:
        child_entropy = (len(left_indices) / len(y)) * entropy(y[left_indices]) + (len(right_indices) / len(y)) * entropy(y[right_indices])

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, feature, split_threshold):
        left_indices = np.argwhere(feature <= split_threshold).flatten()
        right_indices = np.argwhere(feature > split_threshold).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = DecisionTree(max_depth=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
