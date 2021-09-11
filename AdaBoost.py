import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        feature = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[feature < self.threshold] = -1
        else:
            predictions[feature > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Greedy search to find best threshold and feature:
            for feature_idx in range(n_features):
                feature = X[:, feature_idx]
                thresholds = np.unique(feature)

                for threshold in thresholds:
                    p = 1 # Polarity
                    predictions = np.ones(n_samples)
                    predictions[feature < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_idx
                        min_error = error

            eps = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))

            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        y_pred = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(y_pred, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = Adaboost(n_clf=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
