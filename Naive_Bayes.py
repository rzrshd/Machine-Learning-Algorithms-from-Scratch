import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            x_c = X[y == c]
            self._mean[idx, :] = x_c.mean(axis=0)
            self._var[idx, :] = x_c.var(axis=0)
            self._priors[idx] = x_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class:
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = NaiveBayes()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
