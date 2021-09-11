from collections import Counter
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set:
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors:
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples:
        k_neighbor_labels = [self.y_train[i] for i in k_indices]

        # Most common class label:
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    k = 5
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = KNN(k=k)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
