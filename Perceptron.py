import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                h_raw = np.dot(x, self.weights) + self.bias
                y_pred = self._unit_step_func(h_raw)

                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x
                self.bias += update

    def predict(self, X):
        h_raw = np.dot(X, self.weights) + self.bias
        y_pred = self._unit_step_func(h_raw)
        return y_pred

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def main():
    X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.05)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = Perceptron(learning_rate=0.01, n_iters=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)

    x0_1 = np.amin(x_train[:, 0])
    x0_2 = np.amax(x_train[:, 0])

    x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
    x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    y_min = np.amin(x_train[:, 1])
    y_max = np.amax(x_train[:, 1])
    ax.set_ylim([y_min - 3, y_max + 3])

    plt.show()

if __name__ == '__main__':
    main()
