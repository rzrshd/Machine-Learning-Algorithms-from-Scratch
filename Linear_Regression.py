import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent:
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def main():
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    model = LinearRegression(learning_rate=0.01, n_iters=10000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    y_pred_line = model.predict(X)

    mse = mean_squared_error(y_test, predictions)
    print(f'MSE: {mse}')

    accuracy = r2_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')

    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()

if __name__ == '__main__':
    main()
