import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean-centered:
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance, function needs samples as columns:
        cov = np.cov(X.T)

        # Eigenvalues, Eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        # Store first n eigenvectors:
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project data:
        X = X - self.mean
        return np.dot(X, self.components.T)

def main():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    x_projected = pca.transform(X)

    print('Shape of X: ', X.shape)
    print('Shape of transformed X: ', x_projected.shape)

    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
