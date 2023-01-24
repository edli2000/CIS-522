import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        pass

    def fit(self, X, y):
        n = X.shape[0]
        X = np.append(X, np.ones((n, 1)), axis=1)  # Append column for bias
        y = y.reshape(n, 1)
        self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def predict(self, X):
        n = X.shape[0]
        X = np.append(X, np.ones((n, 1)), axis=1)
        return np.dot(X, self.theta)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        self.m = 0
        self.b = 0
        self.n = len(X)
        for i in range(epochs):
            y_pred = self.m * X + self.b
            dm = (-2 / self.n) * sum(X * (y - y_pred))
            db = (-1 / self.n) * sum(y - y_pred)
            self.m = self.m - dm * lr
            self.b = self.b - db * lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.m * X + self.b
