import numpy as np


class LinearRegression:
    """
    A linear regression model that uses a closed form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model for the given input using a closed form solution.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input labels.

        Returns:
            np.ndarray: None.

        """
        n = X.shape[0]
        X = np.append(X, np.ones((n, 1)), axis=1)  # Append column for bias
        y = y.reshape(n, 1)
        self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
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
        """
        Fit the model for the given input using gradient descent.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input labels.

        Returns:
            np.ndarray: None.

        """
        self.w = 0
        self.b = 0
        self.n = len(X)
        for i in range(epochs):
            y_pred = self.w * X + self.b
            dm = (-2 / self.n) * sum(X * (y - y_pred))
            db = (-1 / self.n) * sum(y - y_pred)
            self.w = self.w - dm * lr
            self.b = self.b - db * lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w * X + self.b
