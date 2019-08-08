""""Linear Regression class will be defined here"""

import numpy as np

from utils import Dataset


class LinearRegression(Dataset):
    """
    Missing documentation
    """

    def __init__(self):
        """
        MISSING DOCUMENTATION
        """
        super(LinearRegression, self).__init__()

    def run_experiment(self, N):
        """
        MISSING DOCUMENTATION
        """

        dataset = Dataset()
        df, coeff_vector = dataset.create_dataframe(N)

        X = df[['x1', 'x2', 'bias']].values
        y = df['y'].values

        pseudo_inverse = lambda X: np.linalg.inv(X.T @ X) @ X.T
        X_pinv = pseudo_inverse(X)

        w = X_pinv @ y
        y_pred = np.sign(X @ w)

        E_in = np.mean(y != y_pred)

        test_points = dataset.generate_data(1000)
        y_true = np.sign(test_points @ coeff_vector)
        y_pred = np.sign(test_points @ w)

        E_out = np.mean(y_true != y_pred)

        return [E_in, E_out]

    def run_epoch(self, num_epochs, N):
        """
        Run experiment for num_epochs, using N data points
        """
        return [self.run_experiment(N) for _ in np.arange(num_epochs)]
