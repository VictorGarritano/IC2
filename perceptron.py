"""Perceptron class will be defined here"""

import numpy as np

from utils import Dataset


class Perceptron(Dataset):
    """Docstring for Perceptron. """

    def __init__(self):
        """
        Awesome method description goes here
        """
        super(Perceptron, self).__init__()
        self.w = None

    @staticmethod
    def _activation_function(x, w):
        """
        Awesome method description goes here
        """
        return np.sign(x @ w)

    def run_experiment(self, N):
        """
        Run the experiment and return number of iterations and out-of-sample
        error
        """
        it = 0
        self.w = np.zeros(3)
        dataset = Dataset()
        df, coeff_vector = dataset.create_dataframe(N)

        while True:
            y_pred = self._activation_function(df[['x1', 'x2', 'bias']],
                                               self.w)
            check_y_pred_vs_y_true = np.equal(y_pred, df['class'].values)
            wrong_examples = np.where(check_y_pred_vs_y_true == False)[0]

            if wrong_examples.shape[0] == 0:
                test_points = Dataset.generate_data(1000)
                y_true = self._activation_function(test_points, coeff_vector)
                y_pred = self._activation_function(test_points, self.w)

                mistmatch_prob = np.mean(y_true != y_pred)

                return [it, mistmatch_prob]

            random_example_idx = np.random.choice(wrong_examples)
            random_example = df[['x1', 'x2', 'bias']].iloc[random_example_idx]

            self.w += random_example.values * \
                df['y'].iloc[random_example_idx]
            it += 1

    def run_epoch(self, num_epochs, N):
        """
        Run experiment for num_epochs epochs, using N data points
        """
        return [self.run_experiment(N) for _ in np.arange(num_epochs)]
