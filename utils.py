"""Base classes will be defined here"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Dataset():

    """Docstring for Dataset. """

    def __init__(self):
        """
        MISSING DOCUMENTATION
        """
        self.coeff_vector = None

    @staticmethod
    def generate_data(N):
        """
        MISSING DOCUMENTATION
        """
        return np.concatenate(
            (np.random.uniform(low=-1, size=(N, 2)),
             np.ones(N).reshape(N, 1)),
            axis=1
        )

    def _define_line(self):
        """
        Awesome method description goes here
        """
        x = np.random.uniform(low=-1, size=(2, 2))
        m = (x[0][1] - x[1][1]) / (x[0][0] - x[1][0])
        coeff_vector = np.array([-m, 1, m * x[0][0] - x[0][1]])

        self.coeff_vector = coeff_vector

    def create_dataframe(self, N):
        """
        MISSING DOCUMENTATION
        """
        data = self.generate_data(N)
        self._define_line()

        df = pd.DataFrame(data=data, columns=['x1', 'x2', 'bias'])
        df['y'] = np.sign(
            df[['x1', 'x2', 'bias']].values @ self.coeff_vector)

        return df, self.coeff_vector

    @staticmethod
    def plot_data(df):
        """
        MISSING DOCUMENTATION
        """
        sns.set_style('whitegrid')
        plt.figure(figsize=(8, 8))

        sns.scatterplot(x='x', y='y', data=df, hue='class',
                        legend='full', s=80)

        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

    @staticmethod
    def generate_results_dataframe(results, column_names):
        """
        MISSING DOCUMENTATION
        """
        return pd.DataFrame(data=results, columns=column_names)
