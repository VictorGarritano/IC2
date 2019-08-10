"""Código relativo à implementação do Perceptron"""

import numpy as np

from datasets import Dataset


class Perceptron():
    """Classe base do Perceptron.

    Essa classe define a função de ativação, assim como o processo de
    otimização dos pesos
    """

    def __init__(self):
        """
        Construtor para o perceptron.

        Define o vetor de pesos w, a ser aprendido
        pelo processo de otimização.
        """
        self.weights_vector = None

    @staticmethod
    def _activation_function(sample, weights_vector):
        """Função de ativação do perceptron. Realiza o produto escalar entre um
        exemplo e o vetor de pesos e retorna o sinal do resultado

        Parameters
        ----------
        sample: np.array
            Array que representa um ponto no espaço
        weights_vector:
            Array que representa o vetor de pesos

        Returns
        -------
        float
            sinal do produto escalar entre o exemplo e o vetor de pesos
        """
        return np.sign(sample @ weights_vector)

    def run_experiment(self, N):
        """Realiza o processo de treinamento do perceptron. A cada iteração,
        é gerado o conjunto de pontos classificados incorretamente. Um ponto
        desse conjunto é selecionado aleatoriamente e o vetor de pesos é
        atualizado de acordo com a regra do perceptron

        Parameters
        ----------

        N: int
            Quantidade de pontos a ser gerada no conjunto de dados

        Returns
        -------
        it: int
            Quantidade de iterações realizadas até todos os pontos serem
            corretamente classificados
        mismatch_prob: float
            Percentual de pontos classificados de maneira incorreta numa
            amostra não utilizada no treinamento, gerada de maneira aleatória
        """
        it = 0
        self.weights_vector = np.zeros(3)
        dataset = Dataset()
        df, coeff_vector = dataset.create_dataframe(N)

        while True:
            y_pred = self._activation_function(df[['x1', 'x2', 'bias']],
                                               self.weights_vector)
            check_y_pred_vs_y_true = np.equal(y_pred, df['y'].values)
            wrong_examples = np.where(check_y_pred_vs_y_true == False)[0]

            if wrong_examples.shape[0] == 0:
                test_points = Dataset.generate_data(1000)
                y_true = self._activation_function(test_points, coeff_vector)
                y_pred = self._activation_function(test_points,
                                                   self.weights_vector)

                mistmatch_prob = np.mean(y_true != y_pred)

                return [it, mistmatch_prob]

            random_example_idx = np.random.choice(wrong_examples)
            random_example = df[['x1', 'x2', 'bias']].iloc[random_example_idx]

            self.weights_vector += random_example.values * \
                df['y'].iloc[random_example_idx]
            it += 1

    def run_epoch(self, num_epochs, N):
        """Realiza uma sequência de execuções do processo de treinamento
        do Perceptron

        Parameters
        ----------
        num_epochs: int
            Quantidade de execuções
        N: int
            Tamanho do conjunto de treinamento utilizado em cada execução

        Returns
        -------
        list
            Uma lista contendo a quantidade de iterações e o percentual de
            exemplos de teste classificados incorretamente para cada execução
        """
        return [self.run_experiment(N) for _ in np.arange(num_epochs)]
