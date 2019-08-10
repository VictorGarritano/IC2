"""Implementação da Regressão Logística"""

import numpy as np


class LogisticRegression():
    """Classe que implementa o Regressão Logística, assim como seu processo
    de otimização
    """

    def __init__(self, lr):
        """Construtor para a Regressão Logística

        Parameters
        ----------
        lr: float
            Taxa de aprendizado
        """
        self.lr = lr

    def run_experiment(self, N, dataset):
        """Roda o processo de otimização da regressão logística, calculando
        o gradiente e realizando a atualização dos pesos a cada iteração

        Quando a norma da diferença entre dois vetores de peso consecutivos
        for menos que 0.01, estabelecemos a convergência do algoritmo, e
        calculamos o erro fora da amostra para um conjunto de 1000 pontos de
        teste

        Parameters
        ----------
        N: int
            Quantidade de exemplos de treino a ser gerada
        dataset: Dataset
            Tipo de Dataset a ser utilizado

        Returns
        -------
        list
            Lista contendo o número de épocas de treinamento necessárias até
                a convergência ser atingida e o valor do erro fora da amostra
        """

        df, coeff_vector = dataset.create_dataframe(N)

        X = df[['x1', 'x2', 'bias']].values
        y = df['y'].values
        N = X.shape[0]
        w_old = np.zeros(X.shape[1])

        max_epochs = 10**6

        for epoch in range(max_epochs):
            X_y = np.concatenate((X, y[:, np.newaxis]), axis=1)
            np.random.shuffle(X_y)

            _X = X_y[:, :-1]
            _y = X_y[:, -1]

            num = _y[:, np.newaxis] * _X
            den = 1 + np.exp(_y * (_X @ w_old))
            quotient = num / den[:, np.newaxis]

            grad_E_in = - (1/N) * np.sum(quotient, axis=0)

            w_new = w_old - N * self.lr * grad_E_in

            if np.linalg.norm(w_new - w_old) < 0.01:
                N_test = 1000
                test_points = dataset.generate_data(N_test)
                y_true = np.sign(test_points @ coeff_vector)

                E_out = (1/N_test) * np.sum(
                    np.log(1 + np.exp(- y_true * (test_points @ w_new)))
                )

                return [epoch, E_out]

            w_old = w_new

        return max_epochs

    def run_epoch(self, runs, N, dataset):
        """Roda uma sequência de execuções algoritmo de otimização da
        Regressão Logística

        Parameters
        ----------
        runs: int
            Número de execuções a serem executadas
        N: int
            Quantidade de exemplos de treino a cada execução
        dataset:
            Tipo de Dataset utilizado

        Returns
        -------
        list
            Lista contendo o número de épocas e o erro fora da amostra para
            cada execução
        """
        return [self.run_experiment(N, dataset)
                for _ in np.arange(runs)]
