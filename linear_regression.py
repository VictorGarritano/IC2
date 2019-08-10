""""Implementação da regressão linear"""

import numpy as np

from datasets import Dataset, DatasetNoiseTarget


class LinearRegression():
    """Define o algoritmo utilizado no processo de otimização do vetor normal
    à reta que realiza a separação linear entre duas regiões via Regressão
    Linear
    """

    def __init__(self):
        """
        Construtor para a LinearRegression
        """
        pass

    def run_experiment(self, N, dataset, test):
        """Realiza os experimentos relativos a regressão linear

        Uma base de dados de dimensão N X 3 é criada e a partir dela o vetor
        de pesos ótimo é calculado, levando em consideração o algoritmo
        definido para a regressão linear

        É calculado o erro dentro da amostra e posteriormente é gerado um novo
        conjunto com 1000 pontos de teste, sobre o qual o erro fora da amostra
        é calculado de maneira análoga

        Parameters
        ----------
        N: int
            Quantidade de exemplos de treino a ser gerada
        dataset: Dataset
            Tipo de dataset a ser utilizado no experimento
        test: bool
            Se devemos realizar ou não o cálculo do erro fora da amostra

        Returns
        -------
        list
            Retorna os erros dentro e fora da amostra, caso esse último tenha
            sido calculado
        """

        df, coeff_vector = dataset.create_dataframe(N)

        X = df[['x1', 'x2', 'bias']].values
        y = df['y'].values

        pseudo_inverse = lambda X: np.linalg.inv(X.T @ X) @ X.T
        X_pinv = pseudo_inverse(X)

        w = X_pinv @ y
        y_pred = np.sign(X @ w)

        E_in = np.mean(y != y_pred)

        if test:
            test_points = dataset.generate_data(1000)
            y_true = np.sign(test_points @ coeff_vector)
            y_pred = np.sign(test_points @ w)

            E_out = np.mean(y_true != y_pred)

            return [E_in, E_out]
        else:
            return E_in

    def run_epoch(self, num_epochs, N, dataset, test=True):
        """Executa uma sequência de rodadas do experimento da regressão linear

        Parameters
        ----------
        num_epochs: int
            Número de rodadas a serem executadas
        N: int
            Quantidade de exemplos a serem gerados em cada experimento
        dataset: Dataset
            Tipo de Dataset a ser utilizado no experimento
        test: bool (opcional)
            Se devemos ou não calcular o erro fora da amostra (padrão é True)

        Returns
        -------
        list
            Retorna os erros dentro e fora da amostra para cada uma das
            rodadas
        """
        return [self.run_experiment(N, dataset, test)
                for _ in np.arange(num_epochs)]


class LinearRegressionPerceptron(LinearRegression):
    """Versão modificada da Regressão Linear onde, após a determinação
    do melhor vetor de pesos via regressão linear, utilizamos esse resultado
    como ponto de partida para o algoritmo do perceptron
    """

    def __init__(self):
        super(LinearRegressionPerceptron, self).__init__()

    def run_experiment(self, N, dataset, test):
        """Roda o experimento da regressão linear e em sequência o perceptron
        tomando como ponto de partida a solução encontrada pela regressão

        Parameters
        ----------
        N: int
            Quantidade de exemplos de treino a serem gerados
        dataset: Dataset
            Tipo de Dataset a ser utilizado
        test: bool
            Nesse caso não utilizaremos esse argumento, entretanto ele está
            presente para fins de compatibilidade com métodos da classe pai

        Returns
        -------
        it: int
            Número de iterações necessárias para o perceptron separar os dados
            completamente
        """
        df, coeff_vector = dataset.create_dataframe(N)

        X = df[['x1', 'x2', 'bias']].values
        y = df['y'].values

        pseudo_inverse = lambda X: np.linalg.inv(X.T @ X) @ X.T
        X_pinv = pseudo_inverse(X)

        w = X_pinv @ y

        h = lambda x, w: np.sign(x @ w)
        it = 0

        while True:
            y_pred = h(df[['x1', 'x2', 'bias']], w)
            check_y_pred_vs_y_true = np.equal(y_pred, df['y'].values)
            wrong_examples = np.where(check_y_pred_vs_y_true == False)[0]

            if wrong_examples.shape[0] == 0:
                return it

            random_example_idx = np.random.choice(wrong_examples)
            random_example = df[['x1', 'x2', 'bias']].iloc[random_example_idx]

            w += random_example.values * df['y'].iloc[random_example_idx]
            it += 1


class NonLinearRegression(LinearRegression):
    """Versão da Regressão Linear que utiliza como entrada um conjunto de
    dados com atributos não-lineares
    """

    def __init__(self):
        """
        Construtor para a Regressão Não-Linear
        """
        super(NonLinearRegression, self).__init__()

    def run_experiment(self, N, dataset, test):
        """Roda o experimento da Regressão com a base de dados não-linear

        Após a determinação do conjunto ótimo de pesos, é calculada a
        similaridade da solução encontrada com uma série de hipóteses
        pré-definidas pelo experimento, da seguinte maneira: após a obtenção
        do vetor w, geramos a classificação (y_pred) para cada um dos pontos,
        realizando o produto escalar entre X e w, e tomamos o sinal de cada
        resultado individual

        Realizamos esse mesmo procedimento considerando cada um dos pesos
        pré-definidos e finalmente calculamos o percentual de concordância
        entre os rótulos fornecidos por y_pred e os rótulos de cada uma das
        hipóteses

        Parameters
        ----------
        N: int
            Quantidade de exemplos de treino a ser gerada
        dataset:
            Tipo de dataset a ser utilizado
        test:
            Nesse caso, não utilizamos esse argumento, porém ele é mantido
            para fins de compatibilidade com métodos da classe pai

        Returns
        -------
        list:
            Lista com o percentual de concordância entre o vetor de pesos
            correspondente à solução da regressão linear e cada uma das
            hipóteses pré-definida, e o erro fora da amostra (caso tenhamos
            optado por calculá-lo)
        """

        df = dataset.create_dataframe(N)

        X = df[['bias', 'x1', 'x2', 'x1*x2', 'x1^2', 'x2^2']].values
        y = df['y'].values

        pseudo_inverse = lambda X: np.linalg.inv(X.T @ X) @ X.T
        X_pinv = pseudo_inverse(X)

        w = X_pinv @ y

        y_pred = np.sign(X @ w)

        w_a = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
        w_b = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
        w_c = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
        w_d = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
        w_e = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])

        ws = [w_a, w_b, w_c, w_d, w_e]

        disagreements = []

        for _w in ws:
            y = np.sign(X @ _w)
            disagreements.append(np.mean(y_pred != y))

        if test:
            df_test = dataset.create_dataframe(1000)
            y_true = df_test['y'].values

            X_test = df_test[
                ['bias', 'x1', 'x2', 'x1*x2', 'x1^2', 'x2^2']].values
            y_pred = np.sign(X_test @ w)

            E_out = np.mean(y_true != y_pred)

            return [disagreements, E_out]

        else:
            return disagreements
