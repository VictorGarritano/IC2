"""Classes que definem os conjuntos de dados utilizados no experimentos"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Dataset():

    """Classe Base para o Dataset.

    Contém métodos que serão úteis para a definição de Datasets com alguma
    modificação especial para determinados casos
    """

    def __init__(self):
        """
        Construtor do Dataset
        """
        self.coeff_vector = None

    @staticmethod
    def generate_data(N):
        """Gera um conjunto de dados de dimensão N X 3 (com a última dimensão
        composta unicamente por valores 1). Os valores das duas primeiras
        dimensões são geradas a partir de uma distribuição uniforme entre
        -1 e 1

        Parameters
        ----------
        N: int
            Quantidade de pontos gerada

        Returns
        -------
        np.array
            Matriz N X 3 onde cada linha representa cada um dos N pontos da
            base de dados
        """
        return np.concatenate(
            (np.random.uniform(low=-1, size=(N, 2)),
             np.ones(N).reshape(N, 1)),
            axis=1
        )

    def _define_line(self):
        """Define os coeficientes da forma padrão da linha que estabelece a
        separação entre as regiões de valor +1 e -1 no plano

        São selecionados 2 pontos de dimensão 2 aleatoriamente com
        distribuição uniforme entre -1 e 1. Após isso, o coeficiente angular
        m da reta ligando esses pontos é determinada

        A partir da equação da reta -m * x1 + y1 = b, podemos obter a forma
        padrão Ax + By = C, onde:
        A = -m
        B = 1
        C = m * x1 - y1
        """
        x = np.random.uniform(low=-1, size=(2, 2))
        m = (x[0][1] - x[1][1]) / (x[0][0] - x[1][0])
        coeff_vector = np.array([-m, 1, m * x[0][0] - x[0][1]])

        self.coeff_vector = coeff_vector

    def create_dataframe(self, N):
        """Gera um dataframe com os exemplos e seus respectivos rótulos, onde
        cada dimensão identificada é identificada por um nome conveniente

        Parameters
        ----------
        N: int
            Quantidade de exemplos na base de dados a serem gerados

        Returns
        -------
        df: pd.DataFrame
            DataFrame com os pontos e seus respectivos rótulos
        coeff_vector: np.array
            Array com os coeficientes da forma padrão da reta que foi
            utilizada para definir as regiões com valores +1 e -1
        """
        data = self.generate_data(N)
        self._define_line()

        df = pd.DataFrame(data=data, columns=['x1', 'x2', 'bias'])
        df['y'] = np.sign(
            df[['x1', 'x2', 'bias']].values @ self.coeff_vector)

        return df, self.coeff_vector

    @staticmethod
    def plot_data(df):
        """Função auxiliar para visualização de uma base de dados

        Parameters
        ----------
        df: pd.DataFrame
            Base de dados, com colunas nomeadas como 'x1', 'x2' e 'y'
        """
        sns.set_style('whitegrid')
        plt.figure(figsize=(8, 8))

        sns.scatterplot(x='x1', y='x2', data=df, hue='y',
                        legend='full', s=80)

        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

    @staticmethod
    def generate_results_dataframe(results, column_names):
        """Função auxiliar para geração de um DataFrame com as estatísticas
        computadas após uma série de execuções de um experimento

        Parameters
        ----------
        results: list
            Lista de resultados contendo os valores computados para cada
            execução
        column_names: list
            Lista que define o nome de cada um dos resultados de uma execução

        Returns
        -------
        DataFrame
            DataFrame com as estatíticas de cada uma das execuções
        """
        return pd.DataFrame(data=results, columns=column_names)


class DatasetNoiseTarget(Dataset):

    """Dataset onde 10% dos exemplos terão seus labels com valores invertidos,
    com a finalidade de simular uma função objetivo não-determinística, dada a
    presença de um ruído adicionado artificialmente
    """

    def __init__(self):
        """
        Construtor do DatasetNoiseTarget
        """
        super(DatasetNoiseTarget, self).__init__()

    @staticmethod
    def generate_data(N):
        """Gera um conjunto de dados de dimensão N X 3 (com a primeira
        dimensão composta unicamente por valores 1). Os valores das
        duas primeiras dimensões são geradas a partir de uma
        distribuição uniforme entre -1 e 1

        Parameters
        ----------
        N: int
            Quantidade de pontos gerada

        Returns
        -------
        np.array
            Matriz N X 3 onde cada linha representa cada um dos N pontos da
            base de dados
        """
        return np.concatenate(
            (np.ones(N).reshape(N, 1),
            np.random.uniform(low=-1, size=(N, 2))),
            axis=1
        )

    @staticmethod
    def define_noise_targets(x):
        """Define os rótulos de cada um dos pontos na base de dados, e após
        isso seleciona-se 10% desses, que terão seus rótulos invertidos
        (de 1 para -1 e de -1 para 1)

        Parameters
        ----------
        x: np.array
            Matriz que representa a base de dados

        Returns
        -------
        y: np.array
            Rótulos para cada um dos pontos na base de dados, com ruído
            adicionado
        """
        y = np.sign(x[:, 1]**2 + x[:, 2]**2 - 0.6)

        noise_amount = int(0.1 * x.shape[0])
        noise_idxs = np.random.choice(x.shape[0], noise_amount, replace=False)

        y[noise_idxs] *= -1

        return y

    def create_dataframe(self, N):
        """Gera um dataframe com os exemplos e seus respectivos rótulos, onde
        cada dimensão identificada é identificada por um nome conveniente

        Parameters
        ----------
        N: int
            Quantidade de exemplos na base de dados a serem gerados

        Returns
        -------
        df: pd.DataFrame
            DataFrame com os pontos e seus respectivos rótulos
        None
            Retorna um segundo argumento para fins de compatibilidade com
            métodos da classe pai
        """

        data = self.generate_data(N)
        y = self.define_noise_targets(data)

        df = pd.DataFrame(data=data, columns=['bias', 'x1', 'x2'])
        df['y'] = y

        return df, None


class NonLinearDataset(DatasetNoiseTarget):
    """Classe que gera um vetor de atributos não-linear,  definidos como:
    (1, x_1, x_2, x_1*x_2, x_1**2, x_2**2)
    """

    def __init__(self):
        """Construtor para o Dataset com atributos não-lineares """
        super(NonLinearDataset, self).__init__()

    def create_dataframe(self, N):
        """Gera um conjunto de dados com atributos não-lineares,
        bem como seus respectivos rótulos, onde cada dimensão é identificada
        por um nome conveniente

        Parameters
        ----------
        N: int
            Quantidade de exemplos na base de dados a serem gerados

        Returns
        -------
        df: pd.DataFrame
            DataFrame com os pontos e seus respectivos rótulos
        """

        data = self.generate_data(N)
        y = self.define_noise_targets(data)

        df = pd.DataFrame(data=data, columns=['bias', 'x1', 'x2'])
        df['x1*x2'] = df['x1'].values * df['x2'].values
        df['x1^2'] = np.square(df['x1'].values)
        df['x2^2'] = np.square(df['x2'].values)
        df['y'] = y

        return df
