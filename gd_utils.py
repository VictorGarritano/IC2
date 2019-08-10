"""Implementação do Gradiente Descendente"""

import numpy as np


def E(u, v):
    """
    Superfície de erro

    Parameters
    ----------
    u: float
        primeira coordenada
    v: float
        segunda coordenada

    Returns
    -------
    float
        Valor da função E(u, v) avaliados nos pontos passados como argumento
    """
    return u * np.exp(v) - 2 * v * np.exp(-u)

def E_prime_v(u, v):
    """
    Derivada parcial de E com relação à v

    Parameters
    ----------
    u: float
        primeira coordenada
    v: float
        segund coordenada

    Returns
    -------
    float
        Valor da derivada parcial de E com relação à v nos pontos especificados
    """
    return 2 * E(u, v) * (u * np.exp(v) - 2 * np.exp(-u))

def E_prime_u(u, v):
    """
    Derivada parcial de E com relação à u

    Parameters
    ----------
    u: float
        primeira coordenada
    v: float
        segund coordenada

    Returns
    -------
    float
        Valor da derivada parcial de E com relação à u nos pontos especificados
    """
    return 2 * E(u, v) * (np.exp(v) + 2 * v * np.exp(-u))

def get_closest_point(central_point, options):
    """Calcula a distância euclidiana entre um ponto e uma lista de pontos

    Parameters
    ----------
    central_point: np.array
        Ponto para o qual queremos encontrar o correspondente de menor
        distância
    options: list
        Lista de pontos cuja distância será calculada com relação ao ponto
        central

    Returns
    -------
    np.array
        Retorna o ponto da lista de opções que apresentou a menor distãncia
        em relação ao ponto central
    """
    distances = [np.linalg.norm(central_point - option) for option in options]
    minimum = np.argmin(distances)
    return options[minimum]
