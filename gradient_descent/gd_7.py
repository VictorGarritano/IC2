"""
Codes for Gradient Descent section
"""

import numpy as np


def E(u, v):
    """
    Surface
    """
    return u * np.exp(v) - 2 * v * np.exp(-u)

def E_prime_v(u, v):
    """
    Partial derivative of E w.r.t. v
    """
    return 2 * E(u, v) * (u * np.exp(v) - 2 * np.exp(-u))

def E_prime_u(u, v):
    """
    Partial derivative of E w.r.t. u
    """
    return 2 * E(u, v) * (np.exp(v) + 2 * v * np.exp(-u))

def get_closest_point(central_point, options):
    """
    Return the closest point of central_points considering the options
    """
    distances = [np.linalg.norm(central_point - option) for option in options]
    minimum = np.argmin(distances)
    return options[minimum]


if __name__ == '__main__':
    eta = 0.1
    x = [1, 1]

    error = 10000000

    for it in range(30):

        if it % 2 == 0:
            grad = np.array([E_prime_u(*x), 0])
        else:
            grad = np.array([0, E_prime_v(*x)])
        x -= eta * grad

    print(x)
    print(E(*x)**2)

