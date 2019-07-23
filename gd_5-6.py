"""
Codes for Gradient Descent section
"""

import numpy as np


def E(u, v):
    """
    Awesome function description goes here
    """
    return u * np.exp(v) - 2 * v * np.exp(-u)

def E_prime_v(u, v):
    """
    Awesome function description goes here
    """
    return 2 * E(u, v) * (u * np.exp(v) - 2 * np.exp(-u))

def E_prime_u(u, v):
    """
    Awesome function description goes here
    """
    return 2 * E(u, v) * (np.exp(v) + 2 * v * np.exp(-u))

def get_closest_point(point, options):
    distances = [np.linalg.norm(point - option) for option in options]
    minimum = np.argmin(distances)
    return options[minimum]


if __name__ == '__main__':
    eta = 0.1
    x = [1, 1]

    error = 10000000
    it = 0

    while error > 10**(-14):
        error = E(*x)**2
        print(it, error)

        grad = np.array([E_prime_u(*x), E_prime_v(*x)])
        x -= eta * grad

        it += 1

    print(x)

    L = [[1, 1],
         [0.713, 0.045],
         [0.016, 0.112],
         [-0.083, 0.029],
         [0.045, 0.024]]

    print(get_closest_point(x, L))
