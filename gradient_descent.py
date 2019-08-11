"""Experimentos relativos ao Gradiente Descendente"""

from gd_utils import *

if __name__ == '__main__':
    # Nesses dois experimentos, o objetivo é determinar o número de iterações
    # até encontrarmos um  ponto (u*, v*) que faz com que o valor da função E
    # caia abaixo de 10**-14 (E(u*, v*) < 10**-14), partindo do ponto inicial
    # (u_0, v_0) = (1, 1) e utilizando uma taxa de aprendizado de 0.1

    print("Iniciando experimentos 11 e 12a")
    print('----')
    eta = 0.1
    x = [1, 1]

    error = 10000000
    it = 0

    while error > 10**(-14):
        error = E(*x)**2
        print("iteração: {0} --> E(u, v) = {1}".format(it, error))

        grad = np.array([E_prime_u(*x), E_prime_v(*x)])
        x -= eta * grad

        it += 1

    print("Ponto encontrado: {}".format(x))

    L = [[1, 1],
         [0.713, 0.045],
         [0.016, 0.112],
         [-0.083, 0.029],
         [0.045, 0.024]]

    # Após determinarmos o ponto desejado, o comparamos com a lista de pontos
    # pré-definida e retornamos o mais próximo
    print("Ponto mais próximo: {}".format(get_closest_point(x, L)))
    print('----'*10)

    # O último experimento é avaliar a performance da técnica conhecida como
    # "coordinate descent", onde a cada iteração calculamos o gradiente porém
    # atualizamos apenas uma das coordenadas, de maneira alternada
    print("Iniciando experimento 12b")
    print('----')

    x = [1, 1]

    for it in range(30):

        if it % 2 == 0:
            grad = np.array([E_prime_u(*x), 0])
        else:
            grad = np.array([0, E_prime_v(*x)])
        x -= eta * grad

    print("Vetor final obtido pela coordenada descendente: {0}".format(x))
    print("E(u, v) obtido pela coordenada descendente: {0}".format(E(*x)**2))
