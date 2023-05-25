import numpy as np
from devtool import *


def direct_iter(matrix_a, eps):
    matrix_x = np.ones(len(matrix_a))
    print(matrix_x)
    a = max(matrix_x, key=abs)
    a_p = a
    itera = 0
    while True:
        matrix_x = matrix_a @ (matrix_x / a)
        a = max(matrix_x, key=abs)
        print(a)
        itera = itera+1
        if abs(a - a_p) < eps:
            itera = itera+1
            break
        a_p = a
    print("Maximum eigenvalue by abs: ", a)

    print("Итерации: ", itera)
print(direct_iter(matrix_10v, 0.0001))
