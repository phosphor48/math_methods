import numpy as np
from devtool import *


def reverse_iter(matrix_a, eps):
    matrix_x = np.ones(len(matrix_a))
    print(matrix_x)
    a = max(matrix_x, key=abs)
    a_p = a
    itera = 0
    while True:
        matrix_x = np.linalg.inv(matrix_a) @ (matrix_x / a)
        a = max(matrix_x, key=abs)
        print(matrix_x)
        itera = itera + 1
        if abs(a - a_p) < eps:
            itera=itera+1
            break
        a_p = a
    print("Minimum eigenvalue by abs: ", 1 / a)

    print("Итерации: ", itera)
reverse_iter(matrix_10v, 0.0001)
