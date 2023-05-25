import numpy as np
from devtool import *


def simple_iteration(matrix_a, eps):
    matrix_x = [1] * len(matrix_a)
    print(matrix_x)
    matrix_y = matrix_a @ matrix_x
    matrix_l = matrix_y @ matrix_x
    print(matrix_l)
    matrix_x = matrix_y / np.linalg.norm(matrix_y)
    matrix_lp = matrix_l
    print(matrix_lp)
    itera=0
    while True:
        matrix_y = matrix_a @ matrix_x
        matrix_l = matrix_y @ matrix_x
        matrix_x = matrix_y / np.linalg.norm(matrix_y)
        itera = itera+1
        if abs(matrix_l - matrix_lp) < eps:
            itera=itera+1
            break
        matrix_lp = matrix_l
        print(matrix_lp)
    print("Maximum eigenvalue by abs: ", matrix_l)

    print("Итерации: ", itera)
simple_iteration(matrix_10v, 0.0001)