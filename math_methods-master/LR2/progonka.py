import pprint
import numpy as np
from devtool import *


def progonka(matrix_a, matrix_b):
    vector_c, vector_d = np.zeros(len(matrix_a)), np.zeros(len(matrix_a))
    vector_c[0] = - matrix_a[0][1] / matrix_a[0][0]
    vector_d[0] = matrix_b[0] / matrix_a[0][0]
    for i in range(1, len(matrix_a) - 1):
        vector_c[i] = - matrix_a[i][i + 1] / (matrix_a[i][i - 1] * vector_c[i - 1] + matrix_a[i][i])
        vector_d[i] = (matrix_b[i] - matrix_a[i][i - 1] * vector_d[i - 1]) / (matrix_a[i][i - 1] * vector_c[i - 1] + matrix_a[i][i])

    vector_c[-1] = 0
    vector_d[-1] = (matrix_b[-1] - matrix_a[-1][-2] * vector_d[-2]) / (matrix_a[-1][-2] * vector_c[-2] + matrix_a[-1][-1])

    vector_x = np.zeros(len(matrix_a))
    vector_x[-1] = vector_d[-1]
    for i in range(len(matrix_a) - 1, 0, -1):
        vector_x[i - 1] = vector_c[i - 1] * vector_x[i] + vector_d[i - 1]
    pprint.pprint(vector_x, width=10)


#---------------------


progonka(matrix_16v_4_progonkaA, matrix_16v_4_progonkaB)
print("\n",matrix_16v_4_progonkaA)
print("\n",matrix_16v_4_progonkaB)
print("\n")
print("Невязка: r = ", nevyazka(matrix_16v_4_progonkaA, matrix_16v_4_progonkaB, [0, 0, 0,0]))
