import numpy as np
from devtool import *


def decompose_llt(matrix_a, matrix_b):
    matrix_l = np.matrix(np.zeros([matrix_a.shape[0], matrix_a.shape[1]]))
    c = 0
    for i in range(len(matrix_a)):
        c += 1
        for j in range(len(matrix_a)):
            if i == 0 and j == 0:
                matrix_l[0, 0] = np.sqrt(abs(matrix_a[i, j]))
            elif i == j:
                sum_ = 0
                for k in range(i):
                    sum_ += matrix_l[i, k] ** 2
                matrix_l[i, j] = np.sqrt(abs(matrix_a[i, j] - sum_))
            elif j == 0:
                matrix_l[i, j] = matrix_a[i, j] / matrix_l[j, j]
            else:
                if j <= i:
                    sum_ = 0
                    for k in range(i):
                        sum_ += matrix_l[i, k] * matrix_l[j, k]
                    matrix_l[i, j] = (matrix_a[i, j] - sum_) / matrix_l[j, j]
    lt = np.transpose(matrix_l)
    vector_y, vector_x = np.zeros(len(matrix_a)), np.zeros(len(matrix_a))
    vector_y[0] = matrix_b[0] / matrix_l[0, 0]
    for i in range(len(matrix_a)):
        sum_ = 0
        for k in range(i):
            sum_ += matrix_l[i, k] * vector_y[k]
        vector_y[i] = (matrix_b[i] - sum_) / matrix_l[i, i]
    vector_x[-1] = vector_y[-1]
    for i in range(len(matrix_a) - 1, -1, -1):
        sum_ = 0
        for k in range(i + 1, len(matrix_a)):
            sum_ += matrix_l[k, i] * vector_x[k]
        vector_x[i] = (vector_y[i] - sum_) / matrix_l[i, i]
    return vector_x


#---------------------


print(matrix_16v_1_3_g_l_l_A)
print(decompose_llt(matrix_16v_1_3_g_l_l_A, matrix_16v_1_3_g_l_l_B))
print("Невязка: r = ", nevyazka(matrix_16v_1_3_g_l_l_A, matrix_16v_1_3_g_l_l_B, [0, 0, 0, 0]))
