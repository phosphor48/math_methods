import numpy as np
from devtool import *


def zeydel(A, B, eps):
    n = len(A)
    X = [0] * n
    count =0
    while nevyazka(A, B, X) > eps:
        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i][j] / A[i][i] * X[j]
            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i][j] / A[i][i] * X[j]

            X[i] = B[i] / A[i][i] - sum1 - sum2
            count += 1
            print(X)
    print("X:", X)
    print("epsilon =", eps)
    print("Невязка: r = ", nevyazka(A, B, X))
    print(count)

#---------------------

zeydel(matrix_16v_5_6zadanieA, matrix_16v_5_6zadanieB, 0.0001)