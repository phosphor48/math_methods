import copy
from devtool import *
import numpy as np


def yacobi(A, B, eps):
    n = len(A)
    D = np.eye(n)
    F = copy.deepcopy(A)
    for i in range(n):
        for j in range(n):
            if i == j:
                F[i][j] = 0
                D[i][j] = A[i][j]
                itera = 0
                itera = itera + 1
    D_inv = np.linalg.inv(D)
    C = np.dot(-D_inv, F)
    d = np.dot(D_inv, B)
    X = [0] * n
    r = nevyazka(A, B, X)
    count = 0

    while r > eps:
        count = count + 1
        X = np.dot(C, X) + d
        r = nevyazka(A, B, X)
        print(X)
    print("Метод Якоби.")
    print("X:", X)
    print("epsilon =", eps)
    print("Невязка: r = ", nevyazka(A, B, X))
    print("Количество итераций: ", count)
    print("Количество итераций: ", itera)

#---------------------

yacobi(matrix_16v_5_6zadanieA, matrix_16v_5_6zadanieB, 0.0001)
