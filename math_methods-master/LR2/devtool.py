import numpy as np


def nevyazka(A, B, X):
    r = np.dot(A, X) - B
    R = 0
    for i in range(len(r)):
        R += r[i] ** 2
        print(r[i])
    R = np.sqrt(R)
    return R

#1-3

matrix_16v_5_6zadanieA = np.array([
    [-4.01,14.03,1.37],
    [-2.1,2.27,14.15],
    [7.4,-1.99,-0.1]
])

matrix_16v_5_6zadanieB = np.array([
    -0.43,
    -2.73,
    1.79
])

matrix_16v_4_progonkaA = np.array([
    [4,2,-4,8],
    [2,17,-6,24],
    [-4,-6,9,-21],
    [8,24,-21,73]
])

matrix_16v_4_progonkaB = ([
    16,
    60,
    -65,
    233
])

matrix_16v_1_3_g_l_l_A = [
    [1,8,0,0],
    [-1,0,7,0],
    [0,3,-2,1],
    [0,0,-9,1]
]

matrix_16v_1_3_g_l_l_B = [
    -57,
    -27,
    -11,
    38
]

matrix_second_a = np.array((
    (1, 3, 1, 5),
    (3, 25, 7, -5),
    (1, 7, 11, -12),
    (5, -5, -12, 82)
))

matrix_second_b = np.array((
    -39,
    -129,
    -21,
    -272
))

matrix_third_a = np.array((
    (1,-3,0,0),
    (-3,-3,2,0),
    (0,5,-1,-6),
    (0, 0,-6,1)
))

matrix_third_b = np.array((
    -19,
    -53,
    17,
    47
))

a = np.array((
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5)
))

b = np.array((
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5)
))