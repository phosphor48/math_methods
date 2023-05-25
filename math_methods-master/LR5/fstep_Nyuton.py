from sympy import *
from matplotlib import pyplot as plt

import devtools
from devtools import *
import numpy as np


def function(arg):
    x = Symbol('x')
    return np.sin(np.pi * arg)


def N_func(N, arg):
    x = Symbol('x')
    return N.subs(x, arg)


def nyuton(n, m, c, d):

    h = (d - c) / m
    ind = [i for i in np.arange(c, d + h, h)]

    counter = 0
    for i in range(m):
        h_m = h / n

        counter += 1

        local_ind = [i for i in np.arange(c + h * i, c + h * i + h + h_m, h_m)]
        local_ind = local_ind[0 : n + 1]

        grid = np.array([np.array([0. for i in range(n + 1)]) for i in range(n + 2)])
        grid[0] = local_ind

        for i in range(0, n + 1):
            grid[1][i] = function(grid[0][i])

        for i in range(2, n + 2):
            for j in range(0, n - i + 2):
                grid[i][j] = grid[i - 1][j + 1] - grid[i - 1][j]
                grid[i][j] /= grid[0][j + i - 1] - grid[0][j]

        x = Symbol('x')

        N = 0
        x_step = 1

        for i in range(1, n + 2):
            N += grid[i][0] * x_step
            x_step *= (x - grid[0][i - 1])

        x_plot = np.arange(local_ind[0], local_ind[-1] + 0.1, 0.1)
        y_plot = [N_func(N, i) for i in x_plot]

        plt.plot(x_plot, y_plot)
        plt.scatter(local_ind, [function(i) for i in local_ind])

        print(f'Таблицa {counter} кусочно-заданой функции: {local_ind, [function(i) for i in local_ind]}')

    x_plot = np.arange(ind[0], ind[-1] + 0.1, 0.1)
    y_plot = [function(i) for i in x_plot]

    print(f"\nТаблица разностей: {grid}\n")

    for i in range(len(x_plot)):
        print(f"Значению X: {x_plot[i]} соответсвует значение Y: {y_plot[i]}")

    plt.plot(x_plot, y_plot, label='График заданной функции')
    plt.scatter(ind, [function(i) for i in ind], label='Точки графика')
    plt.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    data = devtools.data_dev
    table = devtools.table_dev
    nyuton(data["n"], data["m"], data["c"], data["d"])
