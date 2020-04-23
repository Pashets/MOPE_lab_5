from copy import deepcopy
from math import sqrt
import numpy as np
from prettytable import PrettyTable

x1_min = 0
x1_max = 3
x2_min = -6
x2_max = 3
x3_min = -4
x3_max = 1

x_average_max = (x1_max + x2_max + x3_max) / 3
x_average_min = (x1_min + x2_min + x3_min) / 3
y_max = 200 + x_average_max
y_min = 200 + x_average_min


def replace_column(list_: list, column, list_replace):
    list_ = deepcopy(list_)
    for i in range(len(list_)):
        list_[i][column] = list_replace[i]
    return list_


def append_to_list_x(x: list, norm=0):
    for i in range(len(x)):
        x[i].append(x[i][0 + norm] * x[i][1 + norm])
        x[i].append(x[i][0 + norm] * x[i][2 + norm])
        x[i].append(x[i][1 + norm] * x[i][2 + norm])
        x[i].append(x[i][0 + norm] * x[i][1 + norm] * x[i][2 + norm])
        x[i].append(x[i][0 + norm] * x[i][0 + norm])
        x[i].append(x[i][1 + norm] * x[i][1 + norm])
        x[i].append(x[i][2 + norm] * x[i][2 + norm])
        for j in range(len(x[i])):
            if round(x[i][j]) == 0:
                x[i][j] = 0
            x[i][j] = round(x[i][j], 3)


def get_value(table: dict, key: int):
    value = table.get(key)
    if value is not None:
        return value
    for i in table:
        if type(i) == range and key in i:
            return table.get(i)


def main(m, n):
    if n == 15:
        l = 1.215
        print(
            'ŷ = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b12 * x1 * x2 + b13 * x1 * x3 + b23 * x2 * x3 + b123 * x1 * x2 * '
            'x3 + b11 * x1 * x1 + b22 * x2 * x2 + b33 * x3 * x3')
        norm_x = [
            [+1, -1, -1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
            [+1, -1, -1, +1],
            [+1, -1, +1, -1],
            [+1, +1, -1, -1],
            [+1, +1, +1, +1],
            [+1, -l, 0, 0],
            [+1, l, 0, 0],
            [+1, 0, -l, 0],
            [+1, 0, l, 0],
            [+1, 0, 0, -l],
            [+1, 0, 0, l],
            [+1, 0, 0, 0]
        ]

        append_to_list_x(norm_x, norm=1)

        delta_x1 = (x1_max - x1_min) / 2
        delta_x2 = (x2_max - x2_min) / 2
        delta_x3 = (x2_max - x3_min) / 2
        x01 = (x1_min + x1_max) / 2
        x02 = (x2_min + x2_max) / 2
        x03 = (x3_min + x3_max) / 2

        x = [
            [1, x1_min, x2_min, x3_min],
            [1, x1_min, x2_max, x3_max],
            [1, x1_max, x2_min, x3_max],
            [1, x1_max, x2_max, x3_min],
            [1, x1_min, x2_min, x3_max],
            [1, x1_min, x2_max, x3_min],
            [1, x1_max, x2_min, x3_min],
            [1, x1_max, x2_max, x3_max],
            [1, -l * delta_x1 + x01, x02, x03],
            [1, l * delta_x1 + x01, x02, x03],
            [1, x01, -l * delta_x2 + x02, x03],
            [1, x01, l * delta_x2 + x02, x03],
            [1, x01, x02, -l * delta_x3 + x03],
            [1, x01, x02, l * delta_x3 + x03],
            [1, x01, x02, x03]
        ]

        append_to_list_x(x, norm=1)

    if n == 8:
        print(
            'ŷ = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b12 * x1 * x2 + b13 * x1 * x3 + b23 * x2 * x3 + b123 * x1 * x2 * x3'
        )
        norm_x = [
            [+1, -1, -1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
            [+1, -1, -1, +1],
            [+1, -1, +1, -1],
            [+1, +1, -1, -1],
            [+1, +1, +1, +1]
        ]

        for i in range(len(norm_x)):
            norm_x[i].append(norm_x[i][1] * norm_x[i][2])
            norm_x[i].append(norm_x[i][1] * norm_x[i][3])
            norm_x[i].append(norm_x[i][2] * norm_x[i][3])
            norm_x[i].append(norm_x[i][1] * norm_x[i][2] * norm_x[i][3])

        x = [
            [x1_min, x2_min, x3_min],
            [x1_min, x2_max, x3_max],
            [x1_max, x2_min, x3_max],
            [x1_max, x2_max, x3_min],
            [x1_min, x2_min, x3_max],
            [x1_min, x2_max, x3_min],
            [x1_max, x2_min, x3_min],
            [x1_max, x2_max, x3_max]
        ]
        for i in range(len(x)):
            x[i].append(x[i][0] * x[i][1])
            x[i].append(x[i][0] * x[i][2])
            x[i].append(x[i][1] * x[i][2])
            x[i].append(x[i][0] * x[i][1] * x[i][2])

    if n == 4:
        print('ŷ = b0 + b1 * x1 + b2 * x2 + b3 * x3')
        norm_x = [
            [+1, -1, -1, -1],
            [+1, -1, +1, +1],
            [+1, +1, -1, +1],
            [+1, +1, +1, -1],
        ]
        x = [
            [x1_min, x2_min, x3_min],
            [x1_min, x2_max, x3_max],
            [x1_max, x2_min, x3_max],
            [x1_max, x2_max, x3_min],
        ]
    y = np.random.randint(y_min, y_max, size=(n, m))
    y_av = list(np.average(y, axis=1))

    for i in range(len(y_av)):
        y_av[i] = round(y_av[i], 3)

    if n == 15:

        t = PrettyTable(['N', 'norm_x_0', 'norm_x_1', 'norm_x_2', 'norm_x_3', 'norm_x_1_x_2', 'norm_x_1_x_3',
                         'norm_x_2_x_3', 'norm_x_1_x_2_x_3', 'norm_x_1_x_1', 'norm_x_2_x_2', 'norm_x_3_x_3', 'x_0',
                         'x_1', 'x_2', 'x_3', 'x_1_x_2', 'x_1_x_3', 'x_2_x_3',
                         'x_1_x_2_x_3', 'x_1_x_1', 'x_2_x_2', 'x_3_x_3'] + [f'y_{i + 1}' for i in range(m)] + ['y_av'])

        for i in range(n):
            t.add_row([i + 1] + list(norm_x[i]) + list(x[i]) + list(y[i]) + [y_av[i]])
        print(t)

        # sums_of_columns_x = [round(i, 3) for i in np.sum(x, axis=0)]
        # m_ij = [[n] + [i for i in sums_of_columns_x]]
        # for i in range(len(x)):
        #     x[i].insert(0, 1)
        m_ij = []
        for i in range(len(x[0])):
            # a = [sums_of_columns_x[i]]
            b = [round(sum([x[k][i] * x[k][j] for k in range(len(x))]), 3) for j in range(len(x[i]))]
            m_ij.append(b)
            # m_ij.append(
            #     a + b
            # )

        ### ####
        def a(first, second):  # first = 1, second = 2 : пошук а12
            """Пошук коефіцієнтів а"""
            need_a = 0
            for j in range(N):
                need_a += matrix_x[j][first - 1] * matrix_x[j][second - 1] / N
            return need_a

        def x(l1, l2, l3):
            """Пошук зоряних точок"""
            x_1 = l1 * delta_x1 + x01
            x_2 = l2 * delta_x2 + x02
            x_3 = l3 * delta_x3 + x03
            return [x_1, x_2, x_3]

        matrix_pfe = [
            [-1, -1, -1, +1, +1, +1, -1, +1, +1, +1],
            [-1, -1, +1, +1, -1, -1, +1, +1, +1, +1],
            [-1, +1, -1, -1, +1, -1, +1, +1, +1, +1],
            [-1, +1, +1, -1, -1, +1, -1, +1, +1, +1],
            [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
            [+1, -1, +1, -1, +1, -1, -1, +1, +1, +1],
            [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
            [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
            [-1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
            [+1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
            [0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
            [0, +1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
            [0, 0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623],
            [0, 0, +1.215, 0, 0, 0, 0, 0, 0, 1.4623],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        # Генеруєм матрицю ПЕ із натуралізованих значень
        N = 15
        matrix_x = [[] for x in range(N)]
        for i in range(len(matrix_x)):
            if i < 8:
                x1 = x1_min if matrix_pfe[i][0] == -1 else x1_max
                x2 = x2_min if matrix_pfe[i][1] == -1 else x2_max
                x3 = x3_min if matrix_pfe[i][2] == -1 else x3_max
            else:
                x_lst = x(matrix_pfe[i][0], matrix_pfe[i][1], matrix_pfe[i][2])
                x1, x2, x3 = x_lst
            matrix_x[i] = [x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2]

        def find_average(lst, orientation):
            """Функція пошуку середнього значення по колонках або по рядках"""
            average = []
            if orientation == 1:  # Середнє значення по рядку
                for rows in range(len(lst)):
                    average.append(sum(lst[rows]) / len(lst[rows]))
            else:  # Середнє значення по колонкі
                for column in range(len(lst[0])):
                    number_lst = []
                    for rows in range(len(lst)):
                        number_lst.append(lst[rows][column])
                    average.append(sum(number_lst) / len(number_lst))
            return average

        average_x = find_average(matrix_x, 0)  # Середні х по колонкам
        mx_i = average_x  # Список середніх значень колонок [Mx1, Mx2, ...]
        unknown = [
            [1, mx_i[0], mx_i[1], mx_i[2], mx_i[3], mx_i[4], mx_i[5], mx_i[6], mx_i[7], mx_i[8], mx_i[9]],
            [mx_i[0], a(1, 1), a(1, 2), a(1, 3), a(1, 4), a(1, 5), a(1, 6), a(1, 7), a(1, 8), a(1, 9), a(1, 10)],
            [mx_i[1], a(2, 1), a(2, 2), a(2, 3), a(2, 4), a(2, 5), a(2, 6), a(2, 7), a(2, 8), a(2, 9), a(2, 10)],
            [mx_i[2], a(3, 1), a(3, 2), a(3, 3), a(3, 4), a(3, 5), a(3, 6), a(3, 7), a(3, 8), a(3, 9), a(3, 10)],
            [mx_i[3], a(4, 1), a(4, 2), a(4, 3), a(4, 4), a(4, 5), a(4, 6), a(4, 7), a(4, 8), a(4, 9), a(4, 10)],
            [mx_i[4], a(5, 1), a(5, 2), a(5, 3), a(5, 4), a(5, 5), a(5, 6), a(5, 7), a(5, 8), a(5, 9), a(5, 10)],
            [mx_i[5], a(6, 1), a(6, 2), a(6, 3), a(6, 4), a(6, 5), a(6, 6), a(6, 7), a(6, 8), a(6, 9), a(6, 10)],
            [mx_i[6], a(7, 1), a(7, 2), a(7, 3), a(7, 4), a(7, 5), a(7, 6), a(7, 7), a(7, 8), a(7, 9), a(7, 10)],
            [mx_i[7], a(8, 1), a(8, 2), a(8, 3), a(8, 4), a(8, 5), a(8, 6), a(8, 7), a(8, 8), a(8, 9), a(8, 10)],
            [mx_i[8], a(9, 1), a(9, 2), a(9, 3), a(9, 4), a(9, 5), a(9, 6), a(9, 7), a(9, 8), a(9, 9), a(9, 10)],
            [mx_i[9], a(10, 1), a(10, 2), a(10, 3), a(10, 4), a(10, 5), a(10, 6), a(10, 7), a(10, 8), a(10, 9),
             a(10, 10)]
        ]
        k_i = []
        for i in range(len(x[0])):
            a = sum(y_av[j] * x[j][i] for j in range(len(x[i])))
            k_i.append(a)
        print('m_ij')
        print(*m_ij, sep='\n')
        print(f'{k_i}')
        det = np.linalg.det(m_ij)
        det_i = [np.linalg.det(replace_column(m_ij, i, k_i)) for i in range(len(k_i))]

        b_i = [round(i / det, 3) for i in det_i]
        # b_i = np.linalg.lstsq(x, y_av, rcond=None)[0]
        # b_i = [round(i, 3) for i in b_i]

        print(
            f"\nThe naturalized regression equation: y = {b_i[0]:.5f} + {b_i[1]:.5f} * x1 + {b_i[2]:.5f} * x2 + "
            f"{b_i[3]:.5f} * x3 + {b_i[4]:.5f} * x1 * x2 + "
            f"{b_i[5]:.5f} * x1 * x3 + {b_i[6]:.5f} * x2 * x3 + {b_i[7]:.5f} * x1 * x2 * x3 + {b_i[8]:.5f} * x1 * x1 + "
            f"{b_i[9]:.5f} * x2 * x2 + {b_i[10]:.5f} * x3 * x3")

        check_i = [
            b_i[0] + b_i[1] * i[1] + b_i[2] * i[2] + b_i[3] * i[3] + b_i[4] * i[4] + b_i[5] * i[5] +
            b_i[6] * i[6] + b_i[7] * i[7] + b_i[8] * i[8] + b_i[9] * i[9] + b_i[10] * i[10] for i in x]

        print("Values are naturalized: ", check_i)

    if n == 8:
        t = PrettyTable(['N', 'norm_x_0', 'norm_x_1', 'norm_x_2', 'norm_x_3', 'norm_x_1_x_2', 'norm_x_1_x_3',
                         'norm_x_2_x_3', 'norm_x_1_x_2_x_3', 'x_1', 'x_2', 'x_3', 'x_1_x_2', 'x_1_x_3', 'x_2_x_3',
                         'x_1_x_2_x_3'] + [f'y_{i + 1}' for i in range(m)] + ['y_av'])
        for i in range(n):
            t.add_row([i + 1] + list(norm_x[i]) + list(x[i]) + list(y[i]) + [y_av[i]])
        print(t)
        sums_of_columns_x = np.sum(x, axis=0)
        m_ij = [[n] + [i for i in sums_of_columns_x]]
        for i in range(len(sums_of_columns_x)):
            m_ij.append(
                [sums_of_columns_x[i]] + [sum([x[k][i] * x[k][j] for k in range(len(x[i]))]) for j in range(len(x[i]))])

        k_i = [sum(y_av)]
        for i in range(len(sums_of_columns_x)):
            k_i.append(sum(y_av[j] * x[j][i] for j in range(len(x[i]))))

        det = np.linalg.det(m_ij)
        det_i = [np.linalg.det(replace_column(m_ij, i, k_i)) for i in range(len(k_i))]

        b_i = [i / det for i in det_i]

        print(
            f"\nThe naturalized regression equation: y = {b_i[0]:.5f} + {b_i[1]:.5f} * x1 + {b_i[2]:.5f} * x2 + "
            f"{b_i[3]:.5f} * x3 + {b_i[4]:.5f} * x1 * x2 + "
            f"{b_i[5]:.5f} * x1 * x3 + {b_i[6]:.5f} * x2 * x3 + {b_i[7]:.5f} * x1 * x2 * x3")

    if n == 4:
        t = PrettyTable(
            ['N', 'norm_x_0', 'norm_x_1', 'norm_x_2', 'norm_x_3', 'x_1', 'x_2', 'x_3'] + [f'y_{i + 1}' for i in
                                                                                          range(m)] + ['y_av'])
        for i in range(n):
            t.add_row([i + 1] + list(norm_x[i]) + list(x[i]) + list(y[i]) + [y_av[i]])
        print(t)

        mx_1, mx_2, mx_3 = [i / len(x) for i in np.sum(x, axis=0)]
        my = sum(y_av) / len(y_av)

        a_1 = sum([x[i][0] * y_av[i] for i in range(len(x))]) / len(x)
        a_2 = sum([x[i][1] * y_av[i] for i in range(len(x))]) / len(x)
        a_3 = sum([x[i][2] * y_av[i] for i in range(len(x))]) / len(x)

        a_11 = sum([x[i][0] ** 2 for i in range(len(x))]) / len(x)
        a_22 = sum([x[i][1] ** 2 for i in range(len(x))]) / len(x)
        a_33 = sum([x[i][2] ** 2 for i in range(len(x))]) / len(x)
        a_12 = sum([x[i][0] * x[i][1] for i in range(len(x))]) / len(x)
        a_13 = sum([x[i][0] * x[i][2] for i in range(len(x))]) / len(x)
        a_23 = a_32 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / len(x)

        matrix = [
            [1, mx_1, mx_2, mx_3],
            [mx_1, a_11, a_12, a_13],
            [mx_2, a_12, a_22, a_32],
            [mx_3, a_13, a_23, a_33]
        ]

        answers = [my, a_1, a_2, a_3]

        det = np.linalg.det(matrix)
        det_i = [np.linalg.det(replace_column(matrix, i, answers)) for i in range(len(answers))]

        b_i = [i / det for i in det_i]
        print(
            f"\nThe naturalized regression equation: y = {b_i[0]:.5f} + {b_i[1]:.5f} * x1 + {b_i[2]:.5f} * x2 + {b_i[3]:.5f} * x3\n")

    print("\n[ Kohren's test ]")
    f_1 = m - 1
    f_2 = n
    s_i = [sum([(i - y_av[j]) ** 2 for i in y[j]]) / m for j in range(len(y))]
    g_p = max(s_i) / sum(s_i)

    table = {3: 0.6841, 4: 0.6287, 5: 0.5892, 6: 0.5598, 7: 0.5365, 8: 0.5175, 9: 0.5017, 10: 0.4884,
             range(11, 17): 0.4366, range(17, 37): 0.3720, range(37, 145): 0.3093}
    g_t = get_value(table, m)

    if g_p < g_t:
        print(f"The variance is homogeneous: Gp = {g_p:.5} < Gt = {g_t}")
    else:
        print(f"The variance is not homogeneous Gp = {g_p:.5} < Gt = {g_t}\nStart again with m = m + 1")
        return main(m=m + 1, n=n)

    print("\n[ Student's test ]")
    s2_b = sum(s_i) / n
    s2_beta_s = s2_b / (n * m)
    s_beta_s = sqrt(s2_beta_s)
    if n == 15:
        beta_i = b_i
    else:
        beta_i = [sum([norm_x[i][j] * y_av[i] for i in range(len(norm_x))]) / n for j in range(len(norm_x[0]))]
    beta_i = [round(i, 3) for i in beta_i]

    t = [abs(i) / s_beta_s for i in beta_i]

    f_3 = f_1 * f_2
    t_table = {8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
               17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.08, 22: 2.074, 23: 2.069, 24: 2.064,
               range(25, 30): 2.06, range(30, 40): 2.042, range(40, 60): 2.021, range(60, 100): 2,
               range(100, 2 ** 100): 1.96}
    d = deepcopy(n)
    for i in range(len(t)):
        if get_value(t_table, f_3) > t[i]:
            beta_i[i] = 0
            d -= 1
    if n == d:
        n = 8 if n == 4 else 15
        print(f"n=d\nStart again with n = {n}")
        main(m=m + 1, n=n)
    if n == 15:
        print(
            f"\nThe naturalized simplified regression equation: y = {beta_i[0]:.5f} + {beta_i[1]:.5f} * x1 + "
            f"{beta_i[2]:.5f} * x2 + {beta_i[3]:.5f} * x3 + {beta_i[4]:.5f} * x1 * x2 + "
            f"{beta_i[5]:.5f} * x1 * x3 + {beta_i[6]:.5f} * x2 * x3 + {beta_i[7]:.5f} * x1 * x2 * x3 + "
            f"{beta_i[8]:.5f} * x1 * x1 + {beta_i[9]:.5f} * x2 * x2 + {beta_i[10]:.5f} * x3 * x3")

        check_i = [
            beta_i[0] + beta_i[1] * i[1] + beta_i[2] * i[2] + beta_i[3] * i[3] + beta_i[4] * i[4] + beta_i[5] * i[5] +
            beta_i[6] * i[6] + beta_i[7] * i[7] + beta_i[8] * i[8] + beta_i[9] * i[9] + beta_i[10] * i[10] for i in x]

        print("Values are normalized: ", check_i)
    if n == 8:
        print(
            f"\nThe normalized regression equation: y = {beta_i[0]:.5f} + {beta_i[1]:.5f} * x1 + {beta_i[2]:.5f} * x2 + "
            f"{beta_i[3]:.5f} * x3 + {beta_i[4]:.5f} * x1 * x2 + "
            f"{beta_i[5]:.5f} * x1 * x3 + {beta_i[6]:.5f} * x2 * x3 + {beta_i[7]:.5f} * x1 * x2 * x3")
        check_i = [
            beta_i[0] + beta_i[1] * i[0] + beta_i[2] * i[1] + beta_i[3] * i[2] + beta_i[4] * i[3] + beta_i[5] * i[4] +
            beta_i[6] * i[5] + beta_i[7] * i[6] for i in norm_x]
        print("Values are normalized: ", check_i)

    if n == 4:
        print(
            f"\nThe normalized regression equation: y = {beta_i[0]:.5f} + {beta_i[1]:.5f} * x1 + {beta_i[2]:.5f} * x2 + "
            f"{beta_i[3]:.5f} * x3")
        check_i = [
            beta_i[0] + beta_i[1] * i[0] + beta_i[2] * i[1] + beta_i[3] * i[2] for i in norm_x]
        print("Values are normalized: ", check_i)

    print("\n[ Fisher's test ]")
    f_4 = n - d
    s2_ad = m / f_4 * sum([(check_i[i] - y_av[i]) ** 2 for i in range(len(y_av))])
    f_p = s2_ad / s2_b
    f_t = {
        1: [164.4, 199.5, 215.7, 224.6, 230.2, 234, 235.8, 237.6],
        2: [18.5, 19.2, 19.2, 19.3, 19.3, 19.3, 19.4, 19.4],
        3: [10.1, 9.6, 9.3, 9.1, 9, 8.9, 8.8, 8.8],
        4: [7.7, 6.9, 6.6, 6.4, 6.3, 6.2, 6.1, 6.1],
        5: [6.6, 5.8, 5.4, 5.2, 5.1, 5, 4.9, 4.9],
        6: [6, 5.1, 4.8, 4.5, 4.4, 4.3, 4.2, 4.2],
        7: [5.5, 4.7, 4.4, 4.1, 4, 3.9, 3.8, 3.8],
        8: [5.3, 4.5, 4.1, 3.8, 3.7, 3.6, 3.5, 3.5],
        9: [5.1, 4.3, 3.9, 3.6, 3.5, 3.4, 3.3, 3.3],
        10: [5, 4.1, 3.7, 3.5, 3.3, 3.2, 3.1, 3.1],
        11: [4.8, 4, 3.6, 3.4, 3.2, 3.1, 3, 3],
        12: [4.8, 3.9, 3.5, 3.3, 3.1, 3, 2.9, 2.9],
        13: [4.7, 3.8, 3.4, 3.2, 3, 2.9, 2.8, 2.8],
        14: [4.6, 3.7, 3.3, 3.1, 3, 2.9, 2.8, 2.7],
        15: [4.5, 3.7, 3.3, 3.1, 2.9, 2.8, 2.7, 2.7],
        16: [4.5, 3.6, 3.2, 3, 2.9, 2.7, 2.6, 2.6],
        17: [4.5, 3.6, 3.2, 3, 2.8, 2.7, 2.5, 2.3],
        18: [4.4, 3.6, 3.2, 2.9, 2.8, 2.7, 2.5, 2.3],
        19: [4.4, 3.5, 3.1, 2.9, 2.7, 2.7, 2.4, 2.3],
        range(20, 22): [4.4, 3.5, 3.1, 2.8, 2.7, 2.7, 2.4, 2.3],
        range(22, 24): [4.3, 3.4, 3.1, 2.8, 2.7, 2.6, 2.4, 2.3],
        range(24, 26): [4.3, 3.4, 3, 2.8, 2.6, 2.5, 2.3, 2.2],
        range(26, 28): [4.2, 3.4, 3, 2.7, 2.6, 2.5, 2.3, 2.2],
        range(28, 30): [4.2, 3.3, 3, 2.7, 2.6, 2.4, 2.3, 2.1],
        range(30, 40): [4.2, 3.3, 3, 2.7, 2.6, 2.4, 2.3, 2.1, 2, 2, 2, 2],
        range(40, 60): [4.1, 3.2, 2.9, 2.6, 2.5, 2.3, 2.2, 2, 1.9, 1.9, 1.9, 1.9],
        range(60, 120): [4, 3.2, 2.8, 2.5, 2.4, 2.3, 2.1, 1.9, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
        range(120, 2 ** 100): [3.8, 3, 2.6, 2.4, 2.2, 2.1, 2, 2, 1.9, 1.9, 1.9, 1.8, 1.8]
    }
    if f_p > get_value(f_t, f_3)[f_4]:
        n = 8 if n == 4 else 15
        print(
            f"fp = {f_p} > ft = {get_value(f_t, f_3)[f_4]}.\nThe mathematical model is not adequate to the experimental "
            f"data\nStart again with m = m + 1 = {m + 1} and n = {n}")
        main(m=m + 1, n=n)
    else:
        print(
            f"fP = {f_p} < fT = {get_value(f_t, f_3)[f_4]}.\nThe mathematical model is adequate to the experimental data\n")


# main(m=3, n=4)
main(m=3, n=15)
