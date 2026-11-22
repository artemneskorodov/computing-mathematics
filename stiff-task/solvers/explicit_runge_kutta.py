import numpy as np

#
# Шаг явного метода Рунге-Кутты
#
def _step_explicit_runge_kutta(A, b, c, f, y_0, t_0, h):
    y_0 = np.asarray(y_0)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or c.ndim != 1 or c.shape[0] != A.shape[0] or b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Ожидается: A = [s, s], b = [s], c = [s]")

    s = A.shape[0] # Стадийность метода
    n = y_0.shape[0] # Размерность задачи

    # Вычисляем k_i
    k = []
    for i in range(s):
        yi = np.zeros(n)
        for j in range(i): yi += A[i, j] * k[j]
        yi = y_0 + h * yi
        ki = f(t_0 + c[i] * h, yi)
        k.append(ki)

    # Вычисляем y_{n+1}
    y_1 = np.zeros(n)
    for i in range(s):
        y_1 += b[i] * k[i]
    y_1 = y_0 + h * y_1

    return y_1

#
# Стандартные матрицы для методов Рунге-Кутта 
#
_BUTCHER_1 = np.array([
    [0, 0],
    [0, 1],
])
_BUTCHER_2 = np.array([
    [  0,   0,   0],
    [1/2, 1/2,   0],
    [  0,   0,   1],
])
_BUTCHER_3 = np.array([
    [  0,   0,   0,   0],
    [1/2, 1/2,   0,   0],
    [  1,  -1,   2,   0],
    [  0, 1/6, 4/6, 1/6],
])
_BUTCHER_4 = np.array([
    [  0,   0,   0,   0,   0],
    [1/2, 1/2,   0,   0,   0],
    [1/2,   0, 1/2,   0,   0],
    [  1,   0,   0,   1,   0],
    [  0, 1/6, 1/3, 1/3, 1/6],
])

def _get_A_b_c(s):
    if s == 1:
        butcher_table = _BUTCHER_1
    elif s == 2:
        butcher_table = _BUTCHER_2
    elif s == 3:
        butcher_table = _BUTCHER_3
    elif s == 4:
        butcher_table = _BUTCHER_4
    else:
        raise ValueError(f"Порядок не поддерживается для явного метода Рунге-Кутты: {s}")

    A = butcher_table[:s, 1:]
    b = butcher_table[s, 1:]
    c = butcher_table[:s, 0]

    return A, b, c

#
# Решение явным методом Рунге-Кутты
#
def solveExplicitRungeKutta(f, y_0, t_0, t_end, h, s):
    A, b, c = _get_A_b_c(s)

    y = y_0.copy()
    t = t_0

    y_values = [y.copy()]
    t_values = [t]

    while t < t_end:
        t_next = t + h
        y_next = _step_explicit_runge_kutta(A, b, c, f, y, t, h)
        y_values.append(y_next)
        t_values.append(t_next)

        y = y_next.copy()
        t = t_next

    return t_values, y_values
