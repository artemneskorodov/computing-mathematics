import numpy as np
from .equation_solvers import solveNewton

def _step_implicit_runge_kutta(A, b, c, f, dfdy, y_0, t_0, h):
    y_0 = np.asarray(y_0)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or c.ndim != 1 or c.shape[0] != A.shape[0] or b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Ожидается: A = [s, s], b = [s], c = [s]")

    s = A.shape[0] # Стадийность метода
    n = y_0.shape[0] # Размерность задачи

    K0 = np.array([f(t_0 + h * c[i], y_0) for i in range(s)])

    # Для решения системы уравнений
    # k_i = f(t_n + c_i h, y_n + h sum(a_ij * k_j for j in range(s)))
    # используется метод Ньютона.
    # При этом для решения используется сразу весь набор векторов k_i.

    def F(flat_K):
        K_matrix = np.asarray(flat_K).reshape(s, n)
        Y_matrix = y_0 + h * (A @ K_matrix)
        F_matrix = np.array([f(t_0 + h * c[i], Y_matrix[i]) for i in range(s)])

        return (K_matrix - F_matrix).ravel()

    def jacobian(flat_K):
        K_matrix = np.asarray(flat_K).reshape(s, n)
        Y_matrix = y_0 + h * (A @ K_matrix)
        J = np.zeros((s * n, s * n))

        for i in range(s):
            Jf = np.asarray(dfdy(t_0 + h * c[i], Y_matrix[i])).reshape(n, n)
            for j in range(s):
                block = -h * A[i, j] * Jf
                if i == j:
                    block += np.eye(n)

                J[i*n : (i+1)*n, j*n : (j+1)*n] = block

        return J

    K = solveNewton(F, jacobian, K0.ravel(), epsilon=1e-13).reshape(s, n)

    return y_0 + h * (b @ K)

#
# Стандартные значения таблиц Бутчера для неявных методов Рунге-Кутты
#
def _get_A_b_c(order):
    if order == 1:
        c = [1]
        A = [[1]]
        b = [1]

    elif order == 2:
        c = [1/2]
        A = [[1/2]]
        b = [1]

    elif order == 3:
        c = [1/3, 1]
        A = [
            [5/12, -1/12],
            [3/4, 1/4],
        ]
        b = [3/4, 1/4]

    elif order == 4:
        r = np.sqrt(3) / 6
        c = [1/2 - r, 1/2 + r]
        A = [
            [1/4, 1/4 - r],
            [1/4 + r, 1/4],
        ]
        b = [1/2, 1/2]

    else:
        raise ValueError("Supported orders: 1, 2, 3, 4")

    return np.array(A), np.array(b), np.array(c)

def solveImplicitRungeKutta(f, dfdy, y0, t0, t_end, h, order):
    A, b, c = _get_A_b_c(order)

    if h <= 0 or t_end <= t0:
        raise ValueError("Require h > 0 and t_end > t0")

    N = int(np.ceil((t_end - t0) / h))
    t_values = np.linspace(t0, t_end, N + 1)

    y = np.asarray(y0, dtype=float).reshape(-1)
    y_values = [y.copy()]

    for n in range(N):
        t = t_values[n]
        h_step = t_values[n + 1] - t

        y = _step_implicit_runge_kutta(A, b, c, f, dfdy, y, t, h_step)

        y_values.append(y.copy())

    return t_values, np.array(y_values)
