import numpy as np

#
# Для методов Адамса-Башфорта нужна история значений y, её получаем с помощью метода Рунге-Кутты 4 порядка.
#
from .explicit_runge_kutta import _get_A_b_c, _step_explicit_runge_kutta

#
# Коэффициенты метода Адамса — Башфорта для производных
# f_n, f_{n-1}, f_{n-2}, f_{n-3}.
#
_ADAMS_COEFFICIENTS = {
    1: np.array([1]),
    2: np.array([3 / 2, -1 / 2]),
    3: np.array([23 / 12, -16 / 12, 5 / 12]),
    4: np.array([55 / 24, -59 / 24, 37 / 24, -9 / 24]),
}

#
# Для многошагового метода нужны начальные значения. Получаем их методом
# Рунге-Кутты 4-го порядка, уже заданным выше.
#
_RK4_A, _RK4_B, _RK4_C = _get_A_b_c(4)

# Решение явным методом Адамса — Башфорта.
def solveExplicitAdams(f, y0, t0: float, t_end: float, n_steps: int, s: int):
    if s not in _ADAMS_COEFFICIENTS:
        raise ValueError(f"Unexpected method order: {s}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")

    coefficients = _ADAMS_COEFFICIENTS[s]
    t = np.linspace(t0, t_end, n_steps + 1)
    y0 = np.asarray(y0, dtype=float)
    y = np.empty((n_steps + 1,) + y0.shape, dtype=float)
    y[0] = y0
    f_history = [np.asarray(f(t[0], y[0]), dtype=float)]

    for n in range(n_steps):
        h = t[n + 1] - t[n]

        if n < s - 1:
            # Запускаем многошаговый метод значениями Рунге-Кутты 4-го порядка.
            y[n + 1] = _step_explicit_runge_kutta(_RK4_A, _RK4_B, _RK4_C, f, y[n], t[n], h)

        else:
            # История хранится от самой свежей производной к более старым.
            weighted_f = sum(coefficients[j] * f_history[-1 - j] for j in range(s))

            y[n + 1] = y[n] + h * weighted_f

        f_history.append(np.asarray(f(t[n + 1], y[n + 1]), dtype=float))

    return t, y
