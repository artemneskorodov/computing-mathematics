import numpy as np

from .equation_solvers import solveNewton
from .implicit_runge_kutta import _get_A_b_c, _step_implicit_runge_kutta

#
# Стандартные коэффициенты метода Гира
#
_BDF = {
    1: np.array([1, -1]),
    2: np.array([3, -4, 1]) / 2,
    3: np.array([11, -18, 9, -2]) / 6,
    4: np.array([25, -48, 36, -16, 3]) / 12,
}

def solveGear(f, dfdy, y0, t0, t_end, h, order):
    if order not in _BDF:
        raise ValueError("Supported orders: 1, 2, 3, 4")
    if h <= 0 or t_end <= t0:
        raise ValueError("Require h > 0 and t_end > t0")

    n_steps = int(np.ceil((t_end - t0) / h))
    t, step = np.linspace(t0, t_end, n_steps + 1, retstep=True)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    y = np.empty((n_steps + 1, y0.size))
    y[0] = y0

    # Стартовые узлы для многошаговой формулы получаем неявным РК4.
    tableau = _get_A_b_c(4)
    for n in range(1, min(order, n_steps + 1)):
        y[n] = _step_implicit_runge_kutta(*tableau, f, dfdy, y[n-1], t[n-1], step)

    alpha = _BDF[order]
    diagonal = alpha[0] * np.eye(y0.size)
    for n in range(order, n_steps + 1):
        history = alpha[1:] @ y[n-order:n][::-1]

        def residual(value):
            return alpha[0] * value + history - step * np.asarray(f(t[n], value))

        def jacobian(value):
            return diagonal - step * np.asarray(dfdy(t[n], value))

        y[n] = solveNewton(residual, jacobian, y[n-1], epsilon=1e-13)

    return t, y
