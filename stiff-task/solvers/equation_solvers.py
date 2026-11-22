import numpy as np
import numpy.typing as nptp

class LinearEquation:
    def __init__(self, A: nptp.NDArray[np.float32], b: nptp.NDArray[np.float32]) -> None:
        if A.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {A.ndim}D matrix")

        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected NxN matrix, got {A.shape[0]}x{A.shape[0]} matrix")

        if b.ndim != 1:
            raise ValueError(f"Expected column, b.ndim = {b.ndim}")

        if b.shape[0] != A.shape[0]:
            raise ValueError(f"Expected column with n = {A.shape[0]} values, got {b.shape[0]}")

        self.A = A
        self.b = b
        self.n = A.shape[0]

#
# Стабилизированный метод бисопряжённых градиентов (BiCGSTAB).
# Подходит в том числе для несимметричных матриц.
#
def solveBiCGSTAB(equation: LinearEquation, n_iter: int):
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.zeros(n)
    r = b - A @ x
    if np.all(r == 0.0):
        return x

    tolerance = 1e-14 * np.linalg.norm(r, np.inf)
    r_hat = r.copy()

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)

    for k in range(n_iter):
        rho = np.dot(r_hat, r)
        if rho == 0.0:
            raise ValueError("BiCGSTAB breakdown: (r_hat, r) = 0")

        if k == 0:
            p = r.copy()
        else:
            if omega == 0.0:
                raise ValueError("BiCGSTAB breakdown: omega = 0")
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

        v = A @ p
        r_hat_v = np.dot(r_hat, v)
        if r_hat_v == 0.0:
            raise ValueError("BiCGSTAB breakdown: (r_hat, A p) = 0")

        alpha = rho / r_hat_v
        s = r - alpha * v

        # При малой промежуточной невязке второй шаг не нужен.
        if np.linalg.norm(s, np.inf) <= tolerance:
            x = x + alpha * p
            return x

        t = A @ s
        tt = np.dot(t, t)
        if tt == 0.0:
            raise ValueError("BiCGSTAB breakdown: (A s, A s) = 0")

        omega = np.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        if np.linalg.norm(r, np.inf) <= tolerance:
            return x

        rho_prev = rho

    return x

_EPSILON = 1e-6

#
# Решение уравнения методом Ньютона
#
def solveNewton(f, jacobian, x0, epsilon=_EPSILON):
    x = x0.copy()

    for _ in range(100):
        residual = f(x)
        if np.linalg.norm(residual, np.inf) < epsilon:
            return x

        equation = LinearEquation(jacobian(x), -residual)
        try:
            delta = solveBiCGSTAB(equation, 100)
        except ValueError:
            # Ортогональность в BiCGSTAB возможна и для невырожденного якобиана.
            delta = np.linalg.solve(equation.A, equation.b)
        else:
            if np.linalg.norm(equation.A @ delta - equation.b, np.inf) > 1e-8 * np.linalg.norm(residual, np.inf):
                delta = np.linalg.solve(equation.A, equation.b)
        x = x + delta
        # При жёстком якобиане невязка может упереться в округление
        # раньше, чем поправка Ньютона перестанет различаться в x.
        if np.linalg.norm(delta, np.inf) <= np.finfo(float).eps * (1 + np.linalg.norm(x, np.inf)):
            return x

    if np.linalg.norm(f(x), np.inf) < epsilon:
        return x
    raise RuntimeError("Метод Ньютона не сошёлся за 100 итераций")
