import numpy as np
import numpy.typing as nptp
import typing as tp

#
# Default zero flag
#
_EPSILON = np.finfo(np.float32).eps

#
# Check for zero with delta
#
def _is_zero(value, delta=_EPSILON):
    if np.abs(value) < delta:
        return True
    return False

#
# Equation class
#
class Equation:
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


IterationCallback = tp.Optional[tp.Callable[[int, nptp.NDArray], None]]


def _call_iteration_callback(callback: IterationCallback, iteration: int, x) -> None:
    if callback is not None:
        callback(iteration, x.copy())

#
# Решение методом Гаусса с выбором главного элемента.
#
def solveGaussWithPivoting(equation: Equation) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    #
    # Прямой ход
    #
    for k in range(n):
        # Строка с наибольшим по модулю a_mk
        m = k + np.argmax(np.abs(A[k:, k]))

        # Меняем строки m и k
        if k != m:
            k_row = A[k, k:].copy()
            A[k, k:] = A[m, k:]
            A[m, k:] = k_row

            k_b = b[k]
            b[k] = b[m]
            b[m] = k_b

        if _is_zero(A[k, k]):
            raise ValueError(f"The equation cannot be solved")

        # Вычитаем из всех строк k-ую
        for i in range(k+1, n):
            t = A[i, k] / A[k, k]
            A[i, k:] -= t * A[k, k:]
            A[i, k] = 0.0
            b[i] -= t * b[k]

    #
    # Обратный ход
    #
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        sum = np.dot(A[k, k+1:], x[k+1:])
        x[k] = (b[k] - sum) / A[k, k]

    return x

#
# Решение методом LU-разложения
#

def canUseLUDecomposition(equation: Equation) -> bool:
    for k in range(1, equation.n + 1):
        det = np.linalg.det(equation.A[:k, :k])
        if _is_zero(det):
            return False
    return True

def _get_LU(equation: Equation):
    A = equation.A
    n = equation.n

    L = np.zeros((n, n))
    U = np.eye(n)

    for m in range(n):
        for i in range(m, n):
            L[i, m] = A[i, m] - np.dot(L[i, :m], U[:m, m])

        if _is_zero(L[m, m]):
            raise ValueError(f"Cannot solve with LU decomposition")
        for j in range(m + 1, n):
            U[m, j] = (A[m, j] - np.dot(L[m, :m], U[:m, j])) / L[m, m]

    return L, U

def solveLUDecomposition(equation: Equation) -> nptp.NDArray[np.float32]:
    L, U = _get_LU(equation)
    b = equation.b
    n = equation.n

    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    x = np.zeros(n)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:]))

    return x

#
# Условие возможности решения с помощью метода Якоби или Зейделя - отличие от нулю элементов
# диагонали матрицы.
# Достаточным условием это не является, оно только необходимо по построению.
#
def canSolveJacobiSeidel(equation: Equation) -> bool:
    for k in range(equation.n):
        if _is_zero(equation.A[k, k]):
            return False
    return True

#
# Решение методом Якоби.
#
def solveJacobiMethod(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        prev_x = x.copy()
        for i in range(n):
            sum_less_i = np.dot(A[i, :i], prev_x[:i])
            sum_greater_i = np.dot(A[i, i+1:], prev_x[i+1:])
            x[i] = (b[i] - sum_less_i - sum_greater_i) / A[i, i]
        _call_iteration_callback(callback, iteration, x)

    return x

#
# Следующая итерация по методу Зейделя
#
def _get_next_seidel(A, b, n, x):
    prev_x = x.copy()
    for i in range(n):
        sum_less_i = np.dot(A[i, :i], x[:i])
        sum_greater_i = np.dot(A[i, i+1:], prev_x[i+1:])
        x[i] = (b[i] - sum_less_i - sum_greater_i) / A[i, i]
    return x
#
# Решение методом Зейделя.
#
def solveSeidelMethod(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        prev_x = x.copy()
        x = _get_next_seidel(A, b, n, prev_x)
        _call_iteration_callback(callback, iteration, x)
    return x

#
# Решение схемой последовательной верхней релаксации (SOR).
#
def solveSOR(
    equation: Equation, omega, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        prev_x = x.copy()
        for i in range(n):
            sum_less_i = np.dot(A[i, :i], x[:i])
            sum_greater_i = np.dot(A[i, i+1:], prev_x[i+1:])
            x_seidel_i = (b[i] - sum_less_i - sum_greater_i) / A[i, i]
            x[i] = prev_x[i] + omega * (x_seidel_i - prev_x[i])
        _call_iteration_callback(callback, iteration, x)
    return x

#
# Метод градиентного спуска (подходит только для симметрических положительно определенных матриц)
#
def solveGradientDescent(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        prev_x = x.copy()
        r = b - A @ prev_x
        alpha = np.dot(r, r) / np.dot(r, A @ r)
        x = prev_x + alpha * r
        _call_iteration_callback(callback, iteration, x)

    return x

#
# Метод минимальных невязок.
#
def solveMinimalResidual(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        prev_x = x.copy()
        r = b - A @ prev_x
        Ar = A @ r
        alpha = np.dot(r, Ar) / np.dot(Ar, Ar)
        x = prev_x + alpha * r
        _call_iteration_callback(callback, iteration, x)

    return x

#
# Метод сопряженных градиентов.
#
def solveCG(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    r = b - A @ x
    p = r
    _call_iteration_callback(callback, 0, x)

    for iteration in range(1, n_iter + 1):
        alpha = np.dot(p, r) / np.dot(p, A @ p)
        next_x = x + alpha * p
        next_r = r - alpha * (A @ p)
        beta = np.dot(next_r, next_r) / np.dot(r, r)
        next_p = next_r + beta * p

        x = next_x
        r = next_r
        p = next_p
        _call_iteration_callback(callback, iteration, x)

    return x

#
# Стабилизированный метод бисопряжённых градиентов (BiCGSTAB).
# Подходит в том числе для несимметричных матриц.
#
def solveBiCGSTAB(
    equation: Equation, n_iter, callback: IterationCallback = None
) -> nptp.NDArray[np.float32]:
    A = equation.A
    b = equation.b
    n = equation.n

    x = np.ones(n)
    r = b - A @ x
    _call_iteration_callback(callback, 0, x)
    if np.all(r == 0.0):
        return x

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

        # Если s равен нулю, точное решение уже получено и второй шаг не нужен.
        if np.all(s == 0.0):
            x = x + alpha * p
            _call_iteration_callback(callback, k + 1, x)
            return x

        t = A @ s
        tt = np.dot(t, t)
        if tt == 0.0:
            raise ValueError("BiCGSTAB breakdown: (A s, A s) = 0")

        omega = np.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        _call_iteration_callback(callback, k + 1, x)
        if np.all(r == 0.0):
            return x

        rho_prev = rho

    return x
