"""Stationary 2D conduction with piecewise constant conductivities.

The outer (street) connected air region is Dirichlet at outdoor temperature.
Heater disks are Dirichlet at their specified temperature. Other boundaries
of the simulation rectangle are insulated (zero normal heat flux).

Finite-volume face fluxes use harmonic-mean conductivity, preserving symmetry
and flux continuity across material interfaces.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# Allow `python main.py` and tests run from this directory to use the shared solver.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from linsolver import Equation, solveCG

from model import Project, Wall

AIR, CONCRETE, WINDOW = 0, 1, 2


class ModelError(ValueError):
    pass


@dataclass
class Raster:
    materials: np.ndarray
    street: np.ndarray
    heaters: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    conductivity: np.ndarray
    hx: float
    hy: float


@dataclass
class Solution:
    temperature: np.ndarray
    raster: Raster
    matrix: csr_matrix
    rhs: np.ndarray
    residual: float
    unknown_count: int


def _distance2_segment(X: np.ndarray, Y: np.ndarray, wall: Wall):
    vx, vy = wall.x2 - wall.x1, wall.y2 - wall.y1
    length2 = vx * vx + vy * vy
    if length2 < 1e-10:
        raise ModelError("Обнаружена стена нулевой длины")
    t = np.clip(((X - wall.x1) * vx + (Y - wall.y1) * vy) / length2, 0.0, 1.0)
    return (X - wall.x1 - t * vx) ** 2 + (Y - wall.y1 - t * vy) ** 2, t


def rasterize(project: Project) -> Raster:
    if project.step <= 0 or project.width <= 0 or project.height <= 0:
        raise ModelError("Размеры области и шаг сетки должны быть положительными")
    if project.step > min(project.width, project.height) / 4:
        raise ModelError("Шаг сетки слишком большой для выбранной области")
    if any(v <= 0 for v in (
        project.air_conductivity,
        project.concrete_conductivity,
        project.window_conductivity,
    )):
        raise ModelError("Теплопроводности должны быть положительными")
    # Cell-centred grid includes the entire rectangle without virtual border nodes.
    nx = round(project.width / project.step)
    ny = round(project.height / project.step)
    if nx < 3 or ny < 3:
        raise ModelError("Слишком мало узлов")
    if nx * ny > 100_000:
        raise ModelError("Слишком мелкая сетка: больше 100 000 ячеек")
    hx, hy = project.width / nx, project.height / ny
    x = (np.arange(nx) + 0.5) * hx
    y = (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(x, y)

    materials = np.full((ny, nx), AIR, dtype=np.uint8)
    # Paint concrete first, then windows replace parts of their parent walls.
    for wall in project.walls:
        if not (0 < wall.thickness <= 2):
            raise ModelError("Толщина стены должна быть в диапазоне (0, 2] м")
        distance2, _ = _distance2_segment(X, Y, wall)
        materials[distance2 <= (wall.thickness / 2) ** 2] = CONCRETE

    for wall in project.walls:
        vx, vy = wall.x2 - wall.x1, wall.y2 - wall.y1
        length = math.hypot(vx, vy)
        for window in wall.windows:
            if window.length <= 0:
                continue
            half_fraction = window.length / (2 * length)
            lo, hi = max(0.0, window.centre - half_fraction), min(1.0, window.centre + half_fraction)
            # A window is a segment with the same thickness as the wall.
            section = Wall(
                wall.x1 + lo * vx, wall.y1 + lo * vy,
                wall.x1 + hi * vx, wall.y1 + hi * vy,
                wall.thickness,
            )
            if math.hypot(section.x2 - section.x1, section.y2 - section.y1) < 1e-8:
                continue
            distance2, t = _distance2_segment(X, Y, section)
            materials[distance2 <= (wall.thickness / 2) ** 2] = WINDOW

    if project.street_seed is None:
        raise ModelError("Отметьте точку на улице инструментом «Улица»")
    seed_x, seed_y = project.street_seed
    if not (0 <= seed_x < project.width and 0 <= seed_y < project.height):
        raise ModelError("Точка улицы должна быть внутри области расчёта")
    si = min(int(seed_y / hy), ny - 1)
    sj = min(int(seed_x / hx), nx - 1)
    if materials[si, sj] != AIR:
        raise ModelError("Точка улицы попала на стену или окно")

    street = np.zeros((ny, nx), dtype=bool)
    street[si, sj] = True
    pending = deque([(si, sj)])
    while pending:
        i, j = pending.popleft()
        for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if (0 <= ii < ny and 0 <= jj < nx
                    and not street[ii, jj] and materials[ii, jj] == AIR):
                street[ii, jj] = True
                pending.append((ii, jj))

    heaters = np.zeros((ny, nx), dtype=bool)
    fixed_values = np.full((ny, nx), project.outdoor_temperature, dtype=float)
    # A fixed-temperature heater is allowed only in interior air cells.
    for heater in project.heaters:
        if heater.radius <= 0:
            raise ModelError("Радиус нагревателя должен быть положительным")
        mask = ((X - heater.x) ** 2 + (Y - heater.y) ** 2 <= heater.radius ** 2)
        mask &= (materials == AIR) & (~street)
        heaters |= mask
        fixed_values[mask] = heater.temperature

    if not np.any(heaters):
        raise ModelError("Добавьте нагреватель внутри помещения (не на улице)")
    # This catches the common case of an unclosed room outline: the marked
    # street flooded all free-space air, leaving no interior air to heat.
    if not np.any((materials == AIR) & (~street)):
        raise ModelError("Не обнаружена внутренняя область: проверьте замкнутость стен")

    fixed = street | heaters
    conductivity = np.choose(
        materials,
        [project.air_conductivity, project.concrete_conductivity,
         project.window_conductivity],
    )
    return Raster(materials, street, heaters, fixed, fixed_values,
                  conductivity, hx, hy)


def _face_coefficient(k1: float, k2: float, h: float) -> float:
    return 2 * k1 * k2 / (k1 + k2) / (h * h)


def assemble(project: Project) -> tuple[csr_matrix, np.ndarray, Raster, np.ndarray]:
    raster = rasterize(project)
    unknown = ~raster.fixed
    numbering = np.full(unknown.shape, -1, dtype=np.int32)
    numbering[unknown] = np.arange(int(unknown.sum()), dtype=np.int32)
    ny, nx = numbering.shape
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(int(unknown.sum()), dtype=float)
    K = raster.conductivity

    for i, j in np.argwhere(unknown):
        p = int(numbering[i, j])
        diag = 0.0
        for ii, jj, h in (
            (i, j - 1, raster.hx),
            (i, j + 1, raster.hx),
            (i - 1, j, raster.hy),
            (i + 1, j, raster.hy),
        ):
            if not (0 <= ii < ny and 0 <= jj < nx):
                continue  # zero heat flux at the external raster border
            a = _face_coefficient(K[i, j], K[ii, jj], h)
            diag += a
            if raster.fixed[ii, jj]:
                b[p] += a * raster.fixed_values[ii, jj]
            else:
                rows.append(p)
                cols.append(int(numbering[ii, jj]))
                vals.append(-a)
        rows.append(p)
        cols.append(p)
        vals.append(diag)

    A = coo_matrix((vals, (rows, cols)), shape=(len(b), len(b))).tocsr()
    return A, b, raster, numbering


def solve(project: Project) -> Solution:
    A, b, raster, numbering = assemble(project)
    if A.shape[0] == 0:
        raise ModelError("Не осталось неизвестных узлов")
    x = solveCG(Equation(A, b), n_iter=10 * len(b), tol=1e-12)
    if not np.isfinite(x).all():
        raise ModelError("Не удалось решить СЛАУ")
    if np.linalg.norm(A @ x - b) > 1e-12 * max(1.0, np.linalg.norm(b)):
        raise ModelError("Метод сопряжённых градиентов не сошёлся")
    T = np.array(raster.fixed_values, copy=True)
    T[numbering >= 0] = x[numbering[numbering >= 0]]
    residual = float(np.linalg.norm(A @ x - b, ord=np.inf))
    return Solution(T, raster, A, b, residual, len(b))
