import numpy as np
import pytest
from scipy.sparse import csr_matrix

from model import Project, Wall, Window, Heater, example_project
from physics import AIR, WINDOW, CONCRETE, assemble, rasterize, solve, ModelError, Equation, solveCG


def test_example_solution_is_bounded_and_residual_small():
    model = example_project()
    answer = solve(model)
    assert answer.unknown_count > 0
    assert np.isfinite(answer.temperature).all()
    assert answer.temperature.min() >= model.outdoor_temperature - 1e-8
    assert answer.temperature.max() <= model.heaters[0].temperature + 1e-8
    assert answer.residual < 1e-7
    diff = answer.matrix - answer.matrix.T
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-8


def test_shared_cg_solver_accepts_sparse_matrix():
    matrix = csr_matrix([[4.0, -1.0], [-1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    answer = solveCG(Equation(matrix, rhs), n_iter=10, tol=1e-12)
    np.testing.assert_allclose(matrix @ answer, rhs, atol=1e-12)


def test_window_replaces_wall_cells():
    r = rasterize(example_project())
    assert np.sum(r.materials == WINDOW) > 0
    assert np.sum(r.materials == CONCRETE) > 0
    assert np.sum((r.materials == AIR) & (~r.street)) > 0


def test_constant_dirichlet_temperature_gives_constant_solution():
    model = example_project()
    model.outdoor_temperature = 20
    model.heaters[0].temperature = 20
    temperature = solve(model).temperature
    np.testing.assert_allclose(temperature, 20, atol=1e-8)


def test_open_room_cannot_have_interior_heater():
    p = Project()
    p.street_seed = (0.3, 0.3)
    p.heaters = [Heater(3.0, 3.0)]
    with pytest.raises(ModelError, match="нагреватель"):
        rasterize(p)


def test_save_load(tmp_path):
    p = example_project()
    path = tmp_path / 'room.json'
    p.save(path)
    loaded = Project.load(path)
    assert loaded.to_dict() == p.to_dict()
