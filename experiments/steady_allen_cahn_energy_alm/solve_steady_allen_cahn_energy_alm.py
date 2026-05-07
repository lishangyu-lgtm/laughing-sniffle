from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    _HERE = Path(__file__).resolve().parent
    _EXPERIMENTS = _HERE.parent
    _ROOT = _EXPERIMENTS.parent
    _PARENT = _ROOT.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))
    __package__ = f"{_ROOT.name}.{_EXPERIMENTS.name}.{_HERE.name}"

from ...baselines.fdm_reference import (
    evaluate_fdm_on_side,
    fdm_plot_arrays,
    solve_interface_fdm,
)
from ...core.methods_tfpm import assemble_lagrange_kkt_system
from ...core.tfpm_local import (
    build_elements,
    build_uniform_grid,
    evaluate_solution_fine,
    evaluate_tfpm_on_side,
    interface_node_from_grid,
)
from ..nonlinear_pde.analysis_plot import (
    compute_errors_vs_reference,
    plot_errors_vs_reference,
    plot_solutions,
    write_summary as write_comparison_summary,
)

ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class ProblemConfig:
    a: float = -0.5
    b: float = 0.5
    x_interface: float = 0.0
    epsilon: float = 0.1
    num_elements: int = 40
    quad_n: int = 12
    n_seg_gauss: int = 8
    left_bc: float = 0.0
    right_bc: float = 0.0
    jump_u: float = 0.0
    jump_du: float = 0.0
    flux_jump_weights: tuple[float, float] = (0.5, 0.5)
    basis_kind: str = "endpoint_auto"
    initial_guess_kind: str = "smooth_periodic"
    initial_tanh_width: float | None = None


@dataclass(slots=True)
class SolverConfig:
    rho: float = 100
    max_space_iter: int = 40
    space_tol: float = 1.0e-10
    energy_tol: float = 1.0e-12
    max_aug_lag_iter: int = 200
    max_inner_newton_iter: int = 80
    tol_primal: float = 1.0e-10
    tol_stationarity: float = 1.0e-10
    tol_inner_grad: float = 1.0e-11
    line_search_max_iter: int = 24
    armijo: float = 1.0e-4
    error_check_points: int = 800
    snapshot_points_per_element: int = 50
    plot_points_per_element: int = 30
    fdm_num_elements: int | None = None
    fdm_reference_num_elements: int = 2**14
    fdm_max_newton_iter: int = 80
    fdm_newton_tol: float = 1.0e-13
    fdm_plot_points_per_cell: int = 20
    comparison_points_per_side: int = 2000


@dataclass(slots=True)
class InnerAlmHistory:
    aug_lag_iters: int
    inner_newton_iters: int
    primal_inf: float
    stationarity_inf: float
    converged: bool
    rho: float


@dataclass(slots=True)
class SpaceIterationRecord:
    iteration: int
    energy: float
    energy_delta: float
    step_inf: float
    primal_inf: float
    stationarity_inf: float
    aug_lag_iters: int
    inner_newton_iters: int
    inner_converged: bool


@dataclass(slots=True)
class IterateState:
    elems: object | None = None
    z: np.ndarray | None = None
    forcing: ArrayFunc | None = None
    y_snapshot: np.ndarray | None = None
    u_snapshot: np.ndarray | None = None

    def snapshot_evaluator(self) -> ArrayFunc:
        y_snapshot = self.y_snapshot
        u_snapshot = self.u_snapshot

        def evaluate(y: np.ndarray) -> np.ndarray:
            y_arr = np.asarray(y, dtype=np.float64)
            if y_snapshot is None or u_snapshot is None:
                return np.zeros_like(y_arr)
            return np.interp(y_arr, y_snapshot, u_snapshot)

        return evaluate

    def update(self, elems: object, z: np.ndarray, forcing: ArrayFunc, config: ProblemConfig, solver: SolverConfig) -> None:
        self.elems = elems
        self.z = z
        self.forcing = forcing
        self.y_snapshot, self.u_snapshot = evaluate_solution_fine(
            elems=elems,
            z=z,
            f_func=forcing,
            points_per_element=solver.snapshot_points_per_element,
            n_seg_gauss=config.n_seg_gauss,
        )


@dataclass(slots=True)
class SolveResult:
    problem: ProblemConfig
    solver: SolverConfig
    state: IterateState
    history: list[SpaceIterationRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    converged: bool = False

    def evaluate_on_side(self, x_eval: np.ndarray, side: str) -> np.ndarray:
        if self.state.elems is None or self.state.z is None or self.state.forcing is None:
            raise RuntimeError("No solution is available for evaluation.")
        y_eval = x_to_y(self.problem, x_eval)
        return evaluate_tfpm_on_side(
            elems=self.state.elems,
            z=self.state.z,
            f_func=self.state.forcing,
            x_eval=y_eval,
            side=side,
            xI=transformed_interface(self.problem),
            n_seg_gauss=self.problem.n_seg_gauss,
        )

    def plot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.state.elems is None or self.state.z is None or self.state.forcing is None:
            raise RuntimeError("No solution is available to plot.")
        y_plot, u_plot = evaluate_solution_fine(
            elems=self.state.elems,
            z=self.state.z,
            f_func=self.state.forcing,
            points_per_element=self.solver.plot_points_per_element,
            n_seg_gauss=self.problem.n_seg_gauss,
        )
        return y_to_x(self.problem, y_plot), u_plot


@dataclass(slots=True)
class FdmIterationRecord:
    iteration: int
    update_inf: float


@dataclass(slots=True)
class FdmSolveResult:
    problem: ProblemConfig
    solver: SolverConfig
    num_elements: int
    y_grid: np.ndarray
    solution: dict[str, np.ndarray] | None = None
    history: list[FdmIterationRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    converged: bool = False

    def evaluate_on_side(self, x_eval: np.ndarray, side: str) -> np.ndarray:
        if self.solution is None:
            raise RuntimeError("No FDM solution is available for evaluation.")
        y_eval = x_to_y(self.problem, x_eval)
        return evaluate_fdm_on_side(self.solution, y_eval, side=side)

    def plot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.solution is None:
            raise RuntimeError("No FDM solution is available to plot.")
        y_plot, u_plot = fdm_plot_arrays(self.solution, points_per_cell=self.solver.fdm_plot_points_per_cell)
        return y_to_x(self.problem, y_plot), u_plot


@dataclass(slots=True)
class FdmComparisonResult:
    energy_result: SolveResult
    fdm_result: FdmSolveResult
    fdm_reference_result: FdmSolveResult
    x_left: np.ndarray
    x_right: np.ndarray
    reference_values: tuple[np.ndarray, np.ndarray]
    method_values: dict[str, tuple[np.ndarray, np.ndarray]]
    relative_l2_errors: dict[str, float]
    timings: dict[str, float]
    solution_plot_path: Path
    error_plot_path: Path
    summary_path: Path


def _validate_epsilon(problem: ProblemConfig) -> None:
    if problem.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")


def x_to_y(problem: ProblemConfig, x: np.ndarray) -> np.ndarray:
    _validate_epsilon(problem)
    return np.asarray(x, dtype=np.float64) / float(problem.epsilon)


def y_to_x(problem: ProblemConfig, y: np.ndarray) -> np.ndarray:
    _validate_epsilon(problem)
    return float(problem.epsilon) * np.asarray(y, dtype=np.float64)


def transformed_interface(problem: ProblemConfig) -> float:
    return float(x_to_y(problem, np.array([problem.x_interface], dtype=np.float64))[0])


def transformed_jump_du(problem: ProblemConfig) -> float:
    return float(problem.epsilon) * float(problem.jump_du)


def build_grid(problem: ProblemConfig) -> np.ndarray:
    _validate_epsilon(problem)
    physical_grid, _h, _interface_node = build_uniform_grid(
        N=problem.num_elements,
        a=problem.a,
        b=problem.b,
        xI=problem.x_interface,
    )
    return x_to_y(problem, physical_grid)


def _build_fdm_grid(problem: ProblemConfig, num_elements: int) -> np.ndarray:
    _validate_grid_count("FDM num_elements", num_elements, problem)
    physical_grid, _h, _interface_node = build_uniform_grid(
        N=int(num_elements),
        a=problem.a,
        b=problem.b,
        xI=problem.x_interface,
    )
    return x_to_y(problem, physical_grid)


def _resolve_fdm_num_elements(problem: ProblemConfig, solver: SolverConfig, num_elements: int | None) -> int:
    if num_elements is not None:
        return int(num_elements)
    if solver.fdm_num_elements is not None:
        return int(solver.fdm_num_elements)
    return int(problem.num_elements)


def _validate_grid_count(name: str, num_elements: int, problem: ProblemConfig) -> None:
    try:
        build_uniform_grid(
            N=int(num_elements),
            a=problem.a,
            b=problem.b,
            xI=problem.x_interface,
        )
    except ValueError as exc:
        raise ValueError(
            f"{name}={num_elements} is incompatible with "
            f"[{problem.a:.16g}, {problem.b:.16g}] and x_interface={problem.x_interface:.16g}."
        ) from exc


def initial_guess_values(problem: ProblemConfig, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    kind = str(problem.initial_guess_kind).strip().lower().replace("-", "_")
    if kind == "zero":
        return np.zeros_like(x_arr)
    if kind == "linear":
        slope = (problem.right_bc - problem.left_bc) / (problem.b - problem.a)
        return problem.left_bc + slope * (x_arr - problem.a)
    if kind in {"smooth_periodic", "cosine", "cos"}:
        length = problem.b - problem.a
        phase = 2.0 * np.pi * (x_arr - problem.a) / length
        endpoint_value = 0.5 * (problem.left_bc + problem.right_bc)
        values = endpoint_value + 0.5 * (1.0 - endpoint_value) * (1.0 - np.cos(phase))
        values = np.asarray(values, dtype=np.float64)
        values[np.isclose(x_arr, problem.a, atol=1.0e-14, rtol=1.0e-14)] = problem.left_bc
        values[np.isclose(x_arr, problem.b, atol=1.0e-14, rtol=1.0e-14)] = problem.right_bc
        return values
    if kind == "tanh":
        if problem.initial_tanh_width is None:
            width = (2.0**0.5) * float(problem.epsilon)
        else:
            width = float(problem.initial_tanh_width)
        width = max(width, 1.0e-12)
        raw = np.tanh((x_arr - problem.x_interface) / width)
        raw_left = float(np.tanh((problem.a - problem.x_interface) / width))
        raw_right = float(np.tanh((problem.b - problem.x_interface) / width))
        denom = raw_right - raw_left
        if abs(denom) <= 1.0e-14:
            slope = (problem.right_bc - problem.left_bc) / (problem.b - problem.a)
            return problem.left_bc + slope * (x_arr - problem.a)
        theta = (raw - raw_left) / denom
        return problem.left_bc + theta * (problem.right_bc - problem.left_bc)
    raise ValueError("initial_guess_kind must be 'smooth_periodic', 'tanh', 'linear', or 'zero'.")


def make_initial_iterate(problem: ProblemConfig) -> ArrayFunc:
    def evaluate(y: np.ndarray) -> np.ndarray:
        x = y_to_x(problem, y)
        return initial_guess_values(problem, x)

    return evaluate


def make_newton_space_functions(
    previous_iterate: ArrayFunc,
    grid: np.ndarray | None = None,
) -> tuple[ArrayFunc, ArrayFunc, Callable[[float], float] | None, Callable[[float], float] | None]:
    if grid is None:
        def c_func(y: np.ndarray, prev: ArrayFunc = previous_iterate) -> np.ndarray:
            u_prev = prev(y)
            return 3.0 * u_prev * u_prev - 1.0

        c_left_trace = None
        c_right_trace = None
    else:
        grid_arr = np.asarray(grid, dtype=np.float64)
        mids = 0.5 * (grid_arr[:-1] + grid_arr[1:])
        coeffs = 3.0 * previous_iterate(mids) ** 2 - 1.0

        def element_index_from_right(x: float) -> int:
            idx = int(np.searchsorted(grid_arr, float(x), side="right") - 1)
            return int(np.clip(idx, 0, coeffs.size - 1))

        def element_index_from_left(x: float) -> int:
            idx = int(np.searchsorted(grid_arr, float(x), side="left") - 1)
            return int(np.clip(idx, 0, coeffs.size - 1))

        def c_func(y: np.ndarray) -> np.ndarray:
            y_arr = np.asarray(y, dtype=np.float64)
            idx = np.searchsorted(grid_arr, y_arr, side="right") - 1
            idx = np.clip(idx, 0, coeffs.size - 1)
            return coeffs[idx]

        def c_left_trace(x: float) -> float:
            return float(coeffs[element_index_from_left(x)])

        def c_right_trace(x: float) -> float:
            return float(coeffs[element_index_from_right(x)])

    def f_func(y: np.ndarray, prev: ArrayFunc = previous_iterate) -> np.ndarray:
        u_prev = prev(y)
        return 2.0 * u_prev * u_prev * u_prev

    return c_func, f_func, c_left_trace, c_right_trace


def _constraint_system(
    problem: ProblemConfig,
    grid: np.ndarray,
    elems: object,
    c_func: ArrayFunc,
    f_func: ArrayFunc,
) -> tuple[np.ndarray, np.ndarray]:
    _kkt, _rhs, _h, _l, C, d = assemble_lagrange_kkt_system(
        grid=grid,
        elems=elems,
        f_func=f_func,
        m=problem.left_bc,
        n_dir=problem.right_bc,
        p=problem.jump_u,
        jump_du=transformed_jump_du(problem),
        xI=transformed_interface(problem),
        c_func=c_func,
        use_true_c=True,
        flux_jump_weights=problem.flux_jump_weights,
    )
    return C.toarray(), np.asarray(d, dtype=np.float64)


def _add_flux_jump_term(
    energy: float,
    grad: np.ndarray,
    problem: ProblemConfig,
    grid: np.ndarray,
    elems: object,
    z: np.ndarray,
) -> float:
    if problem.jump_du == 0.0:
        return energy

    interface_node = interface_node_from_grid(grid, transformed_interface(problem))
    e_left = interface_node - 1
    e_right = interface_node
    el_left = elems[e_left]
    el_right = elems[e_right]
    w_left, w_right = problem.flux_jump_weights
    q = (float(problem.epsilon) ** 2) * float(problem.jump_du)

    idx_left = np.array([2 * e_left, 2 * e_left + 1], dtype=int)
    idx_right = np.array([2 * e_right, 2 * e_right + 1], dtype=int)
    phi_left = np.array([el_left.phi1_R, el_left.phi2_R], dtype=np.float64)
    phi_right = np.array([el_right.phi1_L, el_right.phi2_L], dtype=np.float64)

    u_left = float(phi_left @ z[idx_left] + el_left.up_R)
    u_right = float(phi_right @ z[idx_right] + el_right.up_L)
    energy += q * (w_left * u_left + w_right * u_right)
    grad[idx_left] += q * w_left * phi_left
    grad[idx_right] += q * w_right * phi_right
    return energy


def energy_gradient_hessian(
    problem: ProblemConfig,
    grid: np.ndarray,
    elems: object,
    z: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    ndof = 2 * len(elems)
    grad = np.zeros(ndof, dtype=np.float64)
    hess = np.zeros((ndof, ndof), dtype=np.float64)
    energy = 0.0
    energy_scale = float(problem.epsilon)

    for e, el in enumerate(elems):
        idx = np.array([2 * e, 2 * e + 1], dtype=int)
        ze = z[idx]
        phi = np.vstack([el.phi1_q, el.phi2_q])
        dphi = np.vstack([el.dphi1_q, el.dphi2_q])
        u = el.up_q + ze[0] * el.phi1_q + ze[1] * el.phi2_q
        du = el.dup_q + ze[0] * el.dphi1_q + ze[1] * el.dphi2_q
        w = el.wq

        energy += energy_scale * float(np.sum(w * (0.5 * du * du + 0.25 * (u * u - 1.0) ** 2)))
        residual = u * u * u - u
        grad[idx] += energy_scale * np.array(
            [
                np.sum(w * (du * dphi[0] + residual * phi[0])),
                np.sum(w * (du * dphi[1] + residual * phi[1])),
            ],
            dtype=np.float64,
        )

        local_hess = dphi @ ((w[None, :] * dphi).T)
        curvature = 3.0 * u * u - 1.0
        local_hess += phi @ (((w * curvature)[None, :] * phi).T)
        hess[np.ix_(idx, idx)] += energy_scale * local_hess

    energy = _add_flux_jump_term(energy, grad, problem, grid, elems, z)
    return energy, grad, hess


def _augmented_value(
    energy: float,
    residual: np.ndarray,
    lam: np.ndarray,
    rho: float,
) -> float:
    return float(energy + lam @ residual + 0.5 * rho * (residual @ residual))


def solve_energy_in_fixed_space(
    problem: ProblemConfig,
    solver: SolverConfig,
    grid: np.ndarray,
    elems: object,
    C: np.ndarray,
    d: np.ndarray,
    z_initial: np.ndarray | None = None,
    lam_initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, InnerAlmHistory, float]:
    ndof = 2 * len(elems)
    mcon = C.shape[0]
    z = np.zeros(ndof, dtype=np.float64) if z_initial is None else np.asarray(z_initial, dtype=np.float64).copy()
    if z.shape != (ndof,):
        z = np.zeros(ndof, dtype=np.float64)

    lam = np.zeros(mcon, dtype=np.float64) if lam_initial is None else np.asarray(lam_initial, dtype=np.float64).copy()
    if lam.shape != (mcon,):
        lam = np.zeros(mcon, dtype=np.float64)

    Ct = C.T
    CtC = Ct @ C
    total_newton_iters = 0
    primal_inf = np.inf
    stationarity_inf = np.inf
    last_energy = np.inf

    for aug_iter in range(1, solver.max_aug_lag_iter + 1):
        for _newton_iter in range(1, solver.max_inner_newton_iter + 1):
            energy, grad_energy, hess_energy = energy_gradient_hessian(problem, grid, elems, z)
            residual = C @ z - d
            grad_aug = grad_energy + Ct @ (lam + solver.rho * residual)
            hess_aug = hess_energy + solver.rho * CtC
            grad_inf = float(np.max(np.abs(grad_aug)))
            total_newton_iters += 1
            last_energy = energy

            if grad_inf <= solver.tol_inner_grad:
                break

            try:
                step = np.linalg.solve(hess_aug, -grad_aug)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess_aug, -grad_aug, rcond=None)[0]

            directional = float(grad_aug @ step)
            if not np.isfinite(directional) or directional >= 0.0:
                step = -grad_aug
                directional = float(grad_aug @ step)

            current_aug = _augmented_value(energy, residual, lam, solver.rho)
            alpha = 1.0
            accepted = False
            for _line_iter in range(solver.line_search_max_iter):
                z_trial = z + alpha * step
                trial_energy, _trial_grad, _trial_hess = energy_gradient_hessian(problem, grid, elems, z_trial)
                trial_residual = C @ z_trial - d
                trial_aug = _augmented_value(trial_energy, trial_residual, lam, solver.rho)
                if np.isfinite(trial_aug) and trial_aug <= current_aug + solver.armijo * alpha * directional:
                    z = z_trial
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                z = z + alpha * step

        energy, grad_energy, _hess_energy = energy_gradient_hessian(problem, grid, elems, z)
        residual = C @ z - d
        lam = lam + solver.rho * residual
        stationarity = grad_energy + Ct @ lam
        primal_inf = float(np.max(np.abs(residual)))
        stationarity_inf = float(np.max(np.abs(stationarity)))
        last_energy = energy

        if primal_inf <= solver.tol_primal and stationarity_inf <= solver.tol_stationarity:
            history = InnerAlmHistory(
                aug_lag_iters=aug_iter,
                inner_newton_iters=total_newton_iters,
                primal_inf=primal_inf,
                stationarity_inf=stationarity_inf,
                converged=True,
                rho=float(solver.rho),
            )
            return z, lam, history, last_energy

    history = InnerAlmHistory(
        aug_lag_iters=solver.max_aug_lag_iter,
        inner_newton_iters=total_newton_iters,
        primal_inf=primal_inf,
        stationarity_inf=stationarity_inf,
        converged=False,
        rho=float(solver.rho),
    )
    return z, lam, history, last_energy


def solve_steady_allen_cahn_energy_alm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    verbose: bool = True,
) -> SolveResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()
    grid = build_grid(problem)
    y_interface = transformed_interface(problem)
    state = IterateState()
    history: list[SpaceIterationRecord] = []
    previous_iterate = make_initial_iterate(problem)
    z_guess: np.ndarray | None = None
    lam_guess: np.ndarray | None = None
    previous_energy = np.inf
    converged = False

    if verbose:
        print("=== Starting steady Allen-Cahn Newton-space energy ALM ===")

    t_start = time.perf_counter()
    for iteration in range(1, solver.max_space_iter + 1):
        c_func, f_func, c_left_trace, c_right_trace = make_newton_space_functions(previous_iterate)
        _grid, elems = build_elements(
            c_func=c_func,
            f_func=f_func,
            grid=grid,
            quad_n=problem.quad_n,
            xI=y_interface,
            n_seg_gauss=problem.n_seg_gauss,
            c_left_trace=c_left_trace,
            c_right_trace=c_right_trace,
            basis_kind=problem.basis_kind,
        )
        C, d = _constraint_system(problem, grid, elems, c_func, f_func)
        z_new, lam_guess, inner_history, energy = solve_energy_in_fixed_space(
            problem=problem,
            solver=solver,
            grid=grid,
            elems=elems,
            C=C,
            d=d,
            z_initial=z_guess,
            lam_initial=lam_guess,
        )

        y_check = np.linspace(grid[0], grid[-1], max(int(solver.error_check_points), 2))
        u_old = previous_iterate(y_check)
        state.update(elems=elems, z=z_new, forcing=f_func, config=problem, solver=solver)
        current_iterate = state.snapshot_evaluator()
        u_new = current_iterate(y_check)
        step_inf = float(np.max(np.abs(u_new - u_old)))
        energy_delta = float(abs(previous_energy - energy)) if np.isfinite(previous_energy) else np.inf

        record = SpaceIterationRecord(
            iteration=iteration,
            energy=float(energy),
            energy_delta=energy_delta,
            step_inf=step_inf,
            primal_inf=inner_history.primal_inf,
            stationarity_inf=inner_history.stationarity_inf,
            aug_lag_iters=inner_history.aug_lag_iters,
            inner_newton_iters=inner_history.inner_newton_iters,
            inner_converged=inner_history.converged,
        )
        history.append(record)

        if verbose:
            print(
                f"Space Iter {iteration:2d} | "
                f"E={energy:.8e} | "
                f"dE={energy_delta:.3e} | "
                f"step_inf={step_inf:.3e} | "
                f"primal={inner_history.primal_inf:.3e} | "
                f"stationarity={inner_history.stationarity_inf:.3e} | "
                f"ALM={inner_history.aug_lag_iters}"
            )

        energy_stable = (
            np.isfinite(energy_delta)
            and energy_delta <= solver.energy_tol * max(1.0, abs(float(energy)))
        )
        converged = inner_history.converged and (step_inf <= solver.space_tol or energy_stable)
        if converged:
            break

        previous_iterate = current_iterate
        previous_energy = float(energy)
        z_guess = z_new

    elapsed = time.perf_counter() - t_start
    if verbose:
        if converged:
            print("=> Steady Allen-Cahn energy ALM reached the target tolerance.")
        else:
            print("=> Warning: steady Allen-Cahn energy ALM hit the maximum number of iterations.")
        print(f"Total runtime: {elapsed:.4f} s")

    return SolveResult(
        problem=problem,
        solver=solver,
        state=state,
        history=history,
        elapsed_seconds=elapsed,
        converged=converged,
    )


def _resolve_output_dir(save_dir: Path | None) -> Path:
    if save_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    else:
        output_dir = Path(save_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_fdm_evaluator(solution: dict[str, np.ndarray], y_interface: float) -> ArrayFunc:
    def evaluate(y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        values = np.empty_like(y_arr)
        left_mask = y_arr <= y_interface
        right_mask = ~left_mask

        if np.any(left_mask):
            values[left_mask] = evaluate_fdm_on_side(solution, y_arr[left_mask], side="left")
        if np.any(right_mask):
            values[right_mask] = evaluate_fdm_on_side(solution, y_arr[right_mask], side="right")
        return values

    return evaluate


def solve_newton_fdm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    num_elements: int | None = None,
    verbose: bool = True,
    label: str = "FDM Newton",
) -> FdmSolveResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()

    resolved_num_elements = _resolve_fdm_num_elements(problem, solver, num_elements)
    y_grid = _build_fdm_grid(problem, resolved_num_elements)
    y_interface = transformed_interface(problem)
    previous_iterate = make_initial_iterate(problem)
    current_solution: dict[str, np.ndarray] | None = None
    history: list[FdmIterationRecord] = []

    if verbose:
        print(f"=== Starting {label} Newton iteration ===")

    t_start = time.perf_counter()
    for iteration in range(1, solver.fdm_max_newton_iter + 1):
        c_func, f_func, _c_left_trace, _c_right_trace = make_newton_space_functions(previous_iterate)
        new_solution = solve_interface_fdm(
            N=None,
            c_func=c_func,
            f_func=f_func,
            m=problem.left_bc,
            n_dir=problem.right_bc,
            p=problem.jump_u,
            qjump=transformed_jump_du(problem),
            grid=y_grid,
            xI=y_interface,
        )

        current_iterate = _make_fdm_evaluator(new_solution, y_interface)
        y_check = np.linspace(y_grid[0], y_grid[-1], max(int(solver.error_check_points), 2))
        u_old = previous_iterate(y_check)
        u_new = current_iterate(y_check)
        update_inf = float(np.max(np.abs(u_new - u_old)))

        history.append(FdmIterationRecord(iteration=iteration, update_inf=update_inf))
        current_solution = new_solution
        previous_iterate = current_iterate

        if verbose:
            print(f"{label} Newton Iter {iteration:2d} | L-inf update: {update_inf:.4e}")

        if update_inf < solver.fdm_newton_tol:
            elapsed = time.perf_counter() - t_start
            if verbose:
                print(f"=> {label} Newton iteration reached the target tolerance.")
                print(f"Total runtime: {elapsed:.4f} s")
            return FdmSolveResult(
                problem=problem,
                solver=solver,
                num_elements=resolved_num_elements,
                y_grid=y_grid,
                solution=current_solution,
                history=history,
                elapsed_seconds=elapsed,
                converged=True,
            )

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"=> Warning: {label} Newton iteration hit the maximum number of steps without converging.")
        print(f"Total runtime: {elapsed:.4f} s")
    return FdmSolveResult(
        problem=problem,
        solver=solver,
        num_elements=resolved_num_elements,
        y_grid=y_grid,
        solution=current_solution,
        history=history,
        elapsed_seconds=elapsed,
        converged=False,
    )


def _piecewise_plot_arrays(
    problem: ProblemConfig,
    points_per_side: int,
    evaluator: Callable[[np.ndarray, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_points = max(int(points_per_side), 2)
    x_left = np.linspace(problem.a, problem.x_interface, n_points)
    x_right = np.linspace(problem.x_interface, problem.b, n_points)
    u_left = evaluator(x_left, "left")
    u_right = evaluator(x_right, "right")
    x_plot = np.concatenate([x_left, np.array([np.nan]), x_right])
    u_plot = np.concatenate([u_left, np.array([np.nan]), u_right])
    return x_plot, u_plot


def _write_fdm_comparison_outputs(
    energy_result: SolveResult,
    output_dir: Path,
    verbose: bool = True,
) -> FdmComparisonResult:
    problem = energy_result.problem
    solver = energy_result.solver
    coarse_fdm_num_elements = _resolve_fdm_num_elements(problem, solver, num_elements=None)
    _validate_grid_count("problem.num_elements", problem.num_elements, problem)
    _validate_grid_count("solver.fdm_num_elements", coarse_fdm_num_elements, problem)
    _validate_grid_count("solver.fdm_reference_num_elements", solver.fdm_reference_num_elements, problem)

    if verbose:
        print(f"\n=== Solving FDM Newton comparison (N={coarse_fdm_num_elements}) ===")
    fdm_result = solve_newton_fdm(
        problem=problem,
        solver=solver,
        num_elements=coarse_fdm_num_elements,
        verbose=verbose,
        label=f"FDM Newton (N={coarse_fdm_num_elements})",
    )

    if verbose:
        print(f"\n=== Solving high-resolution FDM Newton reference (N={solver.fdm_reference_num_elements}) ===")
    fdm_reference_result = solve_newton_fdm(
        problem=problem,
        solver=solver,
        num_elements=solver.fdm_reference_num_elements,
        verbose=verbose,
        label=f"FDM Newton reference (N={solver.fdm_reference_num_elements})",
    )

    n_compare = max(int(solver.comparison_points_per_side), 2)
    x_left = np.linspace(problem.a, problem.x_interface, n_compare)
    x_right = np.linspace(problem.x_interface, problem.b, n_compare)
    reference_label = f"FDM Newton fine (N={fdm_reference_result.num_elements})"
    reference_values = (
        fdm_reference_result.evaluate_on_side(x_left, side="left"),
        fdm_reference_result.evaluate_on_side(x_right, side="right"),
    )
    method_values = {
        "Energy ALM": (
            energy_result.evaluate_on_side(x_left, side="left"),
            energy_result.evaluate_on_side(x_right, side="right"),
        ),
        f"FDM Newton (N={fdm_result.num_elements})": (
            fdm_result.evaluate_on_side(x_left, side="left"),
            fdm_result.evaluate_on_side(x_right, side="right"),
        ),
    }
    relative_l2_errors = compute_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        reference_label=reference_label,
    )

    solution_plot_path = output_dir / "steady_allen_cahn_vs_newton_fdm_solution.png"
    error_plot_path = output_dir / "steady_allen_cahn_vs_newton_fdm_errors.png"
    summary_path = output_dir / "steady_allen_cahn_vs_newton_fdm_summary.txt"

    x_ref, u_ref = _piecewise_plot_arrays(problem, n_compare, fdm_reference_result.evaluate_on_side)
    x_energy, u_energy = _piecewise_plot_arrays(problem, n_compare, energy_result.evaluate_on_side)
    x_fdm, u_fdm = _piecewise_plot_arrays(problem, n_compare, fdm_result.evaluate_on_side)
    plot_solutions(
        [
            (reference_label, x_ref, u_ref),
            ("Energy ALM", x_energy, u_energy),
            (f"FDM Newton (N={fdm_result.num_elements})", x_fdm, u_fdm),
        ],
        xI=problem.x_interface,
        save_path=solution_plot_path,
        reference_label=reference_label,
    )
    plot_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        xI=problem.x_interface,
        save_path=error_plot_path,
        reference_label=reference_label,
    )

    timings = {
        "Energy ALM": energy_result.elapsed_seconds,
        f"FDM Newton (N={fdm_result.num_elements})": fdm_result.elapsed_seconds,
        reference_label: fdm_reference_result.elapsed_seconds,
    }
    diagnostics = [
        "FDM Newton equation is solved in the stretched coordinate y = x / epsilon.",
        f"Energy ALM converged = {energy_result.converged}",
        f"FDM Newton (N={fdm_result.num_elements}) converged = {fdm_result.converged}",
        f"{reference_label} converged = {fdm_reference_result.converged}",
        f"Energy ALM space iterations = {len(energy_result.history)}",
        f"FDM Newton (N={fdm_result.num_elements}) Newton steps = {len(fdm_result.history)}",
        f"{reference_label} Newton steps = {len(fdm_reference_result.history)}",
    ]
    if energy_result.history:
        last_energy = energy_result.history[-1]
        diagnostics.extend(
            [
                f"Energy ALM final energy = {last_energy.energy:.12e}",
                f"Energy ALM final update = {last_energy.step_inf:.6e}",
                f"Energy ALM final primal residual = {last_energy.primal_inf:.6e}",
                f"Energy ALM final stationarity = {last_energy.stationarity_inf:.6e}",
            ]
        )
    if fdm_result.history:
        diagnostics.append(
            f"FDM Newton (N={fdm_result.num_elements}) final update = {fdm_result.history[-1].update_inf:.6e}"
        )
    if fdm_reference_result.history:
        diagnostics.append(f"{reference_label} final update = {fdm_reference_result.history[-1].update_inf:.6e}")

    write_comparison_summary(
        summary_path,
        timings=timings,
        errors=relative_l2_errors,
        extra_lines=diagnostics,
        errors_title=f"Relative L2 errors vs {reference_label}:",
    )

    if verbose:
        print(f"\n=== Relative L2 errors vs {reference_label} ===")
        for key, value in relative_l2_errors.items():
            print(f"{key:>48s}: {value:.6e}")
        print("\nSaved FDM comparison files:")
        print(summary_path)
        print(solution_plot_path)
        print(error_plot_path)

    return FdmComparisonResult(
        energy_result=energy_result,
        fdm_result=fdm_result,
        fdm_reference_result=fdm_reference_result,
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        relative_l2_errors=relative_l2_errors,
        timings=timings,
        solution_plot_path=solution_plot_path,
        error_plot_path=error_plot_path,
        summary_path=summary_path,
    )


def compare_steady_allen_cahn_energy_alm_with_fdm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> FdmComparisonResult:
    energy_result = solve_steady_allen_cahn_energy_alm(problem=problem, solver=solver, verbose=verbose)
    output_dir = _resolve_output_dir(save_dir)
    return _write_fdm_comparison_outputs(energy_result, output_dir=output_dir, verbose=verbose)


def plot_solution(result: SolveResult, save_path: Path) -> Path:
    x_plot, u_plot = result.plot_arrays()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, u_plot, linewidth=2.0, label="energy ALM")
    ax.axvline(result.problem.x_interface, color="0.35", linestyle="--", linewidth=1.0, label="interface")
    ax.set_title(r"Steady Allen-Cahn: $-\epsilon^2u'' + u^3 - u = 0$")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_history(result: SolveResult, save_path: Path) -> Path:
    iters = np.array([record.iteration for record in result.history], dtype=np.float64)
    energies = np.array([record.energy for record in result.history], dtype=np.float64)
    steps = np.array([record.step_inf for record in result.history], dtype=np.float64)
    stationarity = np.array([record.stationarity_inf for record in result.history], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(iters, energies, marker="o")
    axes[0].set_xlabel("space iteration")
    axes[0].set_ylabel("energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(iters, np.maximum(steps, np.finfo(float).tiny), marker="o")
    axes[1].set_xlabel("space iteration")
    axes[1].set_ylabel("L-inf update")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(iters, np.maximum(stationarity, np.finfo(float).tiny), marker="o")
    axes[2].set_xlabel("space iteration")
    axes[2].set_ylabel("stationarity")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def write_summary(result: SolveResult, save_path: Path) -> Path:
    lines = [
        "Steady Allen-Cahn Newton-space nonlinear energy ALM summary",
        "equation = -epsilon^2 u'' + u^3 - u = 0",
        "energy = int 0.5 epsilon^2 |u'|^2 + 0.25 (u^2 - 1)^2 dx",
        f"converged = {result.converged}",
        f"elapsed_seconds = {result.elapsed_seconds:.8e}",
        f"interval = [{result.problem.a:.12e}, {result.problem.b:.12e}]",
        f"x_interface = {result.problem.x_interface:.12e}",
        f"epsilon = {result.problem.epsilon:.12e}",
        "coordinate_transform = y = x / epsilon",
        (
            "transformed_interval = "
            f"[{x_to_y(result.problem, np.array([result.problem.a], dtype=np.float64))[0]:.12e}, "
            f"{x_to_y(result.problem, np.array([result.problem.b], dtype=np.float64))[0]:.12e}]"
        ),
        f"transformed_interface = {transformed_interface(result.problem):.12e}",
        f"left_bc = {result.problem.left_bc:.12e}",
        f"right_bc = {result.problem.right_bc:.12e}",
        f"num_elements = {result.problem.num_elements}",
        f"initial_guess_kind = {result.problem.initial_guess_kind}",
        f"space_iterations = {len(result.history)}",
    ]
    if result.history:
        last = result.history[-1]
        lines.extend(
            [
                f"final_energy = {last.energy:.12e}",
                f"final_energy_delta = {last.energy_delta:.12e}",
                f"final_step_inf = {last.step_inf:.12e}",
                f"final_primal_inf = {last.primal_inf:.12e}",
                f"final_stationarity_inf = {last.stationarity_inf:.12e}",
                f"final_aug_lag_iters = {last.aug_lag_iters}",
                f"final_inner_newton_iters = {last.inner_newton_iters}",
                f"final_inner_converged = {last.inner_converged}",
            ]
        )

    lines.append("")
    lines.append("iteration,energy,energy_delta,step_inf,primal_inf,stationarity_inf,aug_lag_iters,inner_newton_iters")
    for record in result.history:
        lines.append(
            f"{record.iteration},"
            f"{record.energy:.12e},"
            f"{record.energy_delta:.12e},"
            f"{record.step_inf:.12e},"
            f"{record.primal_inf:.12e},"
            f"{record.stationarity_inf:.12e},"
            f"{record.aug_lag_iters},"
            f"{record.inner_newton_iters}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


def save_solution_data(result: SolveResult, save_path: Path) -> Path:
    x_plot, u_plot = result.plot_arrays()
    y_plot = x_to_y(result.problem, x_plot)
    iterations = np.array([record.iteration for record in result.history], dtype=np.int64)
    energies = np.array([record.energy for record in result.history], dtype=np.float64)
    steps = np.array([record.step_inf for record in result.history], dtype=np.float64)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x=x_plot,
        y=y_plot,
        u=u_plot,
        iterations=iterations,
        energies=energies,
        steps=steps,
    )
    return save_path


def run_experiment(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    save_dir: Path | None = None,
    compare_fdm: bool = True,
    verbose: bool = True,
) -> SolveResult:
    result = solve_steady_allen_cahn_energy_alm(problem=problem, solver=solver, verbose=verbose)
    output_dir = _resolve_output_dir(save_dir)
    solution_path = plot_solution(result, output_dir / "steady_allen_cahn_solution.png")
    history_path = plot_history(result, output_dir / "steady_allen_cahn_history.png")
    data_path = save_solution_data(result, output_dir / "steady_allen_cahn_solution.npz")
    summary_path = write_summary(result, output_dir / "steady_allen_cahn_summary.txt")
    comparison_result: FdmComparisonResult | None = None
    if compare_fdm:
        comparison_result = _write_fdm_comparison_outputs(result, output_dir=output_dir, verbose=verbose)

    if verbose:
        print("Saved files:")
        print(summary_path)
        print(solution_path)
        print(history_path)
        print(data_path)
        if comparison_result is not None:
            print(comparison_result.summary_path)
            print(comparison_result.solution_plot_path)
            print(comparison_result.error_plot_path)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run steady Allen-Cahn Newton-space energy ALM.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory relative to this experiment.")
    parser.add_argument("--num-elements", type=int, default=None, help="Override the default TFPM element count.")
    parser.add_argument("--epsilon", type=float, default=None, help="Positive epsilon in -epsilon^2 u'' + u^3 - u = 0.")
    parser.add_argument("--fdm-elements", type=int, default=None, help="Override the comparison FDM element count.")
    parser.add_argument(
        "--fdm-reference-elements",
        type=int,
        default=None,
        help="Override the high-resolution FDM reference element count.",
    )
    parser.add_argument("--fdm-newton-tol", type=float, default=None, help="Override the FDM Newton update tolerance.")
    parser.add_argument(
        "--fdm-max-newton-iter",
        type=int,
        default=None,
        help="Override the maximum number of FDM Newton iterations.",
    )
    parser.add_argument(
        "--comparison-points-per-side",
        type=int,
        default=None,
        help="Override the number of error comparison points on each side of the interface.",
    )
    parser.add_argument(
        "--initial-tanh-width",
        type=float,
        default=None,
        help="Override tanh initial width. The default is sqrt(2) * epsilon.",
    )
    parser.add_argument(
        "--initial",
        choices=("smooth_periodic", "smooth-periodic", "cosine", "cos", "tanh", "linear", "zero"),
        default=None,
        help="Override the initial iterate used to build the first Newton space.",
    )
    parser.add_argument("--no-fdm-comparison", action="store_true", help="Only run the energy ALM solve.")
    parser.add_argument("--quiet", action="store_true", help="Suppress iteration logs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    problem = ProblemConfig()
    solver = SolverConfig()
    if args.num_elements is not None:
        problem.num_elements = int(args.num_elements)
    if args.epsilon is not None:
        problem.epsilon = float(args.epsilon)
    if args.initial_tanh_width is not None:
        problem.initial_tanh_width = float(args.initial_tanh_width)
    if args.initial is not None:
        problem.initial_guess_kind = str(args.initial)
    if args.fdm_elements is not None:
        solver.fdm_num_elements = int(args.fdm_elements)
    if args.fdm_reference_elements is not None:
        solver.fdm_reference_num_elements = int(args.fdm_reference_elements)
    if args.fdm_newton_tol is not None:
        solver.fdm_newton_tol = float(args.fdm_newton_tol)
    if args.fdm_max_newton_iter is not None:
        solver.fdm_max_newton_iter = int(args.fdm_max_newton_iter)
    if args.comparison_points_per_side is not None:
        solver.comparison_points_per_side = int(args.comparison_points_per_side)
    run_experiment(
        problem=problem,
        solver=solver,
        save_dir=args.output_dir,
        compare_fdm=not args.no_fdm_comparison,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
