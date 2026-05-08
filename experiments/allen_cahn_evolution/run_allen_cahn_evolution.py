from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

if __package__ is None or __package__ == "":
    _HERE = Path(__file__).resolve().parent
    _EXPERIMENTS = _HERE.parent
    _ROOT = _EXPERIMENTS.parent
    _PARENT = _ROOT.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))
    __package__ = f"{_ROOT.name}.{_EXPERIMENTS.name}.{_HERE.name}"

from ...core.methods_tfpm import LagrangeHistory, assemble_lagrange_kkt_system, solve_lagrange_kkt
from ...core.tfpm_local import build_elements, evaluate_tfpm_state_on_side, make_trace_functions
from .analysis_plot import plot_energy_histories, plot_errors_vs_reference, plot_final_solutions, write_summary
from .problem import (
    ArrayFunc,
    ExperimentConfig,
    NumericalConfig,
    PhysicalGrid,
    ProblemConfig,
    ReferenceConfig,
    SampledState,
    TransformedGrid,
    build_physical_grid,
    build_transformed_grid,
    diffusion_piecewise,
    make_initial_state,
    nonlinearity,
    potential,
    sampled_state_from_values,
    transformed_coefficient_from_physical,
    transformed_rhs_from_physical,
    validate_experiment,
    x_to_y,
)


@dataclass(slots=True)
class LinearSolveResult:
    coeff_y_func: ArrayFunc
    rhs_y_func: ArrayFunc
    y_grid: np.ndarray
    y_interface: float
    solved_on_physical_grid: bool
    elems: object
    z: np.ndarray
    lagrange_history: LagrangeHistory


@dataclass(slots=True)
class StepAttempt:
    state: SampledState
    energy: float
    linear_result: LinearSolveResult
    inner_iterations: int
    linear_solves: int
    lagrange_solves: int
    inner_converged: bool
    stop_reason: str


@dataclass(slots=True)
class StepRecord:
    step: int
    time: float
    dt: float
    mode: str
    stage: str
    inner_iterations: int
    linear_solves: int
    lagrange_solves: int
    lagrange_primal_inf: float
    lagrange_stationarity_inf: float
    inner_converged: bool
    step_l2: float
    step_inf: float
    energy: float
    note: str


@dataclass(slots=True)
class Snapshot:
    time: float
    state: SampledState


@dataclass(slots=True)
class EvolutionResult:
    mode: str
    problem: ProblemConfig
    numerical: NumericalConfig
    initial_time: float
    final_time: float
    completed: bool
    all_inner_converged: bool
    initial_state: SampledState
    final_state: SampledState
    initial_energy: float
    final_linear_result: LinearSolveResult | None
    history: list[StepRecord]
    snapshots: list[Snapshot]
    elapsed_seconds: float


@dataclass(slots=True)
class FdmReferenceResult:
    label: str
    grid: PhysicalGrid
    times: np.ndarray
    states: np.ndarray
    energies: np.ndarray
    final_state: SampledState
    elapsed_seconds: float
    method: str
    nfev: int
    njev: int
    nlu: int
    message: str


def _print(enabled: bool, text: str) -> None:
    if enabled:
        print(text, flush=True)


def _resolve_output_dir(output_dir: Path) -> Path:
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _normalize_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "scheme1": "scheme1",
        "scheme_1": "scheme1",
        "schemei": "scheme1",
        "scheme_i": "scheme1",
        "s1": "scheme1",
        "ptc": "scheme1",
        "scheme2": "scheme2",
        "scheme_2": "scheme2",
        "schemeii": "scheme2",
        "scheme_ii": "scheme2",
        "s2": "scheme2",
        "scheme3": "scheme3",
        "scheme_3": "scheme3",
        "schemeiii": "scheme3",
        "scheme_iii": "scheme3",
        "s3": "scheme3",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("mode must be one of 'scheme1', 'scheme2', 'scheme3', or 'all'.") from exc


def _mode_label(mode: str) -> str:
    return {"scheme1": "Scheme I", "scheme2": "Scheme II", "scheme3": "Scheme III"}[mode]


def _state_l2_norm(grid: PhysicalGrid, values: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.asarray(values, dtype=np.float64) ** 2) * grid.h))


def _relative_step_l2(grid: PhysicalGrid, u_new: np.ndarray, u_old: np.ndarray) -> float:
    base = max(_state_l2_norm(grid, u_old), 1.0e-14)
    return _state_l2_norm(grid, np.asarray(u_new, dtype=np.float64) - np.asarray(u_old, dtype=np.float64)) / base


def _energy_on_state(
    problem: ProblemConfig,
    state: SampledState,
    ux: np.ndarray | None = None,
) -> float:
    x = state.x
    u = state.u
    if ux is None:
        dx = np.diff(x)
        slopes = np.diff(u) / dx
        faces = 0.5 * (x[:-1] + x[1:])
        grad = 0.5 * np.sum(diffusion_piecewise(problem, faces) * slopes**2 * dx)
    else:
        grad_density = 0.5 * diffusion_piecewise(problem, x) * np.asarray(ux, dtype=np.float64) ** 2
        grad = float(np.trapezoid(grad_density, x))
    bulk = potential(u)
    return float(grad + np.trapezoid(bulk, x))


def _reference_time_grid(experiment: ExperimentConfig) -> np.ndarray:
    t0 = float(experiment.time.initial_time)
    tf = float(experiment.time.final_time)
    dt = float(experiment.time.dt)
    if tf <= t0:
        return np.array([t0], dtype=np.float64)
    n_full = int(np.floor((tf - t0) / dt + 1.0e-12))
    times = t0 + dt * np.arange(n_full + 1, dtype=np.float64)
    if times[-1] < tf - 1.0e-14:
        times = np.concatenate([times, np.array([tf], dtype=np.float64)])
    else:
        times[-1] = tf
    return times


def _fdm_reference_label(reference: ReferenceConfig) -> str:
    return f"FDM reference ({reference.method}, N={reference.n_segments})"


def solve_fdm_reference(experiment: ExperimentConfig) -> FdmReferenceResult:
    problem = experiment.problem
    reference = experiment.reference
    grid = build_physical_grid(problem, reference.n_segments)
    initial_state = make_initial_state(problem, grid.x)
    times = _reference_time_grid(experiment)
    label = _fdm_reference_label(reference)
    method = str(reference.method).strip()

    if times.size == 1:
        states = initial_state.u[np.newaxis, :].copy()
        energies = np.array([_energy_on_state(problem, initial_state)], dtype=np.float64)
        return FdmReferenceResult(
            label=label,
            grid=grid,
            times=times,
            states=states,
            energies=energies,
            final_state=initial_state,
            elapsed_seconds=0.0,
            method=method,
            nfev=0,
            njev=0,
            nlu=0,
            message="initial time equals final time",
        )

    x = grid.x
    h = float(grid.h)
    n_interior = x.size - 2
    face_x = 0.5 * (x[:-1] + x[1:])
    face_a = diffusion_piecewise(problem, face_x)
    lower_upper = face_a[1:n_interior] / h**2
    linear_diag = -(face_a[:n_interior] + face_a[1 : n_interior + 1]) / h**2
    y0 = initial_state.u[1:-1].copy()

    def full_state(y: np.ndarray) -> np.ndarray:
        u = np.empty(x.size, dtype=np.float64)
        u[0] = problem.bc_left
        u[-1] = problem.bc_right
        u[1:-1] = np.asarray(y, dtype=np.float64)
        return u

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        u = full_state(y)
        flux_right = face_a[1:] * (u[2:] - u[1:-1]) / h
        flux_left = face_a[:-1] * (u[1:-1] - u[:-2]) / h
        diffusion = (flux_right - flux_left) / h
        return diffusion - nonlinearity(u[1:-1])

    def jac(_t: float, y: np.ndarray):
        nonlinear_diag = 1.0 - 3.0 * np.asarray(y, dtype=np.float64) ** 2
        return diags(
            diagonals=(lower_upper, linear_diag + nonlinear_diag, lower_upper),
            offsets=(-1, 0, 1),
            shape=(n_interior, n_interior),
            format="csc",
        )

    max_step = np.inf if reference.max_step is None else float(reference.max_step)
    t_start = time.perf_counter()
    sol = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        y0,
        method=method,
        t_eval=times,
        rtol=float(reference.rtol),
        atol=float(reference.atol),
        jac=jac,
        max_step=max_step,
    )
    elapsed = time.perf_counter() - t_start
    if not sol.success:
        raise RuntimeError(f"{label} failed: {sol.message}")

    states = np.empty((times.size, x.size), dtype=np.float64)
    states[:, 0] = problem.bc_left
    states[:, -1] = problem.bc_right
    states[:, 1:-1] = sol.y.T
    energies = np.array(
        [_energy_on_state(problem, SampledState(x=x, u=states[i])) for i in range(states.shape[0])],
        dtype=np.float64,
    )
    final_state = SampledState(x=x.copy(), u=states[-1].copy())
    return FdmReferenceResult(
        label=label,
        grid=grid,
        times=times.copy(),
        states=states,
        energies=energies,
        final_state=final_state,
        elapsed_seconds=elapsed,
        method=method,
        nfev=int(getattr(sol, "nfev", 0)),
        njev=int(getattr(sol, "njev", 0)),
        nlu=int(getattr(sol, "nlu", 0)),
        message=str(sol.message),
    )


def _relative_l2_error(x: np.ndarray, values: np.ndarray, reference_values: np.ndarray) -> float:
    numerator = np.trapezoid((values - reference_values) ** 2, x)
    denominator = np.trapezoid(reference_values**2, x)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def _relative_linf_error(values: np.ndarray, reference_values: np.ndarray) -> float:
    numerator = float(np.max(np.abs(values - reference_values)))
    denominator = float(np.max(np.abs(reference_values)))
    return numerator / max(denominator, 1.0e-15)


def _solve_linear_tfpm_lagrange(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    coeff_x_func: ArrayFunc,
    rhs_x_func: ArrayFunc,
) -> LinearSolveResult:
    problem = experiment.problem
    numerical = experiment.numerical
    lagrange = experiment.lagrange

    coeff_y_func = transformed_coefficient_from_physical(problem, coeff_x_func)
    rhs_y_func = transformed_rhs_from_physical(problem, rhs_x_func)
    c_left_trace, c_right_trace = make_trace_functions(coeff_y_func, transformed_grid.y_interface)

    _grid, elems = build_elements(
        c_func=coeff_y_func,
        f_func=rhs_y_func,
        grid=transformed_grid.y,
        quad_n=numerical.quad_n,
        xI=transformed_grid.y_interface,
        n_seg_gauss=numerical.n_seg_gauss,
        c_left_trace=c_left_trace,
        c_right_trace=c_right_trace,
        basis_kind=numerical.tfpm_basis,
    )

    if transformed_grid.solved_on_physical_grid:
        a_interface = float(diffusion_piecewise(problem, np.array([problem.x_interface], dtype=np.float64))[0])
        jump_du = problem.jump_flux / a_interface
    else:
        jump_du = problem.jump_flux

    K, rhs, H, l_vec, C, d = assemble_lagrange_kkt_system(
        grid=transformed_grid.y,
        elems=elems,
        f_func=rhs_y_func,
        m=problem.bc_left,
        n_dir=problem.bc_right,
        p=problem.jump_u,
        jump_du=jump_du,
        xI=transformed_grid.y_interface,
        c_func=coeff_y_func,
        use_true_c=numerical.use_true_c_lagrange,
        flux_jump_weights=(0.5, 0.5),
    )

    z, _lam, lagrange_history = solve_lagrange_kkt(
        K=K,
        rhs=rhs,
        H=H,
        l=l_vec,
        C=C,
        d=d,
        residual_tol=float(lagrange.residual_tol),
    )
    return LinearSolveResult(
        coeff_y_func=coeff_y_func,
        rhs_y_func=rhs_y_func,
        y_grid=transformed_grid.y.copy(),
        y_interface=float(transformed_grid.y_interface),
        solved_on_physical_grid=bool(transformed_grid.solved_on_physical_grid),
        elems=elems,
        z=np.asarray(z, dtype=np.float64),
        lagrange_history=lagrange_history,
    )


def _evaluate_linear_solution_on_x(
    problem: ProblemConfig,
    numerical: NumericalConfig,
    linear: LinearSolveResult,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_eval = x_to_y(problem, x_eval)
    left_mask = x_eval <= problem.x_interface
    right_mask = ~left_mask
    u = np.zeros_like(x_eval)
    ux = np.zeros_like(x_eval)

    if np.any(left_mask):
        ul, duy_l = evaluate_tfpm_state_on_side(
            elems=linear.elems,
            z=linear.z,
            f_func=linear.rhs_y_func,
            x_eval=y_eval[left_mask],
            side="left",
            xI=linear.y_interface,
            n_seg_gauss=numerical.n_seg_gauss,
        )
        u[left_mask] = ul
        if linear.solved_on_physical_grid:
            ux[left_mask] = duy_l
        else:
            ux[left_mask] = duy_l / (problem.eps_left**2)

    if np.any(right_mask):
        ur, duy_r = evaluate_tfpm_state_on_side(
            elems=linear.elems,
            z=linear.z,
            f_func=linear.rhs_y_func,
            x_eval=y_eval[right_mask],
            side="right",
            xI=linear.y_interface,
            n_seg_gauss=numerical.n_seg_gauss,
        )
        u[right_mask] = ur
        if linear.solved_on_physical_grid:
            ux[right_mask] = duy_r
        else:
            ux[right_mask] = duy_r / (problem.eps_right**2)

    if x_eval.size:
        u[np.isclose(x_eval, problem.x_left, atol=1.0e-14, rtol=1.0e-14)] = problem.bc_left
        u[np.isclose(x_eval, problem.x_right, atol=1.0e-14, rtol=1.0e-14)] = problem.bc_right
    return u, ux


def _candidate_from_linear(
    experiment: ExperimentConfig,
    monitor_grid: PhysicalGrid,
    linear: LinearSolveResult,
) -> tuple[SampledState, float]:
    u, ux = _evaluate_linear_solution_on_x(experiment.problem, experiment.numerical, linear, monitor_grid.x)
    state = sampled_state_from_values(experiment.problem, monitor_grid.x, u)
    energy = _energy_on_state(experiment.problem, state, ux=ux)
    return state, energy


def _solve_scheme1_step(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    monitor_grid: PhysicalGrid,
    current_state: SampledState,
    dt: float,
) -> StepAttempt:
    mu = 1.0 / dt + float(experiment.scheme1.stabilization_shift)

    def coeff_x(x: np.ndarray, mu: float = mu) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return mu * np.ones_like(x)

    def rhs_x(x: np.ndarray, state: SampledState = current_state, mu: float = mu) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        un = state.eval(x)
        return -nonlinearity(un) + mu * un

    linear = _solve_linear_tfpm_lagrange(experiment, transformed_grid, coeff_x, rhs_x)
    state, energy = _candidate_from_linear(experiment, monitor_grid, linear)
    return StepAttempt(
        state=state,
        energy=energy,
        linear_result=linear,
        inner_iterations=1,
        linear_solves=1,
        lagrange_solves=int(linear.lagrange_history.iter),
        inner_converged=True,
        stop_reason="single linearized Scheme I step",
    )


def _solve_scheme2_step(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    monitor_grid: PhysicalGrid,
    current_state: SampledState,
    dt: float,
    cfg,
) -> StepAttempt:
    inv_dt = 1.0 / dt
    uk = current_state
    last_state: SampledState | None = None
    last_energy = np.nan
    last_linear: LinearSolveResult | None = None
    total_lagrange_solves = 0

    for inner in range(1, int(cfg.max_inner) + 1):
        def coeff_x(x: np.ndarray, uk: SampledState = uk) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            ukx = uk.eval(x)
            return inv_dt + ukx**2 + 2.0

        def rhs_x(x: np.ndarray, uk: SampledState = uk, un: SampledState = current_state) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            ukx = uk.eval(x)
            unx = un.eval(x)
            return (inv_dt + 1.0) * unx + 2.0 * ukx

        linear = _solve_linear_tfpm_lagrange(experiment, transformed_grid, coeff_x, rhs_x)
        total_lagrange_solves += int(linear.lagrange_history.iter)
        candidate_state, candidate_energy = _candidate_from_linear(experiment, monitor_grid, linear)
        rel_inner = _relative_step_l2(monitor_grid, candidate_state.u, uk.u)

        last_state = candidate_state
        last_energy = candidate_energy
        last_linear = linear
        if rel_inner <= float(cfg.inner_tol_step_rel):
            return StepAttempt(
                state=candidate_state,
                energy=candidate_energy,
                linear_result=linear,
                inner_iterations=inner,
                linear_solves=inner,
                lagrange_solves=total_lagrange_solves,
                inner_converged=True,
                stop_reason="inner fixed-point tolerance reached",
            )
        uk = candidate_state

    assert last_state is not None and last_linear is not None
    return StepAttempt(
        state=last_state,
        energy=float(last_energy),
        linear_result=last_linear,
        inner_iterations=int(cfg.max_inner),
        linear_solves=int(cfg.max_inner),
        lagrange_solves=total_lagrange_solves,
        inner_converged=False,
        stop_reason="maximum inner iterations reached",
    )


def _scheme3_initial_guess(
    problem: ProblemConfig,
    prev_state: SampledState,
    current_state: SampledState,
    weight: float,
) -> SampledState:
    values = current_state.u + float(weight) * (current_state.u - prev_state.u)
    return sampled_state_from_values(problem, current_state.x, values)


def _solve_scheme3_step(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    monitor_grid: PhysicalGrid,
    prev_state: SampledState,
    current_state: SampledState,
    dt: float,
) -> StepAttempt:
    problem = experiment.problem
    cfg = experiment.scheme3
    inv_2dt = 0.5 / dt
    inv_dt_term = 1.5 / dt
    uk = _scheme3_initial_guess(problem, prev_state, current_state, cfg.extrapolation_weight)
    last_state: SampledState | None = None
    last_energy = np.nan
    last_linear: LinearSolveResult | None = None
    total_lagrange_solves = 0

    for inner in range(1, int(cfg.max_inner) + 1):
        def coeff_x(
            x: np.ndarray,
            uk: SampledState = uk,
            un: SampledState = current_state,
            unm1: SampledState = prev_state,
        ) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            vk = 1.5 * uk.eval(x) - 0.5 * un.eval(x)
            vn = 1.5 * un.eval(x) - 0.5 * unm1.eval(x)
            theta = vk**2 + vn**2 + 1.0
            return inv_dt_term + 0.375 * theta

        def rhs_x(
            x: np.ndarray,
            uk: SampledState = uk,
            un: SampledState = current_state,
            unm1: SampledState = prev_state,
        ) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            unx = un.eval(x)
            unm1x = unm1.eval(x)
            vk = 1.5 * uk.eval(x) - 0.5 * unx
            vn = 1.5 * unx - 0.5 * unm1x
            theta = vk**2 + vn**2 + 1.0
            return (
                (4.0 * unx - unm1x) * inv_2dt
                + 0.125 * theta * unx
                + 0.5 * (vn + 1.5 * vk - 0.5 * vn**3 - 0.5 * vk**2 * vn)
            )

        linear = _solve_linear_tfpm_lagrange(experiment, transformed_grid, coeff_x, rhs_x)
        total_lagrange_solves += int(linear.lagrange_history.iter)
        candidate_state, candidate_energy = _candidate_from_linear(experiment, monitor_grid, linear)
        rel_inner = _relative_step_l2(monitor_grid, candidate_state.u, uk.u)

        last_state = candidate_state
        last_energy = candidate_energy
        last_linear = linear
        if rel_inner <= float(cfg.inner_tol_step_rel):
            return StepAttempt(
                state=candidate_state,
                energy=candidate_energy,
                linear_result=linear,
                inner_iterations=inner,
                linear_solves=inner,
                lagrange_solves=total_lagrange_solves,
                inner_converged=True,
                stop_reason="inner fixed-point tolerance reached",
            )
        uk = candidate_state

    assert last_state is not None and last_linear is not None
    return StepAttempt(
        state=last_state,
        energy=float(last_energy),
        linear_result=last_linear,
        inner_iterations=int(cfg.max_inner),
        linear_solves=int(cfg.max_inner),
        lagrange_solves=total_lagrange_solves,
        inner_converged=False,
        stop_reason="maximum inner iterations reached",
    )


def _make_step_record(
    *,
    step: int,
    time_value: float,
    dt: float,
    mode: str,
    stage: str,
    old_state: SampledState,
    attempt: StepAttempt,
    monitor_grid: PhysicalGrid,
) -> StepRecord:
    delta = attempt.state.u - old_state.u
    lag = attempt.linear_result.lagrange_history
    return StepRecord(
        step=step,
        time=float(time_value),
        dt=float(dt),
        mode=mode,
        stage=stage,
        inner_iterations=int(attempt.inner_iterations),
        linear_solves=int(attempt.linear_solves),
        lagrange_solves=int(attempt.lagrange_solves),
        lagrange_primal_inf=float(lag.primal_inf),
        lagrange_stationarity_inf=float(lag.stationarity_inf),
        inner_converged=bool(attempt.inner_converged),
        step_l2=_state_l2_norm(monitor_grid, delta),
        step_inf=float(np.max(np.abs(delta))),
        energy=float(attempt.energy),
        note=attempt.stop_reason,
    )


def solve_evolution(
    mode: str,
    experiment: ExperimentConfig | None = None,
) -> EvolutionResult:
    experiment = ExperimentConfig() if experiment is None else experiment
    validate_experiment(experiment)
    mode = _normalize_mode(mode)

    problem = experiment.problem
    numerical = experiment.numerical
    time_cfg = experiment.time
    verbose = experiment.output.verbose

    physical_grid = build_physical_grid(problem, numerical.n_elements)
    transformed_grid = build_transformed_grid(problem, physical_grid)
    monitor_grid = build_physical_grid(problem, numerical.monitor_segments)

    initial_state = make_initial_state(problem, monitor_grid.x)
    initial_energy = _energy_on_state(problem, initial_state)

    current_state = initial_state
    prev_state: SampledState | None = None
    current_time = float(time_cfg.initial_time)
    final_time = float(time_cfg.final_time)
    base_dt = float(time_cfg.dt)
    history: list[StepRecord] = []
    snapshots: list[Snapshot] = [Snapshot(time=current_time, state=initial_state)]
    final_linear: LinearSolveResult | None = None
    all_inner_converged = True

    _print(verbose, f"=== Running {_mode_label(mode)} for Allen-Cahn evolution ===")
    t_start = time.perf_counter()
    step = 0
    while current_time < final_time - 1.0e-14:
        dt = min(base_dt, final_time - current_time)
        t_new = current_time + dt
        step += 1
        old_state = current_state

        if mode == "scheme1":
            stage = "scheme1"
            attempt = _solve_scheme1_step(experiment, transformed_grid, monitor_grid, current_state, dt)
            current_state = attempt.state
        elif mode == "scheme2":
            stage = "scheme2"
            attempt = _solve_scheme2_step(
                experiment, transformed_grid, monitor_grid, current_state, dt, experiment.scheme2
            )
            current_state = attempt.state
        else:
            if prev_state is None:
                startup_mode = _normalize_mode(experiment.scheme3.startup_mode)
                if startup_mode == "scheme3":
                    raise ValueError("scheme3.startup_mode must be 'scheme1' or 'scheme2'.")
                stage = f"scheme3-startup-{startup_mode}"
                if startup_mode == "scheme1":
                    attempt = _solve_scheme1_step(experiment, transformed_grid, monitor_grid, current_state, dt)
                else:
                    attempt = _solve_scheme2_step(
                        experiment, transformed_grid, monitor_grid, current_state, dt, experiment.scheme3
                    )
                prev_state = old_state
                current_state = attempt.state
            else:
                stage = "scheme3"
                attempt = _solve_scheme3_step(
                    experiment, transformed_grid, monitor_grid, prev_state, current_state, dt
                )
                prev_state = old_state
                current_state = attempt.state

        final_linear = attempt.linear_result
        all_inner_converged = all_inner_converged and attempt.inner_converged
        record = _make_step_record(
            step=step,
            time_value=t_new,
            dt=dt,
            mode=mode,
            stage=stage,
            old_state=old_state,
            attempt=attempt,
            monitor_grid=monitor_grid,
        )
        history.append(record)
        current_time = t_new

        if numerical.snapshot_stride > 0 and (step % int(numerical.snapshot_stride) == 0):
            snapshots.append(Snapshot(time=current_time, state=current_state))

        _print(
            verbose,
            f"{_mode_label(mode)} step={step:04d}, t={current_time:.6e}, "
            f"E={record.energy:.8e}, step_l2={record.step_l2:.3e}, "
            f"inner={record.inner_iterations}",
        )

    elapsed = time.perf_counter() - t_start
    completed = current_time >= final_time - 1.0e-14
    if not snapshots or snapshots[-1].time < current_time - 1.0e-14:
        snapshots.append(Snapshot(time=current_time, state=current_state))

    return EvolutionResult(
        mode=mode,
        problem=problem,
        numerical=numerical,
        initial_time=float(time_cfg.initial_time),
        final_time=float(current_time),
        completed=completed,
        all_inner_converged=all_inner_converged,
        initial_state=initial_state,
        final_state=current_state,
        initial_energy=initial_energy,
        final_linear_result=final_linear,
        history=history,
        snapshots=snapshots,
        elapsed_seconds=elapsed,
    )


def _history_arrays(result: EvolutionResult) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([result.initial_time] + [record.time for record in result.history], dtype=np.float64)
    energies = np.array([result.initial_energy] + [record.energy for record in result.history], dtype=np.float64)
    return times, energies


def _save_result_npz(out_dir: Path, result: EvolutionResult) -> Path:
    path = out_dir / f"{result.mode}_state_history.npz"
    times, energies = _history_arrays(result)
    np.savez(
        path,
        monitor_x=result.final_state.x,
        initial_u=result.initial_state.u,
        final_u=result.final_state.u,
        times=times,
        energies=energies,
        step=np.array([record.step for record in result.history], dtype=np.int32),
        dt=np.array([record.dt for record in result.history], dtype=np.float64),
        step_l2=np.array([record.step_l2 for record in result.history], dtype=np.float64),
        step_inf=np.array([record.step_inf for record in result.history], dtype=np.float64),
        inner_iterations=np.array([record.inner_iterations for record in result.history], dtype=np.int32),
        lagrange_solves=np.array([record.lagrange_solves for record in result.history], dtype=np.int32),
        inner_converged=np.array([record.inner_converged for record in result.history], dtype=np.bool_),
    )
    return path


def _save_reference_npz(out_dir: Path, reference: FdmReferenceResult) -> Path:
    path = out_dir / "fdm_reference_state_history.npz"
    np.savez(
        path,
        x=reference.grid.x,
        times=reference.times,
        states=reference.states,
        final_u=reference.final_state.u,
        energies=reference.energies,
        method=np.array([reference.method]),
        n_segments=np.array([reference.grid.x.size - 1], dtype=np.int32),
        nfev=np.array([reference.nfev], dtype=np.int32),
        njev=np.array([reference.njev], dtype=np.int32),
        nlu=np.array([reference.nlu], dtype=np.int32),
    )
    return path


def _compute_final_errors_vs_reference(
    experiment: ExperimentConfig,
    results: dict[str, EvolutionResult],
    reference: FdmReferenceResult,
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, np.ndarray, np.ndarray]]]:
    x_error = np.linspace(
        experiment.problem.x_left,
        experiment.problem.x_right,
        int(experiment.reference.comparison_samples),
    )
    ref_values = reference.final_state.eval(x_error)
    metrics: dict[str, tuple[float, float]] = {}
    error_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for mode, result in results.items():
        values = result.final_state.eval(x_error)
        diff = values - ref_values
        metrics[_mode_label(mode)] = (
            _relative_l2_error(x_error, values, ref_values),
            _relative_linf_error(values, ref_values),
        )
        error_series.append((_mode_label(mode), x_error, diff))
    return metrics, error_series


def run_experiment(
    experiment: ExperimentConfig | None = None,
    modes: tuple[str, ...] = ("scheme1", "scheme2", "scheme3"),
) -> dict[str, EvolutionResult]:
    experiment = ExperimentConfig() if experiment is None else experiment
    validate_experiment(experiment)
    out_dir = _resolve_output_dir(experiment.output.output_dir)
    modes = tuple(_normalize_mode(mode) for mode in modes)

    results: dict[str, EvolutionResult] = {}
    for mode in modes:
        results[mode] = solve_evolution(mode, experiment)

    reference_result: FdmReferenceResult | None = None
    reference_errors: dict[str, tuple[float, float]] = {}
    reference_error_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    if experiment.reference.enabled:
        _print(experiment.output.verbose, f"=== Running {_fdm_reference_label(experiment.reference)} ===")
        reference_result = solve_fdm_reference(experiment)
        reference_errors, reference_error_series = _compute_final_errors_vs_reference(
            experiment,
            results,
            reference_result,
        )
        _print(
            experiment.output.verbose,
            f"{reference_result.label} elapsed={reference_result.elapsed_seconds:.6f} s, "
            f"nfev={reference_result.nfev}, nlu={reference_result.nlu}",
        )

    x_plot = np.linspace(experiment.problem.x_left, experiment.problem.x_right, experiment.numerical.plot_samples)
    output_files: list[Path] = []
    if experiment.output.save_plots:
        final_series = [
            ("initial", x_plot, results[modes[0]].initial_state.eval(x_plot) if modes else np.zeros_like(x_plot))
        ]
        if reference_result is not None:
            final_series.append((reference_result.label, x_plot, reference_result.final_state.eval(x_plot)))
        for mode, result in results.items():
            final_series.append((_mode_label(mode), x_plot, result.final_state.eval(x_plot)))
        final_plot = out_dir / "allen_cahn_evolution_final.png"
        plot_final_solutions(final_series, experiment.problem.x_interface, final_plot)
        output_files.append(final_plot)

        energy_plot = out_dir / "allen_cahn_evolution_energy.png"
        energy_histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if reference_result is not None:
            energy_histories[reference_result.label] = (reference_result.times, reference_result.energies)
        for mode, result in results.items():
            energy_histories[_mode_label(mode)] = _history_arrays(result)
        plot_energy_histories(
            energy_histories,
            energy_plot,
        )
        output_files.append(energy_plot)

        if reference_result is not None:
            error_plot = out_dir / "allen_cahn_evolution_errors_vs_fdm_reference.png"
            plot_errors_vs_reference(
                reference_error_series,
                experiment.problem.x_interface,
                reference_result.label,
                error_plot,
            )
            output_files.append(error_plot)

    if experiment.output.save_npz:
        for result in results.values():
            output_files.append(_save_result_npz(out_dir, result))
        if reference_result is not None:
            output_files.append(_save_reference_npz(out_dir, reference_result))

    summary_path = out_dir / "summary.txt"
    config_lines = [
        f"domain = [{experiment.problem.x_left}, {experiment.problem.x_right}], x_interface = {experiment.problem.x_interface}",
        f"eps_left = {experiment.problem.eps_left}, eps_right = {experiment.problem.eps_right}",
        f"bc_left = {experiment.problem.bc_left}, bc_right = {experiment.problem.bc_right}",
        f"linear_grid_mode = {experiment.problem.linear_grid_mode}",
        f"n_elements = {experiment.numerical.n_elements}, monitor_segments = {experiment.numerical.monitor_segments}",
        f"tfpm_basis = {experiment.numerical.tfpm_basis}",
        f"time interval = [{experiment.time.initial_time}, {experiment.time.final_time}], dt = {experiment.time.dt}",
        f"lagrange residual tolerance = {experiment.lagrange.residual_tol}",
    ]
    if experiment.reference.enabled:
        config_lines.append(
            "reference = "
            f"{experiment.reference.method} FDM, N={experiment.reference.n_segments}, "
            f"rtol={experiment.reference.rtol}, atol={experiment.reference.atol}, "
            f"max_step={experiment.reference.max_step}, comparison_samples={experiment.reference.comparison_samples}"
        )
    result_lines: list[str] = []
    if reference_result is not None:
        result_lines.extend(
            [
                f"{reference_result.label} elapsed = {reference_result.elapsed_seconds:.6f} s",
                f"{reference_result.label} nfev = {reference_result.nfev}, njev = {reference_result.njev}, nlu = {reference_result.nlu}",
                f"{reference_result.label} final energy = {reference_result.energies[-1]:.8e}",
            ]
        )
    reference_label = reference_result.label if reference_result is not None else "reference"
    for mode, result in results.items():
        final_energy = result.history[-1].energy if result.history else result.initial_energy
        total_lagrange = sum(record.lagrange_solves for record in result.history)
        final_step = result.history[-1].step_l2 if result.history else 0.0
        error_line = []
        if reference_errors:
            rel_l2, rel_linf = reference_errors[_mode_label(mode)]
            error_line.append(f"{_mode_label(mode)} final relative L2 vs {reference_label} = {rel_l2:.8e}")
            error_line.append(f"{_mode_label(mode)} final relative Linf vs {reference_label} = {rel_linf:.8e}")
        result_lines.extend(
            [
                f"{_mode_label(mode)} completed = {result.completed}, all inner converged = {result.all_inner_converged}",
                f"{_mode_label(mode)} steps = {len(result.history)}, elapsed = {result.elapsed_seconds:.6f} s",
                f"{_mode_label(mode)} final time = {result.final_time:.8e}, final energy = {final_energy:.8e}",
                f"{_mode_label(mode)} final step_l2 = {final_step:.8e}, total Lagrange KKT solves = {total_lagrange}",
                *error_line,
            ]
        )
    output_files.append(summary_path)
    write_summary(
        summary_path,
        config_lines=config_lines,
        result_lines=result_lines,
        output_lines=[str(path) for path in output_files],
    )

    if experiment.output.verbose:
        print("\nSaved files:")
        for path in output_files:
            print(path)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TFPM-Lagrange Scheme I-III for time-evolution Allen-Cahn.")
    parser.add_argument("--mode", default="all", help="scheme1, scheme2, scheme3, or all")
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--n-elements", type=int, default=None)
    parser.add_argument("--monitor-segments", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--reference-method", default=None)
    parser.add_argument("--reference-segments", type=int, default=None)
    parser.add_argument("--reference-rtol", type=float, default=None)
    parser.add_argument("--reference-atol", type=float, default=None)
    parser.add_argument("--reference-max-step", type=float, default=None)
    parser.add_argument("--reference-no-max-step", action="store_true")
    parser.add_argument("--reference-samples", type=int, default=None)
    parser.add_argument("--lagrange-residual-tol", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-npz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment = ExperimentConfig()
    if args.final_time is not None:
        experiment.time.final_time = float(args.final_time)
    if args.dt is not None:
        experiment.time.dt = float(args.dt)
    if args.n_elements is not None:
        experiment.numerical.n_elements = int(args.n_elements)
    if args.monitor_segments is not None:
        experiment.numerical.monitor_segments = int(args.monitor_segments)
    if args.output_dir is not None:
        experiment.output.output_dir = Path(args.output_dir)
    if args.no_reference:
        experiment.reference.enabled = False
    if args.reference_method is not None:
        experiment.reference.method = str(args.reference_method)
    if args.reference_segments is not None:
        experiment.reference.n_segments = int(args.reference_segments)
    if args.reference_rtol is not None:
        experiment.reference.rtol = float(args.reference_rtol)
    if args.reference_atol is not None:
        experiment.reference.atol = float(args.reference_atol)
    if args.reference_no_max_step:
        experiment.reference.max_step = None
    elif args.reference_max_step is not None:
        experiment.reference.max_step = float(args.reference_max_step)
    if args.reference_samples is not None:
        experiment.reference.comparison_samples = int(args.reference_samples)
    if args.lagrange_residual_tol is not None:
        experiment.lagrange.residual_tol = float(args.lagrange_residual_tol)
    if args.quiet:
        experiment.output.verbose = False
    if args.no_plots:
        experiment.output.save_plots = False
    if args.no_npz:
        experiment.output.save_npz = False

    if str(args.mode).strip().lower() == "all":
        modes = ("scheme1", "scheme2", "scheme3")
    else:
        modes = (_normalize_mode(args.mode),)
    run_experiment(experiment=experiment, modes=modes)


if __name__ == "__main__":
    main()


__all__ = [
    "EvolutionResult",
    "FdmReferenceResult",
    "LinearSolveResult",
    "Snapshot",
    "StepRecord",
    "solve_evolution",
    "solve_fdm_reference",
    "run_experiment",
]
