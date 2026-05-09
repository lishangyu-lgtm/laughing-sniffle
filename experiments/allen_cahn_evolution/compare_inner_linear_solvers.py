from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    evaluate_fdm_derivative_on_side,
    evaluate_fdm_on_side,
    solve_interface_fdm,
)
from ...core.methods_tfpm import LagrangeHistory
from .analysis_plot import plot_energy_histories, plot_errors_vs_reference, plot_final_solutions
from .problem import (
    ArrayFunc,
    ExperimentConfig,
    NumericalConfig,
    PhysicalGrid,
    ProblemConfig,
    SampledState,
    TransformedGrid,
    build_physical_grid,
    build_transformed_grid,
    diffusion_piecewise,
    make_initial_state,
    nonlinearity,
    sampled_state_from_values,
    transformed_coefficient_from_physical,
    transformed_rhs_from_physical,
    validate_experiment,
    x_to_y,
)
from .run_allen_cahn_evolution import (
    FdmReferenceResult,
    LinearSolveResult as TfpmLinearSolveResult,
    _energy_on_state,
    _mode_label,
    _normalize_mode,
    _relative_l2_error,
    _relative_linf_error,
    _relative_step_l2,
    _solve_linear_tfpm_lagrange,
    _state_l2_norm,
    solve_fdm_reference,
)

LinearSolver = Callable[[ExperimentConfig, TransformedGrid, ArrayFunc, ArrayFunc], "InnerLinearSolveResult"]


@dataclass(slots=True)
class InnerLinearSolveResult:
    inner_solver: str
    coeff_y_func: ArrayFunc
    rhs_y_func: ArrayFunc
    y_grid: np.ndarray
    y_interface: float
    solved_on_physical_grid: bool
    tfpm_result: TfpmLinearSolveResult | None = None
    fdm_solution: dict[str, np.ndarray] | None = None
    lagrange_history: LagrangeHistory | None = None
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class StepAttempt:
    state: SampledState
    energy: float
    linear_result: InnerLinearSolveResult
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
    inner_solver: str
    stage: str
    inner_iterations: int
    linear_solves: int
    lagrange_solves: int
    linear_solve_seconds: float
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
class InnerEvolutionResult:
    mode: str
    inner_solver: str
    problem: ProblemConfig
    numerical: NumericalConfig
    initial_time: float
    final_time: float
    completed: bool
    all_inner_converged: bool
    initial_state: SampledState
    final_state: SampledState
    initial_energy: float
    final_linear_result: InnerLinearSolveResult | None
    history: list[StepRecord]
    snapshots: list[Snapshot]
    elapsed_seconds: float


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


def _default_experiment() -> ExperimentConfig:
    experiment = ExperimentConfig()
    experiment.output.output_dir = Path(__file__).resolve().parent / "results_inner_solver_compare"
    return experiment


def _normalize_inner_solver(inner_solver: str) -> str:
    key = str(inner_solver).strip().lower().replace("-", "_")
    aliases = {
        "tfpm": "tfpm_lagrange",
        "tfpm_lagrange": "tfpm_lagrange",
        "tfpm_kkt": "tfpm_lagrange",
        "lagrange": "tfpm_lagrange",
        "kkt": "tfpm_lagrange",
        "fdm": "fdm",
        "finite_difference": "fdm",
        "finite_difference_method": "fdm",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("inner_solver must be 'tfpm_lagrange', 'fdm', or 'all'.") from exc


def _inner_solver_label(inner_solver: str) -> str:
    return {"tfpm_lagrange": "TFPM-Lagrange", "fdm": "FDM"}[_normalize_inner_solver(inner_solver)]


def _series_label(mode: str, inner_solver: str) -> str:
    return f"{_mode_label(mode)} / {_inner_solver_label(inner_solver)}"


def _mode_output_dir(out_dir: Path, mode: str) -> Path:
    return out_dir / _normalize_mode(mode)


def _jump_du_for_linear_problem(problem: ProblemConfig, transformed_grid: TransformedGrid) -> float:
    if transformed_grid.solved_on_physical_grid:
        a_interface = float(diffusion_piecewise(problem, np.array([problem.x_interface], dtype=np.float64))[0])
        return float(problem.jump_flux) / a_interface
    return float(problem.jump_flux)


def _solve_linear_tfpm_lagrange_backend(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    coeff_x_func: ArrayFunc,
    rhs_x_func: ArrayFunc,
) -> InnerLinearSolveResult:
    t_start = time.perf_counter()
    tfpm_result = _solve_linear_tfpm_lagrange(experiment, transformed_grid, coeff_x_func, rhs_x_func)
    elapsed = time.perf_counter() - t_start
    return InnerLinearSolveResult(
        inner_solver="tfpm_lagrange",
        coeff_y_func=tfpm_result.coeff_y_func,
        rhs_y_func=tfpm_result.rhs_y_func,
        y_grid=tfpm_result.y_grid.copy(),
        y_interface=float(tfpm_result.y_interface),
        solved_on_physical_grid=bool(tfpm_result.solved_on_physical_grid),
        tfpm_result=tfpm_result,
        fdm_solution=None,
        lagrange_history=tfpm_result.lagrange_history,
        elapsed_seconds=elapsed,
    )


def _solve_linear_fdm_backend(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    coeff_x_func: ArrayFunc,
    rhs_x_func: ArrayFunc,
) -> InnerLinearSolveResult:
    problem = experiment.problem
    coeff_y_func = transformed_coefficient_from_physical(problem, coeff_x_func)
    rhs_y_func = transformed_rhs_from_physical(problem, rhs_x_func)
    jump_du = _jump_du_for_linear_problem(problem, transformed_grid)

    t_start = time.perf_counter()
    fdm_solution = solve_interface_fdm(
        N=None,
        c_func=coeff_y_func,
        f_func=rhs_y_func,
        m=problem.bc_left,
        n_dir=problem.bc_right,
        p=problem.jump_u,
        qjump=jump_du,
        grid=transformed_grid.y,
        xI=transformed_grid.y_interface,
    )
    elapsed = time.perf_counter() - t_start
    return InnerLinearSolveResult(
        inner_solver="fdm",
        coeff_y_func=coeff_y_func,
        rhs_y_func=rhs_y_func,
        y_grid=transformed_grid.y.copy(),
        y_interface=float(transformed_grid.y_interface),
        solved_on_physical_grid=bool(transformed_grid.solved_on_physical_grid),
        tfpm_result=None,
        fdm_solution=fdm_solution,
        lagrange_history=None,
        elapsed_seconds=elapsed,
    )


def _linear_solver_for(inner_solver: str) -> LinearSolver:
    inner_solver = _normalize_inner_solver(inner_solver)
    if inner_solver == "tfpm_lagrange":
        return _solve_linear_tfpm_lagrange_backend
    if inner_solver == "fdm":
        return _solve_linear_fdm_backend
    raise ValueError(f"Unsupported inner_solver={inner_solver!r}.")


def _evaluate_linear_solution_on_x(
    problem: ProblemConfig,
    numerical: NumericalConfig,
    linear: InnerLinearSolveResult,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_eval = x_to_y(problem, x_eval)
    left_mask = x_eval <= problem.x_interface
    right_mask = ~left_mask
    u = np.zeros_like(x_eval)
    ux = np.zeros_like(x_eval)

    if linear.inner_solver == "tfpm_lagrange":
        if linear.tfpm_result is None:
            raise ValueError("TFPM-Lagrange linear result is missing tfpm_result.")
        from .run_allen_cahn_evolution import _evaluate_linear_solution_on_x as _eval_tfpm

        return _eval_tfpm(problem, numerical, linear.tfpm_result, x_eval)

    if linear.inner_solver != "fdm":
        raise ValueError(f"Unsupported inner_solver={linear.inner_solver!r}.")
    if linear.fdm_solution is None:
        raise ValueError("FDM linear result is missing fdm_solution.")

    if np.any(left_mask):
        u[left_mask] = evaluate_fdm_on_side(linear.fdm_solution, y_eval[left_mask], side="left")
        duy_l = evaluate_fdm_derivative_on_side(linear.fdm_solution, y_eval[left_mask], side="left")
        if linear.solved_on_physical_grid:
            ux[left_mask] = duy_l
        else:
            ux[left_mask] = duy_l / (problem.eps_left**2)

    if np.any(right_mask):
        u[right_mask] = evaluate_fdm_on_side(linear.fdm_solution, y_eval[right_mask], side="right")
        duy_r = evaluate_fdm_derivative_on_side(linear.fdm_solution, y_eval[right_mask], side="right")
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
    linear: InnerLinearSolveResult,
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
    solve_linear: LinearSolver,
) -> StepAttempt:
    mu = 1.0 / dt + float(experiment.scheme1.stabilization_shift)

    def coeff_x(x: np.ndarray, mu: float = mu) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return mu * np.ones_like(x)

    def rhs_x(x: np.ndarray, state: SampledState = current_state, mu: float = mu) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        un = state.eval(x)
        return -nonlinearity(un) + mu * un

    linear = solve_linear(experiment, transformed_grid, coeff_x, rhs_x)
    state, energy = _candidate_from_linear(experiment, monitor_grid, linear)
    lagrange_solves = 0 if linear.lagrange_history is None else int(linear.lagrange_history.iter)
    return StepAttempt(
        state=state,
        energy=energy,
        linear_result=linear,
        inner_iterations=1,
        linear_solves=1,
        lagrange_solves=lagrange_solves,
        inner_converged=True,
        stop_reason=f"single linearized Scheme I step with {_inner_solver_label(linear.inner_solver)} inner solve",
    )


def _solve_scheme2_step(
    experiment: ExperimentConfig,
    transformed_grid: TransformedGrid,
    monitor_grid: PhysicalGrid,
    current_state: SampledState,
    dt: float,
    cfg,
    solve_linear: LinearSolver,
) -> StepAttempt:
    inv_dt = 1.0 / dt
    uk = current_state
    last_state: SampledState | None = None
    last_energy = np.nan
    last_linear: InnerLinearSolveResult | None = None
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

        linear = solve_linear(experiment, transformed_grid, coeff_x, rhs_x)
        if linear.lagrange_history is not None:
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
    solve_linear: LinearSolver,
) -> StepAttempt:
    problem = experiment.problem
    cfg = experiment.scheme3
    inv_2dt = 0.5 / dt
    inv_dt_term = 1.5 / dt
    uk = _scheme3_initial_guess(problem, prev_state, current_state, cfg.extrapolation_weight)
    last_state: SampledState | None = None
    last_energy = np.nan
    last_linear: InnerLinearSolveResult | None = None
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

        linear = solve_linear(experiment, transformed_grid, coeff_x, rhs_x)
        if linear.lagrange_history is not None:
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
    inner_solver: str,
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
        inner_solver=inner_solver,
        stage=stage,
        inner_iterations=int(attempt.inner_iterations),
        linear_solves=int(attempt.linear_solves),
        lagrange_solves=int(attempt.lagrange_solves),
        linear_solve_seconds=float(attempt.linear_result.elapsed_seconds),
        lagrange_primal_inf=np.nan if lag is None else float(lag.primal_inf),
        lagrange_stationarity_inf=np.nan if lag is None else float(lag.stationarity_inf),
        inner_converged=bool(attempt.inner_converged),
        step_l2=_state_l2_norm(monitor_grid, delta),
        step_inf=float(np.max(np.abs(delta))),
        energy=float(attempt.energy),
        note=attempt.stop_reason,
    )


def solve_evolution_with_inner_solver(
    mode: str,
    inner_solver: str,
    experiment: ExperimentConfig | None = None,
) -> InnerEvolutionResult:
    experiment = _default_experiment() if experiment is None else experiment
    validate_experiment(experiment)
    mode = _normalize_mode(mode)
    inner_solver = _normalize_inner_solver(inner_solver)
    solve_linear = _linear_solver_for(inner_solver)

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
    final_linear: InnerLinearSolveResult | None = None
    all_inner_converged = True

    _print(
        verbose,
        f"=== Running {_mode_label(mode)} with {_inner_solver_label(inner_solver)} inner linear solve ===",
    )
    t_start = time.perf_counter()
    step = 0
    while current_time < final_time - 1.0e-14:
        dt = min(base_dt, final_time - current_time)
        t_new = current_time + dt
        step += 1
        old_state = current_state

        if mode == "scheme1":
            stage = "scheme1"
            attempt = _solve_scheme1_step(
                experiment,
                transformed_grid,
                monitor_grid,
                current_state,
                dt,
                solve_linear,
            )
            current_state = attempt.state
        elif mode == "scheme2":
            stage = "scheme2"
            attempt = _solve_scheme2_step(
                experiment,
                transformed_grid,
                monitor_grid,
                current_state,
                dt,
                experiment.scheme2,
                solve_linear,
            )
            current_state = attempt.state
        else:
            if prev_state is None:
                startup_mode = _normalize_mode(experiment.scheme3.startup_mode)
                if startup_mode == "scheme3":
                    raise ValueError("scheme3.startup_mode must be 'scheme1' or 'scheme2'.")
                stage = f"scheme3-startup-{startup_mode}"
                if startup_mode == "scheme1":
                    attempt = _solve_scheme1_step(
                        experiment,
                        transformed_grid,
                        monitor_grid,
                        current_state,
                        dt,
                        solve_linear,
                    )
                else:
                    attempt = _solve_scheme2_step(
                        experiment,
                        transformed_grid,
                        monitor_grid,
                        current_state,
                        dt,
                        experiment.scheme3,
                        solve_linear,
                    )
                prev_state = old_state
                current_state = attempt.state
            else:
                stage = "scheme3"
                attempt = _solve_scheme3_step(
                    experiment,
                    transformed_grid,
                    monitor_grid,
                    prev_state,
                    current_state,
                    dt,
                    solve_linear,
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
            inner_solver=inner_solver,
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
            f"{_series_label(mode, inner_solver)} step={step:04d}, t={current_time:.6e}, "
            f"E={record.energy:.8e}, step_l2={record.step_l2:.3e}, "
            f"inner={record.inner_iterations}",
        )

    elapsed = time.perf_counter() - t_start
    completed = current_time >= final_time - 1.0e-14
    if not snapshots or snapshots[-1].time < current_time - 1.0e-14:
        snapshots.append(Snapshot(time=current_time, state=current_state))

    return InnerEvolutionResult(
        mode=mode,
        inner_solver=inner_solver,
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


def _history_arrays(result: InnerEvolutionResult) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([result.initial_time] + [record.time for record in result.history], dtype=np.float64)
    energies = np.array([result.initial_energy] + [record.energy for record in result.history], dtype=np.float64)
    return times, energies


def _save_result_npz(out_dir: Path, result: InnerEvolutionResult) -> Path:
    path = out_dir / f"{result.mode}_{result.inner_solver}_state_history.npz"
    times, energies = _history_arrays(result)
    np.savez(
        path,
        inner_solver=np.array([result.inner_solver]),
        monitor_x=result.final_state.x,
        initial_u=result.initial_state.u,
        final_u=result.final_state.u,
        times=times,
        energies=energies,
        step=np.array([record.step for record in result.history], dtype=np.int32),
        dt=np.array([record.dt for record in result.history], dtype=np.float64),
        step_l2=np.array([record.step_l2 for record in result.history], dtype=np.float64),
        step_inf=np.array([record.step_inf for record in result.history], dtype=np.float64),
        linear_solve_seconds=np.array([record.linear_solve_seconds for record in result.history], dtype=np.float64),
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


def _compute_reference_errors(
    experiment: ExperimentConfig,
    results: dict[tuple[str, str], InnerEvolutionResult],
    reference: FdmReferenceResult,
) -> tuple[dict[tuple[str, str], tuple[float, float]], list[tuple[str, np.ndarray, np.ndarray]]]:
    x_error = np.linspace(
        experiment.problem.x_left,
        experiment.problem.x_right,
        int(experiment.reference.comparison_samples),
    )
    ref_values = reference.final_state.eval(x_error)
    metrics: dict[tuple[str, str], tuple[float, float]] = {}
    error_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for (mode, inner_solver), result in results.items():
        values = result.final_state.eval(x_error)
        diff = values - ref_values
        metrics[(mode, inner_solver)] = (
            _relative_l2_error(x_error, values, ref_values),
            _relative_linf_error(values, ref_values),
        )
        error_series.append((_series_label(mode, inner_solver), x_error, diff))
    return metrics, error_series


def _compute_pairwise_inner_differences(
    experiment: ExperimentConfig,
    results: dict[tuple[str, str], InnerEvolutionResult],
    modes: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    x_error = np.linspace(
        experiment.problem.x_left,
        experiment.problem.x_right,
        int(experiment.reference.comparison_samples),
    )
    metrics: dict[str, tuple[float, float]] = {}
    for mode in modes:
        fdm_result = results.get((mode, "fdm"))
        tfpm_result = results.get((mode, "tfpm_lagrange"))
        if fdm_result is None or tfpm_result is None:
            continue
        fdm_values = fdm_result.final_state.eval(x_error)
        tfpm_values = tfpm_result.final_state.eval(x_error)
        metrics[_mode_label(mode)] = (
            _relative_l2_error(x_error, fdm_values, tfpm_values),
            _relative_linf_error(fdm_values, tfpm_values),
        )
    return metrics


def _write_summary(
    save_path: Path,
    *,
    config_lines: list[str],
    result_lines: list[str],
    output_lines: list[str],
) -> None:
    lines = ["Allen-Cahn evolution inner linear solver comparison", ""]
    lines.append("Configuration:")
    lines.extend(f"- {line}" for line in config_lines)
    lines.append("")
    lines.append("Results:")
    lines.extend(f"- {line}" for line in result_lines)
    if output_lines:
        lines.append("")
        lines.append("Output files:")
        lines.extend(f"- {line}" for line in output_lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")


def run_inner_solver_comparison(
    experiment: ExperimentConfig | None = None,
    modes: tuple[str, ...] = ("scheme1", "scheme2", "scheme3"),
    inner_solvers: tuple[str, ...] = ("tfpm_lagrange", "fdm"),
) -> dict[tuple[str, str], InnerEvolutionResult]:
    experiment = _default_experiment() if experiment is None else experiment
    validate_experiment(experiment)
    out_dir = _resolve_output_dir(experiment.output.output_dir)
    modes = tuple(_normalize_mode(mode) for mode in modes)
    inner_solvers = tuple(_normalize_inner_solver(inner_solver) for inner_solver in inner_solvers)

    results: dict[tuple[str, str], InnerEvolutionResult] = {}
    for mode in modes:
        for inner_solver in inner_solvers:
            result = solve_evolution_with_inner_solver(mode, inner_solver, experiment)
            results[(mode, inner_solver)] = result

    reference_result: FdmReferenceResult | None = None
    reference_errors: dict[tuple[str, str], tuple[float, float]] = {}
    if experiment.reference.enabled:
        _print(
            experiment.output.verbose,
            f"=== Running FDM reference ({experiment.reference.method}, N={experiment.reference.n_segments}) ===",
        )
        reference_result = solve_fdm_reference(experiment)
        reference_errors, _ = _compute_reference_errors(
            experiment,
            results,
            reference_result,
        )
        _print(
            experiment.output.verbose,
            f"{reference_result.label} elapsed={reference_result.elapsed_seconds:.6f} s, "
            f"nfev={reference_result.nfev}, nlu={reference_result.nlu}",
        )

    pairwise_errors = _compute_pairwise_inner_differences(experiment, results, modes)

    x_plot = np.linspace(experiment.problem.x_left, experiment.problem.x_right, experiment.numerical.plot_samples)
    output_files: list[Path] = []
    if experiment.output.save_plots:
        first_result = next(iter(results.values()))
        for mode in modes:
            mode_dir = _mode_output_dir(out_dir, mode)
            final_series = [("initial", x_plot, first_result.initial_state.eval(x_plot))]
            if reference_result is not None:
                final_series.append((reference_result.label, x_plot, reference_result.final_state.eval(x_plot)))
            for inner_solver in inner_solvers:
                result = results[(mode, inner_solver)]
                final_series.append((_inner_solver_label(inner_solver), x_plot, result.final_state.eval(x_plot)))
            final_plot = mode_dir / "final_state.png"
            plot_final_solutions(
                final_series,
                experiment.problem.x_interface,
                final_plot,
                title=f"Allen-Cahn Evolution: {_mode_label(mode)} Final State",
            )
            output_files.append(final_plot)

            energy_histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if reference_result is not None:
                energy_histories[reference_result.label] = (reference_result.times, reference_result.energies)
            for inner_solver in inner_solvers:
                result = results[(mode, inner_solver)]
                energy_histories[_inner_solver_label(inner_solver)] = _history_arrays(result)
            energy_plot = mode_dir / "energy_history.png"
            plot_energy_histories(
                energy_histories,
                energy_plot,
                title=f"Allen-Cahn Evolution: {_mode_label(mode)} Energy History",
            )
            output_files.append(energy_plot)

            if reference_result is not None:
                x_error = np.linspace(
                    experiment.problem.x_left,
                    experiment.problem.x_right,
                    int(experiment.reference.comparison_samples),
                )
                ref_values = reference_result.final_state.eval(x_error)
                error_series = []
                for inner_solver in inner_solvers:
                    result = results[(mode, inner_solver)]
                    values = result.final_state.eval(x_error)
                    error_series.append((_inner_solver_label(inner_solver), x_error, values - ref_values))
                error_plot = mode_dir / "errors_vs_fdm_reference.png"
                plot_errors_vs_reference(
                    error_series,
                    experiment.problem.x_interface,
                    reference_result.label,
                    error_plot,
                    title=f"Allen-Cahn Evolution: {_mode_label(mode)} Final Error vs {reference_result.label}",
                )
                output_files.append(error_plot)

    if experiment.output.save_npz:
        for result in results.values():
            output_files.append(_save_result_npz(_mode_output_dir(out_dir, result.mode), result))
        if reference_result is not None:
            output_files.append(_save_reference_npz(out_dir, reference_result))

    summary_path = out_dir / "summary.txt"
    reference_line = "disabled"
    if experiment.reference.enabled:
        reference_line = (
            f"{experiment.reference.method} FDM, N={experiment.reference.n_segments}, "
            f"rtol={experiment.reference.rtol}, atol={experiment.reference.atol}, "
            f"max_step={experiment.reference.max_step}, comparison_samples={experiment.reference.comparison_samples}"
        )
    config_lines = [
        f"domain = [{experiment.problem.x_left}, {experiment.problem.x_right}], x_interface = {experiment.problem.x_interface}",
        f"eps_left = {experiment.problem.eps_left}, eps_right = {experiment.problem.eps_right}",
        f"bc_left = {experiment.problem.bc_left}, bc_right = {experiment.problem.bc_right}",
        f"linear_grid_mode = {experiment.problem.linear_grid_mode}",
        "inner comparison = same outer scheme, same linear grid, different inner linear solver",
        f"inner solvers = {', '.join(_inner_solver_label(s) for s in inner_solvers)}",
        f"modes = {', '.join(_mode_label(m) for m in modes)}",
        f"n_elements = {experiment.numerical.n_elements}, monitor_segments = {experiment.numerical.monitor_segments}",
        f"tfpm_basis = {experiment.numerical.tfpm_basis}",
        f"time interval = [{experiment.time.initial_time}, {experiment.time.final_time}], dt = {experiment.time.dt}",
        f"lagrange residual tolerance = {experiment.lagrange.residual_tol}",
        f"reference = {reference_line}",
    ]

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
    for mode in modes:
        for inner_solver in inner_solvers:
            result = results[(mode, inner_solver)]
            final_energy = result.history[-1].energy if result.history else result.initial_energy
            total_lagrange = sum(record.lagrange_solves for record in result.history)
            total_linear_seconds = sum(record.linear_solve_seconds for record in result.history)
            final_step = result.history[-1].step_l2 if result.history else 0.0
            prefix = _series_label(mode, inner_solver)
            result_lines.extend(
                [
                    f"{prefix} completed = {result.completed}, all inner converged = {result.all_inner_converged}",
                    f"{prefix} steps = {len(result.history)}, elapsed = {result.elapsed_seconds:.6f} s, linear solve time = {total_linear_seconds:.6f} s",
                    f"{prefix} final time = {result.final_time:.8e}, final energy = {final_energy:.8e}",
                    f"{prefix} final step_l2 = {final_step:.8e}, total Lagrange KKT solves = {total_lagrange}",
                ]
            )
            if reference_errors:
                rel_l2, rel_linf = reference_errors[(mode, inner_solver)]
                result_lines.extend(
                    [
                        f"{prefix} final relative L2 vs {reference_label} = {rel_l2:.8e}",
                        f"{prefix} final relative Linf vs {reference_label} = {rel_linf:.8e}",
                    ]
                )
    if pairwise_errors:
        result_lines.append("Pairwise final differences between FDM inner and TFPM-Lagrange inner:")
        for mode_label, (rel_l2, rel_linf) in pairwise_errors.items():
            result_lines.append(
                f"{mode_label} FDM-inner vs TFPM-Lagrange-inner relative L2 = {rel_l2:.8e}, relative Linf = {rel_linf:.8e}"
            )

    output_files.append(summary_path)
    _write_summary(
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
    parser = argparse.ArgumentParser(
        description="Compare FDM and TFPM-Lagrange inner linear solves for Allen-Cahn Scheme I-III."
    )
    parser.add_argument("--mode", default="all", help="scheme1, scheme2, scheme3, or all")
    parser.add_argument("--inner-solver", default="all", help="tfpm_lagrange, fdm, or all")
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
    experiment = _default_experiment()
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

    if str(args.inner_solver).strip().lower() == "all":
        inner_solvers = ("tfpm_lagrange", "fdm")
    else:
        inner_solvers = (_normalize_inner_solver(args.inner_solver),)

    run_inner_solver_comparison(
        experiment=experiment,
        modes=modes,
        inner_solvers=inner_solvers,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "InnerEvolutionResult",
    "InnerLinearSolveResult",
    "Snapshot",
    "StepRecord",
    "run_inner_solver_comparison",
    "solve_evolution_with_inner_solver",
]
