from __future__ import annotations

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
from ...core.methods_tfpm import LagrangeHistory, assemble_lagrange_kkt_system, solve_lagrange_kkt
from ...core.tfpm_local import (
    build_elements,
    build_uniform_grid,
    evaluate_solution_fine,
    evaluate_tfpm_on_side,
)
from .analysis_plot import compute_errors_vs_reference, plot_errors_vs_reference, plot_solutions, write_summary

ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class ProblemConfig:
    a: float = 0.0
    b: float = 1.0
    x_interface: float = 0.5
    num_elements: int = 32
    quad_n: int = 10
    n_seg_gauss: int = 8
    left_bc: float = 0.0
    right_bc: float = 0.0
    jump_u: float = 0.0
    jump_du: float = 0.0


@dataclass(slots=True)
class SolverConfig:
    lagrange_residual_tol: float = 1e-10
    max_newton_iter: int = 50
    newton_tol: float = 1e-12
    error_check_points: int = 200
    comparison_points_per_side: int = 2000
    snapshot_points_per_element: int = 80
    plot_points_per_element: int = 50
    fdm_num_elements: int | None = None
    fdm_reference_num_elements: int = 2**14
    fdm_plot_points_per_cell: int = 20


@dataclass(slots=True)
class NewtonIterationRecord:
    iteration: int
    error_inf: float
    lagrange_history: LagrangeHistory


@dataclass(slots=True)
class IterateState:
    elems: object | None = None
    z: np.ndarray | None = None
    forcing: ArrayFunc | None = None
    x_snapshot: np.ndarray | None = None
    u_snapshot: np.ndarray | None = None

    def snapshot_evaluator(self) -> ArrayFunc:
        x_snapshot = self.x_snapshot
        u_snapshot = self.u_snapshot

        def evaluate(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            if u_snapshot is None or x_snapshot is None:
                return np.zeros_like(x_arr)
            return np.interp(x_arr, x_snapshot, u_snapshot)

        return evaluate

    def update(self, elems: object, z: np.ndarray, forcing: ArrayFunc, config: ProblemConfig, solver: SolverConfig) -> None:
        self.elems = elems
        self.z = z
        self.forcing = forcing
        self.x_snapshot, self.u_snapshot = evaluate_solution_fine(
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
    history: list[NewtonIterationRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    converged: bool = False

    def evaluate_on_side(self, x_eval: np.ndarray, side: str) -> np.ndarray:
        if self.state.elems is None or self.state.z is None or self.state.forcing is None:
            raise RuntimeError("No solution is available for evaluation.")
        return evaluate_tfpm_on_side(
            elems=self.state.elems,
            z=self.state.z,
            f_func=self.state.forcing,
            x_eval=x_eval,
            side=side,
            xI=self.problem.x_interface,
            n_seg_gauss=self.problem.n_seg_gauss,
        )

    def plot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.state.elems is None or self.state.z is None or self.state.forcing is None:
            raise RuntimeError("No solution is available to plot.")
        return evaluate_solution_fine(
            elems=self.state.elems,
            z=self.state.z,
            f_func=self.state.forcing,
            points_per_element=self.solver.plot_points_per_element,
            n_seg_gauss=self.problem.n_seg_gauss,
        )


@dataclass(slots=True)
class FdmIterationRecord:
    iteration: int
    error_inf: float


@dataclass(slots=True)
class FdmSolveResult:
    problem: ProblemConfig
    solver: SolverConfig
    num_elements: int
    solution: dict[str, np.ndarray] | None = None
    history: list[FdmIterationRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    converged: bool = False

    def evaluate_on_side(self, x_eval: np.ndarray, side: str) -> np.ndarray:
        if self.solution is None:
            raise RuntimeError("No FDM solution is available for evaluation.")
        return evaluate_fdm_on_side(self.solution, x_eval, side=side)

    def plot_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.solution is None:
            raise RuntimeError("No FDM solution is available to plot.")
        return fdm_plot_arrays(self.solution, points_per_cell=self.solver.fdm_plot_points_per_cell)


@dataclass(slots=True)
class ComparisonResult:
    tfpm_result: SolveResult
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


def default_source(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return 1e2*x_arr*(1.0 - x_arr)


def make_linearized_functions(previous_iterate: ArrayFunc, source_func: ArrayFunc) -> tuple[ArrayFunc, ArrayFunc]:
    def c_func(x: np.ndarray, prev: ArrayFunc = previous_iterate) -> np.ndarray:
        u_prev = prev(x)
        return  3*u_prev**2

    def f_func(x: np.ndarray, prev: ArrayFunc = previous_iterate, source: ArrayFunc = source_func) -> np.ndarray:
        u_prev = prev(x)
        return source(x) + 2*u_prev**3

    return c_func, f_func
def build_grid(problem: ProblemConfig) -> np.ndarray:
    grid, _, _ = build_uniform_grid(
        N=problem.num_elements,
        a=problem.a,
        b=problem.b,
        xI=problem.x_interface,
    )
    return grid


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


def _make_zero_evaluator() -> ArrayFunc:
    def evaluate(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x_arr)

    return evaluate


def _make_fdm_evaluator(solution: dict[str, np.ndarray], x_interface: float) -> ArrayFunc:
    def evaluate(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        values = np.empty_like(x_arr)
        left_mask = x_arr <= x_interface
        right_mask = ~left_mask

        if np.any(left_mask):
            values[left_mask] = evaluate_fdm_on_side(solution, x_arr[left_mask], side="left")
        if np.any(right_mask):
            values[right_mask] = evaluate_fdm_on_side(solution, x_arr[right_mask], side="right")
        return values

    return evaluate


def _build_piecewise_plot_arrays(
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


def _resolve_output_dir(save_dir: Path | None) -> Path:
    if save_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    else:
        output_dir = Path(save_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def solve_newton_problem(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    source_func: ArrayFunc = default_source,
    verbose: bool = True,
) -> SolveResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()

    grid = build_grid(problem)
    state = IterateState()
    history: list[NewtonIterationRecord] = []

    if verbose:
        print("=== Starting Newton-Raphson iteration ===")

    t_start = time.perf_counter()

    for iteration in range(1, solver.max_newton_iter + 1):
        previous_iterate = state.snapshot_evaluator()
        c_func_k, f_func_k = make_linearized_functions(previous_iterate, source_func)

        _, elems = build_elements(
            c_func=c_func_k,
            f_func=f_func_k,
            grid=grid,
            quad_n=problem.quad_n,
            xI=problem.x_interface,
            n_seg_gauss=problem.n_seg_gauss,
        )

        K, rhs, H, l_vec, C, d = assemble_lagrange_kkt_system(
            grid=grid,
            elems=elems,
            f_func=f_func_k,
            m=problem.left_bc,
            n_dir=problem.right_bc,
            p=problem.jump_u,
            jump_du=problem.jump_du,
            xI=problem.x_interface,
            c_func=c_func_k,
            use_true_c=True,
            flux_jump_weights=(0.5, 0.5),
        )

        z_new, _lam, lagrange_history = solve_lagrange_kkt(
            K=K,
            rhs=rhs,
            H=H,
            l=l_vec,
            C=C,
            d=d,
            residual_tol=solver.lagrange_residual_tol,
        )

        x_check = np.linspace(problem.a, problem.b, solver.error_check_points)
        u_old = previous_iterate(x_check)

        state.update(elems=elems, z=z_new, forcing=f_func_k, config=problem, solver=solver)
        u_new = state.snapshot_evaluator()(x_check)

        error_inf = float(np.max(np.abs(u_new - u_old)))
        record = NewtonIterationRecord(
            iteration=iteration,
            error_inf=error_inf,
            lagrange_history=lagrange_history,
        )
        history.append(record)

        if verbose:
            print(
                f"Newton Iter {iteration:2d} | "
                f"L-inf Error: {error_inf:.4e} | "
                f"KKT residual: {lagrange_history.residual_inf:.4e}"
            )

        if error_inf < solver.newton_tol:
            elapsed = time.perf_counter() - t_start
            if verbose:
                print("=> Newton iteration reached the target tolerance.")
                print(f"Total runtime: {elapsed:.4f} s")
            return SolveResult(
                problem=problem,
                solver=solver,
                state=state,
                history=history,
                elapsed_seconds=elapsed,
                converged=True,
            )

    elapsed = time.perf_counter() - t_start
    if verbose:
        print("=> Warning: Newton iteration hit the maximum number of steps without converging.")
        print(f"Total runtime: {elapsed:.4f} s")
    return SolveResult(
        problem=problem,
        solver=solver,
        state=state,
        history=history,
        elapsed_seconds=elapsed,
        converged=False,
    )


def solve_newton_fdm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    num_elements: int | None = None,
    source_func: ArrayFunc = default_source,
    verbose: bool = True,
    label: str = "FDM",
) -> FdmSolveResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()

    resolved_num_elements = _resolve_fdm_num_elements(problem, solver, num_elements)
    _validate_grid_count(label, resolved_num_elements, problem)
    grid, _, _ = build_uniform_grid(
        N=resolved_num_elements,
        a=problem.a,
        b=problem.b,
        xI=problem.x_interface,
    )

    current_solution: dict[str, np.ndarray] | None = None
    previous_iterate = _make_zero_evaluator()
    history: list[FdmIterationRecord] = []

    if verbose:
        print(f"=== Starting {label} Newton iteration ===")

    t_start = time.perf_counter()

    for iteration in range(1, solver.max_newton_iter + 1):
        c_func_k, f_func_k = make_linearized_functions(previous_iterate, source_func)
        new_solution = solve_interface_fdm(
            N=None,
            c_func=c_func_k,
            f_func=f_func_k,
            m=problem.left_bc,
            n_dir=problem.right_bc,
            p=problem.jump_u,
            qjump=problem.jump_du,
            grid=grid,
            xI=problem.x_interface,
        )

        current_iterate = _make_fdm_evaluator(new_solution, problem.x_interface)
        x_check = np.linspace(problem.a, problem.b, max(int(solver.error_check_points), 2))
        u_old = previous_iterate(x_check)
        u_new = current_iterate(x_check)
        error_inf = float(np.max(np.abs(u_new - u_old)))

        record = FdmIterationRecord(iteration=iteration, error_inf=error_inf)
        history.append(record)
        current_solution = new_solution
        previous_iterate = current_iterate

        if verbose:
            print(f"{label} Newton Iter {iteration:2d} | L-inf Error: {error_inf:.4e}")

        if error_inf < solver.newton_tol:
            elapsed = time.perf_counter() - t_start
            if verbose:
                print(f"=> {label} Newton iteration reached the target tolerance.")
                print(f"Total runtime: {elapsed:.4f} s")
            return FdmSolveResult(
                problem=problem,
                solver=solver,
                num_elements=resolved_num_elements,
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
        solution=current_solution,
        history=history,
        elapsed_seconds=elapsed,
        converged=False,
    )


def default_plot_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "solve_nonlinear_pde_solution.png"


def plot_solution(result: SolveResult, show: bool = True, save_path: Path | None = None) -> Path:
    x_plot, u_plot = result.plot_arrays()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, u_plot, "b-", linewidth=2, label="u(x)")
    ax.set_title(r"Solution of $-u'' + u^3 = f(x)$ using TFPM-Lagrange")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = save_path or default_plot_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    backend = plt.get_backend().lower()
    if show and "agg" not in backend:
        plt.show()
    else:
        if show:
            print(f"Plot not shown because backend '{plt.get_backend()}' is non-interactive.")
        plt.close(fig)
    return output_path


def solve_nonlinear_bvp(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    source_func: ArrayFunc = default_source,
    show_plot: bool = True,
    save_path: Path | None = None,
    verbose: bool = True,
) -> SolveResult:
    result = solve_newton_problem(
        problem=problem,
        solver=solver,
        source_func=source_func,
        verbose=verbose,
    )
    plot_solution(result, show=show_plot, save_path=save_path)
    return result


def compare_nonlinear_bvp_with_fdm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    source_func: ArrayFunc = default_source,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> ComparisonResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()

    coarse_fdm_num_elements = _resolve_fdm_num_elements(problem, solver, num_elements=None)
    _validate_grid_count("problem.num_elements", problem.num_elements, problem)
    _validate_grid_count("solver.fdm_num_elements", coarse_fdm_num_elements, problem)
    _validate_grid_count("solver.fdm_reference_num_elements", solver.fdm_reference_num_elements, problem)

    if verbose:
        print("=== Solving TFPM-Lagrange solution ===")
    tfpm_result = solve_newton_problem(
        problem=problem,
        solver=solver,
        source_func=source_func,
        verbose=verbose,
    )

    if verbose:
        print(f"\n=== Solving coarse-grid FDM solution (N={coarse_fdm_num_elements}) ===")
    fdm_result = solve_newton_fdm(
        problem=problem,
        solver=solver,
        num_elements=coarse_fdm_num_elements,
        source_func=source_func,
        verbose=verbose,
        label=f"FDM (N={coarse_fdm_num_elements})",
    )

    if verbose:
        print(f"\n=== Solving fine-grid FDM reference (N={solver.fdm_reference_num_elements}) ===")
    fdm_reference_result = solve_newton_fdm(
        problem=problem,
        solver=solver,
        num_elements=solver.fdm_reference_num_elements,
        source_func=source_func,
        verbose=verbose,
        label=f"FDM reference (N={solver.fdm_reference_num_elements})",
    )

    n_compare = max(int(solver.comparison_points_per_side), 2)
    x_left = np.linspace(problem.a, problem.x_interface, n_compare)
    x_right = np.linspace(problem.x_interface, problem.b, n_compare)

    ref_left = fdm_reference_result.evaluate_on_side(x_left, side="left")
    ref_right = fdm_reference_result.evaluate_on_side(x_right, side="right")
    reference_values = (ref_left, ref_right)
    reference_label = f"FDM fine (N={fdm_reference_result.num_elements})"

    method_values = {
        "TFPM-Lagrange": (
            tfpm_result.evaluate_on_side(x_left, side="left"),
            tfpm_result.evaluate_on_side(x_right, side="right"),
        ),
        f"FDM (N={fdm_result.num_elements})": (
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

    out_dir = _resolve_output_dir(save_dir)
    solution_plot_path = out_dir / "solve_nonlinear_pde_compare.png"
    error_plot_path = out_dir / "solve_nonlinear_pde_errors_vs_reference.png"
    summary_path = out_dir / "solve_nonlinear_pde_summary.txt"

    x_plot_ref, u_plot_ref = _build_piecewise_plot_arrays(problem, n_compare, fdm_reference_result.evaluate_on_side)
    x_plot_tfpm, u_plot_tfpm = _build_piecewise_plot_arrays(problem, n_compare, tfpm_result.evaluate_on_side)
    x_plot_fdm, u_plot_fdm = _build_piecewise_plot_arrays(problem, n_compare, fdm_result.evaluate_on_side)

    plot_solutions(
        [
            (reference_label, x_plot_ref, u_plot_ref),
            ("TFPM-Lagrange", x_plot_tfpm, u_plot_tfpm),
            (f"FDM (N={fdm_result.num_elements})", x_plot_fdm, u_plot_fdm),
        ],
        xI=problem.x_interface,
        save_path=solution_plot_path,
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
        "TFPM-Lagrange": tfpm_result.elapsed_seconds,
        f"FDM (N={fdm_result.num_elements})": fdm_result.elapsed_seconds,
        reference_label: fdm_reference_result.elapsed_seconds,
    }
    diagnostics = [
        f"TFPM-Lagrange converged = {tfpm_result.converged}",
        f"FDM (N={fdm_result.num_elements}) converged = {fdm_result.converged}",
        f"{reference_label} converged = {fdm_reference_result.converged}",
        f"TFPM-Lagrange Newton steps = {len(tfpm_result.history)}",
        f"FDM (N={fdm_result.num_elements}) Newton steps = {len(fdm_result.history)}",
        f"{reference_label} Newton steps = {len(fdm_reference_result.history)}",
    ]
    if tfpm_result.history:
        diagnostics.append(f"TFPM-Lagrange final Newton increment = {tfpm_result.history[-1].error_inf:.6e}")
    if fdm_result.history:
        diagnostics.append(
            f"FDM (N={fdm_result.num_elements}) final Newton increment = {fdm_result.history[-1].error_inf:.6e}"
        )
    if fdm_reference_result.history:
        diagnostics.append(
            f"{reference_label} final Newton increment = {fdm_reference_result.history[-1].error_inf:.6e}"
        )

    write_summary(
        summary_path,
        timings=timings,
        errors=relative_l2_errors,
        extra_lines=diagnostics,
        errors_title=f"Relative L2 errors vs {reference_label}:",
    )

    if verbose:
        print(f"\n=== Relative L2 Error vs {reference_label} ===")
        for key, value in relative_l2_errors.items():
            print(f"{key:>40s}: {value:.6e}")

        print("\nSaved files:")
        print(summary_path)
        print(solution_plot_path)
        print(error_plot_path)

    return ComparisonResult(
        tfpm_result=tfpm_result,
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


def main() -> None:
    compare_nonlinear_bvp_with_fdm()


if __name__ == "__main__":
    main()
