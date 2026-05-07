from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    _HERE = Path(__file__).resolve().parent
    _EXPERIMENTS = _HERE.parent
    _ROOT = _EXPERIMENTS.parent
    _PARENT = _ROOT.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))
    __package__ = f"{_ROOT.name}.{_EXPERIMENTS.name}.{_HERE.name}"

from ..nonlinear_pde.analysis_plot import (
    compute_errors_vs_reference,
    plot_errors_vs_reference,
    plot_solutions,
    write_summary,
)
from ..nonlinear_pde import solve_nonlinear_pde as pde_module
from .solve_energy_alm import (
    ProblemConfig as EnergyProblemConfig,
    SolverConfig as EnergySolverConfig,
    SolveResult as EnergySolveResult,
    solve_energy_alm,
)


@dataclass(slots=True)
class ComparisonResult:
    energy_result: EnergySolveResult
    tfpm_result: object
    fdm_reference_result: object
    relative_l2_errors: dict[str, float]
    timings: dict[str, float]
    solution_plot_path: Path
    error_plot_path: Path
    summary_path: Path


def make_common_energy_problem() -> EnergyProblemConfig:
    return EnergyProblemConfig(
        a=0.0,
        b=1.0,
        x_interface=0.5,
        num_elements=32,
        quad_n=12,
        n_seg_gauss=8,
        left_bc=0.0,
        right_bc=0.0,
        jump_u=0.0,
        jump_du=0.0,
    )


def make_matching_pde_problem(problem: EnergyProblemConfig) -> pde_module.ProblemConfig:
    return pde_module.ProblemConfig(
        a=problem.a,
        b=problem.b,
        x_interface=problem.x_interface,
        num_elements=problem.num_elements,
        quad_n=problem.quad_n,
        n_seg_gauss=problem.n_seg_gauss,
        left_bc=problem.left_bc,
        right_bc=problem.right_bc,
        jump_u=problem.jump_u,
        jump_du=problem.jump_du,
    )


def make_energy_solver() -> EnergySolverConfig:
    return EnergySolverConfig()


def make_pde_solver() -> pde_module.SolverConfig:
    return pde_module.SolverConfig(
        rho=10.0,
        max_aug_lag_iter=5000,
        tol_primal=1.0e-10,
        tol_stationarity=1.0e-10,
        max_newton_iter=50,
        newton_tol=1.0e-14,
        error_check_points=400,
        comparison_points_per_side=2000,
        snapshot_points_per_element=80,
        plot_points_per_element=60,
        fdm_num_elements=32,
        fdm_reference_num_elements=2**14,
        fdm_plot_points_per_cell=20,
    )


def _resolve_output_dir(save_dir: Path | None) -> Path:
    if save_dir is None:
        output_dir = Path(__file__).resolve().parent / "results_compare"
    else:
        output_dir = Path(save_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _piecewise_plot_arrays(
    problem: EnergyProblemConfig,
    points_per_side: int,
    evaluator,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = max(int(points_per_side), 2)
    x_left = np.linspace(problem.a, problem.x_interface, n_points)
    x_right = np.linspace(problem.x_interface, problem.b, n_points)
    u_left = evaluator(x_left, "left")
    u_right = evaluator(x_right, "right")
    x_plot = np.concatenate([x_left, np.array([np.nan]), x_right])
    u_plot = np.concatenate([u_left, np.array([np.nan]), u_right])
    return x_plot, u_plot


def compare_energy_alm_with_tfpm_auglag(
    energy_problem: EnergyProblemConfig | None = None,
    energy_solver: EnergySolverConfig | None = None,
    pde_solver: pde_module.SolverConfig | None = None,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> ComparisonResult:
    energy_problem = energy_problem or make_common_energy_problem()
    pde_problem = make_matching_pde_problem(energy_problem)
    energy_solver = energy_solver or make_energy_solver()
    pde_solver = pde_solver or make_pde_solver()

    if verbose:
        print("=== Solving nonlinear energy ALM ===")
    energy_result = solve_energy_alm(
        problem=energy_problem,
        solver=energy_solver,
        source_func=pde_module.default_source,
        verbose=verbose,
    )

    if verbose:
        print("\n=== Solving nonlinear_pde TFPM-AugLag ===")
    tfpm_result = pde_module.solve_newton_problem(
        problem=pde_problem,
        solver=pde_solver,
        source_func=pde_module.default_source,
        verbose=verbose,
    )

    if verbose:
        print(f"\n=== Solving nonlinear_pde FDM fine reference (N={pde_solver.fdm_reference_num_elements}) ===")
    fdm_reference_result = pde_module.solve_newton_fdm(
        problem=pde_problem,
        solver=pde_solver,
        num_elements=pde_solver.fdm_reference_num_elements,
        source_func=pde_module.default_source,
        verbose=verbose,
        label=f"FDM fine (N={pde_solver.fdm_reference_num_elements})",
    )

    n_compare = max(int(pde_solver.comparison_points_per_side), 2)
    x_left = np.linspace(energy_problem.a, energy_problem.x_interface, n_compare)
    x_right = np.linspace(energy_problem.x_interface, energy_problem.b, n_compare)
    reference_label = f"FDM fine (N={fdm_reference_result.num_elements})"
    reference_values = (
        fdm_reference_result.evaluate_on_side(x_left, side="left"),
        fdm_reference_result.evaluate_on_side(x_right, side="right"),
    )

    method_values = {
        "Energy ALM": (
            energy_result.evaluate_on_side(x_left, side="left"),
            energy_result.evaluate_on_side(x_right, side="right"),
        ),
        "TFPM-AugLag": (
            tfpm_result.evaluate_on_side(x_left, side="left"),
            tfpm_result.evaluate_on_side(x_right, side="right"),
        ),
    }
    relative_l2_errors = compute_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        reference_label=reference_label,
    )

    output_dir = _resolve_output_dir(save_dir)
    solution_plot_path = output_dir / "energy_alm_vs_tfpm_auglag_solution.png"
    error_plot_path = output_dir / "energy_alm_vs_tfpm_auglag_errors.png"
    summary_path = output_dir / "energy_alm_vs_tfpm_auglag_summary.txt"

    x_ref, u_ref = _piecewise_plot_arrays(energy_problem, n_compare, fdm_reference_result.evaluate_on_side)
    x_energy, u_energy = _piecewise_plot_arrays(energy_problem, n_compare, energy_result.evaluate_on_side)
    x_tfpm, u_tfpm = _piecewise_plot_arrays(energy_problem, n_compare, tfpm_result.evaluate_on_side)
    plot_solutions(
        [
            (reference_label, x_ref, u_ref),
            ("Energy ALM", x_energy, u_energy),
            ("TFPM-AugLag", x_tfpm, u_tfpm),
        ],
        xI=energy_problem.x_interface,
        save_path=solution_plot_path,
        reference_label=reference_label,
    )
    plot_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        xI=energy_problem.x_interface,
        save_path=error_plot_path,
        reference_label=reference_label,
    )

    timings = {
        "Energy ALM": energy_result.elapsed_seconds,
        "TFPM-AugLag": tfpm_result.elapsed_seconds,
        reference_label: fdm_reference_result.elapsed_seconds,
    }
    diagnostics = [
        f"Energy ALM converged = {energy_result.converged}",
        f"TFPM-AugLag converged = {tfpm_result.converged}",
        f"{reference_label} converged = {fdm_reference_result.converged}",
        f"Energy ALM space steps = {len(energy_result.history)}",
        f"TFPM-AugLag outer steps = {len(tfpm_result.history)}",
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
    if tfpm_result.history:
        diagnostics.append(f"TFPM-AugLag final update = {tfpm_result.history[-1].error_inf:.6e}")
    if fdm_reference_result.history:
        diagnostics.append(f"{reference_label} final update = {fdm_reference_result.history[-1].error_inf:.6e}")

    write_summary(
        summary_path,
        timings=timings,
        errors=relative_l2_errors,
        extra_lines=diagnostics,
        errors_title=f"Relative L2 errors vs {reference_label}:",
    )

    if verbose:
        print(f"\n=== Relative L2 errors vs {reference_label} ===")
        for key, value in relative_l2_errors.items():
            print(f"{key:>40s}: {value:.6e}")
        print("\nSaved files:")
        print(summary_path)
        print(solution_plot_path)
        print(error_plot_path)

    return ComparisonResult(
        energy_result=energy_result,
        tfpm_result=tfpm_result,
        fdm_reference_result=fdm_reference_result,
        relative_l2_errors=relative_l2_errors,
        timings=timings,
        solution_plot_path=solution_plot_path,
        error_plot_path=error_plot_path,
        summary_path=summary_path,
    )


def main() -> None:
    compare_energy_alm_with_tfpm_auglag()


if __name__ == "__main__":
    main()
