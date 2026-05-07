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

from ...core.methods_tfpm import assemble_lagrange_kkt_system
from ...core.tfpm_local import (
    build_elements,
    build_uniform_grid,
    evaluate_solution_fine,
    evaluate_tfpm_on_side,
    interface_node_from_grid,
)

ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(slots=True)
class ProblemConfig:
    a: float = 0.0
    b: float = 1.0
    x_interface: float = 0.5
    num_elements: int = 32
    quad_n: int = 12
    n_seg_gauss: int = 8
    left_bc: float = 0.0
    right_bc: float = 0.0
    jump_u: float = 0.0
    jump_du: float = 0.0
    flux_jump_weights: tuple[float, float] = (0.5, 0.5)
    basis_kind: str = "endpoint_auto"


@dataclass(slots=True)
class SolverConfig:
    rho: float = 10.0
    max_space_iter: int = 40
    space_tol: float = 1.0e-10
    energy_tol: float = 1.0e-12
    max_aug_lag_iter: int = 80
    max_inner_newton_iter: int = 30
    tol_primal: float = 1.0e-10
    tol_stationarity: float = 1.0e-10
    tol_inner_grad: float = 1.0e-11
    line_search_max_iter: int = 20
    armijo: float = 1.0e-4
    error_check_points: int = 400
    snapshot_points_per_element: int = 80
    plot_points_per_element: int = 60


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
    x_snapshot: np.ndarray | None = None
    u_snapshot: np.ndarray | None = None

    def snapshot_evaluator(self) -> ArrayFunc:
        x_snapshot = self.x_snapshot
        u_snapshot = self.u_snapshot

        def evaluate(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            if x_snapshot is None or u_snapshot is None:
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
    history: list[SpaceIterationRecord] = field(default_factory=list)
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


def default_source(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return 1.0e2 * x_arr * (1.0 - x_arr)


def build_grid(problem: ProblemConfig) -> np.ndarray:
    grid, _h, _interface_node = build_uniform_grid(
        N=problem.num_elements,
        a=problem.a,
        b=problem.b,
        xI=problem.x_interface,
    )
    return grid


def make_newton_space_functions(
    previous_iterate: ArrayFunc,
    source_func: ArrayFunc,
    grid: np.ndarray | None = None,
) -> tuple[ArrayFunc, ArrayFunc, Callable[[float], float] | None, Callable[[float], float] | None]:
    if grid is None:
        def c_func(x: np.ndarray, prev: ArrayFunc = previous_iterate) -> np.ndarray:
            u_prev = prev(x)
            return 3.0 * u_prev * u_prev

        c_left_trace = None
        c_right_trace = None
    else:
        grid_arr = np.asarray(grid, dtype=np.float64)
        mids = 0.5 * (grid_arr[:-1] + grid_arr[1:])
        coeffs = np.maximum(3.0 * previous_iterate(mids) ** 2, 0.0)

        def element_index_from_right(x: float) -> int:
            idx = int(np.searchsorted(grid_arr, float(x), side="right") - 1)
            return int(np.clip(idx, 0, coeffs.size - 1))

        def element_index_from_left(x: float) -> int:
            idx = int(np.searchsorted(grid_arr, float(x), side="left") - 1)
            return int(np.clip(idx, 0, coeffs.size - 1))

        def c_func(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            idx = np.searchsorted(grid_arr, x_arr, side="right") - 1
            idx = np.clip(idx, 0, coeffs.size - 1)
            return coeffs[idx]

        def c_left_trace(x: float) -> float:
            return float(coeffs[element_index_from_left(x)])

        def c_right_trace(x: float) -> float:
            return float(coeffs[element_index_from_right(x)])

    def f_func(x: np.ndarray, source: ArrayFunc = source_func, prev: ArrayFunc = previous_iterate) -> np.ndarray:
        u_prev = prev(x)
        return source(x) + 2.0 * u_prev * u_prev * u_prev

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
        jump_du=problem.jump_du,
        xI=problem.x_interface,
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

    interface_node = interface_node_from_grid(grid, problem.x_interface)
    e_left = interface_node - 1
    e_right = interface_node
    el_left = elems[e_left]
    el_right = elems[e_right]
    w_left, w_right = problem.flux_jump_weights
    q = float(problem.jump_du)

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
    source_func: ArrayFunc,
) -> tuple[float, np.ndarray, np.ndarray]:
    ndof = 2 * len(elems)
    grad = np.zeros(ndof, dtype=np.float64)
    hess = np.zeros((ndof, ndof), dtype=np.float64)
    energy = 0.0

    for e, el in enumerate(elems):
        idx = np.array([2 * e, 2 * e + 1], dtype=int)
        ze = z[idx]
        phi = np.vstack([el.phi1_q, el.phi2_q])
        dphi = np.vstack([el.dphi1_q, el.dphi2_q])
        u = el.up_q + ze[0] * el.phi1_q + ze[1] * el.phi2_q
        du = el.dup_q + ze[0] * el.dphi1_q + ze[1] * el.dphi2_q
        f = source_func(el.xq)
        w = el.wq

        energy += float(np.sum(w * (0.5 * du * du + 0.25 * u**4 - f * u)))
        residual = u**3 - f
        grad[idx] += np.array(
            [
                np.sum(w * (du * dphi[0] + residual * phi[0])),
                np.sum(w * (du * dphi[1] + residual * phi[1])),
            ],
            dtype=np.float64,
        )

        local_hess = dphi @ ((w[None, :] * dphi).T)
        local_hess += phi @ (((3.0 * w * u * u)[None, :] * phi).T)
        hess[np.ix_(idx, idx)] += local_hess

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
    source_func: ArrayFunc,
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
        for newton_iter in range(1, solver.max_inner_newton_iter + 1):
            energy, grad_energy, hess_energy = energy_gradient_hessian(problem, grid, elems, z, source_func)
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
                trial_energy, _trial_grad, _trial_hess = energy_gradient_hessian(
                    problem, grid, elems, z_trial, source_func
                )
                trial_residual = C @ z_trial - d
                trial_aug = _augmented_value(trial_energy, trial_residual, lam, solver.rho)
                if np.isfinite(trial_aug) and trial_aug <= current_aug + solver.armijo * alpha * directional:
                    z = z_trial
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                z = z + alpha * step

        energy, grad_energy, _hess_energy = energy_gradient_hessian(problem, grid, elems, z, source_func)
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


def solve_energy_alm(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    source_func: ArrayFunc = default_source,
    verbose: bool = True,
) -> SolveResult:
    problem = problem or ProblemConfig()
    solver = solver or SolverConfig()
    grid = build_grid(problem)
    state = IterateState()
    history: list[SpaceIterationRecord] = []
    previous_iterate = state.snapshot_evaluator()
    z_guess: np.ndarray | None = None
    lam_guess: np.ndarray | None = None
    previous_energy = np.inf
    converged = False

    if verbose:
        print("=== Starting Newton-space nonlinear energy ALM ===")

    t_start = time.perf_counter()
    for iteration in range(1, solver.max_space_iter + 1):
        c_func, f_func, c_left_trace, c_right_trace = make_newton_space_functions(previous_iterate, source_func)
        _grid, elems = build_elements(
            c_func=c_func,
            f_func=f_func,
            grid=grid,
            quad_n=problem.quad_n,
            xI=problem.x_interface,
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
            source_func=source_func,
            z_initial=z_guess,
            lam_initial=lam_guess,
        )

        x_check = np.linspace(problem.a, problem.b, max(int(solver.error_check_points), 2))
        u_old = previous_iterate(x_check)
        state.update(elems=elems, z=z_new, forcing=f_func, config=problem, solver=solver)
        current_iterate = state.snapshot_evaluator()
        u_new = current_iterate(x_check)
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
            print("=> Newton-space energy ALM reached the target tolerance.")
        else:
            print("=> Warning: Newton-space energy ALM hit the maximum number of iterations.")
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


def plot_solution(result: SolveResult, save_path: Path) -> Path:
    x_plot, u_plot = result.plot_arrays()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, u_plot, linewidth=2.0, label="energy ALM")
    ax.axvline(result.problem.x_interface, color="0.35", linestyle="--", linewidth=1.0, label="interface")
    ax.set_title(r"Newton-space minimization of $\int 0.5|u'|^2 + 0.25u^4 - fu$")
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iters, energies, marker="o")
    axes[0].set_xlabel("space iteration")
    axes[0].set_ylabel("energy")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(iters, np.maximum(steps, np.finfo(float).tiny), marker="o")
    axes[1].set_xlabel("space iteration")
    axes[1].set_ylabel("L-inf update")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def write_summary(result: SolveResult, save_path: Path) -> Path:
    lines = [
        "Newton-space nonlinear energy ALM summary",
        f"converged = {result.converged}",
        f"elapsed_seconds = {result.elapsed_seconds:.8e}",
        f"num_elements = {result.problem.num_elements}",
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


def run_experiment(
    problem: ProblemConfig | None = None,
    solver: SolverConfig | None = None,
    source_func: ArrayFunc = default_source,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> SolveResult:
    result = solve_energy_alm(problem=problem, solver=solver, source_func=source_func, verbose=verbose)
    output_dir = _resolve_output_dir(save_dir)
    solution_path = plot_solution(result, output_dir / "energy_alm_solution.png")
    history_path = plot_history(result, output_dir / "energy_alm_history.png")
    summary_path = write_summary(result, output_dir / "energy_alm_summary.txt")

    if verbose:
        print("Saved files:")
        print(summary_path)
        print(solution_path)
        print(history_path)

    return result


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
