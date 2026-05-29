from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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

from ..convergence_rate.convergence_study import (  # noqa: E402
    ManufacturedCase,
    SingleRunResult,
    _build_cases,
    _run_single_case,
)


DEFAULT_CASE_NAME = "case_flux_jump_only"
DEFAULT_EPS_VALUES = [1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
DEFAULT_N_ELEMENTS = 64
REFERENCE_EPS_ORDER = -0.5
LAGRANGE_L2_REFERENCE_EPS_ORDER = -1.0


@dataclass(frozen=True)
class EqualEpsRun:
    eps: float
    result: SingleRunResult


@dataclass(frozen=True)
class EpsOrderEstimate:
    metric: str
    order: float
    r2: float


def _case_map() -> dict[str, ManufacturedCase]:
    return {case.name: case for case in _build_cases()}


def _get_case(name: str) -> ManufacturedCase:
    cases = _case_map()
    try:
        return cases[name]
    except KeyError as exc:
        available = ", ".join(sorted(cases))
        raise ValueError(f"Unknown case {name!r}. Available cases: {available}.") from exc


def _parse_eps_values(values: list[str] | None) -> list[float]:
    if not values:
        return list(DEFAULT_EPS_VALUES)
    eps_values = [float(value) for value in values]
    if any((not np.isfinite(value)) or value <= 0.0 for value in eps_values):
        raise ValueError("All eps values must be positive finite numbers.")
    return eps_values


def _fit_eps_order(runs: list[EqualEpsRun], attr: str) -> EpsOrderEstimate:
    """Fit p in error ~= C * eps^p."""
    eps_values = np.array([run.eps for run in runs], dtype=np.float64)
    errors = np.array([getattr(run.result, attr) for run in runs], dtype=np.float64)
    valid = np.isfinite(eps_values) & (eps_values > 0.0) & np.isfinite(errors) & (errors > 0.0)
    if np.count_nonzero(valid) < 2:
        return EpsOrderEstimate(metric=attr, order=float("nan"), r2=float("nan"))

    log_eps = np.log(eps_values[valid])
    log_errors = np.log(errors[valid])
    coeffs = np.polyfit(log_eps, log_errors, deg=1)
    fit = np.polyval(coeffs, log_eps)
    ss_res = float(np.sum((log_errors - fit) ** 2))
    ss_tot = float(np.sum((log_errors - np.mean(log_errors)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
    return EpsOrderEstimate(metric=attr, order=float(coeffs[0]), r2=r2)


def _local_eps_orders(runs: list[EqualEpsRun], attr: str) -> list[float]:
    orders: list[float] = []
    for left, right in zip(runs[:-1], runs[1:]):
        e0 = float(getattr(left.result, attr))
        e1 = float(getattr(right.result, attr))
        if left.eps <= 0.0 or right.eps <= 0.0 or e0 <= 0.0 or e1 <= 0.0:
            orders.append(float("nan"))
            continue
        orders.append(float(np.log(e1 / e0) / np.log(right.eps / left.eps)))
    return orders


def run_sweep(
    *,
    case_name: str = DEFAULT_CASE_NAME,
    eps_values: list[float] | None = None,
    n_elements: int = DEFAULT_N_ELEMENTS,
) -> tuple[ManufacturedCase, list[EqualEpsRun], list[EpsOrderEstimate]]:
    if n_elements <= 0:
        raise ValueError("n_elements must be positive.")

    case = _get_case(case_name)
    selected_eps = list(DEFAULT_EPS_VALUES if eps_values is None else eps_values)
    if any((not np.isfinite(value)) or value <= 0.0 for value in selected_eps):
        raise ValueError("All eps values must be positive finite numbers.")

    print(f"Running equal-eps sweep for {case.name} with N={n_elements}...")
    runs: list[EqualEpsRun] = []
    for eps in selected_eps:
        print(f"  eps = {eps:g}")
        result = _run_single_case(
            case,
            n_elements=n_elements,
            eps1=float(eps),
            eps2=float(eps),
            build_quad_n=12,
            particular_quad_segments=10,
            norm_quad_n=12,
            derivative_weight_mode="eps_weighted",
            flux_jump_average="eps_weighted",
            lagrange_residual_tol=1e-10,
        )
        runs.append(EqualEpsRun(eps=float(eps), result=result))

    estimates = [
        _fit_eps_order(runs, "tfpm_l2"),
        _fit_eps_order(runs, "tfpm_h1"),
        _fit_eps_order(runs, "lagrange_l2"),
        _fit_eps_order(runs, "lagrange_h1"),
    ]
    return case, runs, estimates


def _write_csv(csv_path: Path, runs: list[EqualEpsRun]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eps",
                "n_elements",
                "h",
                "tfpm_l2",
                "tfpm_eps_h1",
                "lagrange_l2",
                "lagrange_eps_h1",
                "lagrange_iter",
                "lagrange_primal_inf",
                "lagrange_stationarity_inf",
                "lagrange_residual_inf",
            ]
        )
        for run in runs:
            result = run.result
            writer.writerow(
                [
                    f"{run.eps:.16e}",
                    result.n_elements,
                    f"{result.h:.16e}",
                    f"{result.tfpm_l2:.16e}",
                    f"{result.tfpm_h1:.16e}",
                    f"{result.lagrange_l2:.16e}",
                    f"{result.lagrange_h1:.16e}",
                    result.lagrange_iter,
                    f"{result.lagrange_primal_inf:.16e}",
                    f"{result.lagrange_stationarity_inf:.16e}",
                    f"{result.lagrange_residual_inf:.16e}",
                ]
            )


def _write_summary(
    summary_path: Path,
    *,
    case: ManufacturedCase,
    runs: list[EqualEpsRun],
    estimates: list[EpsOrderEstimate],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    n_elements = runs[0].result.n_elements if runs else DEFAULT_N_ELEMENTS
    lines: list[str] = []
    lines.append("Equal-eps approximation-error sweep")
    lines.append("All runs set eps1 = eps2 = eps.")
    lines.append(
        "The manufactured solution, coefficient functions, mesh N, quadrature, and solver settings "
        "are fixed; f and jump_flux are rebuilt consistently from the same exact solution for each eps."
    )
    lines.append(
        "The reported eps-H1 error is relative: "
        "sqrt((||e||_L2^2 + eps ||e'||_L2^2) / (||u||_L2^2 + eps ||u'||_L2^2))."
    )
    lines.append("The eps-order p is fitted from error ~= C * eps^p; eps^(-1/2) corresponds to p = -0.5.")
    lines.append("")
    lines.append(f"case = {case.name}")
    lines.append(case.description)
    lines.append(f"N = {n_elements}")
    lines.append(f"c_left(x)  = {case.c_left_expr}")
    lines.append(f"c_right(x) = {case.c_right_expr}")
    lines.append(f"u_left(x)  = {case.u_left_expr}")
    lines.append(f"u_right(x) = {case.u_right_expr}")
    lines.append("")
    lines.append(
        "eps          TFPM_L2       eps-order  TFPM_EpsH1    eps-order  "
        "Lagrange_L2   eps-order  Lagrange_EpsH1   eps-order  KKTIter"
    )

    tfpm_l2_orders = _local_eps_orders(runs, "tfpm_l2")
    tfpm_h1_orders = _local_eps_orders(runs, "tfpm_h1")
    lagrange_l2_orders = _local_eps_orders(runs, "lagrange_l2")
    lagrange_h1_orders = _local_eps_orders(runs, "lagrange_h1")

    for idx, run in enumerate(runs):
        result = run.result
        tfpm_l2_order = "-" if idx == 0 else f"{tfpm_l2_orders[idx - 1]:9.4f}"
        tfpm_h1_order = "-" if idx == 0 else f"{tfpm_h1_orders[idx - 1]:9.4f}"
        lagrange_l2_order = "-" if idx == 0 else f"{lagrange_l2_orders[idx - 1]:9.4f}"
        lagrange_h1_order = "-" if idx == 0 else f"{lagrange_h1_orders[idx - 1]:9.4f}"
        lines.append(
            f"{run.eps:10.3e}  "
            f"{result.tfpm_l2:12.5e}  {tfpm_l2_order:>9s}  "
            f"{result.tfpm_h1:12.5e}  {tfpm_h1_order:>9s}  "
            f"{result.lagrange_l2:12.5e}  {lagrange_l2_order:>9s}  "
            f"{result.lagrange_h1:16.5e}  {lagrange_h1_order:>9s}  "
            f"{result.lagrange_iter:7d}"
        )

    lines.append("")
    lines.append("Global estimated eps-orders p in error ~= C * eps^p:")
    for estimate in estimates:
        inverse_order = -estimate.order
        lines.append(
            f"{estimate.metric}: p = {estimate.order:.6f}, "
            f"equivalent eps^(-{inverse_order:.6f}), R^2 = {estimate.r2:.6f}"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_errors(plot_path: Path, *, case: ManufacturedCase, runs: list[EqualEpsRun]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    eps_values = np.array([run.eps for run in runs], dtype=np.float64)
    tfpm_l2 = np.array([run.result.tfpm_l2 for run in runs], dtype=np.float64)
    tfpm_h1 = np.array([run.result.tfpm_h1 for run in runs], dtype=np.float64)
    lagrange_l2 = np.array([run.result.lagrange_l2 for run in runs], dtype=np.float64)
    lagrange_h1 = np.array([run.result.lagrange_h1 for run in runs], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.loglog(eps_values, tfpm_l2, marker="o", linewidth=1.8, label=r"TFPM $L^2$")
    ax.loglog(eps_values, tfpm_h1, marker="s", linewidth=1.8, label=r"TFPM eps-$H^1$")
    ax.loglog(eps_values, lagrange_l2, marker="^", linewidth=1.8, label=r"Lagrange $L^2$")
    ax.loglog(
        eps_values,
        lagrange_h1,
        marker="D",
        linewidth=1.8,
        label=r"Lagrange eps-$H^1$",
    )
    if eps_values.size > 0:
        anchor_index = eps_values.size - 1
        eps0 = float(eps_values[anchor_index])
        lagrange_l2_reference = float(lagrange_l2[anchor_index]) * (
            eps_values / eps0
        ) ** LAGRANGE_L2_REFERENCE_EPS_ORDER
        ax.loglog(
            eps_values,
            lagrange_l2_reference,
            color="tab:purple",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=r"$O(\epsilon^{-1})$",
        )
        lagrange_h1_reference = float(lagrange_h1[anchor_index]) * (
            eps_values / eps0
        ) ** REFERENCE_EPS_ORDER
        ax.loglog(
            eps_values,
            lagrange_h1_reference,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label=r"$O(\epsilon^{-1/2})$",
        )
    ax.invert_xaxis()
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_xlabel(r"equal $\epsilon$  ($\epsilon_1=\epsilon_2$)")
    ax.set_ylabel("relative error")
    n_elements = runs[0].result.n_elements if runs else DEFAULT_N_ELEMENTS
    ax.set_title(f"Error vs equal eps: {case.name}, N={n_elements}")
    ax.legend(loc="best", frameon=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90)
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)


def write_outputs(
    *,
    out_dir: Path,
    case: ManufacturedCase,
    runs: list[EqualEpsRun],
    estimates: list[EpsOrderEstimate],
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    csv_path = out_dir / "error_vs_equal_eps.csv"
    plot_path = out_dir / "error_vs_equal_eps.png"
    _write_summary(summary_path, case=case, runs=runs, estimates=estimates)
    _write_csv(csv_path, runs)
    _plot_errors(plot_path, case=case, runs=runs)
    return summary_path, csv_path, plot_path


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Sweep eps1 = eps2 = eps and report approximation error."
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE_NAME,
        help=f"Manufactured case name. Default: {DEFAULT_CASE_NAME}.",
    )
    parser.add_argument(
        "--eps-values",
        nargs="+",
        default=None,
        help="Positive eps values to sweep. Default: 1 0.5 0.2 0.1 0.05 0.02 0.01.",
    )
    parser.add_argument(
        "--n-elements",
        type=int,
        default=DEFAULT_N_ELEMENTS,
        help=f"Fixed number of elements. Default: {DEFAULT_N_ELEMENTS}.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "results",
        help="Directory where summary, CSV, and figure are written.",
    )
    args = parser.parse_args()

    eps_values = _parse_eps_values(args.eps_values)
    case, runs, estimates = run_sweep(
        case_name=args.case,
        eps_values=eps_values,
        n_elements=args.n_elements,
    )
    summary_path, csv_path, plot_path = write_outputs(
        out_dir=args.out_dir,
        case=case,
        runs=runs,
        estimates=estimates,
    )
    print(summary_path)
    print(csv_path)
    print(plot_path)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
