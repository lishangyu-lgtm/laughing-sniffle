from __future__ import annotations

import sys
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

from ...core.linear_interface import (
    ExactReference,
    ProblemConfig,
    X_DOMAIN,
    flux_jump_average_weights,
    make_transformed_coefficient,
    make_transformed_coefficient_trace_functions,
    make_transformed_rhs,
    normalize_flux_jump_average,
    transformed_interface,
    x_to_y,
)
from ...core.methods_tfpm import (
    assemble_lagrange_kkt_system,
    solve_augmented_lagrange,
    solve_tfpm_strong_matching,
)
from ...core.tfpm_local import (
    build_elements,
    build_uniform_grid,
    evaluate_tfpm_state_on_side,
    gauss_legendre,
    map_to_interval,
)


ArrayFunc = Callable[[np.ndarray], np.ndarray]
DEFAULT_N_VALUES = [4, 8, 16, 32, 64, 128, 256, 512]


@dataclass(frozen=True)
class ManufacturedCase:
    name: str
    description: str
    c_left_expr: str
    c_right_expr: str
    u_left_expr: str
    u_right_expr: str
    c_left: ArrayFunc
    c_right: ArrayFunc
    u_left: ArrayFunc
    u_right: ArrayFunc
    du_left: ArrayFunc
    du_right: ArrayFunc
    d2u_left: ArrayFunc
    d2u_right: ArrayFunc
    x_interface: float = 0.5

    def build_problem(
        self,
        eps1: float = 1.0,
        eps2: float = 1.0,
    ) -> tuple[ProblemConfig, ArrayFunc, ArrayFunc, ExactReference]:
        xI = float(self.x_interface)
        left_bc = _scalar_eval(self.u_left, X_DOMAIN[0])
        right_bc = _scalar_eval(self.u_right, X_DOMAIN[1])
        jump_u = _scalar_eval(self.u_right, xI) - _scalar_eval(self.u_left, xI)
        jump_flux = eps2 * _scalar_eval(self.du_right, xI) - eps1 * _scalar_eval(self.du_left, xI)

        cfg = ProblemConfig(
            eps1=eps1,
            eps2=eps2,
            x_interface=xI,
            left_bc=left_bc,
            right_bc=right_bc,
            jump_u=jump_u,
            jump_flux=jump_flux,
            reference_source="exact",
        )

        def c_func(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            return np.where(x <= xI, self.c_left(x), self.c_right(x))

        def f_func(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            left = -eps1 * self.d2u_left(x) + self.c_left(x) * self.u_left(x)
            right = -eps2 * self.d2u_right(x) + self.c_right(x) * self.u_right(x)
            return np.where(x <= xI, left, right)

        exact_reference = ExactReference(
            left=self.u_left,
            right=self.u_right,
            left_derivative=self.du_left,
            right_derivative=self.du_right,
            label=f"Exact ({self.name})",
        )
        return cfg, c_func, f_func, exact_reference


@dataclass(frozen=True)
class SingleRunResult:
    n_elements: int
    h: float
    tfpm_l2: float
    tfpm_h1: float
    auglag_l2: float
    auglag_h1: float
    auglag_iter: int
    auglag_primal_inf: float
    auglag_stationarity_inf: float


@dataclass(frozen=True)
class RateEstimate:
    order: float
    r2: float
    start_n: int
    end_n: int
    n_points: int


@dataclass(frozen=True)
class EpsilonScenario:
    name: str
    eps1: float
    eps2: float
    description: str


@dataclass(frozen=True)
class ScenarioCaseReport:
    scenario: EpsilonScenario
    case: ManufacturedCase
    cfg: ProblemConfig
    runs: list[SingleRunResult]
    estimates: dict[str, RateEstimate]


def _scalar_eval(func: ArrayFunc, x: float) -> float:
    return float(np.asarray(func(np.array([x], dtype=np.float64)), dtype=np.float64)[0])


def _integrate_relative_errors(
    cfg: ProblemConfig,
    exact_reference: ExactReference,
    elems,
    y_f_func: ArrayFunc,
    solutions: dict[str, np.ndarray],
    grid_x: np.ndarray,
    x_interface: float,
    y_interface: float,
    n_seg_gauss: int,
    quad_n: int,
    derivative_weight_mode: str = "standard",
) -> dict[str, tuple[float, float]]:
    xi, wi = gauss_legendre(quad_n)

    den_l2 = 0.0
    den_h1 = 0.0
    numerators = {name: {"l2": 0.0, "h1": 0.0} for name in solutions}

    for e in range(grid_x.size - 1):
        xL = float(grid_x[e])
        xR = float(grid_x[e + 1])
        xq, wq = map_to_interval(xi, wi, xL, xR)
        yq = x_to_y(xq, cfg)
        side = "left" if xR <= x_interface + 1e-14 else "right"
        eps_side = cfg.eps1 if side == "left" else cfg.eps2
        if derivative_weight_mode == "standard":
            derivative_weight = 1.0
        elif derivative_weight_mode == "eps_weighted":
            derivative_weight = eps_side
        else:
            raise ValueError("derivative_weight_mode must be 'standard' or 'eps_weighted'.")

        ref_u = exact_reference.evaluate(xq, side=side)
        ref_du = exact_reference.evaluate_derivative(xq, side=side)
        den_l2 += float(np.sum(wq * ref_u**2))
        den_h1 += float(np.sum(wq * (ref_u**2 + derivative_weight * ref_du**2)))

        for name, z in solutions.items():
            num_u, num_du_transformed = evaluate_tfpm_state_on_side(
                elems=elems,
                z=z,
                f_func=y_f_func,
                x_eval=yq,
                side=side,
                xI=y_interface,
                n_seg_gauss=n_seg_gauss,
            )
            num_du = num_du_transformed / (cfg.eps1 if side == "left" else cfg.eps2)
            numerators[name]["l2"] += float(np.sum(wq * (num_u - ref_u) ** 2))
            numerators[name]["h1"] += float(
                np.sum(
                    wq
                    * (
                        (num_u - ref_u) ** 2
                        + derivative_weight * (num_du - ref_du) ** 2
                    )
                )
            )

    den_l2 = max(den_l2, 1e-30)
    den_h1 = max(den_h1, 1e-30)

    return {
        name: (
            float(np.sqrt(values["l2"] / den_l2)),
            float(np.sqrt(values["h1"] / den_h1)),
        )
        for name, values in numerators.items()
    }


def _fit_rate(n_values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    log_h = np.log(1.0 / n_values)
    log_e = np.log(errors)
    coeffs = np.polyfit(log_h, log_e, deg=1)
    fit = np.polyval(coeffs, log_h)
    ss_res = float(np.sum((log_e - fit) ** 2))
    ss_tot = float(np.sum((log_e - np.mean(log_e)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
    return float(coeffs[0]), r2


def _estimate_rate(runs: list[SingleRunResult], attr: str) -> RateEstimate:
    n_values = np.array([run.n_elements for run in runs], dtype=np.float64)
    errors = np.array([getattr(run, attr) for run in runs], dtype=np.float64)

    valid = np.isfinite(errors) & (errors > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError(f"Need at least three positive errors to fit {attr}.")

    n_valid = n_values[valid]
    e_valid = errors[valid]
    first_error = float(e_valid[0])
    floor = max(5e-13, 1e-10 * first_error)

    candidates: list[tuple[float, float, int, int, int, float]] = []
    for window_len in (4, 3):
        if e_valid.size < window_len:
            continue
        for start in range(0, e_valid.size - window_len + 1):
            end = start + window_len - 1
            window_errors = e_valid[start : end + 1]
            if np.any(window_errors <= floor):
                continue
            if np.any(window_errors[1:] >= 0.995 * window_errors[:-1]):
                continue
            local_orders = np.log(window_errors[:-1] / window_errors[1:]) / np.log(
                n_valid[start + 1 : end + 1] / n_valid[start:end]
            )
            order, r2 = _fit_rate(n_valid[start : end + 1], window_errors)
            spread = float(np.std(local_orders))
            candidates.append((r2, -spread, end, window_len, start, order))

    if candidates:
        best_r2, _neg_spread, end, window_len, start, order = max(candidates)
        r2 = best_r2
    else:
        start = 0
        end = min(2, e_valid.size - 1)
        order, r2 = _fit_rate(n_valid[start : end + 1], e_valid[start : end + 1])

    return RateEstimate(
        order=order,
        r2=r2,
        start_n=int(n_valid[start]),
        end_n=int(n_valid[end]),
        n_points=end - start + 1,
    )


def _local_orders(runs: list[SingleRunResult], attr: str) -> list[float]:
    errors = [getattr(run, attr) for run in runs]
    n_values = [run.n_elements for run in runs]
    rates: list[float] = []
    for i in range(len(runs) - 1):
        if errors[i] <= 0.0 or errors[i + 1] <= 0.0:
            rates.append(float("nan"))
            continue
        ratio_h = n_values[i + 1] / n_values[i]
        rates.append(float(np.log(errors[i] / errors[i + 1]) / np.log(ratio_h)))
    return rates


def _run_single_case(
    case: ManufacturedCase,
    n_elements: int,
    *,
    eps1: float = 1.0,
    eps2: float = 1.0,
    build_quad_n: int,
    particular_quad_segments: int,
    norm_quad_n: int,
    derivative_weight_mode: str = "standard",
    flux_jump_average: str,
    auglag_rho: float,
    auglag_max_iter: int,
    auglag_tol: float,
) -> SingleRunResult:
    cfg, x_c_func, x_f_func, exact_reference = case.build_problem(eps1=eps1, eps2=eps2)
    grid_x, h, _interface_node = build_uniform_grid(
        N=n_elements,
        a=X_DOMAIN[0],
        b=X_DOMAIN[1],
        xI=cfg.x_interface,
    )
    y_grid = x_to_y(grid_x, cfg)
    y_interface = transformed_interface(cfg)

    y_c_func = make_transformed_coefficient(cfg, x_c_func)
    y_f_func = make_transformed_rhs(cfg, x_f_func)
    y_c_left_trace, y_c_right_trace = make_transformed_coefficient_trace_functions(cfg, x_c_func)

    _mode = normalize_flux_jump_average(flux_jump_average)
    flux_weights = flux_jump_average_weights(cfg, _mode)

    grid_y, elems = build_elements(
        c_func=y_c_func,
        f_func=y_f_func,
        grid=y_grid,
        quad_n=build_quad_n,
        xI=y_interface,
        n_seg_gauss=particular_quad_segments,
        c_left_trace=y_c_left_trace,
        c_right_trace=y_c_right_trace,
    )

    z_tfpm = solve_tfpm_strong_matching(
        grid=grid_y,
        elems=elems,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        qjump=cfg.jump_flux,
        xI=y_interface,
    )

    _K, _rhs, H, l, C, d = assemble_lagrange_kkt_system(
        grid=grid_y,
        elems=elems,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        jump_du=cfg.jump_flux,
        xI=y_interface,
        c_func=y_c_func,
        use_true_c=True,
        flux_jump_weights=flux_weights,
    )
    z_auglag, _lam, aug_history = solve_augmented_lagrange(
        H=H,
        l=l,
        C=C,
        d=d,
        rho=auglag_rho,
        max_iter=auglag_max_iter,
        tol_primal=auglag_tol,
        tol_stationarity=auglag_tol,
        relax=1.0,
        verbose=False,
    )

    errors = _integrate_relative_errors(
        cfg=cfg,
        exact_reference=exact_reference,
        elems=elems,
        y_f_func=y_f_func,
        solutions={
            "tfpm": z_tfpm,
            "auglag": z_auglag,
        },
        grid_x=grid_x,
        x_interface=cfg.x_interface,
        y_interface=y_interface,
        n_seg_gauss=particular_quad_segments,
        quad_n=norm_quad_n,
        derivative_weight_mode=derivative_weight_mode,
    )

    return SingleRunResult(
        n_elements=n_elements,
        h=h,
        tfpm_l2=errors["tfpm"][0],
        tfpm_h1=errors["tfpm"][1],
        auglag_l2=errors["auglag"][0],
        auglag_h1=errors["auglag"][1],
        auglag_iter=aug_history.iter,
        auglag_primal_inf=aug_history.primal_inf,
        auglag_stationarity_inf=aug_history.stationarity_inf,
    )


def _build_cases() -> list[ManufacturedCase]:
    xI = 0.5
    pi = np.pi

    def base1(x: np.ndarray) -> np.ndarray:
        return 1.0 + 0.4 * x + 0.2 * x**2 + 0.15 * np.sin(pi * x)

    def dbase1(x: np.ndarray) -> np.ndarray:
        return 0.4 + 0.4 * x + 0.15 * pi * np.cos(pi * x)

    def d2base1(x: np.ndarray) -> np.ndarray:
        return 0.4 - 0.15 * pi**2 * np.sin(pi * x)

    def base2(x: np.ndarray) -> np.ndarray:
        return 0.9 + np.exp(0.4 * x) + 0.1 * x**3

    def dbase2(x: np.ndarray) -> np.ndarray:
        return 0.4 * np.exp(0.4 * x) + 0.3 * x**2

    def d2base2(x: np.ndarray) -> np.ndarray:
        return 0.16 * np.exp(0.4 * x) + 0.6 * x

    def base3(x: np.ndarray) -> np.ndarray:
        return 1.1 + 0.25 * np.cos(2.0 * pi * x) + 0.3 * x

    def dbase3(x: np.ndarray) -> np.ndarray:
        return -0.5 * pi * np.sin(2.0 * pi * x) + 0.3

    def d2base3(x: np.ndarray) -> np.ndarray:
        return -(pi**2) * np.cos(2.0 * pi * x)

    return [
        ManufacturedCase(
            name="case_zero_jump",
            description="continuous interface data, smooth polynomial/exponential coefficient",
            c_left_expr="1.8 + x + 0.5*x^2",
            c_right_expr="2.0 + exp(0.6*(x-0.5))",
            u_left_expr="1 + 0.4*x + 0.2*x^2 + 0.15*sin(pi*x)",
            u_right_expr="u_left + 0.35*(x-0.5)^2 + 0.20*(x-0.5)^3",
            c_left=lambda x: 1.8 + x + 0.5 * x**2,
            c_right=lambda x: 2.0 + np.exp(0.6 * (x - xI)),
            u_left=base1,
            u_right=lambda x: base1(x) + 0.35 * (x - xI) ** 2 + 0.20 * (x - xI) ** 3,
            du_left=dbase1,
            du_right=lambda x: dbase1(x) + 0.70 * (x - xI) + 0.60 * (x - xI) ** 2,
            d2u_left=d2base1,
            d2u_right=lambda x: d2base1(x) + 0.70 + 1.20 * (x - xI),
            x_interface=xI,
        ),
        ManufacturedCase(
            name="case_value_flux_jump",
            description="nonzero value and flux jumps with smooth polynomial/exponential coefficient",
            c_left_expr="1.8 + x + 0.5*x^2",
            c_right_expr="2.0 + exp(0.6*(x-0.5))",
            u_left_expr="0.9 + exp(0.4*x) + 0.1*x^3",
            u_right_expr="u_left + 0.5 + 0.25*(x-0.5) + 0.10*(x-0.5)^2",
            c_left=lambda x: 1.8 + x + 0.5 * x**2,
            c_right=lambda x: 2.0 + np.exp(0.6 * (x - xI)),
            u_left=base2,
            u_right=lambda x: base2(x) + 0.5 + 0.25 * (x - xI) + 0.10 * (x - xI) ** 2,
            du_left=dbase2,
            du_right=lambda x: dbase2(x) + 0.25 + 0.20 * (x - xI),
            d2u_left=d2base2,
            d2u_right=lambda x: d2base2(x) + 0.20,
            x_interface=xI,
        ),
        ManufacturedCase(
            name="case_flux_jump_only",
            description="zero value jump, nonzero flux jump, stronger right coefficient growth",
            c_left_expr="2.5 + 1.5*(x+0.1)^2",
            c_right_expr="1.7 + exp(1.2*(x-0.5))",
            u_left_expr="1.1 + 0.25*cos(2*pi*x) + 0.3*x",
            u_right_expr="u_left - 0.15*(x-0.5) + 0.25*(x-0.5)^2 - 0.08*(x-0.5)^4",
            c_left=lambda x: 2.5 + 1.5 * (x + 0.1) ** 2,
            c_right=lambda x: 1.7 + np.exp(1.2 * (x - xI)),
            u_left=base3,
            u_right=lambda x: base3(x) - 0.15 * (x - xI) + 0.25 * (x - xI) ** 2 - 0.08 * (x - xI) ** 4,
            du_left=dbase3,
            du_right=lambda x: dbase3(x) - 0.15 + 0.50 * (x - xI) - 0.32 * (x - xI) ** 3,
            d2u_left=d2base3,
            d2u_right=lambda x: d2base3(x) + 0.50 - 0.96 * (x - xI) ** 2,
            x_interface=xI,
        ),
        ManufacturedCase(
            name="case_small_left_slope",
            description=(
                "nonzero value and flux jumps with a left coefficient whose boundary slope is nearly "
                "zero on fine meshes; this was the previously unstable Airy case"
            ),
            c_left_expr="1.5 + 0.3*cos(1.3*x) + 0.4*x^2",
            c_right_expr="2.2 + 0.25*sin(1.7*x) + 0.6*(x-0.5)^2",
            u_left_expr="0.9 + exp(0.4*x) + 0.1*x^3",
            u_right_expr="u_left + 0.5 + 0.25*(x-0.5) + 0.10*(x-0.5)^2",
            c_left=lambda x: 1.5 + 0.3 * np.cos(1.3 * x) + 0.4 * x**2,
            c_right=lambda x: 2.2 + 0.25 * np.sin(1.7 * x) + 0.6 * (x - xI) ** 2,
            u_left=base2,
            u_right=lambda x: base2(x) + 0.5 + 0.25 * (x - xI) + 0.10 * (x - xI) ** 2,
            du_left=dbase2,
            du_right=lambda x: dbase2(x) + 0.25 + 0.20 * (x - xI),
            d2u_left=d2base2,
            d2u_right=lambda x: d2base2(x) + 0.20,
            x_interface=xI,
        ),
    ]


def _build_eps_scenarios() -> list[EpsilonScenario]:
    return [
        EpsilonScenario(
            name="eps_equal",
            eps1=1.0,
            eps2=1.0,
            description="baseline equal diffusion",
        ),
        EpsilonScenario(
            name="eps_right_smaller",
            eps1=1.0,
            eps2=0.01,
            description="100x smaller diffusion on the right side",
        ),
        EpsilonScenario(
            name="eps_left_smaller",
            eps1=0.01,
            eps2=1.0,
            description="100x smaller diffusion on the left side",
        ),
    ]


def _get_eps_scenario(name: str) -> EpsilonScenario:
    for scenario in _build_eps_scenarios():
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unknown eps scenario: {name}")


def _write_summary(report_path: Path, reports: list[ScenarioCaseReport]) -> None:
    lines: list[str] = []
    lines.append("Convergence study with varying eps1 and eps2")
    lines.append(
        "Reference: exact manufactured solution in physical x. For each eps-scenario, "
        "f is rebuilt from -eps_i*u'' + c*u on each side and jump_flux = eps2*u'_R - eps1*u'_L."
    )
    lines.append(
        "Second norm reported below is sqrt(||u||_{L2}^2 + ||eps*u'||_{L2}^2), "
        "using piecewise eps(x)."
    )
    lines.append("")

    current_scenario: str | None = None
    for report in reports:
        scenario = report.scenario
        case = report.case
        cfg = report.cfg
        runs = report.runs

        if scenario.name != current_scenario:
            if current_scenario is not None:
                lines.append("")
            lines.append(f"=== {scenario.name} ===")
            lines.append(
                f"eps1 = {scenario.eps1:.6g}, eps2 = {scenario.eps2:.6g} "
                f"({scenario.description})"
            )
            lines.append("")
            current_scenario = scenario.name

        lines.append(f"--- {case.name} ---")
        lines.append(case.description)
        lines.append(f"c_left(x)  = {case.c_left_expr}")
        lines.append(f"c_right(x) = {case.c_right_expr}")
        lines.append(f"u_left(x)  = {case.u_left_expr}")
        lines.append(f"u_right(x) = {case.u_right_expr}")
        lines.append(
            f"interface data: jump_u = {cfg.jump_u:.12e}, jump_flux = {cfg.jump_flux:.12e}"
        )
        lines.append("")
        lines.append(
            "N        h            TFPM_L2        ord        TFPM_EpsNorm   ord        "
            "AugLag_L2      ord        AugLag_EpsNorm ord       AugIter"
        )

        tfpm_l2_orders = _local_orders(runs, "tfpm_l2")
        tfpm_h1_orders = _local_orders(runs, "tfpm_h1")
        aug_l2_orders = _local_orders(runs, "auglag_l2")
        aug_h1_orders = _local_orders(runs, "auglag_h1")

        for idx, run in enumerate(runs):
            tfpm_l2_ord = "-" if idx == 0 else f"{tfpm_l2_orders[idx - 1]:8.4f}"
            tfpm_h1_ord = "-" if idx == 0 else f"{tfpm_h1_orders[idx - 1]:8.4f}"
            aug_l2_ord = "-" if idx == 0 else f"{aug_l2_orders[idx - 1]:8.4f}"
            aug_h1_ord = "-" if idx == 0 else f"{aug_h1_orders[idx - 1]:8.4f}"
            lines.append(
                f"{run.n_elements:4d}  "
                f"{run.h:10.3e}  "
                f"{run.tfpm_l2:12.5e}  {tfpm_l2_ord:>8s}  "
                f"{run.tfpm_h1:12.5e}  {tfpm_h1_ord:>8s}  "
                f"{run.auglag_l2:12.5e}  {aug_l2_ord:>8s}  "
                f"{run.auglag_h1:12.5e}  {aug_h1_ord:>8s}  "
                f"{run.auglag_iter:7d}"
            )

        lines.append("")
        lines.append("Estimated orders from the last pre-plateau log-log window:")
        for key, estimate in report.estimates.items():
            lines.append(
                f"{key}: order = {estimate.order:.4f}, "
                f"window N = {estimate.start_n}..{estimate.end_n}, "
                f"points = {estimate.n_points}, R^2 = {estimate.r2:.6f}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_study_for_scenario(
    scenario: EpsilonScenario,
    *,
    n_values: list[int] | None = None,
) -> list[ScenarioCaseReport]:
    scenario_n_values = list(DEFAULT_N_VALUES if n_values is None else n_values)
    print(f"Running {scenario.name} (eps1={scenario.eps1}, eps2={scenario.eps2})...")

    reports: list[ScenarioCaseReport] = []
    for case in _build_cases():
        print(f"  {case.name}")
        cfg, _c_func, _f_func, _exact = case.build_problem(
            eps1=scenario.eps1,
            eps2=scenario.eps2,
        )
        runs = []
        for n in scenario_n_values:
            print(f"    N = {n}")
            runs.append(
                _run_single_case(
                    case,
                    n_elements=n,
                    eps1=scenario.eps1,
                    eps2=scenario.eps2,
                    build_quad_n=12,
                    particular_quad_segments=10,
                    norm_quad_n=12,
                    derivative_weight_mode="eps_weighted",
                    flux_jump_average="eps_weighted",
                    auglag_rho=100.0,
                    auglag_max_iter=50000,
                    auglag_tol=1e-12,
                )
            )
        estimates = {
            "TFPM-Strong L2": _estimate_rate(runs, "tfpm_l2"),
            "TFPM-Strong EpsNorm": _estimate_rate(runs, "tfpm_h1"),
            "AugLag L2": _estimate_rate(runs, "auglag_l2"),
            "AugLag EpsNorm": _estimate_rate(runs, "auglag_h1"),
        }
        reports.append(
            ScenarioCaseReport(
                scenario=scenario,
                case=case,
                cfg=cfg,
                runs=runs,
                estimates=estimates,
            )
        )

    return reports


def run_equal_study(*, n_values: list[int] | None = None) -> list[ScenarioCaseReport]:
    return run_study_for_scenario(_get_eps_scenario("eps_equal"), n_values=n_values)


def run_study(
    *,
    scenarios: list[EpsilonScenario] | None = None,
    n_values: list[int] | None = None,
) -> list[ScenarioCaseReport]:
    selected_scenarios = list(_build_eps_scenarios() if scenarios is None else scenarios)
    reports: list[ScenarioCaseReport] = []
    for scenario in selected_scenarios:
        reports.extend(run_study_for_scenario(scenario, n_values=n_values))
    return reports


def main() -> None:
    reports = run_study()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    _write_summary(summary_path, reports)
    print(summary_path)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
