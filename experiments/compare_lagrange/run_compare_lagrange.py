from __future__ import annotations

import sys
import time
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

from ...baselines.fdm_reference import (
    evaluate_fdm_derivative_on_side,
    evaluate_fdm_on_side,
    fdm_plot_arrays,
    solve_interface_fdm,
)
from ...baselines.fem_solver import (
    assemble_interface_fem_system,
    evaluate_fem_derivative_on_side,
    evaluate_fem_on_side,
    fem_plot_arrays,
    solve_interface_fem_from_system,
)
from ...core.methods_tfpm import (
    assemble_lagrange_kkt_system,
    solve_lagrange_kkt,
    solve_tfpm_strong_matching,
)
from ...core.tfpm_local import (
    build_elements,
    build_uniform_grid,
    evaluate_solution_fine,
    evaluate_tfpm_state_on_side,
)
from .analysis_plot import (
    compute_h1_errors_vs_reference,
    compute_errors_vs_reference,
    plot_errors_vs_reference,
    plot_solutions,
    write_summary,
)
from .problem import (
    COEFFICIENT_LEFT_DESCRIPTION,
    COEFFICIENT_RIGHT_DESCRIPTION,
    DEFAULT_EXPERIMENT,
    ExactReference,
    ExperimentConfig,
    RHS_LEFT_DESCRIPTION,
    RHS_RIGHT_DESCRIPTION,
    X_DOMAIN,
    flux_jump_average_weights,
    make_coefficient,
    make_exact_reference,
    make_rhs,
    make_transformed_coefficient,
    make_transformed_coefficient_trace_functions,
    make_transformed_rhs,
    normalize_flux_jump_average,
    transformed_domain,
    transformed_interface,
    x_to_y,
    y_to_x,
)


def _validate_x_grid(name: str, n_segments: int, cfg) -> None:
    try:
        build_uniform_grid(N=n_segments, a=X_DOMAIN[0], b=X_DOMAIN[1], xI=cfg.x_interface)
    except ValueError as exc:
        raise ValueError(
            f"numerical.{name}={n_segments} is incompatible with the physical x-grid on "
            f"[{X_DOMAIN[0]:.16g}, {X_DOMAIN[1]:.16g}] and interface xI={cfg.x_interface:.16g}. "
            "Adjust the total grid count so the physical interface lies on a grid node."
        ) from exc


def _build_transformed_grid_from_x_count(n_segments: int, cfg) -> np.ndarray:
    x_grid, _hx, _interface_node = build_uniform_grid(
        N=n_segments,
        a=X_DOMAIN[0],
        b=X_DOMAIN[1],
        xI=cfg.x_interface,
    )
    return x_to_y(x_grid, cfg)


def _grid_split_counts(n_segments: int, cfg) -> tuple[int, int]:
    _grid, _h, interface_node = build_uniform_grid(
        N=n_segments,
        a=X_DOMAIN[0],
        b=X_DOMAIN[1],
        xI=cfg.x_interface,
    )
    return interface_node, n_segments - interface_node


def _resolve_output_dir(output_dir: Path) -> Path:
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _normalize_reference_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fdm": "fdm",
        "fdm_fine": "fdm",
        "fine_fdm": "fdm",
        "exact": "exact",
        "analytic": "exact",
        "analytical": "exact",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported problem.reference_source={source!r}. Supported values: {supported}."
        ) from exc


def _exact_plot_arrays(
    reference: ExactReference,
    cfg,
    samples_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = max(int(samples_per_side), 2)
    x_left = np.linspace(X_DOMAIN[0], cfg.x_interface, n_samples)
    x_right = np.linspace(cfg.x_interface, X_DOMAIN[1], n_samples)
    u_left = reference.evaluate(x_left, side="left")
    u_right = reference.evaluate(x_right, side="right")
    x_plot = np.concatenate([x_left, np.array([np.nan]), x_right])
    u_plot = np.concatenate([u_left, np.array([np.nan]), u_right])
    return x_plot, u_plot


def _split_plot_curve_at_interface(
    x_plot: np.ndarray,
    u_plot: np.ndarray,
    x_interface: float,
    u_left_interface: float,
    u_right_interface: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_plot = np.asarray(x_plot, dtype=np.float64)
    u_plot = np.asarray(u_plot, dtype=np.float64)
    tol = max(1e-12, 1e-10 * max(1.0, abs(x_interface)))

    finite_mask = np.isfinite(x_plot) & np.isfinite(u_plot)
    left_mask = finite_mask & (x_plot < x_interface - tol)
    right_mask = finite_mask & (x_plot > x_interface + tol)

    x_left = np.concatenate([x_plot[left_mask], np.array([x_interface], dtype=np.float64)])
    u_left = np.concatenate([u_plot[left_mask], np.array([u_left_interface], dtype=np.float64)])
    x_right = np.concatenate([np.array([x_interface], dtype=np.float64), x_plot[right_mask]])
    u_right = np.concatenate([np.array([u_right_interface], dtype=np.float64), u_plot[right_mask]])

    x_piecewise = np.concatenate([x_left, np.array([np.nan]), x_right])
    u_piecewise = np.concatenate([u_left, np.array([np.nan]), u_right])
    return x_piecewise, u_piecewise


def _plot_series(
    reference_label: str,
    x_plot_lagrange: np.ndarray,
    u_plot_lagrange: np.ndarray,
    x_plot_ref: np.ndarray,
    u_plot_ref: np.ndarray,
    x_plot_fdm: np.ndarray,
    u_plot_fdm: np.ndarray,
    x_plot_fem: np.ndarray,
    u_plot_fem: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        (reference_label, x_plot_ref, u_plot_ref),
        ("Lagrange", x_plot_lagrange, u_plot_lagrange),
        ("FDM", x_plot_fdm, u_plot_fdm),
        ("FEM", x_plot_fem, u_plot_fem),
    ]


def _print_report(
    timings: dict[str, float],
    l2_errors: dict[str, float],
    h1_errors: dict[str, float],
    diagnostics: list[str],
    out_dir: Path,
    reference_label: str,
    save_plots: bool,
) -> None:
    print("=== Timing (s) ===")
    for key, value in timings.items():
        print(f"{key:>28s}: {value:.6f}")

    print(f"\n=== Relative L2 Error vs {reference_label} ===")
    for key, value in l2_errors.items():
        print(f"{key:>28s}: {value:.6e}")

    print(f"\n=== Relative H^1 Error vs {reference_label} ===")
    for key, value in h1_errors.items():
        print(f"{key:>28s}: {value:.6e}")

    print("\n=== Diagnostics ===")
    for line in diagnostics:
        print(line)

    print("\nSaved files:")
    print(out_dir / "summary.txt")
    if save_plots:
        print(out_dir / "solutions_compare.png")
        print(out_dir / "errors_vs_reference.png")


def _transformed_to_physical_derivative(
    du_left_transformed: np.ndarray,
    du_right_transformed: np.ndarray,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    return du_left_transformed / cfg.eps1, du_right_transformed / cfg.eps2


def _summary_config_sections(cfg, num, out_dir: Path, save_plots: bool) -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "Problem settings:",
            {
                "eps1": cfg.eps1,
                "eps2": cfg.eps2,
                "x_interface": cfg.x_interface,
                "left_bc": cfg.left_bc,
                "right_bc": cfg.right_bc,
                "jump_u": cfg.jump_u,
                "jump_flux": cfg.jump_flux,
                "reference_source": cfg.reference_source,
            },
        ),
        (
            "Equation settings:",
            {
                "coefficient_left": COEFFICIENT_LEFT_DESCRIPTION,
                "coefficient_right": COEFFICIENT_RIGHT_DESCRIPTION,
                "rhs_left": RHS_LEFT_DESCRIPTION,
                "rhs_right": RHS_RIGHT_DESCRIPTION,
            },
        ),
        (
            "Numerical settings:",
            {
                "n_elements": num.n_elements,
                "n_ref": num.n_ref,
                "n_fdm": num.n_fdm,
                "n_fem": num.n_fem,
                "quad_n": num.quad_n,
                "n_seg_gauss": num.n_seg_gauss,
                "plot_per_element": num.plot_per_element,
                "fdm_plot_cells": num.fdm_plot_cells,
                "fem_plot_cells": num.fem_plot_cells,
                "error_samples": num.error_samples,
                "use_true_c_lagrange": num.use_true_c_lagrange,
                "lagrange_residual_tol": num.lagrange_residual_tol,
                "flux_jump_average": num.flux_jump_average,
            },
        ),
        (
            "Output settings:",
            {
                "output_dir": out_dir,
                "save_plots": save_plots,
            },
        ),
    ]


def main(experiment: ExperimentConfig = DEFAULT_EXPERIMENT) -> None:
    cfg = experiment.problem
    num = experiment.numerical
    out = experiment.output
    reference_source = _normalize_reference_source(cfg.reference_source)

    grid_attrs = ("n_elements", "n_fdm", "n_fem")
    if reference_source == "fdm":
        grid_attrs = ("n_elements", "n_ref", "n_fdm", "n_fem")
    for attr in grid_attrs:
        n_segments = int(getattr(num, attr))
        if n_segments % 2 != 0:
            raise ValueError(f"numerical.{attr} must be even.")
        _validate_x_grid(attr, n_segments, cfg)

    x_c_func = make_coefficient(cfg)
    x_f_func = make_rhs(cfg)
    y_c_func = make_transformed_coefficient(cfg, x_c_func)
    y_f_func = make_transformed_rhs(cfg, x_f_func)
    y_c_left_trace, y_c_right_trace = make_transformed_coefficient_trace_functions(cfg, x_c_func)
    flux_jump_average = normalize_flux_jump_average(num.flux_jump_average)
    flux_left_weight, flux_right_weight = flux_jump_average_weights(cfg, flux_jump_average)

    y_left_domain, y_right_domain = transformed_domain(cfg)
    y_interface = transformed_interface(cfg)

    y_grid_tfpm = _build_transformed_grid_from_x_count(num.n_elements, cfg)
    y_grid_fdm = _build_transformed_grid_from_x_count(num.n_fdm, cfg)
    y_grid_fem = _build_transformed_grid_from_x_count(num.n_fem, cfg)
    n_left_tfpm, n_right_tfpm = _grid_split_counts(num.n_elements, cfg)
    y_grid_ref = None
    if reference_source == "fdm":
        y_grid_ref = _build_transformed_grid_from_x_count(num.n_ref, cfg)

    out_dir = _resolve_output_dir(out.output_dir)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    grid, elems = build_elements(
        c_func=y_c_func,
        f_func=y_f_func,
        grid=y_grid_tfpm,
        quad_n=num.quad_n,
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
        c_left_trace=y_c_left_trace,
        c_right_trace=y_c_right_trace,
    )
    timings["build_elements"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    z_tfpm = solve_tfpm_strong_matching(
        grid=grid,
        elems=elems,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        qjump=cfg.jump_flux,
        xI=y_interface,
    )
    timings["tfpm_strong_solve"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    K, rhs, H, l, C, d = assemble_lagrange_kkt_system(
        grid=grid,
        elems=elems,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        jump_du=cfg.jump_flux,
        xI=y_interface,
        c_func=y_c_func,
        use_true_c=num.use_true_c_lagrange,
        flux_jump_weights=(flux_left_weight, flux_right_weight),
    )
    timings["lagrange_assemble"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    z_lagrange, lam, lagrange_history = solve_lagrange_kkt(
        K=K,
        rhs=rhs,
        H=H,
        l=l,
        C=C,
        d=d,
        residual_tol=num.lagrange_residual_tol,
    )
    timings["lagrange_solve"] = time.perf_counter() - t3

    sol_fdm_ref = None
    exact_reference = None
    reference_label = "Reference"
    if reference_source == "fdm":
        t4 = time.perf_counter()
        sol_fdm_ref = solve_interface_fdm(
            N=None,
            c_func=y_c_func,
            f_func=y_f_func,
            m=cfg.left_bc,
            n_dir=cfg.right_bc,
            p=cfg.jump_u,
            qjump=cfg.jump_flux,
            grid=y_grid_ref,
            xI=y_interface,
        )
        timings["fdm_reference_solve_total"] = time.perf_counter() - t4
        reference_label = "FDM fine"
    else:
        exact_reference = make_exact_reference(cfg)
        if exact_reference is None:
            raise ValueError(
                "problem.reference_source requests an exact reference, but "
                "make_exact_reference(...) returned None."
            )
        reference_label = exact_reference.label

    t5 = time.perf_counter()
    sol_fdm = solve_interface_fdm(
        N=None,
        c_func=y_c_func,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        qjump=cfg.jump_flux,
        grid=y_grid_fdm,
        xI=y_interface,
    )
    timings["fdm_solve_total"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    fem_system = assemble_interface_fem_system(
        N=None,
        c_func=y_c_func,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        qjump=cfg.jump_flux,
        grid=y_grid_fem,
        xI=y_interface,
    )
    timings["fem_assemble"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    sol_fem = solve_interface_fem_from_system(fem_system)
    timings["fem_solve"] = time.perf_counter() - t7

    x_left = np.linspace(X_DOMAIN[0], cfg.x_interface, num.error_samples)
    x_right = np.linspace(cfg.x_interface, X_DOMAIN[1], num.error_samples)
    y_left = x_to_y(x_left, cfg)
    y_right = x_to_y(x_right, cfg)

    tfpm_left, tfpm_du_left_transformed = evaluate_tfpm_state_on_side(
        elems,
        z_tfpm,
        y_f_func,
        y_left,
        side="left",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    tfpm_right, tfpm_du_right_transformed = evaluate_tfpm_state_on_side(
        elems,
        z_tfpm,
        y_f_func,
        y_right,
        side="right",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    lagrange_left, lagrange_du_left_transformed = evaluate_tfpm_state_on_side(
        elems,
        z_lagrange,
        y_f_func,
        y_left,
        side="left",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    lagrange_right, lagrange_du_right_transformed = evaluate_tfpm_state_on_side(
        elems,
        z_lagrange,
        y_f_func,
        y_right,
        side="right",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    if sol_fdm_ref is not None:
        ref_left = evaluate_fdm_on_side(sol_fdm_ref, y_left, side="left")
        ref_right = evaluate_fdm_on_side(sol_fdm_ref, y_right, side="right")
        ref_du_left_transformed = evaluate_fdm_derivative_on_side(sol_fdm_ref, y_left, side="left")
        ref_du_right_transformed = evaluate_fdm_derivative_on_side(sol_fdm_ref, y_right, side="right")
        ref_du_left, ref_du_right = _transformed_to_physical_derivative(
            ref_du_left_transformed,
            ref_du_right_transformed,
            cfg,
        )
    else:
        t_ref = time.perf_counter()
        ref_left = exact_reference.evaluate(x_left, side="left")
        ref_right = exact_reference.evaluate(x_right, side="right")
        ref_du_left = exact_reference.evaluate_derivative(x_left, side="left")
        ref_du_right = exact_reference.evaluate_derivative(x_right, side="right")
        timings["exact_reference_eval_total"] = time.perf_counter() - t_ref
    fdm_left = evaluate_fdm_on_side(sol_fdm, y_left, side="left")
    fdm_right = evaluate_fdm_on_side(sol_fdm, y_right, side="right")
    fdm_du_left_transformed = evaluate_fdm_derivative_on_side(sol_fdm, y_left, side="left")
    fdm_du_right_transformed = evaluate_fdm_derivative_on_side(sol_fdm, y_right, side="right")
    fem_left = evaluate_fem_on_side(sol_fem, y_left, side="left")
    fem_right = evaluate_fem_on_side(sol_fem, y_right, side="right")
    fem_du_left_transformed = evaluate_fem_derivative_on_side(sol_fem, y_left, side="left")
    fem_du_right_transformed = evaluate_fem_derivative_on_side(sol_fem, y_right, side="right")
    fdm_du_left, fdm_du_right = _transformed_to_physical_derivative(
        fdm_du_left_transformed,
        fdm_du_right_transformed,
        cfg,
    )
    tfpm_du_left, tfpm_du_right = _transformed_to_physical_derivative(
        tfpm_du_left_transformed,
        tfpm_du_right_transformed,
        cfg,
    )
    lagrange_du_left, lagrange_du_right = _transformed_to_physical_derivative(
        lagrange_du_left_transformed,
        lagrange_du_right_transformed,
        cfg,
    )
    fem_du_left, fem_du_right = _transformed_to_physical_derivative(
        fem_du_left_transformed,
        fem_du_right_transformed,
        cfg,
    )

    reference_values = (ref_left, ref_right)
    method_values = {
        "TFPM-Strong": (tfpm_left, tfpm_right),
        "Lagrange": (lagrange_left, lagrange_right),
        "FDM": (fdm_left, fdm_right),
        "FEM": (fem_left, fem_right),
    }
    errors = compute_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        method_values=method_values,
        reference_label=reference_label,
    )
    h1_errors = compute_h1_errors_vs_reference(
        x_left=x_left,
        x_right=x_right,
        reference_values=reference_values,
        reference_derivatives=(ref_du_left, ref_du_right),
        method_values={
            "TFPM-Strong": (tfpm_left, tfpm_right),
            "Lagrange": (lagrange_left, lagrange_right),
            "FDM": (fdm_left, fdm_right),
            "FEM": (fem_left, fem_right),
        },
        method_derivatives={
            "TFPM-Strong": (tfpm_du_left, tfpm_du_right),
            "Lagrange": (lagrange_du_left, lagrange_du_right),
            "FDM": (fdm_du_left, fdm_du_right),
            "FEM": (fem_du_left, fem_du_right),
        },
        reference_label=reference_label,
    )

    tfpm_uL_I, tfpm_duL_I = evaluate_tfpm_state_on_side(
        elems,
        z_tfpm,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="left",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    tfpm_uR_I, tfpm_duR_I = evaluate_tfpm_state_on_side(
        elems,
        z_tfpm,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="right",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    lagrange_uL_I, lagrange_duL_I = evaluate_tfpm_state_on_side(
        elems,
        z_lagrange,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="left",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    lagrange_uR_I, lagrange_duR_I = evaluate_tfpm_state_on_side(
        elems,
        z_lagrange,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="right",
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )

    if out.save_plots:
        y_plot_lagrange, u_plot_lagrange = evaluate_solution_fine(
            elems=elems,
            z=z_lagrange,
            f_func=y_f_func,
            points_per_element=num.plot_per_element,
            n_seg_gauss=num.n_seg_gauss,
        )
        y_plot_fdm, u_plot_fdm = fdm_plot_arrays(sol_fdm, points_per_cell=num.fdm_plot_cells)
        y_plot_fem, u_plot_fem = fem_plot_arrays(sol_fem, points_per_cell=num.fem_plot_cells)
        if sol_fdm_ref is not None:
            y_plot_ref, u_plot_ref = fdm_plot_arrays(sol_fdm_ref, points_per_cell=num.fdm_plot_cells)
            x_plot_ref = y_to_x(y_plot_ref, cfg)
        else:
            x_plot_ref, u_plot_ref = _exact_plot_arrays(exact_reference, cfg, num.error_samples)
        x_plot_lagrange = y_to_x(y_plot_lagrange, cfg)
        x_plot_lagrange, u_plot_lagrange = _split_plot_curve_at_interface(
            x_plot=x_plot_lagrange,
            u_plot=u_plot_lagrange,
            x_interface=cfg.x_interface,
            u_left_interface=float(lagrange_uL_I[0]),
            u_right_interface=float(lagrange_uR_I[0]),
        )
        x_plot_fdm = y_to_x(y_plot_fdm, cfg)
        x_plot_fem = y_to_x(y_plot_fem, cfg)

        plot_solutions(
            _plot_series(
                reference_label,
                x_plot_lagrange,
                u_plot_lagrange,
                x_plot_ref,
                u_plot_ref,
                x_plot_fdm,
                u_plot_fdm,
                x_plot_fem,
                u_plot_fem,
            ),
            xI=cfg.x_interface,
            save_path=out_dir / "solutions_compare.png",
            reference_label=reference_label,
        )
        plot_errors_vs_reference(
            x_left=x_left,
            x_right=x_right,
            reference_values=reference_values,
            method_values={name: values for name, values in method_values.items() if name != "TFPM-Strong"},
            xI=cfg.x_interface,
            save_path=out_dir / "errors_vs_reference.png",
            reference_label=reference_label,
        )

    tfpm_u_jump = float(tfpm_uR_I[0] - tfpm_uL_I[0])
    tfpm_flux_jump = float(tfpm_duR_I[0] - tfpm_duL_I[0])
    lagrange_u_jump = float(lagrange_uR_I[0] - lagrange_uL_I[0])
    lagrange_flux_jump = float(lagrange_duR_I[0] - lagrange_duL_I[0])

    diagnostics = [
        f"Reference source = {reference_label}",
        f"Transformed interval = [{y_left_domain:.6e}, {y_right_domain:.6e}]",
        f"Transformed interface yI = {y_interface:.6e}",
        f"TFPM grid split: left elements = {n_left_tfpm}, right elements = {n_right_tfpm}",
        (
            "Lagrange flux-jump average = "
            f"{flux_jump_average} (left={flux_left_weight:.6e}, right={flux_right_weight:.6e})"
        ),
        f"Lagrange KKT residual = {lagrange_history.residual_inf:.6e}",
        f"Lagrange final primal residual = {lagrange_history.primal_inf:.6e}",
        f"Lagrange final stationarity residual = {lagrange_history.stationarity_inf:.6e}",
        f"TFPM interface [u] residual = {abs(tfpm_u_jump - cfg.jump_u):.6e}",
        f"TFPM interface [eps u'] residual = {abs(tfpm_flux_jump - cfg.jump_flux):.6e}",
        f"Lagrange interface [u] residual = {abs(lagrange_u_jump - cfg.jump_u):.6e}",
        f"Lagrange interface [eps u'] residual = {abs(lagrange_flux_jump - cfg.jump_flux):.6e}",
    ]

    write_summary(
        out_dir / "summary.txt",
        timings=timings,
        errors=errors,
        extra_lines=diagnostics,
        errors_title=f"Relative L2 errors vs {reference_label}:",
        additional_sections=[(f"Relative H^1 errors vs {reference_label}:", h1_errors)],
        config_sections=_summary_config_sections(cfg, num, out_dir=out_dir, save_plots=out.save_plots),
    )
    _print_report(
        timings=timings,
        l2_errors=errors,
        h1_errors=h1_errors,
        diagnostics=diagnostics,
        out_dir=out_dir,
        reference_label=reference_label,
        save_plots=out.save_plots,
    )


if __name__ == "__main__":
    main()
