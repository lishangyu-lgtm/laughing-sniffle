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

from ...core.methods_tfpm import (
    assemble_lagrange_kkt_system,
    solve_augmented_lagrange,
    solve_penalty_method,
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
    compute_pairwise_errors,
    plot_errors_vs_reference,
    plot_solutions,
    write_summary,
)
from .problem import (
    DEFAULT_EXPERIMENT,
    ExactReference,
    ExperimentConfig,
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
    x_plot_ref: np.ndarray,
    u_plot_ref: np.ndarray,
    method_series: list[tuple[str, np.ndarray, np.ndarray]],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [(reference_label, x_plot_ref, u_plot_ref), *method_series]


def _resolve_penalty_gammas(num) -> tuple[float, ...]:
    raw_gammas = getattr(num, "penalty_gammas", ())
    gamma_candidates = tuple(raw_gammas) if raw_gammas is not None else ()
    if len(gamma_candidates) == 0:
        gamma_candidates = (num.penalty_gamma,)

    gammas: list[float] = []
    for raw_gamma in gamma_candidates:
        gamma = float(raw_gamma)
        if gamma <= 0.0:
            raise ValueError("All penalty gammas must be positive.")
        if gamma not in gammas:
            gammas.append(gamma)
    return tuple(gammas)


def _format_gamma_for_label(gamma: float) -> str:
    return f"{gamma:.1e}"


def _penalty_label(gamma: float, total_penalty_runs: int) -> str:
    if total_penalty_runs == 1:
        return "Penalty"
    return f"Penalty (gamma={_format_gamma_for_label(gamma)})"


def _penalty_timing_key(gamma: float, total_penalty_runs: int) -> str:
    if total_penalty_runs == 1:
        return "penalty_solve"
    return f"penalty_solve(gamma={gamma:.6e})"


def _print_report(
    timings: dict[str, float],
    l2_errors: dict[str, float],
    h1_errors: dict[str, float],
    pairwise_errors: dict[str, float],
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

    print("\n=== Pairwise Relative L2 Error ===")
    for key, value in pairwise_errors.items():
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


def _evaluate_method_on_grid(
    elems,
    z: np.ndarray,
    y_f_func,
    y_left: np.ndarray,
    y_right: np.ndarray,
    y_interface: float,
    n_seg_gauss: int,
    cfg,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    u_left, du_left_transformed = evaluate_tfpm_state_on_side(
        elems,
        z,
        y_f_func,
        y_left,
        side="left",
        xI=y_interface,
        n_seg_gauss=n_seg_gauss,
    )
    u_right, du_right_transformed = evaluate_tfpm_state_on_side(
        elems,
        z,
        y_f_func,
        y_right,
        side="right",
        xI=y_interface,
        n_seg_gauss=n_seg_gauss,
    )
    du_left, du_right = _transformed_to_physical_derivative(
        du_left_transformed,
        du_right_transformed,
        cfg,
    )
    return (u_left, u_right), (du_left, du_right)


def _evaluate_interface_state(
    elems,
    z: np.ndarray,
    y_f_func,
    y_interface: float,
    n_seg_gauss: int,
) -> tuple[float, float, float, float]:
    u_left, du_left = evaluate_tfpm_state_on_side(
        elems,
        z,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="left",
        xI=y_interface,
        n_seg_gauss=n_seg_gauss,
    )
    u_right, du_right = evaluate_tfpm_state_on_side(
        elems,
        z,
        y_f_func,
        np.array([y_interface], dtype=np.float64),
        side="right",
        xI=y_interface,
        n_seg_gauss=n_seg_gauss,
    )
    return float(u_left[0]), float(u_right[0]), float(du_left[0]), float(du_right[0])


def _evaluate_plot_curve(
    elems,
    z: np.ndarray,
    y_f_func,
    cfg,
    plot_per_element: int,
    n_seg_gauss: int,
    u_left_interface: float,
    u_right_interface: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_plot, u_plot = evaluate_solution_fine(
        elems=elems,
        z=z,
        f_func=y_f_func,
        points_per_element=plot_per_element,
        n_seg_gauss=n_seg_gauss,
    )
    x_plot = y_to_x(y_plot, cfg)
    return _split_plot_curve_at_interface(
        x_plot=x_plot,
        u_plot=u_plot,
        x_interface=cfg.x_interface,
        u_left_interface=u_left_interface,
        u_right_interface=u_right_interface,
    )


def main(experiment: ExperimentConfig = DEFAULT_EXPERIMENT) -> None:
    cfg = experiment.problem
    num = experiment.numerical
    out = experiment.output

    if str(cfg.reference_source).strip().lower() != "exact":
        raise ValueError(
            "This quick penalty-vs-auglag experiment currently expects "
            "problem.reference_source = 'exact'."
        )

    n_segments = int(num.n_elements)
    if n_segments % 2 != 0:
        raise ValueError("numerical.n_elements must be even.")
    _validate_x_grid("n_elements", n_segments, cfg)

    x_c_func = make_coefficient(cfg)
    x_f_func = make_rhs(cfg)
    y_c_func = make_transformed_coefficient(cfg, x_c_func)
    y_f_func = make_transformed_rhs(cfg, x_f_func)
    y_c_left_trace, y_c_right_trace = make_transformed_coefficient_trace_functions(cfg, x_c_func)
    flux_jump_average = normalize_flux_jump_average(num.flux_jump_average)
    flux_left_weight, flux_right_weight = flux_jump_average_weights(cfg, flux_jump_average)

    y_left_domain, y_right_domain = transformed_domain(cfg)
    y_interface = transformed_interface(cfg)
    y_grid = _build_transformed_grid_from_x_count(num.n_elements, cfg)
    n_left_tfpm, n_right_tfpm = _grid_split_counts(num.n_elements, cfg)

    out_dir = _resolve_output_dir(out.output_dir)
    timings: dict[str, float] = {}
    penalty_gammas = _resolve_penalty_gammas(num)

    t0 = time.perf_counter()
    grid, elems = build_elements(
        c_func=y_c_func,
        f_func=y_f_func,
        grid=y_grid,
        quad_n=num.quad_n,
        xI=y_interface,
        n_seg_gauss=num.n_seg_gauss,
        c_left_trace=y_c_left_trace,
        c_right_trace=y_c_right_trace,
    )
    timings["build_elements"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    _K_aug, _rhs_aug, H_aug, l_aug, C_aug, d_aug = assemble_lagrange_kkt_system(
        grid=grid,
        elems=elems,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        jump_du=cfg.jump_flux,
        xI=y_interface,
        c_func=y_c_func,
        use_true_c=num.use_true_c_auglag,
        flux_jump_weights=(flux_left_weight, flux_right_weight),
    )
    timings["auglag_system_assemble"] = time.perf_counter() - t1

    t1b = time.perf_counter()
    _K_pen, _rhs_pen, H_pen, l_pen, C_pen, d_pen = assemble_lagrange_kkt_system(
        grid=grid,
        elems=elems,
        f_func=y_f_func,
        m=cfg.left_bc,
        n_dir=cfg.right_bc,
        p=cfg.jump_u,
        jump_du=cfg.jump_flux,
        xI=y_interface,
        c_func=y_c_func,
        use_true_c=num.use_true_c_penalty,
        flux_jump_weights=(flux_left_weight, flux_right_weight),
    )
    timings["penalty_system_assemble"] = time.perf_counter() - t1b

    t2 = time.perf_counter()
    z_auglag, lam_auglag, aug_history = solve_augmented_lagrange(
        H=H_aug,
        l=l_aug,
        C=C_aug,
        d=d_aug,
        rho=num.auglag_rho,
        max_iter=num.auglag_max_iter,
        tol_primal=num.auglag_tol_primal,
        tol_stationarity=num.auglag_tol_stationarity,
        relax=num.auglag_relax,
        verbose=num.auglag_verbose,
    )
    timings["auglag_solve"] = time.perf_counter() - t2

    penalty_runs: list[dict[str, object]] = []
    for gamma in penalty_gammas:
        t3 = time.perf_counter()
        z_penalty, lam_penalty, penalty_history = solve_penalty_method(
            H=H_pen,
            l=l_pen,
            C=C_pen,
            d=d_pen,
            gamma=gamma,
            residual_tol=num.penalty_residual_tol,
        )
        timings[_penalty_timing_key(gamma, len(penalty_gammas))] = time.perf_counter() - t3
        penalty_runs.append(
            {
                "gamma": gamma,
                "label": _penalty_label(gamma, len(penalty_gammas)),
                "z": z_penalty,
                "lam": lam_penalty,
                "history": penalty_history,
            }
        )

    exact_reference = make_exact_reference(cfg)
    if exact_reference is None:
        raise ValueError("make_exact_reference(...) returned None for the penalty-vs-auglag experiment.")
    reference_label = exact_reference.label

    x_left = np.linspace(X_DOMAIN[0], cfg.x_interface, num.error_samples)
    x_right = np.linspace(cfg.x_interface, X_DOMAIN[1], num.error_samples)
    y_left = x_to_y(x_left, cfg)
    y_right = x_to_y(x_right, cfg)

    (aug_left, aug_right), (aug_du_left, aug_du_right) = _evaluate_method_on_grid(
        elems=elems,
        z=z_auglag,
        y_f_func=y_f_func,
        y_left=y_left,
        y_right=y_right,
        y_interface=y_interface,
        n_seg_gauss=num.n_seg_gauss,
        cfg=cfg,
    )

    t_ref = time.perf_counter()
    ref_left = exact_reference.evaluate(x_left, side="left")
    ref_right = exact_reference.evaluate(x_right, side="right")
    ref_du_left = exact_reference.evaluate_derivative(x_left, side="left")
    ref_du_right = exact_reference.evaluate_derivative(x_right, side="right")
    timings["exact_reference_eval_total"] = time.perf_counter() - t_ref

    reference_values = (ref_left, ref_right)
    method_values = {
        "AugLag": (aug_left, aug_right),
    }
    method_derivatives = {
        "AugLag": (aug_du_left, aug_du_right),
    }
    for penalty_run in penalty_runs:
        label = str(penalty_run["label"])
        penalty_values, penalty_derivatives = _evaluate_method_on_grid(
            elems=elems,
            z=np.asarray(penalty_run["z"]),
            y_f_func=y_f_func,
            y_left=y_left,
            y_right=y_right,
            y_interface=y_interface,
            n_seg_gauss=num.n_seg_gauss,
            cfg=cfg,
        )
        method_values[label] = penalty_values
        method_derivatives[label] = penalty_derivatives
        penalty_run["values"] = penalty_values
        penalty_run["derivatives"] = penalty_derivatives
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
        method_values=method_values,
        method_derivatives=method_derivatives,
        reference_label=reference_label,
    )
    pairwise_errors = compute_pairwise_errors(
        x_left=x_left,
        x_right=x_right,
        common_values=method_values,
    )

    aug_uL_I, aug_uR_I, aug_duL_I, aug_duR_I = _evaluate_interface_state(
        elems=elems,
        z=z_auglag,
        y_f_func=y_f_func,
        y_interface=y_interface,
        n_seg_gauss=num.n_seg_gauss,
    )
    for penalty_run in penalty_runs:
        pen_uL_I, pen_uR_I, pen_duL_I, pen_duR_I = _evaluate_interface_state(
            elems=elems,
            z=np.asarray(penalty_run["z"]),
            y_f_func=y_f_func,
            y_interface=y_interface,
            n_seg_gauss=num.n_seg_gauss,
        )
        penalty_run["u_left_interface"] = pen_uL_I
        penalty_run["u_right_interface"] = pen_uR_I
        penalty_run["du_left_interface"] = pen_duL_I
        penalty_run["du_right_interface"] = pen_duR_I

    if out.save_plots:
        x_plot_ref, u_plot_ref = _exact_plot_arrays(exact_reference, cfg, num.error_samples)
        x_plot_aug, u_plot_aug = _evaluate_plot_curve(
            elems=elems,
            z=z_auglag,
            y_f_func=y_f_func,
            cfg=cfg,
            plot_per_element=num.plot_per_element,
            n_seg_gauss=num.n_seg_gauss,
            u_left_interface=aug_uL_I,
            u_right_interface=aug_uR_I,
        )
        method_plot_series = [("AugLag", x_plot_aug, u_plot_aug)]
        for penalty_run in penalty_runs:
            x_plot_pen, u_plot_pen = _evaluate_plot_curve(
                elems=elems,
                z=np.asarray(penalty_run["z"]),
                y_f_func=y_f_func,
                cfg=cfg,
                plot_per_element=num.plot_per_element,
                n_seg_gauss=num.n_seg_gauss,
                u_left_interface=float(penalty_run["u_left_interface"]),
                u_right_interface=float(penalty_run["u_right_interface"]),
            )
            method_plot_series.append((str(penalty_run["label"]), x_plot_pen, u_plot_pen))

        plot_solutions(
            _plot_series(
                reference_label,
                x_plot_ref,
                u_plot_ref,
                method_plot_series,
            ),
            xI=cfg.x_interface,
            save_path=out_dir / "solutions_compare.png",
            reference_label=reference_label,
        )
        plot_errors_vs_reference(
            x_left=x_left,
            x_right=x_right,
            reference_values=reference_values,
            method_values=method_values,
            xI=cfg.x_interface,
            save_path=out_dir / "errors_vs_reference.png",
            reference_label=reference_label,
        )

    aug_u_jump = aug_uR_I - aug_uL_I
    aug_flux_jump = aug_duR_I - aug_duL_I

    diagnostics = [
        f"Reference source = {reference_label}",
        f"Transformed interval = [{y_left_domain:.6e}, {y_right_domain:.6e}]",
        f"Transformed interface yI = {y_interface:.6e}",
        f"TFPM grid split: left elements = {n_left_tfpm}, right elements = {n_right_tfpm}",
        (
            "Flux-jump average = "
            f"{flux_jump_average} (left={flux_left_weight:.6e}, right={flux_right_weight:.6e})"
        ),
        f"AugLag rho = {num.auglag_rho:.6e}",
        f"AugLag final iter = {aug_history.iter}",
        f"AugLag final primal residual = {aug_history.primal_inf:.6e}",
        f"AugLag final stationarity residual = {aug_history.stationarity_inf:.6e}",
        f"AugLag approximate lambda inf = {float(np.max(np.abs(lam_auglag))):.6e}",
        f"AugLag interface [u] residual = {abs(aug_u_jump - cfg.jump_u):.6e}",
        f"AugLag interface [eps u'] residual = {abs(aug_flux_jump - cfg.jump_flux):.6e}",
    ]
    if len(penalty_gammas) > 1:
        diagnostics.append(
            "Penalty gamma sweep = "
            + ", ".join(f"{gamma:.6e}" for gamma in penalty_gammas)
        )
    for penalty_run in penalty_runs:
        penalty_history = penalty_run["history"]
        lam_penalty = np.asarray(penalty_run["lam"])
        pen_u_jump = float(penalty_run["u_right_interface"]) - float(penalty_run["u_left_interface"])
        pen_flux_jump = float(penalty_run["du_right_interface"]) - float(penalty_run["du_left_interface"])
        label = str(penalty_run["label"])
        diagnostics.extend(
            [
                f"{label} gamma = {float(penalty_run['gamma']):.6e}",
                f"{label} primal residual = {penalty_history.primal_inf:.6e}",
                f"{label} stationarity residual = {penalty_history.stationarity_inf:.6e}",
                f"{label} approximate lambda inf = {float(np.max(np.abs(lam_penalty))):.6e}",
                f"{label} interface [u] residual = {abs(pen_u_jump - cfg.jump_u):.6e}",
                f"{label} interface [eps u'] residual = {abs(pen_flux_jump - cfg.jump_flux):.6e}",
            ]
        )

    write_summary(
        out_dir / "summary.txt",
        timings=timings,
        errors=errors,
        extra_lines=diagnostics,
        errors_title=f"Relative L2 errors vs {reference_label}:",
        additional_sections=[
            (f"Relative H^1 errors vs {reference_label}:", h1_errors),
            ("Pairwise relative L2 errors:", pairwise_errors),
        ],
    )
    _print_report(
        timings=timings,
        l2_errors=errors,
        h1_errors=h1_errors,
        pairwise_errors=pairwise_errors,
        diagnostics=diagnostics,
        out_dir=out_dir,
        reference_label=reference_label,
        save_plots=out.save_plots,
    )


if __name__ == "__main__":
    main()
