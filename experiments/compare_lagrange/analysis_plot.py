from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_METHOD_PLOT_STYLES: dict[str, dict[str, object]] = {
    "__reference__": {
        "color": "0.45",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 1,
    },
    "Lagrange": {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "o",
        "zorder": 2,
    },
    "FDM": {
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "s",
        "zorder": 2,
    },
    "FEM": {
        "color": "tab:green",
        "linestyle": "-.",
        "linewidth": 2.0,
        "marker": "^",
        "zorder": 2,
    },
    "TFPM-Strong": {
        "color": "tab:red",
        "linestyle": ":",
        "linewidth": 2.0,
        "marker": "D",
        "zorder": 2,
    },
    "__default__": {
        "color": "0.35",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 1,
    },
}


def _resolve_style_key(name: str, reference_label: str | None = None) -> str:
    if reference_label is not None and name == reference_label:
        return "__reference__"
    if name in _METHOD_PLOT_STYLES:
        return name
    if "strong" in name.lower():
        return "TFPM-Strong"
    return "__default__"


def _line_style(name: str, reference_label: str | None = None, n_points: int | None = None) -> dict[str, object]:
    style = dict(_METHOD_PLOT_STYLES[_resolve_style_key(name, reference_label)])
    marker = style.get("marker")
    if marker is not None and n_points is not None and n_points > 0:
        style["markevery"] = max(n_points // 14, 2)
        style["markersize"] = 4.0
        style["markerfacecolor"] = "white"
        style["markeredgewidth"] = 1.0
    return style


def _x_bounds(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    finite_parts = [arr[np.isfinite(arr)] for arr in arrays if np.any(np.isfinite(arr))]
    if not finite_parts:
        return 0.0, 1.0
    all_x = np.concatenate(finite_parts)
    return float(np.min(all_x)), float(np.max(all_x))


def _window_y_limits(
    curves: Iterable[tuple[np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
) -> tuple[float, float] | None:
    xmin, xmax = xlim
    visible_values = []
    for x, y in curves:
        mask = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
        if np.any(mask):
            visible_values.append(y[mask])
    if not visible_values:
        return None

    y_values = np.concatenate(visible_values)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1e-6)
    else:
        pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _zoom_windows(x_min: float, x_max: float, xI: float) -> list[tuple[str, tuple[float, float]]]:
    span = max(x_max - x_min, 1e-12)
    interface_half_width = 0.01 * span
    right_width = 0.01 * span
    return [
        ("Global", (x_min, x_max)),
        (
            f"Interface (x~{xI:.3g})",
            (max(x_min, xI - interface_half_width), min(x_max, xI + interface_half_width)),
        ),
        (
            f"Right End (x~{x_max:.3g})",
            (max(x_min, x_max - right_width), x_max),
        ),
    ]


def _setup_axis(
    ax: plt.Axes,
    curves: list[tuple[np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    xI: float,
    title: str,
    y_label: str | None,
) -> None:
    ax.set_xlim(*xlim)
    ylim = _window_y_limits(curves, xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim[0] <= xI <= xlim[1]:
        ax.axvline(xI, linestyle="--", linewidth=1.1, color="0.35", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def symmetric_relative_l2(
    x_left: np.ndarray,
    x_right: np.ndarray,
    u1_left: np.ndarray,
    u1_right: np.ndarray,
    u2_left: np.ndarray,
    u2_right: np.ndarray,
) -> float:
    num = np.trapezoid((u1_left - u2_left) ** 2, x_left) + np.trapezoid((u1_right - u2_right) ** 2, x_right)
    den1 = np.trapezoid(u1_left**2, x_left) + np.trapezoid(u1_right**2, x_right)
    den2 = np.trapezoid(u2_left**2, x_left) + np.trapezoid(u2_right**2, x_right)
    den = max(den1, den2, 1e-30)
    return float(np.sqrt(num / den))


def compute_pairwise_errors(
    x_left: np.ndarray,
    x_right: np.ndarray,
    common_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for a, b in combinations(common_values.keys(), 2):
        aL, aR = common_values[a]
        bL, bR = common_values[b]
        err = symmetric_relative_l2(x_left, x_right, aL, aR, bL, bR)
        errors[f"{a} vs {b}"] = err
    return errors


def relative_l2_vs_reference(
    x_left: np.ndarray,
    x_right: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray,
    ref_left: np.ndarray,
    ref_right: np.ndarray,
) -> float:
    num = np.trapezoid((u_left - ref_left) ** 2, x_left) + np.trapezoid((u_right - ref_right) ** 2, x_right)
    den = np.trapezoid(ref_left**2, x_left) + np.trapezoid(ref_right**2, x_right)
    den = max(float(den), 1e-30)
    return float(np.sqrt(num / den))


def relative_h1_vs_reference(
    x_left: np.ndarray,
    x_right: np.ndarray,
    u_left: np.ndarray,
    u_right: np.ndarray,
    du_left: np.ndarray,
    du_right: np.ndarray,
    ref_left: np.ndarray,
    ref_right: np.ndarray,
    ref_du_left: np.ndarray,
    ref_du_right: np.ndarray,
) -> float:
    num = (
        np.trapezoid((u_left - ref_left) ** 2 + (du_left - ref_du_left) ** 2, x_left)
        + np.trapezoid((u_right - ref_right) ** 2 + (du_right - ref_du_right) ** 2, x_right)
    )
    den = (
        np.trapezoid(ref_left**2 + ref_du_left**2, x_left)
        + np.trapezoid(ref_right**2 + ref_du_right**2, x_right)
    )
    den = max(float(den), 1e-30)
    return float(np.sqrt(num / den))


def compute_errors_vs_reference(
    x_left: np.ndarray,
    x_right: np.ndarray,
    reference_values: Tuple[np.ndarray, np.ndarray],
    method_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_label: str = "Reference",
) -> Dict[str, float]:
    ref_left, ref_right = reference_values
    errors: Dict[str, float] = {}
    for name, (u_left, u_right) in method_values.items():
        err = relative_l2_vs_reference(
            x_left=x_left,
            x_right=x_right,
            u_left=u_left,
            u_right=u_right,
            ref_left=ref_left,
            ref_right=ref_right,
        )
        errors[f"{name} vs {reference_label}"] = err
    return errors


def compute_h1_errors_vs_reference(
    x_left: np.ndarray,
    x_right: np.ndarray,
    reference_values: Tuple[np.ndarray, np.ndarray],
    reference_derivatives: Tuple[np.ndarray, np.ndarray],
    method_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method_derivatives: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_label: str = "Reference",
) -> Dict[str, float]:
    ref_left, ref_right = reference_values
    ref_du_left, ref_du_right = reference_derivatives
    errors: Dict[str, float] = {}
    for name, (u_left, u_right) in method_values.items():
        du_left, du_right = method_derivatives[name]
        err = relative_h1_vs_reference(
            x_left=x_left,
            x_right=x_right,
            u_left=u_left,
            u_right=u_right,
            du_left=du_left,
            du_right=du_right,
            ref_left=ref_left,
            ref_right=ref_right,
            ref_du_left=ref_du_left,
            ref_du_right=ref_du_right,
        )
        errors[f"{name} vs {reference_label}"] = err
    return errors


def plot_solutions(
    series: List[Tuple[str, np.ndarray, np.ndarray]],
    xI: float,
    save_path: Path,
    reference_label: str = "Reference",
) -> None:
    x_min, x_max = _x_bounds(x for _, x, _ in series)
    windows = _zoom_windows(x_min, x_max, xI)
    curves = [(x, u) for _, x, u in series]

    fig = plt.figure(figsize=(12.8, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.34, wspace=0.24)
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.suptitle("Solution Comparison")

    for ax, (panel_title, xlim) in zip(axes, windows):
        for name, x, u in series:
            n_points = int(np.count_nonzero(np.isfinite(x) & np.isfinite(u)))
            ax.plot(x, u, label=name, **_line_style(name, reference_label=reference_label, n_points=n_points))
        _setup_axis(
            ax,
            curves=curves,
            xlim=xlim,
            xI=xI,
            title=panel_title,
            y_label="u(x)" if ax is axes[0] else None,
        )

    axes[0].legend(loc="best", fontsize=10, frameon=True)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.87)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_errors_vs_reference(
    x_left: np.ndarray,
    x_right: np.ndarray,
    reference_values: Tuple[np.ndarray, np.ndarray],
    method_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
    xI: float,
    save_path: Path,
    reference_label: str = "Reference",
) -> None:
    x_plot = np.concatenate([x_left, np.array([np.nan]), x_right])
    ref_left, ref_right = reference_values
    error_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, (u_left, u_right) in method_values.items():
        diff = np.concatenate([u_left - ref_left, np.array([np.nan]), u_right - ref_right])
        error_curves.append((name, x_plot, diff))

    x_min, x_max = _x_bounds([x_plot])
    windows = _zoom_windows(x_min, x_max, xI)
    curves = [(x, diff) for _, x, diff in error_curves]

    fig = plt.figure(figsize=(12.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.34, wspace=0.24)
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.suptitle(f"Errors vs {reference_label}")

    for ax, (panel_title, xlim) in zip(axes, windows):
        for name, x, diff in error_curves:
            n_points = int(np.count_nonzero(np.isfinite(x) & np.isfinite(diff)))
            ax.plot(x, diff, label=name, **_line_style(name, n_points=n_points))
        _setup_axis(
            ax,
            curves=curves,
            xlim=xlim,
            xI=xI,
            title=panel_title,
            y_label="difference" if ax is axes[0] else None,
        )

    axes[0].legend(loc="best", fontsize=10, frameon=True)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.87)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_pairwise_differences(
    x_left: np.ndarray,
    x_right: np.ndarray,
    common_values: Dict[str, Tuple[np.ndarray, np.ndarray]],
    xI: float,
    save_path: Path,
) -> None:
    plt.figure(figsize=(9, 4.6))

    x_plot = np.concatenate([x_left, np.array([np.nan]), x_right])

    for a, b in combinations(common_values.keys(), 2):
        aL, aR = common_values[a]
        bL, bR = common_values[b]
        diff = np.concatenate([bL - aL, np.array([np.nan]), bR - aR])
        plt.plot(x_plot, diff, linewidth=1.3, label=f"{b} - {a}")

    plt.axvline(xI, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("difference")
    plt.title("Pairwise Differences")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _format_summary_value(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return repr(value)


def write_summary(
    save_path: Path,
    timings: Dict[str, float],
    errors: Dict[str, float],
    extra_lines: Iterable[str] = (),
    errors_title: str = "Pairwise relative L2 errors:",
    additional_sections: Iterable[Tuple[str, Dict[str, float]]] = (),
    config_sections: Iterable[Tuple[str, Dict[str, object]]] = (),
) -> None:
    lines: list[str] = []

    for title, values in config_sections:
        lines.append(title)
        for k, v in values.items():
            lines.append(f"- {k}: {_format_summary_value(v)}")
        lines.append("")

    lines.append("Timings (seconds):")
    for k, v in timings.items():
        lines.append(f"- {k}: {v:.6f}")

    lines.append("")
    lines.append(errors_title)
    for k, v in errors.items():
        lines.append(f"- {k}: {v:.6e}")

    for title, values in additional_sections:
        lines.append("")
        lines.append(title)
        for k, v in values.items():
            lines.append(f"- {k}: {v:.6e}")

    extra_lines = list(extra_lines)
    if extra_lines:
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(f"- {x}" for x in extra_lines)

    save_path.write_text("\n".join(lines), encoding="utf-8")
