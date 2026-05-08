from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


_METHOD_PLOT_STYLES: dict[str, dict[str, object]] = {
    "__reference__": {
        "color": "0.45",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 1,
    },
    "TFPM-Lagrange": {
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
    if name == "TFPM-Lagrange":
        return "TFPM-Lagrange"
    if name.startswith("FDM"):
        return "FDM"
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


def plot_solutions(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    xI: float,
    save_path: Path,
    reference_label: str = "Reference",
) -> None:
    x_min, x_max = _x_bounds(x for _, x, _ in series)
    curves = [(x, u) for _, x, u in series]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for name, x, u in series:
        n_points = int(np.count_nonzero(np.isfinite(x) & np.isfinite(u)))
        ax.plot(x, u, label=name, **_line_style(name, reference_label=reference_label, n_points=n_points))

    _setup_axis(
        ax,
        curves=curves,
        xlim=(x_min, x_max),
        xI=xI,
        title="Nonlinear Solution Comparison",
        y_label="u(x)",
    )

    ax.legend(loc="best", fontsize=10, frameon=True)
    fig.tight_layout()
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
    curves = [(x, diff) for _, x, diff in error_curves]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for name, x, diff in error_curves:
        n_points = int(np.count_nonzero(np.isfinite(x) & np.isfinite(diff)))
        ax.plot(x, diff, label=name, **_line_style(name, n_points=n_points))

    _setup_axis(
        ax,
        curves=curves,
        xlim=(x_min, x_max),
        xI=xI,
        title=f"Errors vs {reference_label}",
        y_label="difference",
    )

    ax.legend(loc="best", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def write_summary(
    save_path: Path,
    timings: Dict[str, float],
    errors: Dict[str, float],
    extra_lines: Iterable[str] = (),
    errors_title: str = "Pairwise relative L2 errors:",
) -> None:
    lines = ["Timings (seconds):"]
    for k, v in timings.items():
        lines.append(f"- {k}: {v:.6f}")

    lines.append("")
    lines.append(errors_title)
    for k, v in errors.items():
        lines.append(f"- {k}: {v:.6e}")

    extra_lines = list(extra_lines)
    if extra_lines:
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(f"- {x}" for x in extra_lines)

    save_path.write_text("\n".join(lines), encoding="utf-8")
