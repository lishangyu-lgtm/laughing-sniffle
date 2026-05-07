from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SELECTIONS = [
    ("eps_equal", "case_flux_jump_only"),
    ("eps_right_smaller", "case_value_flux_jump"),
]
DEFAULT_MAX_N_TO_SHOW = 256

_METHOD_PLOT_STYLES: dict[str, dict[str, object]] = {
    "AugLag-L2": {
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 5.2,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "zorder": 3,
    },
    "AugLag-EpsNorm": {
        "color": "tab:green",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "s",
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "zorder": 3,
    },
}

_SCENARIO_LABELS = {
    "eps_equal": "Equal diffusion example",
    "eps_right_smaller": "Right-side small diffusion example",
    "eps_left_smaller": "Left-side small diffusion example",
}


@dataclass(frozen=True)
class RunRow:
    n_elements: int
    h: float
    tfpm_l2: float
    tfpm_eps_norm: float
    auglag_l2: float
    auglag_eps_norm: float
    auglag_iter: int


@dataclass(frozen=True)
class CaseSummary:
    scenario_name: str
    scenario_description: str
    eps1: float
    eps2: float
    case_name: str
    case_description: str
    jump_u: float
    jump_flux: float
    runs: list[RunRow]


def _parse_selection(value: str) -> tuple[str, str]:
    scenario, sep, case = value.partition(":")
    if not sep or not scenario or not case:
        raise argparse.ArgumentTypeError(
            "Selections must look like 'scenario_name:case_name'."
        )
    return scenario.strip(), case.strip()


def _format_eps(value: float) -> str:
    return f"{value:g}"


def _reference_curve(x: np.ndarray, y: np.ndarray, order: float, anchor_index: int = 3) -> np.ndarray:
    idx = min(max(anchor_index, 0), x.size - 1)
    x0 = float(x[idx])
    y0 = float(y[idx])
    return y0 * (x / x0) ** (-order)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _parse_summary(summary_path: Path) -> dict[tuple[str, str], CaseSummary]:
    scenario_header = re.compile(r"^===\s+(.+?)\s+===\s*$")
    case_header = re.compile(r"^---\s+(.+?)\s+---\s*$")
    eps_line = re.compile(
        r"^eps1 = ([^,]+), eps2 = ([^(]+)\((.+)\)\s*$"
    )
    interface_line = re.compile(
        r"^interface data: jump_u = ([^,]+), jump_flux = (.+)$"
    )
    row_line = re.compile(r"^\d+\s+[0-9.eE+-]+\s+[0-9.eE+-]+")

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    parsed: dict[tuple[str, str], CaseSummary] = {}
    current_scenario_name: str | None = None
    current_scenario_description = ""
    current_eps1 = 0.0
    current_eps2 = 0.0
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        scenario_match = scenario_header.match(line)
        if scenario_match is not None:
            current_scenario_name = scenario_match.group(1)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            eps_match = eps_line.match(lines[i].strip())
            if eps_match is None:
                raise ValueError(f"Failed to parse epsilon line after scenario '{current_scenario_name}'.")
            current_eps1 = float(eps_match.group(1).strip())
            current_eps2 = float(eps_match.group(2).strip())
            current_scenario_description = eps_match.group(3).strip()
            i += 1
            continue

        case_match = case_header.match(line)
        if case_match is None:
            i += 1
            continue
        if current_scenario_name is None:
            raise ValueError(f"Case '{case_match.group(1)}' appeared before any scenario block.")

        case_name = case_match.group(1).strip()
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing description for case '{case_name}'.")
        case_description = lines[i].strip()

        jump_u = 0.0
        jump_flux = 0.0
        while i < len(lines):
            interface_match = interface_line.match(lines[i].strip())
            if interface_match is not None:
                jump_u = float(interface_match.group(1).strip())
                jump_flux = float(interface_match.group(2).strip())
                i += 1
                break
            i += 1
        else:
            raise ValueError(f"Missing interface data for case '{case_name}'.")

        while i < len(lines) and "TFPM_L2" not in lines[i]:
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing table header for case '{case_name}'.")
        i += 1

        runs: list[RunRow] = []
        while i < len(lines):
            row = lines[i].strip()
            if not row_line.match(row):
                break
            parts = row.split()
            if len(parts) < 11:
                raise ValueError(f"Unexpected table row in case '{case_name}': {row}")
            runs.append(
                RunRow(
                    n_elements=int(parts[0]),
                    h=float(parts[1]),
                    tfpm_l2=float(parts[2]),
                    tfpm_eps_norm=float(parts[4]),
                    auglag_l2=float(parts[6]),
                    auglag_eps_norm=float(parts[8]),
                    auglag_iter=int(parts[10]),
                )
            )
            i += 1

        parsed[(current_scenario_name, case_name)] = CaseSummary(
            scenario_name=current_scenario_name,
            scenario_description=current_scenario_description,
            eps1=current_eps1,
            eps2=current_eps2,
            case_name=case_name,
            case_description=case_description,
            jump_u=jump_u,
            jump_flux=jump_flux,
            runs=runs,
        )

    return parsed


def _setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (elements)")
    ax.grid(True, which="major", alpha=0.28)


def _plot_auglag_summary(
    ax: plt.Axes,
    n_values: np.ndarray,
    auglag_l2: np.ndarray,
    auglag_eps: np.ndarray,
    title: str,
    y_label: str,
    l2_order: float,
    eps_order: float,
    reference_note: str,
) -> None:
    ax.plot(n_values, auglag_l2, label=r"AugLag $L^2$", **_METHOD_PLOT_STYLES["AugLag-L2"])
    ax.plot(
        n_values,
        auglag_eps,
        label=r"AugLag eps-weighted $H^1$",
        **_METHOD_PLOT_STYLES["AugLag-EpsNorm"],
    )

    l2_ref = _reference_curve(n_values, auglag_l2, order=l2_order)
    eps_ref = _reference_curve(n_values, auglag_eps, order=eps_order)
    ax.plot(
        n_values,
        l2_ref,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
        label=rf"$O(N^{{-{int(l2_order)}}})$",
        zorder=1,
    )
    ax.plot(
        n_values,
        eps_ref,
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
        label=rf"$O(N^{{-{int(eps_order)}}})$",
        zorder=1,
    )

    _setup_axis(ax, title=title)
    ax.set_ylabel(y_label)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(int(v)) for v in n_values], rotation=0)
    ax.text(
        0.03,
        0.06,
        reference_note,
        transform=ax.transAxes,
        fontsize=9.2,
        color="0.35",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.legend(loc="best", frameon=True)


def plot_case_summary(case_summary: CaseSummary, save_path: Path) -> None:
    n_values = np.array([run.n_elements for run in case_summary.runs], dtype=np.float64)
    auglag_l2 = np.array([run.auglag_l2 for run in case_summary.runs], dtype=np.float64)
    auglag_eps = np.array([run.auglag_eps_norm for run in case_summary.runs], dtype=np.float64)

    visible = n_values <= DEFAULT_MAX_N_TO_SHOW
    n_values = n_values[visible]
    auglag_l2 = auglag_l2[visible]
    auglag_eps = auglag_eps[visible]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    _plot_auglag_summary(
        ax,
        n_values=n_values,
        auglag_l2=auglag_l2,
        auglag_eps=auglag_eps,
        title="",
        y_label="relative error",
        l2_order=4.0,
        eps_order=3.0,
        reference_note=r"ref slopes: $L^2 \sim N^{-4}$, eps-weighted $H^1 \sim N^{-3}$",
    )

    title_prefix = _SCENARIO_LABELS.get(case_summary.scenario_name, case_summary.scenario_name)
    fig.suptitle(
        f"{title_prefix}: {case_summary.case_name}",
        fontsize=13.2,
        y=0.965,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.84)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def generate_selected_figures(
    summary_path: Path,
    output_dir: Path,
    selections: list[tuple[str, str]],
) -> list[Path]:
    parsed = _parse_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for scenario_name, case_name in selections:
        key = (scenario_name, case_name)
        if key not in parsed:
            available = ", ".join(
                f"{scenario}:{case}" for scenario, case in sorted(parsed.keys())
            )
            raise KeyError(
                f"Unknown selection '{scenario_name}:{case_name}'. Available selections: {available}"
            )
        save_path = output_dir / f"example_{_slug(scenario_name)}_{_slug(case_name)}.png"
        plot_case_summary(parsed[key], save_path=save_path)
        created.append(save_path)

    return created


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot selected convergence examples from summary.txt."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=here / "results" / "summary.txt",
        help="Path to the convergence summary file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "results",
        help="Directory where figure files are written.",
    )
    parser.add_argument(
        "--selection",
        action="append",
        type=_parse_selection,
        help="Selection to plot, written as scenario_name:case_name. Repeat to make multiple figures.",
    )
    args = parser.parse_args()

    selections = args.selection if args.selection else DEFAULT_SELECTIONS
    created = generate_selected_figures(
        summary_path=args.summary,
        output_dir=args.out_dir,
        selections=selections,
    )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
