from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_final_solutions(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    x_interface: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for label, x, u in series:
        ax.plot(x, u, linewidth=1.9, label=label)
    ax.axvline(x_interface, linestyle="--", linewidth=1.0, color="0.35", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Allen-Cahn Evolution: Final State")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_energy_histories(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for label, (times, energies) in histories.items():
        if times.size == 0 or energies.size == 0:
            continue
        ax.plot(times, energies, marker="o", markersize=3.5, linewidth=1.5, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.set_title("Allen-Cahn Evolution: Energy History")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_errors_vs_reference(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    x_interface: float,
    reference_label: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for label, x, error in series:
        ax.plot(x, np.abs(error), linewidth=1.7, label=label)
    ax.axvline(x_interface, linestyle="--", linewidth=1.0, color="0.35", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("absolute error")
    ax.set_title(f"Allen-Cahn Evolution: Final Error vs {reference_label}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def write_summary(
    save_path: Path,
    *,
    config_lines: Iterable[str],
    result_lines: Iterable[str],
    output_lines: Iterable[str],
) -> None:
    lines = ["Allen-Cahn time-evolution TFPM/AugLag experiment", ""]
    lines.append("Configuration:")
    lines.extend(f"- {line}" for line in config_lines)
    lines.append("")
    lines.append("Results:")
    lines.extend(f"- {line}" for line in result_lines)
    outputs = list(output_lines)
    if outputs:
        lines.append("")
        lines.append("Output files:")
        lines.extend(f"- {line}" for line in outputs)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["plot_energy_histories", "plot_errors_vs_reference", "plot_final_solutions", "write_summary"]
