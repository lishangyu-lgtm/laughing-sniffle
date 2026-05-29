from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ...core.linear_interface import (
    ExactReference,
    ProblemConfig,
    X_DOMAIN,
    flux_jump_average_weights,
    make_transformed_coefficient,
    make_transformed_coefficient_trace_functions,
    make_transformed_rhs,
    normalize_flux_jump_average,
    transformed_domain,
    transformed_interface,
    x_to_y,
    y_to_x,
)


DEFAULT_EPS1 = 1.0
DEFAULT_EPS2 = 1e-5
DEFAULT_X_INTERFACE = 0.5


def _coefficient_left(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 10 * np.log(1 + x)


def _coefficient_right(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(2 * x)


def _exact_left(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 + x**2


def _exact_right(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x) + 0.5 * x


def _exact_first_left(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * x


def _exact_first_right(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x) + 0.5


def _exact_second_left(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * np.ones_like(x)


def _exact_second_right(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x)


DEFAULT_LEFT_BC = 0.0
DEFAULT_RIGHT_BC = 1.0
DEFAULT_JUMP_U = -0.5
DEFAULT_JUMP_FLUX = 1e-5
COEFFICIENT_LEFT_DESCRIPTION = "10 * log(1 + x)"
COEFFICIENT_RIGHT_DESCRIPTION = "exp(2 * x)"
RHS_LEFT_DESCRIPTION = "1"
RHS_RIGHT_DESCRIPTION = "sin(2 * pi * x)"


@dataclass
class NumericalConfig:
    n_elements: int = 100
    n_ref: int = 1000000
    n_fdm: int = 1000
    n_fem: int = 1000
    quad_n: int = 10
    n_seg_gauss: int = 8
    plot_per_element: int = 250
    fdm_plot_cells: int = 20
    fem_plot_cells: int = 20
    error_samples: int = 4000
    use_true_c_lagrange: bool = True
    lagrange_residual_tol: float = 1e-10
    flux_jump_average: str = "eps_weighted"


@dataclass
class OutputConfig:
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results_2")
    save_plots: bool = True


@dataclass
class ExperimentConfig:
    problem: ProblemConfig = field(
        default_factory=lambda: ProblemConfig(
            eps1=DEFAULT_EPS1,
            eps2=DEFAULT_EPS2,
            x_interface=DEFAULT_X_INTERFACE,
            left_bc=DEFAULT_LEFT_BC,
            right_bc=DEFAULT_RIGHT_BC,
            jump_u=DEFAULT_JUMP_U,
            jump_flux=DEFAULT_JUMP_FLUX,
            reference_source="fdm",
        )
    )
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_EXPERIMENT = ExperimentConfig()


def make_coefficient(config: ProblemConfig) -> Callable[[np.ndarray], np.ndarray]:
    xI = config.x_interface

    def c_func(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left = _coefficient_left(x)
        right = _coefficient_right(x)
        return np.where(x <= xI, left, right)

    return c_func


def make_rhs(config: ProblemConfig) -> Callable[[np.ndarray], np.ndarray]:
    xI = config.x_interface

    def f_func(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left = np.ones_like(x)
        right = np.sin(2 * np.pi * x)
        return np.where(x < xI, left, right)

    return f_func


def make_exact_reference(config: ProblemConfig) -> ExactReference | None:
    _ = config
    return ExactReference(
        left=_exact_left,
        right=_exact_right,
        left_derivative=_exact_first_left,
        right_derivative=_exact_first_right,
        label="Exact",
    )


__all__ = [
    "DEFAULT_EPS1",
    "DEFAULT_EPS2",
    "DEFAULT_EXPERIMENT",
    "DEFAULT_JUMP_FLUX",
    "DEFAULT_JUMP_U",
    "DEFAULT_LEFT_BC",
    "DEFAULT_RIGHT_BC",
    "DEFAULT_X_INTERFACE",
    "COEFFICIENT_LEFT_DESCRIPTION",
    "COEFFICIENT_RIGHT_DESCRIPTION",
    "ExactReference",
    "ExperimentConfig",
    "NumericalConfig",
    "OutputConfig",
    "ProblemConfig",
    "RHS_LEFT_DESCRIPTION",
    "RHS_RIGHT_DESCRIPTION",
    "X_DOMAIN",
    "flux_jump_average_weights",
    "make_coefficient",
    "make_exact_reference",
    "make_rhs",
    "make_transformed_coefficient",
    "make_transformed_coefficient_trace_functions",
    "make_transformed_rhs",
    "normalize_flux_jump_average",
    "transformed_domain",
    "transformed_interface",
    "x_to_y",
    "y_to_x",
]
