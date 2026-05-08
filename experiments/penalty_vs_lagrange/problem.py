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
DEFAULT_EPS2 = 1.0
DEFAULT_X_INTERFACE = 0.5


def _coefficient_left(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2 * np.exp(x)


def _coefficient_right(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1 - x) ** 2


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


DEFAULT_LEFT_BC = float(_exact_left(np.array([X_DOMAIN[0]], dtype=np.float64))[0])
DEFAULT_RIGHT_BC = float(_exact_right(np.array([X_DOMAIN[1]], dtype=np.float64))[0])
DEFAULT_JUMP_U = float(
    _exact_right(np.array([DEFAULT_X_INTERFACE], dtype=np.float64))[0]
    - _exact_left(np.array([DEFAULT_X_INTERFACE], dtype=np.float64))[0]
)
DEFAULT_JUMP_FLUX = float(
    DEFAULT_EPS2 * (np.exp(DEFAULT_X_INTERFACE) + 0.5) - DEFAULT_EPS1 * (2.0 * DEFAULT_X_INTERFACE)
)


@dataclass
class NumericalConfig:
    n_elements: int = 10
    quad_n: int = 10
    n_seg_gauss: int = 8
    plot_per_element: int = 250
    error_samples: int = 4000
    use_true_c_penalty: bool = True
    use_true_c_lagrange: bool = True
    flux_jump_average: str = "eps_weighted"
    penalty_gamma: float = 1.0e4
    penalty_gammas: tuple[float, ...] = (1.0e4, 1.0e5, 1.0e6)
    penalty_residual_tol: float = 1.0e-10
    lagrange_residual_tol: float = 1.0e-10


@dataclass
class OutputConfig:
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results")
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
            reference_source="exact",
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
    c_func = make_coefficient(config)

    def f_func(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        c_val = c_func(x)
        f_left = -config.eps1 * _exact_second_left(x) + c_val * _exact_left(x)
        f_right = -config.eps2 * _exact_second_right(x) + c_val * _exact_right(x)
        return np.where(x <= xI, f_left, f_right)

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
    "ExactReference",
    "ExperimentConfig",
    "NumericalConfig",
    "OutputConfig",
    "ProblemConfig",
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
