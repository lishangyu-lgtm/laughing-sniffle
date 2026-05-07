from dataclasses import dataclass
from typing import Callable

import numpy as np


X_DOMAIN = (0.0, 1.0)


@dataclass(frozen=True)
class ExactReference:
    left: Callable[[np.ndarray], np.ndarray]
    right: Callable[[np.ndarray], np.ndarray] | None = None
    left_derivative: Callable[[np.ndarray], np.ndarray] | None = None
    right_derivative: Callable[[np.ndarray], np.ndarray] | None = None
    label: str = "Exact"

    def evaluate(self, x: np.ndarray, side: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if side == "left":
            func = self.left
        elif side == "right":
            func = self.left if self.right is None else self.right
        else:
            raise ValueError("side must be 'left' or 'right'.")
        values = np.asarray(func(x), dtype=np.float64)
        return np.broadcast_to(values, x.shape)

    def evaluate_derivative(self, x: np.ndarray, side: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if side == "left":
            func = self.left_derivative
        elif side == "right":
            func = self.left_derivative if self.right_derivative is None else self.right_derivative
        else:
            raise ValueError("side must be 'left' or 'right'.")

        if func is None:
            raise ValueError("Exact reference derivative is not available.")

        values = np.asarray(func(x), dtype=np.float64)
        return np.broadcast_to(values, x.shape)


@dataclass
class ProblemConfig:
    eps1: float = 1.0
    eps2: float = 1e-5
    x_interface: float = 0.5
    left_bc: float = 0.0
    right_bc: float = 1.0
    jump_u: float = 0.2
    jump_flux: float = -1e-5
    reference_source: str = "fdm"


def normalize_flux_jump_average(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "equal": "equal",
        "eps": "eps_weighted",
        "eps_weighted": "eps_weighted",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported numerical.flux_jump_average={mode!r}. Supported values: {supported}."
        ) from exc


def flux_jump_average_weights(config: ProblemConfig, mode: str) -> tuple[float, float]:
    normalized = normalize_flux_jump_average(mode)
    if normalized == "equal":
        return 0.5, 0.5

    denom = float(config.eps1 + config.eps2)
    if not np.isfinite(denom) or np.isclose(denom, 0.0, atol=1e-15, rtol=1e-15):
        raise ValueError("eps1 + eps2 must be finite and nonzero for eps-weighted averaging.")

    return float(config.eps2 / denom), float(config.eps1 / denom)


def _eps_piecewise(x: np.ndarray, config: ProblemConfig) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= config.x_interface, config.eps1, config.eps2)


def x_to_y(x: np.ndarray, config: ProblemConfig) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xI = config.x_interface
    left = x / config.eps1
    right = (xI / config.eps1) + (x - xI) / config.eps2
    return np.where(x <= xI, left, right)


def transformed_interface(config: ProblemConfig) -> float:
    return float(x_to_y(np.array([config.x_interface], dtype=np.float64), config)[0])


def y_to_x(y: np.ndarray, config: ProblemConfig) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    yI = transformed_interface(config)
    left = config.eps1 * y
    right = config.x_interface + config.eps2 * (y - yI)
    return np.where(y <= yI, left, right)


def transformed_domain(config: ProblemConfig) -> tuple[float, float]:
    x_left, x_right = X_DOMAIN
    y_left = float(x_to_y(np.array([x_left], dtype=np.float64), config)[0])
    y_right = float(x_to_y(np.array([x_right], dtype=np.float64), config)[0])
    return y_left, y_right


def make_transformed_coefficient(
    config: ProblemConfig,
    c_func: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def c_transformed(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        x = y_to_x(y, config)
        return _eps_piecewise(x, config) * c_func(x)

    return c_transformed


def make_transformed_rhs(
    config: ProblemConfig,
    f_func: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def f_transformed(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        x = y_to_x(y, config)
        return _eps_piecewise(x, config) * f_func(x)

    return f_transformed


def make_transformed_coefficient_trace_functions(
    config: ProblemConfig,
    c_func: Callable[[np.ndarray], np.ndarray],
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Return left/right traces of the transformed coefficient at the interface."""
    xI = float(config.x_interface)
    yI = transformed_interface(config)
    xI_left = np.nextafter(xI, -np.inf)
    xI_right = np.nextafter(xI, np.inf)

    def _eval_at_x(x: float) -> float:
        x_arr = np.array([x], dtype=np.float64)
        return float((_eps_piecewise(x_arr, config) * c_func(x_arr))[0])

    def c_left_trace(y: float) -> float:
        y = float(y)
        if np.isclose(y, yI, atol=1e-12, rtol=1e-12):
            return _eval_at_x(xI_left)
        return _eval_at_x(float(y_to_x(np.array([y], dtype=np.float64), config)[0]))

    def c_right_trace(y: float) -> float:
        y = float(y)
        if np.isclose(y, yI, atol=1e-12, rtol=1e-12):
            return _eval_at_x(xI_right)
        return _eval_at_x(float(y_to_x(np.array([y], dtype=np.float64), config)[0]))

    return c_left_trace, c_right_trace


__all__ = [
    "ExactReference",
    "ProblemConfig",
    "X_DOMAIN",
    "flux_jump_average_weights",
    "make_transformed_coefficient",
    "make_transformed_coefficient_trace_functions",
    "make_transformed_rhs",
    "normalize_flux_jump_average",
    "transformed_domain",
    "transformed_interface",
    "x_to_y",
    "y_to_x",
]
