from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ...core.tfpm_local import build_uniform_grid

ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PhysicalGrid:
    x: np.ndarray
    interface_node: int
    h: float


@dataclass(frozen=True)
class TransformedGrid:
    y: np.ndarray
    y_interface: float
    solved_on_physical_grid: bool


@dataclass(frozen=True)
class SampledState:
    x: np.ndarray
    u: np.ndarray

    def eval(self, xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=np.float64)
        return np.interp(xq, self.x, self.u)


@dataclass
class ProblemConfig:
    x_left: float = 0.0
    x_right: float = 1.0
    x_interface: float = 0.5

    eps_left: float = 0.01
    eps_right: float = 0.01
    linear_grid_mode: str = "auto"

    bc_left: float = 1.0
    bc_right: float = 1.0
    jump_u: float = 0.0
    jump_flux: float = 0.0

    initial_guess_kind: str = "smooth_periodic"
    initial_tanh_scale: float = 1.0


@dataclass
class NumericalConfig:
    n_elements: int = 100
    monitor_segments: int = 800
    quad_n: int = 10
    n_seg_gauss: int = 8
    plot_samples: int = 600
    snapshot_stride: int = 1
    use_true_c_lagrange: bool = True
    tfpm_basis: str = "endpoint_auto"


@dataclass
class LagrangeConfig:
    residual_tol: float = 1.0e-10


@dataclass
class TimeConfig:
    initial_time: float = 0.0
    final_time: float = 5.0
    dt: float = 0.1


@dataclass
class Scheme1Config:
    stabilization_shift: float = 0.0


@dataclass
class Scheme2Config:
    max_inner: int = 20
    inner_tol_step_rel: float = 1.0e-8


@dataclass
class Scheme3Config:
    max_inner: int = 10
    inner_tol_step_rel: float = 1.0e-8
    startup_mode: str = "scheme2"
    extrapolation_weight: float = 1.0


@dataclass
class ReferenceConfig:
    enabled: bool = True
    method: str = "BDF"
    n_segments: int = 2000
    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    max_step: float | None = 1.0e-2
    comparison_samples: int = 2000


@dataclass
class OutputConfig:
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "results")
    save_plots: bool = True
    save_npz: bool = True
    verbose: bool = True


@dataclass
class ExperimentConfig:
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    lagrange: LagrangeConfig = field(default_factory=LagrangeConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    scheme1: Scheme1Config = field(default_factory=Scheme1Config)
    scheme2: Scheme2Config = field(default_factory=Scheme2Config)
    scheme3: Scheme3Config = field(default_factory=Scheme3Config)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_EXPERIMENT = ExperimentConfig()


def diffusion_piecewise(problem: ProblemConfig, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    eps = np.where(x <= problem.x_interface, problem.eps_left, problem.eps_right)
    return eps**2


def nonlinearity(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return u**3 - u


def potential(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    return 0.25 * (u**2 - 1.0) ** 2


def use_direct_physical_grid(problem: ProblemConfig) -> bool:
    mode = str(problem.linear_grid_mode).strip().lower()
    if mode == "x_direct":
        return True
    if mode == "y_transformed":
        return False
    if mode != "auto":
        raise ValueError("linear_grid_mode must be 'auto', 'x_direct', or 'y_transformed'.")
    return bool(np.isclose(problem.eps_left, problem.eps_right, atol=1.0e-14, rtol=1.0e-14))


def validate_experiment(experiment: ExperimentConfig) -> None:
    problem = experiment.problem
    numerical = experiment.numerical
    time = experiment.time
    reference = experiment.reference
    if not (problem.x_left < problem.x_interface < problem.x_right):
        raise ValueError("x_interface must be a strict interior point.")
    if problem.eps_left <= 0.0 or problem.eps_right <= 0.0:
        raise ValueError("eps_left and eps_right must be positive.")
    if numerical.n_elements % 2 != 0:
        raise ValueError("n_elements must be even.")
    if numerical.monitor_segments % 2 != 0:
        raise ValueError("monitor_segments must be even.")
    if time.dt <= 0.0:
        raise ValueError("time.dt must be positive.")
    if time.final_time < time.initial_time:
        raise ValueError("final_time must be greater than or equal to initial_time.")
    if experiment.lagrange.residual_tol <= 0.0:
        raise ValueError("lagrange.residual_tol must be positive.")
    if use_direct_physical_grid(problem) and not np.isclose(
        problem.eps_left, problem.eps_right, atol=1.0e-14, rtol=1.0e-14
    ):
        raise ValueError("linear_grid_mode='x_direct' requires eps_left == eps_right.")
    if reference.enabled:
        if reference.n_segments < 4:
            raise ValueError("reference.n_segments must be at least 4.")
        if reference.n_segments % 2 != 0:
            raise ValueError("reference.n_segments must be even.")
        if reference.rtol <= 0.0 or reference.atol <= 0.0:
            raise ValueError("reference tolerances must be positive.")
        if reference.max_step is not None and reference.max_step <= 0.0:
            raise ValueError("reference.max_step must be positive or None.")
        if reference.comparison_samples < 2:
            raise ValueError("reference.comparison_samples must be at least 2.")
        if abs(problem.jump_u) > 1.0e-14 or abs(problem.jump_flux) > 1.0e-14:
            raise ValueError("FDM reference currently supports only zero interface jumps.")


def build_physical_grid(problem: ProblemConfig, n_segments: int) -> PhysicalGrid:
    x, h, interface_node = build_uniform_grid(
        N=int(n_segments),
        a=problem.x_left,
        b=problem.x_right,
        xI=problem.x_interface,
    )
    return PhysicalGrid(x=x, h=h, interface_node=interface_node)


def x_to_y(problem: ProblemConfig, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if use_direct_physical_grid(problem):
        return x.copy()
    xI = problem.x_interface
    a_left = problem.eps_left**2
    a_right = problem.eps_right**2
    y_left = (x - problem.x_left) / a_left
    y_interface = (xI - problem.x_left) / a_left
    y_right = y_interface + (x - xI) / a_right
    return np.where(x <= xI, y_left, y_right)


def y_to_x(problem: ProblemConfig, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if use_direct_physical_grid(problem):
        return y.copy()
    xI = problem.x_interface
    a_left = problem.eps_left**2
    a_right = problem.eps_right**2
    y_interface = (xI - problem.x_left) / a_left
    x_left = problem.x_left + a_left * y
    x_right = xI + a_right * (y - y_interface)
    return np.where(y <= y_interface, x_left, x_right)


def transformed_interface(problem: ProblemConfig) -> float:
    return float(x_to_y(problem, np.array([problem.x_interface], dtype=np.float64))[0])


def build_transformed_grid(problem: ProblemConfig, physical_grid: PhysicalGrid) -> TransformedGrid:
    return TransformedGrid(
        y=x_to_y(problem, physical_grid.x),
        y_interface=transformed_interface(problem),
        solved_on_physical_grid=use_direct_physical_grid(problem),
    )


def transformed_coefficient_from_physical(
    problem: ProblemConfig,
    coeff_x: ArrayFunc,
) -> ArrayFunc:
    def coeff_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        x = y_to_x(problem, y)
        a = diffusion_piecewise(problem, x)
        if use_direct_physical_grid(problem):
            return coeff_x(x) / a
        return a * coeff_x(x)

    return coeff_y


def transformed_rhs_from_physical(
    problem: ProblemConfig,
    rhs_x: ArrayFunc,
) -> ArrayFunc:
    def rhs_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        x = y_to_x(problem, y)
        a = diffusion_piecewise(problem, x)
        if use_direct_physical_grid(problem):
            return rhs_x(x) / a
        return a * rhs_x(x)

    return rhs_y


def make_initial_state(problem: ProblemConfig, x_grid: np.ndarray) -> SampledState:
    x_grid = np.asarray(x_grid, dtype=np.float64)
    kind = str(problem.initial_guess_kind).strip().lower()
    length = problem.x_right - problem.x_left
    if kind == "linear":
        slope = (problem.bc_right - problem.bc_left) / length
        u0 = problem.bc_left + slope * (x_grid - problem.x_left)
    elif kind in {"sin", "sine"}:
        u0 = np.sin(2.0 * np.pi * (x_grid - problem.x_left) / length)
    elif kind in {"smooth_periodic", "cosine", "cos"}:
        phase = 2.0 * np.pi * (x_grid - problem.x_left) / length
        endpoint_value = 0.5 * (problem.bc_left + problem.bc_right)
        u0 = endpoint_value * np.cos(phase)
    elif kind in {"periodic_interface", "periodic_tanh", "double_interface"}:
        eps_avg = 0.5 * (problem.eps_left + problem.eps_right)
        width = max(problem.initial_tanh_scale * 2.0 * np.sqrt(2.0) * eps_avg, 1.0e-8)
        center = 0.5 * (problem.x_left + problem.x_right)
        half_width = 0.25 * length
        theta = 2.0 * np.pi * (x_grid - center) / length
        theta_half = 2.0 * np.pi * half_width / length
        scale = max((2.0 * np.pi / length) * np.sin(theta_half), 1.0e-14)
        signed_distance = (np.cos(theta) - np.cos(theta_half)) / scale
        u0 = np.tanh(signed_distance / width)
    elif kind == "tanh":
        eps_avg = 0.5 * (problem.eps_left + problem.eps_right)
        width = max(problem.initial_tanh_scale * 2.0 * np.sqrt(2.0) * eps_avg, 1.0e-8)
        u0 = np.tanh((x_grid - problem.x_interface) / width)
    else:
        raise ValueError(f"Unsupported initial_guess_kind={problem.initial_guess_kind!r}.")
    u0 = np.asarray(u0, dtype=np.float64)
    u0[0] = problem.bc_left
    u0[-1] = problem.bc_right
    return SampledState(x=x_grid.copy(), u=u0)


def sampled_state_from_values(problem: ProblemConfig, x: np.ndarray, u: np.ndarray) -> SampledState:
    x = np.asarray(x, dtype=np.float64)
    out = np.asarray(u, dtype=np.float64).copy()
    out[0] = problem.bc_left
    out[-1] = problem.bc_right
    return SampledState(x=x.copy(), u=out)


__all__ = [
    "ArrayFunc",
    "DEFAULT_EXPERIMENT",
    "ExperimentConfig",
    "LagrangeConfig",
    "NumericalConfig",
    "OutputConfig",
    "PhysicalGrid",
    "ProblemConfig",
    "ReferenceConfig",
    "SampledState",
    "Scheme1Config",
    "Scheme2Config",
    "Scheme3Config",
    "TimeConfig",
    "TransformedGrid",
    "build_physical_grid",
    "build_transformed_grid",
    "diffusion_piecewise",
    "make_initial_state",
    "nonlinearity",
    "potential",
    "sampled_state_from_values",
    "transformed_coefficient_from_physical",
    "transformed_interface",
    "transformed_rhs_from_physical",
    "use_direct_physical_grid",
    "validate_experiment",
    "x_to_y",
    "y_to_x",
]
