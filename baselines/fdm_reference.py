from typing import Callable, Dict

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from ..core.interface_utils import InterfaceLifting, derivative_weights
from ..core.tfpm_local import build_uniform_grid, interface_node_from_grid


def _second_derivative_coeffs(xm: float, x0: float, xp: float) -> tuple[float, float, float]:
    hL = x0 - xm
    hR = xp - x0
    cim1 = 2.0 / (hL * (hL + hR))
    ci = -2.0 / (hL * hR)
    cip1 = 2.0 / (hR * (hL + hR))
    return cim1, ci, cip1


def solve_interface_fdm(
    N: int | None,
    c_func: Callable[[np.ndarray], np.ndarray],
    f_func: Callable[[np.ndarray], np.ndarray],
    m: float,
    n_dir: float,
    p: float,
    qjump: float,
    grid: np.ndarray | None = None,
    a: float = 0.0,
    b: float = 1.0,
    xI: float = 0.5,
) -> Dict[str, np.ndarray]:
    if grid is None:
        if N is None:
            raise ValueError("Either grid or N must be provided.")
        x, _h, iI = build_uniform_grid(N=N, a=a, b=b, xI=xI)
    else:
        x = np.asarray(grid, dtype=np.float64)
        iI = interface_node_from_grid(x, xI)
        N = x.size - 1

    lifting = InterfaceLifting.from_jump_data(
        a=float(x[0]),
        b=float(x[-1]),
        x_interface=float(xI),
        left_bc=float(m),
        right_bc=float(n_dir),
        jump_u=float(p),
        jump_du=float(qjump),
    )

    n_unknown = N - 1
    A = lil_matrix((n_unknown, n_unknown), dtype=np.float64)
    rhs = np.zeros(n_unknown, dtype=np.float64)

    def idx(i: int) -> int:
        return i - 1

    row = 0

    for i in range(1, N):
        if i == iI:
            continue

        xi = float(x[i])
        ci = float(c_func(np.array([xi], dtype=np.float64))[0])
        gi = float(lifting.evaluate(np.array([xi], dtype=np.float64))[0])
        fi = float(f_func(np.array([xi], dtype=np.float64))[0]) - ci * gi
        cim1, cii, cip1 = _second_derivative_coeffs(float(x[i - 1]), float(x[i]), float(x[i + 1]))

        if i - 1 > 0:
            A[row, idx(i - 1)] = -cim1
        A[row, idx(i)] = -cii + ci
        if i + 1 < N:
            A[row, idx(i + 1)] = -cip1
        rhs[row] = fi
        row += 1

    left_indices = np.arange(max(0, iI - 2), iI + 1, dtype=int)
    right_indices = np.arange(iI, min(N, iI + 2) + 1, dtype=int)

    left_weights = derivative_weights(float(x[iI]), x[left_indices])
    right_weights = derivative_weights(float(x[iI]), x[right_indices])

    for coeff, node in zip(right_weights, right_indices):
        if 0 < node < N:
            A[row, idx(node)] += coeff
    for coeff, node in zip(left_weights, left_indices):
        if 0 < node < N:
            A[row, idx(node)] -= coeff
    row += 1

    if row != n_unknown:
        raise RuntimeError(f"FDM row mismatch: {row} != {n_unknown}")

    w = np.zeros(N + 1, dtype=np.float64)
    if n_unknown > 0:
        w[1:N] = spsolve(A.tocsr(), rhs)

    hL = float(x[iI] - x[iI - 1])
    hR = float(x[iI + 1] - x[iI])
    u_left = w[: iI + 1] + lifting.evaluate(x[: iI + 1], side="left")
    u_right = w[iI:] + lifting.evaluate(x[iI:], side="right")

    return {
        "x_left": x[: iI + 1],
        "u_left": u_left,
        "x_right": x[iI:],
        "u_right": u_right,
        "h_left_interface": np.array([hL], dtype=np.float64),
        "h_right_interface": np.array([hR], dtype=np.float64),
    }


def evaluate_fdm_on_side(sol: Dict[str, np.ndarray], x_eval: np.ndarray, side: str) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float64)

    if side == "left":
        return np.interp(x_eval, sol["x_left"], sol["u_left"])
    if side == "right":
        return np.interp(x_eval, sol["x_right"], sol["u_right"])

    raise ValueError("side must be 'left' or 'right'.")


def _evaluate_piecewise_linear_derivative(
    x_nodes: np.ndarray,
    u_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    u_nodes = np.asarray(u_nodes, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    if x_nodes.ndim != 1 or u_nodes.ndim != 1 or x_nodes.size != u_nodes.size:
        raise ValueError("x_nodes and u_nodes must be one-dimensional arrays with matching sizes.")
    if x_nodes.size < 2:
        raise ValueError("Need at least two nodes to evaluate a derivative.")

    slopes = np.diff(u_nodes) / np.diff(x_nodes)
    idx = np.searchsorted(x_nodes, x_eval, side="right") - 1
    idx = np.clip(idx, 0, slopes.size - 1)
    return slopes[idx]


def evaluate_fdm_derivative_on_side(sol: Dict[str, np.ndarray], x_eval: np.ndarray, side: str) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float64)

    if side == "left":
        return _evaluate_piecewise_linear_derivative(sol["x_left"], sol["u_left"], x_eval)
    if side == "right":
        return _evaluate_piecewise_linear_derivative(sol["x_right"], sol["u_right"], x_eval)

    raise ValueError("side must be 'left' or 'right'.")


def fdm_plot_arrays(sol: Dict[str, np.ndarray], points_per_cell: int = 20) -> tuple[np.ndarray, np.ndarray]:
    xL = sol["x_left"]
    uL = sol["u_left"]
    xR = sol["x_right"]
    uR = sol["u_right"]

    n_left = (len(xL) - 1) * points_per_cell + 1
    n_right = (len(xR) - 1) * points_per_cell + 1

    xL_dense = np.linspace(xL[0], xL[-1], n_left)
    uL_dense = np.interp(xL_dense, xL, uL)

    xR_dense = np.linspace(xR[0], xR[-1], n_right)
    uR_dense = np.interp(xR_dense, xR, uR)

    x_plot = np.concatenate([xL_dense, np.array([np.nan]), xR_dense])
    u_plot = np.concatenate([uL_dense, np.array([np.nan]), uR_dense])
    return x_plot, u_plot
