from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from ..core.interface_utils import InterfaceLifting
from ..core.tfpm_local import build_uniform_grid, interface_node_from_grid


@dataclass
class FEMLinearSystem:
    A: csr_matrix
    b: np.ndarray
    x: np.ndarray
    iI: int
    lifting: InterfaceLifting


def assemble_interface_fem_system(
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
) -> FEMLinearSystem:
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

    n_nodes = N + 1
    A = lil_matrix((n_nodes, n_nodes), dtype=np.float64)
    rhs = np.zeros(n_nodes, dtype=np.float64)

    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=np.float64)
    wi = np.array([1.0, 1.0], dtype=np.float64)

    def assemble_element(ia: int, ib: int, xa: float, xb: float) -> None:
        he = xb - xa
        k_local = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64) / he
        m_local = np.zeros((2, 2), dtype=np.float64)
        f_local = np.zeros(2, dtype=np.float64)

        for xiq, wiq in zip(xi, wi):
            xq = 0.5 * (xb - xa) * xiq + 0.5 * (xa + xb)
            wq = 0.5 * (xb - xa) * wiq
            N1 = (xb - xq) / he
            N2 = (xq - xa) / he
            Nv = np.array([N1, N2], dtype=np.float64)

            cq = float(c_func(np.array([xq], dtype=np.float64))[0])
            gq = float(lifting.evaluate(np.array([xq], dtype=np.float64))[0])
            fq = float(f_func(np.array([xq], dtype=np.float64))[0]) - cq * gq
            m_local += wq * cq * np.outer(Nv, Nv)
            f_local += wq * fq * Nv

        a_local = k_local + m_local
        ids = (ia, ib)
        for r in range(2):
            rhs[ids[r]] += f_local[r]
            for s in range(2):
                A[ids[r], ids[s]] += a_local[r, s]

    for e in range(N):
        assemble_element(e, e + 1, float(x[e]), float(x[e + 1]))

    def impose_zero_dirichlet(row: int) -> None:
        A.rows[row] = []
        A.data[row] = []
        A[row, row] = 1.0
        rhs[row] = 0.0

    impose_zero_dirichlet(0)
    impose_zero_dirichlet(N)

    return FEMLinearSystem(A=A.tocsr(), b=rhs, x=x, iI=iI, lifting=lifting)


def solve_interface_fem_from_system(system: FEMLinearSystem) -> Dict[str, np.ndarray]:
    w = spsolve(system.A, system.b)
    iI = system.iI
    hL = float(system.x[iI] - system.x[iI - 1])
    hR = float(system.x[iI + 1] - system.x[iI])

    return {
        "x_left": system.x[: iI + 1],
        "u_left": w[: iI + 1] + system.lifting.evaluate(system.x[: iI + 1], side="left"),
        "x_right": system.x[iI:],
        "u_right": w[iI:] + system.lifting.evaluate(system.x[iI:], side="right"),
        "h_left_interface": np.array([hL], dtype=np.float64),
        "h_right_interface": np.array([hR], dtype=np.float64),
    }


def solve_interface_fem(
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
    system = assemble_interface_fem_system(
        N=N,
        c_func=c_func,
        f_func=f_func,
        m=m,
        n_dir=n_dir,
        p=p,
        qjump=qjump,
        grid=grid,
        a=a,
        b=b,
        xI=xI,
    )
    return solve_interface_fem_from_system(system)


def evaluate_fem_on_side(sol: Dict[str, np.ndarray], x_eval: np.ndarray, side: str) -> np.ndarray:
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


def evaluate_fem_derivative_on_side(sol: Dict[str, np.ndarray], x_eval: np.ndarray, side: str) -> np.ndarray:
    x_eval = np.asarray(x_eval, dtype=np.float64)

    if side == "left":
        return _evaluate_piecewise_linear_derivative(sol["x_left"], sol["u_left"], x_eval)
    if side == "right":
        return _evaluate_piecewise_linear_derivative(sol["x_right"], sol["u_right"], x_eval)

    raise ValueError("side must be 'left' or 'right'.")


def fem_plot_arrays(sol: Dict[str, np.ndarray], points_per_cell: int = 20) -> tuple[np.ndarray, np.ndarray]:
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
