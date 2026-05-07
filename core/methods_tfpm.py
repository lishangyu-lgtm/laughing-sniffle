from dataclasses import dataclass
from typing import Callable, List, Tuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import MatrixRankWarning, lsmr, splu, spsolve

from .tfpm_local import ElementData, interface_node_from_grid


@dataclass
class AugLagHistory:
    iter: int
    primal_inf: float
    stationarity_inf: float
    converged: bool
    rho: float


@dataclass
class LagrangeHistory:
    iter: int
    primal_inf: float
    stationarity_inf: float
    residual_inf: float
    converged: bool


@dataclass
class PenaltyHistory:
    gamma: float
    primal_inf: float
    stationarity_inf: float
    converged: bool


def solve_tfpm_strong_matching(
    grid: np.ndarray,
    elems: List[ElementData],
    m: float,
    n_dir: float,
    p: float,
    qjump: float,
    xI: float = 0.5,
) -> np.ndarray:
    N = len(elems)
    ndof = 2 * N
    interface_node = interface_node_from_grid(grid, xI)

    A = lil_matrix((ndof, ndof), dtype=np.float64)
    b = np.zeros(ndof, dtype=np.float64)

    def dof(e: int, r: int) -> int:
        return 2 * e + r

    row = 0

    el0 = elems[0]
    A[row, dof(0, 0)] = el0.phi1_L
    A[row, dof(0, 1)] = el0.phi2_L
    b[row] = m - el0.up_L
    row += 1

    for i_node in range(1, N):
        eL = i_node - 1
        eR = i_node
        elL = elems[eL]
        elR = elems[eR]

        jump_u = p if (i_node == interface_node) else 0.0
        jump_du = qjump if (i_node == interface_node) else 0.0

        A[row, dof(eR, 0)] = elR.phi1_L
        A[row, dof(eR, 1)] = elR.phi2_L
        A[row, dof(eL, 0)] = -elL.phi1_R
        A[row, dof(eL, 1)] = -elL.phi2_R
        b[row] = jump_u + (elL.up_R - elR.up_L)
        row += 1

        A[row, dof(eR, 0)] = elR.dphi1_L
        A[row, dof(eR, 1)] = elR.dphi2_L
        A[row, dof(eL, 0)] = -elL.dphi1_R
        A[row, dof(eL, 1)] = -elL.dphi2_R
        b[row] = jump_du + (elL.dup_R - elR.dup_L)
        row += 1

    elN = elems[-1]
    A[row, dof(N - 1, 0)] = elN.phi1_R
    A[row, dof(N - 1, 1)] = elN.phi2_R
    b[row] = n_dir - elN.up_R
    row += 1

    if row != ndof:
        raise RuntimeError(f"Strong matching assembly row mismatch: {row} != {ndof}")

    return spsolve(A.tocsr(), b)


def assemble_lagrange_kkt_system(
    grid: np.ndarray,
    elems: List[ElementData],
    f_func: Callable[[np.ndarray], np.ndarray],
    m: float,
    n_dir: float,
    p: float,
    jump_du: float,
    xI: float = 0.5,
    c_func: Callable[[np.ndarray], np.ndarray] | None = None,
    use_true_c: bool = False,
    flux_jump_weights: tuple[float, float] = (0.5, 0.5),
) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    N = len(elems)
    ndof = 2 * N
    interface_node = interface_node_from_grid(grid, xI)

    H = lil_matrix((ndof, ndof), dtype=np.float64)
    l = np.zeros(ndof, dtype=np.float64)

    def dof(e: int, r: int) -> int:
        return 2 * e + r

    if use_true_c and c_func is None:
        raise ValueError("use_true_c=True requires c_func.")

    flux_jump_weights = tuple(float(w) for w in flux_jump_weights)
    if len(flux_jump_weights) != 2:
        raise ValueError("flux_jump_weights must contain exactly two entries.")
    if not np.all(np.isfinite(flux_jump_weights)):
        raise ValueError("flux_jump_weights must be finite.")
    if not np.isclose(sum(flux_jump_weights), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("flux_jump_weights must sum to 1.")
    flux_left_weight, flux_right_weight = flux_jump_weights

    for e, el in enumerate(elems):
        xq, wq = el.xq, el.wq
        if use_true_c:
            cq = c_func(xq)
        else:
            cq = el.a * xq + el.b
        fq = f_func(xq)

        Phi = np.vstack([el.phi1_q, el.phi2_q])
        dPhi = np.vstack([el.dphi1_q, el.dphi2_q])

        for r in range(2):
            for s in range(2):
                H[dof(e, r), dof(e, s)] += np.sum(wq * (dPhi[r] * dPhi[s] + cq * Phi[r] * Phi[s]))

        for r in range(2):
            l[dof(e, r)] += np.sum(wq * (el.dup_q * dPhi[r] + cq * el.up_q * Phi[r] - fq * Phi[r]))

    eL = interface_node - 1
    eR = interface_node
    elL = elems[eL]
    elR = elems[eR]

    idxL = np.array([dof(eL, 0), dof(eL, 1)], dtype=int)
    idxR = np.array([dof(eR, 0), dof(eR, 1)], dtype=int)
    PhiL = np.array([elL.phi1_R, elL.phi2_R], dtype=np.float64)
    PhiR = np.array([elR.phi1_L, elR.phi2_L], dtype=np.float64)

    l[idxL] += flux_left_weight * jump_du * PhiL
    l[idxR] += flux_right_weight * jump_du * PhiR

    mcon = N + 1
    C = lil_matrix((mcon, ndof), dtype=np.float64)
    d = np.zeros(mcon, dtype=np.float64)

    row = 0

    el0 = elems[0]
    C[row, dof(0, 0)] = el0.phi1_L
    C[row, dof(0, 1)] = el0.phi2_L
    d[row] = m - el0.up_L
    row += 1

    for i_node in range(1, N):
        eLm = i_node - 1
        eRp = i_node
        elLm = elems[eLm]
        elRp = elems[eRp]

        C[row, dof(eRp, 0)] = elRp.phi1_L
        C[row, dof(eRp, 1)] = elRp.phi2_L
        C[row, dof(eLm, 0)] = -elLm.phi1_R
        C[row, dof(eLm, 1)] = -elLm.phi2_R

        target = p if (i_node == interface_node) else 0.0
        d[row] = target + (elLm.up_R - elRp.up_L)
        row += 1

    elN = elems[-1]
    C[row, dof(N - 1, 0)] = elN.phi1_R
    C[row, dof(N - 1, 1)] = elN.phi2_R
    d[row] = n_dir - elN.up_R
    row += 1

    if row != mcon:
        raise RuntimeError(f"Lagrange constraint row mismatch: {row} != {mcon}")

    K = lil_matrix((ndof + mcon, ndof + mcon), dtype=np.float64)
    K[:ndof, :ndof] = H
    K[:ndof, ndof:] = C.T
    K[ndof:, :ndof] = C

    rhs = np.zeros(ndof + mcon, dtype=np.float64)
    rhs[:ndof] = -l
    rhs[ndof:] = d

    return K.tocsr(), rhs, H.tocsr(), l, C.tocsr(), d


def solve_lagrange_energy(K: csr_matrix, rhs: np.ndarray, residual_tol: float = 1e-8) -> np.ndarray:
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", MatrixRankWarning)
        sol = spsolve(K, rhs)

    rank_warn = any(isinstance(w.message, MatrixRankWarning) for w in ws)
    res_inf = float(np.max(np.abs(K @ sol - rhs)))
    rhs_scale = max(1.0, float(np.max(np.abs(rhs))))

    bad = (not np.all(np.isfinite(sol))) or rank_warn or (res_inf > residual_tol * rhs_scale)
    if bad:
        sol_lsmr = lsmr(K, rhs, atol=1e-12, btol=1e-12, maxiter=20000)[0]
        res2_inf = float(np.max(np.abs(K @ sol_lsmr - rhs)))
        print(
            f"[Lagrange solver] spsolve unstable/singular (res_inf={res_inf:.3e}); "
            f"fallback LSMR (res_inf={res2_inf:.3e})."
        )
        sol = sol_lsmr

    return sol


def solve_lagrange_kkt(
    K: csr_matrix,
    rhs: np.ndarray,
    H: csr_matrix,
    l: np.ndarray,
    C: csr_matrix,
    d: np.ndarray,
    residual_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, LagrangeHistory]:
    ndof = H.shape[0]
    mcon = C.shape[0]
    expected_shape = (ndof + mcon, ndof + mcon)
    if K.shape != expected_shape:
        raise ValueError(f"K shape mismatch: expected {expected_shape}, got {K.shape}.")
    if rhs.shape != (ndof + mcon,):
        raise ValueError(f"rhs shape mismatch: expected {(ndof + mcon,)}, got {rhs.shape}.")

    sol = solve_lagrange_energy(K, rhs, residual_tol=residual_tol)
    z = np.asarray(sol[:ndof], dtype=np.float64)
    lam = np.asarray(sol[ndof:], dtype=np.float64)

    primal = C @ z - d
    stationarity = H @ z + l + C.T @ lam
    residual = K @ sol - rhs

    primal_inf = float(np.max(np.abs(primal)))
    stationarity_inf = float(np.max(np.abs(stationarity)))
    residual_inf = float(np.max(np.abs(residual)))
    rhs_scale = max(1.0, float(np.max(np.abs(rhs))))
    converged = bool(np.all(np.isfinite(sol)) and residual_inf <= residual_tol * rhs_scale)

    history = LagrangeHistory(
        iter=1,
        primal_inf=primal_inf,
        stationarity_inf=stationarity_inf,
        residual_inf=residual_inf,
        converged=converged,
    )
    return z, lam, history


def solve_augmented_lagrange(
    H: csr_matrix,
    l: np.ndarray,
    C: csr_matrix,
    d: np.ndarray,
    rho: float,
    max_iter: int,
    tol_primal: float,
    tol_stationarity: float,
    relax: float,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, AugLagHistory]:
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if relax <= 0.0:
        raise ValueError("relax must be positive.")

    ndof = H.shape[0]
    mcon = C.shape[0]

    lam = np.zeros(mcon, dtype=np.float64)
    Ct = C.T.tocsr()
    Ct_d = Ct @ d

    A = (H + rho * (Ct @ C)).tocsr()

    lu = None
    try:
        lu = splu(A.tocsc())
    except Exception:
        lu = None

    d_scale = max(1.0, float(np.max(np.abs(d))))
    l_scale = max(1.0, float(np.max(np.abs(l))))

    primal_inf = np.inf
    stationarity_inf = np.inf

    for it in range(1, max_iter + 1):
        rhs = -l - (Ct @ lam) + rho * Ct_d

        if lu is not None:
            z = lu.solve(rhs)
        else:
            z = lsmr(A, rhs, atol=1e-12, btol=1e-12, maxiter=20000)[0]

        primal = C @ z - d
        lam = lam + relax * rho * primal

        stationarity = H @ z + l + Ct @ lam

        primal_inf = float(np.max(np.abs(primal)))
        stationarity_inf = float(np.max(np.abs(stationarity)))

        if verbose:
            print(
                f"[AugLag] iter={it:4d}, "
                f"max|Cz-d|={primal_inf:.3e}, "
                f"max|Hz+l+C^T lambda|={stationarity_inf:.3e}"
            )

        if (
            primal_inf <= tol_primal * d_scale
            and stationarity_inf <= tol_stationarity * l_scale
        ):
            history = AugLagHistory(
                iter=it,
                primal_inf=primal_inf,
                stationarity_inf=stationarity_inf,
                converged=True,
                rho=float(rho),
            )
            return z, lam, history

    raise RuntimeError(
        "AugLag did not converge within max_iter="
        f"{max_iter}. final max|Cz-d|={primal_inf:.3e}, "
        f"final max|Hz+l+C^T lambda|={stationarity_inf:.3e}"
    )


def solve_penalty_method(
    H: csr_matrix,
    l: np.ndarray,
    C: csr_matrix,
    d: np.ndarray,
    gamma: float,
    residual_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, PenaltyHistory]:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    Ct = C.T.tocsr()
    Ct_d = Ct @ d
    A = (H + gamma * (Ct @ C)).tocsr()
    rhs = -l + gamma * Ct_d

    lu = None
    try:
        lu = splu(A.tocsc())
    except Exception:
        lu = None

    if lu is not None:
        z = lu.solve(rhs)
    else:
        z = lsmr(A, rhs, atol=1e-12, btol=1e-12, maxiter=20000)[0]

    stationarity = H @ z + l + gamma * (Ct @ (C @ z - d))
    stationarity_inf = float(np.max(np.abs(stationarity)))
    rhs_scale = max(1.0, float(np.max(np.abs(rhs))))

    bad = (not np.all(np.isfinite(z))) or (stationarity_inf > residual_tol * rhs_scale)
    if bad:
        z_lsmr = lsmr(A, rhs, atol=1e-12, btol=1e-12, maxiter=20000)[0]
        stationarity_lsmr = H @ z_lsmr + l + gamma * (Ct @ (C @ z_lsmr - d))
        stationarity_lsmr_inf = float(np.max(np.abs(stationarity_lsmr)))
        if stationarity_lsmr_inf < stationarity_inf or not np.all(np.isfinite(z)):
            z = z_lsmr
            stationarity_inf = stationarity_lsmr_inf

    primal = C @ z - d
    lam = gamma * primal
    primal_inf = float(np.max(np.abs(primal)))
    converged = np.isfinite(stationarity_inf) and stationarity_inf <= residual_tol * rhs_scale

    history = PenaltyHistory(
        gamma=float(gamma),
        primal_inf=primal_inf,
        stationarity_inf=stationarity_inf,
        converged=converged,
    )
    return z, lam, history


def split_kkt_solution(sol: np.ndarray, ndof: int) -> Tuple[np.ndarray, np.ndarray]:
    return sol[:ndof], sol[ndof:]
