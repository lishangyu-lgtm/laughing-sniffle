from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.special import airy, airye


@dataclass
class ElementData:
    xL: float
    xR: float
    h: float
    a: float
    b: float
    xq: np.ndarray
    wq: np.ndarray
    phi1_q: np.ndarray
    phi2_q: np.ndarray
    dphi1_q: np.ndarray
    dphi2_q: np.ndarray
    up_q: np.ndarray
    dup_q: np.ndarray
    phi1_L: float
    phi2_L: float
    dphi1_L: float
    dphi2_L: float
    up_L: float
    dup_L: float
    phi1_R: float
    phi2_R: float
    dphi1_R: float
    dphi2_R: float
    up_R: float
    dup_R: float
    basis_kind: str = "endpoint_airy"
    basis_b0: float = 0.0


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    xi, wi = np.polynomial.legendre.leggauss(n)
    return xi, wi


def map_to_interval(xi: np.ndarray, wi: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w


def build_uniform_grid(N: int, a: float, b: float, xI: float) -> Tuple[np.ndarray, float, int]:
    if N % 2 != 0:
        raise ValueError("N must be even.")
    if not (a < xI < b):
        raise ValueError("xI must be a strict interior point of the interval.")

    grid = np.linspace(a, b, N + 1)
    h = float(grid[1] - grid[0])
    interface_node = int(round((xI - a) / h))

    if interface_node < 1 or interface_node >= N:
        raise ValueError("xI must correspond to an interior grid node.")
    if not np.isclose(grid[interface_node], xI, atol=1e-12, rtol=1e-12):
        raise ValueError(
            f"Interface xI={xI:.16g} does not lie on the uniform grid for N={N}, "
            f"interval=[{a:.16g}, {b:.16g}]. Adjust the total grid count so the interface is a grid node."
        )

    return grid, h, interface_node


def interface_node_from_grid(grid: np.ndarray, xI: float) -> int:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid must be strictly increasing.")

    matches = np.flatnonzero(np.isclose(grid, xI, atol=1e-12, rtol=1e-12))
    if matches.size != 1:
        raise ValueError("xI must coincide with exactly one grid node.")

    interface_node = int(matches[0])
    if interface_node < 1 or interface_node >= grid.size - 1:
        raise ValueError("xI must correspond to a strict interior grid node.")

    return interface_node


def make_trace_functions(
    c_func: Callable[[np.ndarray], np.ndarray],
    xI: float,
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    xI = float(xI)
    xI_left = np.nextafter(xI, -np.inf)
    xI_right = np.nextafter(xI, np.inf)

    def c_left_trace(x: float) -> float:
        x = float(x)
        if np.isclose(x, xI, atol=1e-12, rtol=1e-12):
            x = xI_left
        return float(c_func(np.array([x], dtype=np.float64))[0])

    def c_right_trace(x: float) -> float:
        x = float(x)
        if np.isclose(x, xI, atol=1e-12, rtol=1e-12):
            x = xI_right
        return float(c_func(np.array([x], dtype=np.float64))[0])

    return c_left_trace, c_right_trace


def linearize_c_on_element(
    xL: float,
    xR: float,
    c_left_trace: Callable[[float], float],
    c_right_trace: Callable[[float], float],
) -> Tuple[float, float]:
    h = xR - xL
    cL = c_right_trace(xL)
    cR = c_left_trace(xR)
    a = (cR - cL) / h
    b = cL - a * xL
    return a, b


def _eval_constant_endpoint_basis(
    b0: float,
    xL: float,
    xR: float,
    x: np.ndarray,
    b_tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    h = float(xR - xL)
    if h <= 0.0:
        raise ValueError("Endpoint basis requires xR > xL.")

    if abs(b0) <= b_tol:
        phi1 = (xR - x) / h
        phi2 = (x - xL) / h
        dphi1 = -np.ones_like(x) / h
        dphi2 = np.ones_like(x) / h
        return phi1, phi2, dphi1, dphi2

    if b0 > 0.0:
        return eval_endpoint_constant_positive_basis(b0=b0, xL=xL, xR=xR, x=x)

    beta = -float(b0)
    s = np.sqrt(beta)
    L = s * h
    den = np.sin(L)
    if abs(den) < 1.0e-14:
        raise ValueError(
            "Constant negative endpoint basis is singular or nearly singular "
            f"on [{xL:.16g}, {xR:.16g}]."
        )
    left_dist = s * (x - xL)
    right_dist = s * (xR - x)
    phi1 = np.sin(right_dist) / den
    phi2 = np.sin(left_dist) / den
    dphi1 = -s * np.cos(right_dist) / den
    dphi2 = s * np.cos(left_dist) / den
    return phi1, phi2, dphi1, dphi2


def _normalize_basis_kind(basis_kind: str) -> str:
    key = str(basis_kind).strip().lower().replace("-", "_")
    aliases = {
        "endpoint": "endpoint_auto",
        "endpoints": "endpoint_auto",
        "endpoint_auto": "endpoint_auto",
        "endpoint_airy": "endpoint_airy",
        "endpoint_variable": "endpoint_airy",
        "endpoint_general": "endpoint_airy",
        "endpoint_constant": "endpoint_constant",
        "endpoint_constant_any": "endpoint_constant",
        "endpoint_positive": "endpoint_constant_positive",
        "endpoint_constant_positive": "endpoint_constant_positive",
        "endpoint_negative": "endpoint_constant_negative",
        "endpoint_constant_negative": "endpoint_constant_negative",
        "endpoint_zero": "endpoint_constant_zero",
        "endpoint_linear": "endpoint_constant_zero",
        "endpoint_constant_zero": "endpoint_constant_zero",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "basis_kind must be 'endpoint_auto', 'endpoint_airy', "
            "'endpoint_constant', 'endpoint_constant_positive', "
            "'endpoint_constant_negative', or 'endpoint_constant_zero'."
        ) from exc


def _constant_b0_if_usable(
    a: float,
    b: float,
    xL: float,
    xR: float,
    *,
    rel_tol: float = 1.0e-8,
) -> float | None:
    xmid = 0.5 * (xL + xR)
    b0 = float(a * xmid + b)
    variation = abs(float(a)) * abs(float(xR - xL))
    scale = max(1.0, abs(b0))
    if variation > rel_tol * scale:
        return None
    return b0


def _constant_positive_b0_if_usable(
    a: float,
    b: float,
    xL: float,
    xR: float,
    *,
    rel_tol: float = 1.0e-8,
    abs_tol: float = 1.0e-12,
) -> float | None:
    b0 = _constant_b0_if_usable(a, b, xL, xR, rel_tol=rel_tol)
    if b0 is None or b0 <= abs_tol:
        return None
    return b0


def _constant_basis_kind_for_b0(b0: float, *, zero_tol: float = 1.0e-14) -> str:
    if b0 > zero_tol:
        return "endpoint_constant_positive"
    if b0 < -zero_tol:
        return "endpoint_constant_negative"
    return "endpoint_constant_zero"


def _one_minus_exp_neg_twice(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -np.expm1(-2.0 * x)


def _stable_sinh_over_sinh(a: np.ndarray, L: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if L <= 50.0:
        return np.sinh(a) / np.sinh(L)
    den = max(float(_one_minus_exp_neg_twice(np.array([L]))[0]), np.finfo(float).tiny)
    return np.exp(a - L) * _one_minus_exp_neg_twice(a) / den


def _stable_cosh_over_sinh(a: np.ndarray, L: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if L <= 50.0:
        return np.cosh(a) / np.sinh(L)
    den = max(float(_one_minus_exp_neg_twice(np.array([L]))[0]), np.finfo(float).tiny)
    return np.exp(a - L) * (1.0 + np.exp(-2.0 * a)) / den


def _stable_sinh_sinh_over_sinh(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if L <= 50.0:
        return np.sinh(a) * np.sinh(b) / np.sinh(L)
    den = max(float(_one_minus_exp_neg_twice(np.array([L]))[0]), np.finfo(float).tiny)
    return (
        0.5
        * np.exp(np.minimum(a + b - L, 0.0))
        * _one_minus_exp_neg_twice(a)
        * _one_minus_exp_neg_twice(b)
        / den
    )


def _stable_sinh_cosh_over_sinh(a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if L <= 50.0:
        return np.sinh(a) * np.cosh(b) / np.sinh(L)
    den = max(float(_one_minus_exp_neg_twice(np.array([L]))[0]), np.finfo(float).tiny)
    return (
        0.5
        * np.exp(np.minimum(a + b - L, 0.0))
        * _one_minus_exp_neg_twice(a)
        * (1.0 + np.exp(-2.0 * b))
        / den
    )


def eval_endpoint_constant_positive_basis(
    b0: float,
    xL: float,
    xR: float,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    h = float(xR - xL)
    if h <= 0.0:
        raise ValueError("Endpoint basis requires xR > xL.")
    s = np.sqrt(float(b0))
    L = s * h
    left_dist = s * (x - xL)
    right_dist = s * (xR - x)

    phi1 = _stable_sinh_over_sinh(right_dist, L)
    phi2 = _stable_sinh_over_sinh(left_dist, L)
    dphi1 = -s * _stable_cosh_over_sinh(right_dist, L)
    dphi2 = s * _stable_cosh_over_sinh(left_dist, L)
    return phi1, phi2, dphi1, dphi2


def build_endpoint_constant_positive_particular_on_points(
    b0: float,
    xL: float,
    xR: float,
    x_pts: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_pts = np.asarray(x_pts, dtype=np.float64)
    if x_pts.ndim != 1:
        raise ValueError("x_pts must be a one-dimensional array.")
    if x_pts.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    if np.any((x_pts < xL - 1.0e-14) | (x_pts > xR + 1.0e-14)):
        raise ValueError("x_pts must lie inside [xL, xR].")
    if np.any(np.diff(x_pts) < -1.0e-15):
        raise ValueError("x_pts must be nondecreasing.")

    s = np.sqrt(float(b0))
    L = s * float(xR - xL)
    xi, wi = gauss_legendre(n_seg_gauss)
    up = np.zeros_like(x_pts)
    dup = np.zeros_like(x_pts)

    for j, x in enumerate(x_pts):
        x = float(x)
        if x > xL + 1.0e-15:
            sg, wg = map_to_interval(xi, wi, xL, x)
            fg = f_func(sg)
            a = s * (sg - xL)
            b = s * (xR - x)
            up[j] += np.sum(wg * (_stable_sinh_sinh_over_sinh(a, b, L) / s) * fg)
            dup[j] -= np.sum(wg * _stable_sinh_cosh_over_sinh(a, b, L) * fg)

        if x < xR - 1.0e-15:
            sg, wg = map_to_interval(xi, wi, x, xR)
            fg = f_func(sg)
            a = s * (x - xL)
            b = s * (xR - sg)
            up[j] += np.sum(wg * (_stable_sinh_sinh_over_sinh(a, b, L) / s) * fg)
            dup[j] += np.sum(wg * _stable_sinh_cosh_over_sinh(b, a, L) * fg)

    return up, dup


def build_endpoint_constant_nonpositive_particular_on_points(
    b0: float,
    xL: float,
    xR: float,
    x_pts: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
    b_tol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    x_pts = np.asarray(x_pts, dtype=np.float64)
    if x_pts.ndim != 1:
        raise ValueError("x_pts must be a one-dimensional array.")
    if x_pts.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    if np.any((x_pts < xL - 1.0e-14) | (x_pts > xR + 1.0e-14)):
        raise ValueError("x_pts must lie inside [xL, xR].")
    if np.any(np.diff(x_pts) < -1.0e-15):
        raise ValueError("x_pts must be nondecreasing.")
    if b0 > b_tol:
        raise ValueError("Nonpositive constant particular requires b0 <= 0 within tolerance.")

    dphi2_L = _eval_constant_endpoint_basis(
        b0=b0,
        xL=xL,
        xR=xR,
        x=np.array([xL], dtype=np.float64),
        b_tol=b_tol,
    )[3]
    dphi2_L_value = float(dphi2_L[0])
    if not np.isfinite(dphi2_L_value) or abs(dphi2_L_value) < 1.0e-15:
        raise ValueError("Endpoint constant Green function is singular or nearly singular.")
    green_scale = 1.0 / dphi2_L_value

    xi, wi = gauss_legendre(n_seg_gauss)
    up = np.zeros_like(x_pts)
    dup = np.zeros_like(x_pts)

    for j, x in enumerate(x_pts):
        x = float(x)
        x_arr = np.array([x], dtype=np.float64)
        phi1_x, phi2_x, dphi1_x, dphi2_x = _eval_constant_endpoint_basis(
            b0=b0,
            xL=xL,
            xR=xR,
            x=x_arr,
            b_tol=b_tol,
        )

        left_integral = 0.0
        right_integral = 0.0

        if x > xL + 1.0e-15:
            sg, wg = map_to_interval(xi, wi, xL, x)
            _p1g, p2g, _dp1g, _dp2g = _eval_constant_endpoint_basis(
                b0=b0,
                xL=xL,
                xR=xR,
                x=sg,
                b_tol=b_tol,
            )
            left_integral = float(np.sum(wg * p2g * f_func(sg)))

        if x < xR - 1.0e-15:
            sg, wg = map_to_interval(xi, wi, x, xR)
            p1g, _p2g, _dp1g, _dp2g = _eval_constant_endpoint_basis(
                b0=b0,
                xL=xL,
                xR=xR,
                x=sg,
                b_tol=b_tol,
            )
            right_integral = float(np.sum(wg * p1g * f_func(sg)))

        up[j] = green_scale * (phi1_x[0] * left_integral + phi2_x[0] * right_integral)
        dup[j] = green_scale * (dphi1_x[0] * left_integral + dphi2_x[0] * right_integral)

    up[np.isclose(x_pts, xL, atol=1.0e-14, rtol=1.0e-14)] = 0.0
    up[np.isclose(x_pts, xR, atol=1.0e-14, rtol=1.0e-14)] = 0.0
    return up, dup


def build_endpoint_constant_particular_on_points(
    b0: float,
    xL: float,
    xR: float,
    x_pts: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
    b_tol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    if b0 > b_tol:
        return build_endpoint_constant_positive_particular_on_points(
            b0=b0,
            xL=xL,
            xR=xR,
            x_pts=x_pts,
            f_func=f_func,
            n_seg_gauss=n_seg_gauss,
        )
    return build_endpoint_constant_nonpositive_particular_on_points(
        b0=b0,
        xL=xL,
        xR=xR,
        x_pts=x_pts,
        f_func=f_func,
        n_seg_gauss=n_seg_gauss,
        b_tol=b_tol,
    )


def _airy_positive_scale(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    scale = np.zeros_like(t)
    mask = t > 0.0
    scale[mask] = (2.0 / 3.0) * t[mask] * np.sqrt(t[mask])
    return scale


def _scaled_airy_real(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    eAi, eAip, eBi, eBip = (np.array(v, dtype=np.float64, copy=True) for v in airye(t))
    nonpositive = t <= 0.0
    if np.any(nonpositive):
        Ai, Aip, Bi, Bip = airy(t[nonpositive])
        eAi[nonpositive] = Ai
        eAip[nonpositive] = Aip
        eBi[nonpositive] = Bi
        eBip[nonpositive] = Bip
    return eAi, eAip, eBi, eBip


def _scaled_two_term_difference(
    c1: np.ndarray,
    e1: np.ndarray,
    c2: np.ndarray,
    e2: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return c1 * np.exp(e1 - scale) - c2 * np.exp(e2 - scale)


def _scaled_two_term_ratio(
    num1: np.ndarray,
    nexp1: np.ndarray,
    num2: np.ndarray,
    nexp2: np.ndarray,
    den1: float,
    dexp1: float,
    den2: float,
    dexp2: float,
) -> np.ndarray:
    nexp1 = np.asarray(nexp1, dtype=np.float64)
    nexp2 = np.asarray(nexp2, dtype=np.float64)
    scale = np.maximum(np.maximum(nexp1, nexp2), max(float(dexp1), float(dexp2)))
    numerator = _scaled_two_term_difference(num1, nexp1, num2, nexp2, scale)
    denominator = _scaled_two_term_difference(
        np.asarray(den1, dtype=np.float64),
        np.asarray(dexp1, dtype=np.float64),
        np.asarray(den2, dtype=np.float64),
        np.asarray(dexp2, dtype=np.float64),
        scale,
    )
    if np.any(np.abs(denominator) < 1.0e-15):
        raise ValueError("Endpoint Airy basis is singular or nearly singular on this element.")
    return numerator / denominator


def eval_endpoint_airy_basis(
    a: float,
    b: float,
    xL: float,
    xR: float,
    x: np.ndarray,
    a_tol: float = 1e-12,
    a_tol_rel: float = 1e-8,
    b_tol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    h = float(xR - xL)
    if h <= 0.0:
        raise ValueError("Endpoint Airy basis requires xR > xL.")

    xmid = 0.5 * (xL + xR)
    b0 = float(a * xmid + b)
    delta_c = abs(float(a)) * h
    if abs(a) <= a_tol or delta_c <= a_tol_rel * max(1.0, abs(b0)):
        return _eval_constant_endpoint_basis(b0=b0, xL=xL, xR=xR, x=x, b_tol=b_tol)

    alpha = np.cbrt(float(a))
    alpha2 = alpha * alpha
    t = alpha * x + (b / alpha2)
    tL = float(alpha * xL + (b / alpha2))
    tR = float(alpha * xR + (b / alpha2))

    eAi, eAip, eBi, eBip = _scaled_airy_real(t)
    eAiLR, eAipLR, eBiLR, eBipLR = _scaled_airy_real(np.array([tL, tR], dtype=np.float64))
    eAiL, eAiR = float(eAiLR[0]), float(eAiLR[1])
    eAipL, eAipR = float(eAipLR[0]), float(eAipLR[1])
    eBiL, eBiR = float(eBiLR[0]), float(eBiLR[1])
    eBipL, eBipR = float(eBipLR[0]), float(eBipLR[1])

    if (
        not np.all(np.isfinite(eAi))
        or not np.all(np.isfinite(eAip))
        or not np.all(np.isfinite(eBi))
        or not np.all(np.isfinite(eBip))
        or not np.all(np.isfinite(eAiLR))
        or not np.all(np.isfinite(eAipLR))
        or not np.all(np.isfinite(eBiLR))
        or not np.all(np.isfinite(eBipLR))
    ):
        raise ValueError("Endpoint Airy basis evaluation produced non-finite scaled Airy values.")

    xi = _airy_positive_scale(t)
    xiL = float(_airy_positive_scale(np.array([tL], dtype=np.float64))[0])
    xiR = float(_airy_positive_scale(np.array([tR], dtype=np.float64))[0])

    den1 = eAiL * eBiR
    den2 = eBiL * eAiR
    dexp1 = -xiL + xiR
    dexp2 = xiL - xiR

    phi1 = _scaled_two_term_ratio(
        eAi * eBiR,
        -xi + xiR,
        eBi * eAiR,
        xi - xiR,
        den1,
        dexp1,
        den2,
        dexp2,
    )
    phi2 = _scaled_two_term_ratio(
        eAiL * eBi,
        -xiL + xi,
        eBiL * eAi,
        xiL - xi,
        den1,
        dexp1,
        den2,
        dexp2,
    )
    dphi1 = alpha * _scaled_two_term_ratio(
        eAip * eBiR,
        -xi + xiR,
        eBip * eAiR,
        xi - xiR,
        den1,
        dexp1,
        den2,
        dexp2,
    )
    dphi2 = alpha * _scaled_two_term_ratio(
        eAiL * eBip,
        -xiL + xi,
        eBiL * eAip,
        xiL - xi,
        den1,
        dexp1,
        den2,
        dexp2,
    )

    left_mask = np.isclose(x, xL, atol=1e-14, rtol=1e-14)
    right_mask = np.isclose(x, xR, atol=1e-14, rtol=1e-14)
    phi1[left_mask] = 1.0
    phi2[left_mask] = 0.0
    phi1[right_mask] = 0.0
    phi2[right_mask] = 1.0

    if (
        not np.all(np.isfinite(phi1))
        or not np.all(np.isfinite(phi2))
        or not np.all(np.isfinite(dphi1))
        or not np.all(np.isfinite(dphi2))
    ):
        raise ValueError("Endpoint Airy basis evaluation produced non-finite values.")

    return phi1, phi2, dphi1, dphi2


def build_endpoint_airy_particular_on_points(
    a: float,
    b: float,
    xL: float,
    xR: float,
    x_pts: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_pts = np.asarray(x_pts, dtype=np.float64)
    if x_pts.ndim != 1:
        raise ValueError("x_pts must be a one-dimensional array.")
    if x_pts.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty
    if np.any((x_pts < xL - 1.0e-14) | (x_pts > xR + 1.0e-14)):
        raise ValueError("x_pts must lie inside [xL, xR].")
    if np.any(np.diff(x_pts) < -1.0e-15):
        raise ValueError("x_pts must be nondecreasing.")
    if not np.isclose(x_pts[0], xL, atol=1e-14, rtol=1e-14):
        raise ValueError("x_pts must start at xL.")
    if not np.isclose(x_pts[-1], xR, atol=1e-14, rtol=1e-14):
        raise ValueError("x_pts must end at xR.")

    h = float(xR - xL)
    xmid = 0.5 * (xL + xR)
    b0 = float(a * xmid + b)
    delta_c = abs(float(a)) * h
    if abs(a) <= 1.0e-12 or delta_c <= 1.0e-6 * max(1.0, abs(b0)):
        if b0 > 1.0e-14:
            return build_endpoint_constant_positive_particular_on_points(
                b0=b0,
                xL=xL,
                xR=xR,
                x_pts=x_pts,
                f_func=f_func,
                n_seg_gauss=n_seg_gauss,
            )

        # The non-positive constant case is not exponentially growing, so the
        # normalized Green representation is numerically harmless except at
        # the expected Dirichlet resonance points.
        phi1, phi2, dphi1, dphi2 = eval_endpoint_airy_basis(a, b, xL, xR, x_pts)
        dphi2_L = float(dphi2[0])
        if not np.isfinite(dphi2_L) or abs(dphi2_L) < 1.0e-15:
            raise ValueError("Endpoint Green function is singular or nearly singular.")
        green_scale = 1.0 / dphi2_L

        xi, wi = gauss_legendre(n_seg_gauss)
        left_integral = np.zeros_like(x_pts)
        right_integral = np.zeros_like(x_pts)

        for j in range(1, x_pts.size):
            s0 = float(x_pts[j - 1])
            s1 = float(x_pts[j])
            left_integral[j] = left_integral[j - 1]
            if np.isclose(s1, s0, atol=1e-15, rtol=1e-15):
                continue
            sg, wg = map_to_interval(xi, wi, s0, s1)
            _p1g, p2g, _dp1g, _dp2g = eval_endpoint_airy_basis(a, b, xL, xR, sg)
            left_integral[j] += np.sum(wg * p2g * f_func(sg))

        for j in range(x_pts.size - 2, -1, -1):
            s0 = float(x_pts[j])
            s1 = float(x_pts[j + 1])
            right_integral[j] = right_integral[j + 1]
            if np.isclose(s1, s0, atol=1e-15, rtol=1e-15):
                continue
            sg, wg = map_to_interval(xi, wi, s0, s1)
            p1g, _p2g, _dp1g, _dp2g = eval_endpoint_airy_basis(a, b, xL, xR, sg)
            right_integral[j] += np.sum(wg * p1g * f_func(sg))

        up = green_scale * (phi1 * left_integral + phi2 * right_integral)
        dup = green_scale * (dphi1 * left_integral + dphi2 * right_integral)
        return up, dup

    alpha = np.cbrt(float(a))
    alpha2 = alpha * alpha
    prefactor = np.pi / alpha

    def scaled_components(xx: np.ndarray) -> tuple[np.ndarray, ...]:
        tt = alpha * xx + (b / alpha2)
        eAi, eAip, eBi, eBip = _scaled_airy_real(tt)
        xi_vals = _airy_positive_scale(tt)
        if (
            not np.all(np.isfinite(eAi))
            or not np.all(np.isfinite(eAip))
            or not np.all(np.isfinite(eBi))
            or not np.all(np.isfinite(eBip))
        ):
            raise ValueError("Endpoint Airy Green evaluation produced non-finite scaled Airy values.")
        return eAi, eAip, eBi, eBip, xi_vals

    def scaled_diff(
        c1: np.ndarray,
        e1: np.ndarray,
        c2: np.ndarray,
        e2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        scale = np.maximum(e1, e2)
        value = _scaled_two_term_difference(c1, e1, c2, e2, scale)
        return value, scale

    def multiply_over_den(
        v1: np.ndarray,
        s1: np.ndarray,
        v2: np.ndarray,
        s2: np.ndarray,
    ) -> np.ndarray:
        exponent = s1 + s2 - den_scale
        return prefactor * (v1 * v2 / den_value) * np.exp(exponent)

    eAiLR, _eAipLR, eBiLR, _eBipLR, xiLR = scaled_components(np.array([xL, xR], dtype=np.float64))
    eAiL, eAiR = float(eAiLR[0]), float(eAiLR[1])
    eBiL, eBiR = float(eBiLR[0]), float(eBiLR[1])
    xiL, xiR = float(xiLR[0]), float(xiLR[1])

    den_value_arr, den_scale_arr = scaled_diff(
        np.array(eAiL * eBiR, dtype=np.float64),
        np.array(-xiL + xiR, dtype=np.float64),
        np.array(eBiL * eAiR, dtype=np.float64),
        np.array(xiL - xiR, dtype=np.float64),
    )
    den_value = float(den_value_arr)
    den_scale = float(den_scale_arr)
    if not np.isfinite(den_value) or abs(den_value) < 1.0e-15:
        raise ValueError("Endpoint Airy Green function is singular or nearly singular.")

    def zero_left(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eAi, _eAip, eBi, _eBip, xi = scaled_components(xx)
        return scaled_diff(eAiL * eBi, -xiL + xi, eBiL * eAi, xiL - xi)

    def zero_right(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eAi, _eAip, eBi, _eBip, xi = scaled_components(xx)
        return scaled_diff(eAi * eBiR, -xi + xiR, eBi * eAiR, xi - xiR)

    def dzero_left(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _eAi, eAip, _eBi, eBip, xi = scaled_components(xx)
        value, scale = scaled_diff(eAiL * eBip, -xiL + xi, eBiL * eAip, xiL - xi)
        return alpha * value, scale

    def dzero_right(xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _eAi, eAip, _eBi, eBip, xi = scaled_components(xx)
        value, scale = scaled_diff(eAip * eBiR, -xi + xiR, eBip * eAiR, xi - xiR)
        return alpha * value, scale

    xi, wi = gauss_legendre(n_seg_gauss)
    up = np.zeros_like(x_pts)
    dup = np.zeros_like(x_pts)

    for j, x in enumerate(x_pts):
        x_arr = np.array([float(x)], dtype=np.float64)
        zL_x, zL_x_scale = zero_left(x_arr)
        zR_x, zR_x_scale = zero_right(x_arr)
        dzL_x, dzL_x_scale = dzero_left(x_arr)
        dzR_x, dzR_x_scale = dzero_right(x_arr)

        if x > xL + 1.0e-15:
            sg, wg = map_to_interval(xi, wi, xL, float(x))
            zL_s, zL_s_scale = zero_left(sg)
            fg = f_func(sg)
            g = multiply_over_den(zL_s, zL_s_scale, zR_x[0], zR_x_scale[0])
            dg = multiply_over_den(zL_s, zL_s_scale, dzR_x[0], dzR_x_scale[0])
            up[j] += np.sum(wg * g * fg)
            dup[j] += np.sum(wg * dg * fg)

        if x < xR - 1.0e-15:
            sg, wg = map_to_interval(xi, wi, float(x), xR)
            zR_s, zR_s_scale = zero_right(sg)
            fg = f_func(sg)
            g = multiply_over_den(zL_x[0], zL_x_scale[0], zR_s, zR_s_scale)
            dg = multiply_over_den(dzL_x[0], dzL_x_scale[0], zR_s, zR_s_scale)
            up[j] += np.sum(wg * g * fg)
            dup[j] += np.sum(wg * dg * fg)

    up[np.isclose(x_pts, xL, atol=1e-14, rtol=1e-14)] = 0.0
    up[np.isclose(x_pts, xR, atol=1e-14, rtol=1e-14)] = 0.0
    return up, dup


def evaluate_endpoint_constant_positive_particular_at_points(
    b0: float,
    xL: float,
    xR: float,
    x_eval: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    if x_eval.ndim != 1:
        raise ValueError("x_eval must be a one-dimensional array.")
    if x_eval.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    x_sorted = np.unique(np.concatenate([np.array([xL, xR], dtype=np.float64), x_eval]))
    x_sorted.sort()
    up_sorted, dup_sorted = build_endpoint_constant_positive_particular_on_points(
        b0=b0,
        xL=xL,
        xR=xR,
        x_pts=x_sorted,
        f_func=f_func,
        n_seg_gauss=n_seg_gauss,
    )
    idx = np.searchsorted(x_sorted, x_eval)
    return up_sorted[idx], dup_sorted[idx]


def evaluate_endpoint_constant_particular_at_points(
    b0: float,
    xL: float,
    xR: float,
    x_eval: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    if x_eval.ndim != 1:
        raise ValueError("x_eval must be a one-dimensional array.")
    if x_eval.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    x_sorted = np.unique(np.concatenate([np.array([xL, xR], dtype=np.float64), x_eval]))
    x_sorted.sort()
    up_sorted, dup_sorted = build_endpoint_constant_particular_on_points(
        b0=b0,
        xL=xL,
        xR=xR,
        x_pts=x_sorted,
        f_func=f_func,
        n_seg_gauss=n_seg_gauss,
    )
    idx = np.searchsorted(x_sorted, x_eval)
    return up_sorted[idx], dup_sorted[idx]


def evaluate_endpoint_airy_particular_at_points(
    a: float,
    b: float,
    xL: float,
    xR: float,
    x_eval: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    if x_eval.ndim != 1:
        raise ValueError("x_eval must be a one-dimensional array.")
    if x_eval.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    x_sorted = np.unique(np.concatenate([np.array([xL, xR], dtype=np.float64), x_eval]))
    x_sorted.sort()
    up_sorted, dup_sorted = build_endpoint_airy_particular_on_points(
        a=a,
        b=b,
        xL=xL,
        xR=xR,
        x_pts=x_sorted,
        f_func=f_func,
        n_seg_gauss=n_seg_gauss,
    )
    idx = np.searchsorted(x_sorted, x_eval)
    return up_sorted[idx], dup_sorted[idx]


def _select_endpoint_basis_kind(
    requested_basis: str,
    a: float,
    b: float,
    xL: float,
    xR: float,
) -> tuple[str, float]:
    constant_b0 = _constant_b0_if_usable(a, b, xL, xR)
    constant_positive_b0 = _constant_positive_b0_if_usable(a, b, xL, xR)
    if requested_basis == "endpoint_constant_positive":
        if constant_positive_b0 is None:
            raise ValueError(
                "basis_kind='endpoint_constant_positive' requires every element "
                "to have an approximately constant positive coefficient."
            )
        return "endpoint_constant_positive", constant_positive_b0

    if requested_basis in {
        "endpoint_constant",
        "endpoint_constant_negative",
        "endpoint_constant_zero",
    }:
        if constant_b0 is None:
            raise ValueError(
                f"basis_kind={requested_basis!r} requires every element "
                "to have an approximately constant coefficient."
            )
        selected_kind = _constant_basis_kind_for_b0(constant_b0)
        if (
            requested_basis == "endpoint_constant_negative"
            and selected_kind != "endpoint_constant_negative"
        ):
            raise ValueError(
                "basis_kind='endpoint_constant_negative' requires every element "
                "to have an approximately constant negative coefficient."
            )
        if (
            requested_basis == "endpoint_constant_zero"
            and selected_kind != "endpoint_constant_zero"
        ):
            raise ValueError(
                "basis_kind='endpoint_constant_zero' requires every element "
                "to have an approximately zero coefficient."
            )
        return selected_kind, constant_b0

    if requested_basis == "endpoint_auto" and constant_b0 is not None:
        return _constant_basis_kind_for_b0(constant_b0), constant_b0

    return "endpoint_airy", 0.0


def _eval_endpoint_basis(
    basis_kind: str,
    basis_b0: float,
    a: float,
    b: float,
    xL: float,
    xR: float,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if basis_kind in {
        "endpoint_constant_positive",
        "endpoint_constant_negative",
        "endpoint_constant_zero",
    }:
        return _eval_constant_endpoint_basis(basis_b0, xL, xR, x, b_tol=1.0e-14)
    if basis_kind == "endpoint_airy":
        return eval_endpoint_airy_basis(a, b, xL, xR, x)
    raise ValueError(f"Unsupported endpoint basis kind={basis_kind!r}.")


def _build_zero_endpoint_particular_on_points(
    basis_kind: str,
    basis_b0: float,
    a: float,
    b: float,
    xL: float,
    xR: float,
    x_pts: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if basis_kind in {
        "endpoint_constant_positive",
        "endpoint_constant_negative",
        "endpoint_constant_zero",
    }:
        return build_endpoint_constant_particular_on_points(
            basis_b0, xL, xR, x_pts, f_func, n_seg_gauss=n_seg_gauss
        )
    if basis_kind == "endpoint_airy":
        return build_endpoint_airy_particular_on_points(
            a, b, xL, xR, x_pts, f_func, n_seg_gauss=n_seg_gauss
        )
    raise ValueError(f"Unsupported endpoint basis kind={basis_kind!r}.")


def _evaluate_zero_endpoint_particular_at_points(
    basis_kind: str,
    basis_b0: float,
    a: float,
    b: float,
    xL: float,
    xR: float,
    x_eval: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if basis_kind in {
        "endpoint_constant_positive",
        "endpoint_constant_negative",
        "endpoint_constant_zero",
    }:
        return evaluate_endpoint_constant_particular_at_points(
            basis_b0, xL, xR, x_eval, f_func, n_seg_gauss=n_seg_gauss
        )
    if basis_kind == "endpoint_airy":
        return evaluate_endpoint_airy_particular_at_points(
            a, b, xL, xR, x_eval, f_func, n_seg_gauss=n_seg_gauss
        )
    raise ValueError(f"Unsupported endpoint basis kind={basis_kind!r}.")


def _evaluate_element_components(
    el: ElementData,
    x_eval: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    n_seg_gauss: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi1, phi2, dphi1, dphi2 = _eval_endpoint_basis(
        basis_kind=el.basis_kind,
        basis_b0=el.basis_b0,
        a=el.a,
        b=el.b,
        xL=el.xL,
        xR=el.xR,
        x=x_eval,
    )
    up, dup = _evaluate_zero_endpoint_particular_at_points(
        basis_kind=el.basis_kind,
        basis_b0=el.basis_b0,
        a=el.a,
        b=el.b,
        xL=el.xL,
        xR=el.xR,
        x_eval=x_eval,
        f_func=f_func,
        n_seg_gauss=n_seg_gauss,
    )
    return phi1, phi2, dphi1, dphi2, up, dup


def build_elements(
    c_func: Callable[[np.ndarray], np.ndarray],
    f_func: Callable[[np.ndarray], np.ndarray],
    grid: np.ndarray | None = None,
    N: int | None = None,
    a: float = 0.0,
    b: float = 1.0,
    quad_n: int = 12,
    xI: float = 0.5,
    n_seg_gauss: int = 8,
    c_left_trace: Callable[[float], float] | None = None,
    c_right_trace: Callable[[float], float] | None = None,
    basis_kind: str = "endpoint_auto",
) -> Tuple[np.ndarray, List[ElementData]]:
    if grid is None:
        if N is None:
            raise ValueError("Either grid or N must be provided.")
        grid, _h, _interface_node = build_uniform_grid(N=N, a=a, b=b, xI=xI)
    else:
        grid = np.asarray(grid, dtype=np.float64)
        interface_node_from_grid(grid, xI)
        N = grid.size - 1

    if c_left_trace is None or c_right_trace is None:
        c_left_trace, c_right_trace = make_trace_functions(c_func, xI)

    requested_basis = _normalize_basis_kind(basis_kind)
    xi, wi = gauss_legendre(quad_n)

    elems: List[ElementData] = []
    for e in range(N):
        xL = float(grid[e])
        xR = float(grid[e + 1])

        a_lin, b_lin = linearize_c_on_element(xL, xR, c_left_trace, c_right_trace)
        element_basis, basis_b0 = _select_endpoint_basis_kind(requested_basis, a_lin, b_lin, xL, xR)

        xq, wq = map_to_interval(xi, wi, xL, xR)
        phi1_q, phi2_q, dphi1_q, dphi2_q = _eval_endpoint_basis(
            element_basis, basis_b0, a_lin, b_lin, xL, xR, xq
        )

        x_sorted = np.unique(np.concatenate([np.array([xL, xR]), xq]))
        x_sorted.sort()
        up_sorted, dup_sorted = _build_zero_endpoint_particular_on_points(
            element_basis,
            basis_b0,
            a_lin,
            b_lin,
            xL,
            xR,
            x_sorted,
            f_func,
            n_seg_gauss=n_seg_gauss,
        )

        up_L = float(up_sorted[0])
        dup_L = float(dup_sorted[0])
        up_R = float(up_sorted[-1])
        dup_R = float(dup_sorted[-1])

        phi1_L, phi2_L, dphi1_L, dphi2_L = _eval_endpoint_basis(
            element_basis, basis_b0, a_lin, b_lin, xL, xR, np.array([xL], dtype=np.float64)
        )
        phi1_R, phi2_R, dphi1_R, dphi2_R = _eval_endpoint_basis(
            element_basis, basis_b0, a_lin, b_lin, xL, xR, np.array([xR], dtype=np.float64)
        )

        idx_q = np.searchsorted(x_sorted, xq)
        up_q = up_sorted[idx_q]
        dup_q = dup_sorted[idx_q]

        elems.append(
            ElementData(
                xL=xL,
                xR=xR,
                h=xR - xL,
                a=a_lin,
                b=b_lin,
                xq=xq,
                wq=wq,
                phi1_q=phi1_q,
                phi2_q=phi2_q,
                dphi1_q=dphi1_q,
                dphi2_q=dphi2_q,
                up_q=up_q,
                dup_q=dup_q,
                phi1_L=float(phi1_L[0]),
                phi2_L=float(phi2_L[0]),
                dphi1_L=float(dphi1_L[0]),
                dphi2_L=float(dphi2_L[0]),
                up_L=up_L,
                dup_L=dup_L,
                phi1_R=float(phi1_R[0]),
                phi2_R=float(phi2_R[0]),
                dphi1_R=float(dphi1_R[0]),
                dphi2_R=float(dphi2_R[0]),
                up_R=up_R,
                dup_R=dup_R,
                basis_kind=element_basis,
                basis_b0=basis_b0,
            )
        )

    return grid, elems


def _find_element_index_for_points(
    elems: List[ElementData],
    x_eval: np.ndarray,
    side: str | None = None,
    xI: float | None = None,
) -> np.ndarray:
    xL_all = np.array([el.xL for el in elems], dtype=np.float64)
    xR_all = np.array([el.xR for el in elems], dtype=np.float64)

    idx = np.searchsorted(xR_all, x_eval, side="left")
    idx = np.clip(idx, 0, len(elems) - 1)

    if side is not None and xI is not None:
        left_mask = np.isclose(x_eval, xI, atol=1e-12, rtol=1e-12) & (side == "left")
        right_mask = np.isclose(x_eval, xI, atol=1e-12, rtol=1e-12) & (side == "right")
        idx[left_mask] = np.searchsorted(xR_all, xI, side="left")
        idx[right_mask] = np.searchsorted(xL_all, xI, side="right") - 1

    return idx


def evaluate_solution_fine(
    elems: List[ElementData],
    z: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    points_per_element: int = 200,
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    xs_list = []
    us_list = []

    for e, el in enumerate(elems):
        if e == 0:
            xx = np.linspace(el.xL, el.xR, points_per_element + 1)
        else:
            xx = np.linspace(el.xL, el.xR, points_per_element + 1)[1:]

        phi1, phi2, _dphi1, _dphi2, up, _dup = _evaluate_element_components(
            el=el,
            x_eval=xx,
            f_func=f_func,
            n_seg_gauss=n_seg_gauss,
        )

        c1 = z[2 * e]
        c2 = z[2 * e + 1]
        uu = c1 * phi1 + c2 * phi2 + up

        xs_list.append(xx)
        us_list.append(uu)

    xs = np.concatenate(xs_list)
    us = np.concatenate(us_list)
    return xs, us


def evaluate_tfpm_state_on_side(
    elems: List[ElementData],
    z: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    side: str,
    xI: float,
    n_seg_gauss: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    idx = _find_element_index_for_points(elems, x_eval, side=side, xI=xI)

    u = np.zeros_like(x_eval)
    du = np.zeros_like(x_eval)

    unique_idx = np.unique(idx)
    for e in unique_idx:
        mask = idx == e
        el = elems[e]
        xx = x_eval[mask]

        phi1, phi2, dphi1, dphi2, up, dup = _evaluate_element_components(
            el=el,
            x_eval=xx,
            f_func=f_func,
            n_seg_gauss=n_seg_gauss,
        )

        c1 = z[2 * e]
        c2 = z[2 * e + 1]

        u[mask] = c1 * phi1 + c2 * phi2 + up
        du[mask] = c1 * dphi1 + c2 * dphi2 + dup

    return u, du


def evaluate_tfpm_on_side(
    elems: List[ElementData],
    z: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    side: str,
    xI: float,
    n_seg_gauss: int = 8,
) -> np.ndarray:
    u, _du = evaluate_tfpm_state_on_side(
        elems=elems,
        z=z,
        f_func=f_func,
        x_eval=x_eval,
        side=side,
        xI=xI,
        n_seg_gauss=n_seg_gauss,
    )
    return u
