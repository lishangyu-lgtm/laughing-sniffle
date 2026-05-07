from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterfaceLifting:
    a: float
    b: float
    x_interface: float
    left_value: float
    right_value: float
    left_slope: float
    right_slope: float

    @classmethod
    def from_jump_data(
        cls,
        a: float,
        b: float,
        x_interface: float,
        left_bc: float,
        right_bc: float,
        jump_u: float,
        jump_du: float,
    ) -> "InterfaceLifting":
        total_length = float(b - a)
        if total_length <= 0.0:
            raise ValueError("Require b > a.")

        right_length = float(b - x_interface)
        left_slope = (right_bc - left_bc - jump_u - jump_du * right_length) / total_length
        right_slope = left_slope + jump_du
        left_value = left_bc + left_slope * (x_interface - a)
        right_value = left_value + jump_u
        return cls(
            a=float(a),
            b=float(b),
            x_interface=float(x_interface),
            left_value=float(left_value),
            right_value=float(right_value),
            left_slope=float(left_slope),
            right_slope=float(right_slope),
        )

    def evaluate(self, x: np.ndarray, side: str | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        left_values = self.left_value + self.left_slope * (x - self.x_interface)
        right_values = self.right_value + self.right_slope * (x - self.x_interface)

        if side == "left":
            return left_values
        if side == "right":
            return right_values
        if side is not None:
            raise ValueError("side must be None, 'left', or 'right'.")

        return np.where(x <= self.x_interface, left_values, right_values)


def derivative_weights(x0: float, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 1:
        raise ValueError("points must be one-dimensional.")
    if points.size < 2:
        raise ValueError("Need at least two points for a derivative stencil.")

    shifted = points - float(x0)
    vandermonde = np.vstack([shifted**k for k in range(points.size)])
    rhs = np.zeros(points.size, dtype=np.float64)
    rhs[1] = 1.0
    return np.linalg.solve(vandermonde, rhs)
