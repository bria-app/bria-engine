"""Fourier building blocks for the time-varying parameters of the FTGM.

The FTGM lets the grey-model parameters breathe over the season by writing them as a
finite Fourier series (Eqs. 6-7 in Ye et al., 2024):

    a(t) = a0 + sum_{n=1..N} ( a1n*cos(n*w*t) + a2n*sin(n*w*t) )      # development coeff.
    b(t) = b0 + sum_{n=1..N} ( b1n*cos(n*w*t) + b2n*sin(n*w*t) )      # forcing term

``a(t)`` is the (time-varying) growth rate and ``b(t)`` the (time-varying) baseline push.
This module only knows how to build the harmonic features and how to turn a flat
parameter vector ``theta`` back into ``a(t)`` / ``b(t)`` — the estimation of ``theta``
lives in :mod:`app.ftgm.model`.

Parameter layout of ``theta`` (length ``2 + 4*N``), matching the paper:
    [ a0, b0, a11, a21, a12, a22, ..., a1N, a2N, b11, b21, ..., b1N, b2N ]
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def n_params(order: int) -> int:
    """Number of free parameters of an order-``N`` FTGM: ``a0, b0`` plus 4 per harmonic."""
    return 2 + 4 * order


def harmonics(t: FloatArray, order: int, omega: float) -> FloatArray:
    """Stack the cos/sin harmonics for each time instant.

    Returns an ``(len(t), 2*order)`` matrix with columns ordered as
    ``[cos(1wt), sin(1wt), cos(2wt), sin(2wt), ...]`` so that it lines up with the
    ``(a1n, a2n)`` / ``(b1n, b2n)`` pairs in ``theta``.
    """
    if order < 1:
        return np.empty((t.shape[0], 0), dtype=np.float64)
    n = np.arange(1, order + 1, dtype=np.float64)
    angles = omega * np.outer(t, n)  # (len(t), order)
    cols = np.empty((t.shape[0], 2 * order), dtype=np.float64)
    cols[:, 0::2] = np.cos(angles)
    cols[:, 1::2] = np.sin(angles)
    return cols


def design_matrix(t: FloatArray, z: FloatArray, order: int, omega: float) -> FloatArray:
    """Build the integral-matching design matrix ``Xi`` (Eq. 13-14).

    Each row encodes ``x(t_k) = a(t_k)*z_k + b(t_k)`` expanded over the Fourier basis,
    which is *linear in the parameters* — hence ordinary least squares can estimate it.

    Columns (matching ``theta``):
        [ z, 1, {cos(n w t)*z, sin(n w t)*z}_n, {cos(n w t), sin(n w t)}_n ]
    """
    h = harmonics(t, order, omega)  # (m, 2*order)
    ones = np.ones((t.shape[0], 1), dtype=np.float64)
    z_col = z.reshape(-1, 1)
    a_block = h * z_col  # harmonics that modulate the development coefficient a(t)
    b_block = h  # harmonics of the forcing term b(t)
    return np.hstack([z_col, ones, a_block, b_block])


def evaluate_parameters(
    t: FloatArray, theta: FloatArray, order: int, omega: float
) -> tuple[FloatArray, FloatArray]:
    """Reconstruct the continuous functions ``a(t)`` and ``b(t)`` from ``theta``."""
    a0 = theta[0]
    b0 = theta[1]
    a_coef = theta[2 : 2 + 2 * order]
    b_coef = theta[2 + 2 * order : 2 + 4 * order]
    h = harmonics(t, order, omega)
    a = a0 + h @ a_coef
    b = b0 + h @ b_coef
    return a, b


def angular_frequency(period: int) -> float:
    """Seasonal angular frequency ``w = 2*pi / T`` for a seasonal period ``T``."""
    if period <= 0:
        raise ValueError("seasonal period must be a positive integer")
    return 2.0 * np.pi / float(period)
