"""Fourier Time-Varying Grey Model — FTGM(1, 1, N).

Reference implementation of the model proposed by Ye, Xie, Boylan & Shang,
*"Forecasting seasonal demand for retail: A Fourier time-varying grey model"*,
International Journal of Forecasting (2024).

This is the **clean integral form** (paper Appendix A), which is mathematically
equivalent to the MATLAB reference but simpler to implement and read:

    1. Build the accumulated series  Y(t)               (grey-model smoothing).
    2. Estimate a(t), b(t) by ordinary least squares     (integral matching, Eq. 13-15).
    3. Solve the single first-order ODE                  (numerically, Eq. A.6):
           dy/dt = a(t)*y(t) + b(t),     y(t1) = b(t1)/(1 - a(t1))
       and recover demand as            x_hat(t) = a(t)*y(t) + b(t).

The estimation step is a plain linear regression; the only non-linearity (the ODE) is
handled by SciPy's ``solve_ivp`` (RK45), the same Runge-Kutta family as MATLAB's
``ode45`` recommended by the authors.

The class is deliberately free of any web/serialization concern: it consumes and
returns NumPy arrays so it can be unit-tested and reused in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from app.ftgm import fourier
from app.ftgm.exceptions import InsufficientDataError, ModelNotFittedError, SolverError

FloatArray = NDArray[np.float64]

# Equally-spaced aggregated series: one time unit per period (h = 1).
_TIME_STEP: float = 1.0
# Guard for the initial-value denominator (1 - a(t1)); avoids division blow-ups.
_DENOM_EPS: float = 1e-8


@dataclass(frozen=True)
class FTGMConfig:
    """Tunable knobs shared by a model instance.

    Attributes:
        period: Seasonal period ``T`` (e.g. 12 for monthly data). Drives ``w = 2*pi/T``.
        interval_z: z-score for the residual-based prediction interval (1.96 ≈ 95%).
        solver_rtol / solver_atol: relative / absolute tolerances for the ODE solver.
        max_growth_factor: forecasts above this multiple of the historical max are
            treated as a divergence (grey models can grow exponentially) and rejected.
        ridge: L2 penalty on the Fourier (harmonic) coefficients of a(t) and b(t) in the
            integral-matching regression, on column-standardised features. ``0`` is the
            paper's plain OLS. Short MYPE series (18–24 months) give 10+ parameters only
            ~20 equations, and plain OLS then overfits noise into huge harmonics whose ODE
            solution drifts or collapses to 0; a mild ridge keeps the seasonal shape and
            roughly doubled out-of-sample accuracy on our 24-month retail benchmark.
    """

    period: int = 12
    interval_z: float = 1.96
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9
    max_growth_factor: float = 50.0
    ridge: float = 0.0


@dataclass
class ForecastResult:
    """Output of :meth:`FTGM.predict`."""

    point: FloatArray
    lower: FloatArray
    upper: FloatArray


@dataclass
class FTGM:
    """An order-``N`` Fourier time-varying grey model.

    Typical use::

        model = FTGM(order=2, config=FTGMConfig(period=12)).fit(monthly_demand)
        result = model.predict(horizon=6)

    After :meth:`fit`, ``fitted_`` and ``residuals_`` expose the in-sample diagnostics
    used for metrics and prediction intervals.
    """

    order: int
    config: FTGMConfig = field(default_factory=FTGMConfig)

    # Learned state (populated by ``fit``).
    theta_: FloatArray | None = field(default=None, init=False)
    fitted_: FloatArray | None = field(default=None, init=False)
    residuals_: FloatArray | None = field(default=None, init=False)
    _train: FloatArray | None = field(default=None, init=False)
    _y_end: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("FTGM order must be >= 1")

    # ------------------------------------------------------------------ fit ----
    def fit(self, demand: FloatArray) -> "FTGM":
        """Estimate the model parameters from an aggregated, equally-spaced series.

        Args:
            demand: 1-D array of observed demand per period (already aggregated).

        Raises:
            InsufficientDataError: if the series is too short to identify the order.
        """
        x = np.asarray(demand, dtype=np.float64).ravel()
        m = x.size
        required = fourier.n_params(self.order)
        # We have (m - 1) equations (one per period from t2..tm).
        if m - 1 < required:
            raise InsufficientDataError(
                f"order {self.order} needs at least {required + 1} observations, got {m}"
            )

        omega = fourier.angular_frequency(self.config.period)
        t = self._time_grid(m)

        # Accumulated (grey) series Y(t): smooths noise, exposes the growth law.
        y = np.cumsum(x)  # weights are all 1 for an equally-spaced series (h = 1)

        # z_k = x(t1) + integral approx = 0.5*y_{k-1} + 0.5*y_k + (h/2)*x_1   (k = 2..m)
        z = 0.5 * y[:-1] + 0.5 * y[1:] + (_TIME_STEP / 2.0) * x[0]

        # Integral matching: x = Xi(theta) -> ordinary least squares (Eq. 14-15).
        design = fourier.design_matrix(t[1:], z, self.order, omega)
        theta = self._estimate(design, x[1:])

        self.theta_ = theta
        self._train = x
        # In-sample fit + residuals (drive metrics and the prediction interval).
        fitted, y_path = self._integrate(t, theta, omega)
        self._y_end = float(y_path[-1])
        # Reject in-sample divergence: an order that cannot even reproduce the history
        # without exploding is unusable. Order selection skips it; the service then
        # falls back to the baseline. Grey models are exponential, so this guards
        # against overfit parameters with a strongly positive development coefficient.
        self._assert_stable(fitted, reference_max=float(np.max(np.abs(x))) or 1.0)
        self.fitted_ = fitted
        self.residuals_ = x - fitted
        return self

    # -------------------------------------------------------------- predict ----
    def predict(self, horizon: int) -> ForecastResult:
        """Forecast the next ``horizon`` periods with a residual-based interval."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.theta_ is None or self._train is None or self.residuals_ is None:
            raise ModelNotFittedError("call fit() before predict()")

        m = self._train.size
        omega = fourier.angular_frequency(self.config.period)
        # Continue the ODE solution from the end of the history (same trajectory as
        # integrating history + future in one go, at a fraction of the cost).
        future_grid = self._time_grid(m + horizon)[m - 1 :]
        x_hat, _ = self._integrate(future_grid, self.theta_, omega, y_start=self._y_end)
        point = x_hat[1:]

        # Sanity guard: grey models are exponential, so reject divergence.
        self._assert_stable(point, reference_max=float(np.max(np.abs(self._train))) or 1.0)
        point = np.clip(point, 0.0, None)  # demand cannot be negative
        lower, upper = self._prediction_interval(point)
        return ForecastResult(point=point, lower=lower, upper=upper)

    def fit_predict(self, demand: FloatArray, horizon: int) -> ForecastResult:
        """Convenience: :meth:`fit` then :meth:`predict`."""
        return self.fit(demand).predict(horizon)

    # ------------------------------------------------------------- internals ---
    def _estimate(self, design: FloatArray, target: FloatArray) -> FloatArray:
        """OLS (paper) or ridge on the harmonic coefficients; ``a0``/``b0`` are never shrunk."""
        if self.config.ridge <= 0:
            theta_raw, *_ = np.linalg.lstsq(design, target, rcond=None)
            return np.asarray(theta_raw, dtype=np.float64)
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0.0] = 1.0
        std = design / scale
        # Ridge as an augmented least-squares system (never singular, unlike the normal
        # equations): a0 (development coefficient) and b0 (level) are not penalised.
        weights = np.full(design.shape[1], np.sqrt(self.config.ridge))
        weights[:2] = 0.0
        aug_x = np.vstack([std, np.diag(weights)])
        aug_y = np.concatenate([target, np.zeros(design.shape[1])])
        theta_std, *_ = np.linalg.lstsq(aug_x, aug_y, rcond=None)
        return np.asarray(theta_std / scale, dtype=np.float64)

    @staticmethod
    def _time_grid(n: int) -> FloatArray:
        """Time instants t1..tn for an equally-spaced series (1-based, step = 1)."""
        return np.arange(1, n + 1, dtype=np.float64) * _TIME_STEP

    def _integrate(
        self, t_eval: FloatArray, theta: FloatArray, omega: float, *, y_start: float | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Solve dy/dt = a(t)y + b(t) and recover x_hat(t) = a(t)y(t) + b(t).

        Returns ``(x_hat, y)`` on ``t_eval``. ``y_start`` continues an existing solution
        from ``t_eval[0]``; otherwise the paper's initial value is used.
        """
        a_full, b_full = fourier.evaluate_parameters(t_eval, theta, self.order, omega)

        if y_start is not None:
            y0 = float(y_start)
        else:
            # Initial value (paper Eq. A.6): x(t1) = b(t1) / (1 - a(t1)).
            denom = 1.0 - a_full[0]
            if abs(denom) < _DENOM_EPS:
                denom = np.copysign(_DENOM_EPS, denom) if denom != 0.0 else _DENOM_EPS
            y0 = float(b_full[0] / denom)

        # Scalar right-hand side (same maths as fourier.evaluate_parameters, without
        # building matrices on every solver step — the solver calls it thousands of times).
        n = self.order
        freqs = np.arange(1, n + 1, dtype=np.float64) * omega
        a0, b0 = float(theta[0]), float(theta[1])
        a_cos, a_sin = theta[2 : 2 + 2 * n : 2], theta[3 : 2 + 2 * n : 2]
        b_cos, b_sin = theta[2 + 2 * n :: 2], theta[3 + 2 * n :: 2]

        def rhs(t: float, y: FloatArray) -> list[float]:
            c, s = np.cos(freqs * t), np.sin(freqs * t)
            a = a0 + float(a_cos @ c + a_sin @ s)
            b = b0 + float(b_cos @ c + b_sin @ s)
            return [a * y[0] + b]

        solution = solve_ivp(
            rhs,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=[y0],
            t_eval=t_eval,
            method="RK45",
            rtol=self.config.solver_rtol,
            atol=self.config.solver_atol,
        )
        if not solution.success:
            raise SolverError(f"ODE integration failed: {solution.message}")

        y_series = solution.y[0]
        return np.asarray(a_full * y_series + b_full, dtype=np.float64), np.asarray(y_series, dtype=np.float64)

    def _assert_stable(self, values: FloatArray, reference_max: float) -> None:
        """Raise :class:`SolverError` if the integrated series is non-finite or diverged."""
        if not np.all(np.isfinite(values)) or float(np.max(np.abs(values))) > (
            self.config.max_growth_factor * reference_max
        ):
            raise SolverError(
                "FTGM integration diverged; series unsuitable for this Fourier order"
            )

    def _prediction_interval(self, point: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Residual-based interval that widens with the horizon.

        The FTGM has no native predictive distribution, so we use a simple, defensible
        heuristic: a band of ``z * sigma * sqrt(step)`` around the point forecast, where
        ``sigma`` is the in-sample residual standard deviation. The ``sqrt(step)`` term
        reflects compounding uncertainty further into the future. Lower bound clipped at 0.
        """
        assert self.residuals_ is not None
        sigma = float(np.std(self.residuals_, ddof=1)) if self.residuals_.size > 1 else 0.0
        steps = np.arange(1, point.size + 1, dtype=np.float64)
        half_width = self.config.interval_z * sigma * np.sqrt(steps)
        lower = np.clip(point - half_width, 0.0, None)
        upper = point + half_width
        return lower, upper
