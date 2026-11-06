"""Domain exceptions for the FTGM engine.

These are raised by the pure model layer (``app.ftgm``) and translated into HTTP
responses at the presentation boundary. Keeping them here means the numerical code
never needs to know about FastAPI or pydantic.
"""
from __future__ import annotations


class FTGMError(Exception):
    """Base class for every error raised by the FTGM model layer."""


class InsufficientDataError(FTGMError):
    """The series is too short to identify the model for the requested order.

    The FTGM needs at least ``2 + 4 * N`` observations to estimate the parameters of
    an order-``N`` model (plus a margin for validation). Raised when a series cannot
    support even the smallest model.
    """


class ModelNotFittedError(FTGMError):
    """Raised when ``predict`` is called before ``fit``."""


class SolverError(FTGMError):
    """The ODE solver failed or produced non-finite values.

    Grey models are exponential by nature, so a poorly conditioned series can make the
    integration diverge. We surface this explicitly instead of returning ``NaN``.
    """
