"""P&L transformation layer (spec section 8.2).

Turns a matrix of option *values* - typically the ``(len(vol_range),
len(spot_range))`` grid produced by :func:`src.grid.price_grid` - into the
mark-to-market profit and loss of a LONG position bought at a known fill
price, and summarizes that matrix for the UI captions.

Pure, side-effect free, fully NumPy-vectorized, and free of any UI imports:
like the rest of ``src/`` this module is importable from a bare REPL.

Transformation
--------------
    pnl[i, j] = (value_grid[i, j] - purchase_price) * multiplier

Long positions only in v1 (a short position would simply negate the sign).
A ``purchase_price`` of 0 makes the P&L numerically identical to the value
matrix, which is the app's non-breaking default (test T-9).

Units and conventions
---------------------
value_grid     : option values in CURRENCY UNITS PER SHARE, the same units the
                 pricing engine returns.  Cells are mark-to-market values at
                 the current time to expiry (T in YEARS), NOT expiry payoffs.
purchase_price : the fill price actually paid, per share, in the same currency
                 units, >= 0.
multiplier     : contract multiplier, 1 (quoted per share) or 100 (per standard
                 equity option contract).  It scales P&L only; it never
                 touches the per-share purchase price.

Orientation is irrelevant here: the transform is elementwise, so whatever
row/column convention the caller used survives untouched.  The ascending-vol
row order is only flipped for display, in ``plotting.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Shared return-shape convention for the whole package: arrays stay arrays,
# 0-d results collapse to a ``np.float64`` scalar (which supports round(),
# float() and f"{x:.2f}", whereas a 0-d array does not).
from .black_scholes import _finalize

__all__ = [
    "VALID_MULTIPLIERS",
    "pnl_grid",
    "pnl_summary",
]

#: Contract multipliers offered in the UI: 1 = per share, 100 = per contract.
VALID_MULTIPLIERS: tuple[int, ...] = (1, 100)


# ---------------------------------------------------------------------------
# internal validation helpers
# ---------------------------------------------------------------------------
def _validate_purchase_price(purchase_price: Any) -> float:
    """Return ``purchase_price`` as a float, or raise a plain-language ValueError."""
    if (
        isinstance(purchase_price, (bool, str, bytes))
        or np.ndim(purchase_price) != 0
    ):
        raise ValueError(
            "Purchase price must be a single number - the price you paid per "
            "share, for example 10.45."
        )
    try:
        value = float(purchase_price)
    except (TypeError, ValueError):
        raise ValueError(
            "Purchase price must be a number - the price you paid per share, "
            "for example 10.45."
        ) from None

    if not np.isfinite(value):
        raise ValueError(
            "Purchase price must be a real, finite number - it cannot be "
            "blank, NaN or infinite."
        )
    if value < 0.0:
        raise ValueError(
            "Purchase price cannot be negative. Enter what you paid per share, "
            "or 0 to show the option value itself."
        )
    return value


def _validate_multiplier(multiplier: Any) -> float:
    """Return ``multiplier`` as a float, or raise a plain-language ValueError."""
    message = "Contract multiplier must be 1 (per share) or 100 (per contract)."
    if isinstance(multiplier, (bool, str, bytes)) or np.ndim(multiplier) != 0:
        raise ValueError(message)
    try:
        value = float(multiplier)
    except (TypeError, ValueError):
        raise ValueError(message) from None

    if not np.isfinite(value) or value not in (1.0, 100.0):
        raise ValueError(message)
    return value


def _as_finite_values(value_grid: ArrayLike, label: str) -> NDArray[np.float64]:
    """Coerce ``value_grid`` to a float64 array, rejecting non-finite cells.

    ``label`` names the field in the user-facing error message.
    """
    try:
        values = np.asarray(value_grid, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be numbers - pass the matrix produced by the "
            f"pricing engine."
        ) from None

    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(
            f"{label} must all be real, finite numbers - no blank, NaN or "
            f"infinite cells."
        )
    return values


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------
def pnl_grid(
    value_grid: ArrayLike,
    purchase_price: float,
    multiplier: int = 1,
) -> NDArray[np.float64]:
    """Transform a matrix of option values into position P&L (spec 8.2).

    ``pnl = (value_grid - purchase_price) * multiplier``, applied elementwise
    in one vectorized expression - never a loop over cells.

    Parameters
    ----------
    value_grid : scalar or array of option values in CURRENCY UNITS PER SHARE,
        of any shape (the heat-map grid is ``(n_vol, n_spot)``, ascending-vol
        first).  All cells must be finite.
    purchase_price : the fill price paid PER SHARE, in the same currency units,
        >= 0.  0 leaves the matrix unchanged in value terms (test T-9).
    multiplier : contract multiplier, either 1 (per share) or 100 (per
        contract).  Defaults to 1.

    Returns
    -------
    np.ndarray
        A NEW array of the same shape as ``value_grid`` holding P&L in currency
        units (a ``np.float64`` scalar for scalar input).  The input is never
        mutated and never returned by reference, even in the identity case
        ``purchase_price = 0, multiplier = 1``.

    Raises
    ------
    ValueError
        If ``purchase_price`` is negative, non-finite or not a single number;
        if ``multiplier`` is anything other than 1 or 100; or if ``value_grid``
        holds non-numeric or non-finite cells.  Messages are written for an end
        user, so the UI can surface them verbatim (NFR-2).

    Notes
    -----
    Long positions only in v1.  P&L marks the position at the CURRENT time to
    expiry, not held to expiration.

    Examples
    --------
    >>> pnl_grid([[10.0, 12.0], [8.0, 9.0]], 10.0, 100)
    array([[   0.,  200.],
           [-200., -100.]])
    """
    price_paid = _validate_purchase_price(purchase_price)
    scale = _validate_multiplier(multiplier)
    values = _as_finite_values(value_grid, "Option values")

    # Out-of-place arithmetic: a brand-new array every time, input untouched.
    with np.errstate(over="ignore", invalid="ignore"):
        result = (values - price_paid) * scale
    if not np.all(np.isfinite(result)):
        raise ValueError("Position P&L is too large to calculate. Reduce the option or purchase price.")
    return _finalize(result)


def pnl_summary(pnl: ArrayLike) -> dict[str, Any]:
    """Summarize a P&L matrix for the UI summary captions.

    Parameters
    ----------
    pnl : scalar or array of P&L values in currency units, as returned by
        :func:`pnl_grid`.  All cells must be finite.

    Returns
    -------
    dict
        ``max_profit`` : float
            The best cell in the matrix, signed.  Negative when every scenario
            loses money, i.e. there is no profit anywhere on the grid.
        ``max_loss`` : float
            The worst cell in the matrix, signed - a loss is NEGATIVE, so the
            caller prints it with its own sign.  Positive when every scenario
            makes money.
        ``pct_profitable`` : float
            Share of cells strictly greater than 0, as a PERCENTAGE in 0..100
            (not a 0..1 fraction).  Cells at exactly 0 are break-even, not
            profitable, so they do not count.
        ``breakeven_exists`` : bool
            True when the matrix straddles zero, i.e. it holds at least one
            strictly positive AND one strictly negative cell - exactly when a
            break-even contour can be drawn through it.

    Raises
    ------
    ValueError
        If ``pnl`` is empty, or holds non-numeric or non-finite cells.

    Examples
    --------
    >>> s = pnl_summary([[-1.0, 0.0], [2.0, 4.0]])
    >>> s["max_profit"], s["max_loss"], s["pct_profitable"], s["breakeven_exists"]
    (4.0, -1.0, 50.0, True)
    """
    values = _as_finite_values(pnl, "P&L values")
    if values.size == 0:
        raise ValueError("There is no P&L to summarize - the scenario grid is empty.")

    profitable = values > 0.0
    losing = values < 0.0

    return {
        "max_profit": float(np.max(values)),
        "max_loss": float(np.min(values)),
        "pct_profitable": float(100.0 * np.count_nonzero(profitable) / values.size),
        "breakeven_exists": bool(np.any(profitable) and np.any(losing)),
    }
