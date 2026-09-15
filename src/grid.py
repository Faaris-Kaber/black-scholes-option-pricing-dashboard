"""Vectorized scenario-matrix construction for the sensitivity heat maps (spec 7).

This module turns two 1-D axes - a spot-price range and a volatility range -
into the rectangular price surface the plotting layer draws.

ORIENTATION (the plotting layer depends on this; read before changing anything)
-------------------------------------------------------------------------------
``price_grid`` returns an array of shape ``(len(vol_range), len(spot_range))``::

    values[i, j] == price(S=spot_range[j], sigma=vol_range[i], ...)

    * ROW    index ``i`` -> ``vol_range[i]``   (volatility / y-axis)
    * COLUMN index ``j`` -> ``spot_range[j]``  (spot price / x-axis)

Both ranges are ASCENDING, therefore **row 0 is the LOWEST volatility** and
column 0 is the lowest spot price.  That is exactly the orientation
``np.meshgrid(spot_range, vol_range)`` produces with its default ``"xy"``
indexing, so nothing here is transposed - and nothing downstream should
transpose it either.

Spec section 7 requires volatility to be DISPLAYED ascending bottom -> top.
Storage order and display order therefore disagree by a vertical flip, and
**that flip happens in plotting.py only** (e.g. ``values[::-1, :]``).  Never
store a flipped grid: the rest of the app - P&L transforms, summaries, tests -
assumes ascending-vol-first row order.

WHAT IS HELD FIXED
------------------
``price_grid`` shocks ONLY two variables, spot ``S`` and volatility ``sigma``.
Strike ``K``, time to expiry ``T`` and the risk-free rate ``r`` are scalars,
held constant across every cell of the matrix.  In particular ``T`` does NOT
decay along either axis, so the surface is a MARK-TO-MARKET valuation at the
current, fixed time to expiry - it is *not* an expiry payoff diagram.  A cell
answers "what is this option worth right now if spot and implied vol jumped
here", not "what will this option be worth at expiration".  The UI must say so
in a caption; spec section 7 calls this the single most common misreading.

UNITS (identical to :mod:`src.black_scholes`)
---------------------------------------------
spot_range : currency units, every value > 0.
vol_range  : annualized volatility as a DECIMAL (0.20 == 20%, never 20),
             floored at ``MIN_VOL`` = 0.01 per spec section 6.2.
K          : strike in currency units, > 0.
T          : time to expiry in YEARS (not days), >= 0.
r          : risk-free rate as a DECIMAL per year (0.05 == 5%, never 5).

Pure functions, no side effects, no global mutable state, no UI imports.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.black_scholes import OptionType, price, validate_inputs

__all__ = [
    "MIN_VOL",
    "MIN_GRID_N",
    "MAX_GRID_N",
    "SPEC_MIN_GRID_N",
    "SPEC_MAX_GRID_N",
    "DEFAULT_GRID_N",
    "make_spot_range",
    "make_vol_range",
    "price_grid",
]

#: Volatility floor from spec section 6.2 - the vol slider starts at 0.01.
#: ``sigma = 0`` is legal in the engine but useless on a heat map (the whole
#: row collapses to the deterministic payoff), so the axis builder floors it.
MIN_VOL: float = 0.01

#: Hard minimum number of grid points: a linspace needs two ends to span a range.
MIN_GRID_N: int = 2
#: Hard maximum - a sanity cap well above anything the UI offers; annotated
#: heat maps stop being readable long before this.
MAX_GRID_N: int = 200

#: The range spec section 7 advertises in the UI ("10 x 10, configurable 5-15").
#: These are documentation, not enforcement: an ``n`` outside 5-15 but >= 2 is
#: accepted so tests and notebooks are not artificially constrained.
SPEC_MIN_GRID_N: int = 5
SPEC_MAX_GRID_N: int = 15
#: Default grid size from spec section 7.
DEFAULT_GRID_N: int = 10


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _validate_n(n: int, axis_label: str) -> int:
    """Return ``n`` as an int, or raise a plain-language ``ValueError``.

    Accepts any whole number >= :data:`MIN_GRID_N` (spec section 7 advertises
    5-15; smaller grids are still mathematically well defined and are useful in
    tests).  Booleans and non-integral floats are rejected.
    """
    if isinstance(n, bool):
        raise ValueError(
            f"{axis_label} grid size must be a whole number, not True/False."
        )
    if isinstance(n, (int, np.integer)):
        n_int = int(n)
    elif isinstance(n, (float, np.floating)) and np.isfinite(n) and float(n).is_integer():
        n_int = int(n)
    else:
        raise ValueError(
            f"{axis_label} grid size must be a whole number (received {n!r})."
        )

    if n_int < MIN_GRID_N:
        raise ValueError(
            f"{axis_label} grid size must be at least {MIN_GRID_N} "
            f"(received {n_int}); the heat map needs at least two points per "
            f"axis to span a range."
        )
    if n_int > MAX_GRID_N:
        raise ValueError(
            f"{axis_label} grid size must be at most {MAX_GRID_N} "
            f"(received {n_int})."
        )
    return n_int


def _validate_bounds(low: float, high: float, axis_label: str) -> tuple[float, float]:
    """Coerce ``low``/``high`` to finite floats with ``high > low``, else raise."""
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError):
        raise ValueError(f"{axis_label} range endpoints must be numbers.") from None

    if not np.isfinite(low_f) or not np.isfinite(high_f):
        raise ValueError(
            f"{axis_label} range endpoints must be real, finite numbers - "
            f"they cannot be blank, NaN or infinite."
        )
    if high_f <= low_f:
        raise ValueError(
            f"{axis_label} range maximum ({high_f:g}) must be greater than the "
            f"minimum ({low_f:g})."
        )
    return low_f, high_f


def _as_ascending_axis(values: ArrayLike, axis_label: str) -> NDArray[np.float64]:
    """Validate a heat-map axis: 1-D, non-empty, finite, ascending. Returns float64.

    The returned array is always a fresh float64 array, so ``price_grid`` can
    neither alias nor mutate caller state (NFR-3, pure functions).
    """
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{axis_label} must be a sequence of numbers.") from None

    arr = np.atleast_1d(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"{axis_label} must be a one-dimensional sequence of values "
            f"(received an array with shape {arr.shape})."
        )
    if arr.size == 0:
        raise ValueError(f"{axis_label} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{axis_label} must contain only real, finite numbers (no NaN or "
            f"infinite values)."
        )
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(
            f"{axis_label} must be strictly ascending, without duplicate values. The heat-map grid "
            f"is stored ascending-first (row 0 = lowest volatility, column 0 = "
            f"lowest spot); the vertical flip for display happens in the "
            f"plotting layer."
        )
    return arr


def _require_scalar(value: ArrayLike, label: str) -> float:
    """Return ``value`` as a float, rejecting arrays - K, T and r are held fixed."""
    if np.ndim(value) != 0:
        raise ValueError(
            f"{label} must be a single number: it is held fixed across every "
            f"cell of the heat map. Only spot and volatility vary."
        )
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a single finite number.") from None


# ---------------------------------------------------------------------------
# axis builders
# ---------------------------------------------------------------------------
def make_spot_range(low: float, high: float, n: int) -> NDArray[np.float64]:
    """Build the heat map's ascending spot-price axis (x-axis, COLUMNS).

    Parameters
    ----------
    low, high : float
        Range endpoints in currency units. Both must be > 0 (the engine's
        domain for ``S``) and ``high`` must be strictly greater than ``low``.
    n : int
        Number of points, >= :data:`MIN_GRID_N` (2). Spec section 7 advertises
        5-15 with a default of 10; anything from 2 to :data:`MAX_GRID_N` is
        accepted here so tests and notebooks are not artificially constrained.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, float64, strictly ASCENDING, from ``low`` to ``high``
        inclusive of both endpoints (``np.linspace``).

    Raises
    ------
    ValueError
        If ``n`` is not a whole number in ``[2, 200]``, if either endpoint is
        not finite, if ``high <= low``, or if ``low <= 0``.

    Examples
    --------
    >>> make_spot_range(80.0, 120.0, 5)
    array([ 80.,  90., 100., 110., 120.])
    """
    n_int = _validate_n(n, "Spot")
    low_f, high_f = _validate_bounds(low, high, "Spot")

    if low_f <= 0.0:
        raise ValueError(
            f"Spot range minimum must be greater than 0 (received {low_f:g}); "
            f"the pricing model is undefined at or below a zero underlying."
        )

    spot_range = np.linspace(low_f, high_f, n_int, dtype=np.float64)
    # Belt-and-braces: every point must sit inside the engine's domain (S > 0).
    validate_inputs(spot_range, 1.0, 0.0, 0.0, 0.0)
    return spot_range


def make_vol_range(low: float, high: float, n: int) -> NDArray[np.float64]:
    """Build the heat map's ascending volatility axis (y-axis, ROWS).

    Volatility is a DECIMAL here (0.20 == 20%, never 20). Per spec section 6.2
    the axis is FLOORED at :data:`MIN_VOL` (0.01): a ``low`` between 0 and 0.01
    is raised to 0.01 rather than rejected, because a zero-vol row collapses to
    the deterministic discounted payoff and reads as a dead stripe on the map.
    The floor is applied BEFORE the ordering check, so ``high`` must exceed the
    *floored* minimum.

    Parameters
    ----------
    low, high : float
        Range endpoints as decimals. ``low`` must be >= 0 (negative volatility
        is rejected outright, matching the engine) and is then floored at
        :data:`MIN_VOL`; ``high`` must be strictly greater than that floor.
    n : int
        Number of points, >= :data:`MIN_GRID_N` (2). See :func:`make_spot_range`.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, float64, strictly ASCENDING, from ``max(low, 0.01)`` to
        ``high``. Element 0 is the LOWEST volatility and becomes ROW 0 of the
        grid; the display flip to "lowest vol at the bottom" is the plotting
        layer's job.

    Raises
    ------
    ValueError
        If ``n`` is not a whole number in ``[2, 200]``, if either endpoint is
        not finite, if ``low < 0``, or if ``high`` is not strictly above the
        floored minimum.

    Examples
    --------
    >>> make_vol_range(0.10, 0.30, 5)
    array([0.1 , 0.15, 0.2 , 0.25, 0.3 ])
    >>> make_vol_range(0.0, 0.05, 2)   # floored at 0.01 per spec 6.2
    array([0.01, 0.05])
    """
    n_int = _validate_n(n, "Volatility")

    try:
        raw_low = float(low)
    except (TypeError, ValueError):
        raise ValueError("Volatility range endpoints must be numbers.") from None
    if not np.isfinite(raw_low):
        raise ValueError(
            "Volatility range endpoints must be real, finite numbers - they "
            "cannot be blank, NaN or infinite."
        )
    if raw_low < 0.0:
        raise ValueError(
            f"Volatility range minimum cannot be negative (received "
            f"{raw_low:g}). Enter it as a decimal, for example 0.20 for 20%."
        )

    # Spec section 6.2: the volatility axis is floored at 0.01.
    floored_low = max(raw_low, MIN_VOL)
    try:
        low_f, high_f = _validate_bounds(floored_low, high, "Volatility")
    except ValueError as exc:
        if raw_low < MIN_VOL and "greater than the minimum" in str(exc):
            raise ValueError(
                f"Volatility range maximum ({float(high):g}) must be greater "
                f"than {MIN_VOL:g}; the volatility axis is floored at "
                f"{MIN_VOL:g} (1%)."
            ) from None
        raise

    vol_range = np.linspace(low_f, high_f, n_int, dtype=np.float64)
    # Belt-and-braces: every point must sit inside the engine's domain
    # (sigma >= 0); the floor above already guarantees sigma >= 0.01.
    validate_inputs(1.0, 1.0, 0.0, 0.0, vol_range)
    return vol_range


# ---------------------------------------------------------------------------
# scenario matrix
# ---------------------------------------------------------------------------
def price_grid(
    spot_range: ArrayLike,
    vol_range: ArrayLike,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> NDArray[np.float64]:
    """Price the whole (spot x volatility) scenario matrix in ONE vectorized call.

    Exactly one ``np.meshgrid`` and exactly one call into
    :func:`src.black_scholes.price` - there is no Python loop over cells
    (spec section 7; NFR-1 gives a 10x10 grid a 500 ms budget end to end).

    Parameters
    ----------
    spot_range : array_like
        Ascending 1-D spot prices in currency units, all > 0 - the x-axis /
        COLUMNS. Typically from :func:`make_spot_range`.
    vol_range : array_like
        Ascending 1-D volatilities as DECIMALS, all >= 0 - the y-axis / ROWS.
        Typically from :func:`make_vol_range`.
    K : float
        Strike price in currency units (> 0). HELD FIXED across every cell.
    T : float
        Time to expiry in YEARS (>= 0). HELD FIXED across every cell - the
        surface is a mark-to-market valuation at one instant in time, not a
        payoff at expiration.
    r : float
        Risk-free rate as a DECIMAL per year (0.05 == 5%). HELD FIXED.
    option_type : {"call", "put"}
        Which side to price (case- and whitespace-insensitive).

    Returns
    -------
    np.ndarray
        Shape ``(len(vol_range), len(spot_range))``, float64, always finite and
        >= 0 - never ``NaN`` or ``inf``, including ``T == 0`` and ``sigma == 0``
        cells, which the engine resolves elementwise.

        ``values[i, j]`` is the option value at spot ``spot_range[j]`` and
        volatility ``vol_range[i]``. Row 0 is the LOWEST volatility because
        ``vol_range`` ascends; plotting.py flips vertically for display so the
        lowest volatility appears at the BOTTOM of the chart (spec section 7).
        Do not flip here.

    Raises
    ------
    ValueError
        If either range is empty, non-1-D, non-finite or not ascending; if
        ``K``, ``T`` or ``r`` is an array rather than a scalar; if
        ``option_type`` is not ``"call"``/``"put"``; or if any value falls
        outside the engine's domain (see
        :func:`src.black_scholes.validate_inputs`).

    Examples
    --------
    >>> spots = make_spot_range(80.0, 120.0, 5)
    >>> vols = make_vol_range(0.10, 0.30, 4)
    >>> grid = price_grid(spots, vols, K=100.0, T=1.0, r=0.05, option_type="call")
    >>> grid.shape                      # (n_vol, n_spot)
    (4, 5)
    >>> bool(grid[0, 0] < grid[-1, 0])  # row 0 is the LOWEST volatility
    True
    """
    spots = _as_ascending_axis(spot_range, "Spot range")
    vols = _as_ascending_axis(vol_range, "Volatility range")

    K_f = _require_scalar(K, "Strike price (K)")
    T_f = _require_scalar(T, "Time to expiry (T)")
    r_f = _require_scalar(r, "Risk-free rate (r)")

    # np.meshgrid's default indexing="xy" already returns
    # (len(vol_range), len(spot_range)) for BOTH outputs - exactly the required
    # orientation. DO NOT TRANSPOSE.
    S_grid, vol_grid = np.meshgrid(spots, vols)

    # One vectorized call. K, T and r broadcast as the fixed scalars they are.
    values = price(S_grid, K_f, T_f, r_f, vol_grid, option_type)

    # The shape is already (n_vol, n_spot); this only re-asserts the contract
    # and guarantees callers a plain float64 ndarray.
    return np.asarray(values, dtype=np.float64).reshape(vols.size, spots.size)
