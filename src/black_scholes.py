"""Black-Scholes-Merton pricing engine for European options.

Pure, side-effect-free, fully NumPy-vectorized.  No UI or web-framework
imports: this module must be importable and testable from a bare Python REPL.

Units (these are the units for *every* function in this package)
---------------------------------------------------------------
S      : underlying spot price, in currency units, strictly > 0
K      : strike price, in currency units, strictly > 0
T      : time to expiry in YEARS (not days), >= 0.  ``T = 0`` is legal and
         means "at expiry" -> intrinsic value.
r      : continuously-compounded annual risk-free rate as a DECIMAL
         (0.05 == 5%, never 5).  May be negative.
sigma  : annualized volatility as a DECIMAL (0.20 == 20%, never 20), >= 0.
         ``sigma = 0`` is legal and means "deterministic underlying".

Formulas
--------
    d1 = [ln(S/K) + (r + sigma**2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * N(d1) - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S * N(-d1)

with ``N`` the standard normal CDF.  Put-call parity

    C - P = S - K * exp(-r*T)

holds to floating-point tolerance for every finite input, including the
degenerate branches below.

Degenerate branches (handled ELEMENTWISE, never with a scalar ``if``)
--------------------------------------------------------------------
Arrays may mix regular and degenerate cells (e.g. one row of a heat-map grid
at ``sigma = 0``), so every boundary case is resolved with ``np.where`` over a
sanitized denominator.  No divide-by-zero, no ``RuntimeWarning``, no ``NaN``
is ever produced.

    T == 0      -> intrinsic value:  max(S - K, 0) / max(K - S, 0)
    sigma == 0  -> discounted deterministic payoff:
                   call: max(S - K*exp(-r*T), 0)
                   put : max(K*exp(-r*T) - S, 0)

``T == 0`` is evaluated first and takes precedence over ``sigma == 0``; at
``T == 0`` the discount factor is 1 so the two branches coincide anyway.

In those degenerate cells ``d1_d2`` returns the mathematical limit: ``+inf``
when the option is in the money against the forward (``S > K*exp(-r*T)``),
``-inf`` when it is out of the money, and ``0.0`` exactly at the forward.
Those limits are finite-safe under ``N(.)`` and ``phi(.)`` (``N(+-inf)`` is
1/0 and ``phi(+-inf)`` is 0), which is what makes the closed-form price
formula reproduce the intrinsic value exactly.  Callers computing Greeks must
still mask ``T == 0`` / ``sigma == 0`` cells, because quantities such as
``phi(d1) / (S*sigma*sqrt(T))`` are 0/0 there.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

__all__ = [
    "OptionType",
    "validate_inputs",
    "d1_d2",
    "call_price",
    "put_price",
    "price",
]

OptionType = Literal["call", "put"]

#: Accepted values for ``option_type`` (lower-cased, whitespace-stripped).
_OPTION_TYPES: tuple[str, ...] = ("call", "put")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _as_float_array(value: ArrayLike, label: str) -> NDArray[np.float64]:
    """Coerce ``value`` to a float array, or raise a plain-language ValueError.

    ``label`` is the user-facing name of the field, e.g. "Spot price (S)".
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    return arr


def _broadcast(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    """Broadcast the five pricing inputs to a common shape as float64 arrays."""
    return np.broadcast_arrays(
        np.asarray(S, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )


def _finalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize the return value: arrays stay arrays, 0-d results become scalars.

    Every public function in this module returns the broadcast shape of its
    inputs.  For all-scalar input that shape is ``()``; this collapses the 0-d
    array to a ``np.float64`` (exactly what ``np.exp(1.0)`` does), because a
    numpy scalar supports ``round()``, ``float()``, ``f"{x:.2f}"`` and
    comparisons, whereas a 0-d array has no ``__round__``.  Array input is
    returned untouched as ``np.ndarray``.
    """
    return arr[()] if np.ndim(arr) == 0 else arr


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> None:
    """Validate pricing inputs, raising ``ValueError`` with a plain-language message.

    Works on scalars *or* arrays: an input fails if ANY element is invalid
    (``np.any``), so a whole heat-map grid is checked in one call.

    Parameters
    ----------
    S : spot price in currency units, must be > 0.
    K : strike price in currency units, must be > 0.
    T : time to expiry in YEARS, must be >= 0.  ``T = 0`` is valid and prices
        the option at expiry.
    r : risk-free rate as a DECIMAL per year (0.05 == 5%); may be negative,
        must be finite.
    sigma : volatility as a DECIMAL per year (0.20 == 20%), must be >= 0.
        ``sigma = 0`` is valid.

    Returns
    -------
    None
        Nothing is returned, mutated or logged; the function is pure.

    Raises
    ------
    ValueError
        With a message written for an end user - the UI layer surfaces it
        verbatim as an error banner (NFR-2), so it never mentions internals
        and never reads like a traceback.
    """
    fields: tuple[tuple[ArrayLike, str], ...] = (
        (S, "Spot price (S)"),
        (K, "Strike price (K)"),
        (T, "Time to expiry (T)"),
        (r, "Risk-free rate (r)"),
        (sigma, "Volatility (sigma)"),
    )

    arrays: dict[str, NDArray[np.float64]] = {}
    for value, label in fields:
        arr = _as_float_array(value, label)
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{label} must be a real, finite number - it cannot be blank, "
                f"NaN or infinite."
            )
        arrays[label] = arr

    if np.any(arrays["Spot price (S)"] <= 0.0):
        raise ValueError("Spot price (S) must be greater than 0.")

    if np.any(arrays["Strike price (K)"] <= 0.0):
        raise ValueError("Strike price (K) must be greater than 0.")

    if np.any(arrays["Time to expiry (T)"] < 0.0):
        raise ValueError(
            "Time to expiry (T) cannot be negative. Enter the time remaining "
            "in years, for example 0.5 for six months; 0 means the option "
            "expires right now."
        )

    if np.any(arrays["Volatility (sigma)"] < 0.0):
        raise ValueError(
            "Volatility (sigma) cannot be negative. Enter it as a decimal, "
            "for example 0.20 for 20%; 0 means the underlying is certain."
        )

def _validate_option_type(option_type: str) -> str:
    """Normalize ``option_type`` to 'call'/'put' or raise a plain-language error."""
    if not isinstance(option_type, str):
        raise ValueError("Option type must be the text 'call' or 'put'.")
    normalized = option_type.strip().lower()
    if normalized not in _OPTION_TYPES:
        raise ValueError(
            f"Option type must be 'call' or 'put' (received {option_type!r})."
        )
    return normalized


# ---------------------------------------------------------------------------
# core engine
# ---------------------------------------------------------------------------
def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the Black-Scholes ``(d1, d2)`` terms, computed elementwise.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0).
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0).

    Returns
    -------
    (d1, d2) : tuple of np.ndarray
        Each has the broadcast shape of the inputs; all-scalar input yields a
        ``np.float64`` scalar, as ``np.exp(1.0)`` does.

    Raises
    ------
    ValueError
        If any input is outside its valid range (see :func:`validate_inputs`).

    Notes
    -----
    The division uses a sanitized denominator, so cells with ``T == 0`` or
    ``sigma == 0`` never emit a RuntimeWarning; those cells are then replaced
    by the limiting value ``+inf`` / ``-inf`` / ``0.0`` according to the sign
    of ``S - K*exp(-r*T)``.  Greeks that divide by ``sigma*sqrt(T)`` or
    ``sqrt(T)`` must mask those cells themselves (0/0 there).
    """
    validate_inputs(S, K, T, r, sigma)
    S_a, K_a, T_a, r_a, sig_a = _broadcast(S, K, T, r, sigma)

    # T and sigma are already known to be >= 0; np.maximum is belt-and-braces
    # so np.sqrt can never see a negative value.
    sqrt_T = np.sqrt(np.maximum(T_a, 0.0))
    denom = sig_a * sqrt_T
    degenerate = denom <= 0.0

    # Sanitized denominator: 1.0 wherever the true denominator is 0, so the
    # division below is always well defined and warning-free.
    safe_denom = np.where(degenerate, 1.0, denom)

    # S/K can overflow or underflow even when both prices are finite. Use
    # separate logs in the tails and log1p near the money, where subtracting
    # almost equal logs would discard a small but meaningful price difference.
    price_difference = S_a - K_a
    near_money = np.abs(price_difference) <= 0.5 * K_a
    relative_difference = np.divide(
        price_difference, K_a, out=np.zeros_like(price_difference), where=near_money
    )
    log_moneyness = np.where(
        near_money, np.log1p(relative_difference), np.log(S_a) - np.log(K_a)
    )

    d1_regular = (log_moneyness + (r_a + 0.5 * sig_a**2) * T_a) / safe_denom
    d2_regular = d1_regular - denom

    # Limiting value on the degenerate boundary, decided by forward moneyness.
    forward_edge = S_a - K_a * np.exp(-r_a * T_a)
    limit = np.where(
        forward_edge > 0.0,
        np.inf,
        np.where(forward_edge < 0.0, -np.inf, 0.0),
    )

    d1 = np.where(degenerate, limit, d1_regular)
    d2 = np.where(degenerate, limit, d2_regular)
    return _finalize(d1), _finalize(d2)


def call_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Price a European call option, elementwise over broadcastable inputs.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0); ``T = 0`` returns the intrinsic value
        ``max(S - K, 0)``.
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0); ``sigma = 0``
        returns the discounted deterministic payoff ``max(S - K*e^(-rT), 0)``.

    Returns
    -------
    np.ndarray
        Call value(s) in currency units with the broadcast shape of the inputs
        (a ``np.float64`` scalar for all-scalar input).  Always finite and
        >= 0 - never ``NaN`` or ``inf``, including deep in/out-of-the-money
        cells where ``|d1| > 8``.

    Raises
    ------
    ValueError
        If any input is outside its valid range (see :func:`validate_inputs`).

    Examples
    --------
    >>> round(float(call_price(100.0, 100.0, 1.0, 0.05, 0.2)), 4)
    10.4506
    """
    validate_inputs(S, K, T, r, sigma)
    S_a, K_a, T_a, r_a, sig_a = _broadcast(S, K, T, r, sigma)

    d1, d2 = d1_d2(S_a, K_a, T_a, r_a, sig_a)
    discounted_K = K_a * np.exp(-r_a * T_a)

    # Regular branch.  On degenerate cells d1 = d2 = +-inf, so N(.) is 1 or 0
    # and this expression already collapses to the correct payoff; the
    # np.where below makes that branch explicit rather than implicit.
    regular = S_a * norm.cdf(d1) - discounted_K * norm.cdf(d2)

    intrinsic = np.maximum(S_a - K_a, 0.0)               # T == 0
    deterministic = np.maximum(S_a - discounted_K, 0.0)  # sigma == 0, T > 0

    expired = T_a <= 0.0
    zero_vol = (sig_a <= 0.0) & ~expired  # T == 0 takes precedence

    values = np.where(expired, intrinsic, np.where(zero_vol, deterministic, regular))
    # Clip sub-ulp negatives produced by cancellation in deep-OTM cells.
    return _finalize(np.maximum(values, 0.0))


def put_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Price a European put option, elementwise over broadcastable inputs.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0); ``T = 0`` returns the intrinsic value
        ``max(K - S, 0)``.
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0); ``sigma = 0``
        returns the discounted deterministic payoff ``max(K*e^(-rT) - S, 0)``
        (the put analogue of the call form given in the spec).

    Returns
    -------
    np.ndarray
        Put value(s) in currency units with the broadcast shape of the inputs
        (a ``np.float64`` scalar for all-scalar input).  Always finite
        and >= 0.

    Raises
    ------
    ValueError
        If any input is outside its valid range (see :func:`validate_inputs`).

    Examples
    --------
    >>> round(float(put_price(100.0, 100.0, 1.0, 0.05, 0.2)), 4)
    5.5735
    """
    validate_inputs(S, K, T, r, sigma)
    S_a, K_a, T_a, r_a, sig_a = _broadcast(S, K, T, r, sigma)

    d1, d2 = d1_d2(S_a, K_a, T_a, r_a, sig_a)
    discounted_K = K_a * np.exp(-r_a * T_a)

    regular = discounted_K * norm.cdf(-d2) - S_a * norm.cdf(-d1)

    intrinsic = np.maximum(K_a - S_a, 0.0)               # T == 0
    deterministic = np.maximum(discounted_K - S_a, 0.0)  # sigma == 0, T > 0

    expired = T_a <= 0.0
    zero_vol = (sig_a <= 0.0) & ~expired  # T == 0 takes precedence

    values = np.where(expired, intrinsic, np.where(zero_vol, deterministic, regular))
    return _finalize(np.maximum(values, 0.0))


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType,
) -> NDArray[np.float64]:
    """Price a European option of either type - dispatcher over call/put.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0).
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0).
    option_type : ``"call"`` or ``"put"`` (case- and whitespace-insensitive).

    Returns
    -------
    np.ndarray
        Option value(s) in currency units with the broadcast shape of the
        inputs.

    Raises
    ------
    ValueError
        If any input is outside its valid range, or ``option_type`` is not
        ``"call"`` / ``"put"``.

    Examples
    --------
    >>> round(float(price(100.0, 100.0, 1.0, 0.05, 0.2, "put")), 4)
    5.5735
    """
    normalized = _validate_option_type(option_type)
    if normalized == "call":
        return call_price(S, K, T, r, sigma)
    return put_price(S, K, T, r, sigma)
