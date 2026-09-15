"""First- and second-order Black-Scholes sensitivities (the Greeks).

Pure, side-effect-free, fully NumPy-vectorized.  No UI or web-framework
imports: like :mod:`src.black_scholes`, this module must be importable and
testable from a bare Python REPL.

Units of the INPUTS (identical to :mod:`src.black_scholes`)
-----------------------------------------------------------
S      : underlying spot price, currency units, > 0
K      : strike price, currency units, > 0
T      : time to expiry in YEARS (not days), >= 0
r      : continuously-compounded annual risk-free rate as a DECIMAL
         (0.05 == 5%, never 5); may be negative
sigma  : annualized volatility as a DECIMAL (0.20 == 20%, never 20), >= 0

Formulas
--------
    Delta_call = N(d1)                    Delta_put  = N(d1) - 1
    Gamma      = phi(d1) / (S*sigma*sqrt(T))          (same for both)
    Vega       = S * phi(d1) * sqrt(T)                (same for both)
    Theta_call = -S*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
    Theta_put  = -S*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)
    Rho_call   =  K*T*exp(-r*T)*N(d2)     Rho_put    = -K*T*exp(-r*T)*N(-d2)

with ``N`` the standard normal CDF and ``phi`` the standard normal PDF
(``scipy.stats.norm.pdf``).  ``d1``/``d2`` are NOT recomputed here - they come
from :func:`src.black_scholes.d1_d2`, so the Greeks can never drift out of
sync with the prices.

Units of the OUTPUTS - raw vs. display
--------------------------------------
:func:`call_greeks`, :func:`put_greeks` and :func:`greeks` return RAW,
UNSCALED values in their natural analytic units:

    delta : change in option value per 1.00 change in S
    gamma : change in delta        per 1.00 change in S
    vega  : change in option value per 1.00 (== 100 vol points) change in sigma
    theta : change in option value per 1 YEAR of calendar decay
    rho   : change in option value per 1.00 (== 100pp) change in r

:func:`scale_for_display` converts those to the conventional trading-desk
units, and :data:`GREEK_UNITS` gives the exact label the UI must print next to
each figure:

    delta  /   1  -> "per $1 move"
    gamma  /   1  -> "per $1 move^2"      (printed with a superscript two)
    vega   / 100  -> "per 1 vol point"
    theta  / 365  -> "per calendar day"
    rho    / 100  -> "per 1pp rate move"

NOTE ON RHO.  The corrected spec labels ``rho / 100`` per one percentage-point
rate move (100 basis points). Per-basis-point rho would instead require
``rho / 10_000``. The UI uses the corrected percentage-point label.

Degenerate branches (handled ELEMENTWISE, never with a scalar ``if``)
--------------------------------------------------------------------
A heat-map grid can mix regular cells with ``T == 0`` or ``sigma == 0`` cells,
so every boundary case is resolved with ``np.where`` over a sanitized
denominator.  No divide-by-zero, no ``RuntimeWarning``, no ``NaN`` and no
``inf`` is ever produced.  ``T == 0`` is evaluated first and takes precedence
over ``sigma == 0``.

    delta  T == 0 or sigma == 0 : step function on the FORWARD moneyness
                                  ``S - K*exp(-r*T)`` (which is ``S - K`` at
                                  ``T == 0``): call 1.0 if strictly in the
                                  money else 0.0; put = call - 1.0, so the
                                  identity ``delta_call - delta_put == 1``
                                  holds exactly, including on the boundary.
    gamma  T == 0 or sigma == 0 : 0.0   (the closed form is 0/0 there, and its
    vega   T == 0 or sigma == 0 : 0.0    limit is unbounded for gamma at the
                                         forward-ATM knife edge).
    theta  T == 0               : 0.0   (undefined at the payoff kink)
           sigma == 0, T > 0    : carry term only, ``-r*K*exp(-r*T)`` for an
                                  in-the-money call - which is the exact
                                  derivative of the deterministic payoff.
    rho                         : no masking needed; the closed form is
                                  already exact (it carries a factor ``T``, so
                                  it is 0 at ``T == 0``, and at ``sigma == 0``
                                  ``N(d2)`` collapses to the 0/1 indicator of
                                  forward moneyness).

The single documented simplification: at ``sigma == 0`` with ``S`` exactly on
the forward ``K*exp(-r*T)``, the true one-sided limit of vega is
``S*sqrt(T)*phi(0)`` rather than 0.  That knife edge is a measure-zero
coincidence, gamma is genuinely unbounded there, and returning 0 keeps the
whole ``sigma == 0`` row uniform - so 0.0 is returned for both.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .black_scholes import OptionType, d1_d2, validate_inputs

__all__ = [
    "GREEK_NAMES",
    "GREEK_UNITS",
    "GREEK_DISPLAY_DIVISORS",
    "call_greeks",
    "put_greeks",
    "greeks",
    "scale_for_display",
]

#: The Greeks this module produces, in the order the UI table should show them.
GREEK_NAMES: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")

#: Display-unit label for each Greek, printed verbatim by the UI next to the
#: DISPLAY-SCALED figure from :func:`scale_for_display`.  "²" is a
#: superscript two, so gamma reads "per $1 move" followed by it.
GREEK_UNITS: dict[str, str] = {
    "delta": "per $1 move",
    "gamma": "per $1 move²",
    "vega": "per 1 vol point",
    "theta": "per calendar day",
    "rho": "per 1pp rate move",
}

#: Divisor applied by :func:`scale_for_display` to turn a raw Greek into the
#: unit named by :data:`GREEK_UNITS`.  Delta and gamma are already quoted per
#: $1 move, so they are unscaled.  Rho is /100 == per 1 percentage point of
#: rate (NOT per basis point - see the module docstring).
GREEK_DISPLAY_DIVISORS: dict[str, float] = {
    "delta": 1.0,
    "gamma": 1.0,
    "vega": 100.0,
    "theta": 365.0,
    "rho": 100.0,
}

#: Accepted values for ``option_type`` (lower-cased, whitespace-stripped).
_OPTION_TYPES: tuple[str, ...] = ("call", "put")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _broadcast(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike
) -> tuple[NDArray[np.float64], ...]:
    """Broadcast the five inputs to a common shape as float64 arrays."""
    return np.broadcast_arrays(
        np.asarray(S, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(sigma, dtype=np.float64),
    )


def _finalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sanitize and normalize one Greek before it leaves the module.

    Non-finite cells are replaced by 0.0 as a last-resort guard: a Greek must
    never surface as ``NaN`` or ``inf`` in the UI (only reachable with absurd
    inputs, e.g. an ``r*T`` large enough to overflow ``exp``; every in-range
    input is already finite by construction).  A 0-d result is collapsed to a
    ``np.float64`` scalar, exactly as :mod:`src.black_scholes` does, so the UI
    can call ``round()`` / ``f"{x:.4f}"`` on scalar-input results.
    """
    clean = np.where(np.isfinite(arr), arr, 0.0)
    return clean[()] if np.ndim(clean) == 0 else clean


def _normalize_option_type(option_type: str) -> str:
    """Normalize ``option_type`` to 'call'/'put' or raise a plain-language error."""
    if not isinstance(option_type, str):
        raise ValueError("Option type must be the text 'call' or 'put'.")
    normalized = option_type.strip().lower()
    if normalized not in _OPTION_TYPES:
        raise ValueError(
            f"Option type must be 'call' or 'put' (received {option_type!r})."
        )
    return normalized


class _Common:
    """Intermediate quantities shared by the call and put Greeks.

    A plain value object built once per call so that :func:`call_greeks` and
    :func:`put_greeks` never duplicate - or disagree about - the boundary
    handling.  Every attribute has the broadcast shape of the inputs.
    """

    __slots__ = (
        "S",
        "K",
        "T",
        "r",
        "sigma",
        "d1",
        "d2",
        "discounted_K",
        "forward_edge",
        "expired",
        "degenerate",
        "gamma",
        "vega",
        "theta_diffusion",
        "itm_step",
    )

    def __init__(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
    ) -> None:
        validate_inputs(S, K, T, r, sigma)
        S_a, K_a, T_a, r_a, sig_a = _broadcast(S, K, T, r, sigma)
        self.S, self.K, self.T, self.r, self.sigma = S_a, K_a, T_a, r_a, sig_a

        # Reuse the engine's d1/d2 rather than recomputing them.  On degenerate
        # cells it returns the limits +-inf (or 0.0 exactly at the forward),
        # which are finite-safe under N(.) and phi(.):  N(+-inf) is 1/0 and
        # phi(+-inf) is exactly 0.0.
        d1, d2 = d1_d2(S_a, K_a, T_a, r_a, sig_a)
        self.d1 = np.asarray(d1, dtype=np.float64)
        self.d2 = np.asarray(d2, dtype=np.float64)

        self.discounted_K = K_a * np.exp(-r_a * T_a)
        self.forward_edge = S_a - self.discounted_K

        # T == 0 first, then sigma == 0 (branch order is explicit and shared).
        self.expired = T_a <= 0.0
        self.degenerate = self.expired | (sig_a <= 0.0)

        sqrt_T = np.sqrt(np.maximum(T_a, 0.0))
        phi_d1 = norm.pdf(self.d1)

        # Sanitized denominators: 1.0 wherever the true denominator is 0, so no
        # division ever warns.  The degenerate cells are overwritten below.
        gamma_denom = S_a * sig_a * sqrt_T
        safe_gamma_denom = np.where(self.degenerate, 1.0, gamma_denom)
        safe_sqrt_T = np.where(sqrt_T <= 0.0, 1.0, sqrt_T)

        self.gamma = np.where(self.degenerate, 0.0, phi_d1 / safe_gamma_denom)
        self.vega = np.where(self.degenerate, 0.0, S_a * phi_d1 * sqrt_T)
        self.theta_diffusion = np.where(
            self.degenerate, 0.0, -S_a * phi_d1 * sig_a / (2.0 * safe_sqrt_T)
        )

        # Boundary delta: the step function on forward moneyness.  Ties (S
        # exactly on the forward) resolve to the out-of-the-money side for the
        # call, so delta_call - delta_put == 1 still holds exactly.
        self.itm_step = np.where(self.forward_edge > 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def call_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Return the RAW (unscaled) Greeks of a European call, elementwise.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0); ``T = 0`` is legal.
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0); ``sigma = 0`` is
        legal.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are exactly :data:`GREEK_NAMES`; each value has the broadcast
        shape of the inputs (a ``np.float64`` scalar for all-scalar input) and
        is always finite - never ``NaN`` or ``inf``, on any branch.

        Values are UNSCALED analytic Greeks: delta per 1.00 of S, gamma per
        1.00 of S squared, vega per 1.00 of sigma (100 vol points), theta per
        1 YEAR, rho per 1.00 of r (100 percentage points).  Pass the dict
        through :func:`scale_for_display` to get desk units.

    Raises
    ------
    ValueError
        If any input is outside its valid range (see
        :func:`src.black_scholes.validate_inputs`).

    Examples
    --------
    >>> round(float(call_greeks(100.0, 100.0, 1.0, 0.05, 0.2)["delta"]), 4)
    0.6368
    """
    c = _Common(S, K, T, r, sigma)

    delta = np.where(c.degenerate, c.itm_step, norm.cdf(c.d1))

    # The carry term is exact on every branch, sigma == 0 included, where
    # N(d2) collapses to the 0/1 indicator of forward moneyness.
    theta = np.where(
        c.expired,
        0.0,
        c.theta_diffusion - c.r * c.discounted_K * norm.cdf(c.d2),
    )

    rho = c.T * c.discounted_K * norm.cdf(c.d2)

    return {
        "delta": _finalize(delta),
        "gamma": _finalize(c.gamma),
        "vega": _finalize(c.vega),
        "theta": _finalize(theta),
        "rho": _finalize(rho),
    }


def put_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Return the RAW (unscaled) Greeks of a European put, elementwise.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0); ``T = 0`` is legal.
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0); ``sigma = 0`` is
        legal.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are exactly :data:`GREEK_NAMES`; each value has the broadcast
        shape of the inputs (a ``np.float64`` scalar for all-scalar input) and
        is always finite.  Gamma and vega are identical to the call's; delta,
        theta and rho are not.  Units are the raw analytic ones - see
        :func:`call_greeks` and :func:`scale_for_display`.

    Raises
    ------
    ValueError
        If any input is outside its valid range (see
        :func:`src.black_scholes.validate_inputs`).

    Examples
    --------
    >>> round(float(put_greeks(100.0, 100.0, 1.0, 0.05, 0.2)["delta"]), 4)
    -0.3632
    """
    c = _Common(S, K, T, r, sigma)

    # delta_put = delta_call - 1 on every branch, degenerate cells included,
    # so the parity identity holds to the last bit.
    delta = np.where(c.degenerate, c.itm_step, norm.cdf(c.d1)) - 1.0

    theta = np.where(
        c.expired,
        0.0,
        c.theta_diffusion + c.r * c.discounted_K * norm.cdf(-c.d2),
    )

    rho = -c.T * c.discounted_K * norm.cdf(-c.d2)

    return {
        "delta": _finalize(delta),
        "gamma": _finalize(c.gamma),
        "vega": _finalize(c.vega),
        "theta": _finalize(theta),
        "rho": _finalize(rho),
    }


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType,
) -> dict[str, NDArray[np.float64]]:
    """Return the RAW Greeks of either option type - dispatcher over call/put.

    Parameters
    ----------
    S, K : spot and strike prices in currency units (> 0).
    T : time to expiry in YEARS (>= 0).
    r : risk-free rate, DECIMAL per year (0.05 == 5%).
    sigma : volatility, DECIMAL per year (0.20 == 20%, >= 0).
    option_type : ``"call"`` or ``"put"`` (case- and whitespace-insensitive).

    Returns
    -------
    dict[str, np.ndarray]
        Keys are exactly :data:`GREEK_NAMES`; see :func:`call_greeks` for the
        units and shape contract.

    Raises
    ------
    ValueError
        If any input is outside its valid range, or ``option_type`` is not
        ``"call"`` / ``"put"``.

    Examples
    --------
    >>> round(float(greeks(100.0, 100.0, 1.0, 0.05, 0.2, "put")["rho"]), 4)
    -41.8905
    """
    if _normalize_option_type(option_type) == "call":
        return call_greeks(S, K, T, r, sigma)
    return put_greeks(S, K, T, r, sigma)


def scale_for_display(
    raw: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Convert RAW Greeks to the desk units named by :data:`GREEK_UNITS`.

    Parameters
    ----------
    raw : dict[str, np.ndarray]
        A dict as returned by :func:`call_greeks`, :func:`put_greeks` or
        :func:`greeks`.  Values may be scalars or arrays of any shape.

    Returns
    -------
    dict[str, np.ndarray]
        A NEW dict with the same keys and shapes, each value divided by
        :data:`GREEK_DISPLAY_DIVISORS`:

        ===== ======= =======================================================
        greek divisor meaning
        ===== ======= =======================================================
        delta       1 per $1 move (already in that unit)
        gamma       1 per $1 move squared (already in that unit)
        vega      100 per 1 vol point (1 point == 0.01 of sigma)
        theta     365 per calendar day
        rho       100 per 1 percentage point of rate (== 100bp, NOT per bp)
        ===== ======= =======================================================

        Keys that are not recognized Greeks are passed through unscaled, so
        the function is safe to apply to an enriched dict.  The input is never
        mutated and its arrays are never aliased into the result.

    Raises
    ------
    ValueError
        If ``raw`` is not a dict.

    Notes
    -----
    Scaling is applied ONCE, at the display boundary.  Never feed the result
    back into this function, and never scale before doing arithmetic with the
    raw Greeks.

    Examples
    --------
    >>> disp = scale_for_display(call_greeks(100.0, 100.0, 1.0, 0.05, 0.2))
    >>> round(float(disp["vega"]), 4)      # per 1 vol point
    0.3752
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "scale_for_display expects the dict returned by call_greeks / "
            "put_greeks / greeks."
        )

    scaled: dict[str, NDArray[np.float64]] = {}
    for name, value in raw.items():
        divisor = GREEK_DISPLAY_DIVISORS.get(name, 1.0)
        scaled[name] = _finalize(np.asarray(value, dtype=np.float64) / divisor)
    return scaled
