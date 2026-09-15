"""Pricing-engine tests - spec section 10 rows T-1..T-8 and T-11.

Every randomized test draws from a SEEDED ``np.random.default_rng`` so a
failure is reproducible bit-for-bit; an unseeded generator would make a real
sign error look like a flake.

Tolerance policy used throughout
--------------------------------
* Identities that are algebraically exact in IEEE-754 (intrinsic value at
  ``T = 0``, the deterministic payoff at ``sigma = 0``, vectorized vs. scalar
  evaluation) are asserted EXACTLY - no tolerance at all.  The engine performs
  the same floating-point operations in both paths, so any difference is a
  real behavioural difference, not rounding.
* Put-call parity is asserted with an absolute AND a relative component (see
  :func:`test_t1_put_call_parity_500_random_inputs`).
* The spec's reference values are quoted to 4 dp, so they are asserted to
  ``abs=5e-5`` - half a unit in the last quoted place.
"""

from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np
import pytest

from src.black_scholes import (
    call_price,
    d1_d2,
    price,
    put_price,
    validate_inputs,
)

# The spec quotes its reference values to 4 decimals; half a unit in the last
# quoted place is the tightest honest tolerance against them.
QUOTED_DP_TOL = 5e-5


def _reference_prices(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Scalar BSM reference using only Python's math library, including its CDF."""
    discounted_strike = K * math.exp(-r * T)
    if T == 0.0 or sigma == 0.0:
        return max(S - discounted_strike, 0.0), max(discounted_strike - S, 0.0)
    total_volatility = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + r * T) / total_volatility + total_volatility / 2.0
    d2 = d1 - total_volatility

    def normal_cdf(x: float) -> float:
        return math.erfc(-x / math.sqrt(2.0)) / 2.0

    return (
        S * normal_cdf(d1) - discounted_strike * normal_cdf(d2),
        discounted_strike * normal_cdf(-d2) - S * normal_cdf(-d1),
    )


def test_o1_prices_match_independent_reference_on_500_case_grid() -> None:
    """O-1: both prices agree within 1e-6 absolute across 500 parameter cases.

    The reference uses neither the engine's d1/d2 nor SciPy's normal CDF.
    Include negative rates, expiration, zero volatility and near expiration.
    """
    cases = [
        (S, K, T, (-0.05, 0.0, 0.05, 0.15, 0.25)[(i + i // 5) % 5], sigma)
        for i, (S, K, T, sigma) in enumerate(product(
            (25.0, 75.0, 100.0, 125.0, 300.0),
            (50.0, 100.0, 150.0, 250.0),
            (0.0, 1.0 / 365.0, 0.25, 1.0, 5.0),
            (0.0, 0.05, 0.2, 0.5, 1.0),
        ))
    ]
    assert len(cases) == 500
    inputs = np.asarray(cases).T
    expected_call, expected_put = np.asarray([_reference_prices(*case) for case in cases]).T
    np.testing.assert_allclose(call_price(*inputs), expected_call, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(put_price(*inputs), expected_put, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# T-1  put-call parity
# ---------------------------------------------------------------------------
def test_t1_put_call_parity_500_random_inputs(rng: np.random.Generator) -> None:
    """T-1: |C - P - (S - K*e^(-rT))| < 1e-9 over 500 random valid inputs.

    S and K are drawn from [1, 500].  The bound matters: parity is evaluated as
    a difference of two numbers of magnitude ~S, so the achievable absolute
    error grows with the magnitude of the inputs (about ``eps * S``, i.e. ~1e-13
    at S = 500).  Drawing S, K from [1, 500] keeps the spec's 1e-9 ABSOLUTE
    bound comfortably achievable; drawing from, say, [1, 1e9] would not, and the
    test would then be measuring float64 rather than the model.

    The assertion carries both components: rtol=1e-12 (relative, for the
    large-S cases) and atol=1e-9 (absolute, the spec's own bound, which covers
    the near-ATM cases where S - K*e^(-rT) is itself close to zero and a purely
    relative bound would be meaningless).
    """
    n = 500
    S = rng.uniform(1.0, 500.0, n)
    K = rng.uniform(1.0, 500.0, n)
    T = rng.uniform(0.01, 5.0, n)
    r = rng.uniform(-0.05, 0.25, n)  # spec 5.3 valid range for r
    sigma = rng.uniform(0.01, 1.50, n)

    lhs = call_price(S, K, T, r, sigma) - put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-9)
    # Restate the spec's own bound verbatim so the row is unambiguously covered.
    assert np.max(np.abs(lhs - rhs)) < 1e-9


def test_t1_put_call_parity_holds_on_degenerate_branches() -> None:
    """T-1 (extension): parity must survive the T=0 and sigma=0 branches too.

    Those cells take a different code path (branch selection rather than the
    closed form), which is exactly where a sign error would hide.
    """
    S = np.array([50.0, 100.0, 100.0, 150.0, 80.0, 120.0])
    K = np.full(6, 100.0)
    T = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 2.0])
    sigma = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = 0.05

    lhs = call_price(S, K, T, r, sigma) - put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-9)


# ---------------------------------------------------------------------------
# T-2 / T-3  textbook reference case
# ---------------------------------------------------------------------------
def test_t2_atm_call_reference_value(atm: dict[str, float]) -> None:
    """T-2: ATM call S=K=100, T=1, r=0.05, sigma=0.2 -> 10.4506."""
    assert float(call_price(**atm)) == pytest.approx(10.4506, abs=QUOTED_DP_TOL)


def test_t3_atm_put_reference_value(atm: dict[str, float]) -> None:
    """T-3: ATM put on the same inputs -> 5.5735."""
    assert float(put_price(**atm)) == pytest.approx(5.5735, abs=QUOTED_DP_TOL)


def test_t2_t3_dispatcher_agrees_with_the_direct_functions(
    atm: dict[str, float],
) -> None:
    """T-2/T-3: ``price(..., option_type)`` must dispatch, not re-derive."""
    assert price(**atm, option_type="call") == call_price(**atm)
    assert price(**atm, option_type="put") == put_price(**atm)
    # Case / whitespace tolerance is part of the dispatcher's contract.
    assert price(**atm, option_type="  CALL ") == call_price(**atm)
    with pytest.raises(ValueError):
        price(**atm, option_type="straddle")


# ---------------------------------------------------------------------------
# T-4  T == 0 -> exact intrinsic value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("S", "K", "expected_call", "expected_put"),
    [
        (110.0, 100.0, 10.0, 0.0),  # call ITM  / put OTM
        (90.0, 100.0, 0.0, 10.0),   # call OTM  / put ITM
        (100.0, 100.0, 0.0, 0.0),   # exactly at the money
    ],
)
def test_t4_zero_time_returns_exact_intrinsic_value(
    S: float, K: float, expected_call: float, expected_put: float
) -> None:
    """T-4: T=0 returns the EXACT intrinsic value - call and put, ITM and OTM.

    Asserted with ``==`` rather than a tolerance: ``max(S-K, 0)`` involves no
    transcendental function, so an exactly-representable answer is the only
    acceptable one.  r and sigma must be irrelevant at T=0, so both are varied.
    """
    for r in (0.0, 0.05, 0.25, -0.05):
        for sigma in (0.0, 0.2, 1.5):
            assert float(call_price(S, K, 0.0, r, sigma)) == expected_call
            assert float(put_price(S, K, 0.0, r, sigma)) == expected_put


def test_t4_zero_time_is_elementwise_not_scalar(rng: np.random.Generator) -> None:
    """T-4 (trap A): a grid mixing T=0 and T>0 cells resolves each cell on its own.

    A scalar ``if T == 0`` would either miss the expired cells or wrongly
    collapse the live ones.
    """
    S = rng.uniform(60.0, 140.0, 40)
    T = np.where(np.arange(40) % 2 == 0, 0.0, 1.0)

    values = call_price(S, 100.0, T, 0.05, 0.2)
    expired = T == 0.0

    np.testing.assert_array_equal(values[expired], np.maximum(S[expired] - 100.0, 0.0))
    # The live cells must NOT have collapsed to intrinsic value.
    assert np.all(values[~expired] >= np.maximum(S[~expired] - 100.0, 0.0))
    assert np.any(values[~expired] > np.maximum(S[~expired] - 100.0, 0.0) + 1e-6)


# ---------------------------------------------------------------------------
# T-5  sigma == 0 -> discounted deterministic payoff
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [60.0, 95.0, 100.0, 105.0, 140.0])
@pytest.mark.parametrize("T", [0.25, 1.0, 5.0])
def test_t5_zero_vol_returns_discounted_deterministic_payoff(
    S: float, T: float
) -> None:
    """T-5: sigma=0 -> max(S - K*e^(-rT), 0) for a call.

    Trap B: the spec states only the CALL form.  The put analogue is
    ``max(K*e^(-rT) - S, 0)`` and is asserted here too - without it, a put that
    silently returned the call payoff would pass the spec's own test plan.
    """
    K, r = 100.0, 0.05
    discounted_K = K * np.exp(-r * T)

    assert float(call_price(S, K, T, r, 0.0)) == max(S - discounted_K, 0.0)
    assert float(put_price(S, K, T, r, 0.0)) == max(discounted_K - S, 0.0)


def test_t5_zero_vol_row_inside_a_grid_is_elementwise() -> None:
    """T-5 (trap A): one sigma=0 ROW inside a live grid, resolved elementwise."""
    S = np.linspace(70.0, 130.0, 8)
    sigma = np.array([0.0, 0.1, 0.3])[:, None]
    K, T, r = 100.0, 1.0, 0.05

    values = call_price(S[None, :], K, T, r, sigma)
    assert values.shape == (3, 8)
    np.testing.assert_array_equal(values[0], np.maximum(S - K * np.exp(-r * T), 0.0))
    # A live vol row is worth at least the zero-vol row (vega >= 0).
    assert np.all(values[1] >= values[0])


def test_t5_zero_time_takes_precedence_over_zero_vol() -> None:
    """T-5 (trap C): with T=0 AND sigma=0, the T=0 branch wins - explicitly.

    Both branches agree numerically at T=0 (the discount factor is 1), which is
    precisely why an inconsistent branch order could go unnoticed; pin it.
    """
    assert float(call_price(110.0, 100.0, 0.0, 0.25, 0.0)) == 10.0
    assert float(put_price(90.0, 100.0, 0.0, 0.25, 0.0)) == 10.0


# ---------------------------------------------------------------------------
# T-6  deep OTM / deep ITM numerical stability
# ---------------------------------------------------------------------------
def test_t6_deep_otm_call_is_zero_and_not_nan() -> None:
    """T-6: deep OTM call S=10, K=1000 -> ~0, finite, non-negative, not NaN."""
    value = float(call_price(10.0, 1000.0, 1.0, 0.05, 0.2))
    assert not np.isnan(value)
    assert np.isfinite(value)
    assert 0.0 <= value < 1e-6


def test_t6_mirror_deep_itm_call_is_the_discounted_forward() -> None:
    """T-6 (mirror): deep ITM call S=1000, K=10 -> S - K*e^(-rT), its put -> 0.

    The mirror case is what catches an overflow in the other tail: |d1| > 8 must
    not saturate to inf or produce NaN.
    """
    S, K, T, r, sigma = 1000.0, 10.0, 1.0, 0.05, 0.2
    call = float(call_price(S, K, T, r, sigma))
    put = float(put_price(S, K, T, r, sigma))

    assert np.isfinite(call) and np.isfinite(put)
    assert call == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)
    assert 0.0 <= put < 1e-9


def test_t6_extreme_d1_is_finite_and_prices_stay_bounded() -> None:
    """T-6: |d1| > 8 must not overflow anywhere on the surface (spec 5.4)."""
    S = np.array([1.0, 10.0, 100.0, 1_000.0, 10_000.0])
    K = 100.0
    T = np.array([[1e-6], [0.5], [30.0]])
    sigma = np.array([[[0.01]], [[0.2]], [[3.0]]])

    d1, d2 = d1_d2(S, K, T, 0.05, sigma)
    assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
    assert np.any(np.abs(d1) > 8.0), "the sweep must actually reach |d1| > 8"

    call = call_price(S, K, T, 0.05, sigma)
    put = put_price(S, K, T, 0.05, sigma)
    assert np.all(np.isfinite(call)) and np.all(np.isfinite(put))
    assert not np.any(np.isnan(call)) and not np.any(np.isnan(put))
    # No-arbitrage envelope: 0 <= C <= S and 0 <= P <= K.
    assert np.all(call >= 0.0)
    assert np.all(call <= np.broadcast_to(S, call.shape) + 1e-9)
    assert np.all(put >= 0.0) and np.all(put <= K + 1e-9)


def test_t6_extreme_finite_spot_strike_ratio_does_not_overflow() -> None:
    """Finite prices must not overflow just because their ratio exceeds float64."""
    S = np.array([1e-200, 1e200])
    K = S[::-1]
    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        d1, d2 = d1_d2(S, K, 1.0, 0.0, 0.2)
        calls = call_price(S, K, 1.0, 0.0, 0.2)
        puts = put_price(S, K, 1.0, 0.0, 0.2)
    assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))
    np.testing.assert_array_equal(calls, np.array([0.0, 1e200]))
    np.testing.assert_array_equal(puts, np.array([1e200, 0.0]))


def test_d1_preserves_moneyness_for_almost_equal_prices_and_small_volatility() -> None:
    """A small spot difference still matters when volatility approaches zero."""
    K, sigma = 100.0, 1e-16
    S = np.nextafter(K, np.inf)
    d1, d2 = d1_d2(S, K, 1.0, 0.0, sigma)
    expected = math.log1p((S - K) / K) / sigma
    assert float(d1) == pytest.approx(expected + sigma / 2.0, rel=1e-14)
    assert float(d2) == pytest.approx(expected - sigma / 2.0, rel=1e-14)


def test_prices_obey_discounted_arbitrage_bounds_with_negative_rates() -> None:
    """Negative rates can make a European put worth more than its strike."""
    S = np.array([1.0, 75.0, 100.0, 150.0])[None, :]
    T = np.array([0.0, 0.25, 1.0, 5.0])[:, None]
    K, r = 100.0, -0.05
    discounted_strike = K * np.exp(-r * T)
    calls = call_price(S, K, T, r, 0.2)
    puts = put_price(S, K, T, r, 0.2)
    assert np.all(calls >= np.maximum(S - discounted_strike, 0.0) - 1e-12)
    assert np.all(calls <= S + 1e-12)
    assert np.all(puts >= np.maximum(discounted_strike - S, 0.0) - 1e-12)
    assert np.all(puts <= discounted_strike + 1e-12)
    assert puts[-1, 0] > K


# ---------------------------------------------------------------------------
# T-7  vectorization
# ---------------------------------------------------------------------------
def test_t7_vectorized_10x10_matches_a_scalar_loop(rng: np.random.Generator) -> None:
    """T-7: a (10,10) call returns (10,10) and is elementwise-equal to a loop.

    Asserted EXACTLY (``assert_array_equal``): the vectorized path performs the
    identical sequence of floating-point operations per cell, so any difference
    at all signals a broadcasting or branch-selection bug rather than rounding.
    """
    spots = np.sort(rng.uniform(50.0, 150.0, 10))
    vols = np.sort(rng.uniform(0.05, 0.90, 10))
    K, T, r = 100.0, 1.0, 0.05

    S_grid, vol_grid = np.meshgrid(spots, vols)
    vectorized = call_price(S_grid, K, T, r, vol_grid)

    assert isinstance(vectorized, np.ndarray)
    assert vectorized.shape == (10, 10)

    looped = np.empty((10, 10), dtype=np.float64)
    for i in range(10):
        for j in range(10):
            looped[i, j] = float(call_price(spots[j], K, T, r, vols[i]))

    np.testing.assert_array_equal(vectorized, looped)


def test_t7_vectorized_put_matches_a_scalar_loop(rng: np.random.Generator) -> None:
    """T-7 (put side): same contract for ``put_price`` on a (10,10) array."""
    spots = np.sort(rng.uniform(50.0, 150.0, 10))
    vols = np.sort(rng.uniform(0.05, 0.90, 10))
    S_grid, vol_grid = np.meshgrid(spots, vols)

    vectorized = put_price(S_grid, 100.0, 1.0, 0.05, vol_grid)
    assert vectorized.shape == (10, 10)

    looped = np.array(
        [[float(put_price(s, 100.0, 1.0, 0.05, v)) for s in spots] for v in vols]
    )
    np.testing.assert_array_equal(vectorized, looped)


def test_t7_scalar_input_returns_a_scalar_and_arrays_broadcast() -> None:
    """T-7: the return shape is the broadcast shape of the inputs."""
    scalar = call_price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert np.ndim(scalar) == 0
    assert float(scalar) == pytest.approx(10.4506, abs=QUOTED_DP_TOL)

    assert call_price(np.full(7, 100.0), 100.0, 1.0, 0.05, 0.2).shape == (7,)
    assert call_price(
        np.full((3, 1), 100.0), 100.0, 1.0, 0.05, np.full(4, 0.2)
    ).shape == (3, 4)


# ---------------------------------------------------------------------------
# T-8  monotonicity
# ---------------------------------------------------------------------------
def test_t8_call_price_strictly_increasing_in_spot() -> None:
    """T-8: dC/dS > 0 - the call price is strictly increasing in S."""
    spots = np.linspace(50.0, 150.0, 200)
    values = call_price(spots, 100.0, 1.0, 0.05, 0.2)
    assert np.all(np.diff(values) > 0.0)


def test_t8_call_price_strictly_increasing_in_vol() -> None:
    """T-8: dC/dsigma > 0 - vega is strictly positive for T > 0."""
    vols = np.linspace(0.01, 1.50, 200)
    values = call_price(100.0, 100.0, 1.0, 0.05, vols)
    assert np.all(np.diff(values) > 0.0)


def test_t8_monotonicity_also_holds_for_the_put() -> None:
    """T-8 (extension): the put falls in S and rises in sigma."""
    spots = np.linspace(50.0, 150.0, 200)
    assert np.all(np.diff(put_price(spots, 100.0, 1.0, 0.05, 0.2)) < 0.0)

    vols = np.linspace(0.01, 1.50, 200)
    assert np.all(np.diff(put_price(100.0, 100.0, 1.0, 0.05, vols)) > 0.0)


def test_t8_monotonicity_across_a_full_surface() -> None:
    """T-8: monotone in BOTH directions simultaneously on a 2-D surface."""
    spots = np.linspace(60.0, 140.0, 25)
    vols = np.linspace(0.05, 1.0, 25)
    S_grid, vol_grid = np.meshgrid(spots, vols)
    surface = call_price(S_grid, 100.0, 1.0, 0.05, vol_grid)

    assert np.all(np.diff(surface, axis=1) > 0.0), "must rise left -> right in S"
    assert np.all(np.diff(surface, axis=0) > 0.0), "must rise row 0 -> row n in vol"


# ---------------------------------------------------------------------------
# T-11  input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "what"),
    [
        ({"T": -1.0}, "negative time to expiry"),
        ({"T": -1e-12}, "barely negative time to expiry"),
        ({"sigma": -0.2}, "negative volatility"),
        ({"sigma": -1e-12}, "barely negative volatility"),
        ({"S": 0.0}, "zero spot"),
        ({"S": -100.0}, "negative spot"),
        ({"K": 0.0}, "zero strike"),
        ({"K": -100.0}, "negative strike"),
    ],
)
def test_t11_out_of_range_inputs_raise_value_error(
    atm: dict[str, float], kwargs: dict[str, float], what: str
) -> None:
    """T-11: negative T, negative sigma, S<=0 and K<=0 each raise ValueError.

    Checked on every public entry point, because a UI that reaches the engine
    through ``price`` must fail the same way one calling ``call_price`` does
    (NFR-2: no traceback ever reaches the user).
    """
    bad = {**atm, **kwargs}

    for fn in (validate_inputs, d1_d2, call_price, put_price):
        with pytest.raises(ValueError):
            fn(**bad)

    for option_type in ("call", "put"):
        with pytest.raises(ValueError):
            price(**bad, option_type=option_type)


def test_t11_a_single_bad_cell_rejects_the_whole_array() -> None:
    """T-11: validation is over the WHOLE array - one bad cell is enough."""
    spots = np.linspace(80.0, 120.0, 10)
    spots[4] = -1.0
    with pytest.raises(ValueError):
        call_price(spots, 100.0, 1.0, 0.05, 0.2)

    vols = np.linspace(0.05, 0.5, 10)
    vols[-1] = -0.01
    with pytest.raises(ValueError):
        call_price(100.0, 100.0, 1.0, 0.05, vols)


def test_t11_nan_and_inf_are_rejected() -> None:
    """T-11 (extension): NaN / inf inputs raise instead of poisoning the grid."""
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            call_price(bad, 100.0, 1.0, 0.05, 0.2)
        with pytest.raises(ValueError):
            call_price(100.0, 100.0, 1.0, 0.05, bad)


def test_t11_boundary_values_are_accepted() -> None:
    """T-11 (converse): T=0 and sigma=0 are VALID (spec 5.3), not errors.

    The complement matters as much as the rejection: an over-eager ``T > 0``
    check would break the expiry column of the heat map.
    """
    validate_inputs(100.0, 100.0, 0.0, 0.05, 0.0)
    assert np.isfinite(float(call_price(100.0, 100.0, 0.0, 0.05, 0.0)))
    # r is allowed to be negative (spec 5.3: -0.05 .. 0.25).
    assert np.isfinite(float(call_price(100.0, 100.0, 1.0, -0.05, 0.2)))


# ---------------------------------------------------------------------------
# Implied by the spec but absent from its test plan: no RuntimeWarning
# ---------------------------------------------------------------------------
def _degenerate_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A (4, 6) input surface whose first row is sigma=0 and first column T=0."""
    S = np.broadcast_to(np.linspace(80.0, 120.0, 6)[None, :], (4, 6))
    T = np.broadcast_to(np.array([0.0, 0.5, 1.0, 1.0, 2.0, 2.0])[None, :], (4, 6))
    sigma = np.broadcast_to(np.array([0.0, 0.1, 0.3, 0.8])[:, None], (4, 6))
    return S, T, sigma


def test_no_runtime_warning_on_a_grid_with_zero_time_or_zero_vol() -> None:
    """Trap A: a grid holding T=0 / sigma=0 cells must emit NO RuntimeWarning.

    Division by ``sigma*sqrt(T)`` is the obvious trap, and a plain ``np.where``
    over an UNSANITIZED denominator still evaluates the division on every cell
    and warns.  ``warnings.simplefilter("error", RuntimeWarning)`` turns any
    such warning into a failure, and ``np.errstate(..., "raise")`` catches the
    hardware-level flags numpy would otherwise report only once per session.
    Underflow stays ignored: a deep-OTM probability legitimately underflows.
    """
    S, T, sigma = _degenerate_surface()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            d1, d2 = d1_d2(S, 100.0, T, 0.05, sigma)
            call = call_price(S, 100.0, T, 0.05, sigma)
            put = put_price(S, 100.0, T, 0.05, sigma)

    assert np.all(np.isfinite(call)) and np.all(np.isfinite(put))
    assert not np.any(np.isnan(d1)) and not np.any(np.isnan(d2))


def test_prices_are_never_nan_on_a_wide_sweep() -> None:
    """Spec 5.4: edge cases are 'handled explicitly rather than ... NaN'."""
    S = np.array([1e-6, 0.01, 1.0, 50.0, 100.0, 1e3, 1e6])[:, None, None, None]
    T = np.array([0.0, 1e-9, 0.01, 1.0, 30.0])[None, :, None, None]
    sigma = np.array([0.0, 1e-9, 0.01, 0.2, 5.0])[None, None, :, None]
    r = np.array([-0.05, 0.0, 0.05, 0.25])[None, None, None, :]

    call = call_price(S, 100.0, T, r, sigma)
    put = put_price(S, 100.0, T, r, sigma)

    assert not np.any(np.isnan(call)) and not np.any(np.isnan(put))
    assert np.all(np.isfinite(call)) and np.all(np.isfinite(put))
    assert np.all(call >= 0.0) and np.all(put >= 0.0)
