"""Greeks tests - the sensitivities the spec's own test plan (section 10) omits.

Spec section 5.5 defines the Greeks but section 10 never tests them, so these
tests pin the requirements the spec *implies*:

* no Greek is ever ``NaN`` or ``inf``, anywhere on a wide (S, sigma, T) sweep
  that includes the ``T = 0`` and ``sigma = 0`` boundaries (trap E);
* put-call parity for delta: ``delta_call - delta_put == 1``;
* gamma and vega are identical between call and put;
* no ``RuntimeWarning`` escapes on a degenerate grid (trap A);
* the display scalings and their LABELS are the ones trap D mandates - rho is
  divided by 100, which is per 1 percentage point, NOT per basis point, and
  must never be labelled "basis point".

Reference values are recomputed here from the textbook closed forms with
``scipy.stats.norm`` rather than imported from ``src``, so the assertions are
an independent check rather than a restatement of the implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from src.black_scholes import call_price, put_price
from src.greeks import (
    GREEK_NAMES,
    GREEK_UNITS,
    call_greeks,
    greeks,
    put_greeks,
    scale_for_display,
)

# Display divisors mandated by trap D.  Duplicated deliberately: the test is
# the specification of the scaling, not a mirror of the module's constant.
EXPECTED_DIVISORS: dict[str, float] = {
    "delta": 1.0,
    "gamma": 1.0,
    "vega": 100.0,
    "theta": 365.0,
    "rho": 100.0,
}


def _reference_greeks(
    S: float, K: float, T: float, r: float, sigma: float, kind: str
) -> dict[str, float]:
    """Independent textbook implementation, valid only for T > 0 and sigma > 0."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = K * np.exp(-r * T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    diffusion = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if kind == "call":
        return {
            "delta": norm.cdf(d1),
            "gamma": gamma,
            "vega": vega,
            "theta": diffusion - r * disc * norm.cdf(d2),
            "rho": K * T * np.exp(-r * T) * norm.cdf(d2),
        }
    return {
        "delta": norm.cdf(d1) - 1.0,
        "gamma": gamma,
        "vega": vega,
        "theta": diffusion + r * disc * norm.cdf(-d2),
        "rho": -K * T * np.exp(-r * T) * norm.cdf(-d2),
    }


def _wide_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A broadcastable (S, T, sigma, r) sweep that INCLUDES T=0 and sigma=0."""
    S = np.array([0.01, 1.0, 10.0, 50.0, 99.999, 100.0, 100.001, 250.0, 5_000.0])
    T = np.array([0.0, 1e-9, 1e-4, 0.25, 1.0, 5.0, 30.0])
    sigma = np.array([0.0, 1e-9, 0.01, 0.2, 1.0, 5.0])
    r = np.array([-0.05, 0.0, 0.05, 0.25])
    return (
        S[:, None, None, None],
        T[None, :, None, None],
        sigma[None, None, :, None],
        r[None, None, None, :],
    )


# ---------------------------------------------------------------------------
# closed-form correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_an_independent_textbook_implementation(
    kind: str, atm: dict[str, float]
) -> None:
    """Spec 5.5: every Greek matches the closed form recomputed from scratch."""
    produced = greeks(**atm, option_type=kind)
    expected = _reference_greeks(kind=kind, **atm)

    assert set(produced) == set(GREEK_NAMES)
    for name in GREEK_NAMES:
        assert float(produced[name]) == pytest.approx(expected[name], rel=1e-12)


def test_atm_call_delta_reference_value(atm: dict[str, float]) -> None:
    """Sanity anchor: ATM call delta = N(d1) = 0.6368 for the T-2 case."""
    assert float(call_greeks(**atm)["delta"]) == pytest.approx(0.6368, abs=5e-5)
    assert float(put_greeks(**atm)["delta"]) == pytest.approx(-0.3632, abs=5e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_and_vega_agree_with_a_finite_difference_of_the_price(
    kind: str,
) -> None:
    """The Greeks must differentiate the SAME model the pricer evaluates.

    A central difference with a step of 1e-4 has truncation error O(h^2)=1e-8
    and round-off O(eps*price/h)=~1e-11, so 1e-6 is a comfortable, honest bound.
    """
    S, K, T, r, sigma = 103.0, 100.0, 0.7, 0.03, 0.27
    pricer = call_price if kind == "call" else put_price
    g = greeks(S, K, T, r, sigma, kind)

    h = 1e-4
    fd_delta = (
        float(pricer(S + h, K, T, r, sigma)) - float(pricer(S - h, K, T, r, sigma))
    ) / (2 * h)
    fd_vega = (
        float(pricer(S, K, T, r, sigma + h)) - float(pricer(S, K, T, r, sigma - h))
    ) / (2 * h)

    assert float(g["delta"]) == pytest.approx(fd_delta, abs=1e-6)
    assert float(g["vega"]) == pytest.approx(fd_vega, abs=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("r", [-0.05, 0.05, 0.25])
def test_gamma_theta_and_rho_agree_with_price_finite_differences(
    kind: str, r: float
) -> None:
    """Verify curvature, time-decay sign and rate sensitivity independently."""
    S = np.array([80.0, 100.0, 120.0])
    K, T, sigma = 100.0, 0.7, 0.3
    pricer = call_price if kind == "call" else put_price
    produced = greeks(S, K, T, r, sigma, kind)
    spot_step, time_step, rate_step = 0.01, 1e-5, 1e-5
    gamma = (
        pricer(S + spot_step, K, T, r, sigma)
        - 2.0 * pricer(S, K, T, r, sigma)
        + pricer(S - spot_step, K, T, r, sigma)
    ) / spot_step**2
    theta = (
        pricer(S, K, T - time_step, r, sigma)
        - pricer(S, K, T + time_step, r, sigma)
    ) / (2.0 * time_step)
    rho = (
        pricer(S, K, T, r + rate_step, sigma)
        - pricer(S, K, T, r - rate_step, sigma)
    ) / (2.0 * rate_step)
    np.testing.assert_allclose(produced["gamma"], gamma, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(produced["theta"], theta, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(produced["rho"], rho, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# parity and shared-Greek identities
# ---------------------------------------------------------------------------
def test_put_call_parity_for_delta_is_exactly_one() -> None:
    """delta_call - delta_put == 1, EXACTLY, everywhere on the wide sweep.

    Differentiating parity ``C - P = S - K*e^(-rT)`` with respect to S gives
    exactly 1.  It is asserted bit-exact rather than with a tolerance because
    the identity must survive the degenerate branches too: at T=0 or sigma=0
    both deltas collapse to the same step function, and an implementation that
    resolved the at-the-forward tie differently for calls and puts would break
    parity by a full unit on those cells.
    """
    S, T, sigma, r = _wide_sweep()
    dc = call_greeks(S, 100.0, T, r, sigma)["delta"]
    dp = put_greeks(S, 100.0, T, r, sigma)["delta"]

    np.testing.assert_array_equal(dc - dp, np.ones_like(dc))


def test_vega_and_gamma_are_identical_between_call_and_put() -> None:
    """Spec 5.5: gamma and vega are '(same for both)' - bit-for-bit."""
    S, T, sigma, r = _wide_sweep()
    c = call_greeks(S, 100.0, T, r, sigma)
    p = put_greeks(S, 100.0, T, r, sigma)

    np.testing.assert_array_equal(c["gamma"], p["gamma"])
    np.testing.assert_array_equal(c["vega"], p["vega"])


def test_rho_and_theta_are_not_identical_between_call_and_put(
    atm: dict[str, float],
) -> None:
    """Converse of the above: the type-dependent Greeks must actually differ."""
    c = call_greeks(**atm)
    p = put_greeks(**atm)
    assert float(c["rho"]) != float(p["rho"])
    assert float(c["theta"]) != float(p["theta"])
    assert float(c["rho"]) > 0.0 > float(p["rho"])


def test_greek_sign_and_range_conventions() -> None:
    """Call delta in [0,1], put delta in [-1,0], gamma >= 0, vega >= 0."""
    S, T, sigma, r = _wide_sweep()
    c = call_greeks(S, 100.0, T, r, sigma)
    p = put_greeks(S, 100.0, T, r, sigma)

    assert np.all((c["delta"] >= 0.0) & (c["delta"] <= 1.0))
    assert np.all((p["delta"] >= -1.0) & (p["delta"] <= 0.0))
    assert np.all(c["gamma"] >= 0.0)
    assert np.all(c["vega"] >= 0.0)


# ---------------------------------------------------------------------------
# boundaries (trap E) - never NaN, never inf
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_are_never_nan_or_inf_on_a_wide_sweep(kind: str) -> None:
    """Trap E: no Greek is NaN or inf anywhere, including T=0 and sigma=0.

    gamma and vega divide by ``sigma*sqrt(T)`` and theta by ``2*sqrt(T)``; at
    the boundary those are 0/0.  The sweep spans 9 spots x 7 maturities x 6
    volatilities x 4 rates and deliberately includes the exactly-at-the-money
    knife edge, which is where an unmasked 0/0 shows up first.
    """
    S, T, sigma, r = _wide_sweep()
    produced = greeks(S, 100.0, T, r, sigma, kind)

    for name in GREEK_NAMES:
        values = np.asarray(produced[name])
        assert not np.any(np.isnan(values)), f"{name} produced NaN"
        assert np.all(np.isfinite(values)), f"{name} produced inf"


def test_delta_at_zero_time_is_the_step_function() -> None:
    """Trap E: at T=0 delta is 1.0 if S > K else 0.0 for a call (0/-1 for a put)."""
    S = np.array([50.0, 99.999, 100.0, 100.001, 250.0])
    c = call_greeks(S, 100.0, 0.0, 0.05, 0.2)
    p = put_greeks(S, 100.0, 0.0, 0.05, 0.2)

    np.testing.assert_array_equal(c["delta"], np.array([0.0, 0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(p["delta"], np.array([-1.0, -1.0, -1.0, 0.0, 0.0]))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_gamma_vega_theta_are_zero_at_zero_time(kind: str) -> None:
    """Trap E: gamma / vega / theta return exactly 0.0 at T = 0, never inf."""
    S = np.array([50.0, 100.0, 250.0])
    g = greeks(S, 100.0, 0.0, 0.05, 0.2, kind)

    np.testing.assert_array_equal(g["gamma"], np.zeros(3))
    np.testing.assert_array_equal(g["vega"], np.zeros(3))
    np.testing.assert_array_equal(g["theta"], np.zeros(3))
    # Rho carries a factor T, so it is 0 at expiry by construction.
    np.testing.assert_array_equal(g["rho"], np.zeros(3))


def test_zero_vol_theta_is_the_pure_carry_term() -> None:
    """Trap E: at sigma=0 (T>0) theta keeps only the carry term, with no blow-up.

    That term is the exact derivative of the deterministic payoff
    ``max(S - K*e^(-rT), 0)`` with respect to calendar time, so it is checkable
    in closed form: ``-r*K*e^(-rT)`` for an in-the-money call, 0 otherwise.
    """
    K, T, r = 100.0, 1.0, 0.05
    disc = K * np.exp(-r * T)

    itm_call = call_greeks(150.0, K, T, r, 0.0)
    otm_call = call_greeks(50.0, K, T, r, 0.0)
    assert float(itm_call["theta"]) == pytest.approx(-r * disc, rel=1e-12)
    assert float(otm_call["theta"]) == 0.0
    assert np.isfinite(float(itm_call["gamma"]))
    assert float(itm_call["gamma"]) == 0.0

    itm_put = put_greeks(50.0, K, T, r, 0.0)
    assert float(itm_put["theta"]) == pytest.approx(r * disc, rel=1e-12)


def test_no_runtime_warning_on_a_greeks_grid_with_zero_time_or_zero_vol() -> None:
    """Trap A/E: the degenerate Greek branches must warn-free, for both types.

    ``sigma*sqrt(T)`` is 0 on the first row and the first column of this grid;
    a masked-but-still-evaluated division would raise a RuntimeWarning here.
    """
    S = np.broadcast_to(np.linspace(80.0, 120.0, 6)[None, :], (4, 6))
    T = np.broadcast_to(np.array([0.0, 0.25, 1.0, 1.0, 2.0, 2.0])[None, :], (4, 6))
    sigma = np.broadcast_to(np.array([0.0, 0.05, 0.3, 0.9])[:, None], (4, 6))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            produced = [
                call_greeks(S, 100.0, T, 0.05, sigma),
                put_greeks(S, 100.0, T, 0.05, sigma),
            ]

    for bundle in produced:
        for name in GREEK_NAMES:
            assert np.all(np.isfinite(bundle[name])), f"{name} is not finite"


# ---------------------------------------------------------------------------
# shape / vectorization contract
# ---------------------------------------------------------------------------
def test_greeks_broadcast_and_return_the_broadcast_shape() -> None:
    """Every Greek comes back with the broadcast shape of the inputs (rule 2)."""
    S = np.linspace(80.0, 120.0, 7)[None, :]
    sigma = np.linspace(0.1, 0.5, 4)[:, None]
    bundle = call_greeks(S, 100.0, 1.0, 0.05, sigma)

    for name in GREEK_NAMES:
        assert bundle[name].shape == (4, 7)

    scalar = call_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
    for name in GREEK_NAMES:
        assert np.ndim(scalar[name]) == 0


def test_greeks_vectorized_matches_a_scalar_loop(rng: np.random.Generator) -> None:
    """Vectorized evaluation is elementwise-equal to a scalar loop."""
    spots = np.sort(rng.uniform(60.0, 140.0, 6))
    vols = np.sort(rng.uniform(0.05, 0.8, 5))
    S_grid, vol_grid = np.meshgrid(spots, vols)

    vectorized = call_greeks(S_grid, 100.0, 1.0, 0.05, vol_grid)
    for i, v in enumerate(vols):
        for j, s in enumerate(spots):
            cell = call_greeks(s, 100.0, 1.0, 0.05, v)
            for name in GREEK_NAMES:
                assert vectorized[name][i, j] == cell[name]


# ---------------------------------------------------------------------------
# display scaling and units (trap D)
# ---------------------------------------------------------------------------
def test_greek_names_and_unit_labels_are_complete() -> None:
    """GREEK_NAMES and GREEK_UNITS must agree, with a label for every Greek."""
    assert GREEK_NAMES == ("delta", "gamma", "vega", "theta", "rho")
    assert set(GREEK_UNITS) == set(GREEK_NAMES)
    assert all(isinstance(label, str) and label for label in GREEK_UNITS.values())


def test_rho_is_not_labelled_per_basis_point(atm: dict[str, float]) -> None:
    """Trap D: rho is divided by 100, which is per 1pp - NEVER per basis point.

    The spec (5.5) says "Rho per-1-basis-point-move (/100)".  Those two halves
    contradict each other: a basis point is 0.0001, so rho per bp would be
    ``rho / 10_000``.  The /100 scaling is the useful one, but printing the
    words "basis point" next to it would be a factor-of-100 lie on screen.
    """
    label = GREEK_UNITS["rho"].lower()
    assert "basis point" not in label
    assert "bps" not in label
    assert "1pp" in label or "percentage point" in label or "1%" in label

    raw = call_greeks(**atm)["rho"]
    displayed = scale_for_display(call_greeks(**atm))["rho"]
    assert float(displayed) == pytest.approx(float(raw) / 100.0, rel=1e-15)


def test_scale_for_display_applies_the_documented_divisors(
    atm: dict[str, float],
) -> None:
    """Trap D: vega/100, theta/365, rho/100; delta and gamma unscaled."""
    raw = call_greeks(**atm)
    displayed = scale_for_display(raw)

    assert set(displayed) == set(GREEK_NAMES)
    for name, divisor in EXPECTED_DIVISORS.items():
        assert float(displayed[name]) == pytest.approx(
            float(raw[name]) / divisor, rel=1e-15
        )

    # Spot-check the labels the UI prints next to those figures.
    assert GREEK_UNITS["vega"] == "per 1 vol point"
    assert GREEK_UNITS["theta"] == "per calendar day"


def test_scale_for_display_is_pure_and_does_not_mutate_its_input(
    atm: dict[str, float],
) -> None:
    """NFR-3: the display scaling is a pure function - no aliasing, no mutation."""
    raw = call_greeks(np.linspace(90.0, 110.0, 5), 100.0, 1.0, 0.05, 0.2)
    before = {name: np.array(value, copy=True) for name, value in raw.items()}

    displayed = scale_for_display(raw)

    assert displayed is not raw
    for name in GREEK_NAMES:
        np.testing.assert_array_equal(raw[name], before[name])
        assert displayed[name] is not raw[name]


def test_scale_for_display_preserves_array_shapes() -> None:
    """Scaling is elementwise: shapes survive it untouched."""
    raw = call_greeks(
        np.linspace(90.0, 110.0, 7)[None, :],
        100.0,
        1.0,
        0.05,
        np.linspace(0.1, 0.4, 3)[:, None],
    )
    displayed = scale_for_display(raw)
    for name in GREEK_NAMES:
        assert displayed[name].shape == (3, 7)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [{"T": -1.0}, {"sigma": -0.2}, {"S": 0.0}, {"S": -1.0}, {"K": 0.0}, {"K": -1.0}],
)
def test_greeks_reject_out_of_range_inputs(
    atm: dict[str, float], kwargs: dict[str, float]
) -> None:
    """T-11 applies to the Greeks too: the same domain, the same ValueError."""
    bad = {**atm, **kwargs}
    with pytest.raises(ValueError):
        call_greeks(**bad)
    with pytest.raises(ValueError):
        put_greeks(**bad)
    with pytest.raises(ValueError):
        greeks(**bad, option_type="call")


def test_greeks_dispatcher_rejects_an_unknown_option_type(
    atm: dict[str, float],
) -> None:
    """The dispatcher must fail loudly rather than silently pricing a call."""
    with pytest.raises(ValueError):
        greeks(**atm, option_type="strangle")

    for spelling in ("call", "CALL", " Call "):
        assert float(greeks(**atm, option_type=spelling)["delta"]) == float(
            call_greeks(**atm)["delta"]
        )
