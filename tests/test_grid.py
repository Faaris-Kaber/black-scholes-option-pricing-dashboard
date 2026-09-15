"""Scenario surfaces retain pricing semantics and ascending (vol, spot) axes."""

import numpy as np
import pytest

from src.black_scholes import call_price, put_price
from src.grid import make_spot_range, make_vol_range, price_grid


@pytest.mark.parametrize("n", [5, 10, 15])
def test_axis_builders_cover_spec_sizes_and_endpoints(n: int) -> None:
    spots = make_spot_range(80, 120, n)
    vols = make_vol_range(0.001, 0.3, n)
    assert spots.shape == vols.shape == (n,)
    assert spots.dtype == vols.dtype == np.float64
    np.testing.assert_array_equal(spots[[0, -1]], [80, 120])
    np.testing.assert_array_equal(vols[[0, -1]], [0.01, 0.3])
    assert np.all(np.diff(spots) > 0) and np.all(np.diff(vols) > 0)


@pytest.mark.parametrize("side, scalar_price", [("call", call_price), ("put", put_price)])
@pytest.mark.parametrize("time", [0.0, 1.0])
def test_rectangular_surface_equals_scalar_pricing(side, scalar_price, time: float) -> None:
    spots = np.linspace(70, 130, 7)
    vols = np.array([0.0, 0.1, 0.3, 0.7])
    original_spots, original_vols = spots.copy(), vols.copy()
    actual = price_grid(spots, vols, 100, time, -0.01, side)
    expected = np.array([
        [scalar_price(spot, 100, time, -0.01, vol) for spot in spots]
        for vol in vols
    ])
    assert actual.shape == (4, 7)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(spots, original_spots)
    np.testing.assert_array_equal(vols, original_vols)


@pytest.mark.parametrize("side", ["call", "put"])
def test_surface_monotonicity_explains_axes(side: str) -> None:
    values = price_grid(np.linspace(80, 120, 10), np.linspace(0.1, 0.3, 10),
                        100, 1, 0.05, side)
    assert np.all(np.diff(values, axis=0) > 0)
    assert np.all(np.diff(values, axis=1) > 0) if side == "call" else np.all(np.diff(values, axis=1) < 0)


@pytest.mark.parametrize("builder, low, high", [
    (make_spot_range, 0, 100), (make_spot_range, 120, 80),
    (make_spot_range, 80, 80), (make_spot_range, np.nan, 120),
    (make_vol_range, -0.1, 0.3), (make_vol_range, 0, 0.01),
    (make_vol_range, 0.2, np.inf),
])
def test_invalid_axis_bounds_raise_value_error(builder, low, high) -> None:
    with pytest.raises(ValueError):
        builder(low, high, 10)


@pytest.mark.parametrize("n", [True, 1, 2.5, 201, np.nan, "ten"])
def test_invalid_grid_sizes_raise_value_error(n) -> None:
    with pytest.raises(ValueError):
        make_spot_range(80, 120, n)


@pytest.mark.parametrize("spots, vols", [
    ([], [0.1, 0.2]), ([[80, 120]], [0.1, 0.2]),
    ([120, 80], [0.1, 0.2]), ([80, 80], [0.1, 0.2]),
    ([80, 120], [0.2, 0.1]), ([80, 120], [0.1, 0.1]),
    ([80, np.inf], [0.1, 0.2]), ([80, 120], [-0.1, 0.2]),
])
def test_invalid_scenario_axes_raise_value_error(spots, vols) -> None:
    with pytest.raises(ValueError):
        price_grid(spots, vols, 100, 1, 0.05, "call")


@pytest.mark.parametrize("fixed", [(None, 1, 0.05), (100, [1], 0.05),
                                  (100, 1, [0.01, 0.05]), (100, -1, 0.05)])
def test_contract_parameters_must_be_valid_scalars(fixed) -> None:
    with pytest.raises(ValueError):
        price_grid([80, 120], [0.1, 0.3], *fixed, "call")
