"""Long-position profit and loss uses per-share fills and contract scaling."""

import numpy as np
import pytest

from src.pnl import pnl_grid, pnl_summary


def test_zero_purchase_price_preserves_values_without_aliasing() -> None:
    values = np.arange(100.0).reshape(10, 10)
    result = pnl_grid(values, 0)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize("multiplier", [1, 100])
def test_fill_is_subtracted_before_contract_scaling(multiplier: int) -> None:
    values = np.array([[8.0, 10.0], [12.0, 15.0]])
    original = values.copy()
    np.testing.assert_array_equal(
        pnl_grid(values, 10, multiplier),
        np.array([[-2.0, 0.0], [2.0, 5.0]]) * multiplier,
    )
    np.testing.assert_array_equal(values, original)


def test_scalar_position_uses_same_units() -> None:
    assert float(pnl_grid(12.5, 10, 100)) == 250


@pytest.mark.parametrize("purchase_price", [-1, np.nan, np.inf, [1, 2], None, True])
def test_invalid_fills_raise_value_error(purchase_price) -> None:
    with pytest.raises(ValueError):
        pnl_grid([[1.0]], purchase_price)


@pytest.mark.parametrize("multiplier", [0, -1, 10, 1.5, np.inf, "100", True, [100]])
def test_only_spec_contract_multipliers_are_accepted(multiplier) -> None:
    with pytest.raises(ValueError):
        pnl_grid([[1.0]], 0, multiplier)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], ["invalid"]])
def test_nonfinite_or_nonnumeric_values_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        pnl_grid(values, 0)


def test_unrepresentable_contract_pnl_raises_without_overflow_warning() -> None:
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="too large"):
            pnl_grid([1e308], 0, 100)


def test_summary_preserves_loss_sign_and_excludes_breakeven_from_profits() -> None:
    assert pnl_summary([[-1, 0], [2, 4]]) == {
        "max_profit": 4.0, "max_loss": -1.0,
        "pct_profitable": 50.0, "breakeven_exists": True,
    }


@pytest.mark.parametrize("values, profitable", [([[0, 0]], 0), ([[1, 3]], 100), ([[-3, -1]], 0)])
def test_uniform_sign_summary_has_no_crossing(values, profitable: int) -> None:
    summary = pnl_summary(values)
    assert summary["pct_profitable"] == profitable
    assert summary["breakeven_exists"] is False


@pytest.mark.parametrize("values", [[], [np.nan]])
def test_summary_rejects_missing_values(values) -> None:
    with pytest.raises(ValueError):
        pnl_summary(values)
