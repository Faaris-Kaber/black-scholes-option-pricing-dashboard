"""Exercise the real Streamlit entry point, including widget reruns and errors.

The numerical modules have their own unit tests. These checks cover the UI
boundary: percentages, per-contract units, scenario controls and recovery from
an invalid range. They deliberately render the real heat maps.
"""

from __future__ import annotations

from html import unescape
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.black_scholes import call_price, put_price
from src.grid import price_grid


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def dashboard() -> AppTest:
    """Start a fresh browser session while allowing cold image rendering time."""
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def _assert_rendered(dashboard: AppTest) -> None:
    assert not dashboard.exception, [item.value for item in dashboard.exception]
    assert not dashboard.error, [item.value for item in dashboard.error]
    assert len(dashboard.get("image")) == 2


def _card_text(dashboard: AppTest, side: str) -> str:
    cards = [
        item.value
        for item in dashboard.markdown
        if f">{side} value</div>" in item.value
    ]
    assert len(cards) == 1
    return unescape(re.sub(r"<[^>]*>", " ", cards[0]))


def _card_price(dashboard: AppTest, side: str) -> float:
    match = re.search(r"\$([\d,]+\.\d{2})", _card_text(dashboard, side))
    assert match is not None
    return float(match.group(1).replace(",", ""))


def _images(dashboard: AppTest) -> tuple[str, ...]:
    return tuple(image.proto.imgs[0].url for image in dashboard.get("image"))


def _range_slider(dashboard: AppTest, label: str):
    return next(widget for widget in dashboard.slider if widget.label == label)


def _signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def test_default_dashboard_has_reference_prices_and_two_heatmaps(dashboard: AppTest) -> None:
    _assert_rendered(dashboard)
    assert _card_price(dashboard, "Call") == 10.45
    assert _card_price(dashboard, "Put") == 5.57
    assert dashboard.number_input(key="spot").value == 100.0
    assert dashboard.number_input(key="strike").value == 100.0
    assert dashboard.number_input(key="tte").value == 1.0
    assert dashboard.number_input(key="rate_pct").value == 5.0
    assert dashboard.slider(key="vol_pct").value == 20.0
    assert dashboard.slider(key="grid_n").value == 10
    assert _range_slider(dashboard, "Spot range ($)").value == (80.0, 120.0)
    assert _range_slider(dashboard, "Volatility range").value == (10.0, 30.0)
    assert dashboard.radio(key="mode").value == "Option Value"
    assert dashboard.selectbox(key="multiplier").value == 1
    assert not dashboard.button
    assert {item.value for item in dashboard.sidebar.subheader} == {
        "Contract", "Market", "Heat Map Range", "My Position"
    }
    assert not any(item.value == "Greeks" for item in dashboard.subheader)


def test_widget_reruns_convert_percentages_and_update_prices(dashboard: AppTest) -> None:
    original_images = _images(dashboard)
    dashboard.number_input(key="spot").set_value(125.0)
    dashboard.number_input(key="strike").set_value(110.0)
    dashboard.number_input(key="tte").set_value(0.25)
    dashboard.number_input(key="rate_pct").set_value(-2.0)
    dashboard.slider(key="vol_pct").set_value(35.0)
    dashboard.run()

    _assert_rendered(dashboard)
    inputs = (125.0, 110.0, 0.25, -0.02, 0.35)
    assert _card_price(dashboard, "Call") == round(float(call_price(*inputs)), 2)
    assert _card_price(dashboard, "Put") == round(float(put_price(*inputs)), 2)
    assert _images(dashboard) != original_images
    assert _range_slider(dashboard, "Spot range ($)").value == (100.0, 150.0)
    assert _range_slider(dashboard, "Volatility range").value == (17.5, 52.5)
    assert any("91" in item.value for item in dashboard.sidebar.caption)


def test_pnl_purchases_and_contract_multiplier_reach_cards_and_summaries(
    dashboard: AppTest,
) -> None:
    dashboard.number_input(key="call_paid").set_value(8.0)
    dashboard.number_input(key="put_paid").set_value(4.0)
    dashboard.radio(key="mode").set_value("P&L")
    dashboard.run()
    _assert_rendered(dashboard)
    share_images = _images(dashboard)

    spots = np.linspace(80.0, 120.0, 10)
    vols = np.linspace(0.10, 0.30, 10)
    grids = [
        price_grid(spots, vols, 100.0, 1.0, 0.05, "call") - 8.0,
        price_grid(spots, vols, 100.0, 1.0, 0.05, "put") - 4.0,
    ]
    for multiplier in (1, 100):
        if multiplier == 100:
            dashboard.selectbox(key="multiplier").set_value(multiplier).run()
            _assert_rendered(dashboard)
        expected_metrics = []
        for grid in grids:
            expected_metrics.extend([
                _signed_money(float(grid.max()) * multiplier),
                _signed_money(float(grid.min()) * multiplier),
                f"{np.mean(grid > 0) * 100:.1f}%",
            ])
        assert [item.value for item in dashboard.metric] == expected_metrics
        for side, pricing, paid in (
            ("Call", call_price, 8.0),
            ("Put", put_price, 4.0),
        ):
            value = float(pricing(100.0, 100.0, 1.0, 0.05, 0.20))
            assert _card_price(dashboard, side) == round(value, 2)
            assert _signed_money((value - paid) * multiplier) in _card_text(dashboard, side)
    assert _images(dashboard) != share_images

    # Returning to value mode clears position summaries and keeps scalar values.
    dashboard.radio(key="mode").set_value("Option Value").run()
    _assert_rendered(dashboard)
    assert not dashboard.metric
    assert _card_price(dashboard, "Call") == 10.45
    assert _card_price(dashboard, "Put") == 5.57


@pytest.mark.parametrize("spot,call,put", [(75.0, 0.0, 25.0), (125.0, 25.0, 0.0)])
def test_maturity_zero_renders_intrinsic_prices_and_heatmaps(
    dashboard: AppTest, spot: float, call: float, put: float
) -> None:
    dashboard.number_input(key="tte").set_value(0.0)
    dashboard.number_input(key="spot").set_value(spot).run()
    _assert_rendered(dashboard)
    assert _card_price(dashboard, "Call") == call
    assert _card_price(dashboard, "Put") == put


@pytest.mark.parametrize(
    "label,collapsed,recovered",
    [
        ("Spot range ($)", (100.0, 100.0), (80.0, 120.0)),
        ("Volatility range", (20.0, 20.0), (10.0, 30.0)),
    ],
)
def test_collapsed_range_reports_error_and_recovers(
    dashboard: AppTest, label: str, collapsed: tuple, recovered: tuple
) -> None:
    _range_slider(dashboard, label).set_value(collapsed).run()
    assert not dashboard.exception
    assert dashboard.error
    assert all("Traceback" not in item.value for item in dashboard.error)
    _range_slider(dashboard, label).set_value(recovered).run()
    _assert_rendered(dashboard)


def test_minimum_spot_has_noncollapsed_scenario_range(dashboard: AppTest) -> None:
    dashboard.number_input(key="spot").set_value(0.01).run()
    _assert_rendered(dashboard)
    low, high = _range_slider(dashboard, "Spot range ($)").value
    assert 0 < low < high
    assert _card_price(dashboard, "Call") == 0.0
    assert _card_price(dashboard, "Put") == 95.11


@pytest.mark.parametrize("rate,volatility", [(-5.0, 1.0), (25.0, 100.0)])
def test_rate_and_volatility_widget_boundaries_are_valid(
    dashboard: AppTest, rate: float, volatility: float
) -> None:
    dashboard.number_input(key="rate_pct").set_value(rate)
    dashboard.slider(key="vol_pct").set_value(volatility).run()
    _assert_rendered(dashboard)
    inputs = (100.0, 100.0, 1.0, rate / 100, volatility / 100)
    assert _card_price(dashboard, "Call") == round(float(call_price(*inputs)), 2)
    assert _card_price(dashboard, "Put") == round(float(put_price(*inputs)), 2)
    low, high = _range_slider(dashboard, "Volatility range").value
    assert 1.0 <= low < high


def test_grid_size_limits_rerender_heatmaps_without_leaking_figures(dashboard: AppTest) -> None:
    figure_numbers = plt.get_fignums()
    images_by_size = {}
    for grid_n in (5, 15):
        dashboard.slider(key="grid_n").set_value(grid_n).run()
        _assert_rendered(dashboard)
        images_by_size[grid_n] = _images(dashboard)
        assert plt.get_fignums() == figure_numbers
        assert _card_price(dashboard, "Call") == 10.45
    assert images_by_size[5] != images_by_size[15]
