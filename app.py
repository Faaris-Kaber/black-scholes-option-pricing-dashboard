"""Black-Scholes option pricing dashboard - Streamlit entry point (spec sections 6, 7, 8).

Run with::

    streamlit run app.py

This module is the ONLY place in the project that imports Streamlit: everything
under ``src/`` is a pure, UI-free engine that must stay importable from a bare
Python REPL (spec section 4).  The app layer is responsible for exactly four
things:

1. Reading inputs from the sidebar and converting them at the boundary
   (percentages in the UI -> DECIMALS for the engine; years for T).
2. Orchestrating the engine: scalar prices, scenario grids, and P&L.
3. Rendering: two value cards and side-by-side call/put heat maps.
4. Turning any ``ValueError`` the engine raises into a plain-language
   ``st.error`` banner.  No traceback may ever reach the UI (NFR-2).

Units at this boundary
----------------------
The sidebar shows volatility and the risk-free rate as PERCENTAGES because
that is how humans quote them; both are divided by 100 before they touch the
engine, which speaks decimals only (0.20 == 20%, never 20).  Time to expiry is
entered in YEARS with a calendar-day equivalent shown as a caption.

Caching (NFR-1)
---------------
Scenario matrices and rendered SVGs are cached separately. Streamlit hashes
NumPy arrays by their contents, so changing an axis, value, title or mode
invalidates the rendered chart. Cache sizes are bounded for long sessions.

Figure lifecycle
----------------
``src.plotting`` builds figures through ``matplotlib.figure.Figure`` (never
pyplot), and every figure created here is closed in a ``finally`` block the
moment it has been serialized, so no rerun can leak one. ``plt.show()`` is
never called.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib

# Streamlit renders figures server-side; pin the non-interactive backend before
# pyplot is imported so no GUI toolkit is ever touched.  plt.show() is never called.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

# Make ``src`` importable no matter what the working directory is (src/grid.py
# imports the package absolutely, as ``src.black_scholes``).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.black_scholes import call_price, put_price, validate_inputs  # noqa: E402
from src.grid import (  # noqa: E402
    DEFAULT_GRID_N,
    MIN_VOL,
    SPEC_MAX_GRID_N,
    SPEC_MIN_GRID_N,
    make_spot_range,
    make_vol_range,
    price_grid,
)
from src.plotting import figure_svg, pnl_heatmap, value_heatmap  # noqa: E402
from src.pnl import pnl_grid, pnl_summary  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PAGE_TITLE = "Black-Scholes Option Pricing Dashboard"

# Defaults from spec section 5.3 (r and sigma are shown as percentages).
DEFAULT_SPOT = 100.0
DEFAULT_STRIKE = 100.0
DEFAULT_T_YEARS = 1.0
DEFAULT_RATE_PCT = 5.0
DEFAULT_VOL_PCT = 20.0

# Sidebar bounds.  r is clamped to the valid range in spec section 5.3.
MIN_RATE_PCT, MAX_RATE_PCT = -5.0, 25.0
MIN_VOL_PCT = MIN_VOL * 100.0  # the 1% floor from spec section 6.2
MAX_VOL_PCT = 100.0
MAX_VOL_AXIS_PCT = 200.0  # 1.5 x the highest sigma the slider allows, rounded up
MAX_PRICE = 1_000_000.0
MAX_T_YEARS = 30.0

MODE_VALUE = "Option Value"
MODE_PNL = "P&L"

CALL_ACCENT = "#16a34a"
PUT_ACCENT = "#dc2626"

@dataclass(frozen=True)
class Inputs:
    """Everything the sidebar collected, already converted to ENGINE units.

    ``r`` and ``sigma`` are DECIMALS (0.05 == 5%), ``T`` is in YEARS, prices are
    in currency units and ``multiplier`` is 1 (per share) or 100 (per contract).
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    spot_low: float
    spot_high: float
    vol_low: float
    vol_high: float
    grid_n: int
    call_paid: float
    put_paid: float
    mode: str
    multiplier: int


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def money(value: float) -> str:
    """Format a currency amount to exactly 2 dp with thousands separators."""
    return f"${value:,.2f}"


def signed_money(value: float) -> str:
    """Format a P&L amount with an explicit sign, 2 dp (spec section 8.4)."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _days_caption(T: float) -> str:
    """Human-readable calendar/trading-day equivalent of a time in YEARS."""
    if T <= 0.0:
        return "This option expires now, so its value is based on the payoff today."
    return (
        f"About {T * 365.0:,.0f} calendar days "
        f"({T * 252.0:,.0f} trading days)"
    )


# ---------------------------------------------------------------------------
# cached computation (NFR-1)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def cached_axes(
    spot_low: float,
    spot_high: float,
    vol_low: float,
    vol_high: float,
    n: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the ``(spot_range, vol_range)`` axes as ascending tuples.

    Volatilities are DECIMALS. Tuples (not arrays) so the cached value stays
    trivially hashable/picklable; callers wrap them in ``np.asarray``.
    """
    spots = make_spot_range(spot_low, spot_high, n)
    vols = make_vol_range(vol_low, vol_high, n)
    return tuple(spots.tolist()), tuple(vols.tolist())


@st.cache_data(show_spinner=False, max_entries=128)
def cached_value_grid(
    spot_low: float,
    spot_high: float,
    vol_low: float,
    vol_high: float,
    n: int,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> np.ndarray:
    """Price the whole ``(n_vol, n_spot)`` scenario matrix, memoized on scalars.

    The returned grid is stored ascending-vol-first (row 0 = LOWEST volatility);
    the flip for display lives in ``src.plotting`` and nowhere else.

    Units: ``T`` in YEARS, ``r`` and the vol endpoints as DECIMALS.
    """
    spots = make_spot_range(spot_low, spot_high, n)
    vols = make_vol_range(vol_low, vol_high, n)
    return price_grid(spots, vols, K, T, r, option_type)


# ---------------------------------------------------------------------------
# HTML fragments (inline styles, so they survive with or without a <style> block)
# ---------------------------------------------------------------------------
def value_card_html(
    label: str,
    amount: float,
    accent: str,
    lines: list[str],
) -> str:
    """Render a compact, classic dashboard metric panel."""
    body = "".join(
        f'<div style="font-size:0.84rem;color:#6b7280;margin-top:0.18rem;">'
        f"{escape(line)}</div>"
        for line in lines
    )
    return (
        '<div style="border:1px solid #d5d9dd;border-top:4px solid '
        f'{accent};border-radius:3px;background:#ffffff;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.14);min-height:7.8rem;">'
        '<div style="padding:0.48rem 0.85rem;background:#f5f5f5;'
        'border-bottom:1px solid #dddddd;font-family:Arial,sans-serif;'
        'font-size:0.78rem;font-weight:700;color:#4b5563;'
        f'text-transform:uppercase;">{escape(label)}</div>'
        '<div style="padding:0.72rem 0.90rem 0.80rem;">'
        '<div style="font-family:Arial,sans-serif;font-size:2rem;font-weight:600;'
        f'line-height:1.15;color:#222222;font-variant-numeric:tabular-nums;">'
        f'{money(amount)}</div>{body}</div></div>'
    )


# ---------------------------------------------------------------------------
# sidebar (spec section 6.2)
# ---------------------------------------------------------------------------
def read_sidebar() -> Inputs:
    """Draw the four sidebar groups and return inputs converted to engine units.

    Percentages are divided by 100 HERE, at the boundary: the engine never sees
    a 5 that meant 5%.  No submit button - Streamlit reruns on every change
    (spec section 6.3).
    """
    sb = st.sidebar
    sb.title("Build your scenario")
    sb.caption("Adjust any value below and the dashboard will update automatically.")

    # -- Contract ----------------------------------------------------------
    sb.subheader("Contract")
    K = float(
        sb.number_input(
            "Strike price K ($)",
            min_value=0.01,
            max_value=MAX_PRICE,
            value=DEFAULT_STRIKE,
            step=1.0,
            format="%.2f",
            key="strike",
            help="The price at which the option lets you buy or sell the underlying.",
        )
    )
    T = float(
        sb.number_input(
            "Time to expiry (years)",
            min_value=0.0,
            max_value=MAX_T_YEARS,
            value=DEFAULT_T_YEARS,
            step=0.05,
            format="%.4f",
            key="tte",
            help="Enter years remaining. For example, 0.25 is about three months and 0 means today.",
        )
    )
    sb.caption(_days_caption(T))

    # -- Market ------------------------------------------------------------
    sb.subheader("Market")
    S = float(
        sb.number_input(
            "Spot price S ($)",
            min_value=0.01,
            max_value=MAX_PRICE,
            value=DEFAULT_SPOT,
            step=1.0,
            format="%.2f",
            key="spot",
            help="The underlying asset's price right now.",
        )
    )
    sigma_pct = float(
        sb.slider(
            "Volatility (annualized)",
            min_value=MIN_VOL_PCT,
            max_value=MAX_VOL_PCT,
            value=DEFAULT_VOL_PCT,
            step=1.0,
            format="%.0f%%",
            key="vol_pct",
            help="The market's expected annual price movement. Higher volatility usually makes options more valuable.",
        )
    )
    rate_pct = float(
        sb.number_input(
            "Risk-free rate r (%)",
            min_value=MIN_RATE_PCT,
            max_value=MAX_RATE_PCT,
            value=DEFAULT_RATE_PCT,
            step=0.25,
            format="%.2f",
            key="rate_pct",
            help="The annual risk-free rate, entered as a percentage. For example, enter 5 for 5%.",
        )
    )
    sigma = sigma_pct / 100.0
    r = rate_pct / 100.0

    # -- Heat Map Range ----------------------------------------------------
    sb.subheader("Heat Map Range")
    spot_step = max(0.01, round(S / 100.0, 2))
    spot_bound_low = max(0.01, round(S * 0.40, 2))
    spot_bound_high = max(spot_bound_low + 0.01, round(S * 1.60, 2))
    spot_default_low = max(spot_bound_low, round(S * 0.80, 2))  # -20% of S
    spot_default_high = min(
        spot_bound_high, max(spot_default_low + 0.01, round(S * 1.20, 2))
    )  # +20% of S, with distinct endpoints for prices near one cent
    # No key: the bounds move with S, so the widget must re-seed its defaults
    # instead of holding a stale value that now sits outside the range.
    spot_low, spot_high = sb.slider(
        "Spot range ($)",
        min_value=spot_bound_low,
        max_value=spot_bound_high,
        value=(spot_default_low, spot_default_high),
        step=spot_step,
        format="%.2f",
        help="Choose the lowest and highest spot prices to compare across the heat maps.",
    )

    vol_default_low = max(MIN_VOL_PCT, round(0.5 * sigma_pct * 2.0) / 2.0)
    vol_default_high = min(
        MAX_VOL_AXIS_PCT,
        max(vol_default_low + 0.5, round(1.5 * sigma_pct * 2.0) / 2.0),
    )
    vol_low_pct, vol_high_pct = sb.slider(
        "Volatility range",
        min_value=MIN_VOL_PCT,
        max_value=MAX_VOL_AXIS_PCT,
        value=(vol_default_low, vol_default_high),
        step=0.5,
        format="%.1f%%",
        help="Choose the lowest and highest volatility levels to compare.",
    )

    grid_n = int(
        sb.slider(
            "Grid size (n x n)",
            min_value=SPEC_MIN_GRID_N,
            max_value=SPEC_MAX_GRID_N,
            value=DEFAULT_GRID_N,
            step=1,
            key="grid_n",
            help="More cells show finer detail, while fewer cells are easier to scan.",
        )
    )

    # -- My Position (spec section 8.1) ------------------------------------
    sb.subheader("My Position")
    call_paid = float(
        sb.number_input(
            "Call purchase price ($)",
            min_value=0.0,
            max_value=MAX_PRICE,
            value=0.00,
            step=0.05,
            format="%.2f",
            key="call_paid",
            help="What you paid for the call per share. Leave this at $0 to view its full market value as profit.",
        )
    )
    put_paid = float(
        sb.number_input(
            "Put purchase price ($)",
            min_value=0.0,
            max_value=MAX_PRICE,
            value=0.00,
            step=0.05,
            format="%.2f",
            key="put_paid",
            help="What you paid for the put per share. Leave this at $0 to view its full market value as profit.",
        )
    )
    mode = str(
        sb.radio(
            "Display mode",
            (MODE_VALUE, MODE_PNL),
            index=0,
            key="mode",
            horizontal=True,
        )
    )
    multiplier = int(
        sb.selectbox(
            "Contract multiplier",
            (1, 100),
            index=0,
            format_func=lambda m: "1 (per share)" if m == 1 else "100 (per contract)",
            key="multiplier",
        )
    )
    sb.caption(
        "P&L shows what your purchased option would gain or lose if you sold it "
        "at the value calculated here."
    )

    return Inputs(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        spot_low=float(spot_low),
        spot_high=float(spot_high),
        vol_low=float(vol_low_pct) / 100.0,
        vol_high=float(vol_high_pct) / 100.0,
        grid_n=grid_n,
        call_paid=call_paid,
        put_paid=put_paid,
        mode=mode,
        multiplier=multiplier,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def cached_heatmap_image(
    values: np.ndarray,
    spots: np.ndarray,
    vols: np.ndarray,
    title: str,
    is_pnl: bool,
) -> str:
    """Render SVG once per input set; spots are dollars, vols are decimals."""
    chart = pnl_heatmap if is_pnl else value_heatmap
    fig = chart(values, spots, vols, title)
    try:
        return figure_svg(fig)
    finally:
        plt.close(fig)


def render_value_cards(inp: Inputs, call_value: float, put_value: float) -> None:
    """Main row 1: the two large accented value cards (spec section 6.1)."""
    is_pnl = inp.mode == MODE_PNL
    left, right = st.columns(2, gap="large")

    for column, label, value, paid, accent in (
        (left, "Call value", call_value, inp.call_paid, CALL_ACCENT),
        (right, "Put value", put_value, inp.put_paid, PUT_ACCENT),
    ):
        lines = [
            f"{money(value)} per share"
            if inp.multiplier == 1
            else f"{money(value)} per share  ·  {money(value * inp.multiplier)} per contract"
        ]
        if is_pnl:
            position = (value - paid) * inp.multiplier
            lines.append(
                f"Paid {money(paid)}  ->  P&L {signed_money(position)} at these inputs"
            )
        column.markdown(
            value_card_html(label, value, accent, lines),
            unsafe_allow_html=True,
        )

    st.caption(
        f"Based on a {money(inp.S)} spot price, {money(inp.K)} strike, "
        f"{inp.T * 365.0:,.0f} days remaining, {inp.sigma * 100.0:.1f}% volatility, "
        f"and a {inp.r * 100.0:.2f}% risk-free rate."
    )


def render_heatmaps(
    inp: Inputs,
    spots: np.ndarray,
    vols: np.ndarray,
    call_grid: np.ndarray,
    put_grid: np.ndarray,
) -> None:
    """Main row 3: side-by-side call and put heat maps (spec sections 7 and 8)."""
    is_pnl = inp.mode == MODE_PNL
    st.subheader("P&L heat maps" if is_pnl else "Sensitivity heat maps")

    st.caption(
        f"Each cell answers: “What would this option be worth if spot and volatility "
        f"moved here?” Spot increases from left to right, and volatility increases "
        f"from bottom to top. Strike ({money(inp.K)}), time remaining "
        f"({inp.T * 365.0:,.0f} days), and rate ({inp.r * 100.0:.2f}%) stay unchanged."
    )
    if is_pnl:
        st.caption(
            f"This is the gain or loss if you sold the option with the same time "
            f"remaining—not the payoff at expiration. Results are shown "
            f"{'per share' if inp.multiplier == 1 else 'per 100-share contract'}. "
            "White means break-even, and the dashed line separates gains from losses. "
            "The profitable-scenario percentage describes these cells; it is not a forecast."
        )

    left, right = st.columns(2, gap="large")
    for column, side, grid, paid, accent in (
        (left, "Call", call_grid, inp.call_paid, CALL_ACCENT),
        (right, "Put", put_grid, inp.put_paid, PUT_ACCENT),
    ):
        with column:
            if is_pnl:
                position_pnl = pnl_grid(grid, paid, inp.multiplier)
                suffix = "" if inp.multiplier == 1 else f" x{inp.multiplier}"
                st.image(
                    cached_heatmap_image(
                        position_pnl,
                        spots,
                        vols,
                        f"{side} P&L ($){suffix}  ·  paid {money(paid)}",
                        True,
                    ),
                    width="stretch",
                )
                render_pnl_summary(position_pnl)
            else:
                st.image(
                    cached_heatmap_image(grid, spots, vols, f"{side} value ($)", False),
                    width="stretch",
                )
    if not is_pnl:
        st.caption(
            "These are estimated sale values with the same time remaining—not expiration "
            "payoffs. Brighter cells represent higher option values."
        )


def render_pnl_summary(position_pnl: np.ndarray) -> None:
    """Print the ``pnl_summary`` figures underneath a P&L heat map (section 8)."""
    summary = pnl_summary(position_pnl)
    best, worst, share = st.columns(3)
    best.metric("Best scenario", signed_money(summary["max_profit"]))
    worst.metric("Worst scenario", signed_money(summary["max_loss"]))
    share.metric("Scenarios profitable", f"{summary['pct_profitable']:.1f}%")
    if np.all(position_pnl == 0):
        st.caption("Every scenario is exactly at break-even.")
    elif np.min(position_pnl) == 0 or np.max(position_pnl) == 0:
        st.caption("Some scenarios are at break-even; there is no profit-to-loss crossing.")
    elif not summary["breakeven_exists"]:
        st.caption(
            "No break-even line on this grid: every scenario shown lands on the same "
            "side of zero."
        )


def render(inp: Inputs) -> None:
    """Compute and draw the whole dashboard. Raises ``ValueError`` on bad input."""
    validate_inputs(inp.S, inp.K, inp.T, inp.r, inp.sigma)

    call_value = float(call_price(inp.S, inp.K, inp.T, inp.r, inp.sigma))
    put_value = float(put_price(inp.S, inp.K, inp.T, inp.r, inp.sigma))
    render_value_cards(inp, call_value, put_value)

    st.divider()
    # Compute scenario axes and matrices through bounded caches.
    spot_tuple, vol_tuple = cached_axes(
        inp.spot_low, inp.spot_high, inp.vol_low, inp.vol_high, inp.grid_n
    )
    call_grid = cached_value_grid(
        inp.spot_low,
        inp.spot_high,
        inp.vol_low,
        inp.vol_high,
        inp.grid_n,
        inp.K,
        inp.T,
        inp.r,
        "call",
    )
    put_grid = cached_value_grid(
        inp.spot_low,
        inp.spot_high,
        inp.vol_low,
        inp.vol_high,
        inp.grid_n,
        inp.K,
        inp.T,
        inp.r,
        "put",
    )
    render_heatmaps(
        inp,
        np.asarray(spot_tuple, dtype=float),
        np.asarray(vol_tuple, dtype=float),
        call_grid,
        put_grid,
    )


def main() -> None:
    """Configure the page, read the sidebar, and render - never tracebacks (NFR-2)."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(PAGE_TITLE)
    st.caption(
        "Explore how spot price and volatility affect European call and put values. "
        "Change any setting and the results update automatically."
    )

    try:
        inp = read_sidebar()
        render(inp)
    except ValueError as exc:
        # Plain-language message straight from the engine; no traceback, ever.
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - last-resort guard, NFR-2
        logging.getLogger(__name__).exception("Dashboard rendering failed")
        st.error(
            "We couldn't calculate this scenario. Check the values in the sidebar "
            "and try again."
        )

    st.divider()
    st.caption(
        "Model assumptions: European exercise, no dividends, constant volatility and "
        "interest rate, and one purchased option at a time."
    )


if __name__ == "__main__":
    main()
