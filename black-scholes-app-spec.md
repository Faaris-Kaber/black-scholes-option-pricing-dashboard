# Black-Scholes Option Pricing Dashboard — Technical Spec

**Version:** 1.0
**Date:** 2026-09-14
**Status:** Implemented and validated locally; Streamlit Community Cloud publication pending repository/account setup. See [README](README.md) for run instructions and verification.
**Stack:** Python 3.11+ · NumPy · SciPy · Streamlit · Matplotlib/Seaborn

---

## 1. Overview

An interactive web dashboard that prices European call and put options using the Black-Scholes-Merton model and visualizes how those prices — and a user's actual P&L — respond to shocks in spot price and volatility.

**Purpose:** demonstrate (a) working knowledge of option pricing theory, (b) clean numerical Python, and (c) the ability to ship an interactive analytical tool end-to-end.

## 2. Objectives

| # | Objective | Measure of success |
|---|---|---|
| O-1 | Correct BSM pricing | Matches a reference implementation to ≤ 1e-6 absolute on a 500-case grid |
| O-2 | Immediate interactivity | Full recalculation + re-render in < 500 ms |
| O-3 | Intuitive sensitivity view | A non-quant can read the heat map without explanation |
| O-4 | Personalized P&L | User can enter their fill price and see break-even geography |

## 3. Scope

**In scope**
- European calls and puts, non-dividend-paying underlying
- Closed-form analytical pricing (no simulation, no trees)
- Single-leg positions only
- Long positions (bought options) for the P&L module

**Out of scope (v1)**
- American / Bermudan exercise
- Dividend yield, early exercise premium
- Multi-leg strategies (spreads, straddles)
- Live market data feeds
- Persistence / user accounts

---

## 4. Architecture

```
app.py                 # Streamlit entry point — layout, widgets, orchestration
src/
  black_scholes.py     # Pricing engine (pure functions, zero UI imports)
  greeks.py            # First/second-order sensitivities
  grid.py              # Vectorized scenario matrix construction
  pnl.py               # P&L transformation layer
  plotting.py          # Heat map rendering
tests/
  test_black_scholes.py
  test_grid.py
requirements.txt
README.md
```

**Design rule:** `src/` contains no Streamlit imports. The engine must be importable and testable from a bare Python REPL. This keeps the pricing logic reusable and makes unit testing trivial.

**Data flow:**
```
sidebar inputs → validation → scalar price (2 cards)
                            → meshgrid(S, σ) → vectorized price matrix
                                             → P&L transform → heat map
```

---

## 5. Core Engine (FR-1)

### 5.1 Formulas

```
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T

Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·N(−d1)
```

Where `N(·)` is the standard normal CDF (`scipy.stats.norm.cdf`).

**Validation identity:** put-call parity must hold to floating-point tolerance.
```
C − P = S − K·e^(−rT)
```
Assert this in the test suite — it catches most sign errors immediately.

### 5.2 Public interface

```python
def d1_d2(S, K, T, r, sigma) -> tuple[np.ndarray, np.ndarray]: ...

def call_price(S, K, T, r, sigma) -> np.ndarray: ...
def put_price(S, K, T, r, sigma)  -> np.ndarray: ...
```

All functions must be **NumPy-vectorized** — accept scalars or arrays of any broadcastable shape and return the same shape. This is what makes the heat map a single call instead of a nested loop.

### 5.3 Parameters

| Symbol | Name | Type | Valid range | Default |
|---|---|---|---|---|
| `S` | Underlying spot price | float | > 0 | 100.00 |
| `K` | Strike price | float | > 0 | 100.00 |
| `T` | Time to expiry (years) | float | ≥ 0 | 1.00 |
| `r` | Risk-free rate (annual, decimal) | float | −0.05 to 0.25 | 0.05 |
| `σ` | Volatility (annual, decimal) | float | ≥ 0 | 0.20 |

### 5.4 Edge cases

These break the formula via division by zero and must be handled explicitly rather than allowed to return `NaN`:

| Condition | Required behavior |
|---|---|
| `T == 0` | Return intrinsic value: `max(S−K, 0)` / `max(K−S, 0)` |
| `σ == 0` | Return discounted deterministic payoff: `max(S − K·e^(−rT), 0)` |
| `T < 0` or `σ < 0` | Raise `ValueError` |
| `S ≤ 0` or `K ≤ 0` | Raise `ValueError` |
| Deep ITM/OTM (`\|d1\| > 8`) | Must not overflow; verify numerical stability |

### 5.5 Greeks (FR-1b, recommended)

Cheap to add, disproportionately valuable in an interview — they show you understand *why* the heat map shocks S and σ specifically.

```
Delta_call = N(d1)                       Delta_put = N(d1) − 1
Gamma      = φ(d1) / (S·σ·√T)            (same for both)
Vega       = S·φ(d1)·√T                  (same for both)
Theta_call = −S·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2)
Theta_put  = −S·φ(d1)·σ/(2√T) + r·K·e^(−rT)·N(−d2)
Rho_call   = K·T·e^(−rT)·N(d2)           Rho_put = −K·T·e^(−rT)·N(−d2)
```

**Display conventions:** Vega scaled to per-1-vol-point (÷100), Theta scaled to per-calendar-day (÷365), Rho per-1-percentage-point rate move (÷100, equivalent to 100 basis points). Label the units on screen — unscaled Greeks are a common tell. The original draft's “per basis point” label was inconsistent with ÷100; per basis point would require ÷10,000.

---

## 6. User Interface (FR-2)

### 6.1 Layout

- **Sidebar** — all inputs, grouped under headers: *Contract*, *Market*, *Heat Map Range*, *My Position*
- **Main panel, row 1** — two large metric cards: CALL value (green accent), PUT value (red accent)
- **Main panel, row 2** — Greeks table (if implemented)
- **Main panel, row 3** — side-by-side call and put heat maps in `st.columns(2)`

### 6.2 Widget mapping

| Input | Widget | Notes |
|---|---|---|
| Spot, Strike | `st.number_input` | `step=1.0`, `format="%.2f"` |
| Time to expiry | `st.number_input` | Accept years; show a days equivalent as caption |
| Volatility | `st.slider` | 0.01 → 1.00, `step=0.01`; display as % |
| Risk-free rate | `st.number_input` | Display as %, store as decimal |
| Heat map spot range | `st.slider` (range) | Min/max spot, defaults to ±20% of S |
| Heat map vol range | `st.slider` (range) | Defaults to 0.5σ → 1.5σ, floored at 0.01 |
| Purchase prices | `st.number_input` ×2 | See §8 |

### 6.3 Behavior

- Recalculation is automatic on widget change — **no submit button**. Streamlit's rerun model handles this natively.
- Percentage inputs display as percent, convert to decimal at the boundary. Never let a user type `5` and get 500% vol.
- Invalid input surfaces as `st.error` with a plain-language message; the app must not traceback.
- Render all currency to 2 decimal places.

---

## 7. Sensitivity Heat Map (FR-3)

| Spec | Value |
|---|---|
| Grid size | 10 × 10 (configurable 5–15) |
| X-axis | Underlying spot price, ascending left → right |
| Y-axis | Volatility, ascending **bottom → top** |
| Cell value | Option price at that (S, σ) pair |
| Annotation | Value printed in each cell, 2 dp |
| Color map | `viridis` for raw-value mode |

**Construction:** build with `np.meshgrid` over the two ranges and call the pricing function once on the full matrix. Do not loop.

```python
S_grid, vol_grid = np.meshgrid(spot_range, vol_range)
values = call_price(S_grid, K, T, r, vol_grid)
```

All other parameters (K, T, r) are held at their sidebar values. Make this explicit in a caption — the heat map is a **mark-to-market surface at fixed time**, not an expiry payoff diagram. That distinction is the single most common misreading.

**Performance:** wrap grid computation in `@st.cache_data` keyed on all inputs.

---

## 8. P&L Heat Map (FR-4 — primary upgrade)

### 8.1 Inputs

| Input | Widget | Default |
|---|---|---|
| Call purchase price | `st.number_input`, ≥ 0 | 0.00 |
| Put purchase price | `st.number_input`, ≥ 0 | 0.00 |
| Display mode | `st.radio` — *Option Value* / *P&L* | Option Value |
| Contract multiplier | `st.selectbox` — 1 (per share) / 100 (per contract) | 1 |

### 8.2 Transformation

```
PnL[i,j] = (option_value[i,j] − purchase_price) × multiplier
```

Long positions only in v1. Purchase price of 0 makes P&L identical to value — a reasonable, non-breaking default.

### 8.3 Color scale — the part that must be exactly right

Requirement: **red → white → green, with pure white pinned to P&L = 0.** A naive `cmap="RdYlGn"` normalizes to `[min, max]`, which puts the neutral color at the data midpoint rather than at break-even. If every scenario is profitable, a naive scale still paints the worst cells red. That is a correctness bug, not a cosmetic one.

Use a two-slope norm anchored at zero:

```python
from matplotlib.colors import TwoSlopeNorm

lim = max(abs(pnl.min()), abs(pnl.max()))
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
```

Guard the degenerate case where `lim == 0` (flat matrix) — `TwoSlopeNorm` raises if `vmin == vcenter`.

### 8.4 Additional display requirements

- Format cell annotations with explicit sign: `+2.45` / `−1.80`
- Colorbar labeled `P&L ($)`
- Overlay the break-even contour (`plt.contour(..., levels=[0])`) — the line separating profit from loss is the most decision-relevant feature on the chart
- Caption stating: *P&L assumes the position is marked at the current time to expiry, not held to expiration.*

---

## 9. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Full recalculation + render < 500 ms on a 10×10 grid |
| NFR-2 | No unhandled exception reaches the UI under any input in the valid ranges |
| NFR-3 | Engine functions are pure and side-effect free |
| NFR-4 | Type hints on every public function; docstrings state units (years, decimals) |
| NFR-5 | Deployable to Streamlit Community Cloud from a public repo with no config changes |
| NFR-6 | `requirements.txt` pins major versions |

---

## 10. Test Plan

| ID | Test | Expected |
|---|---|---|
| T-1 | Put-call parity across 500 random valid inputs | `\|C − P − (S − K·e^(−rT))\| < 1e-9` |
| T-2 | ATM call, S=K=100, T=1, r=0.05, σ=0.2 | ≈ 10.4506 |
| T-3 | ATM put, same inputs | ≈ 5.5735 |
| T-4 | `T = 0` | Returns exact intrinsic value |
| T-5 | `σ = 0` | Returns discounted deterministic payoff |
| T-6 | Deep OTM call (S=10, K=1000) | → 0, no `NaN` |
| T-7 | Vectorized call on (10,10) array | Returns (10,10), elementwise-equal to scalar loop |
| T-8 | Monotonicity | Call price strictly increasing in S and in σ |
| T-9 | P&L with purchase price 0 | Identical to value matrix |
| T-10 | All-positive P&L matrix | No cell above break-even renders red; P&L = 0 renders pure white |
| T-11 | Negative `T` / `σ` | Raises `ValueError` |

Reference values for T-2/T-3 are the standard textbook case — useful as a sanity anchor anyone can verify.

---

## 11. Build Order

1. `black_scholes.py` + tests T-1 through T-8 — engine correct before any UI exists
2. Minimal Streamlit app: inputs → two price cards
3. `grid.py` + raw-value heat maps
4. Greeks module and display table
5. P&L transform, zero-anchored diverging color scale, break-even contour
6. Polish: captions, input validation messages, README with a screenshot
7. Deploy to Streamlit Community Cloud

## 12. Future Enhancements

- **Implied volatility solver** — invert the model via Brent's method on an observed market price. This is the natural next step and the one most likely to come up in conversation, since it shows you understand that σ is the only unobservable input.
- Dividend yield `q` (Black-Scholes-Merton extension) — a one-line change to `d1` and the discount factors
- Short positions (`is_long` toggle, negates the P&L sign)
- Multi-leg strategy builder with aggregated payoff
- Monte Carlo pricer alongside the closed form, as a convergence cross-check
- Live spot prices via `yfinance`
- Binomial tree for American exercise comparison
