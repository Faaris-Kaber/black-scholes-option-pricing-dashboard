# Black-Scholes Option Pricing Dashboard

A simple Streamlit app for pricing European call and put options. It also shows how changes in spot price and volatility affect option value and the profit or loss on a position.

## Examples

### Option value

![Option value heat maps](https://github.com/Faaris-Kaber/black-scholes-option-pricing-dashboard/releases/download/screenshots-v1/option-value-example.png)

### Position P&L and break-even

![Position P&L heat maps with break-even lines](https://github.com/Faaris-Kaber/black-scholes-option-pricing-dashboard/releases/download/screenshots-v1/pnl-break-even-example.png)

## Run it locally

You need Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` if it does not open automatically.

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## How to use it

1. Enter the strike price and time remaining.
2. Enter the current spot price, volatility, and risk-free rate.
3. Review the calculated call and put values.
4. Adjust the heat map ranges to test different spot and volatility scenarios.
5. Enter the price you paid under My Position and switch the display to P&L.

Rates and volatility are entered as percentages. Enter `5` for 5 percent and `20` for 20 percent.

## Reading the heat maps

Spot price moves from left to right. Volatility moves from bottom to top. Each cell shows the option value for that combination.

Strike, time remaining, and the risk-free rate stay fixed across the map. The results are current estimated values, not expiration payoffs or market forecasts.

In P&L mode:

- Green means profit
- Red means loss
- White means break-even
- The dashed line marks the break-even boundary

The profitable scenario percentage only describes the cells shown. It is not the probability of making money.

## My Position

Enter the premium paid per share for the call or put.

```text
P&L = (current option value - purchase price) x contract multiplier
```

Use a multiplier of 1 for per-share results or 100 for one standard equity option contract.

## Model assumptions

The app uses the Black-Scholes model and assumes:

- European exercise
- No dividends
- Constant volatility and interest rates
- One long option position
- No fees or transaction costs

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\benchmark.py
```

The test suite checks pricing accuracy, put-call parity, edge cases, heat map behavior, P&L calculations, and the Streamlit interface.

## Main files

- `app.py` contains the Streamlit interface
- `src/black_scholes.py` contains the pricing model
- `src/grid.py` builds the scenario grids
- `src/pnl.py` calculates position profit and loss
- `src/plotting.py` creates the heat maps
- `tests/` contains the automated tests

See [black-scholes-app-spec.md](black-scholes-app-spec.md) for the full technical specification.
