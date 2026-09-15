"""Measure complete Streamlit server reruns, including both SVG charts.

Run ``python scripts/benchmark.py`` from any directory. Cold import/startup is
reported separately; unique strike inputs force new grids and chart rendering.
This measures server execution, not network transfer or browser paint time.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from time import perf_counter

from streamlit.testing.v1 import AppTest


def run_benchmark() -> dict[str, float]:
    """Return median and worst full rerun times in milliseconds for 10x10 grids."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)

    def timed_run() -> float:
        start = perf_counter()
        app.run()
        elapsed = (perf_counter() - start) * 1000
        if len(app.exception) or len(app.error):
            raise RuntimeError("Dashboard did not render successfully during benchmark.")
        if len(app.get("image")) != 2:
            raise RuntimeError("Both charts must render for a valid benchmark.")
        return elapsed

    results = {"cold_start_ms": timed_run()}
    for mode in ("Option Value", "P&L"):
        app.radio(key="mode").set_value(mode)
        app.number_input(key="call_paid").set_value(10.0)
        app.number_input(key="put_paid").set_value(6.0)
        timed_run()
        samples = []
        for strike in (101.25, 102.5, 103.75, 105.0, 106.25):
            app.number_input(key="strike").set_value(strike)
            samples.append(timed_run())
        label = "value" if mode == "Option Value" else "pnl"
        results[f"{label}_uncached_median_ms"] = median(samples)
        results[f"{label}_uncached_max_ms"] = max(samples)
        results[f"{label}_cached_median_ms"] = median(timed_run() for _ in range(3))
    return results


if __name__ == "__main__":
    for label, milliseconds in run_benchmark().items():
        print(f"{label}: {milliseconds:.1f}")
