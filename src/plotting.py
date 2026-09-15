"""Matplotlib heat maps for option values and long-position P&L.

Inputs are stored as ``(volatility, spot)`` with both axes ascending. Only this
module reverses rows so volatility increases bottom to top. Spot prices and
values use currency units; annualized volatility uses decimals (0.20 = 20%).

Figures have reserved margins and no automatic layout engine: serializing a
PNG without ``bbox_inches='tight'`` needs one draw. No figure is registered with
pyplot, and the caller owns its lifecycle. There are no Streamlit imports.
"""

from __future__ import annotations

from io import StringIO
from threading import RLock
from typing import Final

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from numpy.typing import ArrayLike, NDArray

__all__ = ["FIGSIZE", "VALUE_CMAP", "PNL_CMAP", "value_heatmap", "pnl_heatmap", "figure_svg"]

FIGSIZE: Final[tuple[float, float]] = (6.0, 5.0)
VALUE_CMAP: Final[str] = "viridis"

# An odd lookup-table size includes the exact midpoint. Matplotlib's default
# 256 entries otherwise turn normalized 0.5 into a slightly green off-white.
PNL_CMAP: Final[Colormap] = LinearSegmentedColormap.from_list(
    "bs_pnl_rwg",
    [(0.00, "#67000d"), (0.25, "#cb181d"), (0.50, "#ffffff"),
     (0.75, "#238b45"), (1.00, "#00441b")],
    N=257,
)

_MIN_PNL_LIM: Final[float] = 1e-9
_SVG_LOCK = RLock()


def figure_svg(fig: Figure) -> str:
    """Serialize a Figure as SVG with searchable text and no extra layout draw.

    Currency values and decimal-volatility labels retain their displayed units.
    Text stays as text rather than thousands of glyph paths, reducing server
    render time while remaining sharp at any browser zoom. The lock protects
    Matplotlib's temporary SVG setting across concurrent Streamlit sessions;
    the original settings are restored before returning. The caller owns fig.
    """
    with _SVG_LOCK, mpl.rc_context({"svg.fonttype": "none"}):
        output = StringIO()
        fig.savefig(output, format="svg", metadata={"Date": None})
    return output.getvalue()


def _prepare(
    grid: ArrayLike,
    spot_range: ArrayLike,
    vol_range: ArrayLike,
    what: str,
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    """Validate finite values and ascending axes, then reverse display rows."""
    try:
        values = np.asarray(grid, dtype=np.float64)
        spots = np.asarray(spot_range, dtype=np.float64)
        vols = np.asarray(vol_range, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Heat-map values and axes must contain numbers.") from None

    if values.ndim != 2:
        raise ValueError(f"{what} must be a two-dimensional matrix.")
    for axis, name in ((spots, "Spot"), (vols, "Volatility")):
        if axis.ndim != 1 or not axis.size:
            raise ValueError(f"{name} range must be a nonempty one-dimensional sequence.")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} range must contain only finite numbers.")
        if np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} range must be strictly ascending.")
    if np.any(spots <= 0.0):
        raise ValueError("Spot prices must be greater than zero.")
    if np.any(vols < 0.0):
        raise ValueError("Volatility cannot be negative.")
    if values.shape != (vols.size, spots.size):
        raise ValueError(
            f"{what} has shape {values.shape}; expected ({vols.size}, {spots.size}). "
            "Rows represent volatility and columns represent spot price."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} must contain only finite numbers.")

    return (
        values[::-1, :],
        [f"{spot:.2f}" for spot in spots],
        [f"{vol * 100.0:.1f}%" for vol in vols[::-1]],
    )


def _annot_size(n_rows: int, n_cols: int, longest: int) -> float:
    """Keep signed values legible without overlapping adjacent cells."""
    n = max(n_rows, n_cols)
    preferred = 9.0 if n <= 8 else 8.0 if n <= 11 else 7.0 if n <= 13 else 6.0
    # Main axes span 3.84 inches. Reserve a little horizontal cell padding.
    return min(preferred, 3.84 * 72.0 / n_cols / (max(longest, 1) * 0.65))


def _heatmap(
    display: NDArray[np.float64],
    xticklabels: list[str],
    yticklabels: list[str],
    title: str,
    *,
    cmap: str | Colormap,
    norm: Normalize,
    signed: bool,
    colorbar_label: str,
) -> tuple[Figure, Axes]:
    """Construct an annotated heat map without eagerly drawing the canvas."""
    n_rows, n_cols = display.shape
    fig = Figure(figsize=FIGSIZE)
    ax = fig.add_axes((0.145, 0.18, 0.64, 0.72))
    cax = fig.add_axes((0.815, 0.18, 0.025, 0.72))
    mesh = ax.pcolormesh(
        display, cmap=cmap, norm=norm, linewidths=0.5, edgecolors="white",
        shading="flat",
    )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, labels=xticklabels)
    ax.set_yticks(np.arange(n_rows) + 0.5, labels=yticklabels)
    ax.tick_params(axis="both", labelsize=8, length=0)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Spot price ($)", fontsize=10)
    ax.set_ylabel("Volatility (annualized)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10, parse_math=False)

    colorbar = fig.colorbar(mesh, cax=cax, format=FormatStrFormatter("%.2f"))
    colorbar.set_label(colorbar_label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_visible(False)

    rounded = np.round(display, 2) + 0.0  # Avoid a misleading "-0.00".
    fmt = "+.2f" if signed else ".2f"
    labels = [format(value, fmt) for value in rounded.flat]
    fontsize = _annot_size(n_rows, n_cols, max(map(len, labels)))

    # Compute text contrast before the first canvas draw. Calling facecolors
    # here would require an eager draw and double the work on every rerun.
    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgb = np.asarray(colormap(norm(display)))[..., :3]
    linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear_rgb @ np.array([0.2126, 0.7152, 0.0722])
    for (row, col), label in zip(np.ndindex(display.shape), labels):
        ax.text(
            col + 0.5, row + 0.5, label,
            ha="center", va="center", fontsize=fontsize,
            color="white" if luminance[row, col] < 0.179 else "#111111",
        )
    return fig, ax


def value_heatmap(
    values: ArrayLike,
    spot_range: ArrayLike,
    vol_range: ArrayLike,
    title: str,
    *,
    cmap: str = VALUE_CMAP,
) -> Figure:
    """Render option values in currency units using the viridis color map.

    ``values`` has shape ``(len(vol_range), len(spot_range))``. Spot prices
    ascend left to right; annualized decimal volatilities ascend bottom to top.
    The caller holds strike, time in years and annual decimal rate fixed.
    All inputs must be finite, with positive spots and nonnegative volatility.
    The returned Figure is independent of pyplot and belongs to the caller.
    """
    display, xs, ys = _prepare(values, spot_range, vol_range, "The value grid")
    fig, _ = _heatmap(
        display, xs, ys, title, cmap=cmap,
        norm=Normalize(vmin=float(display.min()), vmax=float(display.max())),
        signed=False, colorbar_label="Option value ($)",
    )
    return fig


def pnl_heatmap(
    pnl: ArrayLike,
    spot_range: ArrayLike,
    vol_range: ArrayLike,
    title: str,
) -> Figure:
    """Render signed P&L in currency units with pure white at exactly zero.

    ``pnl`` has shape ``(len(vol_range), len(spot_range))``. Annualized decimal
    volatility ascends bottom to top and spot prices ascend left to right.
    A symmetric TwoSlopeNorm maps losses to red and profits to green, even
    when every scenario has the same sign. A flat-zero grid remains white.
    A labelled zero contour is added only for grids containing both signs.
    Inputs must be finite; the returned Figure belongs to the caller.
    """
    display, xs, ys = _prepare(pnl, spot_range, vol_range, "The P&L grid")
    lo, hi = float(display.min()), float(display.max())
    lim = max(abs(lo), abs(hi), _MIN_PNL_LIM)
    fig, ax = _heatmap(
        display, xs, ys, title, cmap=PNL_CMAP,
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        signed=True, colorbar_label="P&L ($)",
    )
    _draw_breakeven_contour(ax, display, lo, hi)
    return fig


def _draw_breakeven_contour(
    ax: Axes,
    display: NDArray[np.float64],
    lo: float,
    hi: float,
) -> None:
    """Draw a zero contour on the same cell centers as the flipped matrix."""
    n_rows, n_cols = display.shape
    if min(n_rows, n_cols) < 2 or not lo < 0.0 < hi:
        return
    contour = ax.contour(
        np.arange(n_cols) + 0.5,
        np.arange(n_rows) + 0.5,
        display,
        levels=[0.0], colors="#111111", linewidths=1.6, linestyles="--",
    )
    ax.clabel(
        contour, fmt={0.0: "break-even"}, fontsize=7,
        inline=True, inline_spacing=4,
    )
