"""Regression checks on rendered colors, axis orientation and break-even lines."""

import io
import warnings
from xml.etree import ElementTree

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.contour import QuadContourSet

from src.plotting import figure_svg, pnl_heatmap, value_heatmap


def test_svg_serialization_keeps_currency_text_and_restores_settings() -> None:
    original_fonttype = mpl.rcParams["svg.fonttype"]
    fig = pnl_heatmap([[-1, 1], [2, 3]], [80, 120], [0.1, 0.3], "Call P&L ($) · paid $8.00")
    result = figure_svg(fig)
    root = ElementTree.fromstring(result)
    texts = [element.text for element in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "+3.00" in texts
    assert "Call P&L ($) · paid $8.00" in texts
    assert mpl.rcParams["svg.fonttype"] == original_fonttype


@pytest.mark.parametrize("renderer", [value_heatmap, pnl_heatmap])
def test_volatility_increases_bottom_to_top_without_transposing_spot(renderer) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]])
    original = values.copy()
    fig = renderer(values, [80, 100, 120], [0.1, 0.3], "Surface")
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_array(), values[::-1])
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["80.00", "100.00", "120.00"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["30.0%", "10.0%"]
    assert ax.transData.transform((0.5, 0.5))[1] > ax.transData.transform((0.5, 1.5))[1]
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("values", [np.zeros((2, 2)), np.array([[-5, 0], [0, 2]])])
def test_actual_zero_cell_color_is_pure_white(values) -> None:
    fig = pnl_heatmap(values, [80, 120], [0.1, 0.3], "P&L")
    FigureCanvasAgg(fig).draw()
    mesh = fig.axes[0].collections[0]
    assert isinstance(mesh.norm, TwoSlopeNorm)
    assert mesh.norm(0.0) == 0.5
    colors = mesh.get_facecolors().reshape(2, 2, 4)
    np.testing.assert_array_equal(colors[values[::-1] == 0], 1.0)


def test_all_positive_pnl_cells_are_never_red() -> None:
    values = np.array([[0.01, 1.0], [2.0, 100.0]])
    fig = pnl_heatmap(values, [80, 120], [0.1, 0.3], "P&L")
    FigureCanvasAgg(fig).draw()
    colors = fig.axes[0].collections[0].get_facecolors()
    assert np.all(colors[:, 1] >= colors[:, 0])
    assert np.all(colors[:, 1] >= colors[:, 2])


def test_all_negative_pnl_cells_are_never_green() -> None:
    fig = pnl_heatmap([[-1, -10], [-2, -20]], [80, 120], [0.1, 0.3], "P&L")
    mesh = fig.axes[0].collections[0]
    colors = mesh.cmap(mesh.norm(mesh.get_array()))
    assert np.all(colors[..., 0] > colors[..., 1])


def test_signed_annotations_round_to_two_places_and_remove_negative_zero() -> None:
    fig = pnl_heatmap([[-1.8, -0.001], [2.45, 0]], [80, 120], [0.1, 0.3], "P&L")
    cell_labels = [text.get_text() for text in fig.axes[0].texts[:4]]
    assert cell_labels == ["+2.45", "+0.00", "-1.80", "+0.00"]
    assert fig.axes[1].get_ylabel() == "P&L ($)"


def test_zero_contour_matches_flipped_cell_centers() -> None:
    fig = pnl_heatmap([[-1, -1], [3, 3]], [80, 120], [0.1, 0.3], "P&L")
    contours = [artist for artist in fig.axes[0].collections if isinstance(artist, QuadContourSet)]
    assert len(contours) == 1
    assert list(contours[0].levels) == [0.0]
    vertices = np.concatenate([path.vertices for path in contours[0].get_paths()])
    np.testing.assert_allclose(vertices[:, 1], 1.25, rtol=0, atol=1e-12)
    assert np.all((vertices[:, 0] >= 0.5) & (vertices[:, 0] <= 1.5))
    assert any(text.get_text() == "break-even" for text in fig.axes[0].texts)


@pytest.mark.parametrize("values", [np.zeros((2, 2)), np.ones((2, 2)), -np.ones((2, 2)),
                                     np.array([[0, 0], [1, 2]])])
def test_no_warning_or_contour_when_surface_has_no_crossing(values) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = pnl_heatmap(values, [80, 120], [0.1, 0.3], "P&L")
        fig.savefig(io.BytesIO(), format="png")
    assert not any(isinstance(artist, QuadContourSet) for artist in fig.axes[0].collections)


def test_one_row_pnl_surface_skips_two_dimensional_contour() -> None:
    fig = pnl_heatmap([[-1, 1]], [80, 120], [0.2], "P&L")
    assert len(fig.axes[0].collections) == 1


def test_raw_value_uses_viridis_and_unsigned_currency() -> None:
    fig = value_heatmap([[1, 2], [3, 4]], [80, 120], [0.1, 0.3], "Values")
    assert fig.axes[0].collections[0].cmap.name == "viridis"
    assert [text.get_text() for text in fig.axes[0].texts] == ["3.00", "4.00", "1.00", "2.00"]
    assert fig.axes[1].get_ylabel() == "Option value ($)"
    assert fig.axes[1].yaxis.get_major_formatter()(1.5) == "1.50"


def test_currency_in_titles_is_literal_text() -> None:
    fig = pnl_heatmap([[1, 2], [3, 4]], [80, 120], [0.1, 0.3], "Call P&L ($) · paid $8.00")
    assert fig.axes[0].title.get_parse_math() is False


def test_contract_colorbar_currency_labels_fit_reserved_margins() -> None:
    fig = pnl_heatmap([[-1000, 0], [1000, 2000]], [80, 120], [0.1, 0.3], "Contract P&L")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    colorbar = fig.axes[1]
    labels = [colorbar.yaxis.label, *colorbar.get_yticklabels()]
    assert all(label.get_window_extent(canvas.get_renderer()).x1 < fig.bbox.x1 for label in labels)


@pytest.mark.parametrize("renderer", [value_heatmap, pnl_heatmap])
@pytest.mark.parametrize("values, spots, vols", [
    ([[1, 2]], [80], [0.1]), ([1, 2], [80, 120], [0.1]),
    ([[1, np.nan]], [80, 120], [0.1]), ([[1, 2]], [[80, 120]], [0.1]),
    ([[1, 2]], [120, 80], [0.1]), ([[1, 2]], [80, 80], [0.1]),
    ([[1, 2]], [80, 120], [-0.1]), ([[1, 2]], [0, 120], [0.1]),
])
def test_invalid_surface_inputs_raise_value_error(renderer, values, spots, vols) -> None:
    with pytest.raises(ValueError):
        renderer(values, spots, vols, "Invalid")
