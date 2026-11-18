"""
The tools.mapping module contains functions for creating maps with geospatial
raster and vector data using rasterio, geopandas, and matplotlib.
"""

# pylint: disable=too-many-lines
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib
import matplotlib.axes
import matplotlib.colors
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    TextArea,
    VPacker,
)
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Affine2D
import numpy as np
import pyproj
import rasterio
import rasterio.shutil
from rasterio.warp import transform_bounds

import rsgispy
import rsgispy.imageutils.metadata
import rsgispy.imageutils.warp
import rsgispy.tools.files
from rsgispy.tools._legend import (
    _measure_axis_inches,
    _style_legend,
    build_legend_handles,
    calc_categorical_legend_layout,
    create_companion_legend_axis,
    resolve_legend_entries,
    save_standalone_legend,
)
import rsgispy.tools.plotting
import rsgispy.tools.projection

#: Scale bar style: solid horizontal bar (default).
SCALEBAR_STYLE_SOLID = "solid"
#: Scale bar style: alternating black/white segmented bar with tick labels.
SCALEBAR_STYLE_ALTERNATING = "alternating"
#: Scale bar style: horizontal line with vertical ticks and labels.
SCALEBAR_STYLE_TICKS = "ticks"

#: Scale bar unit: metres (``"m"``).
SCALEBAR_UNIT_METRES = "m"
#: Scale bar unit: kilometres (``"km"``).
SCALEBAR_UNIT_KILOMETRES = "km"

#: North arrow style: classic half-shaded notched cartographic triangle pointer.
NORTH_ARROW_STYLE_TRIANGLE = "triangle"
#: North arrow style: sleek vertical shaft with arrowhead.
NORTH_ARROW_STYLE_ARROW = "arrow"
#: North arrow style: four-point shaded compass rose.
NORTH_ARROW_STYLE_COMPASS = "compass"
#: North arrow style: minimalist outlined chevron pointer.
NORTH_ARROW_STYLE_SIMPLE = "simple"

_SCALEBAR_STYLES = {
    SCALEBAR_STYLE_SOLID,
    SCALEBAR_STYLE_ALTERNATING,
    SCALEBAR_STYLE_TICKS,
}
_NORTH_ARROW_STYLES = {
    NORTH_ARROW_STYLE_TRIANGLE,
    NORTH_ARROW_STYLE_ARROW,
    NORTH_ARROW_STYLE_COMPASS,
    NORTH_ARROW_STYLE_SIMPLE,
}
_LOC_ALIASES = {
    "top right": "upper right",
    "top left": "upper left",
    "bottom left": "lower left",
    "bottom right": "lower right",
    "top center": "upper center",
    "top centre": "upper center",
    "bottom center": "lower center",
    "bottom centre": "lower center",
    "center": "center",
    "centre": "center",
    "left": "center left",
    "centre left": "center left",
    "centre right": "center right",
}

__all__ = [
    "SCALEBAR_STYLE_SOLID",
    "SCALEBAR_STYLE_ALTERNATING",
    "SCALEBAR_STYLE_TICKS",
    "SCALEBAR_UNIT_METRES",
    "SCALEBAR_UNIT_KILOMETRES",
    "NORTH_ARROW_STYLE_TRIANGLE",
    "NORTH_ARROW_STYLE_ARROW",
    "NORTH_ARROW_STYLE_COMPASS",
    "NORTH_ARROW_STYLE_SIMPLE",
    "MapScaleBar",
    "MapNorthArrow",
    "add_scale_bar",
    "add_north_arrow",
    "calc_y_fig_size",
    "define_axis_extent",
    "define_map_tick_spacing",
    "get_overview_info",
    "add_overview_maps",
    "add_contextily_basemap",
    "create_wmts_img_map",
    "draw_bboxes_to_axis",
    "add_axis_label",
    "create_vec_lyr_map",
    "create_raster_img_map",
    "create_thematic_raster_map",
    "create_choropleth_vec_lyr_map",
    "create_raster_cmap_img_map",
    "create_vec_pt_density_map",
    "calc_categorical_legend_layout",
    "create_categorical_legend_patches",
    "add_categorical_legend",
    "add_categorical_legend_axis",
    "save_categorical_legend_img",
]


def _get_vec_epsg(
    gp_vec: Union[gpd.GeoDataFrame, gpd.GeoSeries],
) -> Optional[int]:
    """Return EPSG integer code for a GeoDataFrame/GeoSeries, or ``None``."""
    if gp_vec.crs is None:
        return None
    return gp_vec.crs.to_epsg()


def _subset_vec_bbox(
    gp_vec: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    bbox: Tuple[float, float, float, float],
) -> Union[gpd.GeoDataFrame, gpd.GeoSeries]:
    """Spatially subset a GeoDataFrame/GeoSeries using a buffered bounding box."""
    min_x_sub = math.floor(bbox[0] - 4)
    max_x_sub = math.ceil(bbox[1] + 4)
    min_y_sub = math.floor(bbox[2] - 4)
    max_y_sub = math.ceil(bbox[3] + 4)
    return gp_vec.cx[min_x_sub:max_x_sub, min_y_sub:max_y_sub]


def _resolve_mpl_location(loc: str) -> str:
    """Validate and normalise a Matplotlib legend-style placement string."""
    if not isinstance(loc, str):
        raise rsgispy.RSGISPyException(
            f"Location must be a string, got {type(loc).__name__}."
        )
    norm_loc = " ".join(loc.strip().lower().split())
    norm_loc = _LOC_ALIASES.get(norm_loc, norm_loc)
    if norm_loc not in AnchoredOffsetbox.codes:
        valid = ", ".join(sorted(AnchoredOffsetbox.codes))
        raise rsgispy.RSGISPyException(
            f"Invalid placement location '{loc}'. Valid options are: {valid}."
        )
    return norm_loc


def _resolve_scalebar_units(units: Optional[str]) -> Optional[str]:
    """
    Validate scale bar units, allowing only metres (``'m'``), kilometres
    (``'km'``), or ``None``.
    """
    if units is None:
        return None
    if not isinstance(units, str):
        raise rsgispy.RSGISPyException(
            "Scale bar units must be 'm', 'km', or None (auto)."
        )
    norm = units.strip().lower()
    if norm in ("", "auto"):
        return None
    if norm in (SCALEBAR_UNIT_METRES, "metre", "metres", "meter", "meters"):
        return SCALEBAR_UNIT_METRES
    if norm in (
        SCALEBAR_UNIT_KILOMETRES,
        "kilometre",
        "kilometres",
        "kilometer",
        "kilometers",
    ):
        return SCALEBAR_UNIT_KILOMETRES
    raise rsgispy.RSGISPyException(
        f"Unsupported scale bar unit '{units}'. Only 'm' and 'km' are allowed."
    )


def _format_scale_val(val: float) -> str:
    """Format numeric scale bar values cleanly without trailing zeros."""
    if abs(val) < 1e-12:
        return "0"
    rounded = round(float(val), 6)
    if 1e-4 <= rounded < 1e9:
        text = f"{rounded:.6f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text
    return f"{rounded:g}"


def _compute_metres_per_map_unit(
    epsg_code: Optional[int],
    bbox: Tuple[float, float, float, float],
) -> float:
    """Compute metres per x-axis map unit from EPSG code and bounding box."""
    if epsg_code is None:
        return 1.0
    if int(epsg_code) == 4326:
        lat_mid = max(-89.9, min(89.9, 0.5 * (bbox[2] + bbox[3])))
        return rsgispy.tools.projection.great_circle_distance(
            wgs84_p1=(0.0, lat_mid),
            wgs84_p2=(1.0, lat_mid),
        )
    try:
        crs = pyproj.CRS.from_epsg(int(epsg_code))
        if crs.is_geographic:
            lat_mid = max(-89.9, min(89.9, 0.5 * (bbox[2] + bbox[3])))
            return rsgispy.tools.projection.great_circle_distance(
                wgs84_p1=(0.0, lat_mid),
                wgs84_p2=(1.0, lat_mid),
            )
        if crs.axis_info:
            factor = crs.axis_info[0].unit_conversion_factor
            if factor is not None and factor > 0:
                return float(factor)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return 1.0


def _pick_nice_value(target: float, max_val: float) -> float:
    """Pick a cartographically clean 1-2-5 multiple near ``target`` and <= ``max_val``."""
    target = max(target, 1e-9)
    max_val = max(max_val, target)
    exp = math.floor(math.log10(target))
    candidates = [
        mult * (10.0**pwr)
        for pwr in range(exp - 1, exp + 3)
        for mult in (1.0, 2.0, 5.0)
    ]
    bounded = [c for c in candidates if c <= max_val * (1.0 + 1e-9)]
    pool = bounded if bounded else candidates
    return min(pool, key=lambda c: abs(math.log10(c) - math.log10(target)))


def _choose_scale_length(
    target_metres: float,
    max_metres: float,
    units: Optional[str],
    fixed_value: Optional[float],
) -> Tuple[float, float, str]:
    """
    Determine ``(bar_value, bar_metres, unit)`` where ``unit`` is strictly
    ``'m'`` or ``'km'``.
    """
    if fixed_value is not None:
        val = float(fixed_value)
        if val <= 0.0:
            raise rsgispy.RSGISPyException("fixed_value must be greater than 0.")
        resolved_unit = (
            SCALEBAR_UNIT_KILOMETRES
            if units == SCALEBAR_UNIT_KILOMETRES
            else SCALEBAR_UNIT_METRES
        )
        bar_m = val * 1000.0 if resolved_unit == SCALEBAR_UNIT_KILOMETRES else val
        return val, bar_m, resolved_unit

    if units == SCALEBAR_UNIT_KILOMETRES:
        nice_km = _pick_nice_value(target_metres / 1000.0, max_metres / 1000.0)
        return nice_km, nice_km * 1000.0, SCALEBAR_UNIT_KILOMETRES
    if units == SCALEBAR_UNIT_METRES:
        nice_m = _pick_nice_value(target_metres, max_metres)
        return nice_m, nice_m, SCALEBAR_UNIT_METRES

    nice_m = _pick_nice_value(target_metres, max_metres)
    if nice_m >= 1000.0:
        return nice_m / 1000.0, nice_m, SCALEBAR_UNIT_KILOMETRES
    return nice_m, nice_m, SCALEBAR_UNIT_METRES


def _choose_n_segments(bar_value: float, n_segments: Optional[int]) -> int:
    """Choose clean segment subdivision count for alternating/ticks scale bars."""
    if n_segments is not None:
        seg_count = int(n_segments)
        if seg_count < 1:
            raise rsgispy.RSGISPyException("n_segments must be at least 1.")
        return seg_count
    exp = math.floor(math.log10(max(bar_value, 1e-12)))
    mant = round(bar_value / (10.0**exp), 6)
    if abs(mant - 5.0) < 1e-6:
        return 5
    if abs(mant - 2.0) < 1e-6:
        return 4
    if abs(mant - 1.0) < 1e-6:
        return 4 if bar_value >= 100.0 else 2
    return 4


class MapScaleBar(AnchoredOffsetbox):
    """
    Pure-Matplotlib cartographic scale bar artist supporting solid, alternating,
    and ticked line styles in metres (``'m'``) or kilometres (``'km'``).
    """

    # pylint: disable=too-many-ancestors,too-many-instance-attributes,abstract-method
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        bar_map_units: float,
        bar_value: float,
        bar_metres: float,
        units: str,
        style: str = SCALEBAR_STYLE_SOLID,
        loc: str = "upper right",
        n_segments: int = 4,
        color: str = "black",
        alt_color: str = "white",
        box_color: str = "white",
        box_alpha: float = 0.8,
        frameon: bool = True,
        font_size: float = 9.0,
        bar_height: float = 5.0,
        pad: float = 0.4,
        borderpad: float = 0.5,
        zorder: float = 10.0,
    ) -> None:
        self._ax = ax
        self._bar_map_units = float(bar_map_units)
        self.bar_length_value = float(bar_value)
        self.bar_length_metres = float(bar_metres)
        self.units = units
        self.style = style
        self.loc_str = loc
        self.n_segments = int(n_segments)
        self.color = color
        self.alt_color = alt_color
        self.font_size = float(font_size)
        self.bar_height = float(bar_height)
        self.label_text = f"{_format_scale_val(bar_value)} {units}"

        est_pts = self._estimate_initial_bar_pts()
        self._current_bar_pts = -1.0
        self._drawing_area = DrawingArea(est_pts, self.bar_height, 0.0, 0.0)
        child = self._init_child_box()

        super().__init__(
            loc=loc,
            child=child,
            pad=pad,
            borderpad=borderpad,
            frameon=frameon,
        )
        self.set_zorder(zorder)
        if frameon:
            self.patch.set_facecolor(box_color)
            self.patch.set_alpha(box_alpha)
            self.patch.set_edgecolor("none")
        self._populate_drawing(est_pts)

    def _estimate_initial_bar_pts(self) -> float:
        """Estimate bar width in points before the canvas is drawn."""
        xlim = self._ax.get_xlim()
        span = max(abs(xlim[1] - xlim[0]), 1e-12)
        frac = max(0.02, min(0.95, self._bar_map_units / span))
        fig = self._ax.figure
        fig_w_in = float(fig.get_size_inches()[0]) if fig is not None else 6.4
        ax_w_frac = float(self._ax.get_position().width)
        return max(20.0, ax_w_frac * fig_w_in * 72.0 * frac)

    def _init_child_box(self) -> Any:
        """Construct the OffsetBox container matching ``self.style``."""
        if self.style == SCALEBAR_STYLE_SOLID:
            lbl_area = TextArea(
                self.label_text,
                textprops={
                    "fontsize": self.font_size,
                    "color": self.color,
                    "fontweight": "normal",
                },
            )
            return VPacker(
                children=[lbl_area, self._drawing_area],
                align="center",
                pad=0,
                sep=2.5,
            )
        return self._drawing_area

    def _measure_bar_pts(self, renderer: Any) -> float:
        """Compute exact point width of ``self._bar_map_units`` on ``self._ax``."""
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        x_mid = 0.5 * (xlim[0] + xlim[1])
        y_mid = 0.5 * (ylim[0] + ylim[1])
        p0 = self._ax.transData.transform((x_mid, y_mid))
        p1 = self._ax.transData.transform((x_mid + self._bar_map_units, y_mid))
        bar_px = abs(float(p1[0] - p0[0]))
        px_per_pt = float(renderer.points_to_pixels(1.0))
        if bar_px <= 0.0 or px_per_pt <= 0.0:
            return self._estimate_initial_bar_pts()
        return max(6.0, bar_px / px_per_pt)

    def _populate_drawing(self, bar_pts: float) -> None:
        """Populate/rebuild ``self._drawing_area`` children for ``bar_pts``."""
        # pylint: disable=too-many-locals
        if abs(bar_pts - self._current_bar_pts) <= 1e-4:
            return
        self._current_bar_pts = bar_pts
        self._drawing_area.get_children().clear()

        if self.style == SCALEBAR_STYLE_SOLID:
            self._drawing_area.width = bar_pts
            self._drawing_area.height = self.bar_height
            self._drawing_area.add_artist(
                Rectangle(
                    (0.0, 0.0),
                    bar_pts,
                    self.bar_height,
                    facecolor=self.color,
                    edgecolor="none",
                )
            )
            return

        left_pad = max(4.5, self.font_size * 0.55)
        end_num_str = _format_scale_val(self.bar_length_value)
        unit_gap = max(4.0, self.font_size * 0.45)
        unit_w_est = len(self.units) * self.font_size * 0.78
        end_half_w = max(4.5, len(end_num_str) * self.font_size * 0.35)
        right_pad = max(end_half_w, unit_gap + unit_w_est) + 3.0
        total_w = left_pad + bar_pts + right_pad
        lbl_gap = 2.5
        total_h = self.bar_height + lbl_gap + self.font_size * 1.15
        self._drawing_area.width = total_w
        self._drawing_area.height = total_h

        n_seg = self.n_segments
        seg_w = bar_pts / float(n_seg)

        if self.style == SCALEBAR_STYLE_ALTERNATING:
            for idx in range(n_seg):
                fill_c = self.color if idx % 2 == 0 else self.alt_color
                self._drawing_area.add_artist(
                    Rectangle(
                        (left_pad + idx * seg_w, 0.0),
                        seg_w,
                        self.bar_height,
                        facecolor=fill_c,
                        edgecolor=self.color,
                        linewidth=0.8,
                    )
                )
        else:
            y_base = 1.0
            self._drawing_area.add_artist(
                Line2D(
                    [left_pad, left_pad + bar_pts],
                    [y_base, y_base],
                    color=self.color,
                    linewidth=1.5,
                )
            )
            for idx in range(n_seg + 1):
                x_tick = left_pad + idx * seg_w
                tick_h = (
                    self.bar_height
                    if idx in (0, n_seg)
                    else max(2.5, self.bar_height * 0.65)
                )
                self._drawing_area.add_artist(
                    Line2D(
                        [x_tick, x_tick],
                        [y_base, y_base + tick_h],
                        color=self.color,
                        linewidth=1.2 if idx in (0, n_seg) else 0.9,
                    )
                )

        min_spacing = self.font_size * 2.3
        if n_seg > 1 and (bar_pts / float(n_seg)) >= min_spacing:
            tick_indices = list(range(n_seg + 1))
        elif n_seg >= 2 and n_seg % 2 == 0 and (bar_pts / 2.0) >= min_spacing:
            tick_indices = [0, n_seg // 2, n_seg]
        else:
            tick_indices = [0, n_seg]

        y_lbl = self.bar_height + lbl_gap
        for idx in tick_indices:
            x_lbl = left_pad + idx * seg_w
            val = self.bar_length_value * (float(idx) / float(n_seg))
            self._drawing_area.add_artist(
                Text(
                    x_lbl,
                    y_lbl,
                    _format_scale_val(val),
                    fontsize=self.font_size,
                    color=self.color,
                    ha="center",
                    va="bottom",
                )
            )
        self._drawing_area.add_artist(
            Text(
                left_pad + bar_pts + unit_gap,
                0.0,
                self.units,
                fontsize=self.font_size,
                color=self.color,
                ha="left",
                va="bottom",
            )
        )

    def get_bbox(self, renderer: Any) -> Any:
        """Update scale bar width for current data transform and return bbox."""
        bar_pts = self._measure_bar_pts(renderer)
        self._populate_drawing(bar_pts)
        return super().get_bbox(renderer)

    def draw(self, renderer: Any) -> None:
        """Render the scale bar at exact current data scale."""
        bar_pts = self._measure_bar_pts(renderer)
        self._populate_drawing(bar_pts)
        super().draw(renderer)


class MapNorthArrow(AnchoredOffsetbox):
    """
    Pure-Matplotlib cartographic north arrow artist supporting multiple styles.
    """

    # pylint: disable=too-many-ancestors,abstract-method
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        style: str = NORTH_ARROW_STYLE_TRIANGLE,
        loc: str = "upper left",
        size: float = 24.0,
        color: str = "black",
        alt_color: str = "white",
        label: str = "N",
        font_size: float = 10.0,
        box_color: str = "white",
        box_alpha: float = 0.8,
        frameon: bool = True,
        pad: float = 0.4,
        borderpad: float = 0.5,
        rotation: float = 0.0,
        zorder: float = 10.0,
    ) -> None:
        self.style = style
        self.loc_str = loc
        self.size = float(size)
        self.color = color
        self.alt_color = alt_color
        self.label = label
        self.rotation = float(rotation)

        arrow_da = _build_north_arrow_drawing(
            style=style,
            size=self.size,
            color=color,
            alt_color=alt_color,
            rotation=self.rotation,
        )
        if label:
            lbl_area = TextArea(
                label,
                textprops={
                    "fontsize": font_size,
                    "fontweight": "bold",
                    "color": color,
                },
            )
            child: Any = VPacker(
                children=[lbl_area, arrow_da],
                align="center",
                pad=0,
                sep=1.5,
            )
        else:
            child = arrow_da

        super().__init__(
            loc=loc,
            child=child,
            pad=pad,
            borderpad=borderpad,
            frameon=frameon,
        )
        self.set_zorder(zorder)
        if frameon:
            self.patch.set_facecolor(box_color)
            self.patch.set_alpha(box_alpha)
            self.patch.set_edgecolor("none")


def _rotate_points(
    pts: Sequence[Tuple[float, float]],
    cx: float,
    cy: float,
    rotation: float,
) -> np.ndarray:
    """Rotate 2D polygon vertices around ``(cx, cy)`` by ``rotation`` degrees."""
    arr = np.asarray(pts, dtype=float)
    if abs(rotation) < 1e-9:
        return arr
    tf = Affine2D().rotate_deg_around(cx, cy, rotation)
    return tf.transform(arr)


def _build_north_arrow_drawing(
    style: str,
    size: float,
    color: str,
    alt_color: str,
    rotation: float,
) -> DrawingArea:
    """Create a ``DrawingArea`` containing the requested north arrow symbol."""
    # pylint: disable=too-many-locals
    w = size
    h = size
    cx = 0.5 * w
    cy = 0.5 * h
    da = DrawingArea(w, h, 0.0, 0.0)

    def _add_poly(
        coords: Sequence[Tuple[float, float]],
        facecolor: str,
        edgecolor: str,
        linewidth: float = 0.9,
    ) -> None:
        da.add_artist(
            Polygon(
                _rotate_points(coords, cx, cy, rotation),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                joinstyle="miter",
            )
        )

    if style == NORTH_ARROW_STYLE_TRIANGLE:
        apex = (cx, h * 0.96)
        notch = (cx, h * 0.24)
        left_pt = (w * 0.16, h * 0.04)
        right_pt = (w * 0.84, h * 0.04)
        _add_poly([apex, left_pt, notch], facecolor=color, edgecolor=color)
        _add_poly([apex, notch, right_pt], facecolor=alt_color, edgecolor=color)
    elif style == NORTH_ARROW_STYLE_ARROW:
        sw = w * 0.09
        hw = w * 0.28
        shaft_top = h * 0.55
        _add_poly(
            [
                (cx - sw, h * 0.04),
                (cx + sw, h * 0.04),
                (cx + sw, shaft_top),
                (cx + hw, shaft_top),
                (cx, h * 0.96),
                (cx - hw, shaft_top),
                (cx - sw, shaft_top),
            ],
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
        )
    elif style == NORTH_ARROW_STYLE_COMPASS:
        long_r = 0.46 * min(w, h)
        short_r = 0.32 * min(w, h)
        inner_r = 0.11 * min(w, h)
        n_pt = (cx, cy + long_r)
        s_pt = (cx, cy - long_r)
        e_pt = (cx + short_r, cy)
        w_pt = (cx - short_r, cy)
        ne_pt = (cx + inner_r, cy + inner_r)
        nw_pt = (cx - inner_r, cy + inner_r)
        se_pt = (cx + inner_r, cy - inner_r)
        sw_pt = (cx - inner_r, cy - inner_r)
        facets = [
            ([(cx, cy), n_pt, nw_pt], color),
            ([(cx, cy), ne_pt, n_pt], alt_color),
            ([(cx, cy), e_pt, ne_pt], color),
            ([(cx, cy), se_pt, e_pt], alt_color),
            ([(cx, cy), s_pt, se_pt], color),
            ([(cx, cy), sw_pt, s_pt], alt_color),
            ([(cx, cy), w_pt, sw_pt], color),
            ([(cx, cy), nw_pt, w_pt], alt_color),
        ]
        for coords, fc in facets:
            _add_poly(coords, facecolor=fc, edgecolor=color, linewidth=0.7)
    else:
        apex = (cx, h * 0.94)
        notch = (cx, h * 0.26)
        left_pt = (w * 0.20, h * 0.06)
        right_pt = (w * 0.80, h * 0.06)
        _add_poly(
            [apex, right_pt, notch, left_pt],
            facecolor=alt_color,
            edgecolor=color,
            linewidth=1.5,
        )
        stem = _rotate_points([notch, apex], cx, cy, rotation)
        da.add_artist(
            Line2D(stem[:, 0], stem[:, 1], color=color, linewidth=1.2)
        )

    return da


def add_scale_bar(
    ax: matplotlib.axes.Axes,
    epsg_code: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    loc: str = "upper right",
    units: Optional[str] = None,
    style: str = SCALEBAR_STYLE_SOLID,
    length_fraction: float = 0.25,
    fixed_value: Optional[float] = None,
    n_segments: Optional[int] = None,
    dx: Optional[float] = None,
    color: str = "black",
    alt_color: str = "white",
    box_color: str = "white",
    box_alpha: float = 0.8,
    frameon: bool = True,
    font_size: float = 9.0,
    bar_height: float = 5.0,
    pad: float = 0.4,
    borderpad: float = 0.5,
    zorder: float = 10.0,
) -> MapScaleBar:
    """
    Add a cartographic scale bar to ``ax``.

    Only metres (``'m'``) and kilometres (``'km'``) are supported as display units.

    :param ax: Target Matplotlib axis.
    :param epsg_code: Optional EPSG code for ``ax`` coordinates (converts degrees
                      to metres for geographic CRSs such as EPSG:4326).
    :param bbox: Optional ``(MinX, MaxX, MinY, MaxY)`` extent. Inferred from
                 ``ax`` limits when ``None``.
    :param loc: Placement string (e.g. ``'upper right'``, ``'lower left'``).
    :param units: Scale bar units: ``'m'``, ``'km'``, or ``None`` to auto-select
                  between ``'m'`` and ``'km'`` based on map width.
    :param style: Visual style constant: :data:`SCALEBAR_STYLE_SOLID` (default),
                  :data:`SCALEBAR_STYLE_ALTERNATING`, or :data:`SCALEBAR_STYLE_TICKS`.
    :param length_fraction: Target bar width as a fraction of axis width
                            (Default = 0.25).
    :param fixed_value: Optional fixed bar length in ``units`` (defaults to ``'m'``
                        when ``units`` is ``None``).
    :param n_segments: Optional number of subdivisions for ``'alternating'`` and
                       ``'ticks'`` styles.
    :param dx: Optional explicit metres-per-map-unit scale factor override.
    :param color: Primary bar/text colour (Default = ``'black'``).
    :param alt_color: Secondary segment colour for alternating style (Default = ``'white'``).
    :param box_color: Background box fill colour (Default = ``'white'``).
    :param box_alpha: Background box opacity (Default = 0.8).
    :param frameon: Whether to draw a background box behind the scale bar (Default = True).
    :param font_size: Label font size in points (Default = 9.0).
    :param bar_height: Bar height in points (Default = 5.0).
    :param pad: Internal box padding (Default = 0.4).
    :param borderpad: Margin between box and axis edges (Default = 0.5).
    :param zorder: Artist z-order (Default = 10.0).
    :return: Created :class:`MapScaleBar` artist.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.define_axis_extent(
            ax, bbox=(500000.0, 510000.0, 5200000.0, 5210000.0)
        )
        rsgispy.tools.mapping.add_scale_bar(
            ax,
            epsg_code=32630,
            style=rsgispy.tools.mapping.SCALEBAR_STYLE_ALTERNATING,
            units=rsgispy.tools.mapping.SCALEBAR_UNIT_KILOMETRES,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if style not in _SCALEBAR_STYLES:
        valid_styles = ", ".join(sorted(_SCALEBAR_STYLES))
        raise rsgispy.RSGISPyException(
            f"Invalid scale bar style '{style}'. Valid options: {valid_styles}."
        )
    norm_loc = _resolve_mpl_location(loc)
    norm_units = _resolve_scalebar_units(units)
    if length_fraction <= 0.0 or length_fraction > 1.0:
        raise rsgispy.RSGISPyException(
            "length_fraction must be in the interval (0, 1]."
        )

    if bbox is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        eff_bbox = (
            float(min(xlim)),
            float(max(xlim)),
            float(min(ylim)),
            float(max(ylim)),
        )
    else:
        eff_bbox = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )

    map_width_units = abs(eff_bbox[1] - eff_bbox[0])
    if map_width_units <= 0.0:
        raise rsgispy.RSGISPyException(
            "Map axis width must be positive to compute a scale bar."
        )

    if dx is not None:
        metres_per_unit = float(dx)
        if metres_per_unit <= 0.0:
            raise rsgispy.RSGISPyException("dx must be positive.")
    else:
        metres_per_unit = _compute_metres_per_map_unit(epsg_code, eff_bbox)

    map_width_metres = map_width_units * metres_per_unit
    target_metres = map_width_metres * float(length_fraction)
    max_metres = map_width_metres * 0.55

    bar_value, bar_metres, resolved_unit = _choose_scale_length(
        target_metres=target_metres,
        max_metres=max_metres,
        units=norm_units,
        fixed_value=fixed_value,
    )
    bar_map_units = bar_metres / metres_per_unit
    seg_count = _choose_n_segments(bar_value, n_segments)

    artist = MapScaleBar(
        ax=ax,
        bar_map_units=bar_map_units,
        bar_value=bar_value,
        bar_metres=bar_metres,
        units=resolved_unit,
        style=style,
        loc=norm_loc,
        n_segments=seg_count,
        color=color,
        alt_color=alt_color,
        box_color=box_color,
        box_alpha=box_alpha,
        frameon=frameon,
        font_size=font_size,
        bar_height=bar_height,
        pad=pad,
        borderpad=borderpad,
        zorder=zorder,
    )
    ax.add_artist(artist)
    return artist


def add_north_arrow(
    ax: matplotlib.axes.Axes,
    loc: str = "upper left",
    style: str = NORTH_ARROW_STYLE_TRIANGLE,
    size: float = 24.0,
    color: str = "black",
    alt_color: str = "white",
    label: str = "N",
    font_size: float = 10.0,
    box_color: str = "white",
    box_alpha: float = 0.8,
    frameon: bool = True,
    pad: float = 0.4,
    borderpad: float = 0.5,
    rotation: float = 0.0,
    zorder: float = 10.0,
) -> MapNorthArrow:
    """
    Add a north arrow artist to ``ax`` with selectable cartographic styles.

    :param ax: Target Matplotlib axis.
    :param loc: Placement string (Default = ``'upper left'``).
    :param style: Visual style constant: :data:`NORTH_ARROW_STYLE_TRIANGLE` (default),
                  :data:`NORTH_ARROW_STYLE_ARROW`, :data:`NORTH_ARROW_STYLE_COMPASS`,
                  or :data:`NORTH_ARROW_STYLE_SIMPLE`.
    :param size: Symbol size in points (Default = 24.0).
    :param color: Primary stroke and fill colour (Default = ``'black'``).
    :param alt_color: Secondary fill colour for split/outlined styles (Default = ``'white'``).
    :param label: Label drawn above the arrow symbol (Default = ``'N'``).
    :param font_size: Label font size in points (Default = 10.0).
    :param box_color: Background box fill colour (Default = ``'white'``).
    :param box_alpha: Background box opacity (Default = 0.8).
    :param frameon: Whether to draw a background box (Default = True).
    :param pad: Internal box padding (Default = 0.4).
    :param borderpad: Margin between box and axis edges (Default = 0.5).
    :param rotation: Counterclockwise arrow rotation in degrees (Default = 0.0).
    :param zorder: Artist z-order (Default = 10.0).
    :return: Created :class:`MapNorthArrow` artist.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.define_axis_extent(
            ax, bbox=(500000.0, 510000.0, 5200000.0, 5210000.0)
        )
        rsgispy.tools.mapping.add_north_arrow(
            ax,
            loc="upper left",
            style=rsgispy.tools.mapping.NORTH_ARROW_STYLE_COMPASS,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if style not in _NORTH_ARROW_STYLES:
        valid_styles = ", ".join(sorted(_NORTH_ARROW_STYLES))
        raise rsgispy.RSGISPyException(
            f"Invalid north arrow style '{style}'. Valid options: {valid_styles}."
        )
    if size <= 0.0:
        raise rsgispy.RSGISPyException("size must be positive.")
    norm_loc = _resolve_mpl_location(loc)

    artist = MapNorthArrow(
        style=style,
        loc=norm_loc,
        size=size,
        color=color,
        alt_color=alt_color,
        label=label,
        font_size=font_size,
        box_color=box_color,
        box_alpha=box_alpha,
        frameon=frameon,
        pad=pad,
        borderpad=borderpad,
        rotation=rotation,
        zorder=zorder,
    )
    ax.add_artist(artist)
    return artist


def _finalize_map_axis(
    ax: matplotlib.axes.Axes,
    epsg_code: Optional[int],
    bbox: Tuple[float, float, float, float],
    show_scale_bar: bool,
    use_grid: bool,
    show_map_axis: bool,
    scale_bar_loc: str,
    title_str: Optional[str],
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """Apply common map formatting options (grid, axis, scale bar, north arrow, title)."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if use_grid:
        ax.grid()
    if not show_map_axis:
        ax.set_axis_off()
    if show_scale_bar:
        add_scale_bar(
            ax=ax,
            epsg_code=epsg_code,
            bbox=bbox,
            loc=scale_bar_loc,
            units=scale_bar_units,
            style=scale_bar_style,
        )
    if show_north_arrow:
        add_north_arrow(
            ax=ax,
            loc=north_arrow_loc,
            style=north_arrow_style,
        )
    if title_str is not None:
        ax.title.set_text(title_str)


def calc_y_fig_size(
    bbox: Tuple[float, float, float, float],
    fig_x_size: Union[int, float],
) -> float:
    """
    Calculate the figure y-axis size given a bounding box and figure x-axis size
    so the mapped area fills the figure cleanly.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the region of interest.
    :param fig_x_size: Size for the figure in the x-axis.
    :return: Figure size in the y-axis.

    .. code:: python

        import rsgispy.tools.mapping

        fig_y_size = rsgispy.tools.mapping.calc_y_fig_size(
            bbox=(250000.0, 270000.0, 280000.0, 290000.0), fig_x_size=10.0
        )
    """
    x_bbox_size = bbox[1] - bbox[0]
    if x_bbox_size == 0:
        raise rsgispy.RSGISPyException(
            "Bounding box width (MaxX - MinX) must be non-zero."
        )
    y_bbox_size = bbox[3] - bbox[2]
    bbox_xy_ratio = y_bbox_size / x_bbox_size
    return float(fig_x_size * bbox_xy_ratio)


def define_axis_extent(
    ax: matplotlib.axes.Axes,
    bbox: Tuple[float, float, float, float],
) -> None:
    """
    Set equal aspect ratio, x/y limits, and plain tick labels on ``ax`` using
    ``bbox``.

    :param ax: Matplotlib axis for which limits are set.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.define_axis_extent(
            ax, bbox=(250000.0, 270000.0, 280000.0, 290000.0)
        )
    """
    ax.set_aspect("equal", "box")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")


def define_map_tick_spacing(
    ax: matplotlib.axes.Axes,
    tick_spacing: float,
) -> None:
    """
    Define the major tick spacing on both axes of the map.

    :param ax: Matplotlib axis for which tick spacing is defined.
    :param tick_spacing: Spacing between major ticks in map units.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.define_map_tick_spacing(ax, tick_spacing=5000.0)
    """
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.ticklabel_format(useOffset=False, style="plain")


def get_overview_info(
    roi_bbox: Tuple[float, float, float, float],
    roi_epsg: int = 4326,
    overview_buf: float = 30.0,
    out_epsg: int = 4326,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
    """
    Compute an overview map bounding box and centre point from a region of
    interest bounding box, optionally reprojecting between EPSG codes.

    :param roi_bbox: Region of interest bbox ``(MinX, MaxX, MinY, MaxY)``.
    :param roi_epsg: EPSG code of ``roi_bbox`` (Default = 4326).
    :param overview_buf: Buffer added around the ROI centre point in ``out_epsg``
                         units (Default = 30.0).
    :param out_epsg: Output EPSG code for the overview bbox and point (Default = 4326).
    :return: Tuple ``(overview_bbox, centre_point)`` where ``overview_bbox`` is
             ``(MinX, MaxX, MinY, MaxY)`` and ``centre_point`` is ``(X, Y)``.

    .. code:: python

        import rsgispy.tools.mapping

        over_bbox, centre_pt = rsgispy.tools.mapping.get_overview_info(
            roi_bbox=(-4.2, -4.0, 52.3, 52.5),
            roi_epsg=4326,
            overview_buf=5.0,
        )
    """
    if roi_epsg != out_epsg:
        min_x, min_y, max_x, max_y = transform_bounds(
            f"EPSG:{roi_epsg}",
            f"EPSG:{out_epsg}",
            roi_bbox[0],
            roi_bbox[2],
            roi_bbox[1],
            roi_bbox[3],
        )
        roi_proj_bbox = (min_x, max_x, min_y, max_y)
    else:
        roi_proj_bbox = (
            float(roi_bbox[0]),
            float(roi_bbox[1]),
            float(roi_bbox[2]),
            float(roi_bbox[3]),
        )

    c_pt_x = roi_proj_bbox[0] + ((roi_proj_bbox[1] - roi_proj_bbox[0]) / 2.0)
    c_pt_y = roi_proj_bbox[2] + ((roi_proj_bbox[3] - roi_proj_bbox[2]) / 2.0)
    over_bbox: Tuple[float, float, float, float] = (
        c_pt_x - overview_buf,
        c_pt_x + overview_buf,
        c_pt_y - overview_buf,
        c_pt_y + overview_buf,
    )
    return over_bbox, (c_pt_x, c_pt_y)


def add_overview_maps(
    ax: matplotlib.axes.Axes,
    overview_lyr: gpd.GeoDataFrame,
    overview_bbox: Tuple[float, float, float, float],
    overview_pt: Optional[Tuple[float, float]] = None,
    over_size_x: float = 0.1,
    over_size_y: float = 0.1,
    over_x_off: float = 0.0,
    over_y_off: float = 0.0,
    pt_clr: str = "red",
    pt_size: float = 10.0,
    fill_clr: str = "white",
    line_clr: str = "black",
    line_width: float = 0.25,
) -> matplotlib.axes.Axes:
    """
    Add an inset overview map to ``ax`` (defaulting to the top-left corner).

    :param ax: Parent matplotlib axis.
    :param overview_lyr: GeoDataFrame used as the overview map backdrop.
    :param overview_bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` shown in
                          the inset axis.
    :param overview_pt: Optional ``(X, Y)`` marker coordinate on the overview map.
    :param over_size_x: Width of the inset axis in axes fraction coordinates.
    :param over_size_y: Height of the inset axis in axes fraction coordinates.
    :param over_x_off: X-axis offset of the inset axis.
    :param over_y_off: Y-axis offset of the inset axis.
    :param pt_clr: Colour of ``overview_pt`` marker (Default = ``'red'``).
    :param pt_size: Marker size for ``overview_pt`` (Default = 10.0).
    :param fill_clr: Polygon fill colour for ``overview_lyr`` (Default = ``'white'``).
    :param line_clr: Edge line colour for ``overview_lyr`` (Default = ``'black'``).
    :param line_width: Line width for ``overview_lyr`` (Default = 0.25).
    :return: Created inset matplotlib axis.

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        world_gdf = gpd.read_file("world.gpkg")
        ax_inset = rsgispy.tools.mapping.add_overview_maps(
            ax=ax,
            overview_lyr=world_gdf,
            overview_bbox=(-10.0, 5.0, 48.0, 60.0),
            overview_pt=(-4.08, 52.41),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ax_over = ax.inset_axes(
        [over_x_off, (1.0 - over_size_y) + over_y_off, over_size_x, over_size_y]
    )
    ax_over.set_xticks([])
    ax_over.set_yticks([])
    overview_lyr.plot(
        ax=ax_over,
        color=fill_clr,
        edgecolor=line_clr,
        linewidth=line_width,
        alpha=1.0,
    )
    if overview_pt is not None:
        ax_over.scatter(overview_pt[0], overview_pt[1], color=pt_clr, s=pt_size)
    ax_over.set_xlim([overview_bbox[0], overview_bbox[1]])
    ax_over.set_ylim([overview_bbox[2], overview_bbox[3]])
    return ax_over


def add_contextily_basemap(
    ax: matplotlib.axes.Axes,
    epsg_crs: int,
    cx_src: Any,
    cx_zoom_lvl: Union[int, str] = "auto",
    cx_attribution: Optional[Union[str, bool]] = None,
    cx_att_size: int = 8,
) -> None:
    """
    Add a contextily tile basemap beneath existing layers on ``ax``.

    :param ax: Matplotlib axis to which the basemap is added.
    :param epsg_crs: EPSG code of the coordinate reference system used on ``ax``.
    :param cx_src: Contextily tile provider (e.g., ``contextily.providers.OpenTopoMap``).
    :param cx_zoom_lvl: Tile zoom level or ``'auto'`` (Default = ``'auto'``).
    :param cx_attribution: Attribution string, ``False`` to suppress, or ``None``
                           for default attribution.
    :param cx_att_size: Attribution font size (Default = 8).

    .. code:: python

        import contextily
        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.define_axis_extent(
            ax, bbox=(250000.0, 270000.0, 280000.0, 290000.0)
        )
        rsgispy.tools.mapping.add_contextily_basemap(
            ax=ax, epsg_crs=27700, cx_src=contextily.providers.OpenStreetMap.Mapnik
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,import-outside-toplevel
    import contextily

    if cx_zoom_lvl is None:
        cx_zoom_lvl = "auto"

    contextily.add_basemap(
        ax,
        zoom=cx_zoom_lvl,
        crs=f"epsg:{epsg_crs}",
        source=cx_src,
        attribution=cx_attribution,
        attribution_size=cx_att_size,
    )


def _update_wmts_xml(
    in_xml_path: str,
    out_xml_path: str,
    wmts_bbox: Tuple[float, float, float, float],
    wmts_zoom_level: Optional[int],
) -> None:
    """Update DataWindow bounds and optional ZoomLevel in a GDAL WMTS XML file."""
    tree = ET.parse(in_xml_path)
    gdal_wmts_tag = tree.getroot()
    if gdal_wmts_tag.tag != "GDAL_WMTS":
        raise rsgispy.RSGISPyException(
            "Something has gone wrong: Root not GDAL_WMTS."
        )
    data_win_tag = gdal_wmts_tag.find("DataWindow")
    if data_win_tag is None:
        raise rsgispy.RSGISPyException(
            "Something has gone wrong: No DataWindow tag."
        )
    for tag_name, val in (
        ("UpperLeftX", wmts_bbox[0]),
        ("UpperLeftY", wmts_bbox[3]),
        ("LowerRightX", wmts_bbox[1]),
        ("LowerRightY", wmts_bbox[2]),
    ):
        tag = data_win_tag.find(tag_name)
        if tag is None:
            raise rsgispy.RSGISPyException(
                f"Something has gone wrong: No {tag_name} tag."
            )
        tag.text = f"{val}"

    if wmts_zoom_level is not None:
        zoom_lvl_tag = ET.Element("ZoomLevel")
        zoom_lvl_tag.text = f"{wmts_zoom_level}"
        gdal_wmts_tag.append(zoom_lvl_tag)

    ET.indent(tree, space="    ", level=0)
    tree.write(out_xml_path)


def create_wmts_img_map(
    ax: matplotlib.axes.Axes,
    wmts_url: str,
    wmts_lyr: str,
    bbox: Tuple[float, float, float, float],
    bbox_epsg: int,
    wmts_zoom_level: Optional[int] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    tmp_dir: Optional[str] = None,
    wmts_epsg: Optional[int] = None,
    scale_bar_loc: str = "upper right",
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Download an image subset from a WMTS service via ``rasterio`` and display
    it on ``ax``.

    :param ax: Matplotlib axis to which the WMTS image is added.
    :param wmts_url: URL of the WMTS capabilities/service endpoint.
    :param wmts_lyr: Layer name within the WMTS service.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` specifying the
                 spatial region to display.
    :param bbox_epsg: EPSG code of ``bbox`` and the output map.
    :param wmts_zoom_level: Optional WMTS zoom level.
    :param title_str: Optional title for the map (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param tmp_dir: Optional directory for intermediate WMTS files.
    :param wmts_epsg: Optional EPSG code for the WMTS layer if auto-detection fails.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.create_wmts_img_map(
            ax=ax,
            wmts_url="https://example.com/wmts/1.0.0/WMTSCapabilities.xml",
            wmts_lyr="ortho_layer",
            bbox=(250000.0, 260000.0, 280000.0, 290000.0),
            bbox_epsg=27700,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    uid_str = uuid.uuid4().hex[:8]
    with rsgispy.tools.files.work_dir_context(tmp_dir) as work_dir:

        gdal_lyr_url = f"WMTS:{wmts_url},layer={wmts_lyr}"
        out_tmp_xml = os.path.join(work_dir, f"wmts_tmp_xml_{uid_str}.xml")
        rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
            gdal_lyr_url, out_tmp_xml, driver="WMTS"
        )

        resolved_wmts_epsg = (
            wmts_epsg
            if wmts_epsg is not None
            else rsgispy.imageutils.metadata.get_epsg_proj_from_img(out_tmp_xml)
        )
        if resolved_wmts_epsg is not None and resolved_wmts_epsg != bbox_epsg:
            w_min_x, w_min_y, w_max_x, w_max_y = transform_bounds(
                f"EPSG:{bbox_epsg}",
                f"EPSG:{resolved_wmts_epsg}",
                bbox[0],
                bbox[2],
                bbox[1],
                bbox[3],
            )
            wmts_bbox: Tuple[float, float, float, float] = (
                w_min_x,
                w_max_x,
                w_min_y,
                w_max_y,
            )
        else:
            wmts_bbox = bbox

        out_sub_xml = os.path.join(work_dir, f"wmts_tmp_sub_xml_{uid_str}.xml")
        _update_wmts_xml(out_tmp_xml, out_sub_xml, wmts_bbox, wmts_zoom_level)

        wmts_tmp_img = os.path.join(work_dir, f"wmts_tmp_img_{uid_str}.tif")
        if resolved_wmts_epsg is not None and resolved_wmts_epsg != bbox_epsg:
            wmts_raw_img = os.path.join(work_dir, f"wmts_raw_img_{uid_str}.tif")
            rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
                out_sub_xml, wmts_raw_img, driver="GTiff"
            )
            rsgispy.imageutils.warp.reproject_image(
                input_img=wmts_raw_img,
                output_img=wmts_tmp_img,
                out_epsg=bbox_epsg,
                interp_method=rsgispy.INTERP_CUBIC,
                gdalformat="GTIFF",
            )
            if os.path.exists(wmts_raw_img):
                os.remove(wmts_raw_img)
        else:
            rasterio.shutil.copy(  # pylint: disable=c-extension-no-member
                out_sub_xml, wmts_tmp_img, driver="GTiff"
            )

        img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
            wmts_tmp_img, bands=[1, 2, 3]
        )

        for tmp_path in (out_tmp_xml, out_sub_xml, wmts_tmp_img):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    ax.imshow(img_data, extent=img_coords)
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    _finalize_map_axis(
        ax=ax,
        epsg_code=bbox_epsg,
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def draw_bboxes_to_axis(
    ax: matplotlib.axes.Axes,
    bboxes: Sequence[Tuple[float, float, float, float]],
    bbox_labels: Optional[Sequence[str]] = None,
    rect_clr: str = "black",
    line_width: float = 1.0,
    fill_rect: bool = False,
    clr_alpha: float = 1.0,
    lbl_font_size: int = 12,
    lbl_font_weight: str = "normal",
    lbl_font_clr: str = "black",
    lbl_pos: Optional[str] = "centre",
    lbl_pos_buf: float = 0.0,
    lbl_fill_clr: Optional[str] = None,
    lbl_padding: float = 3.0,
) -> None:
    """
    Draw one or more bounding box rectangles and optional labels onto ``ax``.

    :param ax: Matplotlib axis on which rectangles are drawn.
    :param bboxes: Sequence of bboxes ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_labels: Optional sequence of text labels matching ``bboxes``.
    :param rect_clr: Edge/fill colour for the rectangles (Default = ``'black'``).
    :param line_width: Rectangle outline width (Default = 1.0).
    :param fill_rect: Whether to fill the rectangle interior (Default = False).
    :param clr_alpha: Alpha transparency for rectangles (Default = 1.0).
    :param lbl_font_size: Label font size (Default = 12).
    :param lbl_font_weight: Label font weight (Default = ``'normal'``).
    :param lbl_font_clr: Label text colour (Default = ``'black'``).
    :param lbl_pos: Label placement relative to each bbox: ``'centre'``, ``'above'``,
                    ``'below'``, ``'left'``, ``'right'``, or ``None``.
    :param lbl_pos_buf: Offset distance between rectangle edge and label.
    :param lbl_fill_clr: Optional background fill colour behind label text.
    :param lbl_padding: Padding around label text when ``lbl_fill_clr`` is set.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.draw_bboxes_to_axis(
            ax=ax,
            bboxes=[(0.0, 10.0, 0.0, 10.0), (15.0, 25.0, 0.0, 10.0)],
            bbox_labels=["Site A", "Site B"],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox_labels is not None and len(bboxes) != len(bbox_labels):
        raise rsgispy.RSGISPyException(
            "If labels are provided then the list length must be the same as bboxes."
        )

    for idx, bbox in enumerate(bboxes):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        bl_x = bbox[0]
        bl_y = bbox[2]
        rect = Rectangle(
            (bl_x, bl_y),
            width,
            height,
            fill=fill_rect,
            color=rect_clr,
            linewidth=line_width,
            alpha=clr_alpha,
        )
        ax.add_patch(rect)

        if bbox_labels is not None:
            cx = bl_x + width / 2.0
            cy = bl_y + height / 2.0
            ha_val = "center"
            va_val = "center"

            if lbl_pos is not None:
                pos_lower = lbl_pos.lower()
                if pos_lower == "above":
                    cy = bl_y + height + lbl_pos_buf
                    va_val = "bottom"
                elif pos_lower == "below":
                    cy = bl_y - lbl_pos_buf
                    va_val = "top"
                elif pos_lower == "left":
                    cx = bl_x - lbl_pos_buf
                    ha_val = "right"
                elif pos_lower == "right":
                    cx = bl_x + width + lbl_pos_buf
                    ha_val = "left"

            bbox_kw = (
                {"facecolor": lbl_fill_clr, "edgecolor": "none", "pad": lbl_padding}
                if lbl_fill_clr is not None
                else None
            )
            ax.annotate(
                bbox_labels[idx],
                (cx, cy),
                color=lbl_font_clr,
                weight=lbl_font_weight,
                fontsize=lbl_font_size,
                ha=ha_val,
                va=va_val,
                bbox=bbox_kw,
            )


def add_axis_label(
    ax: matplotlib.axes.Axes,
    lbl_text: str,
    lbl_font_clr: str = "black",
    lbl_font_weight: str = "normal",
    lbl_font_size: int = 12,
    fill_clr: str = "white",
    padding: float = 3.0,
) -> None:
    """
    Add a label in the top-left corner of ``ax`` (e.g. panel labels ``a``, ``b``).

    :param ax: Matplotlib axis to which the label is added.
    :param lbl_text: Label string to display.
    :param lbl_font_clr: Text colour (Default = ``'black'``).
    :param lbl_font_weight: Font weight (Default = ``'normal'``).
    :param lbl_font_size: Font size (Default = 12).
    :param fill_clr: Background box fill colour (Default = ``'white'``).
    :param padding: Padding around label text (Default = 3.0).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.add_axis_label(ax, lbl_text="(a)")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    xmin, _ = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.text(
        xmin,
        ymax,
        lbl_text,
        color=lbl_font_clr,
        weight=lbl_font_weight,
        fontsize=lbl_font_size,
        ha="left",
        va="top",
        bbox={"facecolor": fill_clr, "edgecolor": "none", "pad": padding},
    )


def _resolve_layer_param(
    param: Union[Any, Sequence[Any]],
    n_lyrs: int,
    idx: int,
    err_msg: str,
) -> Any:
    """Resolve a scalar or per-layer list parameter for multi-layer plotting."""
    if isinstance(param, list):
        if len(param) != n_lyrs:
            raise rsgispy.RSGISPyException(err_msg)
        return param[idx]
    return param


def create_vec_lyr_map(
    ax: matplotlib.axes.Axes,
    gp_vecs: Union[
        gpd.GeoDataFrame,
        gpd.GeoSeries,
        List[Union[gpd.GeoDataFrame, gpd.GeoSeries]],
    ],
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    vec_fill_clrs: Union[str, List[str]] = "grey",
    vec_line_clrs: Union[str, List[str]] = "black",
    vec_line_widths: Union[float, List[float]] = 0.25,
    vec_fill_alphas: Union[float, List[float]] = 1.0,
    vec_markersize: Optional[Union[str, float]] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorders: Union[float, List[float]] = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Add one or more GeoPandas vector layers to ``ax``.

    :param ax: Matplotlib axis to which the vector layer(s) are added.
    :param gp_vecs: GeoDataFrame/GeoSeries or list of layers sharing the same CRS.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional title for the map (Default = None).
    :param vec_fill_clrs: Fill colour string or list of colours (Default = ``'grey'``).
    :param vec_line_clrs: Edge colour string or list of colours (Default = ``'black'``).
    :param vec_line_widths: Line width float or list of widths (Default = 0.25).
    :param vec_fill_alphas: Alpha value or list of alpha values (Default = 1.0).
    :param vec_markersize: Constant marker size or column name for point layers.
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layers to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorders: Drawing z-order float or list of z-orders (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        gdf = gpd.read_file("polys.gpkg")
        rsgispy.tools.mapping.create_vec_lyr_map(
            ax=ax,
            gp_vecs=gdf,
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
            title_str="Study Area",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vec_list = gp_vecs if isinstance(gp_vecs, list) else [gp_vecs]
    n_vec_lyrs = len(vec_list)

    for idx, gp_vec in enumerate(vec_list):
        gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec
        vec_fill_clr = _resolve_layer_param(
            vec_fill_clrs,
            n_vec_lyrs,
            idx,
            "Number of fill colours not same as number of vector layers",
        )
        vec_line_clr = _resolve_layer_param(
            vec_line_clrs,
            n_vec_lyrs,
            idx,
            "Number of line colours not same as number of vector layers",
        )
        vec_line_width = _resolve_layer_param(
            vec_line_widths,
            n_vec_lyrs,
            idx,
            "Number of line widths not same as number of vector layers",
        )
        vec_fill_alpha = _resolve_layer_param(
            vec_fill_alphas,
            n_vec_lyrs,
            idx,
            "Number of fill alphas not same as number of vector layers",
        )
        plot_zorder = _resolve_layer_param(
            plot_zorders,
            n_vec_lyrs,
            idx,
            "Number of zorders not same as number of vector layers",
        )

        if not gp_vec_sub.empty:
            gp_vec_sub.plot(
                ax=ax,
                color=vec_fill_clr,
                edgecolor=vec_line_clr,
                linewidth=vec_line_width,
                alpha=vec_fill_alpha,
                markersize=vec_markersize,
                zorder=plot_zorder,
            )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    vec_epsg = _get_vec_epsg(vec_list[0]) if vec_list else None
    _finalize_map_axis(
        ax=ax,
        epsg_code=vec_epsg,
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def create_raster_img_map(
    ax: matplotlib.axes.Axes,
    input_img: str,
    img_bands: List[int],
    img_stch: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    img_no_data_val: Optional[float] = None,
    no_data_out_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
    stch_min_max_vals: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
    stch_n_stdevs: float = 2.0,
    img_alpha: float = 1.0,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Display a continuous 1-band or 3-band raster layer onto ``ax`` with stretching.

    :param ax: Matplotlib axis to which the image is added.
    :param input_img: Path to the input raster image file.
    :param img_bands: List of 1 or 3 1-based band indices.
    :param img_stch: Stretch constant (``rsgispy.IMG_STRETCH_*``).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full image bounding box is used.
    :param title_str: Optional map title (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param img_no_data_val: Optional no-data value; read from image header if ``None``.
    :param no_data_out_clr: Optional colour value(s) assigned to no-data pixels.
    :param stch_min_max_vals: Stretch min/max dict(s) required when
                              ``img_stch == rsgispy.IMG_STRETCH_USER``.
    :param stch_n_stdevs: Number of standard deviations when
                          ``img_stch == rsgispy.IMG_STRETCH_STDEV`` (Default = 2.0).
    :param img_alpha: Display alpha transparency (Default = 1.0).
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.create_raster_img_map(
            ax=ax,
            input_img="sentinel2.tif",
            img_bands=[4, 3, 2],
            img_stch=rsgispy.IMG_STRETCH_STDEV,
            title_str="RGB Composite",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox is None:
        bbox = rsgispy.imageutils.metadata.get_img_bbox(input_img)

    if img_no_data_val is None:
        with rasterio.open(input_img, "r") as src:
            img_no_data_val = src.nodata

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=img_bands, bbox=bbox
    )

    if img_stch == rsgispy.IMG_STRETCH_USER:
        if stch_min_max_vals is None:
            raise rsgispy.RSGISPyException(
                "Manual stretch requires user parameters to be passed"
            )
        img_data_strch = rsgispy.tools.plotting.manual_stretch_np_arr(
            img_data,
            min_max_vals=stch_min_max_vals,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_LINEAR:
        img_data_strch = rsgispy.tools.plotting.linear_stretch_np_arr(
            img_data,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_STDEV:
        img_data_strch = rsgispy.tools.plotting.stdev_stretch_np_arr(
            img_data,
            n_stdevs=stch_n_stdevs,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_CUMULATIVE:
        img_data_strch = rsgispy.tools.plotting.cumulative_stretch_np_arr(
            img_data,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    else:
        img_data_strch = img_data

    clr_map = "gist_gray" if len(img_bands) == 1 else None
    ax.imshow(
        img_data_strch,
        cmap=clr_map,
        extent=img_coords,
        alpha=img_alpha,
        zorder=plot_zorder,
    )
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=rsgispy.imageutils.metadata.get_epsg_proj_from_img(input_img),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def create_thematic_raster_map(
    ax: matplotlib.axes.Axes,
    input_img: Union[str, List[str]],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    alpha_backgd: bool = True,
    img_alpha: float = 1.0,
    ctab_alpha: bool = True,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    scale_bar_loc: str = "upper right",
    plot_zorders: Union[float, List[float]] = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
    show_legend: bool = False,
    classes_info: Optional[Any] = None,
    legend_position: str = "right",
    legend_title: Optional[str] = None,
    legend_ncols: Optional[int] = None,
    legend_nrows: Optional[int] = None,
    legend_overlay: bool = False,
    legend_loc: str = "lower right",
    legend_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display one or more thematic raster layers onto ``ax`` using the colour
    table stored in each image, optionally attaching a categorical legend.

    :param ax: Matplotlib axis to which the thematic image(s) are added.
    :param input_img: Path or list of paths to thematic raster images.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full extent of the first image is used.
    :param title_str: Optional map title (Default = None).
    :param alpha_backgd: Make pixel value 0 transparent (Default = True).
    :param img_alpha: Display alpha transparency (Default = 1.0).
    :param ctab_alpha: Use alpha channel values from the colour table (Default = True).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorders: Drawing z-order float or list of z-orders (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :param show_legend: If ``True``, automatically create and attach a
                        categorical classification legend (Default = ``False``).
    :param classes_info: Optional class label/colour definitions (dict,
                         ``ClassificationConfig``, or JSON path) passed to the
                         legend builder (Default = ``None``).
    :param legend_position: Companion axis side when ``legend_overlay=False``
                            (``'right'``, ``'left'``, ``'bottom'``, or ``'top'``;
                            Default = ``'right'``).
    :param legend_title: Optional legend title (Default = ``None``).
    :param legend_ncols: Optional explicit legend column count (Default = auto).
    :param legend_nrows: Optional explicit legend row count (Default = auto).
    :param legend_overlay: If ``True``, overlay the legend inside ``ax`` at
                           ``legend_loc`` instead of appending a companion axis
                           (Default = ``False``).
    :param legend_loc: Overlay legend placement string when
                       ``legend_overlay=True`` (Default = ``'lower right'``).
    :param legend_kwargs: Optional extra keyword arguments forwarded to
                          :func:`add_categorical_legend` or
                          :func:`add_categorical_legend_axis`.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        rsgispy.tools.mapping.create_thematic_raster_map(
            ax=ax,
            input_img="landcover.kea",
            title_str="Land Cover",
            show_legend=True,
            classes_info={1: "Forest", 2: "Grassland", 3: "Water"},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    input_imgs = [input_img] if isinstance(input_img, str) else list(input_img)
    if not input_imgs:
        raise rsgispy.RSGISPyException("At least one input image must be provided.")

    if isinstance(plot_zorders, list):
        if len(plot_zorders) != len(input_imgs):
            raise rsgispy.RSGISPyException(
                "The number of zorders provided does not match the number of input images."
            )
        zorder_list = plot_zorders
    else:
        zorder_list = [plot_zorders] * len(input_imgs)

    if bbox is None:
        bbox = rsgispy.imageutils.metadata.get_img_bbox(input_imgs[0])

    img_data_arr_scns = []
    img_coords_scns = []
    for img_path in input_imgs:
        img_data_arr, img_coords, _ = (
            rsgispy.tools.plotting.get_thematic_raster_mpl_imshow(
                img_path,
                band=1,
                bbox=bbox,
                alpha_lyr=alpha_backgd,
                ctab_alpha=ctab_alpha,
            )
        )
        img_data_arr_scns.append(img_data_arr)
        img_coords_scns.append(img_coords)

    ax.set_xlim([img_coords_scns[0][0], img_coords_scns[0][1]])
    ax.set_ylim([img_coords_scns[0][2], img_coords_scns[0][3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    for img_data_arr, img_coords, plot_zorder in zip(
        img_data_arr_scns, img_coords_scns, zorder_list
    ):
        ax.imshow(img_data_arr, extent=img_coords, alpha=img_alpha, zorder=plot_zorder)

    _finalize_map_axis(
        ax=ax,
        epsg_code=rsgispy.imageutils.metadata.get_epsg_proj_from_img(input_imgs[0]),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )

    if show_legend:
        extra_kw = dict(legend_kwargs or {})
        if legend_overlay:
            add_categorical_legend(
                ax=ax,
                classes_info=classes_info,
                input_img=input_imgs[0],
                bbox=bbox,
                ncols=legend_ncols,
                nrows=legend_nrows,
                title=legend_title,
                loc=legend_loc,
                overlay=True,
                **extra_kw,
            )
        else:
            add_categorical_legend_axis(
                ax=ax,
                classes_info=classes_info,
                input_img=input_imgs[0],
                bbox=bbox,
                position=legend_position,
                ncols=legend_ncols,
                nrows=legend_nrows,
                title=legend_title,
                **extra_kw,
            )


def create_choropleth_vec_lyr_map(
    ax: matplotlib.axes.Axes,
    gp_vec: gpd.GeoDataFrame,
    vec_col: str,
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    vec_fill_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    vec_var_norm: Optional[matplotlib.colors.Normalize] = None,
    vec_line_clr: str = "black",
    vec_line_width: float = 0.25,
    vec_fill_alpha: float = 1.0,
    vec_markersize: Optional[Union[str, float]] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> None:
    """
    Display a choropleth vector map coloured by ``vec_col`` on ``ax``.

    :param ax: Matplotlib axis to which the vector layer is added.
    :param gp_vec: GeoDataFrame containing features and attribute column ``vec_col``.
    :param vec_col: Column name used to colour features.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param vec_fill_cmap: Matplotlib colour map name or object (Default = ``'viridis'``).
    :param vec_var_norm: Optional Matplotlib Normalize instance.
    :param vec_line_clr: Edge line colour (Default = ``'black'``).
    :param vec_line_width: Edge line width (Default = 0.25).
    :param vec_fill_alpha: Fill alpha transparency (Default = 1.0).
    :param vec_markersize: Constant marker size or column name for point layers.
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layer to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        gdf = gpd.read_file("regions.gpkg")
        rsgispy.tools.mapping.create_choropleth_vec_lyr_map(
            ax=ax,
            gp_vec=gdf,
            vec_col="area_ha",
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if vec_fill_cmap is None:
        vec_fill_cmap = matplotlib.colormaps["viridis"]

    gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec

    if not gp_vec_sub.empty:
        gp_vec_sub.plot(
            column=vec_col,
            ax=ax,
            cmap=vec_fill_cmap,
            norm=vec_var_norm,
            edgecolor=vec_line_clr,
            linewidth=vec_line_width,
            alpha=vec_fill_alpha,
            markersize=vec_markersize,
            zorder=plot_zorder,
        )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )


def create_raster_cmap_img_map(
    ax: matplotlib.axes.Axes,
    input_img: str,
    img_band: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    cmap_name: str = "viridis",
    norm_img_vals: bool = False,
    use_log_norm: bool = False,
    norm_vmin: Optional[float] = None,
    norm_vmax: Optional[float] = None,
    vals_under_white: bool = False,
    print_norm_vals: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> Tuple[matplotlib.colors.Colormap, Optional[matplotlib.colors.Normalize]]:
    """
    Display a single-band raster layer on ``ax`` using a Matplotlib colour map and
    optional value normalisation.

    :param ax: Matplotlib axis to which the image is added.
    :param input_img: Path to the input raster image file.
    :param img_band: 1-based image band index.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)``. If ``None``,
                 the full image bounding box is used.
    :param title_str: Optional map title (Default = None).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param cmap_name: Matplotlib colour map name (Default = ``'viridis'``).
    :param norm_img_vals: Whether to normalise pixel values (Default = False).
    :param use_log_norm: Whether to use logarithmic normalisation (Default = False).
    :param norm_vmin: Minimum normalisation value (auto-computed if ``None``).
    :param norm_vmax: Maximum normalisation value (auto-computed if ``None``).
    :param vals_under_white: Render values below ``norm_vmin`` as white.
    :param print_norm_vals: Print computed normalisation min/max values.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colourbar.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping

        fig, ax = plt.subplots()
        cmap, norm = rsgispy.tools.mapping.create_raster_cmap_img_map(
            ax=ax,
            input_img="ndvi.tif",
            img_band=1,
            cmap_name="YlGn",
            norm_img_vals=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if bbox is None:
        bbox = rsgispy.imageutils.metadata.get_img_bbox(input_img)

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=[img_band], bbox=bbox
    )

    c_cmap = plt.get_cmap(cmap_name).copy()
    c_norm: Optional[matplotlib.colors.Normalize] = None
    if norm_img_vals:
        if norm_vmin is None:
            norm_vmin = float(np.nanmin(img_data))
        if norm_vmax is None:
            norm_vmax = float(np.nanmax(img_data))

        if vals_under_white:
            c_cmap = c_cmap.with_extremes(under="white")

        if use_log_norm:
            c_norm = matplotlib.colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax)
        else:
            c_norm = matplotlib.colors.Normalize(vmin=norm_vmin, vmax=norm_vmax)

        if print_norm_vals:
            print(f"{norm_vmin} - {norm_vmax}")

    ax.imshow(img_data, extent=img_coords, cmap=c_cmap, norm=c_norm, zorder=plot_zorder)
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=rsgispy.imageutils.metadata.get_epsg_proj_from_img(input_img),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )

    return c_cmap, c_norm


def create_vec_pt_density_map(
    ax: matplotlib.axes.Axes,
    gp_vec: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    title_str: Optional[str] = None,
    cmap_name: str = "viridis",
    use_log_norm: bool = False,
    density_norm_vmin: float = 1.0,
    density_norm_vmax: Optional[float] = None,
    density_dpi: int = 72,
    show_scale_bar: bool = True,
    use_grid: bool = False,
    show_map_axis: bool = True,
    sub_in_vec: bool = False,
    scale_bar_loc: str = "upper right",
    plot_zorder: float = 1.0,
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    show_north_arrow: bool = False,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
) -> Tuple[matplotlib.colors.Colormap, matplotlib.colors.Normalize]:
    """
    Add a point-density plot from a GeoPandas points layer onto ``ax``.

    Note: ``ax`` must be created with ``projection='scatter_density'``, which
    requires the optional ``mpl_scatter_density`` package.

    :param ax: Matplotlib axis created with ``projection='scatter_density'``.
    :param gp_vec: GeoDataFrame containing point geometries.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)`` for the display extent.
    :param title_str: Optional map title (Default = None).
    :param cmap_name: Matplotlib colour map name (Default = ``'viridis'``).
    :param use_log_norm: Use logarithmic density normalisation (Default = False).
    :param density_norm_vmin: Minimum density normalisation value (Default = 1.0).
    :param density_norm_vmax: Maximum density normalisation value (Default = None).
    :param density_dpi: DPI used to bin point density (Default = 72).
    :param show_scale_bar: Whether to add a scale bar (Default = True).
    :param use_grid: Whether to enable axis gridlines (Default = False).
    :param show_map_axis: Whether to display axis ticks and frame (Default = True).
    :param sub_in_vec: Spatially subset vector layer to ``bbox`` before plotting.
    :param scale_bar_loc: Scale bar placement string (Default = ``'upper right'``).
    :param plot_zorder: Drawing z-order (Default = 1.0).
    :param scale_bar_style: Scale bar style constant (Default = :data:`SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant
                              (Default = :data:`NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colourbar.

    .. code:: python

        import geopandas as gpd
        import matplotlib.pyplot as plt
        import mpl_scatter_density  # noqa: F401
        import rsgispy.tools.mapping

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        pts_gdf = gpd.read_file("points.gpkg")
        cmap, norm = rsgispy.tools.mapping.create_vec_pt_density_map(
            ax=ax,
            gp_vec=pts_gdf,
            bbox=(250000.0, 270000.0, 280000.0, 290000.0),
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel,import-error,unused-import
    import mpl_scatter_density  # noqa: F401

    gp_vec_sub = _subset_vec_bbox(gp_vec, bbox) if sub_in_vec else gp_vec

    c_cmap = plt.get_cmap(cmap_name).copy().with_extremes(under=[1, 1, 1, 0])

    if use_log_norm:
        c_norm: matplotlib.colors.Normalize = matplotlib.colors.LogNorm(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )
    else:
        c_norm = matplotlib.colors.Normalize(
            vmin=density_norm_vmin, vmax=density_norm_vmax
        )

    ax.scatter_density(
        gp_vec_sub["geometry"].x,
        gp_vec_sub["geometry"].y,
        dpi=density_dpi,
        norm=c_norm,
        cmap=c_cmap,
        zorder=plot_zorder,
    )

    ax.set_aspect("equal")
    ax.set_xlim([bbox[0], bbox[1]])
    ax.set_ylim([bbox[2], bbox[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalize_map_axis(
        ax=ax,
        epsg_code=_get_vec_epsg(gp_vec),
        bbox=bbox,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        scale_bar_loc=scale_bar_loc,
        title_str=title_str,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
    )

    return c_cmap, c_norm


def create_categorical_legend_patches(
    classes_info: Optional[Any] = None,
    input_img: Optional[Union[str, os.PathLike]] = None,
    band: int = 1,
    colour_lut: Optional[Mapping[int, Any]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ignore_zero: bool = True,
    include_vals: Optional[Sequence[int]] = None,
    exclude_vals: Optional[Sequence[int]] = None,
    only_present_in_img: bool = False,
    sort_by: str = "id",
    show_pxl_vals: bool = False,
    pxl_val_fmt: str = "{val}: {name}",
    max_label_width: Optional[int] = None,
    swatch_shape: str = "rectangle",
    swatch_edgecolor: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
) -> List[Any]:
    """
    Build styled Matplotlib legend handles (``Patch`` or ``Line2D`` markers)
    for categorical or classification maps.

    Accepts a ``{pixel_val: class_name}`` dict, ``{pixel_val: (name, colour)}``
    dict, ``{class_name: ClassInfoObj}`` dict, :class:`ClassificationConfig`,
    classification config JSON path, or a thematic raster ``input_img`` with
    an embedded colour table.

    :param classes_info: Class definition dict, :class:`ClassificationConfig`,
                         sequence of :class:`ClassInfoObj`, or JSON file path.
    :param input_img: Optional thematic raster used to extract class colours
                      (when not explicitly supplied) or filter present classes.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent ``(MinX, MaxX, MinY, MaxY)`` used when
                 ``only_present_in_img=True``.
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional explicit whitelist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude from the legend.
    :param only_present_in_img: If ``True`` and ``input_img`` is provided,
                                show only classes actually present in the raster.
    :param sort_by: Class ordering: ``'id'``, ``'name'``, or ``'none'``
                    (Default = ``'id'``).
    :param show_pxl_vals: Prefix/format labels with pixel IDs (Default = ``False``).
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``
                        (Default = ``'{val}: {name}'``).
    :param max_label_width: Optional character width for wrapping long labels.
    :param swatch_shape: Marker shape: ``'rectangle'``, ``'square'``, or
                         ``'circle'`` (Default = ``'rectangle'``).
    :param swatch_edgecolor: Swatch outline colour (Default = ``'#333333'``).
    :param swatch_linewidth: Swatch outline width (Default = ``0.6``).
    :param swatch_size: Marker size when ``swatch_shape`` is ``'square'`` or
                        ``'circle'`` (Default = ``9.0``).
    :return: List of Matplotlib legend handles.

    .. code:: python

        import rsgispy.tools.mapping as mapping

        handles = mapping.create_categorical_legend_patches(
            classes_info={1: "Forest", 2: "Grassland", 3: "Urban"},
            colour_lut={1: "#228b22", 2: "#9acd32", 3: "#d62728"},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    entries = resolve_legend_entries(
        classes_info=classes_info,
        input_img=input_img,
        band=band,
        colour_lut=colour_lut,
        bbox=bbox,
        ignore_zero=ignore_zero,
        include_vals=include_vals,
        exclude_vals=exclude_vals,
        only_present_in_img=only_present_in_img,
        sort_by=sort_by,
        show_pxl_vals=show_pxl_vals,
        pxl_val_fmt=pxl_val_fmt,
        max_label_width=max_label_width,
    )
    return build_legend_handles(
        entries=entries,
        swatch_shape=swatch_shape,
        swatch_edgecolor=swatch_edgecolor,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
    )


def add_categorical_legend(
    ax: matplotlib.axes.Axes,
    classes_info: Optional[Any] = None,
    input_img: Optional[Union[str, os.PathLike]] = None,
    band: int = 1,
    colour_lut: Optional[Mapping[int, Any]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    title: Optional[str] = None,
    loc: str = "upper right",
    bbox_to_anchor: Optional[Tuple[float, ...]] = None,
    overlay: bool = True,
    fontsize: float = 9.5,
    title_fontsize: Optional[float] = None,
    ignore_zero: bool = True,
    include_vals: Optional[Sequence[int]] = None,
    exclude_vals: Optional[Sequence[int]] = None,
    only_present_in_img: bool = False,
    sort_by: str = "id",
    show_pxl_vals: bool = False,
    pxl_val_fmt: str = "{val}: {name}",
    max_label_width: Optional[int] = None,
    swatch_shape: str = "rectangle",
    swatch_edgecolor: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
    frameon: bool = True,
    facecolor: str = "white",
    edgecolor: str = "#CCCCCC",
    framealpha: float = 0.92,
    fancybox: bool = True,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    labelspacing: float = 0.45,
    borderpad: float = 0.6,
) -> Any:
    """
    Add a professionally formatted categorical legend onto an existing Matplotlib
    axis ``ax``.

    By default (``overlay=True``), the legend is placed on the map axis at
    ``loc`` (e.g. ``'lower right'``). Set ``overlay=False`` when ``ax`` is a
    dedicated subplot/panel reserved exclusively for the legend (ticks and frame
    on ``ax`` will be hidden and the legend centred cleanly).

    When both ``ncols`` and ``nrows`` are ``None`` (auto mode), column and row
    counts are chosen automatically from the axis size and class label lengths.

    :param ax: Target Matplotlib axis.
    :param classes_info: ``{pixel_val: class_name}`` dict,
                         ``{pixel_val: (name, colour)}`` dict,
                         ``{class_name: ClassInfoObj}`` dict,
                         :class:`ClassificationConfig`, or JSON file path.
    :param input_img: Optional thematic raster for extracting embedded colours
                      or filtering visible classes.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent used when ``only_present_in_img=True``.
    :param ncols: Explicit number of columns (Default = ``None``, auto).
    :param nrows: Explicit number of rows (Default = ``None``, auto).
    :param title: Optional bold legend title.
    :param loc: Matplotlib legend placement string (Default = ``'upper right'``;
                ignored and centred when ``overlay=False``).
    :param bbox_to_anchor: Optional Matplotlib ``bbox_to_anchor`` coordinates.
    :param overlay: If ``True``, overlay onto ``ax``; if ``False``, turn off
                    ``ax`` frame/ticks and draw as a standalone panel legend.
    :param fontsize: Class label font size in points (Default = ``9.5``).
    :param title_fontsize: Optional title font size (Default = ``fontsize + 1``).
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional whitelist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edgecolor: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Default = ``True``).
    :param facecolor: Legend background colour (Default = ``'white'``).
    :param edgecolor: Legend border colour (Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Default = ``0.92``).
    :param fancybox: Rounded legend corners (Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Default = ``0.45``).
    :param borderpad: Internal frame padding (Default = ``0.6``).
    :return: Created :class:`matplotlib.legend.Legend` instance.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping as mapping

        fig, ax = plt.subplots()
        mapping.add_categorical_legend(
            ax=ax,
            classes_info={1: "Woodland", 2: "Grassland", 3: "Urban"},
            colour_lut={1: "#228b22", 2: "#9acd32", 3: "#d62728"},
            title="Land Cover",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    entries = resolve_legend_entries(
        classes_info=classes_info,
        input_img=input_img,
        band=band,
        colour_lut=colour_lut,
        bbox=bbox,
        ignore_zero=ignore_zero,
        include_vals=include_vals,
        exclude_vals=exclude_vals,
        only_present_in_img=only_present_in_img,
        sort_by=sort_by,
        show_pxl_vals=show_pxl_vals,
        pxl_val_fmt=pxl_val_fmt,
        max_label_width=max_label_width,
    )
    ax_w, ax_h = _measure_axis_inches(ax)
    eff_avail_w = ax_w if not overlay else max(2.0, ax_w * 0.75)
    _, eff_cols = calc_categorical_legend_layout(
        labels=[e.label for e in entries],
        ncols=ncols,
        nrows=nrows,
        avail_width_in=eff_avail_w,
        avail_height_in=ax_h,
        fontsize=fontsize,
        prefer_vertical=False,
    )
    handles = build_legend_handles(
        entries=entries,
        swatch_shape=swatch_shape,
        swatch_edgecolor=swatch_edgecolor,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
    )
    if not overlay:
        ax.set_axis_off()
        eff_loc = "center"
        eff_anchor: Optional[Tuple[float, ...]] = (0.5, 0.5)
    else:
        eff_loc = _LOC_ALIASES.get(loc.strip().lower(), loc)
        eff_anchor = bbox_to_anchor

    return _style_legend(
        ax=ax,
        handles=handles,
        eff_cols=eff_cols,
        loc=eff_loc,
        bbox_to_anchor=eff_anchor,
        title=title,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        frameon=frameon,
        facecolor=facecolor,
        edgecolor=edgecolor,
        framealpha=framealpha,
        fancybox=fancybox,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
        borderpad=borderpad,
    )


def add_categorical_legend_axis(
    ax: matplotlib.axes.Axes,
    classes_info: Optional[Any] = None,
    input_img: Optional[Union[str, os.PathLike]] = None,
    band: int = 1,
    colour_lut: Optional[Mapping[int, Any]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    position: str = "right",
    size: Optional[str] = None,
    pad: float = 0.15,
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    title: Optional[str] = None,
    fontsize: float = 9.5,
    title_fontsize: Optional[float] = None,
    ignore_zero: bool = True,
    include_vals: Optional[Sequence[int]] = None,
    exclude_vals: Optional[Sequence[int]] = None,
    only_present_in_img: bool = False,
    sort_by: str = "id",
    show_pxl_vals: bool = False,
    pxl_val_fmt: str = "{val}: {name}",
    max_label_width: Optional[int] = None,
    swatch_shape: str = "rectangle",
    swatch_edgecolor: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
    frameon: bool = True,
    facecolor: str = "white",
    edgecolor: str = "#CCCCCC",
    framealpha: float = 0.95,
    fancybox: bool = True,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    labelspacing: float = 0.45,
    borderpad: float = 0.6,
) -> Tuple[matplotlib.axes.Axes, Any]:
    """
    Create a new companion Matplotlib axis next to ``ax`` and render a
    formatted categorical legend inside it.

    Supports placement on ``'right'``, ``'left'``, ``'bottom'``, or ``'top'``.
    When ``ncols`` and ``nrows`` are omitted, the column/row grid adapts
    automatically to the companion axis dimensions and class label lengths.

    :param ax: Existing map Matplotlib axis to attach the companion legend axis to.
    :param classes_info: Class dictionary, :class:`ClassificationConfig`, or JSON path.
    :param input_img: Optional thematic raster for extracting embedded colours
                      or filtering visible classes.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent used when ``only_present_in_img=True``.
    :param position: Side relative to ``ax``: ``'right'``, ``'left'``,
                     ``'bottom'``, or ``'top'`` (Default = ``'right'``).
    :param size: Optional axes-divider size string (e.g. ``'25%'``; auto when ``None``).
    :param pad: Gap in inches between ``ax`` and the new legend axis (Default = ``0.15``).
    :param ncols: Explicit legend column count (Default = ``None``, auto).
    :param nrows: Explicit legend row count (Default = ``None``, auto).
    :param title: Optional bold legend title.
    :param fontsize: Class label font size in points (Default = ``9.5``).
    :param title_fontsize: Optional title font size (Default = ``fontsize + 1``).
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional whitelist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edgecolor: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Default = ``True``).
    :param facecolor: Legend background colour (Default = ``'white'``).
    :param edgecolor: Legend border colour (Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Default = ``0.95``).
    :param fancybox: Rounded legend corners (Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Default = ``0.45``).
    :param borderpad: Internal frame padding (Default = ``0.6``).
    :return: Tuple ``(legend_ax, legend)``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.mapping as mapping

        fig, ax = plt.subplots()
        mapping.create_thematic_raster_map(ax=ax, input_img="landcover.kea")
        leg_ax, lgd = mapping.add_categorical_legend_axis(
            ax=ax,
            classes_info={1: "Forest", 2: "Grassland", 3: "Water"},
            input_img="landcover.kea",
            position="right",
            title="Classes",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    return create_companion_legend_axis(
        ax=ax,
        classes_info=classes_info,
        input_img=input_img,
        band=band,
        colour_lut=colour_lut,
        bbox=bbox,
        position=position,
        size=size,
        pad=pad,
        ncols=ncols,
        nrows=nrows,
        title=title,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        ignore_zero=ignore_zero,
        include_vals=include_vals,
        exclude_vals=exclude_vals,
        only_present_in_img=only_present_in_img,
        sort_by=sort_by,
        show_pxl_vals=show_pxl_vals,
        pxl_val_fmt=pxl_val_fmt,
        max_label_width=max_label_width,
        swatch_shape=swatch_shape,
        swatch_edgecolor=swatch_edgecolor,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
        frameon=frameon,
        facecolor=facecolor,
        edgecolor=edgecolor,
        framealpha=framealpha,
        fancybox=fancybox,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
        borderpad=borderpad,
    )


def save_categorical_legend_img(
    out_img_file: Union[str, os.PathLike],
    classes_info: Optional[Any] = None,
    input_img: Optional[Union[str, os.PathLike]] = None,
    band: int = 1,
    colour_lut: Optional[Mapping[int, Any]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    fig_width: float = 6.5,
    title: Optional[str] = None,
    fontsize: float = 10.0,
    title_fontsize: Optional[float] = None,
    ignore_zero: bool = True,
    include_vals: Optional[Sequence[int]] = None,
    exclude_vals: Optional[Sequence[int]] = None,
    only_present_in_img: bool = False,
    sort_by: str = "id",
    show_pxl_vals: bool = False,
    pxl_val_fmt: str = "{val}: {name}",
    max_label_width: Optional[int] = None,
    swatch_shape: str = "rectangle",
    swatch_edgecolor: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
    frameon: bool = True,
    facecolor: str = "white",
    edgecolor: str = "#CCCCCC",
    framealpha: float = 1.0,
    fancybox: bool = True,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    labelspacing: float = 0.45,
    borderpad: float = 0.6,
    dpi: int = 300,
    transparent: bool = False,
) -> Tuple[int, int]:
    """
    Create and save a standalone categorical legend image file (e.g. PNG, PDF,
    or SVG) tightly cropped around the formatted legend.

    :param out_img_file: Output image file path (e.g. ``'legend.png'``).
    :param classes_info: Class dictionary, :class:`ClassificationConfig`, or JSON path.
    :param input_img: Optional thematic raster for extracting embedded colours
                      or filtering visible classes.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent used when ``only_present_in_img=True``.
    :param ncols: Explicit legend column count (Default = ``None``, auto).
    :param nrows: Explicit legend row count (Default = ``None``, auto).
    :param fig_width: Target available width in inches for auto-layout
                      calculation (Default = ``6.5``).
    :param title: Optional bold legend title.
    :param fontsize: Class label font size in points (Default = ``10.0``).
    :param title_fontsize: Optional title font size (Default = ``fontsize + 1``).
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional whitelist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edgecolor: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Default = ``True``).
    :param facecolor: Legend background colour (Default = ``'white'``).
    :param edgecolor: Legend border colour (Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Default = ``1.0``).
    :param fancybox: Rounded legend corners (Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Default = ``0.45``).
    :param borderpad: Internal frame padding (Default = ``0.6``).
    :param dpi: Output figure resolution in DPI (Default = ``300``).
    :param transparent: If ``True``, export with a transparent background.
    :return: Tuple ``(nrows, ncols)`` used in the saved legend layout.

    .. code:: python

        import rsgispy.tools.mapping as mapping

        nrows, ncols = mapping.save_categorical_legend_img(
            out_img_file="landcover_legend.png",
            classes_info={1: "Forest", 2: "Grassland", 3: "Urban", 4: "Water"},
            colour_lut={1: "#228b22", 2: "#9acd32", 3: "#d62728", 4: "#1f77b4"},
            title="Land Cover Classes",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    return save_standalone_legend(
        out_img_file=out_img_file,
        classes_info=classes_info,
        input_img=input_img,
        band=band,
        colour_lut=colour_lut,
        bbox=bbox,
        ncols=ncols,
        nrows=nrows,
        fig_width=fig_width,
        title=title,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        ignore_zero=ignore_zero,
        include_vals=include_vals,
        exclude_vals=exclude_vals,
        only_present_in_img=only_present_in_img,
        sort_by=sort_by,
        show_pxl_vals=show_pxl_vals,
        pxl_val_fmt=pxl_val_fmt,
        max_label_width=max_label_width,
        swatch_shape=swatch_shape,
        swatch_edgecolor=swatch_edgecolor,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
        frameon=frameon,
        facecolor=facecolor,
        edgecolor=edgecolor,
        framealpha=framealpha,
        fancybox=fancybox,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
        borderpad=borderpad,
        dpi=dpi,
        transparent=transparent,
    )
