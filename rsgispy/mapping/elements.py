"""
The mapping.elements module provides cartographic map decorations and
annotations—including scale bars, north arrows, bounding-box overlays, and
panel labels—for Matplotlib map axes.
"""

import math
from typing import Any, Optional, Sequence, Tuple

import matplotlib.axes
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    TextArea,
    VPacker,
)
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Affine2D
import numpy as np
import pyproj
import pyproj.exceptions

import rsgispy
from rsgispy.mapping._common import resolve_mpl_location

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
    "draw_bboxes_to_axis",
    "add_axis_label",
]


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
    """
    Compute geodesic ground metres per x-axis map unit at the ``bbox`` midpoint.
    """
    if epsg_code is None:
        return 1.0
    try:
        crs = pyproj.CRS.from_epsg(int(epsg_code))
        x_mid = 0.5 * (float(bbox[0]) + float(bbox[1]))
        y_mid = 0.5 * (float(bbox[2]) + float(bbox[3]))
        span_x = abs(float(bbox[1]) - float(bbox[0]))

        if crs.is_geographic:
            geod = crs.get_geod() or pyproj.Geod(ellps="WGS84")
            lat_mid = max(-89.9, min(89.9, y_mid))
            delta = min(0.5, max(span_x * 0.01, 1e-5))
            _, _, dist_m = geod.inv(
                x_mid - 0.5 * delta, lat_mid, x_mid + 0.5 * delta, lat_mid
            )
            if math.isfinite(dist_m) and dist_m > 0.0:
                return float(dist_m / delta)

        geog_crs = crs.geodetic_crs or pyproj.CRS.from_epsg(4326)
        geod = geog_crs.get_geod() or pyproj.Geod(ellps="WGS84")
        transformer = pyproj.Transformer.from_crs(crs, geog_crs, always_xy=True)
        delta = max(span_x * 0.01, 1e-3) if span_x > 0.0 else 1.0
        lon1, lat1 = transformer.transform(x_mid - 0.5 * delta, y_mid)
        lon2, lat2 = transformer.transform(x_mid + 0.5 * delta, y_mid)
        if (
            all(math.isfinite(v) for v in (lon1, lat1, lon2, lat2))
            and abs(lat1) <= 90.0
            and abs(lat2) <= 90.0
        ):
            _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            if math.isfinite(dist_m) and dist_m > 0.0:
                return float(dist_m / delta)

        if crs.axis_info:
            factor = crs.axis_info[0].unit_conversion_factor
            if factor is not None and factor > 0:
                return float(factor)
    except (pyproj.exceptions.CRSError, ValueError):
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
        colour: str = "black",
        alt_colour: str = "white",
        box_colour: str = "white",
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
        self.colour = colour
        self.alt_colour = alt_colour
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
            self.patch.set_facecolor(box_colour)
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
                    "color": self.colour,
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
                    facecolor=self.colour,
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
                fill_c = self.colour if idx % 2 == 0 else self.alt_colour
                self._drawing_area.add_artist(
                    Rectangle(
                        (left_pad + idx * seg_w, 0.0),
                        seg_w,
                        self.bar_height,
                        facecolor=fill_c,
                        edgecolor=self.colour,
                        linewidth=0.8,
                    )
                )
        else:
            y_base = 1.0
            self._drawing_area.add_artist(
                Line2D(
                    [left_pad, left_pad + bar_pts],
                    [y_base, y_base],
                    color=self.colour,
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
                        color=self.colour,
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
                    color=self.colour,
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
                color=self.colour,
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
        colour: str = "black",
        alt_colour: str = "white",
        label: str = "N",
        font_size: float = 10.0,
        box_colour: str = "white",
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
        self.colour = colour
        self.alt_colour = alt_colour
        self.label = label
        self.rotation = float(rotation)

        arrow_da = _build_north_arrow_drawing(
            style=style,
            size=self.size,
            colour=colour,
            alt_colour=alt_colour,
            rotation=self.rotation,
        )
        if label:
            lbl_area = TextArea(
                label,
                textprops={
                    "fontsize": font_size,
                    "fontweight": "bold",
                    "color": colour,
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
            self.patch.set_facecolor(box_colour)
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
    colour: str,
    alt_colour: str,
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
        _add_poly([apex, left_pt, notch], facecolor=colour, edgecolor=colour)
        _add_poly([apex, notch, right_pt], facecolor=alt_colour, edgecolor=colour)
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
            facecolor=colour,
            edgecolor=colour,
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
            ([(cx, cy), n_pt, nw_pt], colour),
            ([(cx, cy), ne_pt, n_pt], alt_colour),
            ([(cx, cy), e_pt, ne_pt], colour),
            ([(cx, cy), se_pt, e_pt], alt_colour),
            ([(cx, cy), s_pt, se_pt], colour),
            ([(cx, cy), sw_pt, s_pt], alt_colour),
            ([(cx, cy), w_pt, sw_pt], colour),
            ([(cx, cy), nw_pt, w_pt], alt_colour),
        ]
        for coords, fc in facets:
            _add_poly(coords, facecolor=fc, edgecolor=colour, linewidth=0.7)
    else:
        apex = (cx, h * 0.94)
        notch = (cx, h * 0.26)
        left_pt = (w * 0.20, h * 0.06)
        right_pt = (w * 0.80, h * 0.06)
        _add_poly(
            [apex, right_pt, notch, left_pt],
            facecolor=alt_colour,
            edgecolor=colour,
            linewidth=1.5,
        )
        stem = _rotate_points([notch, apex], cx, cy, rotation)
        da.add_artist(
            Line2D(stem[:, 0], stem[:, 1], color=colour, linewidth=1.2)
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
    colour: str = "black",
    alt_colour: str = "white",
    box_colour: str = "white",
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
    :param colour: Primary bar/text colour (Default = ``'black'``).
    :param alt_colour: Secondary segment colour for alternating style (Default = ``'white'``).
    :param box_colour: Background box fill colour (Default = ``'white'``).
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
        import rsgispy.mapping.elements as elements
        import rsgispy.mapping.layout as layout

        fig, ax = plt.subplots()
        layout.define_axis_extent(
            ax, bbox=(500000.0, 510000.0, 5200000.0, 5210000.0)
        )
        elements.add_scale_bar(
            ax,
            epsg_code=32630,
            style=elements.SCALEBAR_STYLE_ALTERNATING,
            units=elements.SCALEBAR_UNIT_KILOMETRES,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if style not in _SCALEBAR_STYLES:
        valid_styles = ", ".join(sorted(_SCALEBAR_STYLES))
        raise rsgispy.RSGISPyException(
            f"Invalid scale bar style '{style}'. Valid options: {valid_styles}."
        )
    norm_loc = resolve_mpl_location(loc)
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
        colour=colour,
        alt_colour=alt_colour,
        box_colour=box_colour,
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
    colour: str = "black",
    alt_colour: str = "white",
    label: str = "N",
    font_size: float = 10.0,
    box_colour: str = "white",
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
    :param colour: Primary stroke and fill colour (Default = ``'black'``).
    :param alt_colour: Secondary fill colour for split/outlined styles (Default = ``'white'``).
    :param label: Label drawn above the arrow symbol (Default = ``'N'``).
    :param font_size: Label font size in points (Default = 10.0).
    :param box_colour: Background box fill colour (Default = ``'white'``).
    :param box_alpha: Background box opacity (Default = 0.8).
    :param frameon: Whether to draw a background box (Default = True).
    :param pad: Internal box padding (Default = 0.4).
    :param borderpad: Margin between box and axis edges (Default = 0.5).
    :param rotation: Anticlockwise arrow rotation in degrees (Default = 0.0).
    :param zorder: Artist z-order (Default = 10.0).
    :return: Created :class:`MapNorthArrow` artist.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.elements as elements
        import rsgispy.mapping.layout as layout

        fig, ax = plt.subplots()
        layout.define_axis_extent(
            ax, bbox=(500000.0, 510000.0, 5200000.0, 5210000.0)
        )
        elements.add_north_arrow(
            ax,
            loc="upper left",
            style=elements.NORTH_ARROW_STYLE_COMPASS,
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
    norm_loc = resolve_mpl_location(loc)

    artist = MapNorthArrow(
        style=style,
        loc=norm_loc,
        size=size,
        colour=colour,
        alt_colour=alt_colour,
        label=label,
        font_size=font_size,
        box_colour=box_colour,
        box_alpha=box_alpha,
        frameon=frameon,
        pad=pad,
        borderpad=borderpad,
        rotation=rotation,
        zorder=zorder,
    )
    ax.add_artist(artist)
    return artist


def _finalise_map_axis(
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
        import rsgispy.mapping.elements as elements

        fig, ax = plt.subplots()
        elements.draw_bboxes_to_axis(
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
        import rsgispy.mapping.elements as elements

        fig, ax = plt.subplots()
        elements.add_axis_label(ax, lbl_text="(a)")
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ax.text(
        0.0,
        1.0,
        lbl_text,
        transform=ax.transAxes,
        color=lbl_font_clr,
        weight=lbl_font_weight,
        fontsize=lbl_font_size,
        ha="left",
        va="top",
        bbox={"facecolor": fill_clr, "edgecolor": "none", "pad": padding},
    )
