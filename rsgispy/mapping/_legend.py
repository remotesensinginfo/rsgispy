"""
Internal helper functions for categorical and classification map legends in
:mod:`rsgispy.mapping.legend`.
"""

from dataclasses import dataclass
import math
import os
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.colors
import matplotlib.legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import ensure_file_exists, iter_dataset_windows
from rsgispy.tools.files import ensure_parent_dir

_DEFAULT_PALETTE: Tuple[str, ...] = (
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#98df8a",
    "#ff9896",
    "#ffbb78",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)


@dataclass
class LegendEntry:
    """Resolved categorical legend item."""

    pxl_val: int
    label: str
    colour: Tuple[float, float, float, float]


def _normalise_rgba(
    clr: Any,
) -> Tuple[float, float, float, float]:
    """Normalise hex string, named colour, or RGB/RGBA tuple to 0..1 RGBA."""
    if isinstance(clr, str):
        return matplotlib.colors.to_rgba(clr)
    if isinstance(clr, (tuple, list, np.ndarray)) and len(clr) in (3, 4):
        vals = [float(v) for v in clr]
        if any(v > 1.0 for v in vals):
            vals = [v / 255.0 for v in vals]
        if len(vals) == 3:
            vals.append(1.0)
        return (
            float(np.clip(vals[0], 0.0, 1.0)),
            float(np.clip(vals[1], 0.0, 1.0)),
            float(np.clip(vals[2], 0.0, 1.0)),
            float(np.clip(vals[3], 0.0, 1.0)),
        )
    raise rsgispy.RSGISPyException(
        f"Invalid legend colour specification: {clr!r}."
    )


def _read_raster_colormap(
    input_img: str,
    band: int = 1,
) -> Dict[int, Tuple[float, float, float, float]]:
    """Read optional raster colour table as ``{pixel_val: (r, g, b, a)}``."""
    ensure_file_exists(input_img, "Input thematic image")
    with rasterio.open(input_img, "r") as src:
        if band < 1 or band > src.count:
            raise rsgispy.RSGISPyException(
                f"Band {band} is out of range for image with {src.count} band(s)."
            )
        try:
            cmap = src.colormap(band)
        except ValueError:
            return {}
        if not cmap:
            return {}
        return {int(k): _normalise_rgba(v) for k, v in cmap.items()}


def _scan_present_values(
    input_img: str,
    band: int = 1,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> set:
    """Scan unique integer pixel values present inside ``input_img`` (or ``bbox``)."""
    ensure_file_exists(input_img, "Input thematic image")
    present: set = set()
    with rasterio.open(input_img, "r") as src:
        if bbox is not None:
            win = src.window(*bbox)
            arr = src.read(band, window=win)
            present.update(int(v) for v in np.unique(arr) if np.isfinite(v))
        else:
            for win in iter_dataset_windows(src.height, src.width, use_pbar=False):
                arr = src.read(band, window=win)
                present.update(int(v) for v in np.unique(arr) if np.isfinite(v))
    return present


def _extract_entries_from_class_objs(
    items: Sequence[Tuple[Optional[str], Any]],
) -> List[Tuple[int, str, Optional[Tuple[float, float, float, float]]]]:
    """Extract raw ``(out_id, name, rgba)`` from ``ClassInfoObj`` instances."""
    out: List[Tuple[int, str, Optional[Tuple[float, float, float, float]]]] = []
    for key_name, obj in items:
        pxl_id = int(obj.out_id)
        cls_name = obj.name if obj.name else (key_name or str(pxl_id))
        rgba = (
            float(obj.red) / 255.0,
            float(obj.green) / 255.0,
            float(obj.blue) / 255.0,
            1.0,
        )
        out.append((pxl_id, str(cls_name), rgba))
    return out


def resolve_legend_entries(
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
) -> List[LegendEntry]:
    """
    Resolve categorical classes from dicts, classification configs, JSON files,
    or thematic raster colour tables into ordered, styled :class:`LegendEntry` items.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,import-outside-toplevel
    from rsgispy.classification.classdata import ClassInfoObj, ClassificationConfig

    img_path = os.fspath(input_img) if input_img is not None else None
    raster_cmap: Dict[int, Tuple[float, float, float, float]] = (
        _read_raster_colormap(img_path, band=band) if img_path is not None else {}
    )
    explicit_lut: Dict[int, Tuple[float, float, float, float]] = (
        {int(k): _normalise_rgba(v) for k, v in colour_lut.items()}
        if colour_lut is not None
        else {}
    )

    raw_items: List[Tuple[int, str, Optional[Tuple[float, float, float, float]]]] = []

    if classes_info is None:
        if not raster_cmap:
            raise rsgispy.RSGISPyException(
                "Provide classes_info or an input_img with an embedded colour table."
            )
        for val, rgba in sorted(raster_cmap.items()):
            raw_items.append((val, f"Class {val}", rgba))
    elif isinstance(classes_info, (str, os.PathLike)):
        cfg = ClassificationConfig.load(os.fspath(classes_info))
        raw_items = _extract_entries_from_class_objs(list(cfg.class_info.items()))
    elif isinstance(classes_info, ClassificationConfig):
        raw_items = _extract_entries_from_class_objs(
            list(classes_info.class_info.items())
        )
    elif isinstance(classes_info, Mapping):
        if not classes_info:
            raise rsgispy.RSGISPyException("classes_info mapping must not be empty.")
        sample_val = next(iter(classes_info.values()))
        if isinstance(sample_val, ClassInfoObj):
            raw_items = _extract_entries_from_class_objs(list(classes_info.items()))
        else:
            for k, v in classes_info.items():
                pxl_id = int(k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    raw_items.append((pxl_id, str(v[0]), _normalise_rgba(v[1])))
                elif isinstance(v, Mapping):
                    lbl = str(v.get("name", v.get("label", pxl_id)))
                    clr = v.get("colour", v.get("color"))
                    raw_items.append(
                        (pxl_id, lbl, _normalise_rgba(clr) if clr is not None else None)
                    )
                else:
                    raw_items.append((pxl_id, str(v), None))
    elif isinstance(classes_info, (list, tuple)):
        if not classes_info:
            raise rsgispy.RSGISPyException("classes_info sequence must not be empty.")
        raw_items = _extract_entries_from_class_objs(
            [(None, obj) for obj in classes_info]
        )
    else:
        raise rsgispy.RSGISPyException(
            f"Unsupported classes_info type: {type(classes_info).__name__}."
        )

    present_vals: Optional[set] = (
        _scan_present_values(img_path, band=band, bbox=bbox)
        if (only_present_in_img and img_path is not None)
        else None
    )
    include_set = {int(v) for v in include_vals} if include_vals is not None else None
    exclude_set = {int(v) for v in exclude_vals} if exclude_vals is not None else set()

    resolved: List[LegendEntry] = []
    palette_idx = 0
    for pxl_id, name, inline_rgba in raw_items:
        if ignore_zero and pxl_id == 0:
            continue
        if pxl_id in exclude_set:
            continue
        if include_set is not None and pxl_id not in include_set:
            continue
        if present_vals is not None and pxl_id not in present_vals:
            continue

        if pxl_id in explicit_lut:
            rgba = explicit_lut[pxl_id]
        elif inline_rgba is not None:
            rgba = inline_rgba
        elif pxl_id in raster_cmap:
            rgba = raster_cmap[pxl_id]
        else:
            rgba = matplotlib.colors.to_rgba(
                _DEFAULT_PALETTE[palette_idx % len(_DEFAULT_PALETTE)]
            )
            palette_idx += 1

        lbl_str = pxl_val_fmt.format(val=pxl_id, name=name) if show_pxl_vals else name
        if max_label_width is not None and int(max_label_width) > 0:
            lbl_str = textwrap.fill(lbl_str, width=int(max_label_width))
        resolved.append(LegendEntry(pxl_val=pxl_id, label=lbl_str, colour=rgba))

    if not resolved:
        raise rsgispy.RSGISPyException("No legend classes remain after filtering.")

    sort_key = str(sort_by).strip().lower()
    if sort_key == "id":
        resolved.sort(key=lambda e: e.pxl_val)
    elif sort_key == "name":
        resolved.sort(key=lambda e: e.label.lower())
    elif sort_key != "none":
        raise rsgispy.RSGISPyException(
            f"Unsupported sort_by={sort_by!r}; expected 'id', 'name', or 'none'."
        )
    return resolved


def _max_label_chars(labels: Sequence[str]) -> int:
    """Return longest line length across formatted (possibly wrapped) labels."""
    best = 1
    for lbl in labels:
        for line in lbl.splitlines() or [lbl]:
            best = max(best, len(line))
    return best


def calc_categorical_legend_layout(
    labels: Sequence[str],
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    avail_width_in: float = 6.0,
    avail_height_in: float = 4.0,
    fontsize: float = 10.0,
    prefer_vertical: bool = False,
) -> Tuple[int, int]:
    """
    Compute a balanced ``(nrows, ncols)`` grid layout for categorical legend
    entries, automatically accounting for label lengths and target space.

    :param labels: Sequence of formatted class label strings.
    :param ncols: Optional explicit number of columns.
    :param nrows: Optional explicit number of rows.
    :param avail_width_in: Available width in inches for auto column sizing
                           (Default = ``6.0``).
    :param avail_height_in: Available height in inches for auto sizing
                            (Default = ``4.0``).
    :param fontsize: Label font size in points (Default = ``10.0``).
    :param prefer_vertical: If ``True`` (e.g. side legend axis), favour a single
                            column until items exceed ``avail_height_in``.
    :return: Tuple ``(eff_nrows, eff_ncols)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_items = len(labels)
    if n_items <= 0:
        raise rsgispy.RSGISPyException("labels must not be empty.")

    if ncols is not None and (isinstance(ncols, bool) or int(ncols) < 1):
        raise rsgispy.RSGISPyException(f"ncols must be >= 1, got {ncols!r}.")
    if nrows is not None and (isinstance(nrows, bool) or int(nrows) < 1):
        raise rsgispy.RSGISPyException(f"nrows must be >= 1, got {nrows!r}.")

    if ncols is not None and nrows is not None:
        c_val, r_val = int(ncols), int(nrows)
        if c_val * r_val < n_items:
            raise rsgispy.RSGISPyException(
                f"Specified nrows ({r_val}) * ncols ({c_val}) = {r_val * c_val} "
                f"cannot fit {n_items} classes."
            )
        return r_val, c_val
    if ncols is not None:
        c_val = min(int(ncols), n_items)
        return math.ceil(n_items / c_val), c_val
    if nrows is not None:
        r_val = min(int(nrows), n_items)
        c_val = math.ceil(n_items / r_val)
        return math.ceil(n_items / c_val), c_val

    max_chars = _max_label_chars(labels)
    char_w_in = max(0.055, 0.0085 * float(fontsize))
    swatch_and_gap_in = 0.85
    entry_w_in = swatch_and_gap_in + float(max_chars) * char_w_in
    max_fit_cols = max(1, min(n_items, int(math.floor(avail_width_in / entry_w_in))))

    max_lines = max(len(lbl.splitlines() or [lbl]) for lbl in labels)
    row_h_in = max(0.24, (float(fontsize) / 72.0) * 2.1 * max_lines)

    if prefer_vertical:
        max_fit_rows = max(1, int(math.floor(avail_height_in / row_h_in)))
        if n_items <= max_fit_rows or max_fit_cols == 1:
            return n_items, 1
        min_cols = math.ceil(n_items / max_fit_rows)
        eff_cols = min(max_fit_cols, max(1, min_cols))
        return math.ceil(n_items / eff_cols), eff_cols

    best_cols = 1
    best_score = float("inf")
    for c_cand in range(1, max_fit_cols + 1):
        r_cand = math.ceil(n_items / c_cand)
        empty = r_cand * c_cand - n_items
        score = float(r_cand) + 0.65 * float(empty) + 0.15 * float(c_cand)
        if score < best_score:
            best_score = score
            best_cols = c_cand
    return math.ceil(n_items / best_cols), best_cols


def build_legend_handles(
    entries: Sequence[LegendEntry],
    swatch_shape: str = "rectangle",
    swatch_edgecolor: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
) -> List[Any]:
    """Create Matplotlib Patch or Line2D handles for categorical legend entries."""
    shape = str(swatch_shape).strip().lower()
    if shape == "rectangle":
        return [
            Patch(
                facecolor=e.colour,
                edgecolor=swatch_edgecolor,
                linewidth=swatch_linewidth,
                label=e.label,
            )
            for e in entries
        ]
    if shape in ("square", "circle"):
        marker = "s" if shape == "square" else "o"
        return [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color="none",
                markerfacecolor=e.colour,
                markeredgecolor=swatch_edgecolor,
                markeredgewidth=swatch_linewidth,
                markersize=float(swatch_size),
                label=e.label,
            )
            for e in entries
        ]
    raise rsgispy.RSGISPyException(
        f"Unsupported swatch_shape={swatch_shape!r}; "
        "expected 'rectangle', 'square', or 'circle'."
    )


def _measure_axis_inches(ax: matplotlib.axes.Axes) -> Tuple[float, float]:
    """Estimate current rendered width and height of ``ax`` in inches."""
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    return max(1.0, float(fig_w * pos.width)), max(1.0, float(fig_h * pos.height))


def _style_legend(
    ax: matplotlib.axes.Axes,
    handles: Sequence[Any],
    eff_cols: int,
    loc: str,
    bbox_to_anchor: Optional[Tuple[float, ...]],
    title: Optional[str],
    fontsize: float,
    title_fontsize: Optional[float],
    frameon: bool,
    facecolor: str,
    edgecolor: str,
    framealpha: float,
    fancybox: bool,
    columnspacing: float,
    handletextpad: float,
    labelspacing: float,
    borderpad: float,
) -> matplotlib.legend.Legend:
    """Build and style a Matplotlib Legend object attached to ``ax``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lgd = ax.legend(
        handles=list(handles),
        ncols=int(eff_cols),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        title=title,
        fontsize=fontsize,
        title_fontsize=title_fontsize if title_fontsize is not None else fontsize + 1.0,
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
    if title is not None and lgd.get_title() is not None:
        lgd.get_title().set_fontweight("bold")
    return lgd


def create_companion_legend_axis(
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
) -> Tuple[matplotlib.axes.Axes, matplotlib.legend.Legend]:
    """Create a companion legend axis adjacent to ``ax`` and draw the legend."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    pos_key = str(position).strip().lower()
    if pos_key not in ("right", "left", "bottom", "top"):
        raise rsgispy.RSGISPyException(
            f"position must be 'right', 'left', 'bottom', or 'top', got {position!r}."
        )

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
    is_side = pos_key in ("right", "left")
    avail_w = max(2.2, ax_w * 0.45) if is_side else ax_w
    avail_h = ax_h if is_side else max(1.0, ax_h * 0.4)

    eff_rows, eff_cols = calc_categorical_legend_layout(
        labels=[e.label for e in entries],
        ncols=ncols,
        nrows=nrows,
        avail_width_in=avail_w,
        avail_height_in=avail_h,
        fontsize=fontsize,
        prefer_vertical=is_side,
    )

    if size is None:
        max_chars = _max_label_chars([e.label for e in entries])
        if is_side:
            est_w = eff_cols * (0.85 + max_chars * max(0.055, 0.0085 * fontsize))
            pct = int(np.clip(round((est_w / max(ax_w, 1.0)) * 100.0), 15, 60))
            eff_size = f"{pct}%"
        else:
            est_h = (eff_rows + (1.2 if title else 0.0)) * max(
                0.24, (fontsize / 72.0) * 2.1
            )
            pct = int(np.clip(round((est_h / max(ax_h, 1.0)) * 100.0), 10, 45))
            eff_size = f"{pct}%"
    else:
        eff_size = size

    divider = make_axes_locatable(ax)
    leg_ax = divider.append_axes(pos_key, size=eff_size, pad=pad)
    leg_ax.set_axis_off()

    loc_map = {
        "right": ("center left", (0.0, 0.5)),
        "left": ("center right", (1.0, 0.5)),
        "bottom": ("upper center", (0.5, 1.0)),
        "top": ("lower center", (0.5, 0.0)),
    }
    loc_str, anchor = loc_map[pos_key]
    handles = build_legend_handles(
        entries,
        swatch_shape=swatch_shape,
        swatch_edgecolor=swatch_edgecolor,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
    )
    lgd = _style_legend(
        ax=leg_ax,
        handles=handles,
        eff_cols=eff_cols,
        loc=loc_str,
        bbox_to_anchor=anchor,
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
    return leg_ax, lgd


def save_standalone_legend(
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
    """Render a standalone categorical legend image and save to ``out_img_file``."""
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
    eff_rows, eff_cols = calc_categorical_legend_layout(
        labels=[e.label for e in entries],
        ncols=ncols,
        nrows=nrows,
        avail_width_in=float(fig_width),
        avail_height_in=max(2.0, float(fig_width) * 0.8),
        fontsize=fontsize,
        prefer_vertical=False,
    )
    max_chars = _max_label_chars([e.label for e in entries])
    est_w = max(2.0, eff_cols * (0.9 + max_chars * max(0.055, 0.0085 * fontsize)))
    est_h = max(1.2, (eff_rows + (1.5 if title else 0.5)) * 0.35)

    fig, leg_ax = plt.subplots(figsize=(est_w, est_h))
    try:
        leg_ax.set_axis_off()
        handles = build_legend_handles(
            entries,
            swatch_shape=swatch_shape,
            swatch_edgecolor=swatch_edgecolor,
            swatch_linewidth=swatch_linewidth,
            swatch_size=swatch_size,
        )
        _style_legend(
            ax=leg_ax,
            handles=handles,
            eff_cols=eff_cols,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
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
        out_path = ensure_parent_dir(out_img_file)
        fig.savefig(
            out_path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.15,
            transparent=transparent,
        )
    finally:
        plt.close(fig)
    return eff_rows, eff_cols
