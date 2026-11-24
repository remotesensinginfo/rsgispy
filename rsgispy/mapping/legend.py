"""
The mapping.legend module provides functions for computing layouts, creating
swatch handles, rendering categorical/classification legends on Matplotlib axes,
and exporting standalone legend image files.
"""

import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.legend

from rsgispy.mapping._common import resolve_mpl_location
from rsgispy.mapping._legend import (
    _measure_axis_inches,
    _style_legend,
    build_legend_handles,
    calc_categorical_legend_layout,
    resolve_legend_entries,
    save_standalone_legend,
)

__all__ = [
    "calc_categorical_legend_layout",
    "create_categorical_legend_patches",
    "add_categorical_legend",
    "add_categorical_legend_axis",
    "save_categorical_legend_img",
]


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
    swatch_edge_colour: str = "#333333",
    swatch_linewidth: float = 0.6,
    swatch_size: float = 9.0,
) -> List[Any]:
    """
    Create styled Matplotlib legend handle objects (``Patch`` or ``Line2D``)
    for categorical or classification classes.

    :param classes_info: ``{pixel_val: class_name}`` dict,
                         ``{pixel_val: (name, colour)}`` dict,
                         ``{class_name: ClassInfoObj}`` dict,
                         :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                         or JSON path.
    :param input_img: Optional thematic raster image path for reading embedded
                      colour tables or filtering present pixel values.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent ``(MinX, MaxX, MinY, MaxY)`` used when
                 ``only_present_in_img=True``.
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional allowlist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes whose pixel value appears in
                                ``input_img`` (Default = ``False``).
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in labels (Default = ``False``).
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: Swatch style: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edge_colour: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :return: List of Matplotlib legend handle objects.

    .. code:: python

        import rsgispy.mapping.legend as legend

        handles = legend.create_categorical_legend_patches(
            classes_info={1: "Forest", 2: "Grassland", 3: "Water"},
            colour_lut={1: "#228b22", 2: "#9acd32", 3: "#1f77b4"},
            show_pxl_vals=True,
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
    return build_legend_handles(
        entries=entries,
        swatch_shape=swatch_shape,
        swatch_edge_colour=swatch_edge_colour,
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
    swatch_edge_colour: str = "#333333",
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
) -> matplotlib.legend.Legend:
    """
    Add a professionally formatted categorical legend onto an existing Matplotlib
    axis ``ax``.

    By default (``overlay=True``), the legend is placed on the map axis at
    ``loc`` (e.g. ``'lower right'``). Set ``overlay=False`` when ``ax`` is a
    dedicated subplot/panel reserved exclusively for the legend (ticks and frame
    on ``ax`` will be hidden and the legend centred cleanly).

    When both ``ncols`` and ``nrows`` are ``None`` (auto mode), column and row
    counts are chosen automatically from the axis size and class label lengths.

    Note that styling and layout keyword arguments (``ncols``, ``loc``,
    ``bbox_to_anchor``, ``fontsize``, ``title_fontsize``, ``frameon``,
    ``facecolor``, ``edgecolor``, ``framealpha``, ``fancybox``,
    ``columnspacing``, ``handletextpad``, ``labelspacing``, ``borderpad``) are
    passed directly through to :meth:`matplotlib.axes.Axes.legend` and
    therefore retain Matplotlib's parameter names.

    :param ax: Target Matplotlib axis.
    :param classes_info: ``{pixel_val: class_name}`` dict,
                         ``{pixel_val: (name, colour)}`` dict,
                         ``{class_name: ClassInfoObj}`` dict,
                         :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                         or JSON file path.
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
    :param include_vals: Optional allowlist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edge_colour: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Matplotlib passthrough; Default = ``True``).
    :param facecolor: Legend background colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'white'``).
    :param edgecolor: Legend border colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Matplotlib passthrough; Default = ``0.92``).
    :param fancybox: Rounded legend corners (Matplotlib passthrough; Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Matplotlib passthrough;
                          Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Matplotlib passthrough; Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Matplotlib passthrough;
                         Default = ``0.45``).
    :param borderpad: Internal frame padding (Matplotlib passthrough; Default = ``0.6``).
    :return: Created :class:`matplotlib.legend.Legend` instance.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.legend as legend

        fig, ax = plt.subplots()
        legend.add_categorical_legend(
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
        swatch_edge_colour=swatch_edge_colour,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
    )
    if not overlay:
        ax.set_axis_off()
        eff_loc = "center"
        eff_anchor: Optional[Tuple[float, ...]] = (0.5, 0.5)
    else:
        eff_loc = resolve_mpl_location(loc)
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
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    title: Optional[str] = None,
    loc: str = "center",
    bbox_to_anchor: Optional[Tuple[float, ...]] = None,
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
    swatch_edge_colour: str = "#333333",
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
) -> matplotlib.legend.Legend:
    """
    Render a formatted categorical legend directly into a dedicated Matplotlib
    axis ``ax`` (hiding axis ticks, labels, and frame).

    This allows the caller to position the legend axis freely anywhere within
    a multi-axis or complex figure (e.g., via ``plt.subplots``, ``GridSpec``,
    or ``fig.add_axes``) without creating an extra axis. When ``ncols`` and
    ``nrows`` are omitted, the column/row grid adapts automatically to the
    dimensions of ``ax`` and class label lengths (favouring a vertical layout
    when ``ax`` is taller than it is wide).

    Note that styling and layout keyword arguments (``ncols``, ``loc``,
    ``bbox_to_anchor``, ``fontsize``, ``title_fontsize``, ``frameon``,
    ``facecolor``, ``edgecolor``, ``framealpha``, ``fancybox``,
    ``columnspacing``, ``handletextpad``, ``labelspacing``, ``borderpad``) are
    passed directly through to :meth:`matplotlib.axes.Axes.legend` and
    therefore retain Matplotlib's parameter names.

    :param ax: Dedicated Matplotlib axis into which the legend is drawn.
    :param classes_info: Class dictionary,
                         :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                         or JSON path.
    :param input_img: Optional thematic raster for extracting embedded colours
                      or filtering visible classes.
    :param band: 1-based band index for reading raster colour table (Default = ``1``).
    :param colour_lut: Optional ``{pixel_val: colour}`` override mapping.
    :param bbox: Optional map extent used when ``only_present_in_img=True``.
    :param ncols: Explicit legend column count (Default = ``None``, auto).
    :param nrows: Explicit legend row count (Default = ``None``, auto).
    :param title: Optional bold legend title.
    :param loc: Legend alignment inside ``ax`` (Default = ``'center'``).
    :param bbox_to_anchor: Optional Matplotlib ``bbox_to_anchor`` coordinates.
    :param fontsize: Class label font size in points (Default = ``9.5``).
    :param title_fontsize: Optional title font size (Default = ``fontsize + 1``).
    :param ignore_zero: Exclude background pixel value ``0`` (Default = ``True``).
    :param include_vals: Optional allowlist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edge_colour: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Matplotlib passthrough; Default = ``True``).
    :param facecolor: Legend background colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'white'``).
    :param edgecolor: Legend border colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Matplotlib passthrough; Default = ``0.95``).
    :param fancybox: Rounded legend corners (Matplotlib passthrough; Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Matplotlib passthrough;
                          Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Matplotlib passthrough; Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Matplotlib passthrough;
                         Default = ``0.45``).
    :param borderpad: Internal frame padding (Matplotlib passthrough; Default = ``0.6``).
    :return: Created :class:`matplotlib.legend.Legend` instance.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.legend as legend
        import rsgispy.mapping.raster as map_raster

        fig, (ax_map, ax_leg) = plt.subplots(
            1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [4, 1]}
        )
        map_raster.create_thematic_raster_map(ax=ax_map, input_img="landcover.kea")
        lgd = legend.add_categorical_legend_axis(
            ax=ax_leg,
            classes_info={1: "Forest", 2: "Grassland", 3: "Water"},
            input_img="landcover.kea",
            title="Classes",
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
    _, eff_cols = calc_categorical_legend_layout(
        labels=[e.label for e in entries],
        ncols=ncols,
        nrows=nrows,
        avail_width_in=ax_w,
        avail_height_in=ax_h,
        fontsize=fontsize,
        prefer_vertical=ax_h > ax_w * 1.25,
    )
    handles = build_legend_handles(
        entries=entries,
        swatch_shape=swatch_shape,
        swatch_edge_colour=swatch_edge_colour,
        swatch_linewidth=swatch_linewidth,
        swatch_size=swatch_size,
    )
    ax.set_axis_off()
    eff_loc = resolve_mpl_location(loc)
    return _style_legend(
        ax=ax,
        handles=handles,
        eff_cols=eff_cols,
        loc=eff_loc,
        bbox_to_anchor=bbox_to_anchor,
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


def save_categorical_legend_img(
    output_img: Union[str, os.PathLike],
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
    swatch_edge_colour: str = "#333333",
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

    Note that styling keyword arguments (``ncols``, ``fontsize``,
    ``title_fontsize``, ``frameon``, ``facecolor``, ``edgecolor``,
    ``framealpha``, ``fancybox``, ``columnspacing``, ``handletextpad``,
    ``labelspacing``, ``borderpad``) are passed directly through to
    :meth:`matplotlib.axes.Axes.legend` and therefore retain Matplotlib's
    parameter names.

    :param output_img: Output image file path (e.g. ``'legend.png'``).
    :param classes_info: Class dictionary,
                         :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                         or JSON path.
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
    :param include_vals: Optional allowlist of pixel values to keep.
    :param exclude_vals: Optional pixel values to exclude.
    :param only_present_in_img: Show only classes present in ``input_img``.
    :param sort_by: Ordering mode: ``'id'``, ``'name'``, or ``'none'``.
    :param show_pxl_vals: Include pixel values in legend labels.
    :param pxl_val_fmt: Format template when ``show_pxl_vals=True``.
    :param max_label_width: Optional line-wrap width for long class names.
    :param swatch_shape: ``'rectangle'``, ``'square'``, or ``'circle'``.
    :param swatch_edge_colour: Colour of swatch border (Default = ``'#333333'``).
    :param swatch_linewidth: Line width of swatch border (Default = ``0.6``).
    :param swatch_size: Marker size for ``'square'`` / ``'circle'`` swatches.
    :param frameon: Draw legend frame box (Matplotlib passthrough; Default = ``True``).
    :param facecolor: Legend background colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'white'``).
    :param edgecolor: Legend border colour (Matplotlib ``ax.legend`` passthrough;
                      Default = ``'#CCCCCC'``).
    :param framealpha: Legend background opacity (Matplotlib passthrough; Default = ``1.0``).
    :param fancybox: Rounded legend corners (Matplotlib passthrough; Default = ``True``).
    :param columnspacing: Horizontal gap between columns (Matplotlib passthrough;
                          Default = ``1.2``).
    :param handletextpad: Gap between swatch and text (Matplotlib passthrough; Default = ``0.6``).
    :param labelspacing: Vertical spacing between entries (Matplotlib passthrough;
                         Default = ``0.45``).
    :param borderpad: Internal frame padding (Matplotlib passthrough; Default = ``0.6``).
    :param dpi: Output figure resolution in DPI (Default = ``300``).
    :param transparent: If ``True``, export with a transparent background.
    :return: Tuple ``(nrows, ncols)`` used in the saved legend layout.

    .. code:: python

        import rsgispy.mapping.legend as legend

        nrows, ncols = legend.save_categorical_legend_img(
            output_img="landcover_legend.png",
            classes_info={1: "Forest", 2: "Grassland", 3: "Urban", 4: "Water"},
            colour_lut={1: "#228b22", 2: "#9acd32", 3: "#d62728", 4: "#1f77b4"},
            title="Land Cover Classes",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    return save_standalone_legend(
        output_img=output_img,
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
        swatch_edge_colour=swatch_edge_colour,
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
