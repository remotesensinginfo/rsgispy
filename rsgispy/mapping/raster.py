"""
The mapping.raster module provides functions for rendering continuous,
colour-mapped, and thematic raster layers onto Matplotlib axes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset
from rsgispy.mapping._legend import create_companion_legend_axis
from rsgispy.mapping.elements import (
    NORTH_ARROW_STYLE_TRIANGLE,
    SCALEBAR_STYLE_SOLID,
    _finalize_map_axis,
)
from rsgispy.mapping.legend import (
    add_categorical_legend,
    add_categorical_legend_axis,
)
import rsgispy.tools.plotting

__all__ = [
    "create_raster_img_map",
    "create_thematic_raster_map",
    "create_raster_cmap_img_map",
]


def _read_raster_map_meta(
    input_img: str,
    band: Optional[int] = None,
) -> Tuple[Tuple[float, float, float, float], Optional[int], Optional[float]]:
    """Open ``input_img`` once and return ``(bbox, epsg_code, nodata)``."""
    with rasterio.open(input_img, "r") as src:
        img_bbox = bbox_from_dataset(src)
        epsg_code: Optional[int] = None
        if src.crs is not None:
            epsg_code = src.crs.to_epsg()
        if band is not None and 1 <= band <= src.count:
            nodata_val = src.nodatavals[band - 1]
        else:
            nodata_val = src.nodata
    return img_bbox, epsg_code, nodata_val


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
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        map_raster.create_raster_img_map(
            ax=ax,
            input_img="sentinel2.tif",
            img_bands=[4, 3, 2],
            img_stch=rsgispy.IMG_STRETCH_STDEV,
            title_str="RGB Composite",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    img_bbox, img_epsg, src_nodata = _read_raster_map_meta(input_img)
    if bbox is None:
        bbox = img_bbox
    if img_no_data_val is None:
        img_no_data_val = src_nodata

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=img_bands, bbox=bbox
    )

    clr_map = None
    if img_stch == rsgispy.IMG_STRETCH_NONE:
        img_data_strch = img_data
        if len(img_bands) == 1:
            clr_map = plt.get_cmap("gray")
    elif img_stch == rsgispy.IMG_STRETCH_USER:
        if stch_min_max_vals is None:
            raise rsgispy.RSGISPyException(
                "IMG_STRETCH_USER requires stch_min_max_vals."
            )
        img_data_strch = rsgispy.tools.plotting.manual_stretch_np_arr(
            img_data,
            stch_min_max_vals,
            no_data_val=img_no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_LINEAR:
        img_data_strch = rsgispy.tools.plotting.linear_stretch_np_arr(
            img_data, no_data_val=img_no_data_val, no_data_clr=no_data_out_clr
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
            img_data, no_data_val=img_no_data_val, no_data_clr=no_data_out_clr
        )
    else:
        raise rsgispy.RSGISPyException(
            "Do not recognise stretch method. Must be rsgispy.IMG_STRETCH_*"
        )

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
        epsg_code=img_epsg,
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
    legend_ax: Optional[matplotlib.axes.Axes] = None,
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
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :param show_legend: If ``True``, automatically create and attach a
                        categorical classification legend (Default = ``False``).
    :param classes_info: Optional class label/colour definitions (dict,
                         ``ClassificationConfig``, or JSON path) passed to the
                         legend builder (Default = ``None``).
    :param legend_position: Companion axis side when ``legend_overlay=False`` and
                            ``legend_ax`` is ``None`` (``'right'``, ``'left'``,
                            ``'bottom'``, or ``'top'``; Default = ``'right'``).
    :param legend_title: Optional legend title (Default = ``None``).
    :param legend_ncols: Optional explicit legend column count (Default = auto).
    :param legend_nrows: Optional explicit legend row count (Default = auto).
    :param legend_overlay: If ``True``, overlay the legend inside ``ax`` at
                           ``legend_loc`` instead of appending a companion axis
                           (Default = ``False``).
    :param legend_loc: Overlay legend placement string when
                       ``legend_overlay=True`` (Default = ``'lower right'``).
    :param legend_kwargs: Optional extra keyword arguments forwarded to
                          :func:`~rsgispy.mapping.legend.add_categorical_legend`,
                          :func:`~rsgispy.mapping.legend.add_categorical_legend_axis`,
                          or the companion legend axis builder.
    :param legend_ax: Optional pre-existing dedicated Matplotlib axis into which
                      the legend is written via
                      :func:`~rsgispy.mapping.legend.add_categorical_legend_axis`
                      when ``show_legend=True`` (Default = ``None``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        map_raster.create_thematic_raster_map(
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

    img_bbox, img_epsg, _ = _read_raster_map_meta(input_imgs[0])
    if bbox is None:
        bbox = img_bbox

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
        epsg_code=img_epsg,
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
        if legend_ax is not None:
            add_categorical_legend_axis(
                ax=legend_ax,
                classes_info=classes_info,
                input_img=input_imgs[0],
                bbox=bbox,
                ncols=legend_ncols,
                nrows=legend_nrows,
                title=legend_title,
                **extra_kw,
            )
        elif legend_overlay:
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
            create_companion_legend_axis(
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
    :param scale_bar_style: Scale bar style constant
                            (Default = :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'``, or ``None`` for auto).
    :param show_north_arrow: Whether to add a north arrow (Default = False).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement string (Default = ``'upper left'``).
    :return: Tuple ``(c_cmap, c_norm)`` for creating a Matplotlib colourbar.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        cmap, norm = map_raster.create_raster_cmap_img_map(
            ax=ax,
            input_img="ndvi.tif",
            img_band=1,
            cmap_name="YlGn",
            norm_img_vals=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    img_bbox, img_epsg, band_nodata = _read_raster_map_meta(
        input_img, band=img_band
    )
    if bbox is None:
        bbox = img_bbox

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=[img_band], bbox=bbox
    )
    if band_nodata is not None and not np.isnan(band_nodata):
        img_data = img_data.astype(np.float64, copy=True)
        img_data[img_data == band_nodata] = np.nan

    c_cmap = plt.get_cmap(cmap_name).copy()
    if vals_under_white:
        c_cmap = c_cmap.with_extremes(under="white")

    c_norm: Optional[matplotlib.colors.Normalize] = None
    if norm_img_vals:
        finite_mask = np.isfinite(img_data)
        if norm_vmin is None:
            norm_vmin = (
                float(np.min(img_data[finite_mask]))
                if np.any(finite_mask)
                else 0.0
            )
        if norm_vmax is None:
            norm_vmax = (
                float(np.max(img_data[finite_mask]))
                if np.any(finite_mask)
                else 1.0
            )

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
        epsg_code=img_epsg,
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
