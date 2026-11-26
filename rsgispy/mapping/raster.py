"""
The mapping.raster module provides functions for rendering continuous,
colour-mapped, and thematic raster layers onto Matplotlib axes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.colors
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset
from rsgispy.imageutils.colourtable import get_thematic_classes
from rsgispy.mapping._common import add_map_colour_bar
from rsgispy.mapping._legend import create_companion_legend_axis
from rsgispy.mapping.elements import (
    NORTH_ARROW_STYLE_TRIANGLE,
    SCALEBAR_STYLE_SOLID,
    _finalise_map_axis,
)
from rsgispy.mapping.legend import (
    add_categorical_legend,
    add_categorical_legend_axis,
)
import rsgispy.tools.plotting
from rsgispy.tools.plotting import use_plot_style

__all__ = [
    "create_raster_img_map",
    "create_thematic_raster_map",
    "create_raster_cmap_img_map",
    "add_hillshade",
    "create_shaded_cmap_img_map",
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


@use_plot_style(plot_type="map")
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
    no_data_val: Optional[float] = None,
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
    interpolation: Optional[str] = None,
    interp_stage: str = "data",
) -> matplotlib.image.AxesImage:
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
    :param no_data_val: Optional no-data value; read from image header if ``None``.
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
    :param interpolation: Optional Matplotlib ``imshow`` interpolation
                          (e.g. ``'nearest'``; Default = ``None``, the
                          Matplotlib default, ``'auto'``).
    :param interp_stage: Stage at which the image is resampled when drawn
                         larger or smaller than its pixels: ``'data'``
                         (Default: the values are resampled before colour
                         mapping, which keeps one-pixel features such as
                         stream channels visible on a small map), ``'rgba'``
                         or ``'auto'`` (see ``AxesImage.set_interpolation_stage``).
    :return: The image artist (``matplotlib.image.AxesImage``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        img = map_raster.create_raster_img_map(
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
    if no_data_val is None:
        no_data_val = src_nodata

    img_data, img_coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
        input_img, bands=img_bands, bbox=bbox
    )

    clr_map = None
    if len(img_bands) == 1:
        clr_map = plt.get_cmap("gray")
    if img_stch == rsgispy.IMG_STRETCH_NONE:
        img_data_strch = img_data
    elif img_stch == rsgispy.IMG_STRETCH_USER:
        if stch_min_max_vals is None:
            raise rsgispy.RSGISPyException(
                "IMG_STRETCH_USER requires stch_min_max_vals."
            )
        img_data_strch = rsgispy.tools.plotting.manual_stretch_np_arr(
            img_data,
            stch_min_max_vals,
            no_data_val=no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_LINEAR:
        img_data_strch = rsgispy.tools.plotting.linear_stretch_np_arr(
            img_data, no_data_val=no_data_val, no_data_clr=no_data_out_clr
        )
    elif img_stch == rsgispy.IMG_STRETCH_STDEV:
        img_data_strch = rsgispy.tools.plotting.stdev_stretch_np_arr(
            img_data,
            n_stdevs=stch_n_stdevs,
            no_data_val=no_data_val,
            no_data_clr=no_data_out_clr,
        )
    elif img_stch == rsgispy.IMG_STRETCH_CUMULATIVE:
        img_data_strch = rsgispy.tools.plotting.cumulative_stretch_np_arr(
            img_data, no_data_val=no_data_val, no_data_clr=no_data_out_clr
        )
    else:
        raise rsgispy.RSGISPyException(
            "Do not recognise stretch method. Must be rsgispy.IMG_STRETCH_*"
        )

    img_artist = ax.imshow(
        img_data_strch,
        cmap=clr_map,
        extent=img_coords,
        alpha=img_alpha,
        zorder=plot_zorder,
        interpolation=interpolation,
        interpolation_stage=interp_stage,
    )
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalise_map_axis(
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

    return img_artist


@use_plot_style(plot_type="map")
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
    interpolation: Optional[str] = None,
    interp_stage: str = "data",
) -> Union[matplotlib.image.AxesImage, List[matplotlib.image.AxesImage]]:
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
                         legend builder. If ``None``, the class names and
                         colours stored in the first image by
                         :func:`~rsgispy.imageutils.colourtable.set_thematic_classes`
                         are used (see
                         :func:`~rsgispy.imageutils.colourtable.get_thematic_classes`),
                         or, when none are stored, every colour table entry
                         as ``Class <value>`` (Default = ``None``).
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
    :param interpolation: Optional Matplotlib ``imshow`` interpolation
                          (e.g. ``'nearest'``; Default = ``None``, the
                          Matplotlib default, ``'auto'``).
    :param interp_stage: Stage at which the image is resampled when drawn
                         larger or smaller than its pixels: ``'data'``
                         (Default: the values are resampled before colour
                         mapping, which keeps one-pixel features such as
                         stream channels visible on a small map), ``'rgba'``
                         or ``'auto'`` (see ``AxesImage.set_interpolation_stage``).
    :return: The image artist (``matplotlib.image.AxesImage``) when
             ``input_img`` is one image, or a list of image artists (one per
             image, in drawing order) when it is a list. The legend, when
             drawn, is available from the axis (``ax.get_legend()``) or
             ``legend_ax``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        # The legend uses the names stored by set_thematic_classes.
        map_raster.create_thematic_raster_map(
            ax=ax,
            input_img="landcover.tif",
            show_legend=True,
        )

        fig, ax = plt.subplots()
        map_raster.create_thematic_raster_map(
            ax=ax,
            input_img="landcover.tif",
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

    img_artists = []
    for img_data_arr, img_coords, plot_zorder in zip(
        img_data_arr_scns, img_coords_scns, zorder_list
    ):
        img_artists.append(
            ax.imshow(
                img_data_arr,
                extent=img_coords,
                alpha=img_alpha,
                zorder=plot_zorder,
                interpolation=interpolation,
                interpolation_stage=interp_stage,
            )
        )

    _finalise_map_axis(
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
        if classes_info is None:
            # Use the class names stored by set_thematic_classes, if any.
            classes_info = get_thematic_classes(input_imgs[0]) or None
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

    if isinstance(input_img, str):
        return img_artists[0]
    return img_artists


@use_plot_style(plot_type="map")
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
    show_colour_bar: bool = False,
    cbar_label: Optional[str] = None,
    img_alpha: float = 1.0,
    interpolation: Optional[str] = None,
    interp_stage: str = "data",
) -> matplotlib.image.AxesImage:
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
    :param show_colour_bar: Draw a colour bar beside the map (Default = False).
    :param cbar_label: Optional colour bar label, e.g. the variable and its
                       units (Default = None).
    :param img_alpha: Display alpha transparency (Default = 1.0).
    :param interpolation: Optional Matplotlib ``imshow`` interpolation
                          (e.g. ``'nearest'``; Default = ``None``, the
                          Matplotlib default, ``'auto'``).
    :param interp_stage: Stage at which the image is resampled when drawn
                         larger or smaller than its pixels: ``'data'``
                         (Default: the values are resampled before colour
                         mapping, which keeps one-pixel features such as
                         stream channels visible on a small map), ``'rgba'``
                         or ``'auto'`` (see ``AxesImage.set_interpolation_stage``).
    :return: The image artist (``matplotlib.image.AxesImage``), e.g. for
             ``fig.colorbar(img)``; with ``show_colour_bar=True`` the colour
             bar is ``img.colorbar``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        img = map_raster.create_raster_cmap_img_map(
            ax=ax,
            input_img="ndvi.tif",
            img_band=1,
            cmap_name="YlGn",
            norm_img_vals=True,
            show_colour_bar=True,
            cbar_label="NDVI",
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

    img_artist = ax.imshow(
        img_data,
        extent=img_coords,
        cmap=c_cmap,
        norm=c_norm,
        alpha=img_alpha,
        zorder=plot_zorder,
        interpolation=interpolation,
        interpolation_stage=interp_stage,
    )
    ax.set_xlim([img_coords[0], img_coords[1]])
    ax.set_ylim([img_coords[2], img_coords[3]])
    ax.ticklabel_format(useOffset=False, style="plain")

    _finalise_map_axis(
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

    if show_colour_bar:
        add_map_colour_bar(ax, img_artist, cbar_label=cbar_label)

    return img_artist


def _check_one_hillshade_input(
    hshade_img: Optional[str], dem_img: Optional[str]
) -> None:
    if (hshade_img is None) == (dem_img is None):
        raise rsgispy.RSGISPyException(
            "Give one of hshade_img (a hillshade image) or dem_img (a DEM)."
        )


def _calc_hillshade_for_bbox(
    dem_img: str,
    bbox: Optional[Tuple[float, float, float, float]],
    azimuth: float,
    zenith: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[float]]:
    """
    Hillshade of the part of ``dem_img`` in ``bbox``, made in memory with
    :func:`~rsgispy.elevation.illumination.hillshade_node`. The DEM is read
    with a one-pixel margin (inside the image), so the values equal those of
    :func:`~rsgispy.elevation.illumination.hillshade` on the whole DEM.
    Returns ``(hillshade, extent, nodata)``.
    """
    # pylint: disable=import-outside-toplevel,too-many-locals
    from rsgispy.elevation.illumination import hillshade_node
    from rsgispy.imageprocess import (
        ArraySource,
        MemorySink,
        TileGraph,
        ensure_file_exists,
        infer_node_meta,
    )

    ensure_file_exists(dem_img, "Input DEM image")
    with rasterio.open(dem_img, "r") as src:
        # pylint: disable=protected-access
        win, coords = rsgispy.tools.plotting._get_subset_window_and_coords(src, bbox)
        col0 = max(0, int(win.col_off) - 1)
        row0 = max(0, int(win.row_off) - 1)
        col1 = min(src.width, int(win.col_off + win.width) + 1)
        row1 = min(src.height, int(win.row_off + win.height) + 1)
        read_win = Window(col0, row0, col1 - col0, row1 - row0)
        dem_arr = src.read(1, window=read_win).astype(np.float64)
        dem_trans = src.window_transform(read_win)
        dem_nodata = src.nodatavals[0]

    dem_src = ArraySource(dem_arr, transform=dem_trans, nodata=dem_nodata)
    hs_node = hillshade_node(dem_src, azimuth, zenith)
    hs_arr = TileGraph([MemorySink(hs_node, name="hshade")]).run(use_pbar=False)[
        "hshade"
    ][0]
    r_off = int(win.row_off) - row0
    c_off = int(win.col_off) - col0
    hs_arr = hs_arr[r_off: r_off + int(win.height), c_off: c_off + int(win.width)]
    return hs_arr, coords, infer_node_meta(hs_node).nodata


@use_plot_style(plot_type="map")
def add_hillshade(
    ax: matplotlib.axes.Axes,
    hshade_img: Optional[str] = None,
    *,
    dem_img: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    alpha: float = 0.4,
    azimuth: float = 315.0,
    zenith: float = 45.0,
    zorder: float = 2.0,
    interpolation: Optional[str] = None,
    interp_stage: str = "data",
) -> matplotlib.image.AxesImage:
    """
    Draw a semi-transparent hillshade over what is already on ``ax``, e.g. to
    show the relief under a colour-mapped or thematic raster map.

    Give either a hillshade image (``hshade_img``, e.g. from
    :func:`~rsgispy.elevation.illumination.hillshade`) or a DEM
    (``dem_img``), from which the hillshade of ``bbox`` is made in memory (no
    file is written; the values equal those of
    :func:`~rsgispy.elevation.illumination.hillshade` with the same
    ``azimuth`` and ``zenith``). The hillshade is drawn in grey, stretched
    linearly between its minimum and maximum; no-data pixels are transparent.

    :param ax: Matplotlib axis to draw on.
    :param hshade_img: Optional path to a single-band hillshade image.
    :param dem_img: Optional path to a DEM, used when ``hshade_img`` is
                    ``None``.
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` (Default =
                 the whole image).
    :param alpha: Transparency of the hillshade (Default = ``0.4``).
    :param azimuth: Solar azimuth in degrees of a hillshade made from
                    ``dem_img`` (Default = ``315.0``).
    :param zenith: Solar zenith in degrees of a hillshade made from
                   ``dem_img`` (Default = ``45.0``).
    :param zorder: Drawing z-order, above the layer to shade (Default =
                   ``2.0``; the raster map functions draw at ``1.0``).
    :param interpolation: Optional Matplotlib ``imshow`` interpolation
                          (Default = ``None``).
    :param interp_stage: Stage at which the image is resampled (Default =
                         ``'data'``; see :func:`create_raster_cmap_img_map`).
    :return: The hillshade image artist (``matplotlib.image.AxesImage``).

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots()
        map_raster.create_thematic_raster_map(ax, "landcover.tif")
        # The hillshade is made in memory from the DEM.
        map_raster.add_hillshade(ax, dem_img="dem.tif", alpha=0.35)
    """
    # pylint: disable=too-many-arguments,too-many-locals
    _check_one_hillshade_input(hshade_img, dem_img)
    if hshade_img is not None:
        hs_arr, coords = rsgispy.tools.plotting.get_raster_mpl_imshow(
            hshade_img, bands=[1], bbox=bbox
        )
        _, _, hs_nodata = _read_raster_map_meta(hshade_img, band=1)
    else:
        hs_arr, coords, hs_nodata = _calc_hillshade_for_bbox(
            dem_img, bbox, azimuth, zenith
        )

    hs_arr = np.asarray(hs_arr, dtype=np.float64)
    if hs_nodata is not None and not np.isnan(hs_nodata):
        hs_arr = np.where(hs_arr == hs_nodata, np.nan, hs_arr)
    finite = np.isfinite(hs_arr)
    vmin = float(np.min(hs_arr[finite])) if np.any(finite) else 0.0
    vmax = float(np.max(hs_arr[finite])) if np.any(finite) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    hs_cmap = plt.get_cmap("gray").with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    return ax.imshow(
        hs_arr,
        extent=coords,
        cmap=hs_cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        zorder=zorder,
        interpolation=interpolation,
        interpolation_stage=interp_stage,
    )


@use_plot_style(plot_type="map")
def create_shaded_cmap_img_map(
    ax: matplotlib.axes.Axes,
    input_img: str,
    hshade_img: Optional[str] = None,
    *,
    dem_img: Optional[str] = None,
    img_band: int = 1,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_log_norm: bool = False,
    vals_under_white: bool = False,
    hshade_alpha: float = 0.4,
    azimuth: float = 315.0,
    zenith: float = 45.0,
    show_colour_bar: bool = True,
    cbar_label: Optional[str] = None,
    title_str: Optional[str] = None,
    show_scale_bar: bool = True,
    show_north_arrow: bool = True,
    show_map_axis: bool = False,
    use_grid: bool = False,
    scale_bar_loc: str = "upper right",
    scale_bar_style: str = SCALEBAR_STYLE_SOLID,
    scale_bar_units: Optional[str] = None,
    north_arrow_style: str = NORTH_ARROW_STYLE_TRIANGLE,
    north_arrow_loc: str = "upper left",
    img_alpha: float = 1.0,
    interpolation: Optional[str] = None,
    interp_stage: str = "data",
) -> matplotlib.image.AxesImage:
    """
    Draw a single-band raster with a colour map, shaded by a hillshade so the
    relief shows through (e.g. flow accumulation, a wetness index or flood
    depth), with an optional colour bar, scale bar and north arrow.

    The raster is drawn with :func:`create_raster_cmap_img_map` (normalised
    between ``vmin`` and ``vmax``) and the hillshade over it with
    :func:`add_hillshade`, from ``hshade_img`` or made in memory from
    ``dem_img``. The images are resampled on their values
    (``interp_stage='data'``), so one-pixel features such as stream channels
    stay visible when a large raster is drawn small.

    :param ax: Matplotlib axis to draw on.
    :param input_img: Path to the raster to colour map.
    :param hshade_img: Optional path to a hillshade image.
    :param dem_img: Optional path to a DEM from which the hillshade is made
                    (when ``hshade_img`` is ``None``).
    :param img_band: 1-based band of ``input_img`` (Default = ``1``).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` (Default =
                 the whole of ``input_img``).
    :param cmap_name: Matplotlib colour map name (Default = ``'viridis'``).
    :param vmin: Minimum of the colour map (Default = the minimum value).
    :param vmax: Maximum of the colour map (Default = the maximum value).
    :param use_log_norm: Use a logarithmic colour scale, e.g. for flow
                         accumulation (Default = ``False``).
    :param vals_under_white: Draw values below ``vmin`` in white (Default =
                             ``False``).
    :param hshade_alpha: Transparency of the hillshade (Default = ``0.4``).
    :param azimuth: Solar azimuth of a hillshade made from ``dem_img``
                    (Default = ``315.0``).
    :param zenith: Solar zenith of a hillshade made from ``dem_img``
                   (Default = ``45.0``).
    :param show_colour_bar: Draw a colour bar beside the map (Default =
                            ``True``).
    :param cbar_label: Optional colour bar label, e.g. the variable and its
                       units.
    :param title_str: Optional map title.
    :param show_scale_bar: Add a scale bar (Default = ``True``).
    :param show_north_arrow: Add a north arrow (Default = ``True``).
    :param show_map_axis: Show the axis ticks and frame (Default = ``False``).
    :param use_grid: Draw grid lines (Default = ``False``).
    :param scale_bar_loc: Scale bar placement (Default = ``'upper right'``).
    :param scale_bar_style: Scale bar style constant (Default =
                            :data:`~rsgispy.mapping.elements.SCALEBAR_STYLE_SOLID`).
    :param scale_bar_units: Scale bar units (``'m'``, ``'km'`` or ``None``
                            for automatic).
    :param north_arrow_style: North arrow style constant (Default =
                              :data:`~rsgispy.mapping.elements.NORTH_ARROW_STYLE_TRIANGLE`).
    :param north_arrow_loc: North arrow placement (Default = ``'upper left'``).
    :param img_alpha: Transparency of the colour-mapped raster (Default =
                      ``1.0``).
    :param interpolation: Optional Matplotlib ``imshow`` interpolation
                          (Default = ``None``).
    :param interp_stage: Stage at which the images are resampled (Default =
                         ``'data'``).
    :return: The colour-mapped image artist (``matplotlib.image.AxesImage``),
             e.g. for another colour bar. The hillshade is the last image on
             the axis (``ax.images[-1]``), and with ``show_colour_bar=True``
             the colour bar is ``img.colorbar``.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.mapping.raster as map_raster

        fig, ax = plt.subplots(figsize=(7, 6))
        img = map_raster.create_shaded_cmap_img_map(
            ax,
            "flow_acc.tif",
            dem_img="dem.tif",
            cmap_name="Blues",
            vmin=1,
            use_log_norm=True,
            title_str="Flow accumulation",
            cbar_label="Upslope cells",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    _check_one_hillshade_input(hshade_img, dem_img)
    if bbox is None:
        bbox, _, _ = _read_raster_map_meta(input_img)
    img_artist = create_raster_cmap_img_map(
        ax,
        input_img,
        img_band,
        bbox=bbox,
        title_str=title_str,
        show_scale_bar=show_scale_bar,
        use_grid=use_grid,
        show_map_axis=show_map_axis,
        cmap_name=cmap_name,
        norm_img_vals=True,
        use_log_norm=use_log_norm,
        norm_vmin=vmin,
        norm_vmax=vmax,
        vals_under_white=vals_under_white,
        scale_bar_loc=scale_bar_loc,
        plot_zorder=1.0,
        scale_bar_style=scale_bar_style,
        scale_bar_units=scale_bar_units,
        show_north_arrow=show_north_arrow,
        north_arrow_style=north_arrow_style,
        north_arrow_loc=north_arrow_loc,
        show_colour_bar=False,
        img_alpha=img_alpha,
        interpolation=interpolation,
        interp_stage=interp_stage,
    )
    add_hillshade(
        ax,
        hshade_img,
        dem_img=dem_img,
        bbox=bbox,
        alpha=hshade_alpha,
        azimuth=azimuth,
        zenith=zenith,
        zorder=2.0,
        interpolation=interpolation,
        interp_stage=interp_stage,
    )
    if show_colour_bar:
        add_map_colour_bar(ax, img_artist, cbar_label=cbar_label)
    return img_artist
