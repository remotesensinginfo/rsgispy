"""
The tools.plotting module provides the rsgispy house plot style
(:func:`rsgispy_plot_style`, :func:`use_plot_style`) and the helpers every
rsgispy plotting function uses (:func:`get_fig_ax`, :func:`get_fig_axs`,
:func:`save_plot` and :func:`add_stats_text_box`), functions for reading
raster data into NumPy arrays and applying image stretches for visualisation
with Matplotlib, and generic plots (:func:`plot_histogram_threshold`,
:func:`plot_img_histogram`, :func:`plot_2d_img_histogram` and
:func:`plot_qq`).

See :ref:`plotting_conventions` for the conventions every rsgispy plotting
function follows.
"""

import contextlib
import functools
import math
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds

import rsgispy
from rsgispy.imageprocess import ensure_file_exists

__all__ = [
    "OKABE_ITO_COLOURS",
    "CATEGORICAL_COLOURS",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "PLOT_DPI",
    "REF_LINE_COLOUR",
    "get_plot_style_rc",
    "rsgispy_plot_style",
    "use_plot_style",
    "get_fig_ax",
    "get_fig_axs",
    "save_plot",
    "add_stats_text_box",
    "plot_feature_importance_bars",
    "get_raster_mpl_imshow",
    "get_thematic_raster_mpl_imshow",
    "limit_range_np_arr",
    "manual_stretch_np_arr",
    "linear_stretch_np_arr",
    "cumulative_stretch_np_arr",
    "stdev_stretch_np_arr",
    "plot_histogram_threshold",
    "plot_img_histogram",
    "plot_2d_img_histogram",
    "draw_histogram",
    "THRESHOLD_COLOURS",
    "plot_qq",
    "draw_qq",
    "describe_shapiro",
]

#: The Okabe-Ito colour-blind-safe palette (Okabe and Ito, 2008), by name.
OKABE_ITO_COLOURS: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

#: The colour cycle for categories (the Okabe-Ito colours, ordered so the
#: first ones are the most distinct). The first colour is used for a single
#: series and the second for fitted lines.
CATEGORICAL_COLOURS: List[str] = [
    OKABE_ITO_COLOURS["blue"],
    OKABE_ITO_COLOURS["vermillion"],
    OKABE_ITO_COLOURS["bluish_green"],
    OKABE_ITO_COLOURS["orange"],
    OKABE_ITO_COLOURS["reddish_purple"],
    OKABE_ITO_COLOURS["sky_blue"],
    OKABE_ITO_COLOURS["yellow"],
    OKABE_ITO_COLOURS["black"],
]

#: Default colour map for sequential data.
SEQUENTIAL_CMAP = "viridis"
#: Default colour map for differences and residuals (centre it on zero).
DIVERGING_CMAP = "RdBu_r"
#: Default resolution (dots per inch) of saved plots.
PLOT_DPI = 300

#: Colour of the one-to-one and zero reference lines.
REF_LINE_COLOUR = "#4d4d4d"

_PLOT_TYPES = ("chart", "map")

_F = TypeVar("_F", bound=Callable[..., Any])


def get_plot_style_rc(plot_type: str = "chart") -> Dict[str, Any]:
    """
    Get the Matplotlib ``rcParams`` of the rsgispy house plot style, as
    applied by :func:`rsgispy_plot_style`.

    The style uses font sizes readable at the default figure sizes, the
    colour-blind-safe :data:`CATEGORICAL_COLOURS` cycle, ``viridis`` for
    images, constrained layout (so labels, colour bars and legends do not
    overlap or clip) and legends with a light frame. Charts
    (``plot_type="chart"``) have light grid lines and no top or right spines;
    maps (``plot_type="map"``) keep all four spines and have no grid.

    :param plot_type: ``'chart'`` (Default) or ``'map'``.
    :return: Dictionary of ``rcParams`` values.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        map_rc = rplot.get_plot_style_rc("map")
    """
    if plot_type not in _PLOT_TYPES:
        raise rsgispy.RSGISPyException(
            f"plot_type must be one of {_PLOT_TYPES}, not '{plot_type}'."
        )
    is_chart = plot_type == "chart"
    return {
        "figure.constrained_layout.use": True,
        "figure.facecolor": "white",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8.0,
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#262626",
        "axes.edgecolor": "#4d4d4d",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "axes.spines.top": not is_chart,
        "axes.spines.right": not is_chart,
        "axes.grid": is_chart,
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL_COLOURS),
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#4d4d4d",
        "ytick.color": "#4d4d4d",
        "xtick.labelcolor": "#262626",
        "ytick.labelcolor": "#262626",
        "legend.fontsize": 9,
        "legend.title_fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.6,
        "scatter.edgecolors": "none",
        "image.cmap": SEQUENTIAL_CMAP,
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }


@contextlib.contextmanager
def rsgispy_plot_style(
    plot_type: str = "chart",
    rc: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    """
    Context manager applying the rsgispy house plot style
    (:func:`get_plot_style_rc`) with ``matplotlib.rc_context``. The
    previous ``rcParams`` are restored on exit, so the style never changes
    the global Matplotlib settings. Every rsgispy plotting function applies
    it internally (see :func:`use_plot_style`); use it to give your own plots
    the same look.

    The layout, spines, grid and colour cycle of a figure or axis are fixed
    when it is created, so create them inside the context.

    :param plot_type: ``'chart'`` (Default) or ``'map'`` (all spines, no grid).
    :param rc: Optional ``rcParams`` overriding the house style.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.plotting as rplot

        with rplot.rsgispy_plot_style():
            fig, ax = plt.subplots()
            ax.plot([0, 1, 2], [0, 1, 4], label="Data")
            ax.legend()
        rplot.save_plot(fig, "my_plot.png")
    """
    style_rc = get_plot_style_rc(plot_type)
    if rc is not None:
        style_rc.update(dict(rc))
    with matplotlib.rc_context(rc=style_rc):
        yield


def use_plot_style(
    func: Optional[Callable[..., Any]] = None, *, plot_type: str = "chart"
) -> Any:
    """
    Decorator running the decorated function inside
    :func:`rsgispy_plot_style`, so a plotting function applies the house
    style without changing the user's ``rcParams``. Use it on every rsgispy
    plotting function: ``plot_type="chart"`` for plots and
    ``plot_type="map"`` for functions that draw maps or map elements.

    :param func: Function to decorate (when used without arguments).
    :param plot_type: ``'chart'`` (Default) or ``'map'``.
    :return: The decorated function, or a decorator when ``func`` is ``None``.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        @rplot.use_plot_style
        def plot_my_values(vals, ax=None, out_plot_file=None, out_dpi=300):
            fig, ax = rplot.get_fig_ax(ax, figsize=(6.0, 4.0))
            ax.plot(vals)
            if out_plot_file is not None:
                rplot.save_plot(fig, out_plot_file, out_dpi=out_dpi)
            return fig, ax

        @rplot.use_plot_style(plot_type="map")
        def add_my_layer(ax):
            ...
    """
    if plot_type not in _PLOT_TYPES:
        raise rsgispy.RSGISPyException(
            f"plot_type must be one of {_PLOT_TYPES}, not '{plot_type}'."
        )

    def _decorate(in_func: _F) -> _F:
        @functools.wraps(in_func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            with rsgispy_plot_style(plot_type=plot_type):
                return in_func(*args, **kwargs)

        return _wrapper  # type: ignore[return-value]

    if func is not None:
        return _decorate(func)
    return _decorate


def get_fig_ax(
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **subplots_kw: Any,
) -> Tuple[Any, Any]:
    """
    Get the figure and axis to draw on: ``ax`` and its figure when ``ax`` is
    given, otherwise a new figure and axis from
    ``matplotlib.pyplot.subplots``.

    :param ax: Optional existing Matplotlib axis to draw on.
    :param figsize: Size ``(width, height)`` in inches of a new figure.
    :param subplots_kw: Extra keyword arguments for ``plt.subplots`` when a
                        new figure is created (e.g.
                        ``subplot_kw={"projection": "polar"}``).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        fig, ax = rplot.get_fig_ax(None, figsize=(6.0, 4.0))
    """
    if ax is not None:
        return ax.get_figure(), ax
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, new_ax = plt.subplots(figsize=figsize, **subplots_kw)
    return fig, new_ax


def get_fig_axs(
    axs: Optional[Any] = None,
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **subplots_kw: Any,
) -> Tuple[Any, np.ndarray]:
    """
    Get the figure and a 2D array of axes for a multi-panel plot: ``axs``
    (reshaped to ``(nrows, ncols)``) and its figure when given, otherwise a
    new figure from ``matplotlib.pyplot.subplots`` with ``squeeze=False``.

    :param axs: Optional existing axes (one axis, a sequence or an array)
                with ``nrows * ncols`` elements.
    :param nrows: Number of rows of panels (Default = 1).
    :param ncols: Number of columns of panels (Default = 1).
    :param figsize: Size ``(width, height)`` in inches of a new figure.
    :param subplots_kw: Extra keyword arguments for ``plt.subplots`` when a
                        new figure is created.
    :return: Tuple ``(fig, axs)``, with ``axs`` an array of shape
             ``(nrows, ncols)``.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        fig, axs = rplot.get_fig_axs(None, nrows=2, ncols=2, figsize=(8, 8))
    """
    if axs is not None:
        ax_arr = np.asarray(axs, dtype=object).ravel()
        if ax_arr.size != nrows * ncols:
            raise rsgispy.RSGISPyException(
                f"{nrows * ncols} axes are needed (nrows={nrows}, "
                f"ncols={ncols}) but {ax_arr.size} were given."
            )
        return ax_arr[0].get_figure(), ax_arr.reshape(nrows, ncols)
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, new_axs = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, **subplots_kw
    )
    return fig, new_axs


def save_plot(
    fig: Any,
    out_plot_file: Union[str, os.PathLike],
    out_dpi: int = PLOT_DPI,
    close: bool = False,
) -> str:
    """
    Save a figure with ``bbox_inches="tight"``, creating the parent directory
    if needed; the format comes from the file extension. The figure is not
    closed, so it still shows in a notebook, unless ``close=True`` (used by
    functions that write many figures, to free their memory).

    :param fig: Matplotlib figure to save.
    :param out_plot_file: Output image file path (e.g. ``.png`` or ``.pdf``).
    :param out_dpi: Resolution in dots per inch (Default = ``300``).
    :param close: Close the figure after saving it (Default = ``False``).
    :return: The output file path.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.plotting as rplot

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        rplot.save_plot(fig, "plots/line.png", out_dpi=150)
    """
    out_path = os.fspath(out_plot_file)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=out_dpi, bbox_inches="tight")
    finally:
        if close:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

            plt.close(fig)
    return out_path


def add_stats_text_box(
    ax: Any,
    text: Union[str, Sequence[str]],
    loc: str = "upper left",
    fontsize: Optional[float] = None,
) -> Any:
    """
    Add a small box of text (e.g. ``r²``, RMSE and ``n``) inside ``ax``, in
    the style of the rsgispy plots.

    :param ax: Matplotlib axis.
    :param text: Text, or a sequence of lines.
    :param loc: Placement, as for a legend (Default = ``'upper left'``).
    :param fontsize: Font size (Default = the legend font size).
    :return: The ``matplotlib.offsetbox.AnchoredText`` artist.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.tools.plotting as rplot

        fig, ax = plt.subplots()
        rplot.add_stats_text_box(ax, ["RMSE = 1.20", "n = 50"])
    """
    from matplotlib.offsetbox import AnchoredText  # pylint: disable=import-outside-toplevel

    if not isinstance(text, str):
        text = "\n".join(str(line) for line in text)
    if fontsize is None:
        fontsize = matplotlib.rcParams["legend.fontsize"]
    txt_box = AnchoredText(
        text,
        loc=loc,
        frameon=True,
        prop={"fontsize": fontsize},
        pad=0.4,
        borderpad=0.6,
    )
    txt_box.patch.set_facecolor("white")
    txt_box.patch.set_alpha(0.9)
    txt_box.patch.set_edgecolor("#cccccc")
    txt_box.patch.set_linewidth(0.6)
    txt_box.set_zorder(5)
    ax.add_artist(txt_box)
    return txt_box


def _get_subset_window_and_coords(
    src: rasterio.io.DatasetReader,
    bbox: Optional[Tuple[float, float, float, float]],
) -> Tuple[Window, Tuple[float, float, float, float]]:
    """
    Compute the clipped rasterio pixel Window and exact spatial extent
    ``(MinX, MaxX, MinY, MaxY)`` for an optional bounding box.
    """
    # pylint: disable=too-many-locals
    if bbox is None:
        win = Window(0, 0, src.width, src.height)
        left, bottom, right, top = src.bounds
        return win, (float(left), float(right), float(bottom), float(top))

    img_left, img_bottom, img_right, img_top = src.bounds
    sub_left, sub_right, sub_bottom, sub_top = (float(v) for v in bbox[:4])

    intersects = (
        (img_right > sub_left)
        and (sub_right > img_left)
        and (img_top > sub_bottom)
        and (sub_top > img_bottom)
    )
    if not intersects:
        raise rsgispy.RSGISPyException(
            "The subset BBOX does not intersect with the image bbox - "
            "is the projection of the bbox correct?"
        )

    inter_left = max(img_left, sub_left)
    inter_right = min(img_right, sub_right)
    inter_bottom = max(img_bottom, sub_bottom)
    inter_top = min(img_top, sub_top)

    raw_win = window_from_bounds(
        inter_left,
        inter_bottom,
        inter_right,
        inter_top,
        transform=src.transform,
    )

    col_start = max(0, min(src.width - 1, math.floor(raw_win.col_off + 0.5)))
    col_stop = max(
        col_start + 1,
        min(src.width, math.floor(raw_win.col_off + raw_win.width + 0.5)),
    )
    row_start = max(0, min(src.height - 1, math.floor(raw_win.row_off + 0.5)))
    row_stop = max(
        row_start + 1,
        min(src.height, math.floor(raw_win.row_off + raw_win.height + 0.5)),
    )

    win = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )
    w_left, w_bottom, w_right, w_top = window_bounds(win, src.transform)
    return win, (float(w_left), float(w_right), float(w_bottom), float(w_top))


def get_raster_mpl_imshow(
    input_img: str,
    bands: Optional[Sequence[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Retrieve raster image data as a NumPy array structured for
    ``matplotlib.axes.Axes.imshow`` together with its spatial extent.

    :param input_img: Path to the input raster image file.
    :param bands: Sequence of 1-based band indices (length 1 or 3). If ``None``,
                  all bands in the image are used (must be 1 or 3 bands).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` in the image
                 CRS used to spatially subset the raster.
    :return: Tuple ``(img_data_arr, coords_bbox)`` where ``img_data_arr`` has
             shape ``(height, width)`` for 1 band or ``(height, width, 3)``
             for 3 bands, and ``coords_bbox`` is ``(MinX, MaxX, MinY, MaxY)``.

    .. code:: python

        import rsgispy.tools.plotting

        img_arr, extent = rsgispy.tools.plotting.get_raster_mpl_imshow(
            "sentinel2.tif", bands=[4, 3, 2]
        )
    """
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        n_bands = src.count
        ref_bands = list(range(1, n_bands + 1))
        sel_bands = list(ref_bands) if bands is None else [int(b) for b in bands]

        for band in sel_bands:
            if band not in ref_bands:
                raise rsgispy.RSGISPyException(
                    f"Band {band} is not valid (i.e., within the image)"
                )

        if len(sel_bands) not in (1, 3):
            raise rsgispy.RSGISPyException(
                "The number of bands specified must be 3 or 1."
            )

        win, coords_bbox = _get_subset_window_and_coords(src, bbox)
        if len(sel_bands) == 1:
            img_data_arr = src.read(sel_bands[0], window=win)
        else:
            img_data_arr = np.moveaxis(src.read(sel_bands, window=win), 0, -1)

    return img_data_arr, coords_bbox


def get_thematic_raster_mpl_imshow(
    input_img: str,
    band: int = 1,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    out_patches: bool = False,
    cls_names_lut: Optional[Mapping[int, Any]] = None,
    alpha_lyr: bool = False,
    ctab_alpha: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[List[Patch]]]:
    """
    Retrieve thematic image data with a colour table as an RGB or RGBA NumPy
    array for ``matplotlib.axes.Axes.imshow``, its spatial extent, and optional
    legend patches.

    :param input_img: Path to the input thematic raster image file.
    :param band: 1-based band index (Default = 1).
    :param bbox: Optional bounding box ``(MinX, MaxX, MinY, MaxY)`` in the image
                 CRS used to spatially subset the raster.
    :param out_patches: If True, build a list of ``matplotlib.patches.Patch``
                        objects for creating a legend (Default = False).
    :param cls_names_lut: Optional mapping from pixel class value to class label.
    :param alpha_lyr: If True, include an alpha channel where background pixel
                      value 0 is transparent (Default = False).
    :param ctab_alpha: If True, include alpha values from the colour table
                       (Default = True).
    :return: Tuple ``(img_clr_data_arr, coords_bbox, lgd_out_patches)``.

    .. code:: python

        import rsgispy.tools.plotting

        img_rgba, extent, patches = (
            rsgispy.tools.plotting.get_thematic_raster_mpl_imshow(
                "landcover.tif", out_patches=True, cls_names_lut={1: "Forest"}
            )
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)

    with rasterio.open(input_img, "r") as src:
        if band <= 0 or band > src.count:
            raise rsgispy.RSGISPyException(
                f"Band {band} is not valid (i.e., within the image)"
            )

        try:
            clr_tab = src.colormap(band)
        except ValueError as err:
            raise rsgispy.RSGISPyException("No colour table was present") from err

        if not clr_tab:
            raise rsgispy.RSGISPyException("No colour table was present")

        win, coords_bbox = _get_subset_window_and_coords(src, bbox)
        img_data_arr = src.read(band, window=win)

    use_alpha = alpha_lyr or ctab_alpha
    lgd_out_patches: Optional[List[Patch]] = [] if out_patches else None
    present_vals = set(int(v) for v in np.unique(img_data_arr))
    max_key = max((int(k) for k in clr_tab.keys()), default=0)

    if (
        img_data_arr.dtype.kind in ("u", "i")
        and int(np.min(img_data_arr)) >= 0
        and max(int(np.max(img_data_arr)), max_key) < 65536
    ):
        lut_size = max(int(np.max(img_data_arr)), max_key) + 1
        n_ch = 4 if use_alpha else 3
        lut = np.zeros((lut_size, n_ch), dtype=np.uint8)
        for val, rgba in clr_tab.items():
            idx = int(val)
            if 0 <= idx < lut_size:
                lut[idx, 0] = int(rgba[0])
                lut[idx, 1] = int(rgba[1])
                lut[idx, 2] = int(rgba[2])
                if use_alpha:
                    a_val = int(rgba[3]) if len(rgba) >= 4 else 255
                    if ctab_alpha:
                        lut[idx, 3] = 0 if (alpha_lyr and idx == 0) else a_val
                    elif alpha_lyr and idx > 0:
                        lut[idx, 3] = 255
        img_clr_data_arr = lut[img_data_arr]
    else:
        red_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
        grn_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
        blu_arr = np.zeros_like(img_data_arr, dtype=np.uint8)
        alp_arr = (
            np.zeros_like(img_data_arr, dtype=np.uint8) if use_alpha else None
        )
        for val in sorted(present_vals):
            if val in clr_tab:
                rgba = clr_tab[val]
                r_val, g_val, b_val = int(rgba[0]), int(rgba[1]), int(rgba[2])
                a_val = int(rgba[3]) if len(rgba) >= 4 else 255
                mask = img_data_arr == val
                red_arr[mask] = r_val
                grn_arr[mask] = g_val
                blu_arr[mask] = b_val
                if alp_arr is not None:
                    if ctab_alpha:
                        alp_arr[mask] = 0 if (alpha_lyr and val == 0) else a_val
                    elif alpha_lyr and val > 0:
                        alp_arr[mask] = 255
        if alp_arr is not None:
            img_clr_data_arr = np.stack(
                [red_arr, grn_arr, blu_arr, alp_arr], axis=-1
            )
        else:
            img_clr_data_arr = np.stack([red_arr, grn_arr, blu_arr], axis=-1)

    if lgd_out_patches is not None:
        for val, rgba in sorted(clr_tab.items()):
            if val <= 0:
                continue
            in_lut = cls_names_lut is not None and val in cls_names_lut
            if val not in present_vals and not in_lut:
                continue
            r_val, g_val, b_val = int(rgba[0]), int(rgba[1]), int(rgba[2])
            a_val = int(rgba[3]) if len(rgba) >= 4 else 255
            cls_name = f"{cls_names_lut[val]}" if in_lut else f"{val}"
            if ctab_alpha:
                patch_clr: Tuple[float, ...] = (
                    r_val / 255.0,
                    g_val / 255.0,
                    b_val / 255.0,
                    a_val / 255.0,
                )
            else:
                patch_clr = (r_val / 255.0, g_val / 255.0, b_val / 255.0)
            lgd_out_patches.append(
                Patch(facecolor=patch_clr, edgecolor=patch_clr, label=cls_name)
            )

    return img_clr_data_arr, coords_bbox, lgd_out_patches


def limit_range_np_arr(
    arr_data: np.ndarray,
    min_thres: float = 0.0,
    min_out_val: float = 0.0,
    max_thres: float = 1.0,
    max_out_val: float = 1.0,
) -> np.ndarray:
    """
    Clamp values of a NumPy array outside ``[min_thres, max_thres]``.

    :param arr_data: Input NumPy array.
    :param min_thres: Lower threshold (Default = 0.0).
    :param min_out_val: Replacement value below ``min_thres`` (Default = 0.0).
    :param max_thres: Upper threshold (Default = 1.0).
    :param max_out_val: Replacement value above ``max_thres`` (Default = 1.0).
    :return: Clamped NumPy array copy.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([-0.2, 0.5, 1.4])
        clamped = rsgispy.tools.plotting.limit_range_np_arr(arr, 0.0, 0.0, 1.0, 1.0)
    """
    arr_data_out = arr_data.copy()
    arr_data_out[arr_data < min_thres] = min_out_val
    arr_data_out[arr_data > max_thres] = max_out_val
    return arr_data_out


def _prep_stretch_array(
    arr_data: np.ndarray, no_data_val: Optional[float]
) -> np.ndarray:
    """Copy ``arr_data`` to float64 and mask ``no_data_val`` as NaN."""
    arr_data_out = arr_data.astype(float, copy=True)
    if no_data_val is not None:
        arr_data_out[arr_data == no_data_val] = np.nan
    return arr_data_out


def _rescale_band(
    band_arr: np.ndarray,
    low_val: float,
    up_val: float,
    out_off: float,
    out_gain: float,
) -> np.ndarray:
    """Rescale a single band array cleanly, guarding against zero range."""
    range_val = up_val - low_val
    if range_val == 0 or not np.isfinite(range_val):
        out_arr = np.full_like(band_arr, out_off, dtype=float)
        out_arr[np.isnan(band_arr)] = np.nan
        return out_arr
    return (((band_arr - low_val) / range_val) * out_gain) + out_off


def _finalise_stretch_array(
    arr_data: np.ndarray,
    arr_data_out: np.ndarray,
    no_data_val: Optional[float],
    no_data_clr: Optional[Union[float, Sequence[float]]],
    min_out_val: float,
    max_out_val: float,
    out_int_type: bool,
) -> np.ndarray:
    """Clamp range, apply optional no-data display colour, and cast output dtype."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = limit_range_np_arr(
        arr_data_out,
        min_thres=min_out_val,
        min_out_val=min_out_val,
        max_thres=max_out_val,
        max_out_val=max_out_val,
    )

    if (no_data_val is not None) and (no_data_clr is not None):
        if arr_data.ndim == 2:
            clr_val = (
                no_data_clr[0]
                if isinstance(no_data_clr, (tuple, list))
                else no_data_clr
            )
            arr_data_out[arr_data == no_data_val] = clr_val
        else:
            for band_idx in range(arr_data.shape[2]):
                clr_val = (
                    no_data_clr[band_idx]
                    if isinstance(no_data_clr, (tuple, list))
                    else no_data_clr
                )
                no_data_msk = arr_data[..., band_idx] == no_data_val
                arr_data_out[no_data_msk, band_idx] = clr_val

    if out_int_type:
        arr_data_out = arr_data_out.astype(int)

    return arr_data_out


def manual_stretch_np_arr(
    arr_data: np.ndarray,
    min_max_vals: Union[Dict[str, float], List[Dict[str, float]]],
    no_data_val: Optional[float] = None,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a manual per-band linear stretch using user-specified ``min``/``max`` values.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param min_max_vals: Dict ``{'min': ..., 'max': ...}`` for a 2D array or list
                         of dicts (one per band) for a 3D array.
    :param no_data_val: Optional no-data pixel value.
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([[0.0, 500.0], [1000.0, 2000.0]])
        stretched = rsgispy.tools.plotting.manual_stretch_np_arr(
            arr, min_max_vals={"min": 0.0, "max": 2000.0}
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        if not isinstance(min_max_vals, dict):
            raise rsgispy.RSGISPyException(
                "Just 1 dimension within arr_data and therefore "
                "min_max_vals variable must be a dict."
            )
        if ("min" not in min_max_vals) or ("max" not in min_max_vals):
            raise rsgispy.RSGISPyException(
                "min and max keys must be provided within the dict"
            )
        arr_data_out = _rescale_band(
            arr_data_out,
            float(min_max_vals["min"]),
            float(min_max_vals["max"]),
            out_off,
            out_gain,
        )
    else:
        if not isinstance(min_max_vals, list):
            raise rsgispy.RSGISPyException(
                "arr_data has more than 1 dimension and therefore "
                "min_max_vals variable must be a list."
            )
        n_bands = arr_data.shape[2]
        if n_bands != len(min_max_vals):
            raise rsgispy.RSGISPyException(
                "length of min_max_vals must be the same as the number "
                "of bands in arr_data."
            )
        for band_idx in range(n_bands):
            band_mm = min_max_vals[band_idx]
            if ("min" not in band_mm) or ("max" not in band_mm):
                raise rsgispy.RSGISPyException(
                    "min and max keys must be provided within the dict"
                )
            arr_data_out[..., band_idx] = _rescale_band(
                arr_data_out[..., band_idx],
                float(band_mm["min"]),
                float(band_mm["max"]),
                out_off,
                out_gain,
            )

    return _finalise_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def linear_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band min-max linear stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.array([[10.0, 50.0], [100.0, 200.0]])
        stretched = rsgispy.tools.plotting.linear_stretch_np_arr(arr)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        min_val = float(np.nanmin(arr_data_out))
        max_val = float(np.nanmax(arr_data_out))
        arr_data_out = _rescale_band(
            arr_data_out, min_val, max_val, out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            min_val = float(np.nanmin(band_slice))
            max_val = float(np.nanmax(band_slice))
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, min_val, max_val, out_off, out_gain
            )

    return _finalise_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def cumulative_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    lower: float = 2.0,
    upper: float = 98.0,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band cumulative percentile stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param lower: Lower percentile (Default = 2.0).
    :param upper: Upper percentile (Default = 98.0).
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.arange(100, dtype=float).reshape(10, 10)
        stretched = rsgispy.tools.plotting.cumulative_stretch_np_arr(
            arr, lower=2.0, upper=98.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    if arr_data.ndim == 2:
        min_val, max_val = np.nanpercentile(arr_data_out, [lower, upper])
        arr_data_out = _rescale_band(
            arr_data_out, float(min_val), float(max_val), out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            min_val, max_val = np.nanpercentile(band_slice, [lower, upper])
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, float(min_val), float(max_val), out_off, out_gain
            )

    return _finalise_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


def stdev_stretch_np_arr(
    arr_data: np.ndarray,
    no_data_val: Optional[float] = None,
    n_stdevs: float = 2.0,
    out_off: float = 0.0,
    out_gain: float = 1.0,
    out_int_type: bool = False,
    min_out_val: float = 0.0,
    max_out_val: float = 1.0,
    no_data_clr: Optional[Union[float, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """
    Apply a per-band standard deviation stretch to a NumPy array.

    :param arr_data: Input NumPy array with shape ``(n, m)`` or ``(n, m, b)``.
    :param no_data_val: Optional no-data pixel value.
    :param n_stdevs: Number of standard deviations around the mean (Default = 2.0).
    :param out_off: Output offset (Default = 0.0).
    :param out_gain: Output gain (Default = 1.0).
    :param out_int_type: If True, cast stretched output to integer (Default = False).
    :param min_out_val: Minimum output clamp value (Default = 0.0).
    :param max_out_val: Maximum output clamp value (Default = 1.0).
    :param no_data_clr: Optional colour value(s) assigned to no-data pixels.
    :return: Stretched NumPy array.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        arr = np.arange(100, dtype=float).reshape(10, 10)
        stretched = rsgispy.tools.plotting.stdev_stretch_np_arr(arr, n_stdevs=2.0)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr_data_out = _prep_stretch_array(arr_data, no_data_val)

    def _stretch_1d_bounds(band_slice: np.ndarray) -> Tuple[float, float]:
        min_val = float(np.nanmin(band_slice))
        max_val = float(np.nanmax(band_slice))
        mean_val = float(np.nanmean(band_slice))
        std_val = float(np.nanstd(band_slice))
        low_val = max(min_val, mean_val - (std_val * n_stdevs))
        up_val = min(max_val, mean_val + (std_val * n_stdevs))
        return low_val, up_val

    if arr_data.ndim == 2:
        low_val, up_val = _stretch_1d_bounds(arr_data_out)
        arr_data_out = _rescale_band(
            arr_data_out, low_val, up_val, out_off, out_gain
        )
    else:
        for band_idx in range(arr_data.shape[2]):
            band_slice = arr_data_out[..., band_idx]
            low_val, up_val = _stretch_1d_bounds(band_slice)
            arr_data_out[..., band_idx] = _rescale_band(
                band_slice, low_val, up_val, out_off, out_gain
            )

    arr_data_out[arr_data_out < out_off] = out_off
    arr_data_out[arr_data_out > (out_gain + out_off)] = out_gain + out_off

    return _finalise_stretch_array(
        arr_data=arr_data,
        arr_data_out=arr_data_out,
        no_data_val=no_data_val,
        no_data_clr=no_data_clr,
        min_out_val=min_out_val,
        max_out_val=max_out_val,
        out_int_type=out_int_type,
    )


@use_plot_style
def plot_histogram_threshold(
    data: np.ndarray,
    threshold: Optional[float] = None,
    *,
    title_str: Optional[str] = None,
    x_label: Optional[str] = None,
    hist_clr: str = "#9a9a9a",
    threshold_label: Optional[str] = None,
    n_bins: Optional[int] = None,
    figsize: Tuple[float, float] = (6.4, 4.2),
    ax: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, Any]:
    """
    Plot a histogram of a 1D NumPy array, optionally with a vertical line at
    a threshold.

    :param data: NumPy array of values to plot (flattened; non-finite values
                 are ignored).
    :param threshold: Optional threshold value drawn as a vertical line.
    :param title_str: Optional plot title.
    :param x_label: Optional x-axis label (Default = ``'Value'``).
    :param hist_clr: Colour of the histogram bars (Default = ``'#9a9a9a'``,
                     a mid grey).
    :param threshold_label: Legend label of the threshold line (Default =
                            ``'Threshold = <value>'``).
    :param n_bins: Number of histogram bins (estimated with the
                   Freedman-Diaconis rule if ``None``).
    :param figsize: Size of a new figure in inches (Default = ``(6.4, 4.2)``).
    :param ax: Optional Matplotlib axis to draw on (Default = a new figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting

        vals = np.concatenate([np.random.normal(2.0, 0.4, 200), np.random.normal(6.0, 0.4, 200)])
        fig, ax = rsgispy.tools.plotting.plot_histogram_threshold(
            vals, threshold=4.0, title_str="Threshold", out_plot_file="hist_thres.png"
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.tools.thresholding import get_nbins_histogram

    data_arr = np.asarray(data, dtype=float).ravel()
    data_arr = data_arr[np.isfinite(data_arr)]
    if data_arr.size == 0:
        raise rsgispy.RSGISPyException(
            "Input data array contains no finite values to plot."
        )

    if n_bins is None:
        iqr = float(np.percentile(data_arr, 75) - np.percentile(data_arr, 25))
        data_span = float(np.max(data_arr) - np.min(data_arr))
        if iqr > 0.0 and data_span > 0.0:
            n_bins, _ = get_nbins_histogram(data_arr)
        else:
            n_bins = min(50, max(1, int(data_arr.size)))

    counts, bin_edges = np.histogram(data_arr, bins=n_bins)
    fig, ax = get_fig_ax(ax, figsize=figsize)
    draw_histogram(
        ax,
        counts,
        bin_edges,
        thresholds=threshold,
        threshold_labels=threshold_label,
        hist_clr=hist_clr,
    )
    if title_str is not None:
        ax.set_title(title_str)
    ax.set_xlabel(x_label if x_label is not None else "Value")
    ax.set_ylabel("Frequency")
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax


#: Colours of the threshold lines on histograms, in order.
THRESHOLD_COLOURS: List[str] = [
    OKABE_ITO_COLOURS["vermillion"],
    OKABE_ITO_COLOURS["bluish_green"],
    OKABE_ITO_COLOURS["orange"],
    OKABE_ITO_COLOURS["reddish_purple"],
    OKABE_ITO_COLOURS["sky_blue"],
    OKABE_ITO_COLOURS["black"],
]


def _as_list(vals: Any) -> List[Any]:
    """``None`` -> ``[]``, a scalar or string -> ``[vals]``, else a list."""
    if vals is None:
        return []
    if isinstance(vals, (str, bytes)) or np.ndim(vals) == 0:
        return [vals]
    return list(vals)


def draw_histogram(
    ax: Any,
    counts: Union[np.ndarray, Sequence[float]],
    bin_edges: Union[np.ndarray, Sequence[float]],
    thresholds: Optional[Union[float, Sequence[float]]] = None,
    threshold_labels: Optional[Union[str, Sequence[str]]] = None,
    hist_clr: str = "#9a9a9a",
    log_y: bool = False,
) -> Any:
    """
    Draw a histogram from its counts and bin edges on ``ax`` as bars, with
    optional thresholds as labelled, dashed vertical lines (and a legend).
    Used by :func:`plot_histogram_threshold` and :func:`plot_img_histogram`
    so the two look the same; no axis labels or title are set.

    :param ax: Matplotlib axis.
    :param counts: Count (or any height) of each bin.
    :param bin_edges: Bin edges (``len(counts) + 1`` values).
    :param thresholds: Optional threshold, or list of thresholds, drawn as
                       vertical lines.
    :param threshold_labels: Optional legend label, or one label per
                             threshold (Default = ``'Threshold = <value>'``).
    :param hist_clr: Colour of the bars (Default = ``'#9a9a9a'``, a mid grey).
    :param log_y: Use a logarithmic y axis (Default = ``False``).
    :return: The bars (``matplotlib.container.BarContainer``).

    .. code:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import rsgispy.tools.plotting as rplot

        counts, edges = np.histogram(np.random.default_rng(0).normal(size=500), bins=30)
        fig, ax = plt.subplots()
        rplot.draw_histogram(ax, counts, edges, thresholds=[-1.0, 1.0],
                             threshold_labels=["Low", "High"])
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    counts_arr = np.asarray(counts, dtype=float)
    edges_arr = np.asarray(bin_edges, dtype=float)
    if edges_arr.size != counts_arr.size + 1:
        raise rsgispy.RSGISPyException(
            f"bin_edges must have one more value than counts ({counts_arr.size} "
            f"counts, {edges_arr.size} edges)."
        )
    bars = ax.bar(
        edges_arr[:-1],
        counts_arr,
        width=np.diff(edges_arr),
        align="edge",
        color=hist_clr,
        edgecolor="none",
        linewidth=0,
    )
    ax.grid(axis="x", visible=False)
    if log_y:
        ax.set_yscale("log")

    thres_lst = [float(val) for val in _as_list(thresholds)]
    lbl_lst = _as_list(threshold_labels)
    if lbl_lst and len(lbl_lst) != len(thres_lst):
        raise rsgispy.RSGISPyException(
            f"{len(thres_lst)} threshold(s) but {len(lbl_lst)} threshold label(s)."
        )
    for idx, thres in enumerate(thres_lst):
        ax.axvline(
            x=thres,
            color=THRESHOLD_COLOURS[idx % len(THRESHOLD_COLOURS)],
            linestyle="--",
            linewidth=1.8,
            label=str(lbl_lst[idx]) if lbl_lst else f"Threshold = {thres:.3g}",
        )
    if thres_lst:
        ax.legend(loc="upper right")
    return bars


@use_plot_style
def plot_img_histogram(
    input_img: Union[str, os.PathLike],
    img_band: int = 1,
    *,
    n_bins: int = 200,
    val_range: Optional[Tuple[float, float]] = None,
    no_data_val: Optional[float] = None,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    thresholds: Optional[Union[float, Sequence[float]]] = None,
    threshold_labels: Optional[Union[str, Sequence[str]]] = None,
    log_y: bool = False,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    hist_clr: str = "#9a9a9a",
    hist_data: Optional[Mapping[str, Any]] = None,
    figsize: Tuple[float, float] = (6.4, 4.2),
    ax: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
    use_pbar: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot the histogram of an image band (e.g. a change distance or an index),
    optionally with thresholds drawn as labelled vertical lines. The histogram
    is computed block by block with
    :func:`~rsgispy.imagecalc.stats.get_img_histogram`, so it works on images
    of any size, and drawn as :func:`plot_histogram_threshold` draws a
    histogram of an array.

    :param input_img: Path to the input image.
    :param img_band: 1-based band to plot (Default = ``1``).
    :param n_bins: Number of bins (Default = ``200``).
    :param val_range: Optional ``(min, max)`` of the values counted and of the
                      bins (Default = the range of the valid pixels).
    :param no_data_val: Optional no-data value (Default = the band's no-data
                        value from the image header, if any).
    :param in_msk_img: Optional mask image; only pixels equal to
                       ``img_msk_val`` are counted.
    :param img_msk_val: Mask value of the pixels to count (Default = ``1``).
    :param thresholds: Optional threshold, or list of thresholds, drawn as
                       vertical lines.
    :param threshold_labels: Optional legend label, or one label per
                             threshold (Default = ``'Threshold = <value>'``).
    :param log_y: Use a logarithmic y axis, e.g. to see the tails of a
                  change distance (Default = ``False``).
    :param title: Optional plot title.
    :param x_label: Optional x-axis label (Default = the band name).
    :param hist_clr: Colour of the bars (Default = ``'#9a9a9a'``).
    :param hist_data: Optional histogram from
                      :func:`~rsgispy.imagecalc.stats.get_img_histogram`
                      (with ``counts`` and ``bin_edges``), so the image is not
                      read again; the image arguments are then not used
                      (except ``input_img`` and ``img_band`` for the default
                      x-axis label).
    :param figsize: Size of a new figure in inches (Default = ``(6.4, 4.2)``).
    :param ax: Optional Matplotlib axis to draw on (Default = a new figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :param use_pbar: Show a progress bar while the histogram is computed
                     (Default = ``True``).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        fig, ax = rplot.plot_img_histogram(
            "cva_distance.tif",
            n_bins=200,
            val_range=(-3, 8),
            in_msk_img="analysis_mask.tif",
            thresholds=0.0,
            threshold_labels="Change threshold",
            title="CVA change distance",
            x_label="Change distance",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    from rsgispy.imagecalc.stats import get_img_histogram

    if hist_data is None:
        hist_data = get_img_histogram(
            input_img,
            img_band=img_band,
            n_bins=n_bins,
            val_range=val_range,
            no_data_val=no_data_val,
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            use_pbar=use_pbar,
        )
    if x_label is None:
        x_label = _get_band_name(input_img, img_band)

    fig, ax = get_fig_ax(ax, figsize=figsize)
    draw_histogram(
        ax,
        hist_data["counts"],
        hist_data["bin_edges"],
        thresholds=thresholds,
        threshold_labels=threshold_labels,
        hist_clr=hist_clr,
        log_y=log_y,
    )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pixel count")
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax


def _get_band_name(input_img: Union[str, os.PathLike], band: int) -> str:
    """The band's description in ``input_img``, or ``'Band <n>'``."""
    with rasterio.open(input_img, "r") as src:
        if 1 <= band <= src.count and src.descriptions[band - 1]:
            return str(src.descriptions[band - 1])
    return f"Band {band}"


@use_plot_style
def plot_2d_img_histogram(
    input_img: Union[str, os.PathLike],
    x_band: int,
    y_band: int,
    *,
    y_img: Optional[Union[str, os.PathLike]] = None,
    bins: Union[int, Sequence[int]] = 256,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    no_data_val: Optional[float] = None,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    val_scale: float = 1.0,
    normalise: bool = True,
    log_scale: bool = True,
    cmap: str = SEQUENTIAL_CMAP,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    hist_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    points: Optional[
        Union[Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]]
    ] = None,
    point_labels: Optional[Sequence[str]] = None,
    show_colour_bar: bool = True,
    figsize: Tuple[float, float] = (6.4, 5.4),
    ax: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
    use_pbar: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot the 2D histogram (feature space) of two image bands, e.g. NIR
    against red, to see the clusters and edges of the data (endmembers are
    often at the corners of the feature space). The histogram is computed
    block by block with :func:`~rsgispy.imagecalc.stats.get_img_2d_histogram`.

    Empty bins are white. The values are divided by ``val_scale`` (e.g. 100
    for reflectance x 10,000 to give %); ``x_range``, ``y_range`` and
    ``points`` are in the image units. The axis aspect is equal when the two
    ranges are equal. Points (e.g. endmembers) can be marked and labelled.

    :param input_img: Path to the input image (x band and, by default, y band).
    :param x_band: 1-based band of the x values in ``input_img``.
    :param y_band: 1-based band of the y values in ``y_img`` (or in
                   ``input_img`` when ``y_img`` is ``None``).
    :param y_img: Optional second image, on the same pixel grid, providing
                  ``y_band`` (Default = ``None``).
    :param bins: Number of bins of both axes, or ``[n_x_bins, n_y_bins]``
                 (Default = ``256``).
    :param x_range: Optional ``(min, max)`` of the x values in image units
                    (Default = the range of the valid pixels).
    :param y_range: Optional ``(min, max)`` of the y values, as ``x_range``.
    :param no_data_val: Optional no-data value of both bands (Default = the
                        bands' no-data values from the image headers).
    :param in_msk_img: Optional mask image; only pixels equal to
                       ``img_msk_val`` are counted.
    :param img_msk_val: Mask value of the pixels to count (Default = ``1``).
    :param val_scale: Divide the values by this for plotting (Default =
                      ``1.0``).
    :param normalise: Plot the proportion of the pixels in each bin (the
                      histogram divided by its sum) instead of the count
                      (Default = ``True``).
    :param log_scale: Use a logarithmic colour scale, so sparse bins at the
                      edges of the feature space are visible (Default =
                      ``True``).
    :param cmap: Matplotlib colour map name (Default = ``'viridis'``).
    :param x_label: Optional x-axis label (Default = the x band name).
    :param y_label: Optional y-axis label (Default = the y band name).
    :param title: Optional plot title.
    :param hist_data: Optional ``(hist, x_edges, y_edges)`` from
                      :func:`~rsgispy.imagecalc.stats.get_img_2d_histogram`,
                      so the image is not read again; the histogram arguments
                      are then not used (except the bands and images for the
                      default axis labels).
    :param points: Optional points to mark, in image units: a list of
                   ``(x, y)`` or a dict ``{label: (x, y)}`` (e.g. the
                   endmembers' values in the two bands).
    :param point_labels: Optional labels of the points when ``points`` is a
                         list (one per point).
    :param show_colour_bar: Draw a colour bar (Default = ``True``).
    :param figsize: Size of a new figure in inches (Default = ``(6.4, 5.4)``).
    :param ax: Optional Matplotlib axis to draw on (Default = a new figure).
    :param out_plot_file: Optional output image file; the figure is saved
                          and left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :param use_pbar: Show a progress bar while the histogram is computed
                     (Default = ``True``).
    :return: Tuple ``(fig, ax)``; the histogram image is ``ax.images[0]``.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        # NIR (band 42) against red (band 13) of a HyMap image (reflectance
        # x 10,000), in %, with two endmembers marked.
        fig, ax = rplot.plot_2d_img_histogram(
            "hymap.tif",
            x_band=42,
            y_band=13,
            bins=100,
            x_range=(1, 6000),
            y_range=(1, 6000),
            val_scale=100,
            x_label="NIR reflectance (%)",
            y_label="Red reflectance (%)",
            points={"Vegetation": (4200, 300), "Soil": (2500, 1900)},
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=import-outside-toplevel
    import matplotlib.colors as mcolors

    from rsgispy.imagecalc.stats import get_img_2d_histogram

    if val_scale == 0:
        raise rsgispy.RSGISPyException("val_scale must not be 0.")
    if hist_data is None:
        hist_data = get_img_2d_histogram(
            input_img,
            x_band,
            y_band,
            y_img=y_img,
            bins=bins,
            x_range=x_range,
            y_range=y_range,
            no_data_val=no_data_val,
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            use_pbar=use_pbar,
        )
    hist, x_edges, y_edges = hist_data
    hist = np.asarray(hist, dtype=np.float64)
    x_edges = np.asarray(x_edges, dtype=np.float64) / val_scale
    y_edges = np.asarray(y_edges, dtype=np.float64) / val_scale
    total = float(hist.sum())
    if normalise and total > 0:
        hist = hist / total
    # Empty bins are masked (drawn white); indexed [x, y] so transposed.
    plot_hist = np.ma.masked_less_equal(hist.T, 0.0)

    c_norm: Optional[mcolors.Normalize] = None
    if np.any(hist > 0.0):
        h_min = float(hist[hist > 0.0].min())
        h_max = float(hist.max())
        if log_scale:
            c_norm = mcolors.LogNorm(vmin=h_min, vmax=max(h_max, h_min * 1.0001))
        else:
            c_norm = mcolors.Normalize(vmin=0.0, vmax=h_max)
    c_cmap = matplotlib.colormaps[cmap].with_extremes(bad="white")

    fig, ax = get_fig_ax(ax, figsize=figsize)
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    equal_ranges = np.isclose(x_edges[0], y_edges[0]) and np.isclose(
        x_edges[-1], y_edges[-1]
    )
    hist_img = ax.imshow(
        plot_hist,
        cmap=c_cmap,
        norm=c_norm,
        origin="lower",
        extent=extent,
        aspect="equal" if equal_ranges else "auto",
        interpolation="nearest",
    )

    if points is not None:
        _draw_feature_space_points(ax, points, point_labels, val_scale)

    if show_colour_bar:
        cbar = fig.colorbar(hist_img, ax=ax, shrink=0.85)
        cbar.set_label("Proportion of pixels" if normalise else "Pixel count")
    if x_label is None:
        x_label = _get_band_name(input_img, x_band)
    if y_label is None:
        y_label = _get_band_name(y_img if y_img is not None else input_img, y_band)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax


def _draw_feature_space_points(
    ax: Any,
    points: Union[Mapping[str, Tuple[float, float]], Sequence[Tuple[float, float]]],
    point_labels: Optional[Sequence[str]],
    val_scale: float,
) -> None:
    """Mark (and label) points, given in image units, on a 2D histogram."""
    if isinstance(points, Mapping):
        point_labels = [str(key) for key in points]
        pt_vals = list(points.values())
    else:
        pt_vals = list(points)
    if point_labels is not None and len(point_labels) != len(pt_vals):
        raise rsgispy.RSGISPyException(
            f"{len(pt_vals)} point(s) but {len(point_labels)} point label(s)."
        )
    # Bright colours that stand out on viridis-like colour maps.
    pt_clrs = [
        OKABE_ITO_COLOURS["vermillion"],
        OKABE_ITO_COLOURS["sky_blue"],
        OKABE_ITO_COLOURS["reddish_purple"],
        OKABE_ITO_COLOURS["yellow"],
        OKABE_ITO_COLOURS["orange"],
        OKABE_ITO_COLOURS["bluish_green"],
    ]
    for idx, (pt_x, pt_y) in enumerate(pt_vals):
        x_val = float(pt_x) / val_scale
        y_val = float(pt_y) / val_scale
        ax.scatter(
            [x_val],
            [y_val],
            s=60,
            marker="o",
            color=pt_clrs[idx % len(pt_clrs)],
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
        if point_labels is not None:
            ax.annotate(
                str(point_labels[idx]),
                (x_val, y_val),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize="small",
                zorder=4,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )


def describe_shapiro(stat: float, p_val: float, alpha: float = 0.05) -> str:
    """
    Describe a Shapiro-Wilk test result in words: with ``p < alpha`` the
    values are unlikely to come from a normal distribution; otherwise the test
    gives no evidence against normality (it does not show they are normal).

    :param stat: Shapiro-Wilk ``W`` statistic.
    :param p_val: Shapiro-Wilk p-value.
    :param alpha: Significance level (Default = ``0.05``).
    :return: Two-line description, e.g. for a plot's text box.

    .. code:: python

        import rsgispy.tools.plotting as rplot

        print(rplot.describe_shapiro(0.95, 0.01))
    """
    verdict = (
        f"p < {alpha:g}: not normal"
        if p_val < alpha
        else f"p ≥ {alpha:g}: no evidence against normality"
    )
    return f"Shapiro–Wilk W = {stat:.3f}, p = {p_val:.3g}\n({verdict})"


def draw_qq(
    ax: Any,
    values: Union[np.ndarray, Sequence[float]],
    dist: Any = "norm",
    show_shapiro: bool = True,
) -> Dict[str, Any]:
    """
    Draw a Q-Q plot of ``values`` on ``ax`` (the points and fitted line of
    ``scipy.stats.probplot``, and optionally a Shapiro-Wilk text box) and
    return the numbers drawn: ``osm`` (theoretical quantiles), ``osr``
    (ordered values), ``slope``, ``intercept``, ``r`` and, when tested,
    ``shapiro_w`` and ``shapiro_p``. Used by :func:`plot_qq`; no labels or
    title are set.

    :param ax: Matplotlib axis.
    :param values: Finite values (flattened).
    :param dist: Distribution name or ``scipy.stats`` distribution object.
    :param show_shapiro: Add the Shapiro-Wilk test result (Default = ``True``).
    :return: Dictionary of the plotted values and statistics.

    .. code:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import rsgispy.tools.plotting as rplot

        fig, ax = plt.subplots()
        qq = rplot.draw_qq(ax, np.random.default_rng(0).normal(size=100))
    """
    # pylint: disable=import-outside-toplevel
    try:
        import scipy.stats
    except ImportError as exc:
        rsgispy.check_optional_dep(False, "plot_qq", "scipy", "stats", error_detail=str(exc))
        raise
    vals = np.asarray(values, dtype=float).ravel()
    (osm, osr), (slope, intercept, r_val) = scipy.stats.probplot(vals, dist=dist, fit=True)
    ax.scatter(osm, osr, s=16, alpha=0.7, color=CATEGORICAL_COLOURS[0], label="Values", zorder=3)
    line_x = np.array([np.min(osm), np.max(osm)])
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color=CATEGORICAL_COLOURS[1],
        linestyle="--",
        label="Fitted line",
    )
    out: Dict[str, Any] = {
        "osm": np.asarray(osm),
        "osr": np.asarray(osr),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_val),
    }
    if show_shapiro:
        if vals.size >= 3:
            sw_stat, sw_p = scipy.stats.shapiro(vals)
            out["shapiro_w"] = float(sw_stat)
            out["shapiro_p"] = float(sw_p)
            add_stats_text_box(ax, describe_shapiro(float(sw_stat), float(sw_p)), loc="upper left")
        else:
            add_stats_text_box(ax, "Shapiro–Wilk needs at least 3 values", loc="upper left")
    ax.legend(loc="lower right")
    return out


@use_plot_style
def plot_qq(
    values: Union[np.ndarray, Sequence[float]],
    *,
    dist: Any = "norm",
    show_shapiro: bool = True,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (5.5, 5.0),
    ax: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, Any]:
    """
    Plot a quantile-quantile (Q-Q) plot of ``values`` against a theoretical
    distribution (normal by default), with the fitted line from
    ``scipy.stats.probplot``. Points close to the line mean the values
    follow the distribution; curved tails show skew or heavy tails. With
    ``show_shapiro`` the Shapiro-Wilk test statistic and p-value are shown
    (normal distribution only); ``p < 0.05`` means the values are unlikely to
    be normal (:func:`describe_shapiro`).

    Use it for any values, e.g. the residuals of a regression or the k-fold
    residuals of :func:`rsgispy.regression.regsklearn.perform_kfold_fit`; see
    also :func:`rsgispy.regression.regmetrics.plot_qq_residuals`.

    :param values: Values to plot (flattened; non-finite values are ignored).
    :param dist: Distribution name or ``scipy.stats`` distribution object
                 (Default = ``'norm'``).
    :param show_shapiro: Show the Shapiro-Wilk test result (only when
                         ``dist='norm'``; Default = ``True``).
    :param title: Optional plot title.
    :param x_label: x-axis label (Default = ``'Theoretical quantiles (normal)'``
                    for the normal distribution).
    :param y_label: y-axis label (Default = ``'Ordered values'``).
    :param figsize: Size of a new figure in inches (Default = ``(5.5, 5.0)``).
    :param ax: Optional Matplotlib axis to draw on (Default = a new figure).
    :param out_plot_file: Optional output image file; the figure is saved and
                          left open (Default = ``None``).
    :param out_dpi: Resolution of the saved image (Default = ``300``).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import numpy as np
        import rsgispy.tools.plotting as rplot

        resid = np.random.default_rng(0).normal(0.0, 2.0, 200)
        fig, ax = rplot.plot_qq(resid, title="Residuals", out_plot_file="qq.png")
    """
    # pylint: disable=too-many-arguments
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        raise rsgispy.RSGISPyException("At least two finite values are needed for a Q-Q plot.")
    fig, ax = get_fig_ax(ax, figsize=figsize)
    dist_name = dist if isinstance(dist, str) else getattr(dist, "name", "distribution")
    draw_qq(ax, vals, dist=dist, show_shapiro=show_shapiro and dist_name == "norm")
    dist_lbl = "normal" if dist_name == "norm" else dist_name
    ax.set_xlabel(x_label if x_label is not None else f"Theoretical quantiles ({dist_lbl})")
    ax.set_ylabel(y_label if y_label is not None else "Ordered values")
    if title is not None:
        ax.set_title(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax


@use_plot_style
def plot_feature_importance_bars(
    features: Sequence[str],
    importances: Sequence[float],
    errors: Optional[Sequence[float]] = None,
    title: str = "Feature Importance",
    xlabel: str = "Importance",
    *,
    ax: Optional[Any] = None,
    out_plot_file: Optional[Union[str, os.PathLike]] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, Any]:
    """
    Plot a horizontal bar chart of feature importances, sorted so the most
    important feature is at the top, with optional error bars (e.g. the
    standard deviation of permutation importances).

    :param features: Feature names.
    :param importances: Importance of each feature.
    :param errors: Optional error of each importance, drawn as error bars.
    :param title: Plot title (Default = ``'Feature Importance'``).
    :param xlabel: x-axis label (Default = ``'Importance'``).
    :param ax: Optional Matplotlib axis to draw on.
    :param out_plot_file: Optional output image file.
    :param out_dpi: Resolution of the saved image (Default = 300).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import rsgispy.tools.plotting

        fig, ax = rsgispy.tools.plotting.plot_feature_importance_bars(
            ["NDVI", "NIR", "Red"], [0.5, 0.3, 0.2], title="Random forest"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    feat_list = list(features)
    val_arr = np.asarray(importances, dtype=float)
    err_arr = np.asarray(errors, dtype=float) if errors is not None else None
    order = np.argsort(val_arr)
    sorted_feats = [feat_list[i] for i in order]
    sorted_vals = val_arr[order]
    sorted_errs = err_arr[order] if err_arr is not None else None

    plt_height = max(3.0, (0.45 * len(sorted_feats)) + 1.5)
    fig, ax = get_fig_ax(ax, figsize=(8, plt_height))
    y_pos = np.arange(len(sorted_feats))
    ax.barh(
        y_pos,
        sorted_vals,
        xerr=sorted_errs,
        color=CATEGORICAL_COLOURS[0],
        ecolor="#262626",
        capsize=3,
        height=0.7,
    )
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax
