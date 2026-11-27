"""
The imagespectral.spectra module provides spectrum analysis and plotting:
continuum removal (convex hull quotient) and absorption feature parameters
for a spectrum or for every pixel of an image, and plots of spectra and of
a continuum removal.
"""

from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy._numba import njit, optional_njit
from rsgispy.imageprocess import (
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_output_dataset,
)
from rsgispy.tools.plotting import (
    CATEGORICAL_COLOURS,
    PLOT_DPI,
    REF_LINE_COLOUR,
    get_fig_ax,
    get_fig_axs,
    save_plot,
    use_plot_style,
)

__all__: List[str] = [
    "calc_continuum_removal",
    "calc_continuum_removal_img",
    "plot_continuum_removal",
    "plot_spectra",
]

SpectraInput = Union[pd.DataFrame, Mapping[str, Sequence[float]], np.ndarray, Sequence[float]]

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _check_wavelengths(wvls: np.ndarray) -> None:
    """Raise if ``wvls`` is not 1D and strictly increasing."""
    if wvls.ndim != 1 or wvls.size < 2:
        raise rsgispy.RSGISPyException("At least two wavelengths are needed.")
    if np.any(np.diff(wvls) <= 0):
        raise rsgispy.RSGISPyException("The wavelengths must be strictly increasing.")


def _wvl_range_mask(
    wvls: np.ndarray, start_wvl: Optional[float], stop_wvl: Optional[float]
) -> np.ndarray:
    """Boolean mask of the wavelengths within ``[start_wvl, stop_wvl]``."""
    keep = np.ones(wvls.shape, dtype=bool)
    if start_wvl is not None:
        keep &= wvls >= float(start_wvl)
    if stop_wvl is not None:
        keep &= wvls <= float(stop_wvl)
    if np.count_nonzero(keep) < 2:
        raise rsgispy.RSGISPyException(
            f"Fewer than two wavelengths between {start_wvl} and {stop_wvl}."
        )
    return keep


def _hull_cr_1d(
    spec: np.ndarray, wvls: np.ndarray, cr_out: np.ndarray, hull_idxs: np.ndarray
) -> int:
    """
    Upper convex hull of the points ``(wvls, spec)`` (monotone chain; points
    on a hull segment are not hull points) and the continuum removed
    spectrum ``spec / hull`` (1 where the hull is 0), written to ``cr_out``.
    The hull point indices are written to the start of ``hull_idxs`` and
    their number is returned. Written for numba (no Python objects).
    """
    # pylint: disable=too-many-locals
    n_hull = 0
    for i in range(spec.shape[0]):
        while n_hull >= 2:
            i_1 = hull_idxs[n_hull - 2]
            i_2 = hull_idxs[n_hull - 1]
            cross = (wvls[i_2] - wvls[i_1]) * (spec[i] - spec[i_1]) - (
                spec[i_2] - spec[i_1]
            ) * (wvls[i] - wvls[i_1])
            if cross >= 0:
                n_hull -= 1
            else:
                break
        hull_idxs[n_hull] = i
        n_hull += 1
    for k in range(n_hull - 1):
        s_idx = hull_idxs[k]
        e_idx = hull_idxs[k + 1]
        x_0 = wvls[s_idx]
        y_0 = spec[s_idx]
        slope = (spec[e_idx] - y_0) / (wvls[e_idx] - x_0)
        for j in range(s_idx, e_idx + 1):
            hull_val = y_0 + slope * (wvls[j] - x_0)
            if hull_val != 0:
                cr_out[j] = spec[j] / hull_val
            else:
                cr_out[j] = 1.0
    return n_hull


def _make_cr_pixels(hull_fn: Callable[..., int]) -> Callable[..., np.ndarray]:
    """The pixel loop of the image continuum removal, calling ``hull_fn``."""

    def _cr_pixels(
        data: np.ndarray, wvls: np.ndarray, valid: np.ndarray, no_data_val: float
    ) -> np.ndarray:
        """
        Continuum removed spectrum of each pixel: ``data`` is
        ``(n_pix, n_bands)`` (float64), ``valid`` ``(n_pix,)``; invalid
        pixels get ``no_data_val``.
        """
        n_pix, n_bands = data.shape
        out = np.empty((n_pix, n_bands), dtype=np.float64)
        hull_idxs = np.empty(n_bands, dtype=np.int64)
        cr_vals = np.empty(n_bands, dtype=np.float64)
        for p_idx in range(n_pix):
            if not valid[p_idx]:
                for b_idx in range(n_bands):
                    out[p_idx, b_idx] = no_data_val
                continue
            hull_fn(data[p_idx], wvls, cr_vals, hull_idxs)
            for b_idx in range(n_bands):
                out[p_idx, b_idx] = cr_vals[b_idx]
        return out

    return _cr_pixels


# The compiled kernel calls a compiled hull function; the fallback (used
# without numba, or within disable_numba()) is the same loop in pure Python.
_cr_pixels_kernel = optional_njit(
    _make_cr_pixels(njit(nogil=True)(_hull_cr_1d)),
    nogil=True,
    cache=False,
    fallback=_make_cr_pixels(_hull_cr_1d),
    fallback_desc="a pure-Python loop over the pixels",
)


def calc_continuum_removal(
    spectrum: Sequence[float],
    wavelengths: Sequence[float],
    start_wvl: Optional[float] = None,
    stop_wvl: Optional[float] = None,
    baseline: float = 1.0,
) -> Dict[str, Any]:
    """
    Continuum removal (convex hull quotient) of one spectrum, and the
    parameters of its absorption features.

    The continuum is the upper convex hull of the spectrum; the continuum
    removed spectrum (CRS) is the spectrum divided by the continuum, so it is
    1 where the spectrum touches the hull and below 1 in absorption
    features. Each pair of neighbouring hull points bounds a possible
    feature; a feature is reported when its minimum CRS is below
    ``baseline``.

    :param spectrum: The spectrum values (e.g. reflectance).
    :param wavelengths: The wavelength of each value (strictly increasing).
    :param start_wvl: Optional first wavelength of the analysis (the
                      wavelengths ``>= start_wvl`` are used).
    :param stop_wvl: Optional last wavelength of the analysis (the
                     wavelengths ``<= stop_wvl`` are used).
    :param baseline: Features whose minimum CRS is below this value are
                     reported (Default = 1.0, i.e. any absorption).
    :return: Dictionary with ``'wvl'`` and ``'spectrum'`` (the values within
             the wavelength range), ``'crs'`` (the continuum removed
             spectrum), ``'continuum'`` (the hull at each wavelength),
             ``'hull_x'`` and ``'hull_y'`` (the hull points), and
             ``'features'``: a list with a dictionary per absorption feature
             with ``'id'`` (from 1), ``'wvl'`` and ``'crs'`` (its values),
             ``'cstart_wvl'`` and ``'cstop_wvl'`` (the hull points bounding
             it), ``'abs_wvl'`` (wavelength of the minimum CRS),
             ``'abs_depth'`` (the minimum CRS, so the depth is
             ``1 - abs_depth``), ``'area'`` (area between the CRS and 1),
             ``'cslope'`` (slope of the continuum), ``'FWHM_x'``
             (wavelengths of the full width at half maximum), ``'FWHM_y'``
             (its CRS level) and ``'FWHM_delta'`` (the width).

    .. code:: python

        import pandas
        import rsgispy.imagespectral.spectra as spectra

        spec_df = pandas.read_csv("spectra.txt", sep="\\t")
        cr_info = spectra.calc_continuum_removal(
            spec_df["Chl 40"] / 100, spec_df["WL"], start_wvl=530, stop_wvl=750
        )
        feat = cr_info["features"][0]
        print(feat["abs_wvl"], 1 - feat["abs_depth"], feat["area"])
    """
    # pylint: disable=too-many-locals
    spec = np.asarray(spectrum, dtype=np.float64).ravel()
    wvls = np.asarray(wavelengths, dtype=np.float64).ravel()
    if spec.size != wvls.size:
        raise rsgispy.RSGISPyException(
            f"spectrum ({spec.size} values) and wavelengths ({wvls.size} values) "
            "must be the same length."
        )
    _check_wavelengths(wvls)
    if not np.all(np.isfinite(spec)):
        raise rsgispy.RSGISPyException("The spectrum must not contain NaN or inf.")
    keep = _wvl_range_mask(wvls, start_wvl, stop_wvl)
    spec = spec[keep]
    wvls = wvls[keep]

    crs = np.empty(spec.size, dtype=np.float64)
    hull_idxs = np.empty(spec.size, dtype=np.int64)
    n_hull = _hull_cr_1d(spec, wvls, crs, hull_idxs)
    hull_idxs = hull_idxs[:n_hull]
    hull_x = wvls[hull_idxs]
    hull_y = spec[hull_idxs]

    features: List[Dict[str, Any]] = []
    for n_idx in range(hull_idxs.size - 1):
        s_idx, e_idx = int(hull_idxs[n_idx]), int(hull_idxs[n_idx + 1])
        f_wvl = wvls[s_idx:e_idx + 1]
        f_crs = crs[s_idx:e_idx + 1]
        if f_crs.min() >= baseline:
            continue
        min_idx = int(np.argmin(f_crs))
        abs_depth = float(f_crs[min_idx])
        half_y = abs_depth + (1.0 - abs_depth) / 2.0
        # Walk out from the minimum to the first values at or above half_y.
        left = np.flatnonzero(f_crs[:min_idx + 1] >= half_y)
        right = np.flatnonzero(f_crs[min_idx:] >= half_y)
        fwhm_x1 = float(f_wvl[left[-1]]) if left.size else float(f_wvl[0])
        fwhm_x2 = float(f_wvl[min_idx + right[0]]) if right.size else float(f_wvl[-1])
        features.append(
            {
                "id": len(features) + 1,
                "wvl": f_wvl,
                "crs": f_crs,
                "cstart_wvl": float(f_wvl[0]),
                "cstop_wvl": float(f_wvl[-1]),
                "abs_wvl": float(f_wvl[min_idx]),
                "abs_depth": abs_depth,
                "area": float(_trapezoid(np.abs(f_crs - 1.0), x=f_wvl)),
                "cslope": float(
                    (hull_y[n_idx + 1] - hull_y[n_idx])
                    / (hull_x[n_idx + 1] - hull_x[n_idx])
                ),
                "FWHM_x": (fwhm_x1, fwhm_x2),
                "FWHM_y": (half_y, half_y),
                "FWHM_delta": fwhm_x2 - fwhm_x1,
            }
        )
    return {
        "wvl": wvls,
        "spectrum": spec,
        "crs": crs,
        "continuum": np.interp(wvls, hull_x, hull_y),
        "hull_x": hull_x,
        "hull_y": hull_y,
        "features": features,
    }


def calc_continuum_removal_img(
    input_img: str,
    output_img: str,
    wavelengths: Sequence[float],
    start_wvl: Optional[float] = None,
    stop_wvl: Optional[float] = None,
    *,
    in_no_data_val: Optional[float] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: float = -9999.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Continuum removal (convex hull quotient) of the spectrum of every pixel
    of an image (see :func:`calc_continuum_removal`). The output has one band
    per input band within ``[start_wvl, stop_wvl]``, named by its wavelength,
    with values from 0 to 1 (1 on the continuum). Uses a ``numba`` kernel
    when available (a pure-Python loop, much slower, otherwise).

    :param input_img: Input image (e.g. reflectance), one band per
                      wavelength.
    :param output_img: Output image of continuum removed values.
    :param wavelengths: The wavelength of each input band (strictly
                        increasing).
    :param start_wvl: Optional first wavelength of the analysis.
    :param stop_wvl: Optional last wavelength of the analysis.
    :param in_no_data_val: No data value of the input (Default = read from
                           the image). Pixels with no data or a non-finite
                           value in any band of the range are no data in the
                           output.
    :param gdalformat: Output format (Default = ``'GTIFF'``).
    :param datatype: Output data type (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Output no data value, written to the file (Default =
                        -9999).
    :param use_pbar: Show a progress bar (Default = True).
    :param creation_opts: Optional GDAL creation options.

    .. code:: python

        import rsgispy.imagespectral.spectra as spectra

        spectra.calc_continuum_removal_img(
            "hymap_refl.tif", "hymap_cr_550_750.tif", wavelengths=hymap_wvls,
            start_wvl=550, stop_wvl=750,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img)
    wvls = np.asarray(wavelengths, dtype=np.float64).ravel()
    _check_wavelengths(wvls)
    keep = _wvl_range_mask(wvls, start_wvl, stop_wvl)
    band_idxs = [int(i) + 1 for i in np.flatnonzero(keep)]
    sub_wvls = np.ascontiguousarray(wvls[keep])
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        if src.count != wvls.size:
            raise rsgispy.RSGISPyException(
                f"{wvls.size} wavelengths were given for an image with "
                f"{src.count} bands."
            )
        in_nd = src.nodata if in_no_data_val is None else float(in_no_data_val)
        grid = compute_spatial_intersection({input_img: src})
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(band_idxs),
            creation_opts=creation_opts,
        )
        dst.nodata = no_data_val
        for out_band, wvl in enumerate(sub_wvls, start=1):
            dst.set_band_description(out_band, f"{wvl:g}")

        for out_win in iter_dataset_windows(grid.height, grid.width, use_pbar=use_pbar):
            win_h, win_w = int(out_win.height), int(out_win.width)
            blk = src.read(band_idxs, window=out_win, out_dtype=np.float64)
            pix = np.ascontiguousarray(blk.reshape(len(band_idxs), -1).T)
            valid = np.all(np.isfinite(pix), axis=1)
            if in_nd is not None and not np.isnan(in_nd):
                valid &= ~np.any(pix == in_nd, axis=1)
            cr_pix = _cr_pixels_kernel(pix, sub_wvls, valid, float(no_data_val))
            out_blk = cr_pix.T.reshape(len(band_idxs), win_h, win_w)
            dst.write(
                np.stack(
                    [cast_result_block(b, win_h, win_w, out_np_dtype) for b in out_blk]
                ),
                window=out_win,
            )


@use_plot_style
def plot_continuum_removal(
    cr_info: Mapping[str, Any],
    *,
    show_features: bool = True,
    title: Optional[str] = None,
    x_label: str = "Wavelength (nm)",
    y_label: str = "Reflectance",
    figsize: Optional[Tuple[float, float]] = None,
    axs: Optional[Any] = None,
    out_plot_file: Optional[str] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, np.ndarray]:
    """
    Plot a continuum removal from :func:`calc_continuum_removal`: the
    spectrum with its continuum (hull) and the continuum removed spectrum
    and, in a second panel, the continuum removed spectrum with the
    absorption depth and full width at half maximum of each feature.

    :param cr_info: The dictionary returned by :func:`calc_continuum_removal`.
    :param show_features: Draw the second panel with the absorption features
                          (Default = True).
    :param title: Optional figure title (e.g. the name of the spectrum).
    :param x_label: x-axis label (Default = ``'Wavelength (nm)'``).
    :param y_label: y-axis label of the spectrum (Default =
                    ``'Reflectance'``).
    :param figsize: Size of a new figure in inches.
    :param axs: Optional axes to draw on (two, or one with
                ``show_features=False``).
    :param out_plot_file: Optional output image file; the figure is saved and
                          left open.
    :param out_dpi: Resolution of the saved image (Default = 300).
    :return: Tuple ``(fig, axs)`` with ``axs`` an array of one or two axes.

    .. code:: python

        import matplotlib.pyplot as plt
        import rsgispy.imagespectral.spectra as spectra

        cr_info = spectra.calc_continuum_removal(refl, wvls, 530, 750)
        fig, axs = spectra.plot_continuum_removal(
            cr_info, title="Chl 40", out_plot_file="chl_40_cr.png"
        )
        plt.close(fig)  # e.g. when plotting many spectra in a loop
    """
    # pylint: disable=too-many-arguments,too-many-locals
    n_axes = 2 if show_features else 1
    if figsize is None:
        figsize = (5.2 * n_axes, 4.0)
    fig, ax_grid = get_fig_axs(axs, nrows=1, ncols=n_axes, figsize=figsize)
    ax_arr = ax_grid.ravel()
    wvls = np.asarray(cr_info["wvl"])
    crs = np.asarray(cr_info["crs"])

    ax_spec = ax_arr[0]
    ax_spec.plot(wvls, cr_info["spectrum"], color=CATEGORICAL_COLOURS[0], label="Spectrum")
    ax_spec.plot(
        cr_info["hull_x"], cr_info["hull_y"], color=REF_LINE_COLOUR, linestyle="--",
        marker="o", markersize=3.5, linewidth=1.2, label="Continuum",
    )
    ax_spec.plot(wvls, crs, color=CATEGORICAL_COLOURS[2], label="Continuum removed")
    ax_spec.set_xlabel(x_label)
    ax_spec.set_ylabel(y_label)
    ax_spec.set_title("Spectrum and continuum")
    ax_spec.legend()

    if show_features:
        ax_feat = ax_arr[1]
        ax_feat.plot(wvls, crs, color=CATEGORICAL_COLOURS[2], label="Continuum removed")
        ax_feat.axhline(1.0, color=REF_LINE_COLOUR, linewidth=0.8, linestyle=":")
        for i, feat in enumerate(cr_info["features"]):
            ax_feat.plot(
                (feat["abs_wvl"], feat["abs_wvl"]), (feat["abs_depth"], 1.0),
                color=CATEGORICAL_COLOURS[1], label="Depth" if i == 0 else None,
            )
            ax_feat.plot(
                feat["FWHM_x"], feat["FWHM_y"], color=CATEGORICAL_COLOURS[0],
                marker="|", markersize=8, label="FWHM" if i == 0 else None,
            )
            ax_feat.annotate(
                str(feat["id"]), (feat["abs_wvl"], feat["abs_depth"]),
                textcoords="offset points", xytext=(5, 2), ha="left", va="bottom",
                fontsize=8,
            )
        ax_feat.set_xlabel(x_label)
        ax_feat.set_ylabel("Continuum removed")
        ax_feat.set_title("Absorption features")
        ax_feat.set_ylim(bottom=max(0.0, float(np.min(crs)) - 0.08))
        ax_feat.legend(loc="lower left")
    if title is not None:
        fig.suptitle(title)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax_arr


def _spectra_table(
    spectra: SpectraInput,
    wavelengths: Optional[Union[str, Sequence[float]]],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """The x values and a DataFrame with one column per spectrum."""
    if isinstance(spectra, pd.DataFrame):
        spec_df = spectra
        if isinstance(wavelengths, str):
            x_vals = spec_df[wavelengths].to_numpy()
            spec_df = spec_df.drop(columns=[wavelengths])
            wavelengths = None
        else:
            x_vals = spec_df.index.to_numpy()
    elif isinstance(spectra, Mapping):
        spec_df = pd.DataFrame({str(k): np.asarray(v, dtype=float) for k, v in spectra.items()})
        x_vals = np.arange(1, len(spec_df) + 1)
    else:
        arr = np.asarray(spectra, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim != 2:
            raise rsgispy.RSGISPyException("spectra must be 1D or 2D (values x spectra).")
        spec_df = pd.DataFrame(arr, columns=[f"Spectrum {i + 1}" for i in range(arr.shape[1])])
        x_vals = np.arange(1, arr.shape[0] + 1)
    if wavelengths is not None and not isinstance(wavelengths, str):
        x_vals = np.asarray(wavelengths)
    if len(x_vals) != len(spec_df):
        raise rsgispy.RSGISPyException(
            f"{len(x_vals)} wavelengths were given for spectra of {len(spec_df)} values."
        )
    return x_vals, spec_df


@use_plot_style
def plot_spectra(
    spectra: SpectraInput,
    wavelengths: Optional[Union[str, Sequence[float]]] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    colours: Optional[Sequence[Any]] = None,
    cmap_name: Optional[str] = None,
    gain: float = 1.0,
    title: Optional[str] = None,
    x_label: str = "Wavelength (nm)",
    y_label: str = "Reflectance",
    legend_title: Optional[str] = None,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.5),
    ax: Optional[Any] = None,
    out_plot_file: Optional[str] = None,
    out_dpi: int = PLOT_DPI,
) -> Tuple[Any, Any]:
    """
    Plot one or more spectra (e.g. reflectance, pixel spectra or sensor
    response functions) against wavelength.

    :param spectra: A :class:`pandas.DataFrame` with one column per spectrum
                    (the index gives the wavelengths unless ``wavelengths``
                    is given), a ``dict`` ``{name: values}``, or an array of
                    shape ``(n_values,)`` or ``(n_values, n_spectra)``.
    :param wavelengths: The wavelengths (x values), or the name of the
                        wavelength column of a DataFrame (Default = the
                        DataFrame index, or 1, 2, … for a dict or array).
    :param labels: Optional legend labels, one per spectrum (Default = the
                   column names or dict keys).
    :param colours: Optional colours, one per spectrum.
    :param cmap_name: Optional colour map to colour ordered spectra (e.g.
                      ``'viridis'`` for spectra of increasing chlorophyll).
                      Default: the house categorical colours for up to 8
                      spectra and ``'viridis'`` for more.
    :param gain: Divide the values by this (e.g. 100 for percent to
                 reflectance; Default = 1).
    :param title: Optional plot title.
    :param x_label: x-axis label (Default = ``'Wavelength (nm)'``).
    :param y_label: y-axis label (Default = ``'Reflectance'``).
    :param legend_title: Optional legend title.
    :param show_legend: Draw a legend (Default = True).
    :param figsize: Size of a new figure in inches (Default = ``(7.0, 4.5)``).
    :param ax: Optional axis to draw on.
    :param out_plot_file: Optional output image file; the figure is saved and
                          left open.
    :param out_dpi: Resolution of the saved image (Default = 300).
    :return: Tuple ``(fig, ax)``.

    .. code:: python

        import pandas
        import rsgispy.imagespectral.spectra as spectra

        spec_df = pandas.read_csv("PROSAIL_spectra.txt", sep="\\t")
        fig, ax = spectra.plot_spectra(
            spec_df, wavelengths="WL", y_label="Reflectance (%)",
            cmap_name="viridis", legend_title="Chlorophyll",
        )

        # Pixel spectra read with get_img_pxl_column
        fig, ax = spectra.plot_spectra(
            {"Original": orig_vals, "Smoothed": smth_vals}, wavelengths=wvls
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    x_vals, spec_df = _spectra_table(spectra, wavelengths)
    n_spec = spec_df.shape[1]
    if labels is None:
        labels = [str(c) for c in spec_df.columns]
    elif len(labels) != n_spec:
        raise rsgispy.RSGISPyException(
            f"{len(labels)} labels were given for {n_spec} spectra."
        )
    if colours is None:
        if cmap_name is None and n_spec <= len(CATEGORICAL_COLOURS):
            colours = CATEGORICAL_COLOURS[:n_spec]
        else:
            import matplotlib  # pylint: disable=import-outside-toplevel

            cmap = matplotlib.colormaps[cmap_name or "viridis"]
            colours = [cmap(v) for v in np.linspace(0.0, 0.9, n_spec)]
    elif len(colours) != n_spec:
        raise rsgispy.RSGISPyException(
            f"{len(colours)} colours were given for {n_spec} spectra."
        )

    fig, ax = get_fig_ax(ax, figsize=figsize)
    vals = spec_df.to_numpy(dtype=np.float64) / float(gain)
    for i in range(n_spec):
        ax.plot(x_vals, vals[:, i], color=colours[i], label=labels[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if show_legend:
        ax.legend(title=legend_title, ncols=2 if n_spec > 8 else 1)
    if out_plot_file is not None:
        save_plot(fig, out_plot_file, out_dpi=out_dpi)
    return fig, ax
