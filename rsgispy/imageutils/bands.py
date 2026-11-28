"""
The imageutils.bands module provides raster band selection, band stacking, and
Shepherd & Dymond local linear band sharpening using ``rasterio`` and ``numpy``.
"""

from contextlib import ExitStack
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    create_output_from_dataset,
    ensure_file_exists,
    fits_dtype_range,
    iter_dataset_windows,
    normalise_creation_options,
    open_multi_image_datasets,
    open_output_dataset,
    pop_stats_on_close,
    read_window_with_halo,
    rsgis_dtype_from_np_dtype,
    validate_band_index,
    validate_crs_and_res,
)

__all__ = [
    "select_img_bands",
    "stack_img_bands",
    "SharpBandInfo",
    "sharpen_low_res_bands",
]


class SharpBandInfo:  # pylint: disable=too-few-public-methods
    """
    Configuration record describing an input image band for
    :func:`sharpen_low_res_bands`.

    :param band: 1-based band index within the input image.
    :param status: Band role constant: :data:`rsgispy.SHARP_RES_IGNORE`,
                   :data:`rsgispy.SHARP_RES_LOW`, or
                   :data:`rsgispy.SHARP_RES_HIGH`.
    :param name: Optional output band name.
    """

    def __init__(
        self,
        band: Optional[int] = None,
        status: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.band = band
        self.status = status
        self.name = name

    def __repr__(self) -> str:
        return (
            f"SharpBandInfo(band={self.band!r}, status={self.status!r}, "
            f"name={self.name!r})"
        )


def select_img_bands(
    input_img: str,
    output_img: str,
    bands: Sequence[int],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Copy selected 1-based ``bands`` from ``input_img`` into a new raster
    ``output_img``.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param bands: Sequence of 1-based band indices to extract in order.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to matching ``input_img``).
    :param no_data_val: Optional output no-data value (defaults to ``input_img``
                        no-data value when ``None``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        rsgispy.imageutils.bands.select_img_bands(
            "in_img.tif", "out_img.tif", [1, 3, 5],
            gdalformat="GTIFF", datatype=rsgispy.TYPE_16UINT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if not bands:
        raise rsgispy.RSGISPyException(
            "bands must contain at least one 1-based band index."
        )
    band_list = [int(b) for b in bands]
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        for bidx in band_list:
            validate_band_index(bidx, src.count, input_img)

        eff_dtype = (
            rsgis_dtype_from_np_dtype(src.dtypes[0])
            if datatype is None
            else datatype
        )
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            count=len(band_list),
            creation_opts=norm_opts,
        )
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        if eff_nd is not None:
            dst.nodata = eff_nd
        pop_stats_on_close(dst, calc_stats, no_data_val=eff_nd, use_pbar=use_pbar)
        for out_idx, src_idx in enumerate(band_list, start=1):
            desc = src.descriptions[src_idx - 1]
            if desc:
                dst.set_band_description(out_idx, desc)

        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            blk = np.stack(
                [src.read(bidx, window=win) for bidx in band_list],
                axis=0,
            )
            dst.write(
                cast_result_block(
                    blk, int(win.height), int(win.width), out_np_dtype
                ),
                window=win,
            )


def stack_img_bands(
    input_imgs: Sequence[str],
    output_img: str,
    band_names: Optional[Sequence[str]] = None,
    skip_value: Optional[float] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Stack all bands from ``input_imgs`` over their spatial intersection into a
    single multi-band output image.

    If ``skip_value`` is provided, any pixel where at least one input band
    equals ``skip_value`` has all output bands at that pixel set to
    ``no_data_val``.

    :param input_imgs: Sequence of input image file paths.
    :param output_img: Path to the output stacked image file.
    :param band_names: Optional sequence of output band names (Default =
                       ``None``: the input band descriptions, as
                       :func:`select_img_bands` does).
    :param skip_value: Optional pixel value treated as invalid across inputs
                       (Default = ``None``).
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (Default = ``None``: the smallest type that holds every
                     input band, ``numpy.result_type`` of their dtypes, e.g.
                     ``uint8`` and ``float32`` inputs give ``float32``). A
                     narrower integer type rounds and clamps the values.
    :param no_data_val: Optional no-data value written where pixels are skipped
                        and set on the output dataset (Default = ``None``: the
                        inputs' no-data value when they all share one that the
                        output type can hold, otherwise no no-data value).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        rsgispy.imageutils.bands.stack_img_bands(
            ["b1.tif", "b2.tif", "b3.tif"],
            "stacked.tif",
            band_names=["Blue", "Green", "Red"],
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_16UINT,
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    skip_is_nan = skip_value is not None and math.isnan(float(skip_value))
    norm_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, input_imgs)
        first_ds = datasets[paths[0]]
        validate_crs_and_res(datasets, first_ds)
        grid = compute_spatial_intersection(datasets)

        if datatype is None:
            common_np_dtype = np.result_type(
                *[np.dtype(dt) for p in paths for dt in datasets[p].dtypes]
            )
            eff_dtype = rsgis_dtype_from_np_dtype(
                common_np_dtype, fallback=rsgispy.TYPE_64FLOAT
            )
        else:
            eff_dtype = datatype
        out_rio_dtype = rsgispy.get_rasterio_datatype(eff_dtype)
        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))

        if no_data_val is None:
            in_nds = [datasets[p].nodata for p in paths]
            first_nd = in_nds[0]
            same_nd = first_nd is not None and all(
                nd is not None
                and (nd == first_nd or (math.isnan(nd) and math.isnan(first_nd)))
                for nd in in_nds
            )
            if same_nd and (
                np.issubdtype(out_np_dtype, np.floating)
                or fits_dtype_range(first_nd, out_np_dtype)
            ):
                no_data_val = float(first_nd)

        total_bands = sum(int(datasets[p].count) for p in paths)
        if band_names is None:
            in_descs = [
                desc for p in paths for desc in datasets[p].descriptions
            ]
            if any(in_descs):
                band_names = [desc or "" for desc in in_descs]
        if band_names is not None and len(band_names) != total_bands:
            raise rsgispy.RSGISPyException(
                f"Length of band_names ({len(band_names)}) must match total "
                f"stacked band count ({total_bands})."
            )

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=total_bands,
            output_exists=False,
            creation_opts=norm_opts,
        )
        if no_data_val is not None:
            dst.nodata = no_data_val
        pop_stats_on_close(
            dst, calc_stats, no_data_val=no_data_val, use_pbar=use_pbar
        )
        if band_names is not None:
            for idx, bname in enumerate(band_names, start=1):
                dst.set_band_description(idx, str(bname))

        for dst_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            col_off, row_off = int(dst_win.col_off), int(dst_win.row_off)
            win_w, win_h = int(dst_win.width), int(dst_win.height)
            band_arrays = []
            for path in paths:
                ds = datasets[path]
                off_x, off_y = grid.offsets[path]
                src_win = Window(
                    off_x + col_off, off_y + row_off, win_w, win_h
                )
                band_arrays.append(ds.read(window=src_win))

            stacked = np.concatenate(band_arrays, axis=0)
            if skip_value is not None:
                if skip_is_nan:
                    skip_mask = np.isnan(stacked).any(axis=0)
                else:
                    skip_mask = (stacked == skip_value).any(axis=0)
                if np.any(skip_mask):
                    fill_val = 0.0 if no_data_val is None else no_data_val
                    stacked = stacked.copy()
                    stacked[:, skip_mask] = fill_val

            dst.write(
                cast_result_block(
                    stacked, win_h, win_w, out_np_dtype, nodata=no_data_val
                ),
                window=dst_win,
            )


def _sharpen_single_high_band(
    high_halo: np.ndarray,
    low_halo: np.ndarray,
    valid_halo: np.ndarray,
    win_size: int,
    no_data_val: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """Vectorised Shepherd & Dymond linear sharpening for one high-res band."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    half = win_size // 2
    win_h = high_halo.shape[0] - (2 * half)
    win_w = high_halo.shape[1] - (2 * half)

    h_centre = high_halo[half : half + win_h, half : half + win_w]
    l_centre = low_halo[half : half + win_h, half : half + win_w]
    v_centre = valid_halo[half : half + win_h, half : half + win_w]

    h_wins = np.lib.stride_tricks.sliding_window_view(
        high_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    l_wins = np.lib.stride_tricks.sliding_window_view(
        low_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    v_wins = np.lib.stride_tricks.sliding_window_view(
        valid_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)

    v_f = v_wins.astype(np.float64)
    n_valid = v_f.sum(axis=2)
    h_clean = np.where(v_wins, h_wins, 0.0)
    l_clean = np.where(v_wins, l_wins, 0.0)

    sum_h = h_clean.sum(axis=2)
    sum_l = l_clean.sum(axis=2)
    mean_h = np.divide(sum_h, n_valid, out=np.zeros_like(sum_h), where=n_valid > 0)
    mean_l = np.divide(sum_l, n_valid, out=np.zeros_like(sum_l), where=n_valid > 0)

    dh = np.where(v_wins, h_clean - mean_h[..., None], 0.0)
    dl = np.where(v_wins, l_clean - mean_l[..., None], 0.0)
    s_hh = (dh * dh).sum(axis=2)
    s_hl = (dh * dl).sum(axis=2)

    can_fit = v_centre & (n_valid >= 3) & (s_hh > 1e-12)
    slope = np.divide(
        s_hl, s_hh, out=np.zeros_like(s_hh), where=can_fit
    )
    pred = mean_l + slope * (h_centre - mean_h)
    pred = np.clip(pred, band_min, band_max)

    out = np.where(v_centre, l_centre, float(no_data_val))
    out = np.where(can_fit, pred, out)
    return out


def _sharpen_multi_high_bands(
    high_halos: Sequence[np.ndarray],
    low_halo: np.ndarray,
    valid_halo: np.ndarray,
    win_size: int,
    no_data_val: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """Vectorised local least-squares sharpening for multiple high-res bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    half = win_size // 2
    win_h = low_halo.shape[0] - (2 * half)
    win_w = low_halo.shape[1] - (2 * half)
    n_high = len(high_halos)

    h_centres = np.stack(
        [arr[half : half + win_h, half : half + win_w] for arr in high_halos],
        axis=2,
    )
    l_centre = low_halo[half : half + win_h, half : half + win_w]
    v_centre = valid_halo[half : half + win_h, half : half + win_w]

    h_wins = np.stack(
        [
            np.lib.stride_tricks.sliding_window_view(
                arr, (win_size, win_size)
            ).reshape(win_h, win_w, -1)
            for arr in high_halos
        ],
        axis=3,
    )
    l_wins = np.lib.stride_tricks.sliding_window_view(
        low_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)
    v_wins = np.lib.stride_tricks.sliding_window_view(
        valid_halo, (win_size, win_size)
    ).reshape(win_h, win_w, -1)

    out = np.where(v_centre, l_centre, float(no_data_val))
    n_valid = v_wins.sum(axis=2)
    fit_mask = v_centre & (n_valid >= n_high + 2)
    fit_r, fit_c = np.where(fit_mask)
    n_fit = fit_r.size
    if n_fit == 0:
        return out

    chunk_size = 8192
    for start in range(0, n_fit, chunk_size):
        end = min(start + chunk_size, n_fit)
        rr = fit_r[start:end]
        cc = fit_c[start:end]
        vm = v_wins[rr, cc]  # (C, K)
        cnt = vm.sum(axis=1, dtype=np.float64)  # (C,)
        hv = np.where(vm[..., None], h_wins[rr, cc], 0.0)  # (C, K, M)
        lv = np.where(vm, l_wins[rr, cc], 0.0)  # (C, K)

        mean_h = hv.sum(axis=1) / cnt[:, None]  # (C, M)
        mean_l = lv.sum(axis=1) / cnt  # (C,)

        dh = np.where(vm[..., None], hv - mean_h[:, None, :], 0.0)
        dl = np.where(vm, lv - mean_l[:, None], 0.0)

        gram = np.einsum("nki,nkj->nij", dh, dh)
        rhs = np.einsum("nki,nk->ni", dh, dl)

        eigvals, eigvecs = np.linalg.eigh(gram)
        tol = np.maximum(eigvals[:, -1:] * 1e-12, 1e-12)
        valid_ev = eigvals > tol
        safe_ev = np.where(valid_ev, eigvals, 1.0)
        inv_ev = np.where(valid_ev, 1.0 / safe_ev, 0.0)

        vt_rhs = np.einsum("nji,nj->ni", eigvecs, rhs)
        coefs = np.einsum("nij,nj->ni", eigvecs, inv_ev * vt_rhs)

        pred = mean_l + np.einsum(
            "ni,ni->n", coefs, h_centres[rr, cc] - mean_h
        )
        out[rr, cc] = np.clip(pred, band_min, band_max)

    return out


def _calc_low_band_bounds(
    src: rasterio.DatasetReader,
    low_bands: Sequence[int],
    no_data_val: float,
) -> Dict[int, List[float]]:
    """Compute valid min/max bounds for each low-resolution band."""
    low_bounds: Dict[int, List[float]] = {}
    for l_band in low_bands:
        arr = src.read(l_band, out_dtype=np.float64)
        vmask = build_valid_mask(arr, no_data_val=no_data_val)
        if np.any(vmask):
            low_bounds[l_band] = [
                float(arr[vmask].min()),
                float(arr[vmask].max()),
            ]
        else:
            low_bounds[l_band] = [0.0, 0.0]
    return low_bounds


def sharpen_low_res_bands(
    input_img: str,
    output_img: str,
    band_info: Sequence[SharpBandInfo],
    win_size: int = 7,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Sharpen lower-resolution image bands using local linear fitting against
    higher-resolution bands (Shepherd & Dymond method).

    :param input_img: Path to the input multi-band image (with low-res bands
                      already resampled to the high-res pixel grid).
    :param output_img: Path to the output sharpened image file.
    :param band_info: Sequence of :class:`SharpBandInfo` describing each band's
                      role (``SHARP_RES_LOW``, ``SHARP_RES_HIGH``, or
                      ``SHARP_RES_IGNORE``).
    :param win_size: Odd local regression window size in pixels (Default = ``7``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: No-data value ignored during regression fitting
                        (Default = ``0.0``; if ``None``, uses input nodata or ``0.0``).
    :param calc_stats: If ``True``, compute image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy
        import rsgispy.imageutils.bands

        band_info = [
            rsgispy.imageutils.bands.SharpBandInfo(1, rsgispy.SHARP_RES_HIGH, "B2"),
            rsgispy.imageutils.bands.SharpBandInfo(2, rsgispy.SHARP_RES_LOW, "B5"),
        ]
        rsgispy.imageutils.bands.sharpen_low_res_bands(
            "in_img.tif", "sharp_img.tif", band_info
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    ensure_file_exists(input_img)
    if win_size < 3 or win_size % 2 == 0:
        raise rsgispy.RSGISPyException(
            "win_size must be an odd integer greater than or equal to 3."
        )
    if not band_info:
        raise rsgispy.RSGISPyException("band_info must not be empty.")

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else datatype
    norm_opts = normalise_creation_options(creation_opts)

    high_bands: List[int] = []
    low_bands: List[int] = []
    valid_statuses = (
        rsgispy.SHARP_RES_IGNORE,
        rsgispy.SHARP_RES_LOW,
        rsgispy.SHARP_RES_HIGH,
    )
    half = win_size // 2

    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        eff_nd = (
            float(no_data_val)
            if no_data_val is not None
            else (float(src.nodata) if src.nodata is not None else 0.0)
        )

        for binfo in band_info:
            if binfo.band is None or binfo.status not in valid_statuses:
                raise rsgispy.RSGISPyException(
                    f"Invalid SharpBandInfo entry: {binfo!r}."
                )
            validate_band_index(int(binfo.band), src.count, input_img)
            if binfo.status == rsgispy.SHARP_RES_HIGH:
                high_bands.append(int(binfo.band))
            elif binfo.status == rsgispy.SHARP_RES_LOW:
                low_bands.append(int(binfo.band))

        if not high_bands or not low_bands:
            raise rsgispy.RSGISPyException(
                "band_info must contain at least one SHARP_RES_HIGH band and "
                "one SHARP_RES_LOW band."
            )

        low_bounds = _calc_low_band_bounds(src, low_bands, eff_nd)
        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(eff_dtype),
            count=len(band_info),
            creation_opts=norm_opts,
        )
        dst.nodata = eff_nd
        pop_stats_on_close(dst, calc_stats, no_data_val=eff_nd, use_pbar=use_pbar)
        for out_idx, binfo in enumerate(band_info, start=1):
            if binfo.name:
                dst.set_band_description(out_idx, str(binfo.name))

        out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
        for dst_win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            high_halos = [
                read_window_with_halo(src, hb, dst_win, half)
                for hb in high_bands
            ]
            high_valid = np.ones_like(high_halos[0], dtype=bool)
            for h_arr in high_halos:
                high_valid &= build_valid_mask(h_arr, no_data_val=eff_nd)

            out_bands = []
            for binfo in band_info:
                src_b = int(binfo.band)
                if binfo.status == rsgispy.SHARP_RES_LOW:
                    l_halo = read_window_with_halo(src, src_b, dst_win, half)
                    l_valid = high_valid & build_valid_mask(
                        l_halo, no_data_val=eff_nd
                    )
                    bmin, bmax = low_bounds[src_b]
                    if len(high_halos) == 1:
                        sharpened = _sharpen_single_high_band(
                            high_halos[0],
                            l_halo,
                            l_valid,
                            win_size,
                            eff_nd,
                            bmin,
                            bmax,
                        )
                    else:
                        sharpened = _sharpen_multi_high_bands(
                            high_halos,
                            l_halo,
                            l_valid,
                            win_size,
                            eff_nd,
                            bmin,
                            bmax,
                        )
                    out_bands.append(sharpened)
                else:
                    out_bands.append(
                        src.read(src_b, window=dst_win, out_dtype=np.float64)
                    )

            dst.write(
                cast_result_block(
                    np.stack(out_bands, axis=0),
                    int(dst_win.height),
                    int(dst_win.width),
                    out_np_dtype,
                ),
                window=dst_win,
            )
