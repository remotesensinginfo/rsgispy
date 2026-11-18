"""
The imagecalc.rescale module provides gain/offset and per-band min-max raster
pixel rescaling utilities using ``rasterio``.
"""

from contextlib import ExitStack
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imagecalc._norm_helpers import (
    ALL_NORM_METHODS,
    DEFAULT_TRIM_METHODS,
    PIXEL_VECTOR_METHODS,
    _extract_user_bounds,
    apply_band_norm_1d,
    apply_pixel_vector_norm,
    compute_pixel_vector_norm_matrix,
    fit_band_norm_params,
    resolve_band_nodata,
    resolve_norm_method,
    resolve_out_bounds,
    resolve_out_nodata,
    subsample_band_values,
)
from rsgispy.imageprocess import (
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    iter_dataset_windows,
    open_multi_image_datasets,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageutils.metadata import get_band_names, set_band_names
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import read_json_to_dict, write_dict_to_json

__all__ = [
    "ALL_NORM_METHODS",
    "ImageBandRescale",
    "calc_img_rescale",
    "rescale_img_pxl_vals",
    "calc_img_norm_params",
    "apply_img_norm_params",
    "normalise_img_pxl_vals",
]


class ImageBandRescale:
    """
    Data container storing per-band min-max rescaling parameters for
    :func:`rescale_img_pxl_vals`.

    :param band: 1-based image band index.
    :param in_min: Input image band minimum value for rescaling.
    :param in_max: Input image band maximum value for rescaling.
    :param no_data_val: No-data value in the input image band.
    :param out_min: Output image band minimum value after rescaling.
    :param out_max: Output image band maximum value after rescaling.
    :param out_no_data: No-data value in the output image band.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods
    def __init__(
        self,
        band: int = 0,
        in_min: float = 0.0,
        in_max: float = 0.0,
        no_data_val: float = 0.0,
        out_min: float = 0.0,
        out_max: float = 0.0,
        out_no_data: float = 0.0,
    ) -> None:
        self.band = band
        self.in_min = in_min
        self.in_max = in_max
        self.no_data_val = no_data_val
        self.out_min = out_min
        self.out_max = out_max
        self.out_no_data = out_no_data

    def __str__(self) -> str:
        return (
            f"Band {self.band} [{self.in_min}, {self.in_max}, {self.no_data_val}] "
            f"[{self.out_min}, {self.out_max}, {self.out_no_data}]"
        )

    def __repr__(self) -> str:
        return self.__str__()


def calc_img_rescale(
    input_imgs: Union[str, Sequence[str]],
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    c_no_data_val: float = 0.0,
    c_offset: float = 0.0,
    c_gain: float = 1.0,
    n_no_data_val: float = 0.0,
    n_offset: float = 0.0,
    n_gain: float = 1.0,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rescale and stack one or more input images by converting from current
    ``(c_offset, c_gain)`` to new ``(n_offset, n_gain)`` scaling.

    Valid pixels are transformed via
    ``(((val - c_offset) / c_gain) * n_gain) + n_offset``, while pixels equal
    to ``c_no_data_val`` (or non-finite) are mapped to ``n_no_data_val``.

    :param input_imgs: Single input image path or sequence of image paths to stack.
    :param output_img: Path to the output stacked image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param c_no_data_val: Current no-data value in the input imagery.
    :param c_offset: Current offset value.
    :param c_gain: Current gain value (must be non-zero).
    :param n_no_data_val: New no-data value written to the output image.
    :param n_offset: New offset value.
    :param n_gain: New gain value.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale

        rsgispy.imagecalc.rescale.calc_img_rescale(
            input_imgs=["img_b1.tif", "img_b2.tif"],
            output_img="rescaled.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            c_no_data_val=0.0,
            c_offset=0.0,
            c_gain=10000.0,
            n_no_data_val=-9999.0,
            n_offset=0.0,
            n_gain=1.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if c_gain == 0.0:
        raise rsgispy.RSGISPyException("c_gain must not be zero.")

    img_list = [input_imgs] if isinstance(input_imgs, str) else list(input_imgs)
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)
        total_bands = sum(datasets[path].count for path in img_paths)

        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=total_bands,
            creation_opts=creation_opts,
        )
        dst.nodata = n_no_data_val

        scale = float(n_gain) / float(c_gain)
        c_off = float(c_offset)
        n_off = float(n_offset)
        c_nd_is_nan = np.isnan(c_no_data_val)

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            out_band_idx = 1
            for path in img_paths:
                ds = datasets[path]
                base_col, base_row = grid.offsets[path]
                src_win = Window(
                    base_col + int(out_win.col_off),
                    base_row + int(out_win.row_off),
                    win_w,
                    win_h,
                )
                in_block = ds.read(window=src_win, out_dtype=np.float64)
                for b_idx in range(ds.count):
                    band_arr = in_block[b_idx]
                    if c_nd_is_nan:
                        nodata_mask = ~np.isfinite(band_arr)
                    else:
                        nodata_mask = (band_arr == c_no_data_val) | ~np.isfinite(
                            band_arr
                        )
                    rescaled = ((band_arr - c_off) * scale) + n_off
                    res = np.where(nodata_mask, n_no_data_val, rescaled)
                    out_block = cast_result_block(res, win_h, win_w, out_np_dtype)
                    dst.write(out_block, out_band_idx, window=out_win)
                    out_band_idx += 1


def rescale_img_pxl_vals(
    input_img: str,
    output_img: str,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    band_rescale_objs: Sequence[ImageBandRescale] = (),
    trim_to_limits: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rescale image pixel values per band using a sequence of
    :class:`ImageBandRescale` specifications.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param band_rescale_objs: Sequence of :class:`ImageBandRescale` objects
                              defining per-band input/output ranges.
    :param trim_to_limits: If ``True``, clamp non-nodata outputs to the
                           ``[out_min, out_max]`` interval (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale

        rescale_objs = [
            rsgispy.imagecalc.rescale.ImageBandRescale(
                band=1,
                in_min=0.0,
                in_max=10000.0,
                no_data_val=0.0,
                out_min=0.0,
                out_max=1.0,
                out_no_data=-9999.0,
            ),
        ]
        rsgispy.imagecalc.rescale.rescale_img_pxl_vals(
            input_img="img.tif",
            output_img="rescaled.tif",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
            band_rescale_objs=rescale_objs,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not band_rescale_objs:
        raise rsgispy.RSGISPyException(
            "band_rescale_objs must contain at least one ImageBandRescale."
        )

    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    rescale_dict: Dict[int, ImageBandRescale] = {}
    for rescale_obj in band_rescale_objs:
        if rescale_obj.band < 1:
            raise rsgispy.RSGISPyException(
                f"Band index must be >= 1, got {rescale_obj.band}."
            )
        if rescale_obj.in_max == rescale_obj.in_min:
            raise rsgispy.RSGISPyException(
                f"Band {rescale_obj.band} has in_max == in_min ({rescale_obj.in_min})."
            )
        rescale_dict[rescale_obj.band - 1] = rescale_obj

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img])
        src = datasets[input_img]
        n_bands = src.count
        for b_idx in range(n_bands):
            if b_idx not in rescale_dict:
                raise rsgispy.RSGISPyException(
                    f"Missing ImageBandRescale specification for band {b_idx + 1}."
                )

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=n_bands,
            creation_opts=creation_opts,
        )

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            in_block = src.read(window=out_win, out_dtype=np.float64)
            out_bands = []
            for b_idx in range(n_bands):
                cfg = rescale_dict[b_idx]
                band_arr = in_block[b_idx]
                if np.isnan(cfg.no_data_val):
                    nodata_mask = ~np.isfinite(band_arr)
                else:
                    nodata_mask = (band_arr == cfg.no_data_val) | ~np.isfinite(
                        band_arr
                    )

                scaled = (
                    ((band_arr - cfg.in_min) / (cfg.in_max - cfg.in_min))
                    * (cfg.out_max - cfg.out_min)
                ) + cfg.out_min
                if trim_to_limits:
                    low = min(cfg.out_min, cfg.out_max)
                    high = max(cfg.out_min, cfg.out_max)
                    scaled = np.clip(scaled, low, high)

                res = np.where(nodata_mask, cfg.out_no_data, scaled)
                out_bands.append(
                    cast_result_block(res, win_h, win_w, out_np_dtype)
                )
            dst.write(np.stack(out_bands, axis=0), window=out_win)


def calc_img_norm_params(
    input_img: Union[str, os.PathLike],
    method: Union[str, int] = "linear",
    no_data_val: Optional[float] = None,
    out_stats_file: Optional[Union[str, os.PathLike]] = None,
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    bands: Optional[Sequence[int]] = None,
    lower: float = 2.0,
    upper: float = 98.0,
    n_stdevs: float = 2.0,
    in_min_max: Optional[
        Union[Mapping[str, float], Sequence[Mapping[str, float]]]
    ] = None,
    method_kwargs: Optional[Mapping[str, Any]] = None,
    max_samples: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Calculate per-band pixel scaling and normalisation parameters for
    ``input_img`` and optionally save them to a JSON statistics file.

    Supported ``method`` values include all raster display stretches (both
    ``rsgispy.IMG_STRETCH_*`` constants and string names: ``'none'``,
    ``'linear'``, ``'stdev'``, ``'cumulative'``, ``'user'``/``'manual'``),
    statistical/numeric approaches (``'zscore'``, ``'robust_zscore'``,
    ``'decimal_scaling'``, ``'log'``, ``'log1p'``, ``'sigmoid'``, ``'tanh'``,
    ``'softmax'``, ``'l1'``, ``'l2'``, ``'max_norm'``), and ``scikit-learn``
    scalers/transformers (``'StandardScaler'``, ``'MinMaxScaler'``,
    ``'MaxAbsScaler'``, ``'RobustScaler'``, ``'PowerTransformer'``,
    ``'QuantileTransformer'``, ``'Normalizer'``).

    :param input_img: Path to the input raster image file.
    :param method: Normalisation algorithm name or ``IMG_STRETCH_*`` constant
                   (Default = ``'linear'``).
    :param no_data_val: Explicit input no-data value overriding raster band
                        metadata (Default = ``None``, uses per-band metadata).
    :param out_stats_file: Optional JSON file path to persist calculated
                           normalisation parameters.
    :param out_min: Optional target output minimum value. Defaults to ``0.0``
                    for stretch methods and ``MinMaxScaler``, or preserves the
                    native scale of unbounded scalers when ``None``.
    :param out_max: Optional target output maximum value. Defaults to ``1.0``
                    for stretch methods and ``MinMaxScaler``, or preserves the
                    native scale of unbounded scalers when ``None``.
    :param bands: Optional 1-based band indices to include (Default = all bands).
    :param lower: Lower percentile for ``'cumulative'`` stretch (Default = ``2.0``).
    :param upper: Upper percentile for ``'cumulative'`` stretch (Default = ``98.0``).
    :param n_stdevs: Number of standard deviations for ``'stdev'`` stretch
                     (Default = ``2.0``).
    :param in_min_max: Dict ``{'min': ..., 'max': ...}`` or sequence of dicts
                       per band required for ``'user'`` / ``'manual'`` stretch.
    :param method_kwargs: Optional keyword arguments passed to the underlying
                          ``scikit-learn`` scaler/transformer or ``'log'`` shift.
    :param max_samples: Optional maximum pixel sample count per band for fitting
                        quantile/power transformers cleanly on large rasters.
    :param rnd_seed: Optional random seed for subsampling / ``QuantileTransformer``.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :return: Dictionary containing calculated per-band normalisation parameters.

    .. code:: python

        import rsgispy.imagecalc.rescale as ic_resc

        params = ic_resc.calc_img_norm_params(
            input_img="s2.tif",
            method="StandardScaler",
            out_stats_file="s2_norm_stats.json",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    in_path = os.fspath(input_img)
    ensure_file_exists(in_path, "Input image")
    norm_method = resolve_norm_method(method)
    eff_out_min, eff_out_max = resolve_out_bounds(norm_method, out_min, out_max)

    if norm_method == "cumulative" and not 0.0 <= float(lower) < float(upper) <= 100.0:
        raise rsgispy.RSGISPyException(
            f"cumulative stretch requires 0 <= lower < upper <= 100, got ({lower}, {upper})."
        )
    if norm_method == "stdev" and float(n_stdevs) <= 0.0:
        raise rsgispy.RSGISPyException("n_stdevs must be strictly positive.")

    with rasterio.open(in_path, "r") as src:
        sel_bands = (
            list(range(1, src.count + 1))
            if bands is None
            else [int(b) for b in bands]
        )
        if not sel_bands:
            raise rsgispy.RSGISPyException("bands must not be empty.")
        for b_idx in sel_bands:
            validate_band_index(b_idx, src.count, in_path)

        band_entries: List[Dict[str, Any]] = []
        if norm_method in PIXEL_VECTOR_METHODS:
            norm_kind = (
                str((method_kwargs or {}).get("norm", "l2")).lower()
                if norm_method == "normalizer"
                else ("max" if norm_method == "max_norm" else norm_method)
            )
            if norm_method == "normalizer" and norm_kind not in ("l1", "l2", "max"):
                raise rsgispy.RSGISPyException(
                    f"Normalizer norm must be 'l1', 'l2', or 'max', got {norm_kind!r}."
                )
            band_nds = [
                resolve_band_nodata(src.nodatavals, b_idx, no_data_val)
                for b_idx in sel_bands
            ]
            vec_min: Optional[float] = None
            vec_max: Optional[float] = None
            if eff_out_min is not None and eff_out_max is not None:
                for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
                    blk = src.read(sel_bands, window=win, out_dtype=np.float64)
                    all_valid = build_valid_mask(blk[0], band_nds[0])
                    for i_b in range(1, len(sel_bands)):
                        all_valid &= build_valid_mask(blk[i_b], band_nds[i_b])
                    if np.any(all_valid):
                        normed = compute_pixel_vector_norm_matrix(
                            norm_method, blk[:, all_valid].T, norm_kind
                        )
                        b_lo = float(np.min(normed))
                        b_hi = float(np.max(normed))
                        vec_min = b_lo if vec_min is None else min(vec_min, b_lo)
                        vec_max = b_hi if vec_max is None else max(vec_max, b_hi)
            for b_idx, eff_nd in zip(sel_bands, band_nds):
                band_entries.append(
                    {
                        "band": int(b_idx),
                        "no_data_val": eff_nd,
                        "out_min": eff_out_min,
                        "out_max": eff_out_max,
                        "norm": norm_kind,
                        "vec_min": vec_min,
                        "vec_max": vec_max,
                    }
                )
        else:
            for sel_pos, b_idx in enumerate(sel_bands):
                eff_nd = resolve_band_nodata(src.nodatavals, b_idx, no_data_val)
                vals = subsample_band_values(
                    src,
                    b_idx,
                    eff_nd,
                    max_samples=max_samples,
                    rnd_seed=rnd_seed,
                    use_pbar=use_pbar,
                )
                u_bounds = (
                    _extract_user_bounds(in_min_max, sel_pos, len(sel_bands))
                    if norm_method == "user"
                    else None
                )
                band_entries.append(
                    fit_band_norm_params(
                        method=norm_method,
                        vals=vals,
                        band_idx=b_idx,
                        eff_nd=eff_nd,
                        eff_out_min=eff_out_min,
                        eff_out_max=eff_out_max,
                        lower=lower,
                        upper=upper,
                        n_stdevs=n_stdevs,
                        user_bounds=u_bounds,
                        method_kwargs=method_kwargs,
                        rnd_seed=rnd_seed,
                    )
                )

    result: Dict[str, Any] = {
        "version": 1,
        "method": norm_method,
        "input_img": in_path,
        "selected_bands": sel_bands,
        "out_min": eff_out_min,
        "out_max": eff_out_max,
        "method_kwargs": dict(method_kwargs or {}),
        "bands": band_entries,
    }
    if out_stats_file is not None:
        out_json = ensure_parent_dir(out_stats_file)
        write_dict_to_json(result, out_json)
    return result


def apply_img_norm_params(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    norm_params: Optional[Mapping[str, Any]] = None,
    stats_file: Optional[Union[str, os.PathLike]] = None,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    trim_to_limits: Optional[bool] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Apply scaling and normalisation parameters (either passed directly via
    ``norm_params`` or loaded from ``stats_file``) to ``input_img`` and write
    the normalised raster to ``output_img``.

    :param input_img: Path to the input raster image file.
    :param output_img: Path to the output normalised raster image file.
    :param norm_params: Normalisation parameter dict returned by
                        :func:`calc_img_norm_params`.
    :param stats_file: Path to a JSON statistics file written by
                       :func:`calc_img_norm_params`.
    :param gdalformat: GDAL/rasterio driver format for output
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param out_min: Optional override for output minimum value.
    :param out_max: Optional override for output maximum value.
    :param out_no_data_val: Optional explicit output no-data value.
    :param trim_to_limits: Clamp transformed values to ``[out_min, out_max]``.
                           Defaults to ``True`` for ``'stdev'``, ``'cumulative'``,
                           and ``'user'`` stretches, and ``False`` otherwise.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale as ic_resc

        ic_resc.apply_img_norm_params(
            input_img="s2.tif",
            output_img="s2_norm.tif",
            stats_file="s2_norm_stats.json",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    if norm_params is None and stats_file is None:
        raise rsgispy.RSGISPyException(
            "Either norm_params or stats_file must be provided."
        )
    if norm_params is not None and stats_file is not None:
        raise rsgispy.RSGISPyException(
            "Provide either norm_params or stats_file, not both."
        )

    params_dict: Mapping[str, Any] = (
        read_json_to_dict(os.fspath(stats_file))
        if stats_file is not None
        else norm_params  # type: ignore[assignment]
    )
    if not isinstance(params_dict, Mapping) or "bands" not in params_dict:
        raise rsgispy.RSGISPyException(
            "Invalid normalisation parameter dictionary."
        )

    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, "Input image")

    norm_method = resolve_norm_method(params_dict["method"])
    band_specs: Sequence[Mapping[str, Any]] = params_dict["bands"]
    if not band_specs:
        raise rsgispy.RSGISPyException("norm_params['bands'] must not be empty.")

    if out_min is not None or out_max is not None:
        eff_out_min, eff_out_max = resolve_out_bounds(
            norm_method, out_min, out_max
        )
    else:
        saved_min = params_dict.get("out_min")
        saved_max = params_dict.get("out_max")
        eff_out_min = float(saved_min) if saved_min is not None else None
        eff_out_max = float(saved_max) if saved_max is not None else None

    eff_trim = (
        (norm_method in DEFAULT_TRIM_METHODS)
        if trim_to_limits is None
        else bool(trim_to_limits)
    )
    out_rio_dtype = rsgispy.get_rasterio_datatype(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    eff_out_nd = resolve_out_nodata(band_specs, out_no_data_val, out_np_dtype)
    in_band_names = get_band_names(in_path)

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_path])
        src = datasets[in_path]
        sel_bands = [int(b_spec["band"]) for b_spec in band_specs]
        for b_idx in sel_bands:
            validate_band_index(b_idx, src.count, in_path)

        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=len(sel_bands),
            creation_opts=creation_opts,
        )
        if eff_out_nd is not None:
            dst.nodata = eff_out_nd

        for out_win in iter_dataset_windows(
            grid.height, grid.width, use_pbar=use_pbar
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            in_block = src.read(sel_bands, window=out_win, out_dtype=np.float64)

            if norm_method in PIXEL_VECTOR_METHODS:
                valid_masks = [
                    build_valid_mask(
                        in_block[idx],
                        band_specs[idx].get("no_data_val"),
                    )
                    for idx in range(len(sel_bands))
                ]
                norm_kind = str(band_specs[0].get("norm", "l2"))
                vec_out = apply_pixel_vector_norm(
                    method=norm_method,
                    block=in_block,
                    valid_masks=valid_masks,
                    norm_type=norm_kind,
                    eff_out_min=eff_out_min,
                    eff_out_max=eff_out_max,
                    trim_to_limits=eff_trim,
                    vec_min=band_specs[0].get("vec_min"),
                    vec_max=band_specs[0].get("vec_max"),
                )
                out_bands = []
                for idx in range(len(sel_bands)):
                    band_res = vec_out[idx]
                    if eff_out_nd is not None:
                        band_res = np.where(np.isfinite(band_res), band_res, eff_out_nd)
                    out_bands.append(
                        cast_result_block(band_res, win_h, win_w, out_np_dtype)
                    )
            else:
                out_bands = []
                for idx, b_spec in enumerate(band_specs):
                    arr = in_block[idx]
                    valid = build_valid_mask(arr, b_spec.get("no_data_val"))
                    res = np.full(
                        (win_h, win_w),
                        np.nan if eff_out_nd is None else float(eff_out_nd),
                        dtype=np.float64,
                    )
                    if np.any(valid):
                        res[valid] = apply_band_norm_1d(
                            method=norm_method,
                            vals=arr[valid],
                            band_params=b_spec,
                            eff_out_min=eff_out_min,
                            eff_out_max=eff_out_max,
                            trim_to_limits=eff_trim,
                        )
                    out_bands.append(
                        cast_result_block(res, win_h, win_w, out_np_dtype)
                    )
            dst.write(np.stack(out_bands, axis=0), window=out_win)

    out_names = [in_band_names[b_idx - 1] for b_idx in sel_bands]
    set_band_names(out_path, out_names, feedback=False)


def normalise_img_pxl_vals(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    method: Union[str, int] = "linear",
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    out_stats_file: Optional[Union[str, os.PathLike]] = None,
    bands: Optional[Sequence[int]] = None,
    lower: float = 2.0,
    upper: float = 98.0,
    n_stdevs: float = 2.0,
    in_min_max: Optional[
        Union[Mapping[str, float], Sequence[Mapping[str, float]]]
    ] = None,
    method_kwargs: Optional[Mapping[str, Any]] = None,
    trim_to_limits: Optional[bool] = None,
    max_samples: Optional[int] = None,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate image scaling/normalisation parameters with
    :func:`calc_img_norm_params` and apply them to ``input_img`` using
    :func:`apply_img_norm_params`.

    Supports all mapping stretch modes (``'none'``, ``'linear'``, ``'stdev'``,
    ``'cumulative'``, ``'user'``/``'manual'`` and ``rsgispy.IMG_STRETCH_*``),
    statistical scaling (``'zscore'``, ``'robust_zscore'``,
    ``'decimal_scaling'``, ``'log'``, ``'log1p'``, ``'sigmoid'``, ``'tanh'``,
    ``'softmax'``, ``'l1'``, ``'l2'``, ``'max_norm'``), and ``scikit-learn``
    scalers/transformers (``'StandardScaler'``, ``'MinMaxScaler'``,
    ``'MaxAbsScaler'``, ``'RobustScaler'``, ``'PowerTransformer'``,
    ``'QuantileTransformer'``, ``'Normalizer'``).

    :param input_img: Path to the input raster image file.
    :param output_img: Path to the output normalised raster image file.
    :param method: Normalisation algorithm name or ``IMG_STRETCH_*`` constant
                   (Default = ``'linear'``).
    :param gdalformat: GDAL/rasterio driver format for output
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional explicit input no-data value overriding metadata.
    :param out_min: Optional target output minimum value.
    :param out_max: Optional target output maximum value.
    :param out_no_data_val: Optional explicit output no-data value.
    :param out_stats_file: Optional JSON path to save computed parameters.
    :param bands: Optional 1-based band indices to normalise (Default = all).
    :param lower: Lower percentile for ``'cumulative'`` stretch (Default = ``2.0``).
    :param upper: Upper percentile for ``'cumulative'`` stretch (Default = ``98.0``).
    :param n_stdevs: Standard deviations for ``'stdev'`` stretch (Default = ``2.0``).
    :param in_min_max: Manual ``{'min': ..., 'max': ...}`` bounds for ``'user'``.
    :param method_kwargs: Optional extra kwargs passed to sklearn scalers / ``'log'``.
    :param trim_to_limits: Clamp normalised pixels to ``[out_min, out_max]``.
    :param max_samples: Optional maximum samples per band when fitting params.
    :param rnd_seed: Optional random seed for subsampling / ``QuantileTransformer``.
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.
    :return: Dictionary of calculated normalisation parameters.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale as ic_resc

        ic_resc.normalise_img_pxl_vals(
            input_img="s2.tif",
            output_img="s2_zscore.tif",
            method="zscore",
            out_min=0.0,
            out_max=1.0,
            out_stats_file="s2_zscore_params.json",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    norm_params = calc_img_norm_params(
        input_img=input_img,
        method=method,
        no_data_val=no_data_val,
        out_stats_file=out_stats_file,
        out_min=out_min,
        out_max=out_max,
        bands=bands,
        lower=lower,
        upper=upper,
        n_stdevs=n_stdevs,
        in_min_max=in_min_max,
        method_kwargs=method_kwargs,
        max_samples=max_samples,
        rnd_seed=rnd_seed,
        use_pbar=use_pbar,
    )
    apply_img_norm_params(
        input_img=input_img,
        output_img=output_img,
        norm_params=norm_params,
        gdalformat=gdalformat,
        datatype=datatype,
        out_min=out_min,
        out_max=out_max,
        out_no_data_val=out_no_data_val,
        trim_to_limits=trim_to_limits,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
    return norm_params
