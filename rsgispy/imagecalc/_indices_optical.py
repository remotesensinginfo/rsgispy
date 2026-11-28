"""
Shared band/scaling helpers and optical spectral index stack execution for
:mod:`rsgispy.imagecalc.calcindices`.
"""

from contextlib import ExitStack
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import rasterio

import rsgispy
from rsgispy.imagecalc._indices_registry import (
    _BAND_ALIASES,
    _IndexSpec,
    _lookup_index_spec,
)
from rsgispy._tilegraph._wrap import WrapperDiskSink, run_wrapper_graph
from rsgispy.imagecalc._indices_nodes import IndicesNode, IndicesSetup, scale_bands
from rsgispy.imagecalc._sar_indices import _SARIndexSpec
from rsgispy.imageprocess import (
    OnlineStatsAccumulator,
    RasterSource,
    TileNode,
    build_valid_mask,
    default_out_nodata,
    ensure_file_exists,
    iter_dataset_windows,
    resolve_out_nodata as _ip_resolve_out_nodata,
    round_clamp_to_dtype,
    validate_band_index,
)


def _normalise_band_names(
    band_names: Mapping[int, str],
    n_bands: int,
    input_img: str,
    band_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, int]:
    """
    Validate and normalise ``{band_idx: band_name}`` into
    ``{canonical_band_name: band_idx}``.
    """
    if not isinstance(band_names, Mapping) or not band_names:
        raise rsgispy.RSGISPyException(
            "band_names must be a non-empty mapping of 1-based band index "
            "to band name (e.g., {1: 'blue', 2: 'green', 5: 'swir1'})."
        )

    aliases = _BAND_ALIASES if band_aliases is None else band_aliases
    canonical_to_idx: Dict[str, int] = {}
    for raw_idx, raw_name in band_names.items():
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, (int, np.integer)):
            raise rsgispy.RSGISPyException(
                f"Band index keys in band_names must be integers, got {raw_idx!r}."
            )
        band_idx = int(raw_idx)
        validate_band_index(band_idx, n_bands, input_img)

        if not isinstance(raw_name, str) or not raw_name.strip():
            raise rsgispy.RSGISPyException(
                f"Band name for band {band_idx} must be a non-empty string."
            )
        norm_name = raw_name.strip().lower()
        canonical_name = aliases.get(norm_name, norm_name)
        if canonical_name in canonical_to_idx:
            raise rsgispy.RSGISPyException(
                f"Duplicate band name '{canonical_name}' mapped to bands "
                f"{canonical_to_idx[canonical_name]} and {band_idx}."
            )
        canonical_to_idx[canonical_name] = band_idx

    return canonical_to_idx


def _resolve_requested_indices(
    indices: Sequence[str], canonical_to_idx: Mapping[str, int]
) -> List[Tuple[str, _IndexSpec]]:
    """Validate requested index names and verify required bands exist."""
    if (
        isinstance(indices, (str, bytes))
        or not isinstance(indices, Sequence)
        or not indices
    ):
        raise rsgispy.RSGISPyException(
            "indices must be a non-empty sequence of spectral index names."
        )

    resolved: List[Tuple[str, _IndexSpec]] = []
    for raw_idx_name in indices:
        if not isinstance(raw_idx_name, str) or not raw_idx_name.strip():
            raise rsgispy.RSGISPyException(
                "Each entry in indices must be a non-empty string."
            )
        out_band_name = raw_idx_name.strip()
        spec = _lookup_index_spec(out_band_name)
        missing = [b for b in spec.req_bands if b not in canonical_to_idx]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Index '{spec.canonical_name}' requires band(s) "
                f"{', '.join(missing)}, which were not provided in band_names."
            )
        resolved.append((out_band_name, spec))

    return resolved


def _resolve_in_scaling(
    val: Optional[Union[float, Sequence[float], Mapping[Union[int, str], float]]],
    default: float,
    req_bands: Sequence[str],
    canonical_to_idx: Mapping[str, int],
    n_bands: int,
    label: str,
    check_nonzero: bool = False,
    band_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """Resolve scalar, per-image-band sequence, or mapping into per-band values."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
    if val is None:
        return {b: float(default) for b in req_bands}

    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        f_val = float(val)
        if check_nonzero and f_val == 0.0:
            raise rsgispy.RSGISPyException(f"{label} must not be zero.")
        return {b: f_val for b in req_bands}

    aliases = _BAND_ALIASES if band_aliases is None else band_aliases
    if isinstance(val, Mapping):
        lower_map: Dict[Union[int, str], float] = {}
        for k, v in val.items():
            if isinstance(k, str):
                k_norm = k.strip().lower()
                lower_map[aliases.get(k_norm, k_norm)] = float(v)
            elif isinstance(k, (int, np.integer)) and not isinstance(k, bool):
                lower_map[int(k)] = float(v)
            else:
                raise rsgispy.RSGISPyException(
                    f"Invalid key {k!r} in {label} mapping."
                )

        out: Dict[str, float] = {}
        for b_name in req_bands:
            b_idx = canonical_to_idx[b_name]
            if b_idx in lower_map:
                f_val = lower_map[b_idx]
            elif b_name in lower_map:
                f_val = lower_map[b_name]
            else:
                raise rsgispy.RSGISPyException(
                    f"Missing {label} value for band '{b_name}' (band {b_idx})."
                )
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for band '{b_name}' must not be zero."
                )
            out[b_name] = f_val
        return out

    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        if len(val) != n_bands:
            raise rsgispy.RSGISPyException(
                f"{label} sequence length ({len(val)}) must match input "
                f"image band count ({n_bands})."
            )
        out = {}
        for b_name in req_bands:
            b_idx = canonical_to_idx[b_name]
            f_val = float(val[b_idx - 1])
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for band '{b_name}' (band {b_idx}) must not be zero."
                )
            out[b_name] = f_val
        return out

    raise rsgispy.RSGISPyException(
        f"{label} must be a scalar, sequence, or mapping."
    )


def _resolve_out_scaling(
    val: Optional[Union[float, Sequence[float], Mapping[str, float]]],
    default: float,
    resolved_indices: Sequence[Tuple[str, Union[_IndexSpec, _SARIndexSpec]]],
    label: str,
    check_nonzero: bool = False,
) -> List[float]:
    """Resolve scalar, per-output-band sequence, or mapping into a list per index."""
    # pylint: disable=too-many-branches
    if val is None:
        return [float(default) for _ in resolved_indices]

    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        f_val = float(val)
        if check_nonzero and f_val == 0.0:
            raise rsgispy.RSGISPyException(f"{label} must not be zero.")
        return [f_val for _ in resolved_indices]

    if isinstance(val, Mapping):
        lower_map = {
            str(k).strip().lower(): float(v) for k, v in val.items()
        }
        out: List[float] = []
        for out_name, spec in resolved_indices:
            key_req = out_name.lower()
            key_can = spec.canonical_name.lower()
            if key_req in lower_map:
                f_val = lower_map[key_req]
            elif key_can in lower_map:
                f_val = lower_map[key_can]
            else:
                raise rsgispy.RSGISPyException(
                    f"Missing {label} value for index '{out_name}'."
                )
            if check_nonzero and f_val == 0.0:
                raise rsgispy.RSGISPyException(
                    f"{label} for index '{out_name}' must not be zero."
                )
            out.append(f_val)
        return out

    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        if len(val) != len(resolved_indices):
            raise rsgispy.RSGISPyException(
                f"{label} sequence length ({len(val)}) must match number of "
                f"requested indices ({len(resolved_indices)})."
            )
        out = [float(v) for v in val]
        if check_nonzero:
            for (out_name, _), f_val in zip(resolved_indices, out):
                if f_val == 0.0:
                    raise rsgispy.RSGISPyException(
                        f"{label} for index '{out_name}' must not be zero."
                    )
        return out

    raise rsgispy.RSGISPyException(
        f"{label} must be a scalar, sequence, or mapping."
    )


def _resolve_out_nodata(
    out_no_data_val: Optional[float], out_np_dtype: np.dtype
) -> float:
    """
    Resolve and validate the output no-data value for the target dtype
    (``None`` gives :func:`rsgispy.imageprocess.default_out_nodata`).
    """
    if out_no_data_val is None:
        return default_out_nodata(out_np_dtype)
    res = _ip_resolve_out_nodata(
        out_np_dtype,
        out_no_data_val=out_no_data_val,
        clamp_int=True,
        strict_explicit=True,
        allow_round_int=True,
        allow_nonfinite_float=True,
    )
    return float(res)


def _resolve_index_params(
    index_params: Optional[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Normalise per-index parameter dictionaries keyed by lower-case index name."""
    if index_params is None:
        return {}
    if not isinstance(index_params, Mapping):
        raise rsgispy.RSGISPyException(
            "index_params must be a mapping of index name to parameter dict."
        )
    out: Dict[str, Dict[str, float]] = {}
    for k, v in index_params.items():
        if not isinstance(v, Mapping):
            raise rsgispy.RSGISPyException(
                f"Parameters for index '{k}' in index_params must be a mapping."
            )
        out[str(k).strip().lower()] = {
            str(pk).strip().lower(): float(pv) for pk, pv in v.items()
        }
    return out


def _read_scaled_bands(
    src: rasterio.io.DatasetReader,
    win: rasterio.windows.Window,
    band_list: Sequence[str],
    canonical_to_idx: Mapping[str, int],
    in_nodata_map: Mapping[str, Optional[float]],
    in_gain_map: Mapping[str, float],
    in_offset_map: Mapping[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read, mask, and unscale a window of required bands."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    band_indices = [canonical_to_idx[b] for b in band_list]
    raw_block = src.read(band_indices, window=win, out_dtype=np.float64)
    scaled_bands: Dict[str, np.ndarray] = {}
    invalid_masks: Dict[str, np.ndarray] = {}

    for pos, b_name in enumerate(band_list):
        raw_arr = raw_block[pos]
        invalid = ~build_valid_mask(raw_arr, in_nodata_map[b_name])
        phys = (raw_arr - in_offset_map[b_name]) / in_gain_map[b_name]
        invalid_masks[b_name] = invalid
        scaled_bands[b_name] = phys

    return scaled_bands, invalid_masks


def _calc_brightness_stretch_bounds(
    src: rasterio.io.DatasetReader,
    canonical_to_idx: Mapping[str, int],
    in_nodata_map: Mapping[str, Optional[float]],
    in_gain_map: Mapping[str, float],
    in_offset_map: Mapping[str, float],
    stretch_std_devs: float,
    use_pbar: bool,
) -> Tuple[float, float]:
    """Compute ``(mean - k*std, mean + k*std)`` for ``Brightness_Scaled``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vis_bands = ("blue", "green", "red")
    acc = OnlineStatsAccumulator(n_bands=1)

    for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
        scaled_bands, invalid_masks = _read_scaled_bands(
            src,
            win,
            vis_bands,
            canonical_to_idx,
            in_nodata_map,
            in_gain_map,
            in_offset_map,
        )
        valid_mask = ~(
            invalid_masks["blue"] | invalid_masks["green"] | invalid_masks["red"]
        )
        if not np.any(valid_mask):
            continue
        brightness = (
            scaled_bands["blue"] + scaled_bands["green"] + scaled_bands["red"]
        ) / 3.0
        vals = brightness[valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            acc.update_band(0, vals)

    if int(acc.counts[0]) == 0:
        return 0.0, 0.0

    mean_val = float(acc.means[0])
    stddev_val = float(acc.stds()[0])
    half_width = stretch_std_devs * stddev_val
    return mean_val - half_width, mean_val + half_width


def _format_output_band(
    index_arr: np.ndarray,
    valid_mask: np.ndarray,
    out_g: float,
    out_o: float,
    out_nd: float,
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """
    Apply output gain/offset scaling, then integerise and mask no-data with
    :func:`rsgispy.imageprocess.round_clamp_to_dtype` (round, clamp, and keep
    valid values off ``out_nd``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    scaled = (index_arr * out_g) + out_o
    return round_clamp_to_dtype(scaled, out_np_dtype, out_nd, valid_mask)


def _setup_optical_indices(
    band_names: Mapping[int, str],
    indices: Sequence[str],
    in_gain: Any,
    in_offset: Any,
    out_gain: Any,
    out_offset: Any,
    n_bands: int,
    band_nodata: Sequence[Optional[float]],
    img_label: str,
) -> IndicesSetup:
    """
    Validate the band mapping, indices and scaling against an input with
    ``n_bands`` bands and per-band no-data values ``band_nodata``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    canonical_to_idx = _normalise_band_names(band_names, n_bands, img_label)
    resolved_indices = _resolve_requested_indices(indices, canonical_to_idx)
    return finish_indices_setup(
        canonical_to_idx,
        resolved_indices,
        (in_gain, in_offset, out_gain, out_offset),
        n_bands,
        band_nodata,
    )


def finish_indices_setup(
    canonical_to_idx: Dict[str, int],
    resolved_indices: List[Tuple[str, Any]],
    scaling: Tuple[Any, Any, Any, Any],
    n_bands: int,
    band_nodata: Sequence[Optional[float]],
    band_aliases: Optional[Mapping[str, str]] = None,
) -> IndicesSetup:
    """
    Resolve the scaling ``(in_gain, in_offset, out_gain, out_offset)`` and
    the no-data values of the bands the indices need.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_gain, in_offset, out_gain, out_offset = scaling
    req_band_set: List[str] = []
    seen_bands = set()
    for _, spec in resolved_indices:
        for b_name in spec.req_bands:
            if b_name not in seen_bands:
                seen_bands.add(b_name)
                req_band_set.append(b_name)

    in_gain_map = _resolve_in_scaling(
        in_gain,
        1.0,
        req_band_set,
        canonical_to_idx,
        n_bands,
        "in_gain",
        check_nonzero=True,
        band_aliases=band_aliases,
    )
    in_offset_map = _resolve_in_scaling(
        in_offset,
        0.0,
        req_band_set,
        canonical_to_idx,
        n_bands,
        "in_offset",
        check_nonzero=False,
        band_aliases=band_aliases,
    )
    out_gain_list = _resolve_out_scaling(
        out_gain, 1.0, resolved_indices, "out_gain", check_nonzero=True
    )
    out_offset_list = _resolve_out_scaling(
        out_offset, 0.0, resolved_indices, "out_offset", check_nonzero=False
    )
    in_nodata_map: Dict[str, Optional[float]] = {
        b_name: band_nodata[canonical_to_idx[b_name] - 1] for b_name in req_band_set
    }
    return IndicesSetup(
        canonical_to_idx=canonical_to_idx,
        resolved_indices=resolved_indices,
        req_band_set=req_band_set,
        in_gain_map=in_gain_map,
        in_offset_map=in_offset_map,
        out_gain_list=out_gain_list,
        out_offset_list=out_offset_list,
        in_nodata_map=in_nodata_map,
    )


def calc_index_band(
    domain: Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]],
    idx_pos: int,
    setup: IndicesSetup,
    parsed_idx_params: Mapping[str, Mapping[str, float]],
    out_fmt: Tuple[float, np.dtype],
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Compute and format output index band ``idx_pos`` from the
    ``(bands, invalid_masks)`` of its domain; ``out_fmt`` is
    ``(out_nd, out_np_dtype)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    domain_bands, domain_invalid = domain
    out_name, spec = setup.resolved_indices[idx_pos]
    req_invalid = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    for b_name in spec.req_bands:
        req_invalid = req_invalid | domain_invalid[b_name]
    per_idx_params = parsed_idx_params.get(
        out_name.lower(),
        parsed_idx_params.get(spec.canonical_name.lower(), {}),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_idx, expr_valid = spec.calc_fn(domain_bands, per_idx_params)
    full_valid = (~req_invalid) & expr_valid & np.isfinite(raw_idx)
    return _format_output_band(
        index_arr=raw_idx,
        valid_mask=full_valid,
        out_g=setup.out_gain_list[idx_pos],
        out_o=setup.out_offset_list[idx_pos],
        out_nd=out_fmt[0],
        out_np_dtype=out_fmt[1],
    )


def _optical_block_fn(
    parsed_idx_params: Mapping[str, Mapping[str, float]],
    out_nd: float,
    out_np_dtype: np.dtype,
) -> Any:
    """Return the block function of an optical :class:`IndicesNode`."""

    def _block(
        data: np.ndarray,
        setup: IndicesSetup,
        positions: Mapping[int, int],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        domain = scale_bands(data, setup, positions)
        return np.stack(
            [
                calc_index_band(
                    domain, idx_pos, setup, parsed_idx_params,
                    (out_nd, out_np_dtype), shape,
                )
                for idx_pos in range(len(setup.resolved_indices))
            ],
            axis=0,
        )

    return _block


def _check_brightness_params(
    parsed_idx_params: Mapping[str, Mapping[str, float]],
) -> Any:
    """Return a setup check requiring given ``Brightness_Scaled`` bounds."""

    def _check(setup: IndicesSetup) -> None:
        if any(spec.requires_brightness_stats for _, spec in setup.resolved_indices):
            b_params = parsed_idx_params.get("brightness_scaled", {})
            if "in_min" not in b_params or "in_max" not in b_params:
                raise rsgispy.RSGISPyException(
                    "Brightness_Scaled needs whole-image statistics: give its "
                    "stretch bounds as index_params={'Brightness_Scaled': "
                    "{'in_min': ..., 'in_max': ...}} (calc_indices calculates "
                    "them in a first pass)."
                )

    return _check


def calc_indices_node(
    source: TileNode,
    band_names: Mapping[int, str],
    indices: Sequence[str],
    in_gain: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 1.0,
    in_offset: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 0.0,
    out_gain: Optional[Union[float, Sequence[float], Mapping[str, float]]] = 1.0,
    out_offset: Optional[Union[float, Sequence[float], Mapping[str, float]]] = 0.0,
    index_params: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    out_dtype: Any = np.float32,
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`calc_indices`: one output band per spectral
    index, named after the index, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The parameters are those of :func:`calc_indices`; ``band_names`` refers to
    the bands of ``source``. The band mapping, indices and scaling are checked
    when the graph is built, with the same exceptions as :func:`calc_indices`.
    The node's blocks have the data type ``out_dtype`` (values are rounded and
    clamped for integer types, as :func:`calc_indices` does for an integer
    ``datatype``) and its no-data value is the resolved ``out_no_data_val``.
    ``Brightness_Scaled`` needs whole-image statistics, so its bounds must be
    given as ``index_params={'Brightness_Scaled': {'in_min': ..., 'in_max':
    ...}}``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param band_names: Mapping from 1-based band index of ``source`` to band
                       name (e.g. ``{1: "blue", 3: "red", 4: "nir"}``).
    :param indices: Index names, one output band each.
    :param in_gain: Input gain, as for :func:`calc_indices` (Default = ``1.0``).
    :param in_offset: Input offset (Default = ``0.0``).
    :param out_gain: Output gain (Default = ``1.0``).
    :param out_offset: Output offset (Default = ``0.0``).
    :param index_params: Optional per-index parameters.
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.float32``, as ``TYPE_32FLOAT``).
    :param no_data_val: Optional input no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param out_no_data_val: Optional output no-data value (Default = ``None``:
                            ``-9999`` for float types, the type's minimum
                            (signed) or maximum (unsigned) for integers).
    :param name: Optional node name (Default = ``'calc_indices'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node
        from rsgispy.imagefilter.stats import median_filter_node

        s2 = RasterSource("s2_refl.tif")
        idx = calc_indices_node(
            s2,
            {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"},
            ["NDVI", "MNDWI"],
            in_gain=10000.0,
        )
        # Smooth the indices in the same pass (no intermediate file).
        smooth = median_filter_node(idx, 3, no_data_val=-999.0)
        TileGraph(DiskSink(smooth, "indices_med3.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_np_dtype = np.dtype(out_dtype)
    out_nd = _resolve_out_nodata(out_no_data_val, out_np_dtype)
    parsed_idx_params = _resolve_index_params(index_params)

    def _setup(
        n_bands: int, band_nodata: Sequence[Optional[float]], label: str
    ) -> IndicesSetup:
        return _setup_optical_indices(
            band_names, indices, in_gain, in_offset, out_gain, out_offset,
            n_bands, band_nodata, label,
        )

    return IndicesNode(
        source,
        _setup,
        _optical_block_fn(parsed_idx_params, out_nd, out_np_dtype),
        out_np_dtype,
        out_nd,
        no_data_val=no_data_val,
        check_fn=_check_brightness_params(parsed_idx_params),
        name=name or "calc_indices",
    )


def calc_indices(
    input_img: str,
    output_img: str,
    band_names: Mapping[int, str],
    indices: Sequence[str],
    in_gain: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 1.0,
    in_offset: Optional[
        Union[float, Sequence[float], Mapping[Union[int, str], float]]
    ] = 0.0,
    out_gain: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = 1.0,
    out_offset: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = 0.0,
    index_params: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate multiple spectral indices in a single pass and write them to a
    multi-band output image with each band named after its index.

    Input pixel values are converted to physical reflectance/units via
    ``physical = (dn - in_offset) / in_gain`` prior to index calculation, and
    computed index values are converted to output pixel values via
    ``output = (index_val * out_gain) + out_offset``. When an integer
    ``datatype`` is used (e.g., ``rsgispy.TYPE_16INT``), valid values are
    rounded to the nearest integer and clamped to the target data type range.

    Index naming disambiguation:

    * ``NDWI_McFeeters`` (also ``NDWI`` and ``GNDWI``):
      ``(green - nir) / (green + nir)`` (McFeeters 1996).
    * ``NDWI_Gao`` (also ``NDMI``, ``NDII``, and ``LSWI``):
      ``(nir - swir1) / (nir + swir1)`` (Gao 1996; Hardisky et al. 1983;
      Xiao et al. 2002; note ``rsgislib.calc_ndwi`` used this formula).
    * ``NDMaI`` (also ``NDWVI``):
      ``(swir2 - green) / (swir2 + green)`` (Normalised Difference Mangrove
      Index, Shi et al. 2016; disambiguated from moisture ``NDMI``).
    * ``MNDWI`` (also ``GMNDWI`` and ``NDSI``):
      ``(green - swir1) / (green + swir1)`` (Xu 2006 / Salomonson & Appel 2004).
    * ``MVI`` computes ``(nir - green) / (swir1 - green)``, whereas ``MVI_Abs``
      computes ``|nir - green| / |swir1 - green|`` (Baloloy et al. 2020).
    * ``WBI_BlueNIR`` (also ``WBI``):
      ``blue / nir`` (multispectral RSGISLib ratio, distinct from narrowband
      ``R900 / R970``).
    * ``SR`` / ``RVI`` / ``RVI_NIR_Red`` compute ``nir / red``, whereas
      ``RVI_Red_NIR`` computes ``red / nir``.

    Canonical band names accepted in ``band_names``: ``'blue'``, ``'green'``,
    ``'red'``, ``'red_edge'`` (alias ``'re'``), ``'re1'`` (~705 nm),
    ``'re2'`` (~740 nm), ``'re3'`` (~783 nm), ``'nir'``, ``'nir_narrow'``
    (aliases ``'re4'``, ``'nir8a'``, ~865 nm), ``'swir1'`` (alias ``'swir'``),
    and ``'swir2'``.

    Full equations and categorised tables for all supported indices are
    documented in :doc:`imagecalc/calcindices <../imagecalc/calcindices>` and
    inspectable via :func:`get_index_equation`.

    :param input_img: Path to the input multi-band raster image file.
    :param output_img: Path to the output multi-band raster image file.
    :param band_names: Mapping from 1-based band index to band name
                       (e.g., ``{1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"}``).
    :param indices: Sequence of spectral index names to calculate and stack
                    into ``output_img`` in the specified order.
    :param in_gain: Scalar, per-band sequence (length matching input band count),
                    or mapping (keyed by band index or band name) used to unscale
                    input values via ``(dn - in_offset) / in_gain``
                    (Default = ``1.0``).
    :param in_offset: Scalar, per-band sequence, or mapping subtracted from
                      input values prior to dividing by ``in_gain``
                      (Default = ``0.0``).
    :param out_gain: Optional scalar, per-index sequence, or mapping applied to
                     output index values via ``(index_val * out_gain) + out_offset``
                     to support integerisation (Default = ``1.0``).
    :param out_offset: Optional scalar, per-index sequence, or mapping added
                       after multiplying by ``out_gain`` (Default = ``0.0``).
    :param index_params: Optional mapping of per-index parameter overrides
                         (e.g., ``{"SAVI": {"l": 0.5}, "WDRVI": {"a": 0.1},
                         "GCMI": {"a": 3.4, "b": 2.9, "c": 4.3, "d": 1.1}}``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional input no-data value. If ``None``, reads the
                        no-data value(s) from the input image header.
    :param out_no_data_val: Optional output no-data value. If ``None``, uses
                            ``-9999`` for floating-point types and the type's
                            minimum (signed) or maximum (unsigned) for integer
                            types (:func:`rsgispy.imageprocess.default_out_nodata`;
                            before 0.5.0 the default was ``-999``).
    :param calc_stats: If ``True``, populate band statistics, histograms, and
                       overview pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.calcindices

        band_names = {1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1"}
        rsgispy.imagecalc.calcindices.calc_indices(
            "s2_refl.tif",
            "indices_int16.tif",
            band_names=band_names,
            indices=["NDVI", "NDWI_McFeeters", "MNDWI", "EVI"],
            in_gain=10000.0,
            in_offset=0.0,
            out_gain=1000.0,
            out_offset=0.0,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_16INT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    out_nd = _resolve_out_nodata(out_no_data_val, out_np_dtype)
    parsed_idx_params = _resolve_index_params(index_params)

    with ExitStack() as stack:
        try:
            src = stack.enter_context(rasterio.open(input_img, "r"))
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Could not open input image '{input_img}': {err}"
            ) from err

        n_bands = int(src.count)
        band_nodata = (
            [float(no_data_val)] * n_bands
            if no_data_val is not None
            else list(src.nodatavals)
        )
        setup = _setup_optical_indices(
            band_names, indices, in_gain, in_offset, out_gain, out_offset,
            n_bands, band_nodata, input_img,
        )

        if any(spec.requires_brightness_stats for _, spec in setup.resolved_indices):
            b_params = parsed_idx_params.get("brightness_scaled", {})
            stretch_k = float(b_params.get("stretch_std_devs", 2.0))
            b_min, b_max = _calc_brightness_stretch_bounds(
                src=src,
                canonical_to_idx=setup.canonical_to_idx,
                in_nodata_map=setup.in_nodata_map,
                in_gain_map=setup.in_gain_map,
                in_offset_map=setup.in_offset_map,
                stretch_std_devs=stretch_k,
                use_pbar=use_pbar,
            )
            parsed_idx_params["brightness_scaled"] = {
                **b_params,
                "in_min": b_min,
                "in_max": b_max,
            }

    # Read only the bands the indices need.
    src_bands = sorted({setup.canonical_to_idx[b] for b in setup.req_band_set})

    def _setup_fn(
        n_bands_: int, band_nodata_: Sequence[Optional[float]], label: str
    ) -> IndicesSetup:
        del n_bands_, band_nodata_, label
        return setup

    node = IndicesNode(
        RasterSource(input_img, bands=src_bands),
        _setup_fn,
        _optical_block_fn(parsed_idx_params, out_nd, out_np_dtype),
        out_np_dtype,
        out_nd,
        src_bands=src_bands,
        name="calc_indices",
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=out_nd,
        band_names=setup.out_names,
        calc_stats=calc_stats,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)
