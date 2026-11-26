"""
SAR spectral/backscatter index stack execution for
:mod:`rsgispy.imagecalc.calcindices`.
"""

from contextlib import ExitStack
import math
from typing import (
    Any,
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
from rsgispy._tilegraph._wrap import WrapperDiskSink, run_wrapper_graph
from rsgispy.imagecalc._indices_nodes import IndicesNode, IndicesSetup, scale_bands
from rsgispy.imagecalc._indices_optical import (
    _normalise_band_names,
    _read_scaled_bands,
    _resolve_index_params,
    _resolve_out_nodata,
    calc_index_band,
    finish_indices_setup,
)
from rsgispy.imagecalc._sar_indices import (
    _SAR_BAND_ALIASES,
    _SARIndexSpec,
    convert_sar_block_units,
    lookup_sar_index_spec,
    normalise_sar_units,
)
from rsgispy.imageprocess import (
    RasterSource,
    TileNode,
    ensure_file_exists,
    iter_dataset_windows,
)


def _resolve_requested_sar_indices(
    indices: Sequence[str], canonical_to_idx: Mapping[str, int]
) -> List[Tuple[str, _SARIndexSpec]]:
    """Validate requested SAR index names and verify required polarisations exist."""
    if (
        isinstance(indices, (str, bytes))
        or not isinstance(indices, Sequence)
        or not indices
    ):
        raise rsgispy.RSGISPyException(
            "indices must be a non-empty sequence of SAR index names."
        )

    resolved: List[Tuple[str, _SARIndexSpec]] = []
    for raw_idx_name in indices:
        if not isinstance(raw_idx_name, str) or not raw_idx_name.strip():
            raise rsgispy.RSGISPyException(
                "Each entry in indices must be a non-empty string."
            )
        out_band_name = raw_idx_name.strip()
        spec = lookup_sar_index_spec(out_band_name)
        missing = [b for b in spec.req_bands if b not in canonical_to_idx]
        if missing:
            raise rsgispy.RSGISPyException(
                f"SAR index '{spec.canonical_name}' requires polarisation(s) "
                f"{', '.join(missing)}, which were not provided in band_names."
            )
        resolved.append((out_band_name, spec))

    return resolved


def _calc_sar_vv_max(
    src: rasterio.io.DatasetReader,
    canonical_to_idx: Mapping[str, int],
    in_nodata_map: Mapping[str, Optional[float]],
    in_gain_map: Mapping[str, float],
    in_offset_map: Mapping[str, float],
    in_units: str,
    use_pbar: bool,
) -> float:
    """Compute the scene-wide maximum linear-power VV backscatter for ``DPSVI``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    vv_max = -math.inf
    for win in iter_dataset_windows(src.height, src.width, use_pbar=use_pbar):
        phys_bands, invalid_masks = _read_scaled_bands(
            src,
            win,
            ("vv",),
            canonical_to_idx,
            in_nodata_map,
            in_gain_map,
            in_offset_map,
        )
        pwr_bands, pwr_invalid, _, _ = convert_sar_block_units(
            phys_bands, invalid_masks, in_units, need_power=True, need_db=False
        )
        valid_mask = ~pwr_invalid["vv"]
        if np.any(valid_mask):
            vv_max = max(vv_max, float(np.max(pwr_bands["vv"][valid_mask])))
    return vv_max if math.isfinite(vv_max) else math.nan


def _setup_sar_indices(
    band_names: Mapping[int, str],
    indices: Sequence[str],
    scaling: Tuple[Any, Any, Any, Any],
    n_bands: int,
    band_nodata: Sequence[Optional[float]],
    img_label: str,
) -> IndicesSetup:
    """
    Validate the polarisation mapping, SAR indices and scaling
    ``(in_gain, in_offset, out_gain, out_offset)`` against an input with
    ``n_bands`` bands and per-band no-data values ``band_nodata``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    canonical_to_idx = _normalise_band_names(
        band_names, n_bands, img_label, band_aliases=_SAR_BAND_ALIASES
    )
    resolved_indices = _resolve_requested_sar_indices(indices, canonical_to_idx)
    return finish_indices_setup(
        canonical_to_idx,
        resolved_indices,
        scaling,
        n_bands,
        band_nodata,
        band_aliases=_SAR_BAND_ALIASES,
    )


def _sar_block_fn(
    norm_units: str,
    parsed_idx_params: Mapping[str, Mapping[str, float]],
    out_nd: float,
    out_np_dtype: np.dtype,
) -> Any:
    """Return the block function of a SAR :class:`IndicesNode`."""

    def _block(
        data: np.ndarray,
        setup: IndicesSetup,
        positions: Mapping[int, int],
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        phys_bands, base_invalid = scale_bands(data, setup, positions)
        need_power = any(spec.units == "power" for _, spec in setup.resolved_indices)
        need_db = any(spec.units == "dB" for _, spec in setup.resolved_indices)
        pwr_bands, pwr_invalid, db_bands, db_invalid = convert_sar_block_units(
            phys_bands,
            base_invalid,
            norm_units,
            need_power=need_power,
            need_db=need_db,
        )
        out_bands: List[np.ndarray] = []
        for idx_pos, (_, spec) in enumerate(setup.resolved_indices):
            domain = (
                (pwr_bands, pwr_invalid)
                if spec.units == "power"
                else (db_bands, db_invalid)
            )
            out_bands.append(
                calc_index_band(
                    domain, idx_pos, setup, parsed_idx_params,
                    (out_nd, out_np_dtype), shape,
                )
            )
        return np.stack(out_bands, axis=0)

    return _block


def _check_vv_max_param(
    parsed_idx_params: Mapping[str, Mapping[str, float]],
) -> Any:
    """Return a setup check requiring a given ``DPSVI`` ``vv_max``."""

    def _check(setup: IndicesSetup) -> None:
        if any(spec.requires_vv_max for _, spec in setup.resolved_indices):
            if "vv_max" not in parsed_idx_params.get("dpsvi", {}):
                raise rsgispy.RSGISPyException(
                    "DPSVI needs the scene maximum of VV: give it as "
                    "index_params={'DPSVI': {'vv_max': ...}} (linear power; "
                    "calc_sar_indices calculates it in a first pass)."
                )

    return _check


def calc_sar_indices_node(
    source: TileNode,
    band_names: Mapping[int, str],
    indices: Sequence[str],
    in_units: str,
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
    Tile-graph node of :func:`calc_sar_indices`: one output band per SAR
    index, named after the index, with the same automatic dB / linear power
    conversions, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`
    (e.g. after a speckle filter) without intermediate files.

    The parameters are those of :func:`calc_sar_indices`; ``band_names``
    refers to the bands of ``source`` and is checked when the graph is built.
    The node's blocks have the data type ``out_dtype`` (integer types are
    rounded and clamped) and its no-data value is the resolved
    ``out_no_data_val``. ``DPSVI`` needs the scene maximum of VV, so it must
    be given as ``index_params={'DPSVI': {'vv_max': ...}}``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param band_names: Mapping from 1-based band index of ``source`` to
                       polarisation (e.g. ``{1: "VV", 2: "VH"}``).
    :param indices: SAR index names, one output band each.
    :param in_units: ``'dB'`` or ``'power'`` (units after ``in_gain`` and
                     ``in_offset``).
    :param in_gain: Input gain (Default = ``1.0``).
    :param in_offset: Input offset (Default = ``0.0``).
    :param out_gain: Output gain (Default = ``1.0``).
    :param out_offset: Output offset (Default = ``0.0``).
    :param index_params: Optional per-index parameters.
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.float32``).
    :param no_data_val: Optional input no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param out_no_data_val: Optional output no-data value (Default = ``None``:
                            ``-999``, or the type's limit for narrow integers).
    :param name: Optional node name (Default = ``'calc_sar_indices'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.sar import lee_filter_node
        from rsgispy.imagecalc.calcindices import calc_sar_indices_node

        s1 = RasterSource("s1_vv_vh_power.tif")
        speckle = lee_filter_node(s1, 7)
        idx = calc_sar_indices_node(
            speckle, {1: "VV", 2: "VH"}, ["RVI_VV_VH", "CR_VH_VV_dB"], "power"
        )
        TileGraph(DiskSink(idx, "s1_indices.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    norm_units = normalise_sar_units(in_units)
    out_np_dtype = np.dtype(out_dtype)
    out_nd = _resolve_out_nodata(out_no_data_val, out_np_dtype)
    parsed_idx_params = _resolve_index_params(index_params)

    def _setup(
        n_bands: int, band_nodata: Sequence[Optional[float]], label: str
    ) -> IndicesSetup:
        return _setup_sar_indices(
            band_names, indices, (in_gain, in_offset, out_gain, out_offset),
            n_bands, band_nodata, label,
        )

    return IndicesNode(
        source,
        _setup,
        _sar_block_fn(norm_units, parsed_idx_params, out_nd, out_np_dtype),
        out_np_dtype,
        out_nd,
        no_data_val=no_data_val,
        check_fn=_check_vv_max_param(parsed_idx_params),
        name=name or "calc_sar_indices",
    )


def calc_sar_indices(
    input_img: str,
    output_img: str,
    band_names: Mapping[int, str],
    indices: Sequence[str],
    in_units: str,
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
    Calculate multiple SAR polarimetric backscatter indices in a single pass
    and write them to a multi-band output image with each band named after its
    index.

    Input pixel values are first converted to physical backscatter units via
    ``physical = (dn - in_offset) / in_gain``. Depending on ``in_units``
    (``'dB'`` or ``'power'``) and the physical domain required by each
    requested index, values are automatically converted between decibels and
    linear power as needed:

    * **dB to linear power**: ``power = 10 ** (dB / 10)`` (required by
      linear-domain indices such as ``RVI``, ``RVI_VV_VH``, ``RVI_HH_HV``,
      ``DpRVIc_VV_VH``, ``DPSVI``, ``DPSVIm``, ``VDDPI``, ``RFDI``,
      ``NDI_VH_VV``, ``BMI``, ``CSI``, ``VSI``, ``SDWI``, ``Span_*``, and
      linear ratios ``CR_*`` / ``PR_*``).
    * **Linear power to dB**: ``dB = 10 * log10(power)`` for ``power > 0``
      (required by dB-domain indices such as ``CR_VH_VV_dB``, ``CR_HV_HH_dB``,
      ``PR_HH_VV_dB``, and ``Mean_VV_VH_dB``; pixels with ``power <= 0`` are
      masked as ``out_no_data_val``).

    Computed index values are scaled to output pixel values via
    ``output = (index_val * out_gain) + out_offset``. When an integer
    ``datatype`` is used (e.g., ``rsgispy.TYPE_16INT``), valid values are
    rounded to the nearest integer and clamped to the target data type range.

    Canonical polarisation band names accepted in ``band_names``: ``'vv'``,
    ``'vh'``, ``'hh'``, and ``'hv'`` (case-insensitive; prefixes such as
    ``sigma0_vv``, ``gamma0_vh``, or ``beta0_hh`` are also recognised).

    Full equations, working domains, and literature references for all
    supported SAR indices are documented in
    :doc:`imagecalc/calcindices <../imagecalc/calcindices>` and inspectable via
    :func:`get_available_sar_indices`, :func:`get_sar_index_req_bands`,
    :func:`get_sar_index_equation`, and :func:`get_sar_index_units`.

    :param input_img: Path to the input multi-band SAR raster image file.
    :param output_img: Path to the output multi-band raster image file.
    :param band_names: Mapping from 1-based band index to polarisation name
                       (e.g., ``{1: "VV", 2: "VH"}`` or
                       ``{1: "HH", 2: "HV", 3: "VV"}``).
    :param indices: Sequence of SAR index names to calculate and stack into
                    ``output_img`` in the specified order.
    :param in_units: Units of the physical input backscatter values (after
                     applying ``in_gain`` and ``in_offset``). Must be
                     ``'dB'`` (decibels; aliases ``'decibel'``, ``'decibels'``)
                     or ``'power'`` (linear intensity; aliases ``'linear'``,
                     ``'lin'``).
    :param in_gain: Scalar, per-band sequence (length matching input band count),
                    or mapping (keyed by band index or polarisation name) used
                    to unscale input values via ``(dn - in_offset) / in_gain``
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
                         (e.g., ``{"DPSVI": {"vv_max": 1.25}}`` to supply an
                         explicit linear-power ``vv_max`` rather than computing
                         the scene maximum).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional input no-data value. If ``None``, reads the
                        no-data value(s) from the input image header.
    :param out_no_data_val: Optional output no-data value. If ``None``, uses
                            ``-999.0`` for signed/floating-point types (or
                            ``0`` / type minimum for narrow integer types).
    :param calc_stats: If ``True``, populate band statistics, histograms, and
                       overview pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.calcindices

        # Sentinel-1 VV/VH input in dB — automatically converts to linear power
        # for RVI_VV_VH, DpRVIc_VV_VH, and SDWI while keeping dB for CR_VH_VV_dB
        rsgispy.imagecalc.calcindices.calc_sar_indices(
            "s1_vv_vh_db.tif",
            "s1_indices.tif",
            band_names={1: "VV", 2: "VH"},
            indices=["RVI_VV_VH", "DpRVIc_VV_VH", "SDWI", "CR_VH_VV_dB"],
            in_units="dB",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )

        # ALOS PALSAR HH/HV input in linear power
        rsgispy.imagecalc.calcindices.calc_sar_indices(
            "palsar_hh_hv_power.tif",
            "palsar_indices.tif",
            band_names={1: "HH", 2: "HV"},
            indices=["RFDI", "RVI_HH_HV", "CR_HV_HH_dB", "Span_HH_HV_dB"],
            in_units="power",
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    norm_units = normalise_sar_units(in_units)
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
        setup = _setup_sar_indices(
            band_names, indices, (in_gain, in_offset, out_gain, out_offset),
            n_bands, band_nodata, input_img,
        )

        if any(spec.requires_vv_max for _, spec in setup.resolved_indices):
            dpsvi_params = parsed_idx_params.get("dpsvi", {})
            if "vv_max" not in dpsvi_params:
                vv_max = _calc_sar_vv_max(
                    src=src,
                    canonical_to_idx=setup.canonical_to_idx,
                    in_nodata_map=setup.in_nodata_map,
                    in_gain_map=setup.in_gain_map,
                    in_offset_map=setup.in_offset_map,
                    in_units=norm_units,
                    use_pbar=use_pbar,
                )
                parsed_idx_params["dpsvi"] = {
                    **dpsvi_params,
                    "vv_max": vv_max,
                }

    # Read only the polarisations the indices need.
    src_bands = sorted({setup.canonical_to_idx[b] for b in setup.req_band_set})

    def _setup_fn(
        n_bands_: int, band_nodata_: Sequence[Optional[float]], label: str
    ) -> IndicesSetup:
        del n_bands_, band_nodata_, label
        return setup

    node = IndicesNode(
        RasterSource(input_img, bands=src_bands),
        _setup_fn,
        _sar_block_fn(norm_units, parsed_idx_params, out_nd, out_np_dtype),
        out_np_dtype,
        out_nd,
        src_bands=src_bands,
        name="calc_sar_indices",
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
