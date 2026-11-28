# pylint: disable=too-many-lines
"""
The imagecalc.rescale module provides gain/offset and per-band min-max raster
pixel rescaling utilities using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio

import rsgispy
from rsgispy.imagecalc._norm_helpers import (
    ALL_NORM_METHODS,
    DEFAULT_TRIM_METHODS,
    PERCENTILE_NORM_METHODS,
    PIXEL_VECTOR_METHODS,
    SUMMARY_NORM_METHODS,
    SUMMARY_SKLEARN_METHODS,
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
    summary_band_params,
    summary_sklearn_band_params,
)
from rsgispy._tilegraph._wrap import (
    GridRasterSource,
    WrapperDiskSink,
    run_wrapper_graph,
)
from rsgispy.imageprocess import (
    NodeMeta,
    PointwiseOp,
    RasterSource,
    SetNoData,
    StackBands,
    TileContext,
    TileNode,
    TileTensor,
    build_valid_mask,
    cast_result_block,
    compute_spatial_intersection,
    ensure_file_exists,
    fits_dtype_range,
    iter_dataset_windows,
    open_multi_image_datasets,
    validate_band_index,
)
from rsgispy.imageprocess import resolve_out_nodata as _ip_resolve_out_nodata
from rsgispy.imageutils.metadata import get_band_names
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import read_json_to_dict, write_dict_to_json

__all__ = [
    "ALL_NORM_METHODS",
    "ImageBandRescale",
    "calc_img_rescale",
    "calc_img_rescale_node",
    "rescale_img_pxl_vals",
    "rescale_img_pxl_vals_node",
    "calc_img_norm_params",
    "norm_params_from_stats",
    "STATS_NORM_METHODS",
    "apply_img_norm_params",
    "img_norm_params_node",
    "normalise_img_pxl_vals",
]


class ImageBandRescale:
    """
    Data container storing per-band min-max rescaling parameters for
    :func:`rescale_img_pxl_vals`.

    :param band: 1-based image band index.
    :param in_min: Input image band minimum value for rescaling.
    :param in_max: Input image band maximum value for rescaling.
    :param no_data_val: No-data value of the input image band. If ``None``
                        (the default), the input band's own no-data value is
                        used; if the band has none, no value is masked
                        (non-finite values are always masked). Give a value
                        only to override the band's no-data value.
    :param out_min: Output image band minimum value after rescaling.
    :param out_max: Output image band maximum value after rescaling.
    :param out_no_data: Value written to the output for masked pixels and
                        recorded as the output band's no-data value. If
                        ``None`` (the default), ``-9999`` is used, or for
                        integer outputs that cannot hold it, the data type's
                        maximum (unsigned) or minimum (signed) value. Choose a
                        value outside ``[out_min, out_max]``.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods
    def __init__(
        self,
        band: int = 0,
        in_min: float = 0.0,
        in_max: float = 0.0,
        no_data_val: Optional[float] = None,
        out_min: float = 0.0,
        out_max: float = 0.0,
        out_no_data: Optional[float] = None,
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


def _resolve_rescale_out_nodata(
    out_no_data: Optional[float], out_np_dtype: np.dtype, band: int
) -> float:
    """
    Resolve the output no-data value of one band for :func:`rescale_img_pxl_vals`.

    ``None`` gives ``-9999`` (or, for integer data types that cannot hold it,
    the type's maximum for unsigned and minimum for signed types). An explicit
    value must be representable by ``out_np_dtype``.
    """
    out_np_dtype = np.dtype(out_np_dtype)
    if out_no_data is None:
        # -9999, clamped to the max (unsigned) or min (signed) integer value.
        dflt_nd = _ip_resolve_out_nodata(
            out_np_dtype, -9999.0, clamp_int=True, unsigned_clamp="max"
        )
        return float(dflt_nd)
    nd_val = float(out_no_data)
    if math.isnan(nd_val) and np.issubdtype(out_np_dtype, np.floating):
        return nd_val
    if not fits_dtype_range(nd_val, out_np_dtype):
        raise rsgispy.RSGISPyException(
            f"out_no_data ({out_no_data}) for band {band} cannot be represented "
            f"by the output data type '{out_np_dtype.name}'."
        )
    return nd_val


def _same_nodata(val_a: float, val_b: float) -> bool:
    """Return ``True`` if two no-data values are equal (``NaN`` equals ``NaN``)."""
    return val_a == val_b or (math.isnan(val_a) and math.isnan(val_b))


def _check_rescale_objs(
    band_rescale_objs: Sequence[ImageBandRescale], out_np_dtype: np.dtype
) -> Tuple[Dict[int, ImageBandRescale], Dict[int, float]]:
    """
    Validate :class:`ImageBandRescale` objects; return ``{band_idx: obj}``
    and ``{band_idx: out_no_data}`` (0-based band indices).
    """
    if not band_rescale_objs:
        raise rsgispy.RSGISPyException(
            "band_rescale_objs must contain at least one ImageBandRescale."
        )
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
    out_nds = {
        b_idx: _resolve_rescale_out_nodata(cfg.out_no_data, out_np_dtype, b_idx + 1)
        for b_idx, cfg in rescale_dict.items()
    }
    return rescale_dict, out_nds


def _rescale_band_out_nds(
    rescale_dict: Mapping[int, ImageBandRescale],
    out_nds: Mapping[int, float],
    n_bands: int,
) -> List[float]:
    """Output no-data value of each of ``n_bands`` bands (all need an object)."""
    for b_idx in range(n_bands):
        if b_idx not in rescale_dict:
            raise rsgispy.RSGISPyException(
                f"Missing ImageBandRescale specification for band {b_idx + 1}."
            )
    return [out_nds[b_idx] for b_idx in range(n_bands)]


class _RescalePxlNode(TileNode):
    """Node of :func:`rescale_img_pxl_vals_node`."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes,too-many-locals

    def __init__(
        self,
        source: TileNode,
        rescale_dict: Dict[int, ImageBandRescale],
        out_nds: Dict[int, float],
        trim_to_limits: bool,
        out_np_dtype: np.dtype,
        *,
        no_data_val: Optional[float] = None,
        band_nodata: Optional[Sequence[Optional[float]]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name or "rescale_img_pxl_vals")
        self.rescale_dict = rescale_dict
        self.out_nds = out_nds
        self.trim_to_limits = bool(trim_to_limits)
        self.out_np_dtype = np.dtype(out_np_dtype)
        self.no_data_val = no_data_val
        self.band_nodata = list(band_nodata) if band_nodata is not None else None
        self._band_out_nds: List[float] = []
        self._in_nds: List[Optional[float]] = []

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        if meta.n_bands is None:
            raise rsgispy.RSGISPyException(
                f"{self.label} needs an input with a known number of bands."
            )
        n_bands = int(meta.n_bands)
        self._band_out_nds = _rescale_band_out_nds(self.rescale_dict, self.out_nds, n_bands)
        if self.band_nodata is not None:
            src_nds = self.band_nodata
        else:
            src_nds = [
                self.no_data_val if self.no_data_val is not None else meta.nodata
            ] * n_bands
        self._in_nds = [
            float(self.rescale_dict[b_idx].no_data_val)
            if self.rescale_dict[b_idx].no_data_val is not None
            else src_nds[b_idx]
            for b_idx in range(n_bands)
        ]
        first = self._band_out_nds[0]
        single = all(_same_nodata(nd, first) for nd in self._band_out_nds)
        return NodeMeta(
            n_bands=n_bands,
            band_names=meta.band_names,
            dtype=self.out_np_dtype,
            nodata=first if single else None,
        )

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = inputs[0].crop(halo_y, halo_x)
        in_block = np.asarray(aligned.data, dtype=np.float64)
        win_h, win_w = int(in_block.shape[1]), int(in_block.shape[2])
        out_bands = []
        for b_idx in range(in_block.shape[0]):
            cfg = self.rescale_dict[b_idx]
            band_arr = in_block[b_idx]
            nodata_mask = ~build_valid_mask(band_arr, no_data_val=self._in_nds[b_idx])
            scaled = (
                ((band_arr - cfg.in_min) / (cfg.in_max - cfg.in_min))
                * (cfg.out_max - cfg.out_min)
            ) + cfg.out_min
            if self.trim_to_limits:
                low = min(cfg.out_min, cfg.out_max)
                high = max(cfg.out_min, cfg.out_max)
                scaled = np.clip(scaled, low, high)
            out_bands.append(
                cast_result_block(
                    scaled,
                    win_h,
                    win_w,
                    self.out_np_dtype,
                    nodata=self._band_out_nds[b_idx],
                    valid_mask=~nodata_mask,
                )
            )
        return aligned.with_data(np.stack(out_bands, axis=0))


def rescale_img_pxl_vals_node(
    source: TileNode,
    band_rescale_objs: Sequence[ImageBandRescale] = (),
    trim_to_limits: bool = True,
    *,
    out_dtype: Any = np.float32,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`rescale_img_pxl_vals`: per-band min-max
    rescaling with :class:`ImageBandRescale` objects (one for every band of
    ``source``), for chaining in a :class:`~rsgispy.imageprocess.TileGraph`.

    Masked input pixels (a band's ``no_data_val``, by default the no-data
    value of ``source`` or ``no_data_val``, and non-finite values) are set to
    the band's ``out_no_data``. The blocks have the data type ``out_dtype``;
    the node's band names are those of ``source``, and its no-data value is
    the bands' ``out_no_data`` (``None`` if they differ).

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param band_rescale_objs: :class:`ImageBandRescale` objects (band numbers
                              are those of ``source``).
    :param trim_to_limits: Clamp to ``[out_min, out_max]`` (Default = ``True``).
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.float32``).
    :param no_data_val: Input no-data value for objects without one
                        (Default = ``None``: that of ``source``).
    :param name: Optional node name (Default = ``'rescale_img_pxl_vals'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.rescale import ImageBandRescale, rescale_img_pxl_vals_node
        from rsgispy.imagecalc.calcindices import calc_indices_node

        src = RasterSource("s2.tif")
        objs = [
            ImageBandRescale(band=b, in_min=0, in_max=10000, out_min=0, out_max=1)
            for b in range(1, 7)
        ]
        refl = rescale_img_pxl_vals_node(src, objs, out_dtype=np.float64)
        ndvi = calc_indices_node(refl, {3: "red", 4: "nir"}, ["NDVI"], no_data_val=-9999)
        TileGraph(DiskSink(ndvi, "ndvi.tif")).run()
    """
    # pylint: disable=too-many-arguments
    out_np_dtype = np.dtype(out_dtype)
    rescale_dict, out_nds = _check_rescale_objs(band_rescale_objs, out_np_dtype)
    return _RescalePxlNode(
        source,
        rescale_dict,
        out_nds,
        trim_to_limits,
        out_np_dtype,
        no_data_val=no_data_val,
        name=name,
    )


def _calc_img_rescale_block(
    in_block: np.ndarray,
    params: Tuple[float, float, float, float, float],
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """
    Rescale a block from ``(c_offset, c_gain)`` to ``(n_offset, n_gain)``;
    ``params`` is ``(c_no_data_val, c_offset, scale, n_offset, n_no_data_val)``.
    """
    # pylint: disable=too-many-locals
    c_nd, c_off, scale, n_off, n_nd = params
    in_block = np.asarray(in_block, dtype=np.float64)
    win_h, win_w = int(in_block.shape[1]), int(in_block.shape[2])
    out_bands = []
    for b_idx in range(in_block.shape[0]):
        band_arr = in_block[b_idx]
        nodata_mask = ~build_valid_mask(band_arr, no_data_val=float(c_nd))
        rescaled = ((band_arr - c_off) * scale) + n_off
        out_bands.append(
            cast_result_block(
                rescaled,
                win_h,
                win_w,
                out_np_dtype,
                nodata=n_nd,
                valid_mask=~nodata_mask,
            )
        )
    return np.stack(out_bands, axis=0)


def calc_img_rescale_node(
    source: TileNode,
    c_no_data_val: float = 0.0,
    c_offset: float = 0.0,
    c_gain: float = 1.0,
    n_no_data_val: float = 0.0,
    n_offset: float = 0.0,
    n_gain: float = 1.0,
    *,
    out_dtype: Any = np.float32,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`calc_img_rescale`: converts every band of
    ``source`` from ``(c_offset, c_gain)`` to ``(n_offset, n_gain)`` scaling,
    ``(((val - c_offset) / c_gain) * n_gain) + n_offset``, with pixels equal
    to ``c_no_data_val`` (or non-finite) set to ``n_no_data_val``. Stack
    several images with :class:`~rsgispy.imageprocess.StackBands` first.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param c_no_data_val: Current no-data value.
    :param c_offset: Current offset.
    :param c_gain: Current gain (must be non-zero).
    :param n_no_data_val: New no-data value (the node's no-data value).
    :param n_offset: New offset.
    :param n_gain: New gain.
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.float32``).
    :param name: Optional node name (Default = ``'calc_img_rescale'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, StackBands, TileGraph
        from rsgispy.imagecalc.rescale import calc_img_rescale_node

        stack = StackBands([RasterSource("b1.tif"), RasterSource("b2.tif")])
        refl = calc_img_rescale_node(stack, c_gain=10000.0, n_no_data_val=-9999.0)
        TileGraph(DiskSink(refl, "refl.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if c_gain == 0.0:
        raise rsgispy.RSGISPyException("c_gain must not be zero.")
    out_np_dtype = np.dtype(out_dtype)
    params = (
        float(c_no_data_val),
        float(c_offset),
        float(n_gain) / float(c_gain),
        float(n_offset),
        n_no_data_val,
    )

    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        return inputs[0].replace(dtype=out_np_dtype, nodata=n_no_data_val)

    return PointwiseOp(
        source,
        lambda arr: _calc_img_rescale_block(arr, params, out_np_dtype),
        name=name or "calc_img_rescale",
        meta_fn=_meta,
    )


def calc_img_rescale(
    input_imgs: Union[str, Sequence[str]],
    output_img: str,
    c_no_data_val: float = 0.0,
    c_offset: float = 0.0,
    c_gain: float = 1.0,
    n_no_data_val: float = 0.0,
    n_offset: float = 0.0,
    n_gain: float = 1.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
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
    :param c_no_data_val: Current no-data value in the input imagery.
    :param c_offset: Current offset value.
    :param c_gain: Current gain value (must be non-zero).
    :param n_no_data_val: New no-data value written to the output image.
    :param n_offset: New offset value.
    :param n_gain: New gain value.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale

        rsgispy.imagecalc.rescale.calc_img_rescale(
            input_imgs=["img_b1.tif", "img_b2.tif"],
            output_img="rescaled.tif",
            c_no_data_val=0.0,
            c_offset=0.0,
            c_gain=10000.0,
            n_no_data_val=-9999.0,
            n_offset=0.0,
            n_gain=1.0,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if c_gain == 0.0:
        raise rsgispy.RSGISPyException("c_gain must not be zero.")

    img_list = [input_imgs] if isinstance(input_imgs, str) else list(input_imgs)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)

    sources = [GridRasterSource(path, grid, grid.offsets[path]) for path in img_paths]
    if len(sources) > 1:
        # The rescale applies c_no_data_val to every band itself, so the
        # stack shares it and does not mask across the images.
        stack_in: TileNode = StackBands(
            [SetNoData(src, c_no_data_val) for src in sources]
        )
    else:
        stack_in = sources[0]
    node = calc_img_rescale_node(
        stack_in,
        c_no_data_val=c_no_data_val,
        c_offset=c_offset,
        c_gain=c_gain,
        n_no_data_val=n_no_data_val,
        n_offset=n_offset,
        n_gain=n_gain,
        out_dtype=out_np_dtype,
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=n_no_data_val,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def rescale_img_pxl_vals(
    input_img: str,
    output_img: str,
    band_rescale_objs: Sequence[ImageBandRescale] = (),
    trim_to_limits: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Rescale image pixel values per band using a sequence of
    :class:`ImageBandRescale` specifications.

    Pixels equal to a band's input no-data value (by default the value stored
    in ``input_img``) or non-finite are written as that band's
    ``out_no_data``, which is also recorded as the output no-data value.
    Formats such as GeoTIFF store one no-data value for all bands, so they
    need the same ``out_no_data`` for every band; an exception is raised
    otherwise.

    :param input_img: Path to the input image file.
    :param output_img: Path to the output image file.
    :param band_rescale_objs: Sequence of :class:`ImageBandRescale` objects
                              defining per-band input/output ranges.
    :param trim_to_limits: If ``True``, clamp non-nodata outputs to the
                           ``[out_min, out_max]`` interval (Default = ``True``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.rescale

        # Reflectance x 10000 to 0-1; masks the image's own no-data value
        # and writes -9999 (the default out_no_data) as the output no data
        rescale_objs = [
            rsgispy.imagecalc.rescale.ImageBandRescale(
                band=1,
                in_min=0.0,
                in_max=10000.0,
                out_min=0.0,
                out_max=1.0,
            ),
        ]
        rsgispy.imagecalc.rescale.rescale_img_pxl_vals(
            input_img="img.tif",
            output_img="rescaled.tif",
            band_rescale_objs=rescale_objs,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    rescale_dict, out_nds = _check_rescale_objs(band_rescale_objs, out_np_dtype)

    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [input_img])
        src = datasets[input_img]
        n_bands = src.count
        band_out_nds = _rescale_band_out_nds(rescale_dict, out_nds, n_bands)
        single_nd = all(_same_nodata(nd, band_out_nds[0]) for nd in band_out_nds)
        if not single_nd and gdalformat.upper().strip() in ("GTIFF", "COG"):
            raise rsgispy.RSGISPyException(
                f"The bands have different out_no_data values ({band_out_nds}), "
                f"but {gdalformat} stores one no-data value for all bands: use "
                "the same out_no_data for every band or another format."
            )
        src_nodatavals = list(src.nodatavals)

    node = _RescalePxlNode(
        RasterSource(input_img),
        rescale_dict,
        out_nds,
        trim_to_limits,
        out_np_dtype,
        band_nodata=src_nodatavals,
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=band_out_nds[0] if single_nd else None,
        per_band_nodata=None if single_nd else band_out_nds,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


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


#: Methods :func:`norm_params_from_stats` supports.
STATS_NORM_METHODS = SUMMARY_NORM_METHODS + SUMMARY_SKLEARN_METHODS


def _hist_percentiles(
    hist: np.ndarray, bin_edges: np.ndarray, percents: Sequence[float]
) -> np.ndarray:
    """
    Percentiles of the values a histogram summarises, assuming the values of
    each bin are spread evenly over it (exact to within a bin width).
    """
    counts = np.asarray(hist, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(counts)])
    total = cum[-1]
    targets = np.asarray(percents, dtype=np.float64) / 100.0 * total
    return np.interp(targets, cum, np.asarray(bin_edges, dtype=np.float64))


def _hist_mad(hist: np.ndarray, bin_edges: np.ndarray, median: float) -> float:
    """Median absolute deviation from ``median`` of a histogram's values."""
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = float(np.max(np.diff(bin_edges)))
    dev = np.abs(centres - median)
    order = np.argsort(dev)
    dev_sorted = dev[order]
    cnt_sorted = np.asarray(hist, dtype=np.float64)[order]
    # Spread each bin's deviations over [dev - w/2, dev + w/2] (clipped at 0).
    lo_edges = np.maximum(dev_sorted - width / 2.0, 0.0)
    edges = np.concatenate([lo_edges, [dev_sorted[-1] + width / 2.0]])
    edges = np.maximum.accumulate(edges)
    cum = np.concatenate([[0.0], np.cumsum(cnt_sorted)])
    return float(np.interp(0.5 * cum[-1], cum, edges))


def _hist_order_stats(hist: np.ndarray, bin_edges: np.ndarray) -> Tuple[Any, Any, Any]:
    """``(percentile_fn, median_fn, mad_fn)`` of a band's histogram."""

    def _percentiles(percents: Sequence[float]) -> np.ndarray:
        return _hist_percentiles(hist, bin_edges, percents)

    def _median() -> float:
        return float(_hist_percentiles(hist, bin_edges, [50.0])[0])

    def _mad(median: float) -> float:
        return _hist_mad(hist, bin_edges, median)

    return _percentiles, _median, _mad


def norm_params_from_stats(
    stats: Any,
    method: Union[str, int] = "linear",
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    bands: Optional[Sequence[int]] = None,
    n_stdevs: float = 2.0,
    lower: float = 2.0,
    upper: float = 98.0,
    in_min_max: Optional[
        Union[Mapping[str, float], Sequence[Mapping[str, float]]]
    ] = None,
    method_kwargs: Optional[Mapping[str, Any]] = None,
    out_stats_file: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """
    Normalisation parameters from the per-band statistics of a
    :class:`~rsgispy.imageprocess.StatsSink`
    (:class:`~rsgispy.imageprocess.RasterStats`), in the format of
    :func:`calc_img_norm_params`. This makes a normalisation inside a
    :class:`~rsgispy.imageprocess.TileGraph` a two-pass workflow without
    writing the image: a ``StatsSink`` on any node (pass 1), then
    :func:`img_norm_params_node` with these parameters (pass 2). The result
    is accepted unchanged by :func:`img_norm_params_node`,
    :func:`apply_img_norm_params` and, written with ``out_stats_file``, by
    their ``stats_file`` argument.

    The band numbers are those of the node the ``StatsSink`` was given, and
    each band's no-data value is the statistics' ``nodata``.

    Supported methods:

    * exact from the statistics: ``'none'``, ``'user'`` / ``'manual'``,
      ``'linear'``, ``'stdev'``, ``'zscore'``, ``'decimal_scaling'``,
      ``'standard_scaler'``, ``'minmax_scaler'`` and ``'maxabs_scaler'``
      (they agree with :func:`calc_img_norm_params` on the same pixels to
      rounding error);
    * from the histogram: ``'cumulative'``, ``'robust_zscore'`` and
      ``'robust_scaler'`` need percentiles (the median, quartiles or the
      ``lower`` / ``upper`` range), which are interpolated from the
      histogram (``StatsSink(..., bins=..., hist_range=...)``, with a range
      that covers every value) and so are approximate: the share of pixels
      below each estimate differs from the requested one by at most the
      share of pixels in one bin (use more bins for a closer estimate);
    * other methods (fitted transformers such as ``'log'``,
      ``'power_transformer'`` or ``'quantile_transformer'``, and the
      pixel-vector norms such as ``'l1'``, ``'l2'`` or ``'softmax'``) need
      the pixel values: use :func:`calc_img_norm_params` on a file.

    The standard deviations must be population standard deviations (the
    ``StatsSink`` default ``ddof=0``), as :func:`calc_img_norm_params` uses.

    :param stats: :class:`~rsgispy.imageprocess.RasterStats` returned by a
                  :class:`~rsgispy.imageprocess.StatsSink`.
    :param method: Normalisation method name or ``IMG_STRETCH_*`` constant
                   (see above; Default = ``'linear'``).
    :param out_min: Optional output minimum (as :func:`calc_img_norm_params`).
    :param out_max: Optional output maximum (as :func:`calc_img_norm_params`).
    :param bands: Optional 1-based bands to include (Default = all bands).
    :param n_stdevs: Number of standard deviations for ``'stdev'``
                     (Default = ``2.0``).
    :param lower: Lower percentile for ``'cumulative'`` (Default = ``2.0``).
    :param upper: Upper percentile for ``'cumulative'`` (Default = ``98.0``).
    :param in_min_max: ``{'min': ..., 'max': ...}`` or one per band, for
                       ``'user'``.
    :param method_kwargs: Optional scaler options (e.g. ``with_mean`` for
                          ``'standard_scaler'``, ``feature_range`` for
                          ``'minmax_scaler'``, ``quantile_range`` for
                          ``'robust_scaler'``).
    :param out_stats_file: Optional JSON file to write the parameters to.
    :return: Normalisation parameters dictionary.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, StatsSink, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node
        from rsgispy.imagecalc.rescale import img_norm_params_node, norm_params_from_stats

        idx = calc_indices_node(RasterSource("s2.tif"),
                                {2: "Green", 3: "Red", 4: "NIR"}, ["NDVI", "NDWI"])
        stats = TileGraph(StatsSink(idx, name="st")).run()["st"]  # pass 1
        params = norm_params_from_stats(stats, method="zscore")
        TileGraph(DiskSink(img_norm_params_node(idx, params),
                           "indices_zscore.tif")).run()  # pass 2
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    for attr in ("count", "min", "max", "mean", "std"):
        if not hasattr(stats, attr):
            raise rsgispy.RSGISPyException(
                "stats must be the RasterStats result of a StatsSink."
            )
    norm_method = resolve_norm_method(method)
    if norm_method not in STATS_NORM_METHODS:
        raise rsgispy.RSGISPyException(
            f"norm_params_from_stats does not support method {norm_method!r}: it "
            "needs the pixel values (use calc_img_norm_params on a file). "
            f"Supported methods: {', '.join(STATS_NORM_METHODS)}."
        )
    if int(getattr(stats, "ddof", 0)) != 0:
        raise rsgispy.RSGISPyException(
            "norm_params_from_stats needs population standard deviations; "
            f"use StatsSink(..., ddof=0), not ddof={stats.ddof}."
        )
    eff_out_min, eff_out_max = resolve_out_bounds(norm_method, out_min, out_max)
    if norm_method == "cumulative" and not 0.0 <= float(lower) < float(upper) <= 100.0:
        raise rsgispy.RSGISPyException(
            f"cumulative stretch requires 0 <= lower < upper <= 100, got ({lower}, {upper})."
        )
    if norm_method == "stdev" and float(n_stdevs) <= 0.0:
        raise rsgispy.RSGISPyException("n_stdevs must be strictly positive.")
    hist = getattr(stats, "hist", None)
    bin_edges = getattr(stats, "bin_edges", None)
    if norm_method in PERCENTILE_NORM_METHODS and (hist is None or bin_edges is None):
        raise rsgispy.RSGISPyException(
            f"Method {norm_method!r} needs percentiles: give the StatsSink a "
            "histogram (bins=..., hist_range=...) covering every value."
        )

    n_bands = int(np.asarray(stats.count).size)
    sel_bands = list(range(1, n_bands + 1)) if bands is None else [int(b) for b in bands]
    if not sel_bands:
        raise rsgispy.RSGISPyException("bands must not be empty.")
    eff_nd = None if stats.nodata is None else float(stats.nodata)
    band_entries: List[Dict[str, Any]] = []
    for sel_pos, b_idx in enumerate(sel_bands):
        if not 1 <= b_idx <= n_bands:
            raise rsgispy.RSGISPyException(
                f"Band {b_idx} is not in the statistics ({n_bands} band(s))."
            )
        i_b = b_idx - 1
        if int(stats.count[i_b]) == 0:
            raise rsgispy.RSGISPyException(
                f"Band {b_idx} contains no valid finite non-nodata pixels."
            )
        data_min = float(stats.min[i_b])
        data_max = float(stats.max[i_b])
        mean_val = float(stats.mean[i_b])
        std_val = float(stats.std[i_b])
        pct_fn = med_fn = mad_fn = None
        if norm_method in PERCENTILE_NORM_METHODS:
            b_hist = np.asarray(hist)[i_b]
            if int(b_hist.sum()) != int(stats.count[i_b]):
                raise rsgispy.RSGISPyException(
                    f"The histogram of band {b_idx} holds {int(b_hist.sum())} of its "
                    f"{int(stats.count[i_b])} pixels; set the StatsSink hist_range "
                    f"to cover every value ({data_min} to {data_max})."
                )
            pct_fn, med_fn, mad_fn = _hist_order_stats(
                b_hist, np.asarray(bin_edges, dtype=np.float64)
            )
        if norm_method in SUMMARY_NORM_METHODS:
            u_bounds = (
                _extract_user_bounds(in_min_max, sel_pos, len(sel_bands))
                if norm_method == "user"
                else None
            )
            extra, transform = summary_band_params(
                norm_method, data_min, data_max, mean_val, std_val,
                eff_out_min, eff_out_max, lower, upper, n_stdevs, u_bounds,
                percentile_fn=pct_fn, median_fn=med_fn, mad_fn=mad_fn,
            )
        else:
            extra, transform = summary_sklearn_band_params(
                norm_method, data_min, data_max, mean_val, std_val,
                eff_out_min, eff_out_max, dict(method_kwargs or {}),
                percentile_fn=pct_fn,
            )
        # Every supported transform is monotonic increasing, so the
        # transformed range is the transform of the data range.
        trans = transform(np.array([data_min, data_max], dtype=np.float64))
        band_rec: Dict[str, Any] = {
            "band": int(b_idx),
            "no_data_val": eff_nd,
            "in_min": data_min,
            "in_max": data_max,
            "mean": mean_val,
            "std": std_val,
            "trans_min": float(np.min(trans)),
            "trans_max": float(np.max(trans)),
            "out_min": eff_out_min,
            "out_max": eff_out_max,
        }
        band_rec.update(extra)
        band_entries.append(band_rec)

    result: Dict[str, Any] = {
        "version": 1,
        "method": norm_method,
        "input_img": None,
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


def _load_norm_params(
    norm_params: Optional[Mapping[str, Any]],
    stats_file: Optional[Union[str, os.PathLike]],
) -> Mapping[str, Any]:
    """Return the normalisation parameters given directly or in a JSON file."""
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
    return params_dict


class _NormApply:
    """Resolved settings for applying normalisation parameters."""

    # pylint: disable=too-few-public-methods,too-many-arguments
    # pylint: disable=too-many-positional-arguments

    def __init__(
        self,
        method: str,
        band_specs: Sequence[Mapping[str, Any]],
        eff_out_min: Optional[float],
        eff_out_max: Optional[float],
        eff_trim: bool,
        eff_out_nd: Optional[float],
    ) -> None:
        self.method = method
        self.band_specs = band_specs
        self.eff_out_min = eff_out_min
        self.eff_out_max = eff_out_max
        self.eff_trim = eff_trim
        self.eff_out_nd = eff_out_nd


def _resolve_norm_apply(
    params_dict: Mapping[str, Any],
    out_min: Optional[float],
    out_max: Optional[float],
    trim_to_limits: Optional[bool],
    out_no_data_val: Optional[float],
    out_np_dtype: np.dtype,
) -> _NormApply:
    """Resolve the method, band specs, output bounds, trimming and no data."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
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
    eff_out_nd = resolve_out_nodata(band_specs, out_no_data_val, out_np_dtype)
    return _NormApply(
        norm_method, band_specs, eff_out_min, eff_out_max, eff_trim, eff_out_nd
    )


def _apply_norm_block(
    in_block: np.ndarray, norm: _NormApply, out_np_dtype: np.dtype
) -> np.ndarray:
    """Normalise a ``(n_selected_bands, H, W)`` ``float64`` block."""
    # pylint: disable=too-many-locals
    win_h, win_w = int(in_block.shape[1]), int(in_block.shape[2])
    band_specs = norm.band_specs
    eff_out_nd = norm.eff_out_nd
    out_bands = []
    if norm.method in PIXEL_VECTOR_METHODS:
        valid_masks = [
            build_valid_mask(in_block[idx], band_specs[idx].get("no_data_val"))
            for idx in range(len(band_specs))
        ]
        vec_out = apply_pixel_vector_norm(
            method=norm.method,
            block=in_block,
            valid_masks=valid_masks,
            norm_type=str(band_specs[0].get("norm", "l2")),
            eff_out_min=norm.eff_out_min,
            eff_out_max=norm.eff_out_max,
            trim_to_limits=norm.eff_trim,
            vec_min=band_specs[0].get("vec_min"),
            vec_max=band_specs[0].get("vec_max"),
        )
        for idx in range(len(band_specs)):
            band_res = vec_out[idx]
            out_bands.append(
                cast_result_block(
                    band_res,
                    win_h,
                    win_w,
                    out_np_dtype,
                    nodata=eff_out_nd,
                    valid_mask=None if eff_out_nd is None else np.isfinite(band_res),
                )
            )
    else:
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
                    method=norm.method,
                    vals=arr[valid],
                    band_params=b_spec,
                    eff_out_min=norm.eff_out_min,
                    eff_out_max=norm.eff_out_max,
                    trim_to_limits=norm.eff_trim,
                )
            out_bands.append(
                cast_result_block(
                    res,
                    win_h,
                    win_w,
                    out_np_dtype,
                    nodata=eff_out_nd,
                    valid_mask=None if eff_out_nd is None else valid,
                )
            )
    return np.stack(out_bands, axis=0)


class _NormParamsNode(TileNode):
    """Node of :func:`img_norm_params_node`."""

    def __init__(
        self,
        source: TileNode,
        norm: _NormApply,
        out_np_dtype: np.dtype,
        *,
        preselected: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name or "img_norm_params")
        self.norm = norm
        self.out_np_dtype = np.dtype(out_np_dtype)
        self.sel_bands = [int(b_spec["band"]) for b_spec in norm.band_specs]
        self.preselected = bool(preselected)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        names = None
        if self.preselected:
            names = meta.band_names
        elif meta.n_bands is not None:
            for b_idx in self.sel_bands:
                validate_band_index(b_idx, int(meta.n_bands), f"input of {self.label}")
            if meta.band_names is not None:
                names = tuple(meta.band_names[b - 1] for b in self.sel_bands)
        return NodeMeta(
            n_bands=len(self.sel_bands),
            band_names=names,
            dtype=self.out_np_dtype,
            nodata=self.norm.eff_out_nd,
        )

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = inputs[0].crop(halo_y, halo_x)
        data = aligned.data
        if not self.preselected:
            data = data[[b - 1 for b in self.sel_bands]]
        in_block = np.asarray(data, dtype=np.float64)
        return aligned.with_data(_apply_norm_block(in_block, self.norm, self.out_np_dtype))


def img_norm_params_node(
    source: TileNode,
    norm_params: Optional[Mapping[str, Any]] = None,
    stats_file: Optional[Union[str, os.PathLike]] = None,
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    trim_to_limits: Optional[bool] = None,
    *,
    out_dtype: Any = np.float32,
    out_no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`apply_img_norm_params`: applies normalisation
    parameters calculated beforehand with :func:`calc_img_norm_params` (given
    directly or as its JSON file) to the bands of ``source`` that they
    select, for chaining in a :class:`~rsgispy.imageprocess.TileGraph`.

    Normalisation parameters are whole-image statistics, so they must be
    calculated first (a two-pass workflow); applying them is tile-local. The
    band numbers in the parameters are those of ``source``. The blocks have
    the data type ``out_dtype`` and the node's no-data value is resolved as
    in :func:`apply_img_norm_params`.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param norm_params: Parameters returned by :func:`calc_img_norm_params`.
    :param stats_file: JSON file written by :func:`calc_img_norm_params`.
    :param out_min: Optional override of the output minimum.
    :param out_max: Optional override of the output maximum.
    :param trim_to_limits: Clamp to ``[out_min, out_max]`` (default by method).
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.float32``).
    :param out_no_data_val: Optional output no-data value.
    :param name: Optional node name (Default = ``'img_norm_params'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.rescale import calc_img_norm_params, img_norm_params_node

        params = calc_img_norm_params("s2.tif", method="zscore")  # pass 1
        src = RasterSource("s2.tif")
        norm = img_norm_params_node(src, params, out_dtype=np.float64)  # pass 2
        TileGraph(DiskSink(norm, "s2_zscore.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    params_dict = _load_norm_params(norm_params, stats_file)
    out_np_dtype = np.dtype(out_dtype)
    norm = _resolve_norm_apply(
        params_dict, out_min, out_max, trim_to_limits, out_no_data_val, out_np_dtype
    )
    return _NormParamsNode(source, norm, out_np_dtype, name=name)


def apply_img_norm_params(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    norm_params: Optional[Mapping[str, Any]] = None,
    stats_file: Optional[Union[str, os.PathLike]] = None,
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
    trim_to_limits: Optional[bool] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    out_no_data_val: Optional[float] = None,
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
    :param out_min: Optional override for output minimum value.
    :param out_max: Optional override for output maximum value.
    :param trim_to_limits: Clamp transformed values to ``[out_min, out_max]``.
                           Defaults to ``True`` for ``'stdev'``, ``'cumulative'``,
                           and ``'user'`` stretches, and ``False`` otherwise.
    :param gdalformat: GDAL/rasterio driver format for output
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param out_no_data_val: Optional explicit output no-data value.
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
    params_dict = _load_norm_params(norm_params, stats_file)
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path, "Input image")
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    norm = _resolve_norm_apply(
        params_dict, out_min, out_max, trim_to_limits, out_no_data_val, out_np_dtype
    )
    in_band_names = get_band_names(in_path)
    with ExitStack() as stack:
        datasets, _ = open_multi_image_datasets(stack, [in_path])
        src = datasets[in_path]
        sel_bands = [int(b_spec["band"]) for b_spec in norm.band_specs]
        for b_idx in sel_bands:
            validate_band_index(b_idx, src.count, in_path)

    node = _NormParamsNode(
        RasterSource(in_path, bands=sel_bands), norm, out_np_dtype, preselected=True
    )
    sink = WrapperDiskSink(
        node,
        out_path,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=norm.eff_out_nd,
        band_names=[in_band_names[b_idx - 1] for b_idx in sel_bands],
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def normalise_img_pxl_vals(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    method: Union[str, int] = "linear",
    out_min: Optional[float] = None,
    out_max: Optional[float] = None,
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
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32FLOAT,
    no_data_val: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
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
    :param out_min: Optional target output minimum value.
    :param out_max: Optional target output maximum value.
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
    :param gdalformat: GDAL/rasterio driver format for output
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32FLOAT``).
    :param no_data_val: Optional explicit input no-data value overriding metadata.
    :param out_no_data_val: Optional explicit output no-data value.
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
