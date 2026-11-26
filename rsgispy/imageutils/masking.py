"""
The imageutils.masking module provides raster masking and valid-data mask
generation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import rsgispy
from rsgispy._tilegraph._wrap import (
    GridRasterSource,
    WrapperDiskSink,
    run_wrapper_graph,
)
from rsgispy.imageprocess import (
    NodeMeta,
    PointwiseOp,
    RasterSource,
    TileContext,
    TileNode,
    TileTensor,
    build_valid_cube_mask,
    build_valid_mask,
    compute_spatial_intersection,
    normalise_creation_options,
    open_multi_image_datasets,
    rsgis_dtype_from_np_dtype,
    validate_band_index,
)
from rsgispy.imageutils.colourtable import set_img_thematic

__all__ = [
    "ImageBandInfo",
    "mask_img",
    "mask_img_node",
    "gen_finite_mask",
    "gen_finite_mask_node",
    "gen_valid_mask",
    "gen_valid_mask_node",
    "gen_img_edge_mask",
    "gen_img_edge_mask_node",
    "mask_all_band_zero_vals",
    "mask_all_band_zero_vals_node",
    "create_valid_mask",
    "create_valid_mask_node",
]


class ImageBandInfo:
    """
    Class storing an image file path, logical name, optional list of bands,
    and optional output band names.

    :param file_name: Path to the raster image file.
    :param name: Logical name describing the image.
    :param bands: Optional list of 1-based band indices to consider.
    :param band_names: Optional sequence of output band/feature names matching
                       ``bands`` (or all bands in ``file_name`` when
                       ``bands`` is ``None``).
    """

    def __init__(
        self,
        file_name: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
        bands: Optional[Sequence[int]] = None,
        band_names: Optional[Sequence[str]] = None,
    ):
        self.file_name: Optional[str] = (
            os.fspath(file_name) if file_name is not None else None
        )
        self.name = name
        self.bands: Optional[List[int]] = (
            [int(b) for b in bands] if bands is not None else None
        )
        if band_names is not None:
            names_list = [str(bn) for bn in band_names]
            if any(not bn.strip() for bn in names_list):
                raise rsgispy.RSGISPyException(
                    "band_names must contain non-empty strings."
                )
            if self.bands is not None and len(names_list) != len(self.bands):
                raise rsgispy.RSGISPyException(
                    f"Number of band_names ({len(names_list)}) must match "
                    f"number of bands ({len(self.bands)})."
                )
            self.band_names: Optional[List[str]] = names_list
        else:
            self.band_names = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        out: Dict[str, Any] = {
            "file_name": self.file_name,
            "name": self.name,
            "bands": list(self.bands) if self.bands is not None else None,
        }
        if self.band_names is not None:
            out["band_names"] = list(self.band_names)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageBandInfo":
        """Construct an :class:`ImageBandInfo` from a dictionary."""
        if not isinstance(data, Mapping):
            raise rsgispy.RSGISPyException(
                "ImageBandInfo data must be a dictionary."
            )
        return cls(
            file_name=data.get("file_name"),
            name=data.get("name"),
            bands=data.get("bands"),
            band_names=data.get("band_names"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageBandInfo):
            return NotImplemented
        return (
            self.file_name == other.file_name
            and self.name == other.name
            and self.bands == other.bands
            and self.band_names == other.band_names
        )

    def __repr__(self) -> str:
        if self.band_names is None:
            return (
                f"ImageBandInfo(file_name={self.file_name!r}, "
                f"name={self.name!r}, bands={self.bands!r})"
            )
        return (
            f"ImageBandInfo(file_name={self.file_name!r}, "
            f"name={self.name!r}, bands={self.bands!r}, "
            f"band_names={self.band_names!r})"
        )


def _mask_img_block(
    data: np.ndarray,
    msk_arr: np.ndarray,
    out_value: float,
    mask_vals: Tuple[List[float], bool],
) -> np.ndarray:
    """Set all bands of ``data`` to ``out_value`` where ``msk_arr`` is masked."""
    finite_mask_vals, has_nan_msk = mask_vals
    data = np.array(data, dtype=np.float64, copy=True)
    if finite_mask_vals:
        is_masked = np.isin(msk_arr, finite_mask_vals)
    else:
        is_masked = np.zeros(msk_arr.shape, dtype=bool)
    if has_nan_msk:
        is_masked = is_masked | np.isnan(msk_arr)
    data[:, is_masked] = float(out_value)
    return data


def _parse_mask_values(
    mask_value: Union[float, int, Sequence[Union[float, int]]],
) -> Tuple[List[float], bool]:
    """Return the finite mask values and whether ``NaN`` is a mask value."""
    if isinstance(mask_value, (int, float, np.number)):
        mask_vals = [float(mask_value)]
    else:
        mask_vals = [float(v) for v in mask_value]
    if not mask_vals:
        raise rsgispy.RSGISPyException(
            "mask_value must contain at least one numeric value."
        )
    has_nan_msk = any(math.isnan(v) for v in mask_vals)
    return [v for v in mask_vals if not math.isnan(v)], has_nan_msk


def mask_img_node(
    source: TileNode,
    mask_source: TileNode,
    out_value: float,
    mask_value: Union[float, int, Sequence[Union[float, int]]],
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`mask_img`: wherever band 1 of ``mask_source``
    has one of the ``mask_value`` values, every band of ``source`` is set to
    ``out_value``; for chaining in a :class:`~rsgispy.imageprocess.TileGraph`.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its no-data value is ``no_data_val`` (Default = ``None``: that
    of ``source``). ``mask_source`` can itself be a node, e.g. a thresholded
    index.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode` to mask.
    :param mask_source: Node whose first band is the mask.
    :param out_value: Value written to masked pixels.
    :param mask_value: Mask value or values (``NaN`` allowed).
    :param no_data_val: Optional no-data value of the output.
    :param name: Optional node name (Default = ``'mask_img'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.masking import mask_img_node

        img = RasterSource("s2.tif")
        clouds = RasterSource("cloud_mask.tif", bands=1)
        masked = mask_img_node(img, clouds, 0, [1, 2], no_data_val=0)
        TileGraph(DiskSink(masked, "s2_masked.tif", datatype=rsgispy.TYPE_16UINT)).run()
    """
    # pylint: disable=too-many-arguments
    mask_vals = _parse_mask_values(mask_value)

    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        first = inputs[0]
        return first.replace(
            dtype=np.float64,
            nodata=no_data_val if no_data_val is not None else first.nodata,
        )

    def _func(data: np.ndarray, msk: np.ndarray) -> np.ndarray:
        return _mask_img_block(
            data, np.asarray(msk[0], dtype=np.float64), out_value, mask_vals
        )

    return PointwiseOp(
        [source, mask_source], _func, name=name or "mask_img", meta_fn=_meta
    )


def mask_img(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    out_value: float,
    mask_value: Union[float, int, Sequence[Union[float, int]]],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Mask ``input_img`` using ``in_msk_img``: wherever band 1 of ``in_msk_img``
    matches any value in ``mask_value``, output pixels across all bands are set
    to ``out_value``; otherwise input pixel values are preserved.

    :param input_img: Path to the input image file.
    :param in_msk_img: Path to the mask image file (band 1 is used).
    :param output_img: Path to the output masked image file.
    :param out_value: Value written to masked pixels in the output image.
    :param mask_value: Scalar or sequence of mask values in ``in_msk_img`` to mask.
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
        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_img(
            "in_img.tif", "msk.tif", "out_img.tif", 0.0, 0,
            gdalformat="GTIFF", datatype=rsgispy.TYPE_32FLOAT,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    _parse_mask_values(mask_value)
    norm_opts = normalise_creation_options(creation_opts)
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(
            stack, [input_img, in_msk_img]
        )
        grid = compute_spatial_intersection(datasets)
        in_ds = datasets[paths[0]]
        eff_dtype = (
            rsgis_dtype_from_np_dtype(in_ds.dtypes[0])
            if datatype is None
            else datatype
        )
        eff_nd = no_data_val if no_data_val is not None else in_ds.nodata
        descs = [desc if desc else None for desc in in_ds.descriptions]

    node = mask_img_node(
        GridRasterSource(paths[0], grid, grid.offsets[paths[0]]),
        GridRasterSource(paths[1], grid, grid.offsets[paths[1]], bands=1),
        out_value,
        mask_value,
        no_data_val=eff_nd,
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=eff_dtype,
        nodata=eff_nd,
        clamp_int=False,
        band_names=descs,
        calc_stats=calc_stats,
        creation_opts=norm_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def _thematic_sink(
    node: TileNode,
    output_img: Union[str, os.PathLike],
    gdalformat: str,
    datatype: int,
    no_data_val: Optional[float],
    calc_stats: bool,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]],
    *,
    set_nodata: bool = True,
) -> WrapperDiskSink:
    """Sink of a thematic mask: ``LAYER_TYPE=thematic`` and thematic statistics."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=no_data_val if set_nodata else None,
        clamp_int=False,
        calc_stats=calc_stats,
        creation_opts=normalise_creation_options(creation_opts),
        thematic=True,
        close_actions=[set_img_thematic],
    )


def gen_finite_mask_node(
    source: TileNode,
    *,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`gen_finite_mask`: a ``uint8`` mask, ``1`` where
    every band of ``source`` is finite and ``0`` where any band is ``NaN`` or
    ``+/-Inf``.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param name: Optional node name (Default = ``'gen_finite_mask'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node
        from rsgispy.imageutils.masking import gen_finite_mask_node

        idx = calc_indices_node(RasterSource("s2.tif"), {3: "red", 4: "nir"},
                                ["NDVI"], out_no_data_val=float("nan"))
        msk = gen_finite_mask_node(idx)
        TileGraph(DiskSink(msk, "ndvi_valid.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """

    def _func(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        return build_valid_cube_mask(arr, axis=0).astype(np.uint8)[np.newaxis]

    return PointwiseOp(
        source,
        _func,
        name=name or "gen_finite_mask",
        meta_fn=lambda inputs: NodeMeta(n_bands=1, dtype=np.uint8),
    )


def gen_finite_mask(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic mask (``1`` where all bands are finite, ``0`` where
    any band contains ``NaN`` or ``+/-Inf``).

    :param input_img: Path to the input image file.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_finite_mask("in_img.tif", "finite_msk.tif")
    """
    # pylint: disable=too-many-arguments,too-many-locals
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    with ExitStack() as stack:
        open_multi_image_datasets(stack, [input_img])
    node = gen_finite_mask_node(RasterSource(input_img))
    sink = _thematic_sink(
        node, output_img, gdalformat, eff_dtype, no_data_val, calc_stats, creation_opts
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


class _ValidMaskNode(TileNode):
    """
    ``uint8`` mask, ``1`` where every band of every input is finite and not
    equal to its no-data value; ``band_nodata[i]`` gives the no-data value of
    each band of input ``i`` (a scalar for all bands, or ``None``).
    """

    def __init__(
        self,
        sources: Sequence[TileNode],
        band_nodata: Optional[Sequence[Any]],
        per_band: bool,
        name: Optional[str] = None,
        *,
        default_no_data_val: Optional[float] = None,
    ) -> None:
        super().__init__(inputs=tuple(sources), name=name)
        self.band_nodata = list(band_nodata) if band_nodata is not None else None
        self.per_band = bool(per_band)
        self.default_no_data_val = default_no_data_val
        self._nodata: List[Any] = list(band_nodata) if band_nodata is not None else []

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        if self.band_nodata is None:
            # Per band: each input's no-data value, or the default.
            self._nodata = [
                [meta.nodata if meta.nodata is not None else self.default_no_data_val]
                * int(meta.n_bands or 0)
                for meta in inputs
            ]
        return NodeMeta(n_bands=1, dtype=np.uint8)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        valid = np.ones(aligned[0].data.shape[1:], dtype=bool)
        for tensor, nodata in zip(aligned, self._nodata):
            arr = np.asarray(tensor.data, dtype=np.float64)
            if self.per_band:
                for b_idx in range(arr.shape[0]):
                    b_nd = nodata[b_idx]
                    valid &= build_valid_mask(
                        arr[b_idx],
                        no_data_val=float(b_nd) if b_nd is not None else None,
                    )
            else:
                valid &= build_valid_cube_mask(
                    arr,
                    no_data_val=float(nodata) if nodata is not None else None,
                    axis=0,
                )
        return aligned[0].with_data(valid.astype(np.uint8)[np.newaxis])


def gen_valid_mask_node(
    sources: Union[TileNode, Sequence[TileNode]],
    no_data_val: Optional[float] = 0.0,
    *,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`gen_valid_mask`: a ``uint8`` mask, ``1`` where
    every band of every input is finite and differs from ``no_data_val``,
    else ``0``.

    :param sources: Upstream node or nodes.
    :param no_data_val: No-data value (Default = ``0.0``; ``None`` checks only
                        finiteness).
    :param name: Optional node name (Default = ``'gen_valid_mask'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.masking import gen_valid_mask_node

        msk = gen_valid_mask_node([RasterSource("t1.tif"), RasterSource("t2.tif")], 0)
        TileGraph(DiskSink(msk, "valid.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    src_list = [sources] if isinstance(sources, TileNode) else list(sources)
    if not src_list:
        raise rsgispy.RSGISPyException("gen_valid_mask_node needs at least one input.")
    return _ValidMaskNode(
        src_list, [no_data_val] * len(src_list), False, name=name or "gen_valid_mask"
    )


def gen_valid_mask(
    input_imgs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic valid-pixel mask (``1`` where all bands across all
    input images are finite and differ from ``no_data_val``, else ``0``).

    :param input_imgs: Input image path or sequence of image paths.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Value representing no-data (Default = ``0.0``; pass ``None``
                        to check only finiteness).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_valid_mask(
            ["img1.tif", "img2.tif"], "valid_msk.tif", no_data_val=0.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(input_imgs, (str, os.PathLike)):
        img_list: List[Union[str, os.PathLike]] = [input_imgs]
    else:
        img_list = list(input_imgs)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, img_list)
        grid = compute_spatial_intersection(datasets)
    node = gen_valid_mask_node(
        [GridRasterSource(path, grid, grid.offsets[path]) for path in paths],
        no_data_val=no_data_val,
    )
    sink = _thematic_sink(
        node, output_img, gdalformat, eff_dtype, no_data_val, calc_stats,
        creation_opts, set_nodata=False,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def gen_img_edge_mask_node(
    source: TileNode,
    n_edge_pxls: int,
    *,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`gen_img_edge_mask`: a ``uint8`` mask, ``1``
    within ``n_edge_pxls`` of any edge of the image and ``0`` in the interior.
    The values depend only on the pixel position (``source`` gives the grid).

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param n_edge_pxls: Number of edge pixels on each side marked ``1``.
    :param name: Optional node name (Default = ``'gen_img_edge_mask'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.masking import gen_img_edge_mask_node

        edge = gen_img_edge_mask_node(RasterSource("img.tif", bands=1), 10)
        TileGraph(DiskSink(edge, "edge.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    edge_px = int(n_edge_pxls)
    if edge_px < 0:
        raise rsgispy.RSGISPyException(
            f"n_edge_pxls must be non-negative, got {n_edge_pxls}."
        )

    def _func(arr: np.ndarray, ctx: TileContext) -> np.ndarray:
        row0 = int(ctx.row_off) - int(ctx.halo_y)
        col0 = int(ctx.col_off) - int(ctx.halo_x)
        rows = np.arange(row0, row0 + arr.shape[1])[:, np.newaxis]
        cols = np.arange(col0, col0 + arr.shape[2])[np.newaxis, :]
        edge = (
            (rows < edge_px)
            | (rows >= ctx.img_height - edge_px)
            | (cols < edge_px)
            | (cols >= ctx.img_width - edge_px)
        )
        return edge.astype(np.uint8)[np.newaxis]

    return PointwiseOp(
        source,
        _func,
        pass_ctx=True,
        name=name or "gen_img_edge_mask",
        meta_fn=lambda inputs: NodeMeta(n_bands=1, dtype=np.uint8),
    )


def gen_img_edge_mask(
    input_img: str,
    output_img: str,
    n_edge_pxls: int,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Generate a binary thematic mask highlighting image border pixels (``1`` within
    ``n_edge_pxls`` of any edge of ``input_img``, ``0`` in the interior).

    :param input_img: Path to the input reference image.
    :param output_img: Path to the output mask image file (8-bit unsigned by default).
    :param n_edge_pxls: Number of edge pixels along each side to mark as ``1``.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional output no-data value (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.gen_img_edge_mask(
            "in_img.tif", "edge_msk.tif", 5, gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    edge_px = int(n_edge_pxls)
    if edge_px < 0:
        raise rsgispy.RSGISPyException(
            f"n_edge_pxls must be non-negative, got {n_edge_pxls}."
        )
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    with ExitStack() as stack:
        open_multi_image_datasets(stack, [input_img])
    node = gen_img_edge_mask_node(RasterSource(input_img, bands=1), edge_px)
    sink = _thematic_sink(
        node, output_img, gdalformat, eff_dtype, no_data_val, calc_stats, creation_opts
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def mask_all_band_zero_vals_node(
    source: TileNode,
    out_val: float = 1.0,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`mask_all_band_zero_vals`: band values ``<= 0``
    are replaced by ``out_val`` where the sum of all bands is ``> 0``.

    The node outputs ``float64`` blocks with the bands (and band names) of
    ``source``; its no-data value is ``no_data_val`` (Default = ``None``: that
    of ``source``).

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param out_val: Replacement value (Default = ``1.0``).
    :param no_data_val: Optional no-data value of the output.
    :param name: Optional node name (Default = ``'mask_all_band_zero_vals'``).
    :return: :class:`~rsgispy.imageprocess.PointwiseOp` node.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.masking import mask_all_band_zero_vals_node

        fixed = mask_all_band_zero_vals_node(RasterSource("s2.tif"), out_val=1)
        TileGraph(DiskSink(fixed, "s2_fixed.tif", datatype=rsgispy.TYPE_16UINT)).run()
    """

    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        first = inputs[0]
        return first.replace(
            dtype=np.float64,
            nodata=no_data_val if no_data_val is not None else first.nodata,
        )

    def _func(arr: np.ndarray) -> np.ndarray:
        data = np.array(arr, dtype=np.float64, copy=True)
        valid_pxl = np.sum(data, axis=0) > 0.0
        replace_mask = (data <= 0.0) & valid_pxl[np.newaxis, :, :]
        data[replace_mask] = float(out_val)
        return data

    return PointwiseOp(
        source, _func, name=name or "mask_all_band_zero_vals", meta_fn=_meta
    )


def mask_all_band_zero_vals(
    input_img: str,
    output_img: str,
    out_val: float = 1.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Replace zero/non-positive values in individual bands with ``out_val`` where
    the sum across all bands at that pixel is ``> 0``; pixels where all bands are
    ``<= 0`` remain unchanged.

    :param input_img: Path to the input multi-band image file.
    :param output_img: Path to the output image file.
    :param out_val: Replacement value when a pixel has positive band sum but
                    a band value ``<= 0`` (Default = ``1.0``).
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

        import rsgispy.imageutils.masking

        rsgispy.imageutils.masking.mask_all_band_zero_vals(
            "in_img.tif", "out_img.tif", out_val=1.0, gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    norm_opts = normalise_creation_options(creation_opts)
    with ExitStack() as stack:
        datasets, paths = open_multi_image_datasets(stack, [input_img])
        in_ds = datasets[paths[0]]
        eff_dtype = (
            rsgis_dtype_from_np_dtype(in_ds.dtypes[0])
            if datatype is None
            else datatype
        )
        eff_nd = no_data_val if no_data_val is not None else in_ds.nodata
        descs = [desc if desc else None for desc in in_ds.descriptions]
    node = mask_all_band_zero_vals_node(RasterSource(input_img), out_val, no_data_val=eff_nd)
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=eff_dtype,
        nodata=eff_nd,
        clamp_int=False,
        band_names=descs,
        calc_stats=calc_stats,
        creation_opts=norm_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def create_valid_mask_node(
    sources: Union[TileNode, Sequence[TileNode]],
    default_no_data_val: Optional[float] = 0.0,
    *,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`create_valid_mask`: a ``uint8`` mask, ``1``
    where every band of every input is finite and differs from that input's
    no-data value (``default_no_data_val`` for inputs without one), else
    ``0``. Select the bands to check with
    :class:`~rsgispy.imageprocess.SelectBands` or ``RasterSource(bands=...)``.

    :param sources: Upstream node or nodes.
    :param default_no_data_val: No-data value for inputs without one
                                (Default = ``0.0``; ``None`` checks only
                                finiteness).
    :param name: Optional node name (Default = ``'create_valid_mask'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imageutils.masking import create_valid_mask_node

        msk = create_valid_mask_node(
            [RasterSource("s2.tif", bands=[1, 4]), RasterSource("s1.tif")]
        )
        TileGraph(DiskSink(msk, "valid.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    src_list = [sources] if isinstance(sources, TileNode) else list(sources)
    if not src_list:
        raise rsgispy.RSGISPyException(
            "create_valid_mask_node needs at least one input."
        )
    return _ValidMaskNode(
        src_list,
        None,
        True,
        name=name or "create_valid_mask",
        default_no_data_val=default_no_data_val,
    )


def create_valid_mask(
    img_band_info: Sequence[ImageBandInfo],
    output_img: Union[str, os.PathLike],
    tmp_dir: str = "",
    default_no_data_val: Optional[float] = 0.0,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Create a binary thematic valid-pixel mask (``1`` where all bands of every
    image in ``img_band_info`` are finite and differ from their image's
    no-data value, ``0`` otherwise).

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects describing
                          input images and optional bands to check.
    :param output_img: Path to the output mask image file.
    :param tmp_dir: Temporary directory path (accepted for API compatibility).
    :param default_no_data_val: Fallback no-data value checked when an input
                                band has no embedded raster no-data value
                                (``None`` in ``ds.nodatavals``; Default = ``0.0``).
                                Pass ``None`` to check only finiteness for bands
                                without embedded no-data metadata.
    :param gdalformat: Output GDAL format name (e.g., ``'KEA'``, ``'GTIFF'``;
                       Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` constant for the output image
                     (defaults to :data:`rsgispy.TYPE_8UINT`).
    :param no_data_val: Optional no-data metadata value to set on ``output_img``
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute thematic statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.classification.classdata
        import rsgispy.imageutils

        img_info = [
            rsgispy.classification.classdata.ImageBandInfo("img1.tif", "img1", [1, 2]),
            rsgispy.classification.classdata.ImageBandInfo("img2.tif", "img2"),
        ]
        rsgispy.imageutils.create_valid_mask(
            img_info, "valid_msk.tif", gdalformat="GTIFF"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if not img_band_info:
        raise rsgispy.RSGISPyException(
            "img_band_info must contain at least one ImageBandInfo item."
        )
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    out_msk_path = os.fspath(output_img)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else datatype
    img_paths: List[str] = []
    for info in img_band_info:
        if not info.file_name:
            raise rsgispy.RSGISPyException(
                "Each ImageBandInfo object must specify a valid 'file_name'."
            )
        img_paths.append(os.fspath(info.file_name))
    sources: List[TileNode] = []
    band_nodata: List[List[Optional[float]]] = []
    with ExitStack() as stack:
        datasets, resolved_paths = open_multi_image_datasets(stack, img_paths)
        grid = compute_spatial_intersection(datasets)
        for info, path in zip(img_band_info, resolved_paths):
            ds = datasets[path]
            bands = (
                [int(b) for b in info.bands]
                if info.bands is not None
                else list(range(1, ds.count + 1))
            )
            if not bands:
                raise rsgispy.RSGISPyException(
                    f"ImageBandInfo for '{path}' specified an empty band list."
                )
            for b_idx in bands:
                validate_band_index(b_idx, ds.count, path)
            sources.append(GridRasterSource(path, grid, grid.offsets[path], bands=bands))
            band_nodata.append(
                [
                    default_no_data_val
                    if ds.nodatavals[b_idx - 1] is None
                    else float(ds.nodatavals[b_idx - 1])
                    for b_idx in bands
                ]
            )

    node = _ValidMaskNode(sources, band_nodata, True, name="create_valid_mask")
    sink = _thematic_sink(
        node, out_msk_path, gdalformat, eff_dtype, no_data_val, calc_stats, creation_opts
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)
