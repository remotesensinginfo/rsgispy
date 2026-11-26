"""
The imagecalc.encoding module provides integer recoding and one-hot category
encoding utilities for raster images using ``rasterio``.
"""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio

import rsgispy
from rsgispy._tilegraph._wrap import WrapperDiskSink, run_wrapper_graph
from rsgispy.imagecalc.stats import get_unique_values
from rsgispy.imageprocess import (
    NodeMeta,
    PointwiseOp,
    RasterSource,
    TileContext,
    TileNode,
    TileTensor,
    cast_result_block,
    ensure_file_exists,
    fits_dtype_range,
)

__all__ = [
    "recode_int_raster",
    "recode_int_raster_node",
    "one_hot_category_encoding",
    "one_hot_category_encoding_node",
]


def _resolve_recoded_nodata(
    src_nodata: Optional[float],
    recode_dict: Mapping[int, int],
    keep_vals_not_in_dict: bool,
    out_np_dtype: np.dtype,
) -> Optional[float]:
    """Validate and optionally remap ``src_nodata`` against ``out_np_dtype`` range."""
    if src_nodata is None:
        return None
    cand = float(src_nodata)
    if math.isfinite(cand) and cand.is_integer():
        src_nd_int = int(cand)
        if src_nd_int in recode_dict:
            cand = float(recode_dict[src_nd_int])
        elif not keep_vals_not_in_dict:
            cand = 0.0
    if not math.isfinite(cand) and np.issubdtype(out_np_dtype, np.floating):
        return cand
    if fits_dtype_range(cand, out_np_dtype):
        return float(int(cand)) if np.issubdtype(out_np_dtype, np.integer) else cand
    return None


def _read_band1_header(input_img: str) -> Tuple[np.dtype, Optional[float]]:
    """Return the data type and no-data value of ``input_img`` (band 1)."""
    try:
        with rasterio.open(input_img, "r") as src:
            return np.dtype(src.dtypes[0]), src.nodata
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open input image '{input_img}': {err}"
        ) from err


def _recode_block(
    in_block: np.ndarray,
    recode_dict: Mapping[int, int],
    keep_vals_not_in_dict: bool,
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """Recode the values of a 2D block."""
    win_h, win_w = int(in_block.shape[0]), int(in_block.shape[1])
    if keep_vals_not_in_dict:
        out_arr = cast_result_block(in_block, win_h, win_w, out_np_dtype)
    else:
        out_arr = np.zeros((win_h, win_w), dtype=out_np_dtype)
    for src_val, dst_val in recode_dict.items():
        out_arr[in_block == src_val] = dst_val
    return out_arr


def recode_int_raster_node(
    source: TileNode,
    recode_dict: Mapping[int, int],
    keep_vals_not_in_dict: bool = True,
    *,
    out_dtype: Any = np.int32,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`recode_int_raster`: recodes the values of
    every band of ``source`` with ``recode_dict``, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` (e.g. after a classification or
    a majority filter).

    The blocks have the data type ``out_dtype``; the node's no-data value is
    that of ``source`` recoded as :func:`recode_int_raster` does (``None`` if
    it cannot be stored in ``out_dtype``). Use a source with the image's own
    integer data type (e.g. ``RasterSource(path, dtype=np.uint8)``) for
    integer comparisons.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param recode_dict: Mapping of ``{input_val: output_val}`` integer pairs.
    :param keep_vals_not_in_dict: Copy values not in ``recode_dict``
                                  (``False`` sets them to ``0``; Default =
                                  ``True``).
    :param out_dtype: NumPy dtype of the output blocks
                      (Default = ``numpy.int32``, as ``TYPE_32INT``).
    :param name: Optional node name (Default = ``'recode_int_raster'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.encoding import recode_int_raster_node

        cls = RasterSource("classes.tif", dtype=np.uint8)
        merged = recode_int_raster_node(cls, {1: 10, 2: 10, 3: 20}, out_dtype=np.uint8)
        TileGraph(DiskSink(merged, "merged.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    if recode_dict is None:
        raise rsgispy.RSGISPyException("recode_dict must not be None.")
    out_np_dtype = np.dtype(out_dtype)
    rec = dict(recode_dict)
    keep = bool(keep_vals_not_in_dict)

    def _meta(inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        return meta.replace(
            dtype=out_np_dtype,
            nodata=_resolve_recoded_nodata(meta.nodata, rec, keep, out_np_dtype),
        )

    def _func(arr: np.ndarray) -> np.ndarray:
        return np.stack(
            [_recode_block(band, rec, keep, out_np_dtype) for band in arr], axis=0
        )

    return PointwiseOp(source, _func, name=name or "recode_int_raster", meta_fn=_meta)


class _OneHotNode(TileNode):
    """Node of :func:`one_hot_category_encoding_node`."""

    def __init__(
        self,
        source: TileNode,
        lcl_recode_dict: Mapping[Any, int],
        n_out_bands: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name or "one_hot_category_encoding")
        self.lcl_recode_dict = dict(lcl_recode_dict)
        self.n_out_bands = int(n_out_bands)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return NodeMeta(n_bands=self.n_out_bands, dtype=np.uint8)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = inputs[0].crop(halo_y, halo_x)
        in_block = aligned.data[0]
        out_arr = np.zeros((self.n_out_bands,) + in_block.shape, dtype=np.uint8)
        for img_val, band_idx in self.lcl_recode_dict.items():
            out_arr[band_idx, in_block == img_val] = 1
        return aligned.with_data(out_arr)


def one_hot_category_encoding_node(
    source: TileNode,
    recode_dict: Mapping[int, int],
    *,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`one_hot_category_encoding`: one ``uint8``
    band per category of band 1 of ``source`` (``1`` where the pixel has the
    category's value), for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` (e.g. as features of a model).

    The categories must be given: ``recode_dict`` maps pixel values to
    1-based output bands. (:func:`one_hot_category_encoding` can find them
    from the image's unique values, a whole-image pass.)

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param recode_dict: Mapping ``{pixel_value: output_band_1_based}``.
    :param name: Optional node name (Default = ``'one_hot_category_encoding'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.encoding import one_hot_category_encoding_node

        lc = RasterSource("landcover.tif", dtype=np.uint8)
        onehot = one_hot_category_encoding_node(lc, {1: 1, 2: 2, 5: 3})
        TileGraph(DiskSink(onehot, "onehot.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    if recode_dict is None or len(recode_dict) == 0:
        raise rsgispy.RSGISPyException("recode_dict must not be empty.")
    lcl_recode_dict: Dict[Any, int] = {}
    n_out_bands = 0
    for img_val, band in recode_dict.items():
        band_int = int(band)
        if band_int < 1:
            raise rsgispy.RSGISPyException(
                "The band number must be greater than or equal to 1."
            )
        n_out_bands = max(n_out_bands, band_int)
        lcl_recode_dict[img_val] = band_int - 1
    return _OneHotNode(source, lcl_recode_dict, n_out_bands, name=name)


def recode_int_raster(
    input_img: str,
    output_img: str,
    recode_dict: Mapping[int, int],
    keep_vals_not_in_dict: bool = True,
    *,
    gdalformat: str = "GTIFF",
    datatype: int = rsgispy.TYPE_32INT,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Recode pixel values in band 1 of an integer raster using ``recode_dict``.

    :param input_img: Path to the input image file (band 1 is recoded).
    :param output_img: Path to the output image file.
    :param recode_dict: Mapping of ``{input_val: output_val}`` integer pairs.
    :param keep_vals_not_in_dict: If ``True``, copy pixels not in ``recode_dict``
                                  to the output; if ``False``, set them to ``0``
                                  (Default = ``True``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Output data type constant from ``rsgispy.TYPE_*``
                     (Default = ``rsgispy.TYPE_32INT``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy
        import rsgispy.imagecalc.encoding

        rsgispy.imagecalc.encoding.recode_int_raster(
            input_img="classes.tif",
            output_img="classes_recoded.tif",
            recode_dict={1: 10, 2: 20},
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_32INT,
        )
    """
    # pylint: disable=too-many-arguments
    if recode_dict is None:
        raise rsgispy.RSGISPyException("recode_dict must not be None.")
    ensure_file_exists(input_img)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(datatype))
    src_dtype, src_nodata = _read_band1_header(input_img)
    dst_nodata = _resolve_recoded_nodata(
        src_nodata, recode_dict, keep_vals_not_in_dict, out_np_dtype
    )
    node = recode_int_raster_node(
        RasterSource(input_img, bands=1, dtype=src_dtype),
        recode_dict,
        keep_vals_not_in_dict,
        out_dtype=out_np_dtype,
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=datatype,
        nodata=dst_nodata,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def one_hot_category_encoding(
    input_img: str,
    output_img: str,
    recode_dict: Optional[Mapping[int, int]] = None,
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[int] = None,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Encode a categorical single-band raster image into a multi-band binary
    (``uint8``) one-hot encoded image.

    If ``recode_dict`` is provided, keys map input pixel values to 1-based output
    band indices. Otherwise, unique values in band 1 of ``input_img`` (excluding
    ``no_data_val``) are sorted ascending and mapped to consecutive bands.

    :param input_img: Path to the input categorical image file.
    :param output_img: Path to the output one-hot encoded image file.
    :param recode_dict: Optional mapping ``{pixel_value: output_band_1_based}``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: Optional no-data value to exclude when inferring unique
                        categories from ``input_img``.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.encoding

        rsgispy.imagecalc.encoding.one_hot_category_encoding(
            input_img="landcover.tif",
            output_img="landcover_onehot.tif",
            gdalformat="GTIFF",
            no_data_val=0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    ensure_file_exists(input_img)

    lcl_recode_dict: Dict[Any, int] = {}
    if recode_dict is None:
        unq_vals = sorted(get_unique_values(input_img, img_band=1).tolist())
        if no_data_val is not None:
            unq_vals = [val for val in unq_vals if val != no_data_val]
        if not unq_vals:
            raise rsgispy.RSGISPyException(
                "No valid category values found to encode."
            )
        for idx, unq_val in enumerate(unq_vals):
            lcl_recode_dict[unq_val] = idx
        n_out_bands = len(lcl_recode_dict)
    else:
        if len(recode_dict) == 0:
            raise rsgispy.RSGISPyException("recode_dict must not be empty.")
        n_out_bands = 0
        for img_val, band in recode_dict.items():
            band_int = int(band)
            if band_int < 1:
                raise rsgispy.RSGISPyException(
                    "The band number must be greater than or equal to 1."
                )
            n_out_bands = max(n_out_bands, band_int)
            lcl_recode_dict[img_val] = band_int - 1

    src_dtype, _ = _read_band1_header(input_img)
    node = _OneHotNode(
        RasterSource(input_img, bands=1, dtype=src_dtype), lcl_recode_dict, n_out_bands
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=rsgispy.TYPE_8UINT,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)
