"""
The imagecalc.percentiles module provides band-level and pixel-level raster
percentile calculation functions using ``rasterio``.
"""

from contextlib import ExitStack
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._wrap import (
    GridRasterSource,
    WrapperDiskSink,
    run_wrapper_graph,
)
from rsgispy.imageprocess import (
    BandDefn,
    NodeMeta,
    TileContext,
    TileNode,
    TileTensor,
    compute_spatial_intersection,
    ensure_file_exists,
    fits_dtype_range,
    iter_image_blocks,
    open_input_datasets,
    open_multi_image_datasets,
)
from rsgispy.tools.stats import (
    compute_nanpercentiles,
    validate_percentiles_0_100,
)

__all__ = [
    "calc_band_percentile",
    "calc_band_percentile_msk",
    "calc_imgs_pxl_percentiles",
    "calc_imgs_pxl_percentiles_node",
    "calc_img_band_pxl_percentiles",
    "calc_img_band_pxl_percentiles_node",
]


def _pct_label(percentile: float) -> str:
    """Return the band name suffix for ``percentile`` (e.g. ``'p10'``, ``'p2.5'``)."""
    return f"p{float(percentile):g}"


def _resolve_pct_output(
    datasets: Mapping[str, Any],
    img_paths: Sequence[str],
    no_data_val: Optional[float],
) -> Tuple[str, np.dtype, Any]:
    """
    Return the output rasterio dtype, NumPy dtype and the dtype used to read
    the inputs, and check that ``no_data_val`` can be stored in the output.
    """
    rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])
    out_rio_dtype = rsgispy.get_rasterio_datatype(rsgis_dt)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(rsgis_dt))
    if no_data_val is not None:
        nd_is_nan = math.isnan(float(no_data_val))
        if (nd_is_nan and not np.issubdtype(out_np_dtype, np.floating)) or (
            not nd_is_nan and not fits_dtype_range(float(no_data_val), out_np_dtype)
        ):
            raise rsgispy.RSGISPyException(
                f"no_data_val ({no_data_val}) cannot be stored in the output data "
                f"type '{out_np_dtype.name}' (the data type of the first input "
                "image)."
            )
    # Read in float32 unless an input needs more precision (e.g. float64, int32).
    read_dtype = np.result_type(
        np.float32,
        *[np.dtype(dt) for path in img_paths for dt in datasets[path].dtypes],
    )
    return out_rio_dtype, out_np_dtype, read_dtype


def calc_band_percentile(
    input_img: str,
    percentile: float,
    no_data_val: Optional[float] = None,
) -> List[float]:
    """
    Calculate a percentile value for each band of an input image.

    :param input_img: Path to the input image file.
    :param percentile: Percentile fraction between ``0.0`` and ``1.0`` inclusive.
    :param no_data_val: Optional no-data value to exclude (or ``None``).
    :return: List of float percentile values, one per band.

    .. code:: python

        import rsgispy.imagecalc.percentiles

        pct_vals = rsgispy.imagecalc.percentiles.calc_band_percentile(
            input_img="img.tif",
            percentile=0.95,
            no_data_val=0.0,
        )
    """
    if percentile < 0.0 or percentile > 1.0:
        raise rsgispy.RSGISPyException("Percentile value must be between 0 - 1.")
    ensure_file_exists(input_img)

    pct_val = float(percentile) * 100.0
    out_vals: List[float] = []
    with rasterio.open(input_img, "r") as src:
        for band_idx in range(1, src.count + 1):
            arr = src.read(band_idx, out_dtype=np.float64).ravel()
            if no_data_val is not None:
                if isinstance(no_data_val, float) and math.isnan(no_data_val):
                    arr = arr[~np.isnan(arr)]
                else:
                    arr = arr[arr != no_data_val]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out_vals.append(float("nan"))
            else:
                out_vals.append(float(np.percentile(arr, pct_val)))

    return out_vals


def calc_band_percentile_msk(
    input_img: str,
    in_msk_img: str,
    msk_val: int,
    percentiles: Sequence[float],
    no_data_val: Optional[float] = None,
    use_pbar: bool = True,
) -> np.ndarray:
    """
    Calculate percentiles for all bands of ``input_img`` over pixels matching
    ``msk_val`` in ``in_msk_img``.

    :param input_img: Input image path on which percentiles are calculated.
    :param in_msk_img: Mask image path (band 1 is compared against ``msk_val``).
    :param msk_val: Pixel value within ``in_msk_img`` identifying valid pixels.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param no_data_val: Optional no-data value to exclude across all bands.
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :return: NumPy array of shape ``(len(percentiles), n_bands)`` (or ``(n_bands,)``
             if ``percentiles`` is a scalar-like single float passed to ``np.percentile``).

    .. code:: python

        import rsgispy.imagecalc.percentiles

        pcts = rsgispy.imagecalc.percentiles.calc_band_percentile_msk(
            input_img="img.tif",
            in_msk_img="mask.tif",
            msk_val=1,
            percentiles=[5.0, 50.0, 95.0],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    ensure_file_exists(input_img)
    ensure_file_exists(in_msk_img, label="Mask image")

    with rasterio.open(input_img, "r") as img_src:
        n_bands = img_src.count

    norm_defs = [
        BandDefn(band_name=f"b{idx + 1}", input_img=input_img, img_band=idx + 1)
        for idx in range(n_bands)
    ]

    collected: List[np.ndarray] = []
    with ExitStack() as stack:
        datasets = open_input_datasets(stack, norm_defs)
        if in_msk_img not in datasets:
            datasets[in_msk_img] = stack.enter_context(
                rasterio.open(in_msk_img, "r")
            )
        grid = compute_spatial_intersection(datasets)
        msk_ds = datasets[in_msk_img]
        base_col, base_row = grid.offsets[in_msk_img]

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=norm_defs,
            grid=grid,
            read_dtype=None,
            use_pbar=use_pbar,
        ):
            msk_win = Window(
                base_col + int(out_win.col_off),
                base_row + int(out_win.row_off),
                int(out_win.width),
                int(out_win.height),
            )
            msk_flat = msk_ds.read(1, window=msk_win).ravel()
            sel = msk_flat == msk_val
            if not np.any(sel):
                continue

            band_cols = np.column_stack(
                [block_vars[f"b{idx + 1}"].ravel()[sel] for idx in range(n_bands)]
            )
            valid = np.isfinite(band_cols).all(axis=1)
            if no_data_val is not None and not math.isnan(float(no_data_val)):
                valid &= (band_cols != no_data_val).all(axis=1)
            band_cols = band_cols[valid]
            if band_cols.shape[0] > 0:
                collected.append(band_cols)

    if not collected:
        raise rsgispy.RSGISPyException(
            "No valid pixels found within the specified mask."
        )

    pxl_vals = np.concatenate(collected, axis=0)
    return np.percentile(pxl_vals, percentiles, axis=0)


class _PxlPercentilesNode(TileNode):
    """
    Per-pixel percentiles across the bands of all inputs (``per_band=False``)
    or, for each band, across the inputs (``per_band=True``).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        sources: Sequence[TileNode],
        pcts: Sequence[float],
        no_data_val: Optional[float],
        out_np_dtype: Optional[np.dtype],
        per_band: bool,
        name: Optional[str] = None,
    ) -> None:
        src_list = tuple(sources)
        if not src_list:
            raise rsgispy.RSGISPyException("At least one input node is required.")
        default = "calc_img_band_pxl_percentiles" if per_band else "calc_imgs_pxl_percentiles"
        super().__init__(inputs=src_list, name=name or default)
        self.pcts = [float(pct) for pct in pcts]
        self.no_data_val = no_data_val
        self.out_np_dtype = np.dtype(out_np_dtype) if out_np_dtype is not None else None
        self.per_band = bool(per_band)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        dtype = self.out_np_dtype if self.out_np_dtype is not None else np.dtype(np.float64)
        if not self.per_band:
            return NodeMeta(
                n_bands=len(self.pcts),
                band_names=tuple(_pct_label(pct) for pct in self.pcts),
                dtype=dtype,
                nodata=self.no_data_val,
            )
        counts = [meta.n_bands for meta in inputs]
        if all(c is not None for c in counts) and len(set(counts)) > 1:
            raise rsgispy.RSGISPyException(
                "Input images have a different number of bands."
            )
        n_bands = counts[0]
        names = None
        if n_bands is not None:
            first = inputs[0].band_names
            base_names = [
                first[b_idx] if first is not None else f"b{b_idx + 1}"
                for b_idx in range(n_bands)
            ]
            names = tuple(f"{nm}_{_pct_label(pct)}" for nm in base_names for pct in self.pcts)
        return NodeMeta(
            n_bands=None if n_bands is None else n_bands * len(self.pcts),
            band_names=names,
            dtype=dtype,
            nodata=self.no_data_val,
        )

    def _cast(self, arr: np.ndarray) -> np.ndarray:
        if self.out_np_dtype is None:
            return arr
        return arr.astype(self.out_np_dtype, copy=False)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        blocks = [t.data for t in aligned]
        if not self.per_band:
            out = self._cast(
                compute_nanpercentiles(
                    np.concatenate(blocks, axis=0), self.pcts, self.no_data_val
                )
            )
        else:
            out = self._cast(
                np.concatenate(
                    [
                        compute_nanpercentiles(
                            np.stack([blk[b_idx] for blk in blocks], axis=0),
                            self.pcts,
                            self.no_data_val,
                        )
                        for b_idx in range(blocks[0].shape[0])
                    ],
                    axis=0,
                )
            )
        return aligned[0].with_data(out)


def calc_imgs_pxl_percentiles_node(
    sources: Union[TileNode, Sequence[TileNode]],
    percentiles: Sequence[float],
    *,
    no_data_val: Optional[float] = 0.0,
    out_dtype: Any = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`calc_imgs_pxl_percentiles`: per-pixel
    percentiles across all bands of all ``sources`` (e.g. a multi-date
    stack), one output band per percentile named ``p<percentile>``, for
    chaining in a :class:`~rsgispy.imageprocess.TileGraph` (e.g. a median
    composite of index nodes).

    The percentiles match ``numpy.nanpercentile`` (linear interpolation);
    ``no_data_val`` values are ignored and written to pixels without valid
    values. The blocks are ``float64`` unless ``out_dtype`` is given (then
    cast with ``astype``, as :func:`calc_imgs_pxl_percentiles` does).

    :param sources: Upstream node or nodes.
    :param percentiles: Percentiles in ``[0, 100]``.
    :param no_data_val: Input no-data value (Default = ``0.0``; ``None``
                        ignores only ``NaN``).
    :param out_dtype: Optional NumPy dtype of the output blocks.
    :param name: Optional node name (Default = ``'calc_imgs_pxl_percentiles'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.calcindices import calc_indices_node
        from rsgispy.imagecalc.percentiles import calc_imgs_pxl_percentiles_node

        bands = {3: "red", 4: "nir"}
        ndvis = [
            calc_indices_node(RasterSource(img), bands, ["NDVI"])
            for img in ["s2_t1.tif", "s2_t2.tif", "s2_t3.tif"]
        ]
        med = calc_imgs_pxl_percentiles_node(ndvis, [50], no_data_val=-999)
        TileGraph(DiskSink(med, "ndvi_median.tif")).run()
    """
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]
    src_list = [sources] if isinstance(sources, TileNode) else list(sources)
    return _PxlPercentilesNode(
        src_list, pcts, no_data_val, out_dtype, per_band=False, name=name
    )


def calc_img_band_pxl_percentiles_node(
    sources: Sequence[TileNode],
    percentiles: Sequence[float],
    *,
    no_data_val: Optional[float] = 0.0,
    out_dtype: Any = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`calc_img_band_pxl_percentiles`: for each band,
    per-pixel percentiles across ``sources`` (which must have the same
    number of bands). The output has ``n_bands * len(percentiles)`` bands
    (band 1 percentiles, band 2 percentiles, ...) named
    ``<band name>_p<percentile>`` from the first source's band names (or
    ``b<band>_p<percentile>``).

    :param sources: Upstream nodes (e.g. one per date).
    :param percentiles: Percentiles in ``[0, 100]``.
    :param no_data_val: Input no-data value (Default = ``0.0``; ``None``
                        ignores only ``NaN``).
    :param out_dtype: Optional NumPy dtype of the output blocks
                      (Default = ``None``: ``float64``).
    :param name: Optional node name
                 (Default = ``'calc_img_band_pxl_percentiles'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagecalc.percentiles import calc_img_band_pxl_percentiles_node

        dates = [RasterSource(img) for img in ["t1.tif", "t2.tif", "t3.tif"]]
        comp = calc_img_band_pxl_percentiles_node(dates, [50], no_data_val=0)
        TileGraph(DiskSink(comp, "median_composite.tif")).run()
    """
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]
    return _PxlPercentilesNode(
        list(sources), pcts, no_data_val, out_dtype, per_band=True, name=name
    )


def calc_imgs_pxl_percentiles(
    input_imgs: Sequence[str],
    output_img: str,
    percentiles: Sequence[float],
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate percentiles on a per-pixel basis across all bands of ``input_imgs``.

    The output image has ``len(percentiles)`` bands, named ``p<percentile>``
    (e.g. ``p10``, ``p50``, ``p2.5``), and matches the spatial intersection
    and data type of the first input image. The images are read and
    processed block by block (:func:`rsgispy.imageprocess.get_default_block_size`),
    so memory use does not depend on the image size. The percentiles match
    ``numpy.nanpercentile`` (linear interpolation) and are cast to the output
    data type (truncated for integer types).

    :param input_imgs: Sequence of input image paths (all bands are stacked).
    :param output_img: Path to the output image file.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: No-data value of the inputs, ignored when calculating
                        the percentiles. It is written to the pixels with no
                        valid values and set as the no-data value of the
                        output, so it must be representable in the output data
                        type (Default = 0). ``None`` ignores only ``NaN``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, ignoring ``no_data_val``
                       (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_imgs_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            output_img="pxl_pcts.tif",
            percentiles=[10.0, 50.0, 90.0],
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        grid = compute_spatial_intersection(datasets)
        _, out_np_dtype, read_dtype = _resolve_pct_output(
            datasets, img_paths, no_data_val
        )
        out_rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])

    sources = [
        GridRasterSource(path, grid, grid.offsets[path], dtype=read_dtype)
        for path in img_paths
    ]
    node = _PxlPercentilesNode(
        sources, pcts, no_data_val, out_np_dtype, per_band=False
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=out_rsgis_dt,
        nodata=no_data_val,
        band_names=[_pct_label(pct) for pct in pcts],
        calc_stats=calc_stats,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)


def calc_img_band_pxl_percentiles(
    input_imgs: Sequence[str],
    output_img: str,
    percentiles: Sequence[float],
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = 0.0,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Calculate per-pixel percentiles across ``input_imgs`` on a per-band basis.

    All input images must have the same band count ``n_bands``. The output
    image contains ``n_bands * len(percentiles)`` bands ordered as band 1
    percentiles, band 2 percentiles, ..., band ``n`` percentiles. The output
    bands are named ``<band name>_p<percentile>`` (e.g. ``Blue_p50``), using
    the band names of the first input image, or ``b<band>_p<percentile>``
    (e.g. ``b1_p50``) for bands without a name. The output matches the
    spatial intersection and data type of the first input image. The images
    are read and processed block by block
    (:func:`rsgispy.imageprocess.get_default_block_size`), so memory use does
    not depend on the image size. The percentiles match
    ``numpy.nanpercentile`` (linear interpolation) and are cast to the output
    data type (truncated for integer types).

    :param input_imgs: Sequence of input image paths with identical band counts.
    :param output_img: Path to the output image file.
    :param percentiles: Sequence of percentiles in ``[0, 100]``.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param no_data_val: No-data value of the inputs, ignored when calculating
                        the percentiles. It is written to the pixels with no
                        valid values and set as the no-data value of the
                        output, so it must be representable in the output data
                        type (Default = 0). ``None`` ignores only ``NaN``.
    :param calc_stats: If ``True``, calculate image statistics and pyramids
                       for the output, ignoring ``no_data_val``
                       (Default = ``True``).
    :param use_pbar: If True, display a tqdm progress bar (Default = True).
    :param creation_opts: Optional mapping of GDAL creation options.

    .. code:: python

        import rsgispy.imagecalc.percentiles

        rsgispy.imagecalc.percentiles.calc_img_band_pxl_percentiles(
            input_imgs=["img1.tif", "img2.tif", "img3.tif"],
            output_img="band_pxl_pcts.tif",
            percentiles=[10.0, 50.0, 90.0],
            gdalformat="GTIFF",
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    validate_percentiles_0_100(percentiles)
    pcts = [float(pct) for pct in np.atleast_1d(np.asarray(percentiles, dtype=np.float64))]

    with ExitStack() as stack:
        datasets, img_paths = open_multi_image_datasets(stack, input_imgs)
        n_bands = datasets[img_paths[0]].count
        for path in img_paths[1:]:
            if datasets[path].count != n_bands:
                raise rsgispy.RSGISPyException(
                    "Input images have a different number of bands."
                )

        grid = compute_spatial_intersection(datasets)
        _, out_np_dtype, read_dtype = _resolve_pct_output(
            datasets, img_paths, no_data_val
        )
        out_rsgis_dt = rsgispy.get_rsgis_datatype(datasets[img_paths[0]].dtypes[0])
        in_names = datasets[img_paths[0]].descriptions

    base_names = [
        in_names[b_idx] if in_names[b_idx] else f"b{b_idx + 1}" for b_idx in range(n_bands)
    ]
    sources = [
        GridRasterSource(path, grid, grid.offsets[path], dtype=read_dtype)
        for path in img_paths
    ]
    node = _PxlPercentilesNode(
        sources, pcts, no_data_val, out_np_dtype, per_band=True
    )
    sink = WrapperDiskSink(
        node,
        output_img,
        gdalformat=gdalformat,
        datatype=out_rsgis_dt,
        nodata=no_data_val,
        band_names=[f"{name}_{_pct_label(pct)}" for name in base_names for pct in pcts],
        calc_stats=calc_stats,
        creation_opts=creation_opts,
    )
    run_wrapper_graph(sink, use_pbar=use_pbar)
