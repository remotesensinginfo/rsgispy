"""
The elevation.terrain module provides slope and aspect terrain derivative
functions for digital elevation models (DEMs) using ``rasterio`` and ``numpy``.

Each function also has a tile-graph node builder (:func:`slope_node`,
:func:`slope_pxl_res_img_node`, :func:`aspect_node`,
:func:`aspect_pxl_res_img_node`, :func:`categorise_aspect_node`) for chaining
it with other operations in a :class:`~rsgispy.imageprocess.TileGraph`
without intermediate files.

.. note::

   The functions without a pixel-resolution image (``slope``, ``aspect``,
   ``hillshade``, ``local_incidence_angle``, ``local_exitance_angle``,
   ``shadow_mask`` and their nodes) take the pixel size from the DEM's
   transform, in metres, and raise :class:`rsgispy.RSGISPyException` for a
   DEM in a geographic CRS. Reproject such a DEM, or use the
   ``*_pxl_res_img`` variants with per-pixel sizes in metres.
"""

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

import rsgispy
from rsgispy._io import (
    DiskSink,
    NodeMeta,
    RasterSource,
    TileContext,
    TileGraph,
    TileNode,
    TileTensor,
)
from rsgispy.elevation._helpers import (
    Horn3x3Node as _Horn3x3Node,
    HornBlockFunc as _HornBlockFunc,
    calc_slope_aspect_from_diffs as _calc_slope_aspect_from_diffs,
    run_terrain_node_to_file as _run_terrain_node_to_file,
    validate_slope_unit as _validate_slope_unit,
)
from rsgispy.imageprocess import ensure_file_exists

__all__ = [
    "slope",
    "slope_pxl_res_img",
    "aspect",
    "aspect_pxl_res_img",
    "categorise_aspect",
    "slope_node",
    "slope_pxl_res_img_node",
    "aspect_node",
    "aspect_pxl_res_img_node",
    "categorise_aspect_node",
]


def _slope_block_fn(unit_norm: str) -> _HornBlockFunc:
    """Block function of the slope (in ``unit_norm``) from Horn differences."""

    def _block(
        dx: np.ndarray,
        dy: np.ndarray,
        n_vals: np.ndarray,
        ew_res: Any,
        ns_res: Any,
    ) -> np.ndarray:
        slope_rad, _, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
        if unit_norm == "degrees":
            return np.degrees(slope_rad)
        return slope_rad

    return _block


def _aspect_block(
    dx: np.ndarray,
    dy: np.ndarray,
    n_vals: np.ndarray,
    ew_res: Any,
    ns_res: Any,
) -> np.ndarray:
    """Block function of the aspect (degrees, NaN for flat areas)."""
    _, aspect_deg, _ = _calc_slope_aspect_from_diffs(dx, dy, n_vals, ew_res, ns_res)
    return aspect_deg


def _make_slope_node(
    source: TileNode,
    pxl_res: Optional[TileNode],
    unit: str,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
    out_datatype: int = rsgispy.TYPE_32FLOAT,
    res_nodata: Optional[Tuple[Any, Any]] = None,
) -> TileNode:
    """Slope node (``out_datatype`` sets the no-data fill as the file function)."""
    # pylint: disable=too-many-arguments
    unit_norm = _validate_slope_unit(unit)
    return _Horn3x3Node(
        source,
        _slope_block_fn(unit_norm),
        out_datatype,
        pxl_res=pxl_res,
        no_data_val=no_data_val,
        res_nodata=res_nodata,
        op_desc=f"slope unit={unit_norm}",
        name=name or ("slope_pxl_res_img" if pxl_res is not None else "slope"),
    )


def _make_aspect_node(
    source: TileNode,
    pxl_res: Optional[TileNode],
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
    out_datatype: int = rsgispy.TYPE_32FLOAT,
    res_nodata: Optional[Tuple[Any, Any]] = None,
) -> TileNode:
    """Aspect node (``out_datatype`` sets the no-data fill as the file function)."""
    # pylint: disable=too-many-arguments
    return _Horn3x3Node(
        source,
        _aspect_block,
        out_datatype,
        pxl_res=pxl_res,
        no_data_val=no_data_val,
        res_nodata=res_nodata,
        op_desc="aspect",
        name=name or ("aspect_pxl_res_img" if pxl_res is not None else "aspect"),
    )


class _CategoriseAspectNode(TileNode):
    """Pixel-wise node classifying aspect (degrees) into 8 classes (0 = none)."""

    def __init__(
        self,
        source: TileNode,
        no_data_val: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(source, TileNode):
            raise rsgispy.RSGISPyException(
                "categorise_aspect_node source must be a TileNode, got "
                f"{type(source).__name__}."
            )
        super().__init__(inputs=(source,), name=name)
        self.no_data_val = no_data_val
        self._eff_nd: Optional[float] = no_data_val

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        if meta.n_bands is not None and meta.n_bands != 1:
            raise rsgispy.RSGISPyException(
                f"{self.label}: the aspect input must have 1 band, got "
                f"{meta.n_bands} (select one with SelectBands)."
            )
        self._eff_nd = self.no_data_val if self.no_data_val is not None else meta.nodata
        return NodeMeta(n_bands=1, dtype=np.uint8, nodata=None)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # pylint: disable=too-many-locals
        aligned = inputs[0].crop(halo_y, halo_x)
        if aligned.data.shape[0] != 1:
            raise rsgispy.RSGISPyException(
                f"{self.label}: the aspect input must have 1 band, got "
                f"{aligned.data.shape[0]}."
            )
        arr = aligned.data[0].astype(np.float64, copy=False)
        eff_nd = self._eff_nd
        invalid_mask = np.isnan(arr)
        if eff_nd is not None and not math.isnan(float(eff_nd)):
            invalid_mask |= arr == eff_nd
        valid = ~invalid_mask
        # Check the pixels inside the image only (not the padded margins).
        h_in, w_in = arr.shape
        in_img = (
            slice(aligned.oob_top, h_in - aligned.oob_bottom),
            slice(aligned.oob_left, w_in - aligned.oob_right),
        )
        if np.any(arr[in_img][valid[in_img]] < 0.0):
            raise rsgispy.RSGISPyException(
                "The input image pixel values must be between 0 and 360 degrees."
            )
        norm = np.where(arr > 360.0, np.mod(arr, 360.0), arr)
        out_cls = np.zeros(arr.shape, dtype=np.uint8)
        bins = np.clip((norm[valid] // 45.0).astype(np.int64) + 1, 1, 8)
        out_cls[valid] = bins.astype(np.uint8)
        return aligned.with_data(out_cls)

    def describe_params(self) -> str:
        nd = "from input" if self.no_data_val is None else repr(self.no_data_val)
        return f"no_data_val={nd}"


def slope(
    input_img: str,
    output_img: str,
    unit: str = "degrees",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a slope layer from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output slope image file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.slope(
            input_img="dem.tif",
            output_img="slope.tif",
            unit="degrees",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    unit_norm = _validate_slope_unit(unit)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, _res, _res_nd: _make_slope_node(
            dem, None, unit_norm, no_data_val=no_data_val, out_datatype=eff_dtype
        ),
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def slope_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    unit: str = "degrees",
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate a slope layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    Useful for geographic projections (e.g., EPSG:4326) whose horizontal units
    differ from the elevation units.

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output slope image file.
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.slope_pxl_res_img(
            in_dem_img="dem.tif",
            in_pxl_res_img="pxl_res.tif",
            output_img="slope.tif",
            unit="degrees",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    unit_norm = _validate_slope_unit(unit)
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, res, res_nd: _make_slope_node(
            dem,
            res,
            unit_norm,
            no_data_val=no_data_val,
            out_datatype=eff_dtype,
            res_nodata=res_nd,
        ),
        in_dem_img,
        output_img,
        gdalformat,
        eff_dtype,
        in_pxl_res_img=in_pxl_res_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def aspect(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate an aspect layer (in degrees clockwise from North, ``[0, 360)``,
    ``NaN`` for flat areas) from an input elevation model.

    :param input_img: Path to the input DEM image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.aspect(
            input_img="dem.tif",
            output_img="aspect.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, _res, _res_nd: _make_aspect_node(
            dem, None, no_data_val=no_data_val, out_datatype=eff_dtype
        ),
        input_img,
        output_img,
        gdalformat,
        eff_dtype,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def aspect_pxl_res_img(
    in_dem_img: str,
    in_pxl_res_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Calculate an aspect layer from a DEM and an external 2-band pixel resolution
    image (Band 1: East-West resolution, Band 2: North-South resolution).

    :param in_dem_img: Path to the input DEM image file.
    :param in_pxl_res_img: Path to the 2-band pixel resolution image file.
    :param output_img: Path to the output aspect image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_32FLOAT`` when ``None``).
    :param no_data_val: Optional input DEM no-data value (read from header
                        when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.aspect_pxl_res_img(
            in_dem_img="dem.tif",
            in_pxl_res_img="pxl_res.tif",
            output_img="aspect.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    _run_terrain_node_to_file(
        lambda dem, res, res_nd: _make_aspect_node(
            dem, res, no_data_val=no_data_val, out_datatype=eff_dtype, res_nodata=res_nd
        ),
        in_dem_img,
        output_img,
        gdalformat,
        eff_dtype,
        in_pxl_res_img=in_pxl_res_img,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def categorise_aspect(
    input_img: str,
    output_img: str,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Categorise an aspect image (in degrees) into 8 classes (1-8 for 45-degree
    bins: 0-45, 45-90, ..., 315-360) and ``0`` for ``NaN``/flat pixels.

    :param input_img: Path to the input aspect image file (in degrees).
    :param output_img: Path to the output categorised image file.
    :param gdalformat: GDAL/rasterio driver format for the output file
                       (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` output data type constant
                     (Default = ``rsgispy.TYPE_8UINT`` when ``None``).
    :param no_data_val: Optional input aspect no-data value treated as
                        invalid/flat (read from header when ``None``).
    :param calc_stats: If ``True``, populate band statistics and overview
                       pyramids on ``output_img`` (Default = ``True``).
    :param use_pbar: Show a progress bar during calculation (Default = ``True``).

    .. code:: python

        import rsgispy.elevation.terrain

        rsgispy.elevation.terrain.categorise_aspect(
            input_img="aspect.tif",
            output_img="aspect_cats.tif",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments
    ensure_file_exists(input_img)
    eff_dtype = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
    node = categorise_aspect_node(
        RasterSource(input_img, bands=1), no_data_val=no_data_val
    )
    TileGraph(
        DiskSink(
            node,
            output_img,
            gdalformat=gdalformat,
            datatype=eff_dtype,
            nodata=None,
            calc_stats=calc_stats,
            inherit_meta=False,
        ),
        block_size=None,
    ).run(use_pbar=use_pbar)


def slope_node(
    source: TileNode,
    unit: str = "degrees",
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`slope`: the slope of a single-band DEM node
    (Horn 3x3 differences), for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph` without intermediate files.

    The pixel size is that of the graph grid (the DEM's geotransform, as in
    :func:`slope`; use :func:`slope_pxl_res_img_node` for geographic DEMs).
    The node has a halo radius of 1 and pads the image borders by repeating
    the edge pixels. It outputs one ``float64`` band; pixels whose DEM value
    is no data are set to the DEM no-data value (``NaN`` when the DEM has
    none), which is the node's no-data value. Written with a
    :class:`~rsgispy.imageprocess.DiskSink` of ``rsgispy.TYPE_32FLOAT``, the
    output equals that of :func:`slope`.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`
                   (e.g. a :class:`~rsgispy.imageprocess.RasterSource` with
                   ``bands=1``).
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``, e.g. the DEM file's).
    :param name: Optional node name (Default = ``'slope'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    Smooth the DEM, calculate the slope and classify it, in one pass:

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.stats import mean_filter_node
        from rsgispy.elevation.terrain import slope_node

        dem = RasterSource("dem.tif", bands=1)
        smooth = mean_filter_node(dem, 5, no_data_val=-9999)
        slp = slope_node(smooth, "degrees")
        classes = BandMathOp(
            {"s": slp}, "(s >= 5) + (s >= 15) + (s >= 30)", nodata=-9999
        )
        TileGraph(
            DiskSink(classes, "slope_classes.tif", datatype=rsgispy.TYPE_8UINT)
        ).run()
    """
    return _make_slope_node(source, None, unit, no_data_val=no_data_val, name=name)


def slope_pxl_res_img_node(
    source: TileNode,
    pxl_res: TileNode,
    unit: str = "degrees",
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`slope_pxl_res_img`: the slope of a single-band
    DEM node using the per-pixel resolution in bands 1 (East-West) and 2
    (North-South) of ``pxl_res``, e.g. for a DEM in geographic coordinates.

    As :func:`slope_node`, with pixels whose resolution is no data, zero or
    not finite also set to the no-data value. The no-data value of the
    resolution bands is that of ``pxl_res``. Both inputs must be on the same
    pixel grid (checked by the :class:`~rsgispy.imageprocess.TileGraph`).

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param pxl_res: :class:`~rsgispy.imageprocess.TileNode` with at least 2
                    bands (East-West and North-South pixel resolution, in
                    the DEM's elevation units).
    :param unit: Output angular unit, either ``'degrees'`` or ``'radians'``
                 (Default = ``'degrees'``).
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'slope_pxl_res_img'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.elevation.terrain import slope_pxl_res_img_node

        dem = RasterSource("dem_wgs84.tif", bands=1)
        res = RasterSource("pxl_res_m.tif", bands=[1, 2])
        slp = slope_pxl_res_img_node(dem, res, "degrees")
        TileGraph(DiskSink(slp, "slope.tif")).run()
    """
    # pylint: disable=too-many-arguments
    return _make_slope_node(source, pxl_res, unit, no_data_val=no_data_val, name=name)


def aspect_node(
    source: TileNode,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`aspect`: the aspect of a single-band DEM node
    in degrees clockwise from North, ``[0, 360)``, ``NaN`` for flat areas.

    The pixel size is that of the graph grid (the DEM's geotransform). The
    node has a halo radius of 1 and pads the image borders by repeating the
    edge pixels. It outputs one ``float64`` band; pixels whose DEM value is
    no data are set to the DEM no-data value (``NaN`` when the DEM has
    none), which is the node's no-data value. Written with a
    :class:`~rsgispy.imageprocess.DiskSink` of ``rsgispy.TYPE_32FLOAT``, the
    output equals that of :func:`aspect`.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'aspect'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    Aspect classes of the smoothed DEM, in one pass:

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagefilter.smooth import gaussian_smooth_filter_node
        from rsgispy.elevation.terrain import aspect_node, categorise_aspect_node

        dem = RasterSource("dem.tif", bands=1)
        asp = aspect_node(gaussian_smooth_filter_node(dem, 5, no_data_val=-9999))
        cats = categorise_aspect_node(asp)
        TileGraph(DiskSink(cats, "aspect_cats.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    return _make_aspect_node(source, None, no_data_val=no_data_val, name=name)


def aspect_pxl_res_img_node(
    source: TileNode,
    pxl_res: TileNode,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`aspect_pxl_res_img`: the aspect of a
    single-band DEM node using the per-pixel resolution in bands 1
    (East-West) and 2 (North-South) of ``pxl_res``.

    As :func:`aspect_node`, with pixels whose resolution is no data, zero or
    not finite also set to the no-data value. Both inputs must be on the same
    pixel grid.

    :param source: Single-band DEM :class:`~rsgispy.imageprocess.TileNode`.
    :param pxl_res: :class:`~rsgispy.imageprocess.TileNode` with at least 2
                    bands (East-West and North-South pixel resolution).
    :param no_data_val: Optional DEM no-data value (Default = ``None``: the
                        no-data value of ``source``).
    :param name: Optional node name (Default = ``'aspect_pxl_res_img'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.elevation.terrain import (
            aspect_pxl_res_img_node, categorise_aspect_node,
        )

        dem = RasterSource("dem_wgs84.tif", bands=1)
        res = RasterSource("pxl_res_m.tif", bands=[1, 2])
        cats = categorise_aspect_node(aspect_pxl_res_img_node(dem, res))
        TileGraph(DiskSink(cats, "aspect_cats.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    return _make_aspect_node(source, pxl_res, no_data_val=no_data_val, name=name)


def categorise_aspect_node(
    source: TileNode,
    *,
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`categorise_aspect`: classifies an aspect node
    (degrees) into 8 classes (1-8 for the 45-degree bins 0-45, 45-90, ...,
    315-360; values above 360 are wrapped) and ``0`` for ``NaN``, no-data or
    flat pixels. A pixel-wise node (halo radius 0).

    The node outputs one ``uint8`` band without a no-data value (as
    :func:`categorise_aspect`). Negative aspect values raise
    :class:`rsgispy.RSGISPyException` when the graph runs.

    :param source: Single-band aspect :class:`~rsgispy.imageprocess.TileNode`
                   (e.g. :func:`aspect_node`).
    :param no_data_val: Optional aspect no-data value treated as invalid/flat
                        (Default = ``None``: the no-data value of ``source``).
    :param name: Optional node name (Default = ``'categorise_aspect'``).
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.elevation.terrain import aspect_node, categorise_aspect_node

        cats = categorise_aspect_node(aspect_node(RasterSource("dem.tif", bands=1)))
        TileGraph(DiskSink(cats, "aspect_cats.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    return _CategoriseAspectNode(
        source, no_data_val=no_data_val, name=name or "categorise_aspect"
    )
