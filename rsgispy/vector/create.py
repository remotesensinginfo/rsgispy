"""
Vector geometry creation, grid generation, and raster-to-vector conversion
utilities for :mod:`rsgispy.vector.create` using ``geopandas``,
``pyogrio``, ``shapely``, ``rasterio``, and ``h3``.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import shapely
from shapely.geometry import LineString, Point, Polygon, box, shape

import rsgispy
from rsgispy.imageprocess import (
    _INT_DTYPES,
    bbox_from_dataset,
    build_valid_cube_mask,
    ensure_file_exists,
    iter_dataset_windows,
    validate_band_index,
)
from rsgispy.imageutils.metadata import get_img_no_data_value
import rsgispy.tools.files
from rsgispy.tools.geometry import (
    find_extent_on_whole_num_grid,
    get_bbox_geojson_poly,
    get_bbox_grid,
    reproj_bbox,
    unwrap_wgs84_bbox,
)
from rsgispy.tools.projection import get_deg_coord_as_str
import rsgispy.tools.utm
from rsgispy.vector._io_meta import get_vec_lyr_bbox
from rsgispy.vectorprocess import (
    CRSLike,
    VectorInput,
    build_atts_dataframe,
    check_format_name,
    finish_vec_output,
    get_file_vec_extension,
    is_geoparquet_file,
    is_geoparquet_format,
    is_multi_layer_driver,
    list_vec_layers,
    parse_crs_input,
    prepare_optional_out_vec,
    prepare_out_vec_file,
    read_vec_gdf,
    read_vec_info,
    resolve_out_lyr_name,
    write_vec_gdf,
)

__all__ = [
    "polygonise_raster_to_vec_lyr",
    "polygonise_raster_tiled",
    "vectorise_pxls_to_pts",
    "extract_img_footprint",
    "create_poly_vec_for_lst_bboxs",
    "define_grid",
    "create_poly_vec_bboxs",
    "write_pts_to_vec",
    "create_bboxs_for_pts",
    "create_wgs84_vec_grid",
    "create_vec_for_imgs",
    "create_hex_grid_bbox",
    "create_hex_grid_poly",
    "create_hex_grid_polys",
    "create_random_pts_in_radius",
    "create_random_pts_in_bbox",
    "create_lines_vec",
    "create_img_transects",
    "create_bbox_vec_lyr",
]


def _get_h3() -> Any:
    """Import ``h3`` lazily and raise :class:`rsgispy.RSGISPyException` if absent."""
    try:
        import h3  # pylint: disable=import-outside-toplevel

        return h3
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "The 'h3' package is required for hexagon grid functions."
        ) from err


def polygonise_raster_to_vec_lyr(
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    input_img: str,
    img_band: int = 1,
    mask_img: Optional[str] = None,
    mask_band: int = 1,
    *,
    replace_file: bool = True,
    replace_lyr: bool = True,
    pxl_val_fieldname: str = "PXLVAL",
    use_8_conn: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Polygonise connected pixel regions in ``input_img`` into a vector layer
    using ``rasterio.features.shapes``.

    The whole band, every polygon and the output GeoDataFrame are held in
    memory and a single core is used. For rasters producing many polygons
    (e.g., segmentation label rasters) use
    :func:`polygonise_raster_tiled`, which processes the raster in tiles on
    several cores, streams the output to disk and is much faster.

    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver (e.g., ``'GPKG'``).
    :param input_img: Path to input raster file.
    :param img_band: 1-based band index in ``input_img`` to polygonise
                     (Default = ``1``).
    :param mask_img: Optional path to mask raster where non-zero pixels are
                     polygonised (Default = ``None``).
    :param mask_band: 1-based band index in ``mask_img`` (Default = ``1``).
    :param replace_file: If ``True``, delete an existing ``out_vec_file``
                         before writing (Default = ``True``).
    :param replace_lyr: If ``True``, allow overwriting an existing layer of the
                        same name when ``replace_file=False`` (Default = ``True``).
    :param pxl_val_fieldname: Attribute column name holding raster pixel values
                              (Default = ``'PXLVAL'``).
    :param use_8_conn: If ``True``, use 8-connected pixel grouping; otherwise
                       4-connected (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.polygonise_raster_to_vec_lyr(
            "classes.gpkg", "classes", "GPKG", "classes.tif", img_band=1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    target_lyr = (
        resolve_out_lyr_name(out_vec_file, out_vec_lyr)
        if out_vec_file is not None
        else out_vec_lyr
    )
    if (
        out_vec_file is not None
        and (not replace_file)
        and (not replace_lyr)
        and os.path.exists(out_vec_file)
    ):
        fmt = check_format_name(out_format)
        if is_geoparquet_format(fmt) or is_geoparquet_file(out_vec_file):
            raise rsgispy.RSGISPyException(
                f"Output GeoParquet file '{out_vec_file}' already exists and "
                "replace_file=False, replace_lyr=False."
            )
        if not is_multi_layer_driver(out_vec_file, fmt):
            raise rsgispy.RSGISPyException(
                f"Output vector file '{out_vec_file}' already exists and "
                "replace_file=False, replace_lyr=False."
            )
        existing_lyrs = list_vec_layers(out_vec_file)
        if target_lyr in existing_lyrs:
            raise rsgispy.RSGISPyException(
                f"Layer '{target_lyr}' already exists in '{out_vec_file}' "
                "and replace_lyr=False."
            )

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        band_arr = src.read(img_band)
        src_crs = src.crs
        src_transform = src.transform
        is_int = (
            src.dtypes[img_band - 1] in _INT_DTYPES
            or bool(np.issubdtype(band_arr.dtype, np.integer))
        )

    mask_arr: Optional[np.ndarray] = None
    if mask_img is not None:
        ensure_file_exists(mask_img, label="Mask image")
        with rasterio.open(mask_img, "r") as msk_src:
            validate_band_index(mask_band, msk_src.count, mask_img)
            mask_arr = msk_src.read(mask_band).astype(bool)

    connectivity = 8 if use_8_conn else 4
    # rasterio < 1.4 rejects uint32 / int64 / uint64 in rasterio.features.shapes
    supported_shapes_dtypes = {
        np.dtype("int8"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("float32"),
        np.dtype("float64"),
    }
    shapes_arr = band_arr
    remap_vals: Optional[np.ndarray] = None
    if band_arr.dtype not in supported_shapes_dtypes:
        active = band_arr[mask_arr] if mask_arr is not None else band_arr
        if active.size == 0:
            shapes_arr = np.zeros(band_arr.shape, dtype=np.int32)
        elif np.issubdtype(band_arr.dtype, np.integer):
            min_v = int(active.min())
            max_v = int(active.max())
            if -2_147_483_648 <= min_v and max_v <= 2_147_483_647:
                if mask_arr is not None:
                    shapes_arr = np.where(mask_arr, band_arr, 0).astype(np.int32)
                else:
                    shapes_arr = band_arr.astype(np.int32)
        else:
            shapes_arr = band_arr.astype(np.float64)

    geoms: List[Any] = []
    vals: List[Union[int, float]] = []
    try:
        shape_iter = list(
            rasterio.features.shapes(
                shapes_arr,
                mask=mask_arr,
                connectivity=connectivity,
                transform=src_transform,
            )
        )
    except ValueError:
        if np.issubdtype(band_arr.dtype, np.integer):
            active = band_arr[mask_arr] if mask_arr is not None else band_arr.ravel()
            remap_vals, inv = np.unique(active, return_inverse=True)
            remap_arr = np.zeros(band_arr.shape, dtype=np.int32)
            if mask_arr is not None:
                remap_arr[mask_arr] = (inv + 1).astype(np.int32)
            else:
                remap_arr.ravel()[:] = (inv + 1).astype(np.int32)
            shape_iter = list(
                rasterio.features.shapes(
                    remap_arr,
                    mask=mask_arr,
                    connectivity=connectivity,
                    transform=src_transform,
                )
            )
        else:
            raise

    for geom_dict, pxl_val in shape_iter:
        geoms.append(shape(geom_dict))
        if remap_vals is not None:
            vals.append(int(remap_vals[int(pxl_val) - 1]))
        else:
            vals.append(int(pxl_val) if is_int else float(pxl_val))

    if geoms:
        gdf = gpd.GeoDataFrame(
            {pxl_val_fieldname: vals}, geometry=geoms, crs=src_crs
        )
    else:
        col_dtype = np.int64 if is_int else np.float64
        gdf = gpd.GeoDataFrame(
            {pxl_val_fieldname: pd.Series(dtype=col_dtype)},
            geometry=gpd.GeoSeries([], crs=src_crs),
            crs=src_crs,
        )

    if out_vec_file is None:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=target_lyr,
        out_format=out_format,
        overwrite=replace_file,
        out_compress=out_compress,
        replace_layer=replace_lyr,
    )
    return None


def polygonise_raster_tiled(
    input_img: Union[str, os.PathLike],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    img_band: int = 1,
    mask_img: Optional[Union[str, os.PathLike]] = None,
    mask_band: int = 1,
    no_data_val: Optional[float] = None,
    pxl_val_fieldname: str = "PXLVAL",
    use_8_conn: bool = False,
    dissolve_by_value: bool = False,
    sort_by_value: bool = False,
    inc_pxl_count: bool = False,
    tile_size: Union[int, Tuple[int, int]] = 2048,
    n_jobs: int = 1,
    backend: str = "auto",
    *,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    chunk_size: int = 100_000,
    pxl_count_fieldname: str = "pxl_count",
    tmp_dir: Optional[Union[str, os.PathLike]] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> Optional[gpd.GeoDataFrame]:
    """
    Polygonise connected pixel regions in ``input_img`` for rasters producing
    very large numbers of polygons (e.g., segment label rasters from
    :mod:`rsgispy.segmentation` or fine-scale classifications).

    The raster is processed in tiles of ``tile_size`` pixels, optionally in
    ``n_jobs`` worker processes, so memory use does not depend on the image
    size, and the output is streamed to disk in chunks (GeoParquet with
    compression by default, or any OGR vector format). The polygons are the
    same as those from :func:`polygonise_raster_to_vec_lyr` (same coverage,
    values, holes and connectivity) whatever the tile size and number of
    cores, and the output is identical for any ``n_jobs``.

    Polygons that do not touch an internal tile edge are written as soon as
    their tile is processed. Polygons cut by tile edges are merged at the end
    (pieces of the same value sharing an edge, or any contact for
    8-connectivity, are unioned in exact pixel coordinates), so only these
    *seam pieces* are held in memory; their number grows with the total
    length of the tile edges, not with the image area.

    Backends:

    * ``'numba'``: a custom polygonisation kernel (4-connected component
      labelling, then boundary tracing along pixel edges) compiled with
      ``numba``, producing coordinate arrays and WKB directly without
      creating Python objects per polygon. Without ``numba`` the same
      algorithm runs in pure Python (correct, but slow).
    * ``'rasterio'``: :func:`rasterio.features.shapes` (GDAL) for each tile.
    * ``'auto'`` (default): ``'numba'`` when ``numba`` is available,
      otherwise ``'rasterio'``.

    Notes:

    * The output geometries are always valid (:func:`shapely.is_valid`).
      With ``use_8_conn=True``, regions whose parts touch only diagonally
      (at a pixel corner) are written as MultiPolygons with the parts
      touching at that corner. GDAL writes these as a single self-touching
      ring, which is invalid, so in this case the geometries equal
      ``shapely.make_valid`` of GDAL's.
    * Vertices are only written at polygon corners. Map coordinates are
      calculated from the pixel corner coordinates with the image
      geotransform (``x = c + col * a + row * b``), as GDAL does; for
      awkward (e.g., geographic) pixel sizes they can differ from GDAL's in
      the last bit (GDAL builds using fused multiply-add on some platforms).
      Exteriors are counter-clockwise and holes clockwise (north-up
      images).
    * Integer rasters of any type (including ``uint32``, ``int64`` and
      ``uint64``) are supported. Float rasters are polygonised on exact
      value equality (as GDAL does) with ``NaN`` pixels ignored.
    * ``dissolve_by_value=True`` writes one feature per pixel value (a
      MultiPolygon when the value has several disconnected parts), as needed
      for segment label rasters (one row per segment ID). Without
      ``sort_by_value`` the features are in order of each value's first
      polygon.
    * ``sort_by_value`` / ``dissolve_by_value`` work for outputs larger than
      memory: the rows are staged in a temporary Arrow file and sorted with
      a bucket sort (one ``int64`` key per row is held in memory).
    * Memory: each worker (or the main process when ``n_jobs=1``) holds
      the tile it is processing; for a dense segmentation the peak was about
      0.8 GB per worker with ``tile_size=1024`` and 1.9 GB with ``2048``.
      The main process holds the seam pieces of about one row of tiles.
    * Worker processes take 1-2 s to start, so ``n_jobs > 1`` only helps
      for large rasters (tens of millions of pixels).

    Measured with ``benchmarks/polygonise_benchmark.py`` (Apple M2 Max, 8
    performance + 4 efficiency cores) on a 10,000 x 10,000 segmentation-like
    label raster with 16.2 million polygons, writing GeoParquet:
    :func:`polygonise_raster_to_vec_lyr` took 311 s and 38 GB of memory;
    this function took 18.5 s on one core and 8.1 s with ``n_jobs=4``
    (``'numba'`` backend, ``tile_size=2048``), and 107 s / 17 s with the
    ``'rasterio'`` backend on 1 / 8 cores (``tile_size=1024``). A
    ``tile_size`` of 1024-2048 and ``n_jobs`` of about the number of
    performance cores are good defaults (the ``'rasterio'`` backend gains
    more from extra cores).

    :param input_img: Path to the input raster.
    :param out_vec_file: Output vector file (e.g., ``'segs.parquet'``), or
                         ``None`` to return an in-memory
                         :class:`geopandas.GeoDataFrame` (which holds every
                         polygon in memory).
    :param out_vec_lyr: Optional output layer name (multi-layer formats).
    :param img_band: 1-based band to polygonise (Default = ``1``).
    :param mask_img: Optional mask raster (same size as ``input_img``); only
                     pixels where it is non-zero are polygonised.
    :param mask_band: 1-based band of ``mask_img`` (Default = ``1``).
    :param no_data_val: Optional pixel value not polygonised (Default =
                        ``None``: every pixel is polygonised, as in
                        :func:`polygonise_raster_to_vec_lyr`).
    :param pxl_val_fieldname: Column holding the pixel value (Default =
                              ``'PXLVAL'``).
    :param use_8_conn: If ``True``, group pixels with 8-connectivity,
                       otherwise 4-connectivity (Default = ``False``).
    :param dissolve_by_value: If ``True``, write one feature per pixel value.
    :param sort_by_value: If ``True``, write the features sorted by value.
    :param inc_pxl_count: If ``True``, add the number of pixels in each
                          feature (column ``pxl_count_fieldname``).
    :param tile_size: Tile size in pixels, ``int`` or ``(rows, cols)``
                      (Default = ``2048``).
    :param n_jobs: Number of worker processes (Default = ``1``).
    :param backend: ``'auto'`` (default), ``'numba'`` or ``'rasterio'``.
    :param out_format: Output format (``'GeoParquet'`` or an OGR driver name;
                       Default = inferred from the file extension).
    :param out_compress: GeoParquet compression (Default = ``'snappy'``).
    :param chunk_size: Rows per written chunk (Default = ``100000``).
    :param pxl_count_fieldname: Pixel count column (Default =
                                ``'pxl_count'``).
    :param tmp_dir: Directory for temporary files when sorting/dissolving
                    (Default = the output directory).
    :param overwrite: If ``True``, replace an existing output file.
    :param use_pbar: If ``True``, show a progress bar over the tiles.
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        # One feature per segment, sorted by segment ID, using 8 cores.
        rsgispy.vector.create.polygonise_raster_tiled(
            "s2_segs.tif",
            "s2_segs.parquet",
            pxl_val_fieldname="seg_id",
            no_data_val=0,
            dissolve_by_value=True,
            sort_by_value=True,
            n_jobs=8,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=import-outside-toplevel
    from rsgispy._numba import numba_enabled
    from rsgispy._polygonise import (
        PolygoniseParams,
        _TableBuilder,
        check_n_jobs,
        plan_tile_specs,
        run_tiled_polygonise,
    )
    from rsgispy._polygonise_kernels import _label4_kernel
    from rsgispy.ml.vector import ChunkedVecWriter

    in_path = os.path.abspath(os.fspath(input_img))
    ensure_file_exists(in_path)
    backend_l = str(backend).strip().lower()
    if backend_l not in ("auto", "numba", "rasterio"):
        raise rsgispy.RSGISPyException(
            f"backend must be 'auto', 'numba' or 'rasterio', got {backend!r}."
        )
    if backend_l == "auto":
        backend_l = (
            "numba"
            if numba_enabled() and _label4_kernel.jit_func is not None
            else "rasterio"
        )
    jobs = check_n_jobs(n_jobs)
    plan_tile_specs(1, 1, tile_size)  # validate tile_size before reading
    if int(chunk_size) < 1:
        raise rsgispy.RSGISPyException(f"chunk_size must be >= 1, got {chunk_size}.")
    if not pxl_val_fieldname:
        raise rsgispy.RSGISPyException("pxl_val_fieldname must not be empty.")
    count_col = pxl_count_fieldname if inc_pxl_count else None
    if count_col is not None and count_col == pxl_val_fieldname:
        raise rsgispy.RSGISPyException(
            "pxl_count_fieldname and pxl_val_fieldname must be different."
        )

    with rasterio.open(in_path, "r") as src:
        validate_band_index(img_band, src.count, in_path)
        height, width = int(src.height), int(src.width)
        src_dtype = np.dtype(src.dtypes[img_band - 1])
        crs = src.crs
        tr = src.transform
    if np.issubdtype(src_dtype, np.complexfloating):
        raise rsgispy.RSGISPyException("Complex rasters cannot be polygonised.")
    mask_path: Optional[str] = None
    if mask_img is not None:
        mask_path = os.path.abspath(os.fspath(mask_img))
        ensure_file_exists(mask_path, label="Mask image")
        with rasterio.open(mask_path, "r") as msk:
            validate_band_index(mask_band, msk.count, mask_path)
            if (msk.height, msk.width) != (height, width):
                raise rsgispy.RSGISPyException(
                    f"Mask image '{mask_path}' is {msk.height} x {msk.width} pixels "
                    f"but '{in_path}' is {height} x {width}."
                )
    if np.issubdtype(src_dtype, np.floating):
        val_dtype = np.dtype(np.float64)
    elif src_dtype == np.uint64:
        val_dtype = np.dtype(np.uint64)
    else:
        val_dtype = np.dtype(np.int64)

    params = PolygoniseParams(
        input_img=in_path,
        img_band=int(img_band),
        mask_img=mask_path,
        mask_band=int(mask_band),
        no_data_val=None if no_data_val is None else no_data_val,
        use_8_conn=bool(use_8_conn),
        backend=backend_l,
        use_numba=numba_enabled(),
        img_height=height,
        img_width=width,
        geotransform=(tr.c, tr.a, tr.b, tr.f, tr.d, tr.e),
        flip=(tr.a * tr.e - tr.b * tr.d) > 0,
    )
    tiles = plan_tile_specs(height, width, tile_size)
    builder = _TableBuilder(pxl_val_fieldname, val_dtype, count_col)
    pa_mod = builder.pa

    if out_vec_file is None:
        tables: List[Any] = []
        run_tiled_polygonise(
            params,
            tiles,
            jobs,
            builder,
            tables.append,
            int(chunk_size),
            bool(sort_by_value),
            bool(dissolve_by_value),
            None,
            use_pbar,
        )
        tbl = pa_mod.concat_tables(tables) if tables else builder.empty()
        geoms = shapely.from_wkb(tbl.column("geometry").to_numpy(zero_copy_only=False))
        attrs = tbl.drop_columns(["geometry"]).to_pandas()
        return gpd.GeoDataFrame(attrs, geometry=gpd.GeoSeries(geoms, crs=crs), crs=crs)

    out_path = os.path.abspath(os.fspath(out_vec_file))
    work_dir = os.fspath(tmp_dir) if tmp_dir is not None else os.path.dirname(out_path)
    os.makedirs(work_dir, exist_ok=True)
    geom_type = (
        "Polygon"
        if (not use_8_conn and not dissolve_by_value)
        or (out_format is not None and check_format_name(out_format) == "ESRI Shapefile")
        or out_path.lower().endswith(".shp")
        else "Unknown"
    )
    writer = ChunkedVecWriter(
        out_path,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
        geom_col="geometry",
        crs=crs.to_wkt() if crs is not None else None,
        geometry_type=geom_type,
        overwrite=overwrite,
    )
    try:
        n_written = run_tiled_polygonise(
            params,
            tiles,
            jobs,
            builder,
            writer.write,
            int(chunk_size),
            bool(sort_by_value),
            bool(dissolve_by_value),
            work_dir,
            use_pbar,
        )
        if n_written == 0:
            writer.write(builder.empty())
    except BaseException:
        writer.close(success=False)
        raise
    writer.close(success=True)
    return None


def vectorise_pxls_to_pts(
    input_img: str,
    img_band: int,
    img_msk_val: Union[int, float],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_epsg_code: Optional[int] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a point vector layer at the centre of every pixel in ``input_img``
    whose value equals ``img_msk_val``.

    :param input_img: Input raster image path.
    :param img_band: 1-based band index to inspect.
    :param img_msk_val: Pixel value identifying pixels to vectorise as points.
    :param out_vec_file: Optional output vector file path. If ``None``
                         (default), returns an in-memory
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_epsg_code: Optional EPSG code overriding the input image CRS.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar across raster
                     windows (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.vectorise_pxls_to_pts(
            "mask.tif", 1, 1, "mask_pts.gpkg", "mask_pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        if out_epsg_code is not None:
            out_crs: Any = parse_crs_input(int(out_epsg_code))
        elif src.crs is not None:
            out_crs = src.crs
        else:
            raise rsgispy.RSGISPyException(
                "Input image has no CRS defined; supply out_epsg_code."
            )

        pts_x: List[float] = []
        pts_y: List[float] = []
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            arr = src.read(img_band, window=win)
            rows, cols = np.where(arr == img_msk_val)
            if len(rows) == 0:
                continue
            xs, ys = rasterio.transform.xy(
                src.transform,
                rows + int(win.row_off),
                cols + int(win.col_off),
                offset="center",
            )
            pts_x.extend(np.asarray(xs, dtype=float).tolist())
            pts_y.extend(np.asarray(ys, dtype=float).tolist())

    if pts_x:
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pts_x, pts_y), crs=out_crs
        )
        if out_path is None:
            return gdf
        write_vec_gdf(
            gdf,
            out_vec_file=out_path,
            out_vec_lyr=resolve_out_lyr_name(out_path, out_vec_lyr),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
    elif out_path is None:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=out_crs), crs=out_crs
        )
    return None


def extract_img_footprint(
    input_img: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    no_data_val: Optional[float] = None,
    tmp_dir: Optional[str] = None,
    reproj_to: Optional[CRSLike] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Extract the valid-data polygon footprint of ``input_img`` and annotate each
    polygon with a ``FileName`` column containing the raster basename.

    :param input_img: Input raster image path.
    :param out_vec_file: Optional output vector file path. If ``None``
                         (default), returns an in-memory
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param no_data_val: Input no-data pixel value (read from image header if
                        ``None``).
    :param tmp_dir: Optional working directory for intermediate valid mask
                    raster; uses a temporary directory if ``None``.
    :param reproj_to: Optional target CRS (e.g., ``'EPSG:4326'`` or ``4326``)
                      for the output footprint layer.
    :param use_pbar: If ``True``, display a tqdm progress bar during mask
                     generation (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.extract_img_footprint(
            "scene.tif", "footprint.gpkg", "footprint", no_data_val=0.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if (
        out_vec_file is not None
        and not rsgispy.tools.files.does_path_exists_or_creatable(
            os.fspath(out_vec_file)
        )
    ):
        raise rsgispy.RSGISPyException(
            f"Output file path is not creatable: {out_vec_file}"
        )

    eff_nd = (
        no_data_val
        if no_data_val is not None
        else get_img_no_data_value(input_img)
    )
    if eff_nd is None:
        raise rsgispy.RSGISPyException(
            "A no-data value was not specified and is unset on the image."
        )

    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)
    _ = tmp_dir  # retained for signature compatibility; valid mask is computed in memory

    with rasterio.open(input_img, "r") as src:
        height, width = src.height, src.width
        valid_mask = np.ones((height, width), dtype=bool)
        nd_val = float(eff_nd)
        for win in iter_dataset_windows(height, width, use_pbar=use_pbar):
            arr = src.read(window=win)
            block_valid = build_valid_cube_mask(arr, no_data_val=nd_val, axis=0)
            r0 = int(win.row_off)
            c0 = int(win.col_off)
            valid_mask[r0 : r0 + int(win.height), c0 : c0 + int(win.width)] = block_valid

        msk_u8 = valid_mask.astype(np.uint8)
        geoms = [
            shape(geom_dict)
            for geom_dict, _ in rasterio.features.shapes(
                msk_u8,
                mask=valid_mask,
                connectivity=4,
                transform=src.transform,
            )
        ]
        src_crs = src.crs

    gdf = gpd.GeoDataFrame(
        {"FileName": [os.path.basename(input_img)] * len(geoms)},
        geometry=geoms,
        crs=src_crs,
    )
    if reproj_to is not None and len(gdf) > 0:
        gdf = gdf.to_crs(parse_crs_input(reproj_to))

    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def create_poly_vec_for_lst_bboxs(
    csv_file: str,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    epsg_code: CRSLike,
    min_x_col: int = 0,
    max_x_col: int = 1,
    min_y_col: int = 2,
    max_y_col: int = 3,
    ignore_rows: int = 0,
    *,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a polygon vector layer from bounding boxes stored one per row in a
    CSV file.

    :param csv_file: Input CSV file path.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param min_x_col: 0-based CSV column index for minimum X (Default = ``0``).
    :param max_x_col: 0-based CSV column index for maximum X (Default = ``1``).
    :param min_y_col: 0-based CSV column index for minimum Y (Default = ``2``).
    :param max_y_col: 0-based CSV column index for maximum Y (Default = ``3``).
    :param ignore_rows: Number of header rows to skip (Default = ``0``).
    :param del_exist_vec: If ``True``, delete existing ``out_vec_file`` first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_poly_vec_for_lst_bboxs(
            "bboxes.csv", "bboxes.gpkg", "bboxes", "GPKG", 4326, ignore_rows=1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(csv_file, label="CSV file")
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)

    bboxes: List[Tuple[float, float, float, float]] = []
    with open(csv_file, "r", encoding="utf-8") as handle:
        for row_idx, raw_line in enumerate(handle):
            if row_idx < ignore_rows:
                continue
            line = raw_line.strip()
            if not line:
                continue
            comps = line.split(",")
            bboxes.append(
                (
                    float(comps[min_x_col]),
                    float(comps[max_x_col]),
                    float(comps[min_y_col]),
                    float(comps[max_y_col]),
                )
            )

    return create_poly_vec_bboxs(
        vec_file=out_path,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=epsg_code,
        bboxs=bboxes,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def create_poly_vec_bboxs(
    vec_file: Optional[Union[str, os.PathLike]],
    vec_lyr: Optional[str],
    out_format: str,
    epsg_code: CRSLike,
    bboxs: Sequence[Sequence[float]],
    *,
    atts: Optional[Dict[str, Sequence[Any]]] = None,
    att_types: Optional[Dict[str, Sequence[Any]]] = None,
    overwrite: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a polygon vector layer from a sequence of ``(MinX, MaxX, MinY, MaxY)``
    bounding boxes with optional attributes.

    :param vec_file: Output vector file path, or ``None`` to return an
                     in-memory :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param bboxs: Sequence of ``(MinX, MaxX, MinY, MaxY)`` bounding boxes.
    :param atts: Optional mapping ``{col_name: values_sequence}``.
    :param att_types: Optional mapping ``{'names': [...], 'types': [...]}``.
    :param overwrite: If ``True``, replace an existing output file; set
                      ``False`` when adding multiple layers to a GeoPackage
                      (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_poly_vec_bboxs(
            "grid.gpkg",
            "grid",
            "GPKG",
            32630,
            [(500000.0, 501000.0, 5600000.0, 5601000.0)],
            atts={"tile_id": [1]},
            att_types={"names": ["tile_id"], "types": [int]},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    crs_obj = parse_crs_input(epsg_code)
    att_cols = build_atts_dataframe(
        len(bboxs), atts=atts, att_types=att_types, length_label="bboxs"
    )
    geoms = [
        box(float(b[0]), float(b[2]), float(b[1]), float(b[3])) for b in bboxs
    ]
    gdf = gpd.GeoDataFrame(att_cols, geometry=geoms, crs=crs_obj)
    return finish_vec_output(
        gdf,
        out_vec_file=vec_file,
        out_vec_lyr=vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )


def define_grid(
    bbox: Sequence[float],
    x_size: float,
    y_size: float,
    in_epsg_code: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_epsg_code: Optional[int] = None,
    utm_grid: bool = False,
    utm_hemi: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]]:
    """
    Create a regular polygon grid covering ``bbox``, optionally reprojected to
    ``out_epsg_code`` or split across UTM zones when ``utm_grid=True``.

    When ``utm_grid=True`` spans multiple UTM zones, separate layers named
    ``{out_vec_lyr}_utm{zone}{n|s}`` are written into a multi-layer format such
    as ``GPKG``, or returned as a dictionary ``{layer_name: GeoDataFrame}``
    when ``out_vec_file`` is ``None``.

    :param bbox: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param x_size: Grid cell width in output CRS units.
    :param y_size: Grid cell height in output CRS units.
    :param in_epsg_code: EPSG integer code for ``bbox``.
    :param out_vec_file: Optional output vector file path. If ``None``
                         (default), returns an in-memory
                         :class:`geopandas.GeoDataFrame` (or a dictionary of
                         ``GeoDataFrame`` objects for multi-UTM-zone grids).
    :param out_vec_lyr: Output vector layer name (or prefix for multi-UTM grids).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_epsg_code: Optional output EPSG integer code.
    :param utm_grid: If ``True``, output grid tiles in UTM zone projections
                     with ``x_size``/``y_size`` in metres (Default = ``False``).
    :param utm_hemi: If ``True``, distinguish Southern hemisphere UTM zones
                     (``utmXXs``); otherwise always use Northern hemisphere
                     (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` (or ``dict`` mapping layer names to
             ``GeoDataFrame`` objects for multi-UTM-zone grids) when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.define_grid(
            (-4.2, -3.8, 52.2, 52.5),
            10000.0,
            10000.0,
            4326,
            "utm_grid.gpkg",
            "tiles",
            utm_grid=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_bbox: Tuple[float, float, float, float] = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )

    if not utm_grid:
        target_epsg = (
            int(out_epsg_code)
            if out_epsg_code is not None
            else int(in_epsg_code)
        )
        proj_bbox = (
            reproj_bbox(in_bbox, int(in_epsg_code), target_epsg)
            if target_epsg != int(in_epsg_code)
            else in_bbox
        )
        grid_bboxes = get_bbox_grid(proj_bbox, x_size, y_size)
        return create_poly_vec_bboxs(
            vec_file=out_vec_file,
            vec_lyr=out_vec_lyr,
            out_format=out_format,
            epsg_code=target_epsg,
            bboxs=grid_bboxes,
            overwrite=True,
            out_compress=out_compress,
        )

    wgs84_bbox = (
        reproj_bbox(in_bbox, int(in_epsg_code), 4326)
        if int(in_epsg_code) != 4326
        else in_bbox
    )
    split_wgs84 = unwrap_wgs84_bbox(wgs84_bbox)

    if len(split_wgs84) == 1 and (-180.0 <= split_wgs84[0][0] <= 180.0) and (
        -180.0 <= split_wgs84[0][1] <= 180.0
    ):
        wb = split_wgs84[0]
        utm_tl = rsgispy.tools.utm.from_latlon(wb[3], wb[0])
        utm_tr = rsgispy.tools.utm.from_latlon(wb[3], wb[1])
        utm_br = rsgispy.tools.utm.from_latlon(wb[2], wb[1])
        utm_bl = rsgispy.tools.utm.from_latlon(wb[2], wb[0])
        if utm_tl[2] == utm_tr[2] == utm_br[2] == utm_bl[2]:
            hemi = "S" if (utm_hemi and wb[3] < 0) else "N"
            utm_epsg = int(rsgispy.tools.utm.epsg_for_utm(utm_tl[2], hemi))
            return define_grid(
                bbox=in_bbox,
                x_size=x_size,
                y_size=y_size,
                in_epsg_code=in_epsg_code,
                out_vec_file=out_vec_file,
                out_vec_lyr=out_vec_lyr,
                out_format=out_format,
                out_epsg_code=utm_epsg,
                utm_grid=False,
                utm_hemi=False,
                out_compress=out_compress,
            )

    utm_zone_bboxs: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for sub_wb in split_wgs84:
        utm_zone_bboxs.extend(rsgispy.tools.utm.split_wgs84_bbox_utm_zones(sub_wb))

    if out_vec_file is not None and len(utm_zone_bboxs) > 1:
        fmt = check_format_name(out_format)
        if is_geoparquet_format(fmt) or is_geoparquet_file(out_vec_file):
            raise rsgispy.RSGISPyException(
                "GeoParquet does not support multiple layers in a single file; "
                "multi-UTM-zone grids require a multi-layer format such as GPKG."
            )
        if not is_multi_layer_driver(out_vec_file, fmt):
            raise rsgispy.RSGISPyException(
                f"Output format '{fmt}' does not support multiple layers in a "
                "single file; multi-UTM-zone grids require a multi-layer format "
                "such as GPKG."
            )

    lyr_prefix = out_vec_lyr or "grid"
    out_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    first = True
    for zone_num, zone_bbox in utm_zone_bboxs:
        hemi = "S" if (utm_hemi and zone_bbox[3] < 0) else "N"
        utm_epsg = int(rsgispy.tools.utm.epsg_for_utm(zone_num, hemi))
        utm_bbox = reproj_bbox(zone_bbox, 4326, utm_epsg)
        grid_bboxes = get_bbox_grid(utm_bbox, x_size, y_size)
        utm_lyr = f"{lyr_prefix}_utm{zone_num}{hemi.lower()}"
        res_gdf = create_poly_vec_bboxs(
            vec_file=out_vec_file,
            vec_lyr=utm_lyr,
            out_format=out_format,
            epsg_code=utm_epsg,
            bboxs=grid_bboxes,
            overwrite=first,
            out_compress=out_compress,
        )
        if out_vec_file is None and res_gdf is not None:
            out_gdfs[utm_lyr] = res_gdf
        first = False

    if out_vec_file is None:
        return out_gdfs
    return None


def write_pts_to_vec(
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    epsg_code: CRSLike,
    pts_x: Sequence[float],
    pts_y: Sequence[float],
    *,
    atts: Optional[Dict[str, Sequence[Any]]] = None,
    att_types: Optional[Dict[str, Sequence[Any]]] = None,
    replace: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a point vector layer from X/Y coordinates and optional attributes.

    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param pts_x: Sequence of point X coordinates.
    :param pts_y: Sequence of point Y coordinates.
    :param atts: Optional mapping ``{col_name: values_sequence}``.
    :param att_types: Optional mapping ``{'names': [...], 'types': [...]}``.
    :param replace: If ``True``, delete an existing ``out_vec_file`` before
                    writing (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.write_pts_to_vec(
            "pts.gpkg",
            "pts",
            "GPKG",
            32630,
            [500100.0, 500200.0],
            [5600100.0, 5600200.0],
            atts={"id": [1, 2]},
            att_types={"names": ["id"], "types": [int]},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if len(pts_x) != len(pts_y):
        raise rsgispy.RSGISPyException(
            "The X and Y coordinate sequences are not the same length."
        )

    crs_obj = parse_crs_input(epsg_code)
    att_cols = build_atts_dataframe(
        len(pts_x), atts=atts, att_types=att_types, length_label="pts_x"
    )
    gdf = gpd.GeoDataFrame(
        att_cols,
        geometry=gpd.points_from_xy(pts_x, pts_y),
        crs=crs_obj,
    )
    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=replace,
        out_compress=out_compress,
    )


def create_bboxs_for_pts(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    bbox_width: float,
    bbox_height: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    epsg_code: Optional[int] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create rectangular polygon bounding boxes centred on each point in an input
    point vector layer.

    :param vec_file: Input point vector file path or GeoDataFrame.
    :param vec_lyr: Input point vector layer name.
    :param bbox_width: Output rectangle width in CRS units.
    :param bbox_height: Output rectangle height in CRS units.
    :param out_vec_file: Output polygon vector file path, or ``None`` to return
                         an in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output polygon vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete existing ``out_vec_file`` first
                          (Default = ``False``).
    :param epsg_code: Optional EPSG code overriding the input layer CRS.
    :param use_pbar: If ``True``, display a tqdm progress bar across points
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Polygon :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_bboxs_for_pts(
            "pts.gpkg", "pts", 100.0, 100.0, "pt_boxes.gpkg", "pt_boxes"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, epsg=epsg_code, copy=False)
    if gdf.crs is None:
        raise rsgispy.RSGISPyException(
            "Input layer has no CRS defined; supply epsg_code."
        )

    h_width = bbox_width / 2.0
    h_height = bbox_height / 2.0
    pbar_cls = rsgispy.get_tqdm()
    out_bboxs: List[Tuple[float, float, float, float]] = []

    for geom in pbar_cls(gdf.geometry, disable=not use_pbar):
        if geom is None or geom.is_empty:
            continue
        pt_x = float(geom.x)
        pt_y = float(geom.y)
        out_bboxs.append(
            (pt_x - h_width, pt_x + h_width, pt_y - h_height, pt_y + h_height)
        )

    if out_vec_file is None:
        att_cols = build_atts_dataframe(
            len(out_bboxs), atts=None, att_types=None, length_label="bboxs"
        )
        geoms = [
            box(float(b[0]), float(b[2]), float(b[1]), float(b[3]))
            for b in out_bboxs
        ]
        return gpd.GeoDataFrame(att_cols, geometry=geoms, crs=gdf.crs)

    create_poly_vec_bboxs(
        vec_file=out_vec_file,
        vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        epsg_code=gdf.crs,
        bboxs=out_bboxs,
        overwrite=False,
        out_compress=out_compress,
    )
    return None


def create_wgs84_vec_grid(
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    grid_x: float,
    grid_y: float,
    bbox: Sequence[float],
    *,
    overlap: Optional[float] = None,
    tile_names_col: str = "tile_names",
    tile_name_prefix: str = "",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a regular WGS84 (``EPSG:4326``) grid of polygon tiles named by
    their top-left degree coordinate.

    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver.
    :param grid_x: Cell width in longitude degrees.
    :param grid_y: Cell height in latitude degrees.
    :param bbox: Extent ``(MinX, MaxX, MinY, MaxY)`` in degrees.
    :param overlap: Optional overlap buffer added around each tile in degrees.
    :param tile_names_col: Attribute column name for formatted tile names
                           (Default = ``'tile_names'``).
    :param tile_name_prefix: Prefix prepended to every tile name
                             (Default = ``''``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Polygon grid :class:`geopandas.GeoDataFrame` when
             ``out_vec_file`` is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_wgs84_vec_grid(
            "wgs84_grid.gpkg", "grid", "GPKG", 1.0, 1.0, (-5.0, -3.0, 51.0, 53.0)
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    min_x = float(bbox[0])
    max_x = float(bbox[1])
    min_y = float(bbox[2])
    max_y = float(bbox[3])
    cell_x = float(grid_x)
    cell_y = float(grid_y)

    n_x_cells = int(np.floor((max_x - min_x) / cell_x))
    x_remain = (max_x - min_x) - (cell_x * n_x_cells)
    if x_remain < 1e-9:
        x_remain = 0.0

    n_y_cells = int(np.floor((max_y - min_y) / cell_y))
    y_remain = (max_y - min_y) - (cell_y * n_y_cells)
    if y_remain < 1e-9:
        y_remain = 0.0

    x_spans: List[Tuple[float, float]] = [
        (min_x + j * cell_x, min_x + (j + 1) * cell_x)
        for j in range(n_x_cells)
    ]
    if x_remain > 0:
        x_spans.append((min_x + n_x_cells * cell_x, max_x))

    y_spans: List[Tuple[float, float]] = [
        (max_y - (i + 1) * cell_y, max_y - i * cell_y)
        for i in range(n_y_cells)
    ]
    if y_remain > 0:
        y_spans.append((min_y, max_y - n_y_cells * cell_y))

    ovl = float(overlap) if overlap is not None else 0.0
    bboxes: List[Tuple[float, float, float, float]] = []
    tile_names: List[str] = []

    for c_min_y, c_max_y in y_spans:
        for c_min_x, c_max_x in x_spans:
            coord_str = get_deg_coord_as_str(lat=c_max_y, lon=c_min_x, n_chars=4)
            tile_names.append(f"{tile_name_prefix}{coord_str}")
            bx0 = max(-180.0, c_min_x - ovl)
            bx1 = min(180.0, c_max_x + ovl)
            by0 = max(-90.0, c_min_y - ovl)
            by1 = min(90.0, c_max_y + ovl)
            bboxes.append((bx0, bx1, by0, by1))

    return create_poly_vec_bboxs(
        vec_file=out_vec_file,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=4326,
        bboxs=bboxes,
        atts={tile_names_col: tile_names},
        att_types={"names": [tile_names_col], "types": [str]},
        overwrite=True,
        out_compress=out_compress,
    )


def create_vec_for_imgs(
    input_imgs: Sequence[str],
    output_dir: Optional[str] = None,
    *,
    out_format: str = "GeoJSON",
    geometry_type: int = rsgispy.GEOM_PT,
    out_name_replace: Optional[Dict[str, str]] = None,
    out_file_ext: Optional[str] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[List[str], List[gpd.GeoDataFrame]]:
    """
    Create a single-feature geometry vector layer (point at centre, diagonal
    line from top-left to bottom-right, or bounding-box polygon) for each raster
    in ``input_imgs``.

    :param input_imgs: Sequence of input raster image file paths.
    :param output_dir: Output directory path (created if absent), or ``None``
                       to return a list of in-memory
                       :class:`geopandas.GeoDataFrame` objects instead of
                       writing files.
    :param out_format: Output vector format driver (Default = ``'GeoJSON'``).
    :param geometry_type: One of ``rsgispy.GEOM_PT`` (default),
                          ``rsgispy.GEOM_LINE``, or ``rsgispy.GEOM_POLY``.
    :param out_name_replace: Optional mapping ``{old_sub: new_sub}`` applied to
                             each input basename before naming outputs.
    :param out_file_ext: Output file extension without dot (derived from
                         ``out_format`` when ``None``).
    :param del_exist_vec: If ``True``, overwrite existing output vector files
                          (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar across images
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: List of created output vector file paths when ``output_dir`` is
             provided, or a list of :class:`geopandas.GeoDataFrame` objects
             when ``output_dir`` is ``None``.

    .. code:: python

        import rsgispy
        import rsgispy.vector.create

        out_vecs = rsgispy.vector.create.create_vec_for_imgs(
            ["img1.tif", "img2.tif"],
            "out_vecs",
            geometry_type=rsgispy.GEOM_POLY,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if geometry_type not in (rsgispy.GEOM_PT, rsgispy.GEOM_LINE, rsgispy.GEOM_POLY):
        raise rsgispy.RSGISPyException(
            "geometry_type must be rsgispy.GEOM_PT, rsgispy.GEOM_LINE, "
            "or rsgispy.GEOM_POLY."
        )

    fmt = check_format_name(out_format)
    ext = (
        out_file_ext
        if out_file_ext is not None
        else get_file_vec_extension(fmt)
    )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    out_gdfs: List[gpd.GeoDataFrame] = []
    for img in pbar_cls(input_imgs, disable=not use_pbar):
        ensure_file_exists(img)

        with rasterio.open(img, "r") as src:
            img_crs = src.crs
            img_bbox = bbox_from_dataset(src)

        if geometry_type == rsgispy.GEOM_PT:
            geom: Any = Point(
                img_bbox[0] + ((img_bbox[1] - img_bbox[0]) / 2.0),
                img_bbox[2] + ((img_bbox[3] - img_bbox[2]) / 2.0),
            )
        elif geometry_type == rsgispy.GEOM_LINE:
            geom = LineString(
                [(img_bbox[0], img_bbox[3]), (img_bbox[1], img_bbox[2])]
            )
        else:
            geom = box(img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3])

        gdf = gpd.GeoDataFrame(geometry=[geom], crs=img_crs)
        if output_dir is None:
            out_gdfs.append(gdf)
            continue

        basename = rsgispy.tools.files.get_file_basename(img, check_valid=True)
        if out_name_replace is not None:
            for old_s, new_s in out_name_replace.items():
                basename = basename.replace(old_s, new_s)

        out_vec_file = os.path.join(output_dir, f"{basename}.{ext}")
        prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
        write_vec_gdf(
            gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=basename,
            out_format=fmt,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
        out_files.append(out_vec_file)

    if output_dir is None:
        return out_gdfs
    return out_files


def _h3_ids_to_gdf(hex_ids: Sequence[Any], h3_mod: Any) -> Optional[gpd.GeoDataFrame]:
    """Convert H3 cell identifiers to a WGS84 Polygon GeoDataFrame."""
    ids = list(hex_ids)
    if not ids:
        return None
    polys = [
        Polygon([(lon, lat) for lat, lon in h3_mod.cell_to_boundary(hid)])
        for hid in ids
    ]
    return gpd.GeoDataFrame(
        {"hex_id": ids}, geometry=polys, crs="EPSG:4326"
    )


def create_hex_grid_bbox(
    bbox: Sequence[float],
    bbox_epsg: int,
    hex_scale: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create an H3 hexagon grid covering ``bbox`` and export it in ``bbox_epsg``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_epsg: EPSG integer code for ``bbox`` and the output layer.
    :param hex_scale: H3 resolution level (integer ``0..15``; higher means
                      smaller hexagons).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Hexagon :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_hex_grid_bbox(
            (-4.2, -3.9, 52.3, 52.5), 4326, 7, "hex_bbox.gpkg", "hex"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    h3_mod = _get_h3()
    in_bbox: Tuple[float, float, float, float] = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )
    bbox_wgs84 = (
        reproj_bbox(in_bbox, int(bbox_epsg), 4326)
        if int(bbox_epsg) != 4326
        else in_bbox
    )
    bbox_geojson = get_bbox_geojson_poly(bbox_wgs84)
    hex_ids = h3_mod.h3shape_to_cells(
        h3_mod.geo_to_h3shape(bbox_geojson), int(hex_scale)
    )

    h3_gdf = _h3_ids_to_gdf(hex_ids, h3_mod)
    if h3_gdf is None:
        if out_vec_file is None:
            crs_obj = parse_crs_input(int(bbox_epsg))
            return gpd.GeoDataFrame(
                {"hex_id": pd.Series(dtype="object")},
                geometry=gpd.GeoSeries([], crs=crs_obj),
                crs=crs_obj,
            )
        return None
    if int(bbox_epsg) != 4326:
        h3_gdf = h3_gdf.to_crs(epsg=int(bbox_epsg))

    return finish_vec_output(
        h3_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def create_hex_grid_poly(
    poly_series: Any,
    hex_scale: int,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create an H3 hexagon GeoDataFrame covering a single WGS84
    (``EPSG:4326``) polygon row or geometry.

    :param poly_series: GeoSeries row (having ``.geometry``) or Shapely
                        Polygon/MultiPolygon in ``EPSG:4326``.
    :param hex_scale: H3 resolution level (integer ``0..15``).
    :return: GeoDataFrame with ``hex_id`` and ``geometry`` in ``EPSG:4326``,
             or ``None`` if no hexagon centres fall inside the polygon.

    .. code:: python

        import geopandas as gpd
        import rsgispy.vector.create

        gdf = gpd.read_file("roi_wgs84.gpkg")
        hex_gdf = rsgispy.vector.create.create_hex_grid_poly(gdf.iloc[0], 7)
    """
    h3_mod = _get_h3()
    geom = (
        poly_series.geometry
        if hasattr(poly_series, "geometry")
        else poly_series
    )
    if geom is None or geom.is_empty:
        return None
    if geom.has_z:
        geom = shapely.force_2d(geom)

    hex_ids = h3_mod.h3shape_to_cells(
        h3_mod.geo_to_h3shape(geom.__geo_interface__), int(hex_scale)
    )
    return _h3_ids_to_gdf(hex_ids, h3_mod)


def create_hex_grid_polys(
    vec_in_file: VectorInput,
    vec_in_lyr: Optional[str],
    hex_scale: int,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create an H3 hexagon grid covering all polygons in ``vec_in_file`` and
    export the result in the input layer's coordinate reference system.

    :param vec_in_file: Input polygon vector file path or GeoDataFrame.
    :param vec_in_lyr: Input vector layer name.
    :param hex_scale: H3 resolution level (integer ``0..15``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar across polygons
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Hexagon :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_hex_grid_polys(
            "polys.gpkg", "polys", 7, "hex_polys.gpkg", "hex_polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    polys_gdf = read_vec_gdf(vec_in_file, vec_lyr=vec_in_lyr, copy=False)
    orig_crs = polys_gdf.crs
    reproj_poly = orig_crs is not None and (
        orig_crs != parse_crs_input(4326)
    )
    if reproj_poly:
        polys_gdf = polys_gdf.to_crs(epsg=4326)

    pbar_cls = rsgispy.get_tqdm()
    hex_parts: List[gpd.GeoDataFrame] = []
    for geom in pbar_cls(
        polys_gdf.geometry, total=len(polys_gdf), disable=not use_pbar
    ):
        hex_tmp = create_hex_grid_poly(geom, hex_scale)
        if hex_tmp is not None and len(hex_tmp) > 0:
            hex_parts.append(hex_tmp)

    if not hex_parts:
        if out_vec_file is None:
            return gpd.GeoDataFrame(
                {"hex_id": pd.Series(dtype="object")},
                geometry=gpd.GeoSeries([], crs=orig_crs),
                crs=orig_crs,
            )
        return None

    hex_gdf = gpd.GeoDataFrame(
        pd.concat(hex_parts, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    if reproj_poly and orig_crs is not None:
        hex_gdf = hex_gdf.to_crs(orig_crs)

    return finish_vec_output(
        hex_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def create_random_pts_in_radius(
    centre_x: float,
    centre_y: float,
    radius: float,
    n_pts: int,
    epsg_code: CRSLike,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    n_pts_multi_bbox: float = 3.0,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate ``n_pts`` uniformly distributed random points inside a circle of
    ``radius`` around ``(centre_x, centre_y)``, recording ``x_coords``,
    ``y_coords``, ``dist``, and clockwise north-referenced azimuth ``angle``.

    :param centre_x: X coordinate of circle centre (projected CRS).
    :param centre_y: Y coordinate of circle centre (projected CRS).
    :param radius: Circle radius in CRS units.
    :param n_pts: Number of points to generate.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param n_pts_multi_bbox: Oversampling factor within the enclosing square
                             (Default = ``3.0``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Points :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_random_pts_in_radius(
            500000.0, 5600000.0, 250.0, 50, 32630, "circle_pts.gpkg", "pts",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rng = np.random.default_rng(rnd_seed)
    n_cand = int(n_pts * n_pts_multi_bbox)
    x_cand = rng.uniform(centre_x - radius, centre_x + radius, n_cand)
    y_cand = rng.uniform(centre_y - radius, centre_y + radius, n_cand)

    dist = np.hypot(x_cand - centre_x, y_cand - centre_y)
    in_mask = dist <= radius
    x_in = x_cand[in_mask]
    y_in = y_cand[in_mask]

    if len(x_in) < n_pts:
        raise rsgispy.RSGISPyException(
            "An insufficient number of points fell within the radius; "
            "increase n_pts_multi_bbox."
        )

    sel_idx = rng.choice(len(x_in), size=n_pts, replace=False)
    x_sel = x_in[sel_idx]
    y_sel = y_in[sel_idx]

    dx = x_sel - centre_x
    dy = y_sel - centre_y
    angles = (90.0 - np.rad2deg(np.arctan2(dy, dx))) % 360.0

    data_gdf = gpd.GeoDataFrame(
        {
            "x_coords": x_sel,
            "y_coords": y_sel,
            "dist": np.hypot(dx, dy),
            "angle": angles,
        },
        geometry=gpd.points_from_xy(x_sel, y_sel),
        crs=parse_crs_input(epsg_code),
    )
    return finish_vec_output(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def create_random_pts_in_bbox(
    bbox: Sequence[float],
    n_pts: int,
    epsg_code: CRSLike,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Generate ``n_pts`` uniformly distributed random points within
    ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param n_pts: Number of random points to generate.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Points :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_random_pts_in_bbox(
            (500000.0, 501000.0, 5600000.0, 5601000.0),
            100,
            32630,
            "bbox_pts.gpkg",
            "pts",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rng = np.random.default_rng(rnd_seed)
    x_coords = rng.uniform(float(bbox[0]), float(bbox[1]), n_pts)
    y_coords = rng.uniform(float(bbox[2]), float(bbox[3]), n_pts)

    data_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=x_coords, y=y_coords),
        crs=parse_crs_input(epsg_code),
    )
    return finish_vec_output(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )


def create_lines_vec(
    vec_file: Optional[Union[str, os.PathLike]],
    vec_lyr: Optional[str],
    out_format: str,
    epsg_code: CRSLike,
    lines: Sequence[Sequence[Tuple[float, float]]],
    *,
    overwrite: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a vector layer of polylines (each given as ``[(x, y), ...]``
    vertices) with sequential ``line_id`` attributes.

    :param vec_file: Output vector file path, or ``None`` to return an
                     in-memory :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Output vector layer name (or ``None``).
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param lines: Sequence of vertex sequences defining each ``LineString``.
    :param overwrite: If ``True``, delete an existing ``vec_file`` first;
                      set ``False`` when appending layers to a GeoPackage
                      (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_lines_vec(
            "lines.gpkg",
            "lines",
            "GPKG",
            32630,
            [[(0.0, 0.0), (10.0, 10.0)], [(0.0, 10.0), (10.0, 0.0)]],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    geoms = [
        LineString([(float(pt[0]), float(pt[1])) for pt in line])
        for line in lines
    ]
    gdf = gpd.GeoDataFrame(
        {"line_id": list(range(1, len(lines) + 1))},
        geometry=geoms,
        crs=parse_crs_input(epsg_code),
    )
    if vec_file is None:
        return gdf
    write_vec_gdf(
        gdf,
        out_vec_file=vec_file,
        out_vec_lyr=vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )
    return None


def create_img_transects(
    input_img: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    x_intervals: Optional[Sequence[float]] = None,
    y_intervals: Optional[Sequence[float]] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create axis-aligned transect lines across ``input_img`` at fractional
    positions ``0..1`` along X and/or Y.

    :param input_img: Input raster image path.
    :param out_vec_file: Optional output vector file path. If ``None``
                         (default), returns an in-memory
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name (or ``None``).
    :param x_intervals: Sequence of fractional positions ``[0..1]`` along X for
                        vertical (North-South) transects.
    :param y_intervals: Sequence of fractional positions ``[0..1]`` along Y for
                        horizontal (East-West) transects.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: :class:`geopandas.GeoDataFrame` if ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_img_transects(
            "scene.tif", "transects.gpkg", "transects", x_intervals=[0.25, 0.5, 0.75]
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if (x_intervals is None) and (y_intervals is None):
        raise rsgispy.RSGISPyException(
            "Provide at least one of x_intervals or y_intervals."
        )

    for label, vals in (("x_intervals", x_intervals), ("y_intervals", y_intervals)):
        if vals is not None:
            for val in vals:
                if (val < 0.0) or (val > 1.0):
                    raise rsgispy.RSGISPyException(
                        f"{label} values must be between 0 and 1."
                    )

    ensure_file_exists(input_img)
    with rasterio.open(input_img, "r") as src:
        img_bbox = bbox_from_dataset(src)
        img_crs = src.crs
    if img_crs is None:
        raise rsgispy.RSGISPyException(
            f"Input image '{input_img}' does not have a CRS defined."
        )

    img_width = img_bbox[1] - img_bbox[0]
    img_height = img_bbox[3] - img_bbox[2]

    out_lines: List[List[Tuple[float, float]]] = []
    if x_intervals is not None:
        for frac in x_intervals:
            x_pos = img_bbox[0] + (img_width * float(frac))
            out_lines.append([(x_pos, img_bbox[2]), (x_pos, img_bbox[3])])
    if y_intervals is not None:
        for frac in y_intervals:
            y_pos = img_bbox[2] + (img_height * float(frac))
            out_lines.append([(img_bbox[0], y_pos), (img_bbox[1], y_pos)])

    return create_lines_vec(
        vec_file=out_vec_file,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=img_crs,
        lines=out_lines,
        overwrite=True,
        out_compress=out_compress,
    )


def create_bbox_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    whole_num_grid: bool = False,
    whole_num_base_grid: float = 10.0,
    whole_num_round: Optional[int] = None,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a single-polygon vector layer representing the bounding box of
    ``vec_file``, optionally snapped to a whole-number grid.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param whole_num_grid: Snap bounding box onto a whole-number grid via
                           :func:`rsgispy.tools.geometry.find_extent_on_whole_num_grid`
                           (Default = ``False``).
    :param whole_num_base_grid: Grid cell size in CRS units (Default = ``10.0``).
    :param whole_num_round: Optional decimal places to round coordinates.
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :return: Bounding-box :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.create

        rsgispy.vector.create.create_bbox_vec_lyr(
            "polys.gpkg", "polys", "bbox.gpkg", "bbox", whole_num_grid=True
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    info = read_vec_info(vec_file, vec_lyr=vec_lyr)
    vec_bbox = get_vec_lyr_bbox(vec_file, vec_lyr=vec_lyr, compute_if_exp=True)
    x_min, x_max, y_min, y_max = (
        float(vec_bbox[0]),
        float(vec_bbox[1]),
        float(vec_bbox[2]),
        float(vec_bbox[3]),
    )
    if not all(np.isfinite([x_min, x_max, y_min, y_max])):
        raise rsgispy.RSGISPyException(
            "Input vector layer has no valid geometries to compute a bounding box."
        )

    if whole_num_grid:
        x_min, x_max, y_min, y_max = find_extent_on_whole_num_grid(
            base_extent=(x_min, x_max, y_min, y_max),
            base_grid=whole_num_base_grid,
            full_contain=True,
            round_vals=whole_num_round,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(x_min, y_min, x_max, y_max)],
        crs=info.get("crs"),
    )
    if out_vec_file is None:
        return bbox_gdf
    write_vec_gdf(
        bbox_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None
