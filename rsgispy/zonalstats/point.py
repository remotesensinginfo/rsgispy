"""
The zonalstats.point module provides point-in-pixel zonal extraction functions
using ``rasterio`` and ``geopandas``.

.. versionchanged:: 0.5.0
   Output column names are written as given: ``out_field="B1"`` creates
   ``B1`` (or ``B1_<band>`` for each band when several are read). They were
   lowercased, as in RSGISLib.
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio.transform
from rasterio.windows import Window

import rsgispy
from rsgispy.vectorprocess import VectorInput
from rsgispy.zonalstats._helpers import (
    load_zonal_vec_gdf as _load_zonal_vec_gdf,
    open_validated_zonal_multiband_inputs as _open_validated_zonal_multiband_inputs,
    read_bands_native as _read_bands_native,
    run_zonal_file_op as _run_zonal_file_op,
)

__all__ = [
    "ext_point_band_values_file",
    "ext_point_band_values",
]


def _resolve_point_out_fields(
    bands: List[int],
    is_scalar_band: bool,
    out_field: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    """Resolve output column name(s) for single- or multi-band point extraction."""
    if is_scalar_band and isinstance(out_field, str):
        if not out_field.strip():
            raise rsgispy.RSGISPyException(
                "Output field specified as empty, a name needs to be given."
            )
        return [out_field]

    if out_field is None:
        cols = [f"band_{b}" for b in bands]
    elif isinstance(out_field, str):
        if not out_field.strip():
            raise rsgispy.RSGISPyException(
                "Output field specified as empty, a name needs to be given."
            )
        cols = [f"{out_field}_{b}" for b in bands]
    else:
        field_list = [str(f) for f in out_field]
        if not field_list or any(not f.strip() for f in field_list):
            raise rsgispy.RSGISPyException(
                "Output field list must contain non-empty column names."
            )
        if len(field_list) != len(bands):
            raise rsgispy.RSGISPyException(
                f"Number of out_field names ({len(field_list)}) must match "
                f"number of bands ({len(bands)})."
            )
        cols = list(field_list)

    if len(set(cols)) != len(cols):
        raise rsgispy.RSGISPyException(
            f"Output field names must be unique: {cols}."
        )
    return cols


def _sample_points_by_block(
    src,
    bands_list: List[int],
    nodata_vals: List[Optional[float]],
    geom_series: gpd.GeoSeries,
    min_thres: float,
    max_thres: float,
    out_no_data_f: float,
    use_pbar: bool,
) -> np.ndarray:
    """
    Sample raster values for ``geom_series`` using block-grouped window reads.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_pts = len(geom_series)
    n_bands = len(bands_list)
    out_arr = np.full((n_bands, n_pts), out_no_data_f, dtype=np.float64)
    if n_pts == 0:
        return out_arr

    valid_geom_mask = (geom_series.notna() & ~geom_series.is_empty).to_numpy()
    valid_indices = np.flatnonzero(valid_geom_mask)
    if valid_indices.size == 0:
        return out_arr

    valid_geoms = geom_series.iloc[valid_indices]
    xs = valid_geoms.x.to_numpy(dtype=np.float64)
    ys = valid_geoms.y.to_numpy(dtype=np.float64)
    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    in_bounds = (
        (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    )
    ib_indices = valid_indices[in_bounds]
    if ib_indices.size == 0:
        return out_arr
    ib_rows = rows[in_bounds]
    ib_cols = cols[in_bounds]

    blk_shapes = getattr(src, "block_shapes", None)
    if blk_shapes and len(blk_shapes) >= bands_list[0]:
        blk_h, blk_w = blk_shapes[bands_list[0] - 1]
    else:
        blk_h, blk_w = 256, 256
    blk_h = max(int(blk_h), 1)
    blk_w = max(int(blk_w), 1)

    groups: Dict[Tuple[int, int], List[int]] = {}
    for pos in range(ib_indices.size):
        key = (int(ib_rows[pos] // blk_h), int(ib_cols[pos] // blk_w))
        groups.setdefault(key, []).append(pos)

    tqdm_cls = rsgispy.get_tqdm()
    with tqdm_cls(total=n_pts, disable=not use_pbar) as pbar:
        skipped = n_pts - int(ib_indices.size)
        if skipped > 0:
            pbar.update(skipped)
        for pos_list in groups.values():
            pos_arr = np.asarray(pos_list, dtype=np.intp)
            g_rows = ib_rows[pos_arr]
            g_cols = ib_cols[pos_arr]
            g_orig = ib_indices[pos_arr]

            r0 = int(g_rows.min())
            r1 = int(g_rows.max()) + 1
            c0 = int(g_cols.min())
            c1 = int(g_cols.max()) + 1
            win = Window(c0, r0, c1 - c0, r1 - r0)
            # Each band in its own data type (bands of different types, e.g.
            # from a VRT stack, cannot be read together).
            block_data = _read_bands_native(src, bands_list, window=win)

            for b_idx in range(n_bands):
                b_native = block_data[b_idx][g_rows - r0, g_cols - c0]
                b_vals = b_native.astype(np.float64, copy=False)
                valid = ~np.isnan(b_vals)
                nd = nodata_vals[b_idx]
                if nd is not None and not (
                    isinstance(nd, float) and math.isnan(nd)
                ):
                    # Compare in the band's data type (e.g. a float32 no
                    # data value of 0.1 is not 0.1 in float64).
                    valid &= b_native != nd
                valid &= (b_vals >= min_thres) & (b_vals <= max_thres)
                out_arr[b_idx, g_orig[valid]] = b_vals[valid]
            pbar.update(int(pos_arr.size))

    return out_arr


def ext_point_band_values_file(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: Optional[Union[int, Sequence[int]]],
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    out_field: Optional[Union[str, Sequence[str]]] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Extract raster band pixel values at point locations in ``vec_file``
    (a vector file path or an in-memory :class:`geopandas.GeoDataFrame`).

    Supports either a single 1-based band index or a sequence of band indices
    (or ``None`` for all bands), exporting one attribute column per band.

    :param vec_file: Input point vector file path (updated in-place on disk
                     unless ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats /
                    in-memory GeoDataFrames).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index, sequence of 1-based band indices, or
                     ``None`` to export all bands in ``input_img``.
    :param min_thres: Lower inclusive threshold for valid pixel values.
    :param max_thres: Upper inclusive threshold for valid pixel values.
    :param out_no_data_val: Output fill value when a point is outside the image
                            or the sampled pixel is invalid/nodata.
    :param out_field: Output column name (for a single band), prefix string
                      (generating ``'<prefix>_<band>'`` for multiple bands),
                      sequence of column names matching ``img_band``, or
                      ``None`` to default to ``'band_<n>'``.
    :param reproj_vec: Reproject point coordinates on the fly when True if vector
                       and raster CRSs differ (Default = False).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path. Required if
                         writing to disk when ``vec_file`` is a
                         :class:`geopandas.GeoDataFrame`; when ``vec_file`` is a
                         ``GeoDataFrame`` and ``out_vec_file`` is ``None``, the
                         updated GeoDataFrame is returned in memory.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.point

        rsgispy.zonalstats.point.ext_point_band_values_file(
            vec_file="points.gpkg",
            vec_lyr="points",
            input_img="img.tif",
            img_band=[1, 2, 3],
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            out_field=["b1_val", "b2_val", "b3_val"],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _run_zonal_file_op(
        vec_file,
        vec_lyr,
        lambda vec_gdf: ext_point_band_values(
            vec_gdf,
            input_img,
            img_band,
            min_thres,
            max_thres,
            out_no_data_val,
            out_field=out_field,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
        ),
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_compress=out_compress,
    )


def ext_point_band_values(
    vec_gdf: VectorInput,
    input_img: str,
    img_band: Optional[Union[int, Sequence[int]]],
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    out_field: Optional[Union[str, Sequence[str]]] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Extract raster band pixel values at point locations in a
    :class:`geopandas.GeoDataFrame` or vector file path.

    Supports either a single 1-based band index or a sequence of band indices
    (or ``None`` for all bands), writing one attribute column per band. The
    bands may have different data types (e.g. a VRT stacking ``uint16`` and
    ``float32`` images); each is compared with its own no data value in its
    own data type.

    When given a :class:`geopandas.GeoDataFrame`, it is updated in-place and
    returned (original CRS and geometries are preserved even when
    ``reproj_vec=True``). When given a file path, the layer is loaded into a
    new GeoDataFrame and returned.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing point geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index, sequence of 1-based band indices, or
                     ``None`` to export all bands in ``input_img``.
    :param min_thres: Lower inclusive threshold for valid pixel values.
    :param max_thres: Upper inclusive threshold for valid pixel values.
    :param out_no_data_val: Output fill value when a point is outside the image
                            or the sampled pixel is invalid/nodata.
    :param out_field: Output column name (for a single band), prefix string
                      (generating ``'<prefix>_<band>'`` for multiple bands),
                      sequence of column names matching ``img_band``, or
                      ``None`` to default to ``'band_<n>'``.
    :param reproj_vec: Reproject point coordinates on the fly when True if vector
                       and raster CRSs differ (Default = False).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.zonalstats.point

        vec_gdf = gpd.read_file("points.gpkg", layer="points")
        out_gdf = rsgispy.zonalstats.point.ext_point_band_values(
            vec_gdf=vec_gdf,
            input_img="img.tif",
            img_band=[1, 2, 3],
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            out_field=["b1_val", "b2_val", "b3_val"],
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    vec_gdf = _load_zonal_vec_gdf(vec_gdf, vec_lyr)

    geom_types = set(vec_gdf.geom_type.dropna().unique())
    if not geom_types or not geom_types.issubset({"Point"}):
        raise rsgispy.RSGISPyException(
            f"The layer geometry type must be point: Got '{sorted(geom_types)}'."
        )

    is_scalar_band = isinstance(img_band, (int, np.integer)) and not isinstance(
        img_band, bool
    )
    with _open_validated_zonal_multiband_inputs(
        vec_gdf,
        input_img,
        img_bands=img_band,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, nodata_vals, bands_list, geom_series):
        out_cols = _resolve_point_out_fields(
            bands_list, is_scalar_band, out_field
        )
        out_arr = _sample_points_by_block(
            src=src,
            bands_list=bands_list,
            nodata_vals=nodata_vals,
            geom_series=geom_series,
            min_thres=min_thres,
            max_thres=max_thres,
            out_no_data_f=float(out_no_data_val),
            use_pbar=use_pbar,
        )

    for b_idx, col_name in enumerate(out_cols):
        vec_gdf[col_name] = out_arr[b_idx].tolist()
    return vec_gdf
