"""
The zonalstats.polygon module provides pixel-in-polygon and polygon-centroid
zonal statistics functions using ``rasterio`` and ``geopandas``.

* If pixels are small compared to the polygons so each polygon contains at least
  one pixel centre, use :func:`calc_zonal_band_stats` or
  :func:`calc_zonal_band_stats_file`.
* If pixels are large compared to the polygons, use
  :func:`calc_zonal_poly_pts_band_stats` or
  :func:`calc_zonal_poly_pts_band_stats_file` to sample each polygon centroid.
* If polygon sizes vary relative to pixel size, use
  :func:`calc_zonal_band_stats_test_poly_pts` or
  :func:`calc_zonal_band_stats_test_poly_pts_file`, which computes polygon
  statistics where pixels fall inside the polygon and falls back to the polygon
  centroid otherwise.
* For categorical integer rasters, use :func:`calc_zonal_band_pxl_counts` or
  :func:`calc_zonal_band_pxl_counts_file`.
* To export all individual pixels within polygons across selected bands to a
  :class:`pandas.DataFrame` or tabular file (including compressed Parquet), use
  :func:`export_poly_pxls_to_df`.
"""

# pylint: disable=too-many-lines

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import rasterio.windows

import rsgispy
import rsgispy.tools.stats
from rsgispy.tools.tables import (
    resolve_table_output_spec as _resolve_table_output_spec,
    write_dataframe_table as _write_dataframe_table,
)
from rsgispy.vectorprocess import VectorInput
from rsgispy.zonalstats._helpers import (
    is_valid_pixel_value as _is_valid_pixel_value,
    open_validated_zonal_inputs as _open_validated_zonal_inputs,
    open_validated_zonal_multiband_inputs as _open_validated_zonal_multiband_inputs,
    read_vec_file as _read_vec_file,
    sample_point_pixel as _sample_point_pixel,
    write_vec_file as _write_vec_file,
)

__all__ = [
    "calc_zonal_band_stats_file",
    "calc_zonal_band_stats",
    "calc_zonal_poly_pts_band_stats_file",
    "calc_zonal_poly_pts_band_stats",
    "calc_zonal_band_stats_test_poly_pts_file",
    "calc_zonal_band_stats_test_poly_pts",
    "calc_zonal_band_pxl_counts_file",
    "calc_zonal_band_pxl_counts",
    "export_poly_pxls_to_df",
]


def _get_feature_window(
    geom_bounds: Tuple[float, float, float, float],
    transform: Any,
    img_width: int,
    img_height: int,
) -> Window:
    """Compute a clipped integer pixel :class:`rasterio.windows.Window` for ``geom_bounds``."""
    min_x, min_y, max_x, max_y = geom_bounds
    raw_win = rasterio.windows.from_bounds(
        min_x, min_y, max_x, max_y, transform=transform
    )
    col_off = max(0, int(math.floor(raw_win.col_off)))
    row_off = max(0, int(math.floor(raw_win.row_off)))
    col_end = min(img_width, int(math.ceil(raw_win.col_off + raw_win.width)))
    row_end = min(img_height, int(math.ceil(raw_win.row_off + raw_win.height)))
    return Window(
        col_off,
        row_off,
        max(0, col_end - col_off),
        max(0, row_end - row_off),
    )


def _read_polygon_pixels(
    src: rasterio.DatasetReader,
    img_band: int,
    geom: Any,
    img_nodata: Optional[float],
    thresholds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Rasterise ``geom`` over its intersecting raster window and return a 1D array
    of pixel values matching the polygon, nodata, and optional threshold filters.
    """
    if geom is None or geom.is_empty:
        return np.empty(0, dtype=np.float64)

    window = _get_feature_window(
        geom.bounds, src.transform, src.width, src.height
    )
    if window.width <= 0 or window.height <= 0:
        return np.empty(0, dtype=np.float64)

    src_array = src.read(img_band, window=window)
    sub_transform = rasterio.windows.transform(window, src.transform)
    rv_array = rasterize(
        [(geom, 1)],
        out_shape=(int(window.height), int(window.width)),
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    valid_mask = rv_array == 1
    if img_nodata is not None:
        if isinstance(img_nodata, float) and math.isnan(img_nodata):
            valid_mask &= ~np.isnan(src_array)
        else:
            valid_mask &= src_array != img_nodata

    if thresholds is not None:
        valid_mask &= (src_array >= thresholds[0]) & (src_array <= thresholds[1])

    return src_array[valid_mask]


def _compute_stats_dict(
    vals: np.ndarray,
    field_map: Dict[str, Optional[str]],
    percentile: Optional[float] = None,
) -> Dict[str, float]:
    """Compute requested zonal statistics for a non-empty 1D pixel array."""
    stats_out: Dict[str, float] = {}
    if field_map.get("min") is not None:
        stats_out["min"] = float(vals.min())
    if field_map.get("max") is not None:
        stats_out["max"] = float(vals.max())
    if field_map.get("mean") is not None:
        stats_out["mean"] = float(vals.mean(dtype=np.float64))
    if field_map.get("stddev") is not None:
        stats_out["stddev"] = float(vals.std(dtype=np.float64))
    if field_map.get("sum") is not None:
        stats_out["sum"] = float(vals.sum(dtype=np.float64))
    if field_map.get("count") is not None:
        stats_out["count"] = float(vals.shape[0])
    if field_map.get("mode") is not None:
        unique_vals, counts = np.unique(vals, return_counts=True)
        stats_out["mode"] = float(unique_vals[int(np.argmax(counts))])
    if field_map.get("median") is not None:
        stats_out["median"] = float(np.median(vals))
    if field_map.get("percentile") is not None and percentile is not None:
        stats_out["percentile"] = float(np.percentile(vals, float(percentile)))
    return stats_out


def _validate_percentile(
    percentile: Optional[float],
    percentile_field: Optional[str],
) -> None:
    """Validate percentile arguments when ``percentile_field`` is requested."""
    if percentile_field is not None:
        if percentile is None:
            raise rsgispy.RSGISPyException(
                "percentile_field specified but percentile value was None."
            )
        rsgispy.tools.stats.validate_percentiles_0_100(percentile)


def _centroid_fallback_stats(
    src: rasterio.DatasetReader,
    img_band: int,
    geom: Any,
    thresholds: Tuple[float, float],
    out_no_data_val: float,
) -> Tuple[bool, float]:
    """Sample a polygon's centroid and return ``(is_valid, value_or_nodata)``."""
    img_nodata = src.nodatavals[img_band - 1]
    sampled: Optional[float] = None
    if geom is not None and not geom.is_empty:
        centroid = geom.centroid
        sampled = _sample_point_pixel(
            src, img_band, centroid.x, centroid.y, clamp=False
        )

    if _is_valid_pixel_value(sampled, img_nodata, thresholds[0], thresholds[1]):
        return True, float(sampled)  # type: ignore[arg-type]
    return False, float(out_no_data_val)


def _calc_zonal_stats_core(
    vec_gdf: gpd.GeoDataFrame,
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    field_map: Dict[str, Optional[str]],
    vec_def_epsg: Optional[int] = None,
    percentile: Optional[float] = None,
    allow_centroid_fallback: bool = False,
    use_pbar: bool = True,
) -> gpd.GeoDataFrame:
    """Shared implementation for polygon zonal statistics with optional centroid fallback."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if all(name is None for name in field_map.values()):
        raise rsgispy.RSGISPyException(
            "At least one field needs to be specified for there to be an output."
        )

    if allow_centroid_fallback:
        _validate_percentile(percentile, field_map.get("percentile"))

    with _open_validated_zonal_inputs(
        vec_gdf, input_img, img_band, vec_def_epsg=vec_def_epsg
    ) as (src, img_nodata, geom_series):
        results: Dict[str, List[float]] = {
            key: [] for key, name in field_map.items() if name is not None
        }
        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(
            geom_series, total=len(vec_gdf), disable=not use_pbar
        ):
            vals = _read_polygon_pixels(
                src,
                img_band,
                geom,
                img_nodata,
                thresholds=(min_thres, max_thres),
            )
            if vals.shape[0] > 0:
                feat_stats = _compute_stats_dict(
                    vals, field_map, percentile=percentile
                )
            elif allow_centroid_fallback:
                is_valid, c_val = _centroid_fallback_stats(
                    src,
                    img_band,
                    geom,
                    (min_thres, max_thres),
                    out_no_data_val,
                )
                if is_valid:
                    feat_stats = {
                        key: (
                            0.0
                            if key == "stddev"
                            else (1.0 if key == "count" else c_val)
                        )
                        for key in results
                    }
                else:
                    feat_stats = {
                        key: (0.0 if key == "count" else c_val) for key in results
                    }
            else:
                feat_stats = {
                    key: (0.0 if key == "count" else float(out_no_data_val))
                    for key in results
                }

            for key, stat_vals in results.items():
                stat_vals.append(feat_stats[key])

    for key, stat_vals in results.items():
        col_name = field_map[key]
        if col_name is not None:
            vec_gdf[col_name.lower()] = stat_vals

    return vec_gdf


def calc_zonal_band_stats_file(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    min_field: Optional[str] = None,
    max_field: Optional[str] = None,
    mean_field: Optional[str] = None,
    stddev_field: Optional[str] = None,
    sum_field: Optional[str] = None,
    count_field: Optional[str] = None,
    mode_field: Optional[str] = None,
    median_field: Optional[str] = None,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate zonal statistics for a raster band over polygons in ``vec_file``
    (a vector file path or an in-memory :class:`geopandas.GeoDataFrame`).

    Use this function when raster pixels are small relative to the polygons so
    each polygon contains at least one pixel centre.

    :param vec_file: Input vector file path (updated in-place on disk unless
                     ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats /
                    in-memory GeoDataFrames).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for pixels included in stats.
    :param max_thres: Upper inclusive threshold for pixels included in stats.
    :param out_no_data_val: Output fill value when a polygon has no valid pixels.
    :param min_field: Output column name for minimum value (or ``None``).
    :param max_field: Output column name for maximum value (or ``None``).
    :param mean_field: Output column name for mean value (or ``None``).
    :param stddev_field: Output column name for standard deviation (or ``None``).
    :param sum_field: Output column name for sum value (or ``None``).
    :param count_field: Output column name for valid pixel count (or ``None``).
    :param mode_field: Output column name for modal value (or ``None``).
    :param median_field: Output column name for median value (or ``None``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.polygon

        rsgispy.zonalstats.polygon.calc_zonal_band_stats_file(
            vec_file="polygons.gpkg",
            vec_lyr="polygons",
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            min_field="b1_min",
            max_field="b1_max",
            mean_field="b1_mean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_band_stats(
        vec_gdf,
        input_img,
        img_band,
        min_thres,
        max_thres,
        out_no_data_val,
        min_field=min_field,
        max_field=max_field,
        mean_field=mean_field,
        stddev_field=stddev_field,
        sum_field=sum_field,
        count_field=count_field,
        mode_field=mode_field,
        median_field=median_field,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
    )
    if out_vec_file is not None:
        _write_vec_file(
            vec_gdf,
            out_vec_file,
            out_vec_lyr if out_vec_lyr is not None else vec_lyr,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return vec_gdf
    _write_vec_file(
        vec_gdf,
        vec_file,  # type: ignore[arg-type]
        out_vec_lyr if out_vec_lyr is not None else vec_lyr,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def calc_zonal_band_stats(
    vec_gdf: VectorInput,
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    min_field: Optional[str] = None,
    max_field: Optional[str] = None,
    mean_field: Optional[str] = None,
    stddev_field: Optional[str] = None,
    sum_field: Optional[str] = None,
    count_field: Optional[str] = None,
    mode_field: Optional[str] = None,
    median_field: Optional[str] = None,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics for a raster band over polygons in a
    :class:`geopandas.GeoDataFrame` or vector file path.

    When given a :class:`geopandas.GeoDataFrame`, it is updated in-place and
    returned. When given a file path, the layer is loaded into a new
    GeoDataFrame and returned.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing polygon geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for pixels included in stats.
    :param max_thres: Upper inclusive threshold for pixels included in stats.
    :param out_no_data_val: Output fill value when a polygon has no valid pixels.
    :param min_field: Output column name for minimum value (or ``None``).
    :param max_field: Output column name for maximum value (or ``None``).
    :param mean_field: Output column name for mean value (or ``None``).
    :param stddev_field: Output column name for standard deviation (or ``None``).
    :param sum_field: Output column name for sum value (or ``None``).
    :param count_field: Output column name for valid pixel count (or ``None``).
    :param mode_field: Output column name for modal value (or ``None``).
    :param median_field: Output column name for median value (or ``None``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.zonalstats.polygon

        vec_gdf = gpd.read_file("polygons.gpkg", layer="polygons")
        out_gdf = rsgispy.zonalstats.polygon.calc_zonal_band_stats(
            vec_gdf=vec_gdf,
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            mean_field="b1_mean",
            stddev_field="b1_std",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)
    field_map: Dict[str, Optional[str]] = {
        "min": min_field,
        "max": max_field,
        "mean": mean_field,
        "stddev": stddev_field,
        "sum": sum_field,
        "count": count_field,
        "mode": mode_field,
        "median": median_field,
    }
    return _calc_zonal_stats_core(
        vec_gdf=vec_gdf,
        input_img=input_img,
        img_band=img_band,
        min_thres=min_thres,
        max_thres=max_thres,
        out_no_data_val=out_no_data_val,
        field_map=field_map,
        vec_def_epsg=vec_def_epsg,
        allow_centroid_fallback=False,
        use_pbar=use_pbar,
    )


def calc_zonal_poly_pts_band_stats_file(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    out_field: str,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Extract raster band values at each polygon's centroid for ``vec_file``
    (a vector file path or an in-memory :class:`geopandas.GeoDataFrame`).

    Useful when intersecting coarse-resolution imagery with small polygons.

    :param vec_file: Input vector file path (updated in-place on disk unless
                     ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats /
                    in-memory GeoDataFrames).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param out_field: Output column name in the vector layer.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.polygon

        rsgispy.zonalstats.polygon.calc_zonal_poly_pts_band_stats_file(
            vec_file="polygons.gpkg",
            vec_lyr="polygons",
            input_img="img.tif",
            img_band=1,
            out_field="b1_pt_val",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_poly_pts_band_stats(
        vec_gdf,
        input_img,
        img_band,
        out_field,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
    )
    if out_vec_file is not None:
        _write_vec_file(
            vec_gdf,
            out_vec_file,
            out_vec_lyr if out_vec_lyr is not None else vec_lyr,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return vec_gdf
    _write_vec_file(
        vec_gdf,
        vec_file,  # type: ignore[arg-type]
        out_vec_lyr if out_vec_lyr is not None else vec_lyr,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def calc_zonal_poly_pts_band_stats(
    vec_gdf: VectorInput,
    input_img: str,
    img_band: int,
    out_field: str,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Extract raster band values at each polygon's centroid for a
    :class:`geopandas.GeoDataFrame` or vector file path.

    When given a :class:`geopandas.GeoDataFrame`, it is updated in-place and
    returned. When given a file path, the layer is loaded into a new
    GeoDataFrame and returned.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing polygon geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param out_field: Output column name in ``vec_gdf``.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.zonalstats.polygon

        vec_gdf = gpd.read_file("polygons.gpkg", layer="polygons")
        out_gdf = rsgispy.zonalstats.polygon.calc_zonal_poly_pts_band_stats(
            vec_gdf=vec_gdf,
            input_img="img.tif",
            img_band=1,
            out_field="b1_pt_val",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)

    if not out_field:
        raise rsgispy.RSGISPyException(
            "Output field specified as None or empty, a name needs to be given."
        )

    with _open_validated_zonal_inputs(
        vec_gdf, input_img, img_band, vec_def_epsg=vec_def_epsg
    ) as (src, _, geom_series):
        out_vals: List[float] = []
        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(
            geom_series, total=len(vec_gdf), disable=not use_pbar
        ):
            if geom is None or geom.is_empty:
                out_vals.append(float("nan"))
            else:
                centroid = geom.centroid
                sampled = _sample_point_pixel(
                    src, img_band, centroid.x, centroid.y, clamp=False
                )
                out_vals.append(
                    float(sampled) if sampled is not None else float("nan")
                )

    vec_gdf[out_field.lower()] = out_vals
    return vec_gdf


def calc_zonal_band_stats_test_poly_pts_file(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    percentile: Optional[float] = None,
    percentile_field: Optional[str] = None,
    min_field: Optional[str] = None,
    max_field: Optional[str] = None,
    mean_field: Optional[str] = None,
    stddev_field: Optional[str] = None,
    sum_field: Optional[str] = None,
    count_field: Optional[str] = None,
    mode_field: Optional[str] = None,
    median_field: Optional[str] = None,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate zonal statistics for polygons in ``vec_file`` (a vector file
    path or an in-memory :class:`geopandas.GeoDataFrame`), falling back to the
    polygon centroid if no valid pixels fall within the polygon.

    :param vec_file: Input vector file path (updated in-place on disk unless
                     ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats /
                    in-memory GeoDataFrames).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for pixels included in stats.
    :param max_thres: Upper inclusive threshold for pixels included in stats.
    :param out_no_data_val: Output fill value when neither polygon nor centroid
                            yields a valid pixel.
    :param percentile: Percentile in ``[0, 100]`` when ``percentile_field`` is set.
    :param percentile_field: Output column name for percentile value (or ``None``).
    :param min_field: Output column name for minimum value (or ``None``).
    :param max_field: Output column name for maximum value (or ``None``).
    :param mean_field: Output column name for mean value (or ``None``).
    :param stddev_field: Output column name for standard deviation (or ``None``).
    :param sum_field: Output column name for sum value (or ``None``).
    :param count_field: Output column name for valid pixel count (or ``None``).
    :param mode_field: Output column name for modal value (or ``None``).
    :param median_field: Output column name for median value (or ``None``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.polygon

        rsgispy.zonalstats.polygon.calc_zonal_band_stats_test_poly_pts_file(
            vec_file="polygons.gpkg",
            vec_lyr="polygons",
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            percentile=95.0,
            percentile_field="b1_p95",
            mean_field="b1_mean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_band_stats_test_poly_pts(
        vec_gdf,
        input_img,
        img_band,
        min_thres,
        max_thres,
        out_no_data_val,
        percentile=percentile,
        percentile_field=percentile_field,
        min_field=min_field,
        max_field=max_field,
        mean_field=mean_field,
        stddev_field=stddev_field,
        sum_field=sum_field,
        count_field=count_field,
        mode_field=mode_field,
        median_field=median_field,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
    )
    if out_vec_file is not None:
        _write_vec_file(
            vec_gdf,
            out_vec_file,
            out_vec_lyr if out_vec_lyr is not None else vec_lyr,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return vec_gdf
    _write_vec_file(
        vec_gdf,
        vec_file,  # type: ignore[arg-type]
        out_vec_lyr if out_vec_lyr is not None else vec_lyr,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def calc_zonal_band_stats_test_poly_pts(
    vec_gdf: VectorInput,
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    percentile: Optional[float] = None,
    percentile_field: Optional[str] = None,
    min_field: Optional[str] = None,
    max_field: Optional[str] = None,
    mean_field: Optional[str] = None,
    stddev_field: Optional[str] = None,
    sum_field: Optional[str] = None,
    count_field: Optional[str] = None,
    mode_field: Optional[str] = None,
    median_field: Optional[str] = None,
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics for polygons in a :class:`geopandas.GeoDataFrame`
    or vector file path, falling back to the polygon centroid if no valid pixels
    fall inside.

    When given a :class:`geopandas.GeoDataFrame`, it is updated in-place and
    returned. When given a file path, the layer is loaded into a new
    GeoDataFrame and returned.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing polygon geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for pixels included in stats.
    :param max_thres: Upper inclusive threshold for pixels included in stats.
    :param out_no_data_val: Output fill value when neither polygon nor centroid
                            yields a valid pixel.
    :param percentile: Percentile in ``[0, 100]`` when ``percentile_field`` is set.
    :param percentile_field: Output column name for percentile value (or ``None``).
    :param min_field: Output column name for minimum value (or ``None``).
    :param max_field: Output column name for maximum value (or ``None``).
    :param mean_field: Output column name for mean value (or ``None``).
    :param stddev_field: Output column name for standard deviation (or ``None``).
    :param sum_field: Output column name for sum value (or ``None``).
    :param count_field: Output column name for valid pixel count (or ``None``).
    :param mode_field: Output column name for modal value (or ``None``).
    :param median_field: Output column name for median value (or ``None``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.zonalstats.polygon

        vec_gdf = gpd.read_file("polygons.gpkg", layer="polygons")
        out_gdf = rsgispy.zonalstats.polygon.calc_zonal_band_stats_test_poly_pts(
            vec_gdf=vec_gdf,
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            mean_field="b1_mean",
            median_field="b1_med",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)
    field_map: Dict[str, Optional[str]] = {
        "min": min_field,
        "max": max_field,
        "mean": mean_field,
        "stddev": stddev_field,
        "sum": sum_field,
        "count": count_field,
        "mode": mode_field,
        "median": median_field,
        "percentile": percentile_field,
    }
    return _calc_zonal_stats_core(
        vec_gdf=vec_gdf,
        input_img=input_img,
        img_band=img_band,
        min_thres=min_thres,
        max_thres=max_thres,
        out_no_data_val=out_no_data_val,
        field_map=field_map,
        vec_def_epsg=vec_def_epsg,
        percentile=percentile,
        allow_centroid_fallback=True,
        use_pbar=use_pbar,
    )


def calc_zonal_band_pxl_counts_file(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    vals_lut: Dict[int, str],
    total_field: str = "pxl_count",
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Count pixels matching specific integer values in ``vals_lut`` within each
    polygon of ``vec_file`` (a vector file path or an in-memory
    :class:`geopandas.GeoDataFrame`).

    :param vec_file: Input vector file path (updated in-place on disk unless
                     ``out_vec_file`` is given) or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats /
                    in-memory GeoDataFrames).
    :param input_img: Path to the input integer/categorical raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param vals_lut: Mapping ``{pixel_value: output_field_name}``.
    :param total_field: Output column name for total pixels inside each polygon.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting (``None`` to
                         infer from extension or write uncompressed, ``True``
                         for ``'snappy'``, ``False``/``'none'`` for uncompressed,
                         or codec name ``'snappy'``, ``'gzip'``, ``'zstd'``,
                         ``'brotli'``, ``'lz4'``).
    :param out_vec_file: Optional separate output vector file path.
    :param out_vec_lyr: Optional output vector layer name (defaults to
                        ``vec_lyr``).
    :return: Updated :class:`geopandas.GeoDataFrame` when ``vec_file`` is a
             ``GeoDataFrame`` and ``out_vec_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.polygon

        rsgispy.zonalstats.polygon.calc_zonal_band_pxl_counts_file(
            vec_file="polygons.gpkg",
            vec_lyr="polygons",
            input_img="landcover.tif",
            img_band=1,
            vals_lut={1: "forest_pxls", 2: "water_pxls"},
            total_field="total_pxls",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_band_pxl_counts(
        vec_gdf,
        input_img,
        img_band,
        vals_lut,
        total_field=total_field,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
    )
    if out_vec_file is not None:
        _write_vec_file(
            vec_gdf,
            out_vec_file,
            out_vec_lyr if out_vec_lyr is not None else vec_lyr,
            out_compress=out_compress,
        )
        return None
    if is_gdf_input:
        return vec_gdf
    _write_vec_file(
        vec_gdf,
        vec_file,  # type: ignore[arg-type]
        out_vec_lyr if out_vec_lyr is not None else vec_lyr,
        replace_layer=True,
        out_compress=out_compress,
    )
    return None


def calc_zonal_band_pxl_counts(
    vec_gdf: VectorInput,
    input_img: str,
    img_band: int,
    vals_lut: Dict[int, str],
    total_field: str = "pxl_count",
    vec_def_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Count pixels matching specific integer values in ``vals_lut`` within each
    polygon of a :class:`geopandas.GeoDataFrame` or vector file path.

    When given a :class:`geopandas.GeoDataFrame`, it is updated in-place and
    returned. When given a file path, the layer is loaded into a new
    GeoDataFrame and returned.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing polygon geometries.
    :param input_img: Path to the input integer/categorical raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param vals_lut: Mapping ``{pixel_value: output_field_name}``.
    :param total_field: Output column name for total pixels inside each polygon.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas as gpd
        import rsgispy.zonalstats.polygon

        vec_gdf = gpd.read_file("polygons.gpkg", layer="polygons")
        out_gdf = rsgispy.zonalstats.polygon.calc_zonal_band_pxl_counts(
            vec_gdf=vec_gdf,
            input_img="landcover.tif",
            img_band=1,
            vals_lut={1: "forest_pxls", 2: "water_pxls"},
            total_field="total_pxls",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)

    if not vals_lut:
        raise rsgispy.RSGISPyException(
            "At least one field needs to be specified within the dictionary."
        )

    with _open_validated_zonal_inputs(
        vec_gdf, input_img, img_band, vec_def_epsg=vec_def_epsg
    ) as (src, img_nodata, geom_series):
        total_col = total_field.lower()
        lut_cols = {pxl_val: name.lower() for pxl_val, name in vals_lut.items()}
        totals: List[float] = []
        counts_by_val: Dict[int, List[float]] = {
            pxl_val: [] for pxl_val in vals_lut
        }

        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(
            geom_series, total=len(vec_gdf), disable=not use_pbar
        ):
            vals = _read_polygon_pixels(src, img_band, geom, img_nodata=None)
            totals.append(float(vals.shape[0]))
            for pxl_val, count_list in counts_by_val.items():
                if vals.shape[0] > 0:
                    count_list.append(float(np.sum(vals == pxl_val)))
                else:
                    count_list.append(0.0)

    vec_gdf[total_col] = totals
    for pxl_val, col_name in lut_cols.items():
        vec_gdf[col_name] = counts_by_val[pxl_val]

    return vec_gdf


def _resolve_poly_pxl_columns(
    vec_gdf: gpd.GeoDataFrame,
    bands_list: List[int],
    band_names: Optional[Sequence[str]],
    attr_cols: Optional[Sequence[str]],
    include_coords: bool,
) -> Tuple[List[str], List[str]]:
    """Validate and return ``(band_col_names, attr_col_names)``."""
    if band_names is None:
        band_col_names = [f"band_{b}" for b in bands_list]
    else:
        band_col_names = [str(name) for name in band_names]
        if any(not name.strip() for name in band_col_names):
            raise rsgispy.RSGISPyException(
                "band_names must contain non-empty strings."
            )
        if len(band_col_names) != len(bands_list):
            raise rsgispy.RSGISPyException(
                f"Number of band_names ({len(band_col_names)}) must match "
                f"number of selected bands ({len(bands_list)})."
            )

    if len(set(band_col_names)) != len(band_col_names):
        raise rsgispy.RSGISPyException(
            f"Band output column names must be unique: {band_col_names}."
        )

    attr_col_names: List[str] = []
    if attr_cols is not None:
        attr_col_names = [str(c) for c in attr_cols]
        missing = [c for c in attr_col_names if c not in vec_gdf.columns]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Attribute column(s) not found in vector layer: {missing}."
            )
        if vec_gdf.geometry.name in attr_col_names:
            raise rsgispy.RSGISPyException(
                "Geometry column cannot be included in attr_cols."
            )

    reserved = {"poly_fid"} | ({"x", "y"} if include_coords else set())
    all_cols = list(reserved) + attr_col_names + band_col_names
    if len(set(all_cols)) != len(all_cols):
        raise rsgispy.RSGISPyException(
            "Output DataFrame column names collide across poly_fid, x/y, "
            "attr_cols, and band_names."
        )
    return band_col_names, attr_col_names


def _extract_poly_pxl_chunk(
    src: rasterio.DatasetReader,
    bands_list: List[int],
    nodata_vals: List[Optional[float]],
    geom: Any,
    poly_fid: int,
    band_col_names: List[str],
    attr_values: Dict[str, Any],
    mask_nodata: bool,
    all_touched: bool,
    include_coords: bool,
) -> Optional[pd.DataFrame]:
    """Extract all valid pixels inside ``geom`` for ``bands_list`` into a DataFrame."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    if geom is None or geom.is_empty:
        return None

    window = _get_feature_window(
        geom.bounds, src.transform, src.width, src.height
    )
    if window.width <= 0 or window.height <= 0:
        return None

    arr = src.read(bands_list, window=window)
    sub_transform = rasterio.windows.transform(window, src.transform)
    rv_array = rasterize(
        [(geom, 1)],
        out_shape=(int(window.height), int(window.width)),
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )

    valid_mask = rv_array == 1
    if mask_nodata:
        for b_idx, nodata in enumerate(nodata_vals):
            if nodata is not None:
                band_arr = arr[b_idx]
                if isinstance(nodata, float) and math.isnan(nodata):
                    valid_mask &= ~np.isnan(band_arr)
                else:
                    valid_mask &= band_arr != nodata

    rows, cols = np.nonzero(valid_mask)
    n_pxls = int(rows.shape[0])
    if n_pxls == 0:
        return None

    data: Dict[str, Any] = {
        "poly_fid": np.full(n_pxls, poly_fid, dtype=np.int64),
    }
    for attr_name, attr_val in attr_values.items():
        data[attr_name] = [attr_val] * n_pxls

    if include_coords:
        xs, ys = rasterio.transform.xy(sub_transform, rows, cols, offset="center")
        data["x"] = np.asarray(xs, dtype=np.float64)
        data["y"] = np.asarray(ys, dtype=np.float64)

    for b_idx, col_name in enumerate(band_col_names):
        data[col_name] = arr[b_idx, rows, cols]

    return pd.DataFrame(data)


def export_poly_pxls_to_df(
    vec_gdf: VectorInput,
    input_img: str,
    img_bands: Optional[Union[int, Sequence[int]]] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = None,
    band_names: Optional[Sequence[str]] = None,
    attr_cols: Optional[Sequence[str]] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    *,
    vec_lyr: Optional[str] = None,
    use_pbar: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Export all pixel values falling within each polygon of a vector layer to
    a :class:`pandas.DataFrame` or tabular file (including compressed Parquet).

    Each row in the exported table corresponds to one pixel inside a polygon,
    with ``poly_fid`` identifying the 0-based polygon index, optional polygon
    attribute columns (``attr_cols``), optional pixel-centre coordinates
    (``x``, ``y``), and one column per selected raster band.

    :param vec_gdf: Input :class:`geopandas.GeoDataFrame` or vector file path
                    containing polygon geometries.
    :param input_img: Path to the input raster image.
    :param img_bands: 1-based band index, sequence of 1-based band indices, or
                      ``None`` (default) to export all bands in ``input_img``.
    :param out_file: Optional output file path (e.g., ``'pixels.parquet'``,
                     ``'pixels.parquet.zstd'``, ``'pixels.csv'``). When
                     ``None`` (default), returns the :class:`pandas.DataFrame`
                     instead of writing to disk.
    :param out_format: Optional output table format overriding extension
                       detection (e.g., ``'parquet'``, ``'csv'``, ``'tsv'``,
                       ``'feather'``, ``'excel'``, ``'json'``, ``'pickle'``,
                       ``'hdf'``).
    :param out_compress: Optional compression setting when writing to disk.
                         For Parquet: ``None`` to infer from extension (e.g.,
                         ``.parquet.gzip``, ``.parquet.zstd``) or write
                         uncompressed; ``True`` for ``'snappy'``; ``False`` or
                         ``'none'`` for uncompressed; or codec name
                         ``'snappy'``, ``'gzip'``, ``'zstd'``, ``'brotli'``,
                         or ``'lz4'``.
    :param band_names: Optional sequence of output column names matching
                       ``img_bands``. Defaults to ``'band_<n>'``.
    :param attr_cols: Optional sequence of non-geometry column names from
                      ``vec_gdf`` to copy onto each exported pixel row.
    :param mask_nodata: If ``True`` (default), exclude pixels where any
                        selected band matches the raster's nodata value.
    :param all_touched: If ``True``, include all pixels touched by polygon
                        boundaries; if ``False`` (default), include pixels
                        whose centres fall inside the polygon.
    :param include_coords: If ``True``, include pixel-centre ``x`` and ``y``
                           columns in the raster CRS (Default = ``False``).
    :param reproj_vec: Reproject vector geometries on the fly when ``True`` if
                       vector and raster CRSs differ (Default = ``False``).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :param vec_lyr: Optional layer name when ``vec_gdf`` is a file path.
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: :class:`pandas.DataFrame` if ``out_file`` is ``None``, otherwise
             ``None``.

    .. code:: python

        import rsgispy.zonalstats.polygon

        pxls_df = rsgispy.zonalstats.polygon.export_poly_pxls_to_df(
            vec_gdf="polygons.gpkg",
            vec_lyr="polygons",
            input_img="img.tif",
            img_bands=[1, 2, 3],
            band_names=["blue", "green", "red"],
            attr_cols=["class_name"],
        )
        rsgispy.zonalstats.polygon.export_poly_pxls_to_df(
            vec_gdf="polygons.gpkg",
            vec_lyr="polygons",
            input_img="img.tif",
            img_bands=[1, 2, 3],
            out_file="pixels.parquet",
            out_compress="zstd",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    if out_file is not None:
        _resolve_table_output_spec(
            out_file, out_format=out_format, out_compress=out_compress
        )

    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)

    geom_types = set(vec_gdf.geom_type.dropna().unique())
    if geom_types and not geom_types.issubset({"Polygon", "MultiPolygon"}):
        raise rsgispy.RSGISPyException(
            f"The layer geometry type must be Polygon or MultiPolygon: "
            f"Got '{sorted(geom_types)}'."
        )

    chunks: List[pd.DataFrame] = []
    with _open_validated_zonal_multiband_inputs(
        vec_gdf,
        input_img,
        img_bands=img_bands,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, nodata_vals, bands_list, geom_series):
        band_col_names, attr_col_names = _resolve_poly_pxl_columns(
            vec_gdf, bands_list, band_names, attr_cols, include_coords
        )
        attr_arrays = {
            col: vec_gdf[col].to_numpy() for col in attr_col_names
        }
        tqdm_cls = rsgispy.get_tqdm()
        for poly_fid, geom in enumerate(
            tqdm_cls(geom_series, total=len(vec_gdf), disable=not use_pbar)
        ):
            attr_values = {
                col: attr_arrays[col][poly_fid] for col in attr_col_names
            }
            chunk_df = _extract_poly_pxl_chunk(
                src=src,
                bands_list=bands_list,
                nodata_vals=nodata_vals,
                geom=geom,
                poly_fid=poly_fid,
                band_col_names=band_col_names,
                attr_values=attr_values,
                mask_nodata=mask_nodata,
                all_touched=all_touched,
                include_coords=include_coords,
            )
            if chunk_df is not None:
                chunks.append(chunk_df)

    if chunks:
        out_df = pd.concat(chunks, ignore_index=True)
    else:
        empty_data: Dict[str, Any] = {
            "poly_fid": pd.Series(dtype="int64"),
        }
        for col in attr_col_names:
            empty_data[col] = pd.Series(dtype=vec_gdf[col].dtype)
        if include_coords:
            empty_data["x"] = pd.Series(dtype="float64")
            empty_data["y"] = pd.Series(dtype="float64")
        for col in band_col_names:
            empty_data[col] = pd.Series(dtype="float64")
        out_df = pd.DataFrame(empty_data)

    if out_file is not None:
        _write_dataframe_table(
            out_df,
            out_file,
            out_format=out_format,
            out_compress=out_compress,
        )
        return None
    return out_df
