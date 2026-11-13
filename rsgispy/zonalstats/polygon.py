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
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import geopandas
import numpy
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import rasterio.windows

import rsgispy
import rsgispy.tools.stats
from rsgispy.zonalstats._helpers import (
    is_valid_pixel_value as _is_valid_pixel_value,
    open_validated_zonal_inputs as _open_validated_zonal_inputs,
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
) -> numpy.ndarray:
    """
    Rasterize ``geom`` over its intersecting raster window and return a 1D array
    of pixel values matching the polygon, nodata, and optional threshold filters.
    """
    if geom is None or geom.is_empty:
        return numpy.empty(0, dtype=numpy.float64)

    window = _get_feature_window(
        geom.bounds, src.transform, src.width, src.height
    )
    if window.width <= 0 or window.height <= 0:
        return numpy.empty(0, dtype=numpy.float64)

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
            valid_mask &= ~numpy.isnan(src_array)
        else:
            valid_mask &= src_array != img_nodata

    if thresholds is not None:
        valid_mask &= (src_array >= thresholds[0]) & (src_array <= thresholds[1])

    return src_array[valid_mask]


def _compute_stats_dict(
    vals: numpy.ndarray,
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
        stats_out["mean"] = float(vals.mean())
    if field_map.get("stddev") is not None:
        stats_out["stddev"] = float(vals.std())
    if field_map.get("sum") is not None:
        stats_out["sum"] = float(vals.sum())
    if field_map.get("count") is not None:
        stats_out["count"] = float(vals.shape[0])
    if field_map.get("mode") is not None:
        unique_vals, counts = numpy.unique(vals, return_counts=True)
        stats_out["mode"] = float(unique_vals[int(numpy.argmax(counts))])
    if field_map.get("median") is not None:
        stats_out["median"] = float(numpy.median(vals))
    if field_map.get("percentile") is not None and percentile is not None:
        stats_out["percentile"] = float(numpy.percentile(vals, float(percentile)))
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
    vec_gdf: geopandas.GeoDataFrame,
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    field_map: Dict[str, Optional[str]],
    vec_def_epsg: Optional[int] = None,
    percentile: Optional[float] = None,
    allow_centroid_fallback: bool = False,
) -> geopandas.GeoDataFrame:
    """Shared implementation for polygon zonal statistics with optional centroid fallback."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if all(name is None for name in field_map.values()):
        raise rsgispy.RSGISPyException(
            "At least one field needs to be specified for there is to an output."
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
        for geom in tqdm_cls(geom_series, total=len(vec_gdf)):
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
                feat_stats = {key: float(out_no_data_val) for key in results}

            for key, stat_vals in results.items():
                stat_vals.append(feat_stats[key])

    for key, stat_vals in results.items():
        col_name = field_map[key]
        if col_name is not None:
            vec_gdf[col_name.lower()] = stat_vals

    return vec_gdf


def calc_zonal_band_stats_file(
    vec_file: str,
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
) -> None:
    """
    Calculate zonal statistics for a raster band over polygons in ``vec_file``.

    Use this function when raster pixels are small relative to the polygons so
    each polygon contains at least one pixel centre.

    :param vec_file: Input vector file path (updated in-place on disk).
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats).
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
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
    )
    _write_vec_file(vec_gdf, vec_file, vec_lyr)


def calc_zonal_band_stats(
    vec_gdf: geopandas.GeoDataFrame,
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
) -> geopandas.GeoDataFrame:
    """
    Calculate zonal statistics for a raster band over polygons in a
    :class:`geopandas.GeoDataFrame`.

    The GeoDataFrame is updated in-place and returned.

    :param vec_gdf: Input GeoDataFrame containing polygon geometries.
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
    :return: Updated :class:`geopandas.GeoDataFrame`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
    )


def calc_zonal_poly_pts_band_stats_file(
    vec_file: str,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    out_field: str,
    vec_def_epsg: Optional[int] = None,
) -> None:
    """
    Extract raster band values at each polygon's centroid for ``vec_file``.

    Useful when intersecting coarse-resolution imagery with small polygons.

    :param vec_file: Input vector file path (updated in-place on disk).
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param out_field: Output column name in the vector layer.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_poly_pts_band_stats(
        vec_gdf,
        input_img,
        img_band,
        out_field,
        vec_def_epsg=vec_def_epsg,
    )
    _write_vec_file(vec_gdf, vec_file, vec_lyr)


def calc_zonal_poly_pts_band_stats(
    vec_gdf: geopandas.GeoDataFrame,
    input_img: str,
    img_band: int,
    out_field: str,
    vec_def_epsg: Optional[int] = None,
) -> geopandas.GeoDataFrame:
    """
    Extract raster band values at each polygon's centroid for a
    :class:`geopandas.GeoDataFrame`.

    The GeoDataFrame is updated in-place and returned.

    :param vec_gdf: Input GeoDataFrame containing polygon geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param out_field: Output column name in ``vec_gdf``.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :return: Updated :class:`geopandas.GeoDataFrame`.
    """
    if not out_field:
        raise rsgispy.RSGISPyException(
            "Output field specified as None or empty, a name needs to be given."
        )

    with _open_validated_zonal_inputs(
        vec_gdf, input_img, img_band, vec_def_epsg=vec_def_epsg
    ) as (src, _, geom_series):
        out_vals: List[float] = []
        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(geom_series, total=len(vec_gdf)):
            if geom is None or geom.is_empty:
                out_vals.append(float("nan"))
            else:
                centroid = geom.centroid
                sampled = _sample_point_pixel(
                    src, img_band, centroid.x, centroid.y, clamp=True
                )
                out_vals.append(
                    float(sampled) if sampled is not None else float("nan")
                )

    vec_gdf[out_field.lower()] = out_vals
    return vec_gdf


def calc_zonal_band_stats_test_poly_pts_file(
    vec_file: str,
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
) -> None:
    """
    Calculate zonal statistics for polygons in ``vec_file``, falling back to the
    polygon centroid if no valid pixels fall within the polygon.

    :param vec_file: Input vector file path (updated in-place on disk).
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats).
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
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
    )
    _write_vec_file(vec_gdf, vec_file, vec_lyr)


def calc_zonal_band_stats_test_poly_pts(
    vec_gdf: geopandas.GeoDataFrame,
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
) -> geopandas.GeoDataFrame:
    """
    Calculate zonal statistics for polygons in a :class:`geopandas.GeoDataFrame`,
    falling back to the polygon centroid if no valid pixels fall inside.

    The GeoDataFrame is updated in-place and returned.

    :param vec_gdf: Input GeoDataFrame containing polygon geometries.
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
    :return: Updated :class:`geopandas.GeoDataFrame`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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
    )


def calc_zonal_band_pxl_counts_file(
    vec_file: str,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    vals_lut: Dict[int, str],
    total_field: str = "pxl_count",
    vec_def_epsg: Optional[int] = None,
) -> None:
    """
    Count pixels matching specific integer values in ``vals_lut`` within each
    polygon of ``vec_file``.

    :param vec_file: Input vector file path (updated in-place on disk).
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats).
    :param input_img: Path to the input integer/categorical raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param vals_lut: Mapping ``{pixel_value: output_field_name}``.
    :param total_field: Output column name for total pixels inside each polygon.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = calc_zonal_band_pxl_counts(
        vec_gdf,
        input_img,
        img_band,
        vals_lut,
        total_field=total_field,
        vec_def_epsg=vec_def_epsg,
    )
    _write_vec_file(vec_gdf, vec_file, vec_lyr)


def calc_zonal_band_pxl_counts(
    vec_gdf: geopandas.GeoDataFrame,
    input_img: str,
    img_band: int,
    vals_lut: Dict[int, str],
    total_field: str = "pxl_count",
    vec_def_epsg: Optional[int] = None,
) -> geopandas.GeoDataFrame:
    """
    Count pixels matching specific integer values in ``vals_lut`` within each
    polygon of a :class:`geopandas.GeoDataFrame`.

    The GeoDataFrame is updated in-place and returned.

    :param vec_gdf: Input GeoDataFrame containing polygon geometries.
    :param input_img: Path to the input integer/categorical raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param vals_lut: Mapping ``{pixel_value: output_field_name}``.
    :param total_field: Output column name for total pixels inside each polygon.
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :return: Updated :class:`geopandas.GeoDataFrame`.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not vals_lut:
        raise rsgispy.RSGISPyException(
            "At least one field needs to be specified within the dictionary."
        )

    with _open_validated_zonal_inputs(
        vec_gdf, input_img, img_band, vec_def_epsg=vec_def_epsg
    ) as (src, _, geom_series):
        total_col = total_field.lower()
        lut_cols = {pxl_val: name.lower() for pxl_val, name in vals_lut.items()}
        totals: List[float] = []
        counts_by_val: Dict[int, List[float]] = {
            pxl_val: [] for pxl_val in vals_lut
        }

        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(geom_series, total=len(vec_gdf)):
            vals = _read_polygon_pixels(src, img_band, geom, img_nodata=None)
            totals.append(float(vals.shape[0]))
            for pxl_val, count_list in counts_by_val.items():
                if vals.shape[0] > 0:
                    count_list.append(float(numpy.sum(vals == pxl_val)))
                else:
                    count_list.append(0.0)

    vec_gdf[total_col] = totals
    for pxl_val, col_name in lut_cols.items():
        vec_gdf[col_name] = counts_by_val[pxl_val]

    return vec_gdf
