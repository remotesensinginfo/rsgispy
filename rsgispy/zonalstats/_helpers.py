"""
Internal shared vector/raster helpers for :mod:`rsgispy.zonalstats`.
"""

from contextlib import contextmanager
import math
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset, validate_band_index
import rsgispy.tools.geometry
from rsgispy.vectorprocess import (
    get_gdf_bbox,
    read_vec_gdf as read_vec_file,
    resolve_vector_crs,
    write_vec_gdf as write_vec_file,
)

__all__ = [
    "check_bbox_overlap",
    "get_gdf_bbox",
    "is_valid_pixel_value",
    "normalize_band_list",
    "open_raster_band_info",
    "open_validated_zonal_inputs",
    "open_validated_zonal_multiband_inputs",
    "read_vec_file",
    "resolve_vector_crs",
    "sample_point_pixel",
    "sample_point_pixels",
    "write_vec_file",
]


def normalize_band_list(
    img_band: Optional[Union[int, Sequence[int]]],
    band_count: int,
    input_img: str,
) -> List[int]:
    """
    Normalise ``img_band`` (``None`` for all bands, an integer, or a sequence of
    integers) into a validated list of 1-based band indices.
    """
    if img_band is None:
        return list(range(1, band_count + 1))
    if isinstance(img_band, bool):
        raise rsgispy.RSGISPyException(
            "img_band must be an integer or sequence of integers, not bool."
        )
    if isinstance(img_band, (int, np.integer)):
        bands = [int(img_band)]
    elif isinstance(img_band, (str, bytes)):
        raise rsgispy.RSGISPyException(
            "img_band must be an integer or sequence of integers, not str."
        )
    else:
        bands = [int(b) for b in img_band]
    if not bands:
        raise rsgispy.RSGISPyException(
            "img_band must contain at least one band index."
        )
    for band_idx in bands:
        validate_band_index(band_idx, band_count, input_img)
    return bands


def open_raster_band_info(
    src: rasterio.DatasetReader,
    input_img: str,
    img_band: int,
) -> Tuple[pyproj.CRS, Optional[float], Tuple[float, float, float, float]]:
    """Validate raster CRS and band index, returning ``(crs, nodata, bbox)``."""
    validate_band_index(img_band, src.count, input_img)
    if src.crs is None:
        raise rsgispy.RSGISPyException(
            f"Could not retrieve a projection from '{input_img}'."
        )
    img_crs = pyproj.CRS.from_user_input(src.crs)
    nodata = src.nodatavals[img_band - 1]
    img_bbox = bbox_from_dataset(src)
    return img_crs, nodata, img_bbox


def check_bbox_overlap(
    img_bbox: Tuple[float, float, float, float],
    vec_bbox: Tuple[float, float, float, float],
) -> None:
    """Raise :class:`rsgispy.RSGISPyException` if raster and vector bboxes do not overlap."""
    if not rsgispy.tools.geometry.do_bboxes_intersect(img_bbox, vec_bbox):
        print(f"Image BBOX: {img_bbox}")
        print(f"Vector BBOX: {vec_bbox}")
        raise rsgispy.RSGISPyException("The vector layer and image do not overlap.")


@contextmanager
def open_validated_zonal_inputs(
    vec_gdf: gpd.GeoDataFrame,
    input_img: str,
    img_band: int,
    vec_def_epsg: Optional[int] = None,
    reproj_vec: bool = False,
) -> Iterator[Tuple[rasterio.DatasetReader, Optional[float], gpd.GeoSeries]]:
    """
    Validate vector/raster inputs, check CRS and spatial overlap, optionally
    reproject the geometry series to match ``input_img``, and yield
    ``(src_dataset, img_nodata, geom_series)``.
    """
    with open_validated_zonal_multiband_inputs(
        vec_gdf,
        input_img,
        img_bands=img_band,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, nodata_vals, _bands, geom_series):
        yield src, nodata_vals[0], geom_series


@contextmanager
def open_validated_zonal_multiband_inputs(
    vec_gdf: gpd.GeoDataFrame,
    input_img: str,
    img_bands: Optional[Union[int, Sequence[int]]] = None,
    vec_def_epsg: Optional[int] = None,
    reproj_vec: bool = False,
) -> Iterator[
    Tuple[
        rasterio.DatasetReader,
        List[Optional[float]],
        List[int],
        gpd.GeoSeries,
    ]
]:
    """
    Validate vector/raster inputs for one or more bands, check CRS and spatial
    overlap, optionally reproject geometries to match ``input_img``, and yield
    ``(src_dataset, band_nodatavals, bands_list, geom_series)``.
    """
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    vec_crs = resolve_vector_crs(vec_gdf, vec_def_epsg)
    vec_bbox = get_gdf_bbox(vec_gdf)

    with rasterio.open(input_img, "r") as src:
        if src.crs is None:
            raise rsgispy.RSGISPyException(
                f"Could not retrieve a projection from '{input_img}'."
            )
        bands_list = normalize_band_list(img_bands, src.count, input_img)
        img_crs = pyproj.CRS.from_user_input(src.crs)
        img_bbox = bbox_from_dataset(src)
        nodata_vals = [src.nodatavals[b - 1] for b in bands_list]

        geom_series = vec_gdf.geometry
        if vec_crs != img_crs:
            if not reproj_vec:
                raise rsgispy.RSGISPyException(
                    f"Inputted raster and vector layers have different projections: "
                    f"(Vector Layer: '{vec_crs}' '{input_img}': '{img_crs}')"
                )
            vec_bbox = rsgispy.tools.geometry.reproj_bbox(
                vec_bbox, vec_crs, img_crs
            )
            geom_series = geom_series.set_crs(
                vec_crs, allow_override=True
            ).to_crs(img_crs)

        check_bbox_overlap(img_bbox, vec_bbox)
        yield src, nodata_vals, bands_list, geom_series


def sample_point_pixel(
    src: rasterio.DatasetReader,
    img_band: int,
    x_coord: float,
    y_coord: float,
    clamp: bool = False,
) -> Optional[float]:
    """Read a single pixel value at ``(x_coord, y_coord)`` or return ``None`` if outside."""
    row, col = src.index(x_coord, y_coord)
    if clamp:
        row = min(max(int(row), 0), src.height - 1)
        col = min(max(int(col), 0), src.width - 1)
    elif row < 0 or row >= src.height or col < 0 or col >= src.width:
        return None

    arr = src.read(img_band, window=Window(int(col), int(row), 1, 1))
    return float(arr[0, 0])


def sample_point_pixels(
    src: rasterio.DatasetReader,
    img_bands: Sequence[int],
    x_coord: float,
    y_coord: float,
    clamp: bool = False,
) -> Optional[np.ndarray]:
    """
    Read pixel values for ``img_bands`` at ``(x_coord, y_coord)`` or return
    ``None`` if outside raster bounds.
    """
    row, col = src.index(x_coord, y_coord)
    if clamp:
        row = min(max(int(row), 0), src.height - 1)
        col = min(max(int(col), 0), src.width - 1)
    elif row < 0 or row >= src.height or col < 0 or col >= src.width:
        return None

    arr = src.read(list(img_bands), window=Window(int(col), int(row), 1, 1))
    return arr[:, 0, 0]


def is_valid_pixel_value(
    val: Optional[float],
    img_nodata: Optional[float],
    min_thres: float,
    max_thres: float,
) -> bool:
    """Check whether a sampled pixel value passes nodata and threshold criteria."""
    if val is None or math.isnan(val):
        return False
    if img_nodata is not None:
        if not (
            isinstance(img_nodata, float) and math.isnan(img_nodata)
        ) and val == img_nodata:
            return False
    return min_thres <= val <= max_thres
