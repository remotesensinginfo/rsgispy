"""
Internal shared vector/raster helpers for :mod:`rsgispy.zonalstats`.
"""

from contextlib import contextmanager
import math
from typing import Iterator, Optional, Tuple

import geopandas
import pyproj
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import bbox_from_dataset, validate_band_index
import rsgispy.tools.geometry


def resolve_vector_crs(
    vec_gdf: geopandas.GeoDataFrame,
    vec_def_epsg: Optional[int] = None,
) -> pyproj.CRS:
    """Resolve the vector CRS from ``vec_def_epsg`` or ``vec_gdf.crs``."""
    if vec_def_epsg is not None:
        return pyproj.CRS.from_epsg(int(vec_def_epsg))
    if vec_gdf.crs is None:
        raise rsgispy.RSGISPyException(
            "Could not retrieve a projection object from the vector layer - "
            "projection might not be defined."
        )
    return pyproj.CRS.from_user_input(vec_gdf.crs)


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


def get_gdf_bbox(
    vec_gdf: geopandas.GeoDataFrame,
) -> Tuple[float, float, float, float]:
    """Return ``(MinX, MaxX, MinY, MaxY)`` from a GeoDataFrame."""
    if vec_gdf.empty:
        raise rsgispy.RSGISPyException(
            "The inputted vector layer contains no features."
        )
    min_x, min_y, max_x, max_y = vec_gdf.total_bounds
    if not all(math.isfinite(val) for val in (min_x, min_y, max_x, max_y)):
        raise rsgispy.RSGISPyException(
            "Could not compute valid bounding box for the vector layer."
        )
    return (float(min_x), float(max_x), float(min_y), float(max_y))


@contextmanager
def open_validated_zonal_inputs(
    vec_gdf: geopandas.GeoDataFrame,
    input_img: str,
    img_band: int,
    vec_def_epsg: Optional[int] = None,
    reproj_vec: bool = False,
) -> Iterator[Tuple[rasterio.DatasetReader, Optional[float], geopandas.GeoSeries]]:
    """
    Validate vector/raster inputs, check CRS and spatial overlap, optionally
    reproject the geometry series to match ``input_img``, and yield
    ``(src_dataset, img_nodata, geom_series)``.
    """
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    vec_crs = resolve_vector_crs(vec_gdf, vec_def_epsg)
    vec_bbox = get_gdf_bbox(vec_gdf)

    with rasterio.open(input_img, "r") as src:
        img_crs, img_nodata, img_bbox = open_raster_band_info(
            src, input_img, img_band
        )
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
        yield src, img_nodata, geom_series


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
        if isinstance(img_nodata, float) and math.isnan(img_nodata):
            return False
        if val == img_nodata:
            return False
    return min_thres <= val <= max_thres


def read_vec_file(
    vec_file: str, vec_lyr: Optional[str]
) -> geopandas.GeoDataFrame:
    """Read a vector file layer into a GeoDataFrame, raising RSGISPyException on error."""
    try:
        return geopandas.read_file(vec_file, layer=vec_lyr)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not open '{vec_file}' (layer='{vec_lyr}'): {err}"
        ) from err


def write_vec_file(
    vec_gdf: geopandas.GeoDataFrame,
    vec_file: str,
    vec_lyr: Optional[str],
) -> None:
    """Write an updated GeoDataFrame back to ``vec_file`` (and optional ``vec_lyr``)."""
    vec_gdf.to_file(vec_file, layer=vec_lyr)
