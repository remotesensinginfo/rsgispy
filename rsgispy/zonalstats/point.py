"""
The zonalstats.point module provides point-in-pixel zonal extraction functions
using ``rasterio`` and ``geopandas``.
"""

from typing import List, Optional

import geopandas

import rsgispy
from rsgispy.zonalstats._helpers import (
    is_valid_pixel_value as _is_valid_pixel_value,
    open_validated_zonal_inputs as _open_validated_zonal_inputs,
    read_vec_file as _read_vec_file,
    sample_point_pixel as _sample_point_pixel,
    write_vec_file as _write_vec_file,
)

__all__ = [
    "ext_point_band_values_file",
    "ext_point_band_values",
]


def ext_point_band_values_file(
    vec_file: str,
    vec_lyr: Optional[str],
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    out_field: str,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
) -> None:
    """
    Extract raster band pixel values at point locations in ``vec_file``.

    :param vec_file: Input point vector file path (updated in-place on disk).
    :param vec_lyr: Vector layer name (or ``None`` for single-layer formats).
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for valid pixel values.
    :param max_thres: Upper inclusive threshold for valid pixel values.
    :param out_no_data_val: Output fill value when a point is outside the image
                            or the sampled pixel is invalid/nodata.
    :param out_field: Output column name in the vector layer.
    :param reproj_vec: Reproject point coordinates on the fly when True if vector
                       and raster CRSs differ (Default = False).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.

    .. code:: python

        import rsgispy.zonalstats.point

        rsgispy.zonalstats.point.ext_point_band_values_file(
            vec_file="points.gpkg",
            vec_lyr="points",
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            out_field="b1_val",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = ext_point_band_values(
        vec_gdf,
        input_img,
        img_band,
        min_thres,
        max_thres,
        out_no_data_val,
        out_field,
        reproj_vec=reproj_vec,
        vec_def_epsg=vec_def_epsg,
    )
    _write_vec_file(vec_gdf, vec_file, vec_lyr)


def ext_point_band_values(
    vec_gdf: geopandas.GeoDataFrame,
    input_img: str,
    img_band: int,
    min_thres: float,
    max_thres: float,
    out_no_data_val: float,
    out_field: str,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
) -> geopandas.GeoDataFrame:
    """
    Extract raster band pixel values at point locations in a
    :class:`geopandas.GeoDataFrame`.

    The GeoDataFrame is updated in-place and returned (original CRS and
    geometries are preserved even when ``reproj_vec=True``).

    :param vec_gdf: Input GeoDataFrame containing point geometries.
    :param input_img: Path to the input raster image.
    :param img_band: 1-based band index within ``input_img``.
    :param min_thres: Lower inclusive threshold for valid pixel values.
    :param max_thres: Upper inclusive threshold for valid pixel values.
    :param out_no_data_val: Output fill value when a point is outside the image
                            or the sampled pixel is invalid/nodata.
    :param out_field: Output column name in ``vec_gdf``.
    :param reproj_vec: Reproject point coordinates on the fly when True if vector
                       and raster CRSs differ (Default = False).
    :param vec_def_epsg: Optional EPSG code overriding vector CRS if undefined.
    :return: Updated :class:`geopandas.GeoDataFrame`.

    .. code:: python

        import geopandas
        import rsgispy.zonalstats.point

        vec_gdf = geopandas.read_file("points.gpkg", layer="points")
        out_gdf = rsgispy.zonalstats.point.ext_point_band_values(
            vec_gdf=vec_gdf,
            input_img="img.tif",
            img_band=1,
            min_thres=0.0,
            max_thres=10000.0,
            out_no_data_val=-999.0,
            out_field="b1_val",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    if not out_field:
        raise rsgispy.RSGISPyException(
            "Output field specified as None or empty, a name needs to be given."
        )

    geom_types = set(vec_gdf.geom_type.dropna().unique())
    if not geom_types or not geom_types.issubset({"Point"}):
        raise rsgispy.RSGISPyException(
            f"The layer geometry type must be point: Got '{sorted(geom_types)}'."
        )

    with _open_validated_zonal_inputs(
        vec_gdf,
        input_img,
        img_band,
        vec_def_epsg=vec_def_epsg,
        reproj_vec=reproj_vec,
    ) as (src, img_nodata, geom_series):
        out_no_data_f = float(out_no_data_val)
        out_vals: List[float] = []
        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(geom_series, total=len(vec_gdf)):
            if geom is None or geom.is_empty:
                out_vals.append(out_no_data_f)
                continue
            sampled = _sample_point_pixel(
                src, img_band, geom.x, geom.y, clamp=False
            )
            if _is_valid_pixel_value(
                sampled, img_nodata, min_thres, max_thres
            ):
                out_vals.append(float(sampled))  # type: ignore[arg-type]
            else:
                out_vals.append(out_no_data_f)

    vec_gdf[out_field.lower()] = out_vals
    return vec_gdf
