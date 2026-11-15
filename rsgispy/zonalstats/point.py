"""
The zonalstats.point module provides point-in-pixel zonal extraction functions
using ``rasterio`` and ``geopandas``.
"""

import os
from typing import List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np

import rsgispy
from rsgispy.vectorprocess import VectorInput
from rsgispy.zonalstats._helpers import (
    is_valid_pixel_value as _is_valid_pixel_value,
    open_validated_zonal_multiband_inputs as _open_validated_zonal_multiband_inputs,
    read_vec_file as _read_vec_file,
    sample_point_pixels as _sample_point_pixels,
    write_vec_file as _write_vec_file,
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
        return [out_field.lower()]

    if out_field is None:
        cols = [f"band_{b}" for b in bands]
    elif isinstance(out_field, str):
        if not out_field.strip():
            raise rsgispy.RSGISPyException(
                "Output field specified as empty, a name needs to be given."
            )
        cols = [f"{out_field.lower()}_{b}" for b in bands]
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
        cols = [f.lower() for f in field_list]

    if len(set(cols)) != len(cols):
        raise rsgispy.RSGISPyException(
            f"Output field names must be unique: {cols}."
        )
    return cols


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
    is_gdf_input = isinstance(vec_file, gpd.GeoDataFrame)
    vec_gdf = _read_vec_file(vec_file, vec_lyr)
    vec_gdf = ext_point_band_values(
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
        out_compress=out_compress,
    )
    return None


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
    use_pbar: bool = True,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Extract raster band pixel values at point locations in a
    :class:`geopandas.GeoDataFrame` or vector file path.

    Supports either a single 1-based band index or a sequence of band indices
    (or ``None`` for all bands), writing one attribute column per band.

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
    if vec_gdf is None:
        raise rsgispy.RSGISPyException("The inputted vector layer was None")

    if isinstance(vec_gdf, (str, os.PathLike)):
        vec_gdf = _read_vec_file(vec_gdf, vec_lyr)

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
        out_no_data_f = float(out_no_data_val)
        n_bands = len(bands_list)
        out_vals_by_band: List[List[float]] = [[] for _ in range(n_bands)]

        tqdm_cls = rsgispy.get_tqdm()
        for geom in tqdm_cls(
            geom_series, total=len(vec_gdf), disable=not use_pbar
        ):
            if geom is None or geom.is_empty:
                for b_idx in range(n_bands):
                    out_vals_by_band[b_idx].append(out_no_data_f)
                continue
            sampled = _sample_point_pixels(
                src, bands_list, geom.x, geom.y, clamp=False
            )
            if sampled is None:
                for b_idx in range(n_bands):
                    out_vals_by_band[b_idx].append(out_no_data_f)
                continue
            for b_idx in range(n_bands):
                val = float(sampled[b_idx])
                if _is_valid_pixel_value(
                    val, nodata_vals[b_idx], min_thres, max_thres
                ):
                    out_vals_by_band[b_idx].append(val)
                else:
                    out_vals_by_band[b_idx].append(out_no_data_f)

    for col_name, col_vals in zip(out_cols, out_vals_by_band):
        vec_gdf[col_name] = col_vals
    return vec_gdf
