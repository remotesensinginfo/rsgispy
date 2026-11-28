"""
Shared pytest fixtures and file-writing helpers.

rsgispy plotting functions return their figures open (so they show in a
notebook), so a test or worked example that does not close a figure would
leave it open for the next test in the same process. Every figure still open
after a test is closed here, so tests that check that no figures are left
open only see their own.

:func:`write_raster` and :func:`write_vector` write the small synthetic
rasters and vector layers the tests use. Test modules import them with
``from conftest import write_raster, write_vector``; the ``tmp_raster`` and
``tmp_vector`` fixtures write into the test's ``tmp_path``.
"""

import os
import sys

import numpy as np
import pytest

# Default grid of the synthetic test rasters: UTM zone 30N, 10 m pixels.
DEFAULT_CRS = "EPSG:32630"
DEFAULT_ORIGIN = (500000.0, 5200000.0)
DEFAULT_PIXEL_SIZE = 10.0


def write_raster(
    path,
    data,
    *,
    dtype=None,
    nodata=None,
    crs=DEFAULT_CRS,
    transform=None,
    origin=DEFAULT_ORIGIN,
    pixel_size=DEFAULT_PIXEL_SIZE,
    band_names=None,
    colormap=None,
    colormap_band=1,
    tags=None,
    band_tags=None,
    driver="GTiff",
    **creation_options,
):
    """
    Write a NumPy array to a raster file (a GeoTIFF by default).

    :param path: the output file path.
    :param data: a 2D array (one band), a 3D ``(bands, rows, cols)`` array or a
                 sequence of 2D arrays (one per band).
    :param dtype: the output data type; the array is cast to it. None keeps the
                  array's data type.
    :param nodata: the no-data value; None writes no no-data value.
    :param crs: the CRS (anything rasterio accepts); None writes no CRS.
    :param transform: the affine transform. None builds a north-up transform
                      from ``origin`` and ``pixel_size``; when ``origin`` is
                      also None no transform is written.
    :param origin: the ``(x, y)`` of the top-left corner.
    :param pixel_size: the pixel size, a number or ``(x_size, y_size)``.
    :param band_names: optional band descriptions, one per band.
    :param colormap: an optional colour table ``{value: (r, g, b[, a])}``.
    :param colormap_band: the band the colour table is written to.
    :param tags: optional dataset metadata tags.
    :param band_tags: optional ``{band: {key: value}}`` band metadata tags.
    :param driver: the GDAL driver.
    :param creation_options: other profile entries or creation options
                             (e.g. ``tiled=True``, ``blockxsize=16``).
    :return: the path as a string.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    import rasterio  # pylint: disable=import-outside-toplevel
    from rasterio.transform import from_origin  # pylint: disable=import-outside-toplevel

    if isinstance(data, (list, tuple)):
        arr = np.stack([np.asarray(band) for band in data])
    else:
        arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if dtype is not None:
        arr = arr.astype(dtype)
    count, height, width = arr.shape
    profile = {
        "driver": driver,
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
    }
    if crs is not None:
        profile["crs"] = crs
    if transform is None and origin is not None:
        if np.ndim(pixel_size) == 0:
            x_size = y_size = float(pixel_size)
        else:
            x_size, y_size = (float(val) for val in pixel_size)
        transform = from_origin(origin[0], origin[1], x_size, y_size)
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata
    profile.update(creation_options)
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)
        if band_names is not None:
            for idx, name in enumerate(band_names, start=1):
                dst.set_band_description(idx, name)
        if colormap is not None:
            dst.write_colormap(colormap_band, colormap)
        if tags:
            dst.update_tags(**tags)
        if band_tags:
            for band, band_tag_dict in band_tags.items():
                dst.update_tags(band, **band_tag_dict)
    return str(path)


def write_vector(path, data, *, geometry=None, crs=DEFAULT_CRS, layer=None, driver=None):
    """
    Write a vector layer (GeoPackage, GeoJSON, shapefile or GeoParquet).

    :param path: the output file path; ``.parquet`` / ``.geoparquet`` files are
                 written as GeoParquet.
    :param data: a GeoDataFrame, or a dict / DataFrame of attribute columns
                 with ``geometry`` giving the geometries.
    :param geometry: the geometries when ``data`` is not a GeoDataFrame.
    :param crs: the CRS used when ``data`` is not a GeoDataFrame (or has no CRS).
    :param layer: the layer name (for multi-layer formats).
    :param driver: the OGR driver; None picks it from the file extension.
    :return: the path as a string.
    """
    import geopandas as gpd  # pylint: disable=import-outside-toplevel

    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
    else:
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    path = str(path)
    if os.path.splitext(path)[1].lower() in (".parquet", ".geoparquet"):
        gdf.to_parquet(path)
    else:
        kwargs = {}
        if layer is not None:
            kwargs["layer"] = layer
        if driver is not None:
            kwargs["driver"] = driver
        gdf.to_file(path, **kwargs)
    return path


@pytest.fixture
def tmp_raster(tmp_path):
    """Write a raster into ``tmp_path``: ``tmp_raster(name, data, **kwargs)``
    (the keyword arguments of :func:`write_raster`) returns its path."""

    def _tmp_raster(name, data, **kwargs):
        return write_raster(tmp_path / name, data, **kwargs)

    return _tmp_raster


@pytest.fixture
def tmp_vector(tmp_path):
    """Write a vector layer into ``tmp_path``: ``tmp_vector(name, data, **kwargs)``
    (the keyword arguments of :func:`write_vector`) returns its path."""

    def _tmp_vector(name, data, **kwargs):
        return write_vector(tmp_path / name, data, **kwargs)

    return _tmp_vector


@pytest.fixture(autouse=True)
def _close_open_figures():
    """Close all Matplotlib figures left open by a test."""
    yield
    plt_mod = sys.modules.get("matplotlib.pyplot")
    if plt_mod is not None:
        plt_mod.close("all")
