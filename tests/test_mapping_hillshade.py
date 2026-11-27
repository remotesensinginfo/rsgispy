"""
Tests for the hillshade map helpers in ``rsgispy.mapping.raster``
(``add_hillshade``, ``create_shaded_cmap_img_map``) and the ``img_alpha`` /
``interpolation`` / ``interp_stage`` arguments of the raster map functions.
"""

import os

# pylint: disable=redefined-outer-name

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy._numba import disable_numba
import rsgispy.elevation.illumination as illum
import rsgispy.mapping.raster as map_raster

matplotlib.use("Agg")

X_MIN, Y_MAX, RES = 400000.0, 6000000.0, 5.0
N_ROWS, N_COLS = 40, 50
DEM_NODATA = -9999.0


@pytest.fixture(scope="module")
def dem_data(tmp_path_factory):
    """A small DEM (with a no-data corner), its hillshade file and a value raster."""
    data_dir = tmp_path_factory.mktemp("hshade")
    rows, cols = np.mgrid[0:N_ROWS, 0:N_COLS]
    dem = (
        100.0
        + 30.0 * np.sin(cols / 7.0)
        + 20.0 * np.cos(rows / 5.0)
        + 0.5 * rows
    ).astype(np.float32)
    dem[:3, :3] = DEM_NODATA
    profile = {
        "driver": "GTiff",
        "width": N_COLS,
        "height": N_ROWS,
        "count": 1,
        "crs": "EPSG:32630",
        "transform": from_origin(X_MIN, Y_MAX, RES, RES),
    }
    dem_img = os.path.join(data_dir, "dem.tif")
    with rasterio.open(dem_img, "w", dtype="float32", nodata=DEM_NODATA, **profile) as dst:
        dst.write(dem[np.newaxis])
    vals_img = os.path.join(data_dir, "vals.tif")
    with rasterio.open(vals_img, "w", dtype="float32", nodata=-1.0, **profile) as dst:
        dst.write((rows + cols).astype(np.float32)[np.newaxis])
    hshade_img = os.path.join(data_dir, "hshade.tif")
    illum.hillshade(dem_img, hshade_img, 315.0, 45.0, use_pbar=False)
    return {"dem": dem_img, "hshade": hshade_img, "vals": vals_img}


# Pixel-aligned bounding boxes: interior, touching the image edge, whole image.
BBOXES = [
    (X_MIN + 50.0, X_MIN + 150.0, Y_MAX - 120.0, Y_MAX - 40.0),
    (X_MIN, X_MIN + 100.0, Y_MAX - 80.0, Y_MAX),
    None,
]


def _full_extent():
    return [X_MIN, X_MIN + N_COLS * RES, Y_MAX - N_ROWS * RES, Y_MAX]


@pytest.mark.parametrize("bbox", BBOXES)
def test_add_hillshade_extent_matches_bbox(dem_data, bbox):
    """The hillshade artist covers the bbox (the whole image without one)."""
    fig, ax = plt.subplots()
    for kw in ({"hshade_img": dem_data["hshade"]}, {"dem_img": dem_data["dem"]}):
        img = map_raster.add_hillshade(ax, bbox=bbox, **kw)
        exp = _full_extent() if bbox is None else list(bbox)
        np.testing.assert_allclose(img.get_extent(), exp)
        assert img in ax.images
        assert img.get_alpha() == pytest.approx(0.4)
        assert img.get_zorder() == 2.0
        assert img.get_interpolation_stage() == "data"
    plt.close(fig)


@pytest.mark.parametrize("bbox", BBOXES)
def test_add_hillshade_dem_matches_hillshade_file(dem_data, bbox):
    """The in-memory hillshade from the DEM equals the hillshade() file in the bbox."""
    fig, ax = plt.subplots()
    from_file = map_raster.add_hillshade(ax, dem_data["hshade"], bbox=bbox)
    from_dem = map_raster.add_hillshade(ax, dem_img=dem_data["dem"], bbox=bbox)
    arr_file = np.ma.filled(from_file.get_array(), np.nan)
    arr_dem = np.ma.filled(from_dem.get_array(), np.nan)
    assert arr_file.shape == arr_dem.shape
    np.testing.assert_array_equal(arr_file, arr_dem)
    # No-data pixels (the DEM corner) are transparent (NaN, masked).
    if bbox is None or bbox[0] == X_MIN:
        assert np.isnan(arr_dem[0, 0])
        assert np.isfinite(arr_dem).sum() > 0
    plt.close(fig)


def test_add_hillshade_dem_numpy_path(dem_data):
    """The NumPy (no numba) path gives the same in-memory hillshade."""
    fig, ax = plt.subplots()
    from_file = map_raster.add_hillshade(ax, dem_data["hshade"], bbox=BBOXES[1])
    with disable_numba():
        from_dem = map_raster.add_hillshade(ax, dem_img=dem_data["dem"], bbox=BBOXES[1])
    np.testing.assert_array_equal(
        np.ma.filled(from_file.get_array(), np.nan),
        np.ma.filled(from_dem.get_array(), np.nan),
    )
    plt.close(fig)


def test_add_hillshade_values_and_style(dem_data):
    """The drawn values are those of the file, in grey from its min to max."""
    fig, ax = plt.subplots()
    img = map_raster.add_hillshade(ax, dem_data["hshade"], alpha=0.25, zorder=5)
    with rasterio.open(dem_data["hshade"]) as src:
        ref = src.read(1).astype(float)
        ref[ref == src.nodata] = np.nan
    np.testing.assert_array_equal(np.ma.filled(img.get_array(), np.nan), ref)
    assert img.norm.vmin == np.nanmin(ref) and img.norm.vmax == np.nanmax(ref)
    assert img.get_alpha() == 0.25 and img.get_zorder() == 5
    assert img.get_cmap().name.startswith("gray")
    plt.close(fig)


def test_add_hillshade_needs_one_input(dem_data):
    """Exactly one of hshade_img and dem_img must be given."""
    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        map_raster.add_hillshade(ax)
    with pytest.raises(rsgispy.RSGISPyException):
        map_raster.add_hillshade(ax, dem_data["hshade"], dem_img=dem_data["dem"])
    with pytest.raises(rsgispy.RSGISPyException):
        map_raster.create_shaded_cmap_img_map(ax, dem_data["vals"])
    plt.close(fig)


@pytest.mark.parametrize("hshade_kw", ["hshade", "dem"])
def test_create_shaded_cmap_img_map(dem_data, hshade_kw):
    """A colour-mapped raster under a hillshade, with a colour bar."""
    fig, ax = plt.subplots()
    kw = (
        {"hshade_img": dem_data["hshade"]}
        if hshade_kw == "hshade"
        else {"dem_img": dem_data["dem"]}
    )
    img = map_raster.create_shaded_cmap_img_map(
        ax,
        dem_data["vals"],
        vmin=5.0,
        vmax=60.0,
        cmap_name="magma",
        hshade_alpha=0.3,
        cbar_label="Value",
        title_str="Shaded",
        **kw,
    )
    assert plt.get_fignums() == [fig.number]
    assert len(ax.images) == 2
    assert ax.images[0] is img
    hshade = ax.images[1]
    assert hshade.get_zorder() > img.get_zorder()
    assert hshade.get_alpha() == pytest.approx(0.3)
    assert img.norm.vmin == 5.0 and img.norm.vmax == 60.0
    assert img.get_cmap().name.startswith("magma")
    assert img.colorbar is not None
    assert img.colorbar.ax.get_ylabel() == "Value"
    assert img.get_interpolation_stage() == "data"
    assert hshade.get_interpolation_stage() == "data"
    np.testing.assert_allclose(img.get_extent(), hshade.get_extent())
    assert ax.get_title() == "Shaded"
    plt.close(fig)


def test_create_shaded_cmap_img_map_bbox_and_no_colour_bar(dem_data):
    """The bbox is used for both layers; the colour bar is optional."""
    fig, ax = plt.subplots()
    bbox = BBOXES[0]
    img = map_raster.create_shaded_cmap_img_map(
        ax,
        dem_data["vals"],
        dem_img=dem_data["dem"],
        bbox=bbox,
        use_log_norm=True,
        vmin=1.0,
        show_colour_bar=False,
        show_scale_bar=False,
        show_north_arrow=False,
    )
    assert img.colorbar is None
    assert isinstance(img.norm, matplotlib.colors.LogNorm)
    for artist in ax.images:
        np.testing.assert_allclose(artist.get_extent(), bbox)
    plt.close(fig)


def test_raster_map_alpha_and_interpolation(dem_data):
    """The raster map functions pass img_alpha, interpolation and interp_stage to imshow."""
    fig, ax = plt.subplots()
    img = map_raster.create_raster_cmap_img_map(
        ax, dem_data["vals"], 1, img_alpha=0.5, interpolation="nearest",
        interp_stage="rgba",
    )
    assert img.get_alpha() == 0.5
    assert img.get_interpolation() == "nearest"
    assert img.get_interpolation_stage() == "rgba"
    img = map_raster.create_raster_img_map(
        ax, dem_data["vals"], [1], rsgispy.IMG_STRETCH_LINEAR, interpolation="nearest"
    )
    assert img.get_interpolation() == "nearest"
    assert img.get_interpolation_stage() == "data"
    plt.close(fig)
