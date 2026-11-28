"""
Tests that the dataset-iterating functions default to ``use_pbar=True`` and
honour ``use_pbar=False``.
"""

import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy.imageutils.warp as iu_warp


@pytest.fixture
def pbar_events(monkeypatch):
    """Record the ``disable`` flag of every progress bar rsgispy creates."""
    import rsgispy

    events = []

    class _TrackingTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            events.append(bool(kwargs.get("disable", False)))
            self._iter = iter(iterable) if iterable is not None else iter(())

        def __iter__(self):
            return self._iter

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            pass

    monkeypatch.setattr(rsgispy, "get_tqdm", lambda: _TrackingTqdm)
    return events


@pytest.fixture
def pbar_src_img(tmp_path):
    """A 6 x 6 float32 UTM raster."""
    in_p = tmp_path / "src.tif"
    arr = np.arange(1, 37, dtype=np.float32).reshape((1, 6, 6))
    with rasterio.open(
        in_p,
        "w",
        driver="GTiff",
        width=6,
        height=6,
        count=1,
        dtype="float32",
        crs="EPSG:32630",
        transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
    ) as ds:
        ds.write(arr)
    return in_p


CUT_GEOM = box(500010.0, 5199950.0, 500050.0, 5199990.0)


def test_gdal_translate_and_warp_progress_bar_on_by_default(tmp_path, pbar_events, pbar_src_img):
    """gdal_translate and cutline gdal_warp show a progress bar by default."""
    in_p = pbar_src_img
    iu_warp.gdal_translate(str(in_p), str(tmp_path / "t_on.tif"))
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_on.tif"),
        out_epsg=3857,
        cutline=CUT_GEOM,
        cutline_srs=32630,
    )
    assert len(pbar_events) >= 2 and all(d is False for d in pbar_events)


def test_gdal_translate_and_warp_use_pbar_false(tmp_path, pbar_events, pbar_src_img):
    """gdal_translate and cutline gdal_warp honour use_pbar=False."""
    in_p = pbar_src_img
    iu_warp.gdal_translate(
        str(in_p), str(tmp_path / "t_off.tif"), use_pbar=False
    )
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_off.tif"),
        out_epsg=3857,
        cutline=CUT_GEOM,
        cutline_srs=32630,
        use_pbar=False,
    )
    assert len(pbar_events) >= 2 and all(d is True for d in pbar_events)


def test_gdal_translate_and_warp_quiet_option_disables_pbar(tmp_path, pbar_events, pbar_src_img):
    """The GDAL -q option disables the gdal_translate / gdal_warp progress bar."""
    in_p = pbar_src_img
    iu_warp.gdal_translate(
        str(in_p), str(tmp_path / "t_quiet.tif"), options="-q"
    )
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_quiet.tif"),
        options="-t_srs EPSG:3857 -q",
        cutline=CUT_GEOM,
        cutline_srs=32630,
    )
    assert len(pbar_events) >= 2 and all(d is True for d in pbar_events)


def _run_dataset_functions(in_p, tmp_path, suffix, **pbar_kw):
    """Run masking, subsetting, spatial, imagecalc, elevation, imagemorphology,
    zonalstats and checkdatasets functions, passing ``pbar_kw`` to each."""
    import geopandas as gpd
    from shapely.geometry import Point

    import rsgispy
    import rsgispy.elevation.terrain as el_terrain
    import rsgispy.imagecalc.maths as ic_maths
    import rsgispy.imagemorphology as imorph
    import rsgispy.imageutils.masking as iu_mask
    import rsgispy.imageutils.spatial as iu_spatial
    import rsgispy.imageutils.subsetting as iu_sub
    import rsgispy.tools.checkdatasets as chk_ds
    import rsgispy.zonalstats.point as zs_point

    pts_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(500015.0, 5199985.0)],
        crs="EPSG:32630",
    )
    bdef = [ic_maths.BandDefn("b1", str(in_p), 1)]
    iu_mask.mask_img(
        str(in_p),
        str(in_p),
        str(tmp_path / f"m_{suffix}.tif"),
        1,
        0.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        **pbar_kw,
    )
    iu_sub.subset_bbox(
        str(in_p),
        str(tmp_path / f"s_{suffix}.tif"),
        500000.0,
        500040.0,
        5199960.0,
        5200000.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        **pbar_kw,
    )
    iu_spatial.calc_pixel_locations(
        str(in_p), str(tmp_path / f"loc_{suffix}.tif"), gdalformat="GTIFF", **pbar_kw
    )
    ic_maths.band_maths(
        str(tmp_path / f"bm_{suffix}.tif"),
        "b1 * 2.0",
        bdef,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        **pbar_kw,
    )
    el_terrain.slope(
        str(in_p), str(tmp_path / f"sl_{suffix}.tif"), gdalformat="GTIFF", **pbar_kw
    )
    imorph.image_dilate(
        str(in_p),
        str(tmp_path / f"dil_{suffix}.tif"),
        "",
        False,
        3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        **pbar_kw,
    )
    zs_point.ext_point_band_values(
        pts_gdf.copy(), str(in_p), 1, 0.0, 100.0, -999.0, "val", **pbar_kw
    )
    chk_ds.cmp_to_ref_imgs([str(in_p)], str(tmp_path), "tif", **pbar_kw)


def test_dataset_functions_progress_bar_on_by_default(tmp_path, pbar_events, pbar_src_img):
    """masking, subsetting, spatial, imagecalc, elevation, imagemorphology,
    zonalstats and checkdatasets functions show a progress bar by default."""
    _run_dataset_functions(pbar_src_img, tmp_path, "on")
    assert len(pbar_events) >= 8 and all(d is False for d in pbar_events)


def test_dataset_functions_use_pbar_false(tmp_path, pbar_events, pbar_src_img):
    """The same functions honour use_pbar=False."""
    _run_dataset_functions(pbar_src_img, tmp_path, "off", use_pbar=False)
    assert len(pbar_events) >= 8 and all(d is True for d in pbar_events)
