"""
Tests that the dataset-iterating functions default to ``use_pbar=True`` and
honour ``use_pbar=False``.
"""

import numpy as np
import rasterio
import rasterio.crs
from rasterio.transform import from_origin

import rsgispy.imageutils.warp as iu_warp


def test_progress_bar_defaults_and_suppression(tmp_path, monkeypatch):
    """Verify dataset-iterating functions default to use_pbar=True and honour use_pbar=False."""
    import geopandas as gpd
    from shapely.geometry import Point, box

    import rsgispy
    import rsgispy.elevation.terrain as el_terrain
    import rsgispy.imagecalc.maths as ic_maths
    import rsgispy.imagemorphology as imorph
    import rsgispy.imageutils.masking as iu_mask
    import rsgispy.imageutils.spatial as iu_spatial
    import rsgispy.imageutils.subsetting as iu_sub
    import rsgispy.tools.checkdatasets as chk_ds
    import rsgispy.zonalstats.point as zs_point

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

    pts_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(500015.0, 5199985.0)],
        crs="EPSG:32630",
    )
    cut_geom = box(500010.0, 5199950.0, 500050.0, 5199990.0)
    bdef = [ic_maths.BandDefn("b1", str(in_p), 1)]

    # 1. gdal_translate & cutline gdal_warp default True vs False vs -q
    events.clear()
    iu_warp.gdal_translate(str(in_p), str(tmp_path / "t_on.tif"))
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_on.tif"),
        out_epsg=3857,
        cutline=cut_geom,
        cutline_srs=32630,
    )
    assert len(events) >= 2 and all(d is False for d in events)

    events.clear()
    iu_warp.gdal_translate(
        str(in_p), str(tmp_path / "t_off.tif"), use_pbar=False
    )
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_off.tif"),
        out_epsg=3857,
        cutline=cut_geom,
        cutline_srs=32630,
        use_pbar=False,
    )
    assert len(events) >= 2 and all(d is True for d in events)

    events.clear()
    iu_warp.gdal_translate(
        str(in_p), str(tmp_path / "t_quiet.tif"), options="-q"
    )
    iu_warp.gdal_warp(
        str(in_p),
        str(tmp_path / "w_quiet.tif"),
        options="-t_srs EPSG:3857 -q",
        cutline=cut_geom,
        cutline_srs=32630,
    )
    assert len(events) >= 2 and all(d is True for d in events)

    # 2. masking, subsetting, spatial, imagecalc, elevation, imagemorphology,
    #    zonalstats, checkdatasets
    events.clear()
    iu_mask.mask_img(
        str(in_p),
        str(in_p),
        str(tmp_path / "m_on.tif"),
        1,
        0.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    iu_sub.subset_bbox(
        str(in_p),
        str(tmp_path / "s_on.tif"),
        500000.0,
        500040.0,
        5199960.0,
        5200000.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    iu_spatial.calc_pixel_locations(
        str(in_p), str(tmp_path / "loc_on.tif"), gdalformat="GTIFF"
    )
    ic_maths.band_maths(
        str(tmp_path / "bm_on.tif"),
        "b1 * 2.0",
        bdef,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    el_terrain.slope(str(in_p), str(tmp_path / "sl_on.tif"), gdalformat="GTIFF")
    imorph.image_dilate(
        str(in_p),
        str(tmp_path / "dil_on.tif"),
        "",
        False,
        3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    zs_point.ext_point_band_values(
        pts_gdf.copy(), str(in_p), 1, 0.0, 100.0, -999.0, "val"
    )
    chk_ds.cmp_to_ref_imgs([str(in_p)], str(tmp_path), "tif")
    assert len(events) >= 8 and all(d is False for d in events)

    events.clear()
    iu_mask.mask_img(
        str(in_p),
        str(in_p),
        str(tmp_path / "m_off.tif"),
        1,
        0.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    iu_sub.subset_bbox(
        str(in_p),
        str(tmp_path / "s_off.tif"),
        500000.0,
        500040.0,
        5199960.0,
        5200000.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    iu_spatial.calc_pixel_locations(
        str(in_p), str(tmp_path / "loc_off.tif"), gdalformat="GTIFF", use_pbar=False
    )
    ic_maths.band_maths(
        str(tmp_path / "bm_off.tif"),
        "b1 * 2.0",
        bdef,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    el_terrain.slope(
        str(in_p), str(tmp_path / "sl_off.tif"), gdalformat="GTIFF", use_pbar=False
    )
    imorph.image_dilate(
        str(in_p),
        str(tmp_path / "dil_off.tif"),
        "",
        False,
        3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    zs_point.ext_point_band_values(
        pts_gdf.copy(), str(in_p), 1, 0.0, 100.0, -999.0, "val", use_pbar=False
    )
    chk_ds.cmp_to_ref_imgs([str(in_p)], str(tmp_path), "tif", use_pbar=False)
    assert len(events) >= 8 and all(d is True for d in events)
