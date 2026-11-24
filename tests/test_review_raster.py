"""Targeted regression tests for Workstream 2 raster/elevation/morphology/calc/spectral fixes."""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from shapely.geometry import Point, Polygon

import rsgispy
from rsgispy._io import BandMathOp, DiskSink, RasterSource, TileGraph
import rsgispy.changedetect.imgchngmthds as icm
import rsgispy.elevation.filtering as elev_filt
import rsgispy.imagecalc._norm_helpers as norm_helpers
import rsgispy.imagecalc.calc_pt_win_smpls as pt_win
import rsgispy.imagecalc.encoding as img_enc
import rsgispy.imagecalc.fill as img_fill
import rsgispy.imagecalc.leastcostpath as lcp
import rsgispy.imagecalc.percentiles as img_pct
import rsgispy.imagecalc.stats as calc_stats
import rsgispy.imagemorphology as morph
import rsgispy.imageprocess as imgproc
import rsgispy.imagespectral.endmembers as endm
import rsgispy.imagespectral.transforms as spec_tr
import rsgispy.imagespectral.unmixing as unmix
import rsgispy.imageutils.bands as img_bands
import rsgispy.imageutils.blank as img_blank
import rsgispy.imageutils.mosaic as img_mosaic
import rsgispy.imageutils.rasterise as img_ras
import rsgispy.imageutils.spatial as img_spatial
import rsgispy.imageutils.stats as img_stats
import rsgispy.imageutils.warp as img_warp
import rsgispy.segmentation.clump as img_seg


def _write_raster(
    path: Path,
    arr: np.ndarray,
    *,
    transform: Affine = from_origin(500000.0, 5800000.0, 10.0, 10.0),
    crs: object = "EPSG:32630",
    nodata=None,
) -> str:
    data = arr[np.newaxis, :, :] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[1]),
        "width": int(data.shape[2]),
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


# ---------------------------------------------------------------------------
# 1. elevation/filtering.py — resampling_detrend_dem tiled WarpedVRT
# ---------------------------------------------------------------------------
def test_resampling_detrend_dem_with_and_without_tmp_dir(tmp_path):
    y, x = np.mgrid[0:32, 0:32].astype(np.float32)
    dem = 100.0 + 2.0 * x + 1.5 * y + 5.0 * np.sin(x / 2.0)
    dem[0:4, 0:4] = -9999.0
    in_dem = _write_raster(tmp_path / "dem.tif", dem, nodata=-9999.0)

    out_mem = str(tmp_path / "detrend_mem.tif")
    elev_filt.resampling_detrend_dem(
        input_img=in_dem,
        output_img=out_mem,
        resmp_res=40.0,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )

    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir()
    out_disk = str(tmp_path / "detrend_disk.tif")
    elev_filt.resampling_detrend_dem(
        input_img=in_dem,
        output_img=out_disk,
        resmp_res=40.0,
        tmp_dir=str(tmp_dir),
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )

    with rasterio.open(out_mem) as a, rasterio.open(out_disk) as b:
        arr_a = a.read(1)
        arr_b = b.read(1)
        assert arr_a.shape == (32, 32)
        assert np.allclose(arr_a, arr_b, equal_nan=True)
        assert arr_a[0, 0] == -9999.0
        assert np.isfinite(arr_a[16, 16])


# ---------------------------------------------------------------------------
# 2. imageutils/warp.py — reproject_image preserves nodata & densified bounds
# ---------------------------------------------------------------------------
def test_reproject_image_preserves_src_nodata_and_bounds(tmp_path):
    arr = np.full((24, 24), 42.0, dtype=np.float32)
    arr[0:4, 0:4] = -9999.0
    in_img = _write_raster(
        tmp_path / "utm.tif",
        arr,
        transform=from_origin(500000.0, 5800000.0, 100.0, 100.0),
        crs="EPSG:32630",
        nodata=-9999.0,
    )
    out_img = str(tmp_path / "reproj.tif")
    img_warp.reproject_image(
        input_img=in_img,
        output_img=out_img,
        out_epsg=3857,
        out_pxl_res="auto",
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        assert dst.nodata == -9999.0
        assert dst.crs.to_epsg() == 3857
        out_arr = dst.read(1)
        assert np.any(out_arr == 42.0)
        assert np.any(out_arr == -9999.0)


# ---------------------------------------------------------------------------
# 3. imageutils/rasterise.py — int64, pd.NA skip, deterministic overlap
# ---------------------------------------------------------------------------
def test_rasterise_int64_na_and_deterministic_overlap(tmp_path):
    ref_img = _write_raster(
        tmp_path / "ref.tif",
        np.zeros((20, 20), dtype=np.int64),
        transform=from_origin(0.0, 200.0, 10.0, 10.0),
        crs="EPSG:32630",
        nodata=0,
    )
    poly_a = Polygon([(0, 0), (150, 0), (150, 150), (0, 150)])
    poly_b = Polygon([(50, 50), (200, 50), (200, 200), (50, 200)])
    poly_na = Polygon([(160, 0), (200, 0), (200, 40), (160, 40)])
    gdf = gpd.GeoDataFrame(
        {"val": pd.Series([100, 200, pd.NA], dtype="Int64")},
        geometry=[poly_a, poly_b, poly_na],
        crs="EPSG:32630",
    )
    out_img = str(tmp_path / "ras_i64.tif")
    img_ras.rasterise_vec_lyr(
        vec_file=gdf,
        vec_lyr=None,
        input_img=ref_img,
        output_img=out_img,
        att_column="val",
        datatype=rsgispy.TYPE_64INT,
        no_data_val=0,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        data = dst.read(1)
        assert data.dtype == np.int64
        # Overlap pixel (x=100, y=100 -> row=10, col=10): later feature (200) wins
        assert data[10, 10] == 200
        # Non-overlapping part of poly_a (x=25, y=25 -> row=17, col=2)
        assert data[17, 2] == 100
        # pd.NA polygon skipped -> remains 0
        assert data[18, 18] == 0


# ---------------------------------------------------------------------------
# 4. imageutils/mosaic.py — finite skip_val guards NaN & partial overlap bounds
# ---------------------------------------------------------------------------
def test_mosaic_nan_and_partial_overlap(tmp_path):
    t1 = np.full((10, 10), 5.0, dtype=np.float32)
    t1[2, 2] = np.nan
    t1[3, 3] = 0.0
    img1 = _write_raster(
        tmp_path / "t1.tif",
        t1,
        transform=from_origin(0.0, 100.0, 10.0, 10.0),
        nodata=0.0,
    )
    t2 = np.full((10, 10), 9.0, dtype=np.float32)
    t2[2, 2] = 7.0
    img2 = _write_raster(
        tmp_path / "t2.tif",
        t2,
        transform=from_origin(0.0, 100.0, 10.0, 10.0),
        nodata=0.0,
    )
    out_mos = str(tmp_path / "mos.tif")
    img_mosaic.create_img_mosaic(
        input_imgs=[img2, img1],
        output_img=out_mos,
        background_val=0.0,
        skip_val=0.0,
        overlap_behaviour=0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_mos) as dst:
        arr = dst.read(1)
        # NaN in img1 must NOT overwrite valid pixel 7.0 from img2
        assert arr[2, 2] == 7.0
        # skip_val (0.0) in img1 must NOT overwrite valid pixel 9.0 from img2
        assert arr[3, 3] == 9.0
        # Valid pixel in img1 overwrites img2
        assert arr[0, 0] == 5.0


# ---------------------------------------------------------------------------
# 5. imageutils/bands.py — sharpen_low_res_bands nodata & vectorized multi-high
# ---------------------------------------------------------------------------
def test_sharpen_low_res_bands_sets_nodata_and_multi_high(tmp_path):
    rng = np.random.default_rng(123)
    h1 = rng.uniform(10.0, 100.0, size=(20, 20)).astype(np.float32)
    h2 = rng.uniform(5.0, 50.0, size=(20, 20)).astype(np.float32)
    low = (1.5 * h1 + 0.5 * h2 + 3.0).astype(np.float32)
    low[0, 0] = -999.0
    h1[0, 0] = -999.0
    h2[0, 0] = -999.0

    in_img = _write_raster(
        tmp_path / "sharp_in.tif",
        np.stack([h1, h2, low], axis=0),
        nodata=-999.0,
    )
    out_img = str(tmp_path / "sharp_out.tif")
    band_info = [
        img_bands.SharpBandInfo(
            band=1, status=rsgispy.SHARP_RES_HIGH, name="H1"
        ),
        img_bands.SharpBandInfo(
            band=2, status=rsgispy.SHARP_RES_HIGH, name="H2"
        ),
        img_bands.SharpBandInfo(
            band=3, status=rsgispy.SHARP_RES_LOW, name="L1"
        ),
    ]
    img_bands.sharpen_low_res_bands(
        input_img=in_img,
        output_img=out_img,
        band_info=band_info,
        win_size=5,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        assert dst.nodata == -999.0
        out_low = dst.read(3)
        assert out_low[0, 0] == -999.0
        # Exact linear relation should be recovered closely on interior pixels
        assert np.allclose(out_low[5:15, 5:15], low[5:15, 5:15], atol=1e-2)


# ---------------------------------------------------------------------------
# 6. imageutils/stats.py — direct build_img_pyramids, multi_core & int64 stats
# ---------------------------------------------------------------------------
def test_build_img_pyramids_direct_and_multi_core_and_int64_thematic(tmp_path):
    # Direct test for build_img_pyramids (min_size // 4 > 33 => >= 136x136)
    arr = (np.arange(160 * 160, dtype=np.uint16).reshape(160, 160) % 250) + 1
    pyr_img = _write_raster(tmp_path / "pyr.tif", arr, nodata=0)
    img_stats.build_img_pyramids(pyr_img, resampling_method=Resampling.average)
    with rasterio.open(pyr_img) as src:
        assert 4 in src.overviews(1)

    # Small image (< 136px) should be a clean no-op
    small_img = _write_raster(
        tmp_path / "small.tif", np.ones((32, 32), dtype=np.uint8)
    )
    img_stats.build_img_pyramids(small_img)
    with rasterio.open(small_img) as src:
        assert src.overviews(1) == []

    # Multi-core pyramids across multiple levels reusing a single pool
    mc_img = _write_raster(tmp_path / "mc_pyr.tif", arr, nodata=0)
    img_stats.build_img_pyramids_multi_core(
        mc_img,
        resampling_method=Resampling.average,
        levels=[2, 4, 8],
        tile_size=64,
        n_cores=2,
        external=True,
        use_pbar=False,
    )
    assert os.path.exists(f"{mc_img}.ovr")
    with rasterio.open(mc_img) as src:
        assert src.overviews(1) == [2, 4, 8]

    # int64 thematic stats (histogram + stats tags on 64-bit integer)
    cls_arr = np.array([[0, 1, 1], [2, 2, 2], [0, 3, 3]], dtype=np.int64)
    i64_img = _write_raster(tmp_path / "cls_i64.tif", cls_arr, nodata=0)
    img_stats.pop_thmt_img_stats(
        i64_img, add_clr_tab=True, calc_pyramids=False, use_pbar=False
    )
    with rasterio.open(i64_img) as src:
        tags = src.tags(1)
        assert float(tags["STATISTICS_MINIMUM"]) == 1.0
        assert float(tags["STATISTICS_MAXIMUM"]) == 3.0


# ---------------------------------------------------------------------------
# 7. imageutils/blank.py — custom WKT CRS & rotated Affine preserved
# ---------------------------------------------------------------------------
def test_blank_copy_and_buf_preserve_non_epsg_wkt_and_rotated_transform(tmp_path):
    custom_crs = CRS.from_proj4(
        "+proj=aea +lat_0=52 +lon_0=-2 +lat_1=50 +lat_2=55 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    rot_tf = Affine(8.0, 6.0, 1000.0, 6.0, -8.0, 2000.0)
    ref_img = _write_raster(
        tmp_path / "rot_ref.tif",
        np.ones((10, 12), dtype=np.float32),
        transform=rot_tf,
        crs=custom_crs,
        nodata=-99.0,
    )

    copy_img = str(tmp_path / "rot_copy.tif")
    img_blank.create_copy_img(
        ref_img, copy_img, n_bands=1, pxl_val=3.0, use_pbar=False
    )
    with rasterio.open(copy_img) as dst:
        assert dst.transform == rot_tf
        assert dst.crs == custom_crs
        assert dst.width == 12 and dst.height == 10

    buf_img = str(tmp_path / "rot_buf.tif")
    img_blank.create_blank_buf_img_from_ref_img(
        ref_img, buf_img, buf_pxl_ext=2, use_pbar=False
    )
    with rasterio.open(buf_img) as dst:
        assert dst.width == 16 and dst.height == 14
        assert dst.crs == custom_crs
        # Origin shifted by (-2, -2) in pixel space of rotated transform
        assert dst.transform == rot_tf @ Affine.translation(-2, -2)


# ---------------------------------------------------------------------------
# 8. _io.py — multi-tile BandMathOp parser reuse across boundary tiles
# ---------------------------------------------------------------------------
def test_band_math_op_reuses_parser_across_non_square_tiles(tmp_path):
    arr = np.arange(25 * 30, dtype=np.float32).reshape(25, 30)
    in_img = _write_raster(tmp_path / "bm_in.tif", arr)
    out_img = str(tmp_path / "bm_out.tif")

    src_node = RasterSource(in_img)
    math_node = BandMathOp(
        inputs={"b1": src_node},
        expr="b1 * 2.0 + 1.0",
    )
    sink = DiskSink(
        source=math_node,
        output_img=out_img,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        nodata=None,
    )
    TileGraph(sinks=sink, block_size=16).run(use_pbar=False)

    with rasterio.open(out_img) as dst:
        np.testing.assert_allclose(dst.read(1), arr * 2.0 + 1.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# 9. imageutils/spatial.py — vectorized grid_scattered_pts
# ---------------------------------------------------------------------------
def test_grid_scattered_pts_vectorised_algorithms(tmp_path):
    ref_img = _write_raster(
        tmp_path / "grid_ref.tif",
        np.zeros((10, 10), dtype=np.float32),
        transform=from_origin(0.0, 100.0, 10.0, 10.0),
    )
    pts = [
        Point(15.0, 85.0),
        Point(85.0, 85.0),
        Point(15.0, 15.0),
        Point(85.0, 15.0),
        Point(50.0, 50.0),
    ]
    gdf = gpd.GeoDataFrame(
        {"z": [10.0, 20.0, 30.0, 40.0, 25.0]}, geometry=pts, crs="EPSG:32630"
    )
    for alg in (
        "nearest",
        "invdist:power=2.0",
        "invdistnn:power=2.0:radius=60.0:max_points=4",
        "average:radius1=60.0:radius2=60.0",
        "count:radius1=60.0:radius2=60.0",
    ):
        out_img = str(tmp_path / f"grid_{alg.split(':')[0]}.tif")
        img_spatial.grid_scattered_pts(
            vec_file=gdf,
            vec_lyr=None,
            vec_col="z",
            input_img=ref_img,
            output_img=out_img,
            gdal_grid_alg=alg,
            no_data_val=-999.0,
            use_pbar=False,
        )
        with rasterio.open(out_img) as dst:
            arr = dst.read(1)
            assert arr.shape == (10, 10)
            if alg == "nearest":
                assert arr[1, 1] == pytest.approx(10.0)
                assert arr[8, 8] == pytest.approx(40.0)
            elif alg.startswith("count"):
                assert arr[5, 5] >= 1.0


# ---------------------------------------------------------------------------
# 10. imagemorphology — edge padding & nodata exclusion
# ---------------------------------------------------------------------------
def test_morphology_edge_padding_and_nodata_no_bleed(tmp_path):
    arr = np.full((12, 12), 50.0, dtype=np.float32)
    arr[5, 5] = -9999.0  # nodata hole
    arr[6, 6] = np.nan
    in_img = _write_raster(tmp_path / "morph_in.tif", arr, nodata=-9999.0)

    op_file = str(tmp_path / "op3.gmtxt")
    morph.create_circular_op(op_file, 3)

    out_erode = str(tmp_path / "erode.tif")
    morph.image_erode(
        in_img,
        out_erode,
        op_file,
        use_op_file=True,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )
    out_dilate = str(tmp_path / "dilate.tif")
    morph.image_dilate(
        in_img,
        out_dilate,
        op_file,
        use_op_file=True,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )
    out_open = str(tmp_path / "open.tif")
    morph.image_opening(
        in_img,
        out_open,
        op_file,
        use_op_file=True,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )

    for out_p in (out_erode, out_dilate, out_open):
        with rasterio.open(out_p) as dst:
            res = dst.read(1)
            # Border pixels must stay 50.0 (no 0-padding bleed)
            assert res[0, 0] == 50.0
            assert res[-1, -1] == 50.0
            # Neighbours of nodata/NaN hole must stay 50.0 (no -9999 or NaN bleed)
            assert res[5, 4] == 50.0
            assert res[4, 5] == 50.0
            assert res[6, 7] == 50.0


# ---------------------------------------------------------------------------
# 11. imagecalc/fill.py — calc_fill_regions_knn with NaN fill_region_val & clamp
# ---------------------------------------------------------------------------
def test_calc_fill_regions_knn_nan_fill_val_and_int_clamp(tmp_path):
    ref_arr = np.full((8, 8), 300.0, dtype=np.float32)  # exceeds uint8 max 255
    ref_arr[0, 0] = -1.0
    mask_arr = np.ones((8, 8), dtype=np.float32)
    mask_arr[3:5, 3:5] = np.nan

    ref_img = _write_raster(tmp_path / "knn_ref.tif", ref_arr, nodata=-1.0)
    msk_img = _write_raster(tmp_path / "knn_msk.tif", mask_arr)
    out_img = str(tmp_path / "knn_out.tif")

    img_fill.calc_fill_regions_knn(
        in_ref_img=ref_img,
        in_fill_regions_img=msk_img,
        output_img=out_img,
        ref_no_data=-1.0,
        fill_region_val=float("nan"),
        k=3,
        summary=rsgispy.SUMTYPE_MEAN,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out_img) as dst:
        out_arr = dst.read(1)
        assert out_arr.dtype == np.uint8
        # Filled NaN region clamped safely to 255 rather than wrapping mod 256
        assert np.all(out_arr[3:5, 3:5] == 255)
        assert out_arr[0, 0] == 0


# ---------------------------------------------------------------------------
# 12. imagecalc/percentiles.py — NaN/inf filtering in calc_band_percentile_msk
# ---------------------------------------------------------------------------
def test_calc_band_percentile_msk_filters_non_finite(tmp_path):
    arr = np.array(
        [[10.0, 20.0, np.nan, np.inf], [30.0, 40.0, -np.inf, 50.0]],
        dtype=np.float32,
    )
    msk = np.ones((2, 4), dtype=np.uint8)
    in_img = _write_raster(tmp_path / "pct_in.tif", arr)
    msk_img = _write_raster(tmp_path / "pct_msk.tif", msk)

    pcts = img_pct.calc_band_percentile_msk(
        input_img=in_img,
        in_msk_img=msk_img,
        msk_val=1,
        percentiles=[0.0, 50.0, 100.0],
        no_data_val=float("nan"),
        use_pbar=False,
    )
    np.testing.assert_allclose(pcts[:, 0], [10.0, 30.0, 50.0])


# ---------------------------------------------------------------------------
# 13. imagecalc/leastcostpath.py — NaN / nodata treated as impassable barriers
# ---------------------------------------------------------------------------
def test_least_cost_path_avoids_nan_and_nodata_barriers(tmp_path):
    cost = np.ones((7, 7), dtype=np.float32)
    # Build vertical wall at col=3 except row=6, mixing NaN and nodata (-999)
    cost[0:3, 3] = np.nan
    cost[3:6, 3] = -999.0
    tf = from_origin(0.0, 70.0, 10.0, 10.0)
    cost_img = _write_raster(
        tmp_path / "cost.tif", cost, transform=tf, nodata=-999.0
    )
    out_img = str(tmp_path / "lcp.tif")

    # Start (row=1, col=1) -> (15, 55), Stop (row=1, col=5) -> (55, 55)
    lcp.perform_least_cost_path_calc(
        cost_surface_img=cost_img,
        output_img=out_img,
        start_coord=(15.0, 55.0),
        stop_coord=(55.0, 55.0),
    )
    with rasterio.open(out_img) as dst:
        path_arr = dst.read(1)
        assert path_arr[1, 1] == 1
        assert path_arr[1, 5] == 1
        # Must not cross any barrier cell, and must pass through gap at (6, 3)
        assert not np.any(path_arr[0:6, 3] == 1)
        assert path_arr[6, 3] == 1

    # Start inside nodata cell must raise RSGISPyException
    with pytest.raises(rsgispy.RSGISPyException):
        lcp.perform_least_cost_path_calc(
            cost_surface_img=cost_img,
            output_img=str(tmp_path / "lcp_bad.tif"),
            start_coord=(35.0, 35.0),  # (row=3, col=3) is -999
            stop_coord=(55.0, 55.0),
        )


# ---------------------------------------------------------------------------
# 14. imagecalc/calc_pt_win_smpls.py — rotation around actual pixel center
# ---------------------------------------------------------------------------
def test_extract_rotated_window_near_edge_pivots_on_point(tmp_path):
    arr = np.zeros((20, 20), dtype=np.float32)
    # Put a bright impulse exactly at pixel (row=2, col=3) -> x=35, y=175
    arr[2, 3] = 100.0
    tf = from_origin(0.0, 200.0, 10.0, 10.0)
    in_img = _write_raster(tmp_path / "rot_pt.tif", arr, transform=tf)

    with rasterio.open(in_img) as src:
        img_bbox = (
            float(src.bounds.left),
            float(src.bounds.right),
            float(src.bounds.bottom),
            float(src.bounds.top),
        )
        # Near top-left edge where sub_win gets clipped at row=0, col=0
        roi, roi_tf, _, _ = pt_win._extract_rotated_window(  # pylint: disable=protected-access
            src_ds=src,
            img_bbox=img_bbox,
            pt_xy=(35.0, 175.0),
            pt_xbox=4.0,
            pt_ybox=4.0,
            pt_angle=90.0,
            rio_interp=Resampling.nearest,
            no_data_val=0.0,
        )
    # The point itself must stay invariant under rotation even when clipped at edge
    pt_c, pt_r = ~roi_tf @ (35.0, 175.0)
    assert roi[0, int(pt_r), int(pt_c)] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 15. imagecalc/encoding.py — recode_int_raster validates/remaps dst_nodata
# ---------------------------------------------------------------------------
def test_recode_int_raster_remaps_and_validates_nodata(tmp_path):
    arr = np.array([[-9999, 1, 2], [3, -9999, 1]], dtype=np.int32)
    in_img = _write_raster(tmp_path / "enc_in.tif", arr, nodata=-9999)

    # Case 1: nodata remapped via recode_dict into valid uint8 range
    out1 = str(tmp_path / "enc_out1.tif")
    img_enc.recode_int_raster(
        input_img=in_img,
        output_img=out1,
        recode_dict={-9999: 255, 1: 10, 2: 20},
        keep_vals_not_in_dict=False,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out1) as dst:
        assert dst.nodata == 255.0
        assert dst.read(1)[0, 0] == 255

    # Case 2: nodata not in dict, keep_vals_not_in_dict=True, out of uint8 range
    out2 = str(tmp_path / "enc_out2.tif")
    img_enc.recode_int_raster(
        input_img=in_img,
        output_img=out2,
        recode_dict={1: 10, 2: 20},
        keep_vals_not_in_dict=True,
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(out2) as dst:
        assert dst.nodata is None


# ---------------------------------------------------------------------------
# 16. imagecalc/stats.py — count_pxls_of_val reads only requested band
# ---------------------------------------------------------------------------
def test_count_pxls_of_val_single_band(tmp_path):
    b1 = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
    b2 = np.array([[3, 3, 3], [3, 1, 1]], dtype=np.uint8)
    in_img = _write_raster(tmp_path / "cnt.tif", np.stack([b1, b2], axis=0))

    c1 = calc_stats.count_pxls_of_val(in_img, [1, 2, 3], img_band=1, use_pbar=False)
    c2 = calc_stats.count_pxls_of_val(in_img, [1, 2, 3], img_band=2, use_pbar=False)
    call = calc_stats.count_pxls_of_val(
        in_img, [1, 2, 3], img_band=None, use_pbar=False
    )
    np.testing.assert_array_equal(c1, [2, 2, 2])
    np.testing.assert_array_equal(c2, [2, 0, 4])
    np.testing.assert_array_equal(call, [4, 2, 6])


# ---------------------------------------------------------------------------
# 17. imagecalc/_norm_helpers.py — tiled reservoir sampling in subsample_band_values
# ---------------------------------------------------------------------------
def test_subsample_band_values_tiled_reservoir(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    arr = np.arange(1, 20 * 20 + 1, dtype=np.float32).reshape(20, 20)
    arr[0, 0] = -999.0
    arr[1, 1] = np.nan
    in_img = _write_raster(tmp_path / "res.tif", arr, nodata=-999.0)

    with rasterio.open(in_img) as src:
        s1 = norm_helpers.subsample_band_values(
            src, band_idx=1, eff_nd=-999.0, max_samples=50, rnd_seed=42
        )
        s2 = norm_helpers.subsample_band_values(
            src, band_idx=1, eff_nd=-999.0, max_samples=50, rnd_seed=42
        )
        assert s1.shape == (50,)
        np.testing.assert_array_equal(s1, s2)
        assert np.all(np.isfinite(s1))
        assert not np.any(s1 == -999.0)
        # Samples should span across multiple tiles (not just first 50 pixels)
        assert float(np.max(s1)) > 100.0


# ---------------------------------------------------------------------------
# 18. imagespectral/endmembers.py — tiled streaming calc_ppi
# ---------------------------------------------------------------------------
def test_calc_ppi_tiled_streaming_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(7)
    cube = rng.uniform(0.2, 0.8, size=(3, 16, 16)).astype(np.float32)
    # Plant extreme pure endmembers at opposite corners
    cube[:, 0, 0] = [1.0, 0.0, 0.0]
    cube[:, 15, 15] = [0.0, 1.0, 0.0]
    in_img = _write_raster(tmp_path / "ppi_in.tif", cube)

    out1 = str(tmp_path / "ppi1.tif")
    out2 = str(tmp_path / "ppi2.tif")
    endm.calc_ppi(
        in_img, out1, niters=200, rnd_seed=99, calc_stats=False, use_pbar=False
    )
    endm.calc_ppi(
        in_img, out2, niters=200, rnd_seed=99, calc_stats=False, use_pbar=False
    )
    with rasterio.open(out1) as d1, rasterio.open(out2) as d2:
        a1 = d1.read(1)
        a2 = d2.read(1)
        assert a1.dtype == np.uint16
        np.testing.assert_array_equal(a1, a2)
        assert a1[0, 0] > 0
        assert a1[15, 15] > 0


# ---------------------------------------------------------------------------
# 19. imagespectral/transforms.py — whiten_image filters NaN/nodata & streams
# ---------------------------------------------------------------------------
def test_whiten_image_filters_nan_and_nodata(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(11)
    n = 16 * 16
    base = rng.normal(size=(n, 3))
    mix = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, -0.4], [0.3, -0.4, 1.2]])
    cube = (base @ mix + np.array([10.0, 20.0, 30.0])).T.reshape(3, 16, 16)
    cube = cube.astype(np.float32)
    cube[:, 0, 0] = np.nan
    cube[:, 0, 1] = -9999.0

    msk = np.ones((16, 16), dtype=np.uint8)
    msk[15, 15] = 0

    in_img = _write_raster(tmp_path / "w_in.tif", cube, nodata=-9999.0)
    msk_img = _write_raster(tmp_path / "w_msk.tif", msk)
    out_img = str(tmp_path / "w_out.tif")

    spec_tr.whiten_image(in_img, out_img, msk_img, 1, use_pbar=False)
    with rasterio.open(out_img) as dst:
        out = dst.read()
        assert np.all(out[:, 0, 0] == 0.0)
        assert np.all(out[:, 0, 1] == 0.0)
        assert np.all(out[:, 15, 15] == 0.0)
        valid = np.ones((16, 16), dtype=bool)
        valid[0, 0] = False
        valid[0, 1] = False
        valid[15, 15] = False
        vecs = out[:, valid]
        cov = np.cov(vecs)
        np.testing.assert_allclose(cov, np.eye(3), atol=5e-2)


# ---------------------------------------------------------------------------
# 20. imagespectral/unmixing.py — aligned fill value and no_data_val (999.0)
# ---------------------------------------------------------------------------
def test_summarise_multi_endmember_unmixing_nodata_alignment(tmp_path):
    em_mat = np.array([[0.8, 0.2, 0.1], [0.1, 0.7, 0.9]], dtype=np.float32)
    em_df = pd.DataFrame(em_mat.T, columns=["EM_A", "EM_B"])

    # Pixel (0,0) is all-zero (invalid); remaining pixels are valid combinations
    orig = np.zeros((3, 4, 4), dtype=np.float32)
    for b in range(3):
        orig[b, :, :] = 0.6 * em_mat[0, b] + 0.4 * em_mat[1, b]
    orig[:, 0, 0] = 0.0

    abund = np.zeros((2, 4, 4), dtype=np.float32)
    abund[0, :, :] = 0.6
    abund[1, :, :] = 0.4

    orig_img = _write_raster(tmp_path / "orig.tif", orig)
    abund_img = _write_raster(tmp_path / "abund1.tif", abund)
    out_unmix = str(tmp_path / "sum_unmix.tif")
    out_ref = str(tmp_path / "sum_ref.tif")

    ds_info = endm.ImageEndmemberInfo(
        in_unmix_img=abund_img,
        endmembers=em_df,
        endmember_names=["EM_A", "EM_B"],
    )
    unmix.summarise_multi_endmember_linear_unmixing(
        input_img=orig_img,
        out_unmix_img=out_unmix,
        out_ref_img=out_ref,
        in_unmixed_datasets=[ds_info],
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_unmix) as dst_u, rasterio.open(out_ref) as dst_r:
        assert dst_u.nodata == 999.0
        assert dst_r.nodata == 0.0
        u_arr = dst_u.read()
        r_arr = dst_r.read(1)
        assert np.all(u_arr[:, 0, 0] == 999.0)
        assert r_arr[0, 0] == 0
        assert np.allclose(u_arr[:, 1, 1], [0.6, 0.4], atol=1e-4)
        assert r_arr[1, 1] == 1


# ---------------------------------------------------------------------------
# 21. changedetect/imgchngmthds.py — streaming diff/ratio & binary nodata=255
# ---------------------------------------------------------------------------
def test_changedetect_streaming_and_binary_nodata_255(tmp_path, monkeypatch):
    monkeypatch.setenv(imgproc.ENV_VAR_BLOCK_SIZE, "8")
    rng = np.random.default_rng(19)
    base = rng.uniform(10.0, 30.0, size=(3, 16, 16)).astype(np.float32)
    chng = base + rng.normal(0.0, 0.5, size=(3, 16, 16)).astype(np.float32)
    chng[:, 8:14, 8:14] += 15.0  # clear change patch
    msk = np.ones((16, 16), dtype=np.uint8)
    msk[0:2, 0:2] = 0  # masked-out corner

    base_img = _write_raster(tmp_path / "cd_base.tif", base)
    chng_img = _write_raster(tmp_path / "cd_chng.tif", chng)
    msk_img = _write_raster(tmp_path / "cd_msk.tif", msk)

    out_diff = str(tmp_path / "diff.tif")
    icm.image_difference(
        base_img,
        chng_img,
        out_diff,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        apply_std=True,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_diff) as dst:
        assert dst.nodata == -999.0
        d_arr = dst.read()
        assert np.all(d_arr[:, 0:2, 0:2] == -999.0)
        assert np.all(np.isfinite(d_arr))

    out_ratio = str(tmp_path / "ratio.tif")
    icm.image_ratio(
        base_img,
        chng_img,
        out_ratio,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        apply_std=False,
        no_data_val=-999.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_ratio) as dst:
        r_arr = dst.read()
        assert np.all(r_arr[:, 0:2, 0:2] == -999.0)
        np.testing.assert_allclose(
            r_arr[:, 5, 5], base[:, 5, 5] / chng[:, 5, 5], rtol=1e-5
        )

    cva_bin = str(tmp_path / "cva_bin.tif")
    cva_dist = str(tmp_path / "cva_dist.tif")
    icm.change_vector_analysis(
        base_img,
        chng_img,
        cva_bin,
        cva_dist,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        calc_stats=False,
        use_pbar=False,
    )
    sfa_bin = str(tmp_path / "sfa_bin.tif")
    sfa_dist = str(tmp_path / "sfa_dist.tif")
    icm.slow_feature_analysis(
        base_img,
        chng_img,
        sfa_bin,
        sfa_dist,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        sfa_max_iter=5,
        calc_stats=False,
        use_pbar=False,
    )
    mad_bin = str(tmp_path / "mad_bin.tif")
    mad_dist = str(tmp_path / "mad_dist.tif")
    mad_vars = str(tmp_path / "mad_vars.tif")
    icm.multivariate_alteration_detection(
        base_img,
        chng_img,
        mad_bin,
        mad_dist,
        mad_vars,
        vld_msk_img=msk_img,
        vld_msk_val=1,
        mad_max_iter=5,
        use_kmeans=False,
        calc_stats=False,
        use_pbar=False,
    )
    for b_path in (cva_bin, sfa_bin, mad_bin):
        with rasterio.open(b_path) as dst:
            b_arr = dst.read(1)
            assert dst.nodata == 255.0
            assert np.all(b_arr[0:2, 0:2] == 255)
            assert set(np.unique(b_arr[msk == 1]).tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 22. segmentation/_helpers.py — integer clump ID overflow raises exception
# ---------------------------------------------------------------------------
def test_segmentation_clump_id_overflow_raises(tmp_path):
    # 20x20 grid with unique pixel value per cell => 400 clumps (> uint8 max 255)
    vals = np.arange(1, 401, dtype=np.uint16).reshape(20, 20)
    in_img = _write_raster(tmp_path / "clump_in.tif", vals, nodata=0)

    # In-memory overflow
    with pytest.raises(
        rsgispy.RSGISPyException, match="exceeds maximum representable value"
    ):
        img_seg.clump(
            input_img=in_img,
            output_img=str(tmp_path / "clump_mem.tif"),
            in_memory=True,
            datatype=rsgispy.TYPE_8UINT,
            calc_stats=False,
            use_pbar=False,
        )

    # Tiled overflow (block_size=8 forces multi-tile path)
    with pytest.raises(
        rsgispy.RSGISPyException, match="exceeds maximum representable value"
    ):
        img_seg.clump(
            input_img=in_img,
            output_img=str(tmp_path / "clump_tiled.tif"),
            in_memory=False,
            block_size=8,
            datatype=rsgispy.TYPE_8UINT,
            calc_stats=False,
            use_pbar=False,
        )
