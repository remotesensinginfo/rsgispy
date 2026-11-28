"""
Tests for :mod:`rsgispy.imageutils.warp`: ``gdal_translate``, ``gdal_warp``,
``resample_img_to_match`` and ``reproject_image``.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import box

import rsgispy
import rsgispy.imageutils.warp as iu_warp

from conftest import write_raster


def test_gdal_translate_bands_and_windows(tmp_path):
    """Test gdal_translate band selection, srcwin, projwin, projwin_srs, epo/eco."""
    arr = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    in_p = tmp_path / "win_in.tif"
    write_raster(in_p, arr)

    b_out = tmp_path / "bands_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(b_out),
        bands=[3, 1],
        src_win=(2, 1, 4, 3),
        use_pbar=False,
    )
    with rasterio.open(b_out) as dst:
        assert dst.count == 2
        assert dst.width == 4 and dst.height == 3
        assert np.array_equal(dst.read(1), arr[2, 1:4, 2:6])
        assert np.array_equal(dst.read(2), arr[0, 1:4, 2:6])

    pw_out = tmp_path / "projwin_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(pw_out),
        proj_win=(500020.0, 5199990.0, 500060.0, 5199960.0),
        use_pbar=False,
    )
    with rasterio.open(pw_out) as dst:
        assert dst.width == 4 and dst.height == 3
        assert np.array_equal(dst.read(), arr[:, 1:4, 2:6])

    # projwin_srs in EPSG:4326 covering the full image extent
    wgs_bounds = transform_bounds(
        "EPSG:32630", "EPSG:4326", 500000.0, 5199920.0, 500080.0, 5200000.0
    )
    pws_out = tmp_path / "projwin_srs_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(pws_out),
        proj_win=(wgs_bounds[0], wgs_bounds[3], wgs_bounds[2], wgs_bounds[1]),
        proj_win_srs=4326,
        use_pbar=False,
    )
    with rasterio.open(pws_out) as dst:
        assert dst.width == 8 and dst.height == 8

    # Partial outside clips unless error_partially_outside=True
    clip_out = tmp_path / "clip_out.tif"
    iu_warp.gdal_translate(
        str(in_p), str(clip_out), src_win=(-2, -1, 4, 3), use_pbar=False
    )
    with rasterio.open(clip_out) as dst:
        assert dst.width == 2 and dst.height == 2

    with pytest.raises(rsgispy.RSGISPyException):
        iu_warp.gdal_translate(
            str(in_p),
            str(tmp_path / "epo.tif"),
            src_win=(-1, 0, 4, 4),
            error_partially_outside=True,
            use_pbar=False,
        )

    with pytest.raises(rsgispy.RSGISPyException):
        iu_warp.gdal_translate(
            str(in_p),
            str(tmp_path / "eco.tif"),
            src_win=(20, 20, 4, 4),
            error_completely_outside=True,
            use_pbar=False,
        )


def test_gdal_translate_resize_resample_and_ovr(tmp_path):
    """Test out_size, target_res, resample_alg, and ovr options."""
    arr = np.arange(16 * 8, dtype=np.float32).reshape(1, 8, 16)
    in_p = tmp_path / "res_in.tif"
    write_raster(in_p, arr)
    with rasterio.open(in_p, "r+") as ds:
        ds.build_overviews([2], Resampling.average)

    pct_out = tmp_path / "pct_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(pct_out),
        out_size=("50%", "50%"),
        resample_alg="bilinear",
        use_pbar=False,
    )
    with rasterio.open(pct_out) as dst:
        assert dst.width == 8 and dst.height == 4
        assert dst.transform.a == pytest.approx(20.0)

    aspect_out = tmp_path / "aspect_out.tif"
    iu_warp.gdal_translate(
        str(in_p), str(aspect_out), out_size=(4, 0), use_pbar=False
    )
    with rasterio.open(aspect_out) as dst:
        assert dst.width == 4 and dst.height == 2

    tr_out = tmp_path / "tr_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(tr_out),
        target_res=(20.0, 20.0),
        resample_alg=Resampling.average,
        use_pbar=False,
    )
    with rasterio.open(tr_out) as dst:
        assert dst.width == 8 and dst.height == 4
        assert abs(dst.transform.e) == pytest.approx(20.0)

    ovr_out = tmp_path / "ovr_out.tif"
    iu_warp.gdal_translate(str(in_p), str(ovr_out), ovr=0, use_pbar=False)
    with rasterio.open(ovr_out) as dst:
        assert dst.width == 8 and dst.height == 4


def test_gdal_translate_scale_unscale_and_expand(tmp_path):
    """Test unscale, scale_params, exponent, expand colour table, and strict."""
    dn = np.array(
        [
            [[0, 50, 100, 255], [10, 20, 30, 40]],
            [[2, 4, 6, 8], [10, 12, 14, 16]],
        ],
        dtype=np.uint8,
    )
    in_p = tmp_path / "scale_in.tif"
    write_raster(in_p, dn, nodata=255)
    with rasterio.open(in_p, "r+") as ds:
        ds.scales = (0.1, 0.5)
        ds.offsets = (10.0, -1.0)

    unsc_out = tmp_path / "unscale_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(unsc_out),
        unscale=True,
        no_data_val=-999.0,
        use_pbar=False,
    )
    with rasterio.open(unsc_out) as dst:
        assert dst.dtypes == ("float64", "float64")
        u_arr = dst.read()
        assert u_arr[0, 0, 1] == pytest.approx(15.0)
        assert u_arr[0, 0, 2] == pytest.approx(20.0)
        assert u_arr[0, 0, 3] == pytest.approx(-999.0)
        assert u_arr[1, 0, 0] == pytest.approx(0.0)

    sc_out = tmp_path / "scaled_u8.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(sc_out),
        bands=[1],
        datatype=rsgispy.TYPE_8UINT,
        scale_params=[0.0, 100.0, 0.0, 200.0],
        exponent=2.0,
        no_data_val=255,
        use_pbar=False,
    )
    with rasterio.open(sc_out) as dst:
        s_arr = dst.read(1)
        assert s_arr[0, 0] == 0
        # 50/100 = 0.5 -> **2 = 0.25 -> *200 = 50
        assert s_arr[0, 1] == 50
        assert s_arr[0, 2] == 200
        assert s_arr[0, 3] == 255

    # Auto min/max per-band CLI scaling
    cli_sc_out = tmp_path / "cli_scaled.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(cli_sc_out),
        options="-scale_1 -scale_2 2 16 0 100 -ot Float32 -q",
    )
    with rasterio.open(cli_sc_out) as dst:
        assert dst.dtypes == ("float32", "float32")
        cs = dst.read()
        assert cs[0, 0, 0] == pytest.approx(0.0)
        assert cs[0, 0, 2] == pytest.approx(255.0)
        assert cs[1, 1, 3] == pytest.approx(100.0)

    # strict overflow raises
    over_in = tmp_path / "over_in.tif"
    write_raster(over_in, np.array([[10.0, 300.0]], dtype=np.float32))
    with pytest.raises(rsgispy.RSGISPyException):
        iu_warp.gdal_translate(
            str(over_in),
            str(tmp_path / "over_out.tif"),
            datatype=rsgispy.TYPE_8UINT,
            strict=True,
            use_pbar=False,
        )

    # Colour table expansion (GTiff palette stores RGB and reads alpha=255)
    pal_in = tmp_path / "pal_in.tif"
    write_raster(pal_in, np.array([[0, 1]], dtype=np.uint8))
    with rasterio.open(pal_in, "r+") as ds:
        ds.write_colormap(1, {0: (255, 0, 0, 255), 1: (0, 128, 255, 255)})

    for mode, exp_count in (("rgb", 3), ("rgba", 4), ("gray", 1)):
        exp_p = tmp_path / f"exp_{mode}.tif"
        iu_warp.gdal_translate(
            str(pal_in), str(exp_p), expand=mode, use_pbar=False
        )
        with rasterio.open(exp_p) as dst:
            assert dst.count == exp_count
            e_arr = dst.read()
            if mode == "rgb":
                assert list(e_arr[:, 0, 0]) == [255, 0, 0]
                assert list(dst.colorinterp) == [
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                ]
            elif mode == "rgba":
                assert list(e_arr[:, 0, 1]) == [0, 128, 255, 255]
                assert list(dst.colorinterp) == [
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.alpha,
                ]
            else:
                assert dst.colorinterp == (ColorInterp.gray,)


def test_gdal_translate_metadata_spatial_mask_gcps_and_cli(tmp_path):
    """Test a_srs, ullr, geotransform, mask, gcps, metadata, stats, and CLI string."""
    arr = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + 1.0
    in_p = tmp_path / "meta_in.tif"
    write_raster(in_p, arr, nodata=-1.0)
    with rasterio.open(in_p, "r+") as ds:
        ds.update_tags(SRC_TAG="old")
        ds.update_tags(1, BAND_TAG="b1")
        ds.write_mask(np.ones((4, 4), dtype=np.uint8) * 255)

    out_p = tmp_path / "meta_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(out_p),
        bands=[1, 2],
        out_srs=4326,
        ullr=(0.0, 10.0, 8.0, 2.0),
        no_data_val="none",
        band_scale=[0.5, 0.25],
        band_offset=[1.0, 2.0],
        colour_interp=["blue", "green"],
        metadata={"AUTHOR": "rsgispy"},
        domain_metadata={"MYDOM": {"K": "V"}},
        no_metadata=True,
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(out_p) as dst:
        assert dst.crs.to_epsg() == 4326
        assert dst.nodata is None
        assert dst.transform.c == pytest.approx(0.0)
        assert dst.transform.f == pytest.approx(10.0)
        assert dst.transform.a == pytest.approx(2.0)
        assert dst.scales == pytest.approx((0.5, 0.25))
        assert dst.offsets == pytest.approx((1.0, 2.0))
        assert list(dst.colorinterp) == [ColorInterp.blue, ColorInterp.green]
        assert "SRC_TAG" not in dst.tags()
        assert dst.tags().get("AUTHOR") == "rsgispy"
        assert dst.tags(ns="MYDOM").get("K") == "V"
        assert "STATISTICS_MEAN" in dst.tags(1)
        assert MaskFlags.per_dataset in dst.mask_flag_enums[0]

    # Mask from band + -mask none + geotransform + gcps + approx_stats
    mask_in = tmp_path / "mask_src.tif"
    m_data = np.ones((2, 4, 4), dtype=np.uint8) * 5
    m_data[1, 0, 0] = 0
    write_raster(mask_in, m_data)

    mask_out = tmp_path / "mask_out.tif"
    iu_warp.gdal_translate(
        str(mask_in),
        str(mask_out),
        bands=[1],
        mask_band=2,
        geotransform=(100.0, 2.0, 0.0, 200.0, 0.0, -2.0),
        approx_stats=True,
        use_pbar=False,
    )
    with rasterio.open(mask_out) as dst:
        assert dst.transform.to_gdal() == pytest.approx(
            (100.0, 2.0, 0.0, 200.0, 0.0, -2.0)
        )
        msk = dst.dataset_mask()
        assert msk[0, 0] == 0 and msk[1, 1] == 255
        assert dst.tags(1).get("STATISTICS_APPROXIMATE") == "YES"

    nomask_out = tmp_path / "nomask_out.tif"
    iu_warp.gdal_translate(
        str(mask_out),
        str(nomask_out),
        mask_band="none",
        out_srs="none",
        use_pbar=False,
    )
    with rasterio.open(nomask_out) as dst:
        assert dst.crs is None
        assert MaskFlags.per_dataset not in dst.mask_flag_enums[0]

    # GCPs via CLI and nogcp
    gcp_out = tmp_path / "gcp_out.tif"
    iu_warp.gdal_translate(
        str(in_p),
        str(gcp_out),
        options=(
            "-of GTIFF -ot Float32 -b 2 -b 1 -a_srs EPSG:4326 "
            "-gcp 0 0 10.0 20.0 5.0 -gcp 4 4 14.0 16.0 "
            "-colorinterp_2 red -mo KEY=VAL -dmo DOM:DK=DV "
            "-co COMPRESS=DEFLATE -stats -q"
        ),
    )
    with rasterio.open(gcp_out) as dst:
        gcps_read, gcp_crs = dst.gcps
        assert len(gcps_read) == 2
        assert (gcps_read[0].col, gcps_read[0].row, gcps_read[0].z) == pytest.approx(
            (0.0, 0.0, 5.0)
        )
        assert gcp_crs.to_epsg() == 4326
        assert dst.colorinterp[1] == ColorInterp.red
        assert dst.tags().get("KEY") == "VAL"
        assert dst.tags(ns="DOM").get("DK") == "DV"

    nogcp_out = tmp_path / "nogcp_out.tif"
    iu_warp.gdal_translate(str(gcp_out), str(nogcp_out), no_gcp=True, use_pbar=False)
    with rasterio.open(nogcp_out) as dst:
        assert len(dst.gcps[0]) == 0


def test_gdal_warp_srs_res_size_extent_tap(tmp_path):
    """Test gdal_warp CRS overrides, -tr, -ts, -te, -te_srs, and -tap."""
    arr = np.arange(16 * 8, dtype=np.uint8).reshape(1, 8, 16)
    in_p = tmp_path / "gw_in.tif"
    write_raster(in_p, arr)

    # -tr + -tap + same CRS (no out_epsg required)
    tr_out = tmp_path / "gw_tr_tap.tif"
    iu_warp.gdal_warp(
        str(in_p),
        str(tr_out),
        target_res=(20.0, 20.0),
        tap=True,
        interp_method="bilinear",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(tr_out) as dst:
        assert dst.crs.to_epsg() == 32630
        assert dst.dtypes == ("float32",)
        assert dst.transform.a == pytest.approx(20.0)
        assert dst.transform.c % 20.0 == pytest.approx(0.0)
        assert dst.transform.f % 20.0 == pytest.approx(0.0)
        assert dst.width == 8 and dst.height == 4

    # -ts with 0 aspect-preserving + -tr square
    ts_out = tmp_path / "gw_ts.tif"
    iu_warp.gdal_warp(str(in_p), str(ts_out), target_size=(8, 0))
    with rasterio.open(ts_out) as dst:
        assert dst.width == 8 and dst.height == 4

    sq_out = tmp_path / "gw_sq.tif"
    iu_warp.gdal_warp(str(in_p), str(sq_out), out_epsg=3857, target_res="square")
    with rasterio.open(sq_out) as dst:
        assert dst.crs.to_epsg() == 3857
        assert dst.transform.a == pytest.approx(abs(dst.transform.e))

    # -te + -te_srs + -s_srs
    wgs_b = transform_bounds(
        "EPSG:32630", "EPSG:4326", 500020.0, 5199940.0, 500100.0, 5199980.0
    )
    te_out = tmp_path / "gw_te.tif"
    iu_warp.gdal_warp(
        str(in_p),
        str(te_out),
        src_srs=32630,
        t_srs="EPSG:32630",
        target_extent=wgs_b,
        target_extent_srs=4326,
        target_res=(10.0, 10.0),
    )
    with rasterio.open(te_out) as dst:
        assert dst.width == 8 and dst.height == 4
        assert dst.bounds.left == pytest.approx(500020.0, abs=1.0)


def test_gdal_warp_nodata_bands_alpha_mosaic_and_overwrite(tmp_path):
    """Test src/dst nodata, bands, alpha, multi-input mosaic, and overwrite=False."""
    b1 = np.full((6, 6), 10.0, dtype=np.float32)
    b1[0, 0] = -999.0
    b2 = np.full((6, 6), 25.0, dtype=np.float32)
    b2[0, 1] = -999.0
    in_p = tmp_path / "nd_in.tif"
    write_raster(in_p, np.stack([b1, b2], axis=0), nodata=-999.0)
    with rasterio.open(in_p, "r+") as ds:
        ds.update_tags(COMMON="A", DIFF="V1")

    # src_nodata="none" preserves -999.0 as real data; src_bands reorders; dst_alpha adds alpha
    no_nd_out = tmp_path / "no_nd.tif"
    iu_warp.gdal_warp(
        str(in_p),
        str(no_nd_out),
        out_epsg=32631,
        src_nodata="none",
        dst_nodata=-500.0,
        src_bands=[2, 1],
        dst_alpha=True,
        set_ci=True,
    )
    with rasterio.open(no_nd_out) as dst:
        assert dst.count == 3
        assert dst.colorinterp[2] == ColorInterp.alpha
        assert dst.nodata == pytest.approx(-500.0)
        d = dst.read()
        assert np.any(np.isclose(d[1], -999.0))
        assert np.any(d[2] == 255)

    # -nosrcalpha keeps alpha band as regular band
    rgba_p = tmp_path / "rgba_in.tif"
    rgba_arr = np.full((2, 6, 6), 50, dtype=np.uint8)
    rgba_arr[1, 0, 0] = 0
    write_raster(rgba_p, rgba_arr)
    with rasterio.open(rgba_p, "r+") as ds:
        ds.colorinterp = [ColorInterp.gray, ColorInterp.alpha]

    nosrc_out = tmp_path / "nosrca.tif"
    iu_warp.gdal_warp(str(rgba_p), str(nosrc_out), src_alpha=False)
    with rasterio.open(nosrc_out) as dst:
        assert dst.count == 2
        assert dst.read(2)[0, 0] == 0 and dst.read(1)[0, 0] == 50

    # Multi-input mosaic + meta_conflict_value + overwrite=False
    in_b = tmp_path / "in_b.tif"
    write_raster(
        in_b,
        np.stack([b1 * 2.0, b2 * 2.0], axis=0),
        origin=(500060.0, 5200000.0),
        nodata=-999.0,
    )
    with rasterio.open(in_b, "r+") as ds:
        ds.update_tags(COMMON="A", DIFF="V2")

    mosaic_out = tmp_path / "mosaic_out.tif"
    iu_warp.gdal_warp(
        [str(in_p), str(in_b)],
        str(mosaic_out),
        meta_conflict_value="MIXED",
    )
    with rasterio.open(mosaic_out) as dst:
        assert dst.width == 12
        assert dst.tags().get("COMMON") == "A"
        assert dst.tags().get("DIFF") == "MIXED"
        assert dst.read(1)[1, 1] == pytest.approx(10.0)
        assert dst.read(1)[1, 10] == pytest.approx(20.0)

    # overwrite=False updates existing raster
    upd_src = tmp_path / "upd_src.tif"
    write_raster(
        upd_src,
        np.full((2, 2, 2), 99.0, dtype=np.float32),
        origin=(500020.0, 5199980.0),
        nodata=-999.0,
    )
    iu_warp.gdal_warp(str(upd_src), str(mosaic_out), overwrite=False)
    with rasterio.open(mosaic_out) as dst:
        assert dst.read(1)[2, 2] == pytest.approx(99.0)
        assert dst.read(1)[1, 1] == pytest.approx(10.0)


def test_gdal_warp_cutline_crop_blend_gcps_and_cli(tmp_path):
    """Test cutline (vector/WKT), crop_to_cutline, cblend, GCPs, and CLI string."""
    arr = np.full((1, 10, 10), 100.0, dtype=np.float32)
    in_p = tmp_path / "cut_src.tif"
    write_raster(in_p, arr)

    vec_p = tmp_path / "cut.gpkg"
    gdf = gpd.GeoDataFrame(
        {"keep": [1, 0]},
        geometry=[
            box(500020.0, 5199930.0, 500070.0, 5199980.0),
            box(500080.0, 5199910.0, 500095.0, 5199925.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(vec_p, layer="poly", driver="GPKG")

    crop_out = tmp_path / "crop_cut.tif"
    iu_warp.gdal_warp(
        str(in_p),
        str(crop_out),
        cutline=str(vec_p),
        cutline_layer="poly",
        cutline_where="keep = 1",
        crop_to_cutline=True,
        dst_nodata=-999.0,
    )
    with rasterio.open(crop_out) as dst:
        assert dst.width == 5 and dst.height == 5
        assert np.allclose(dst.read(1), 100.0)

    # WKT cutline + cblend + dst_alpha
    blend_out = tmp_path / "blend_cut.tif"
    wkt_poly = box(500020.0, 5199920.0, 500080.0, 5199980.0).wkt
    iu_warp.gdal_warp(
        str(in_p),
        str(blend_out),
        cutline=wkt_poly,
        cutline_srs=32630,
        cutline_blend=3.0,
        dst_alpha=True,
        dst_nodata=0.0,
    )
    with rasterio.open(blend_out) as dst:
        alpha = dst.read(2)
        assert alpha[0, 0] == 0
        assert 0 < alpha[2, 2] < 255
        assert alpha[5, 5] == 255

    # GCP TPS warp
    gcp_in = tmp_path / "gcp_warp_in.tif"
    gcps = [
        GroundControlPoint(0, 0, 500000.0, 5200000.0),
        GroundControlPoint(0, 9, 500090.0, 5200000.0),
        GroundControlPoint(9, 0, 500000.0, 5199910.0),
        GroundControlPoint(9, 9, 500090.0, 5199910.0),
    ]
    with rasterio.open(
        gcp_in,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:32630",
        gcps=gcps,
    ) as ds:
        ds.write(np.full((1, 10, 10), 12.0, dtype=np.float32))

    tps_out = tmp_path / "tps_out.tif"
    iu_warp.gdal_warp(str(gcp_in), str(tps_out), out_epsg=3857, tps=True)
    with rasterio.open(tps_out) as dst:
        assert dst.crs.to_epsg() == 3857
        assert np.any(np.isclose(dst.read(1), 12.0))

    # Full CLI options string
    cli_out = tmp_path / "cli_warp.tif"
    iu_warp.gdal_warp(
        str(in_p),
        str(cli_out),
        options=(
            "-of GTIFF -ot Float32 -t_srs EPSG:3857 -tr 25 25 -tap "
            "-r bilinear -dstnodata -1 -et 0.1 -wm 32 -wo NUM_THREADS=1 "
            "-co COMPRESS=DEFLATE -nomd -q"
        ),
    )
    with rasterio.open(cli_out) as dst:
        assert dst.crs.to_epsg() == 3857
        assert dst.dtypes == ("float32",)
        assert dst.nodata == pytest.approx(-1.0)
        assert dst.transform.a == pytest.approx(25.0)


def test_resample_img_to_match_uses_header_source_nodata(tmp_path):
    """H8: the source nodata comes from the input header, not no_data_val."""
    ref = str(tmp_path / "ref.tif")
    src = str(tmp_path / "src.tif")
    write_raster(ref, np.zeros((4, 4), dtype=np.float32))
    arr = np.array([[5, 1, 2, 3], [4, 5, 6, 7], [8, 9, 5, 1], [2, 3, 4, 5]],
                   dtype=np.float32)
    write_raster(src, arr, nodata=5)
    out = str(tmp_path / "out.tif")
    iu_warp.resample_img_to_match(ref, src, out, no_data_val=-1.0,
                                  calc_stats=False, use_pbar=False)
    with rasterio.open(out) as ds:
        assert ds.nodata == -1.0
        np.testing.assert_array_equal(ds.read(1), np.where(arr == 5, -1.0, arr))


def test_reproject_image_without_nodata_keeps_zeros(tmp_path, monkeypatch):
    """Without a header nodata or no_data_val, no source pixels are masked."""
    src = str(tmp_path / "src.tif")
    write_raster(src, np.zeros((4, 4), dtype=np.float32))
    seen = {}
    real = iu_warp.rio_reproject

    def _spy(*args, **kwargs):
        seen["src_nodata"] = kwargs.get("src_nodata")
        return real(*args, **kwargs)

    monkeypatch.setattr(iu_warp, "rio_reproject", _spy)
    iu_warp.reproject_image(src, str(tmp_path / "out.tif"), 32630,
                            calc_stats=False, use_pbar=False)
    assert seen["src_nodata"] is None


def test_gdal_warp_src_nodata_none_keeps_alpha_and_zeros(tmp_path):
    """src_nodata='none' keeps header-nodata pixels as data and honours src_alpha."""
    src = str(tmp_path / "src.tif")
    data = np.array([[0, 1, 2, 3], [4, 0, 6, 7], [8, 9, 0, 1], [2, 3, 4, 0]],
                    dtype=np.uint8)
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    alpha[0, 3] = 0
    write_raster(src, np.stack([data, alpha]), nodata=0)
    out = str(tmp_path / "out.tif")
    iu_warp.gdal_warp(src, out, src_nodata="none", dst_nodata=99, src_alpha=True)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    expected = data.copy()
    expected[0, 3] = 99  # masked by the alpha band
    np.testing.assert_array_equal(arr, expected)


def test_gdal_warp_per_band_nodata_alpha_covers_all_bands(tmp_path):
    """With per-band nodata, dst_alpha is opaque wherever any band has data."""
    src = str(tmp_path / "src.tif")
    b1 = np.array([[1, 1], [1, 1]], dtype=np.int16)
    b2 = np.array([[2, 2], [2, 2]], dtype=np.int16)
    b1[0, 0] = -1  # band 1 nodata only
    b2[1, 1] = -2  # band 2 nodata only
    b1[0, 1] = -1
    b2[0, 1] = -2  # both bands nodata
    write_raster(src, np.stack([b1, b2]))
    out = str(tmp_path / "out.tif")
    iu_warp.gdal_warp(src, out, src_nodata=[-1, -2], dst_nodata=[-1, -2],
                      dst_alpha=True)
    with rasterio.open(out) as ds:
        assert ds.count == 3
        alpha = ds.read(3)
    assert alpha[0, 1] == 0
    assert alpha[0, 0] > 0 and alpha[1, 1] > 0 and alpha[1, 0] > 0
