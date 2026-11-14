"""
Tests for extended rsgispy.imageutils submodules: metadata tags/formats/bands,
checks, arrays, sampling, bands, gdal_translate, spatial, and spectral.
"""

import geopandas
import numpy as np
import pytest
import rasterio
import rasterio.crs
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box

import rsgispy
import rsgispy.imageutils.arrays as iu_arr
import rsgispy.imageutils.bands as iu_bands
import rsgispy.imageutils.checks as iu_chk
import rsgispy.imageutils.metadata as iu_meta
import rsgispy.imageutils.sampling as iu_samp
import rsgispy.imageutils.spatial as iu_spat
import rsgispy.imageutils.spectral as iu_spec
import rsgispy.imageutils.warp as iu_warp


def _write_test_tif(
    path,
    data,
    tl_x=500000.0,
    tl_y=5200000.0,
    res=10.0,
    crs="EPSG:32630",
    nodata=None,
):
    """Write a 2D or 3D GeoTIFF test image."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "crs": crs,
        "transform": from_origin(tl_x, tl_y, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)


def test_band_names_datatypes_formats_and_metadata(tmp_path):
    """Test band names, datatype queries, formats, layer rename/delete, and tags."""
    img = tmp_path / "meta.tif"
    arr = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    _write_test_tif(img, arr)

    iu_meta.set_band_names(str(img), ["Red", "NIR"], feedback=True)
    assert iu_meta.get_band_names(str(img)) == ["Red", "NIR"]
    with pytest.raises(rsgispy.RSGISPyException):
        iu_meta.set_band_names(str(img), ["B1", "B2", "B3"])

    assert iu_meta.get_gdal_datatype_from_img(str(img)) == 6
    assert (
        iu_meta.get_rsgislib_datatype_from_img(str(img))
        == rsgispy.TYPE_32FLOAT
    )
    assert iu_meta.get_gdal_datatype_name_from_img(str(img)) == "Float32"

    assert iu_meta.get_gdal_format_from_ext("scene.kea") == "KEA"
    assert iu_meta.get_gdal_format_from_ext("scene.tif") == "GTiff"
    assert iu_meta.get_file_img_extension("GTIFF") == "tif"
    assert iu_meta.get_file_img_extension("KEA") == "kea"
    with pytest.raises(rsgispy.RSGISPyException):
        iu_meta.get_gdal_format_from_ext("scene.xyz")
    with pytest.raises(rsgispy.RSGISPyException):
        iu_meta.get_file_img_extension("UNKNOWN_FMT")

    iu_meta.set_img_metadata(str(img), "SENSOR", "Sentinel-2")
    assert iu_meta.get_img_metadata(str(img), "SENSOR") == "Sentinel-2"
    assert "SENSOR" in iu_meta.get_img_metadata_fields(str(img))
    assert (
        iu_meta.get_img_metadata_fields_dict(str(img))["SENSOR"]
        == "Sentinel-2"
    )

    iu_meta.set_img_band_metadata(str(img), 2, "WAVELENGTH", "842nm")
    assert iu_meta.get_img_band_metadata(str(img), 2, "WAVELENGTH") == "842nm"
    assert "WAVELENGTH" in iu_meta.get_img_band_metadata_fields(str(img), 2)
    assert (
        iu_meta.get_img_band_metadata_fields_dict(str(img), 2)["WAVELENGTH"]
        == "842nm"
    )

    renamed = tmp_path / "meta_renamed.tif"
    iu_meta.rename_gdal_layer(str(img), str(renamed))
    assert not img.exists()
    assert renamed.exists()
    iu_meta.delete_gdal_layer(str(renamed))
    assert not renamed.exists()


def test_image_checks_and_overlap(tmp_path):
    """Test resolution, bbox, projection, list checks, and overlap detection."""
    img1 = tmp_path / "a.tif"
    img2 = tmp_path / "b.tif"
    img3 = tmp_path / "c_other_res.tif"
    img4 = tmp_path / "d_wgs84.tif"
    z = np.zeros((10, 10), dtype=np.float32)

    _write_test_tif(img1, z, tl_x=500000.0, tl_y=5200000.0, res=10.0)
    _write_test_tif(img2, z, tl_x=500050.0, tl_y=5199950.0, res=10.0)
    _write_test_tif(img3, z, tl_x=500000.0, tl_y=5200000.0, res=20.0)
    _write_test_tif(
        img4, z, tl_x=-3.05, tl_y=47.0, res=0.01, crs="EPSG:4326"
    )

    assert iu_chk.do_img_res_match(str(img1), str(img2))
    assert not iu_chk.do_img_res_match(str(img1), str(img3))

    passed = iu_chk.check_img_lst(
        [str(img1), str(img2), str(img3)],
        exp_x_res=10.0,
        exp_y_res=10.0,
        bbox=(500000.0, 500040.0, 5199960.0, 5200000.0),
        print_errors=False,
    )
    assert passed == [str(img1)]

    assert iu_chk.check_img_file_comparison(
        str(img1),
        str(img2),
        test_n_bands=True,
        test_eql_bbox=False,
        print_errors=False,
    )
    assert not iu_chk.check_img_file_comparison(
        str(img1), str(img2), test_eql_bbox=True, print_errors=False
    )
    assert iu_chk.do_gdal_layers_have_same_proj(str(img1), str(img2))
    assert not iu_chk.do_gdal_layers_have_same_proj(str(img1), str(img4))

    assert iu_chk.do_images_overlap(str(img1), str(img2), over_thres=25.0)
    assert not iu_chk.do_images_overlap(str(img1), str(img2), over_thres=60.0)
    assert iu_chk.do_images_overlap(str(img1), str(img4))

    iu_chk.test_img_lst_intersects([str(img1), str(img2)], stop_err=True)
    with pytest.raises(rsgispy.RSGISPyException):
        iu_chk.test_img_lst_intersects([str(img1), str(img3)], stop_err=True)


def test_arrays_and_pixel_access(tmp_path):
    """Test array-to-image creation and pixel/column/subset extraction."""
    arr = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    wkt = rasterio.crs.CRS.from_epsg(32630).to_wkt()  # pylint: disable=c-extension-no-member
    out1 = tmp_path / "from_arr.tif"
    iu_arr.create_img_from_array(
        arr,
        str(out1),
        tl_x=500000.0,
        tl_y=5200060.0,
        out_img_res_x=10.0,
        out_img_res_y=-10.0,
        wkt_string=wkt,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        no_data_val=-999.0,
    )

    out2 = tmp_path / "from_ref.tif"
    iu_arr.create_img_from_array_ref_img(
        arr[0],
        str(out2),
        str(out1),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    assert np.allclose(iu_arr.get_img_data_as_arr(str(out2))[0], arr[0])
    assert np.allclose(
        iu_arr.get_img_data_as_arr(str(out1), img_bands=[2, 1]),
        arr[[1, 0]],
    )

    vals_pxl = iu_arr.get_img_pxl_values(str(out1), 2, [0, 3], [0, 2])
    assert np.allclose(vals_pxl, [arr[1, 0, 0], arr[1, 2, 3]])

    vals_coord = iu_arr.get_img_coords_pxl_values(
        str(out1), 1, [500005.0, 500035.0], [5200055.0, 5200035.0]
    )
    assert np.allclose(vals_coord, [arr[0, 0, 0], arr[0, 2, 3]])

    col_pxl = iu_arr.get_img_pxl_column(str(out1), 2, 1)
    col_spa = iu_arr.get_img_column(str(out1), 500025.0, 5200045.0)
    assert np.allclose(col_pxl, [arr[0, 1, 2], arr[1, 1, 2]])
    assert np.allclose(col_spa, col_pxl)

    iu_arr.set_img_pxl_values(str(out1), 1, [1], [1], pxl_value=777.0)
    assert iu_arr.get_img_pxl_values(str(out1), 1, [1], [1])[0] == (
        pytest.approx(777.0)
    )

    sub_arr = iu_arr.get_img_band_pxl_data(
        str(out1), 2, [500020.0, 500050.0, 5200020.0, 5200050.0]
    )
    assert sub_arr.shape == (3, 3)
    assert np.allclose(sub_arr, arr[1, 1:4, 2:5])


def test_sampling_functions(tmp_path):
    """Test mask sampling, strided pixel extraction, random pixels, and random raster."""
    msk_path = tmp_path / "msk.tif"
    msk = np.zeros((10, 10), dtype=np.uint8)
    msk[1:5, 1:5] = 1
    msk[6:9, 6:9] = 2
    _write_test_tif(msk_path, msk)

    sampled_msk = tmp_path / "sampled_msk.tif"
    iu_samp.perform_random_pxl_sample_in_mask(
        str(msk_path), str(sampled_msk), "GTIFF", [1, 2], 4, rnd_seed=42
    )
    with rasterio.open(sampled_msk) as src:
        s_arr = src.read(1)
        assert int((s_arr == 1).sum()) == 4
        assert int((s_arr == 2).sum()) == 4

    img_path = tmp_path / "vals.tif"
    vals = np.arange(2 * 10 * 10, dtype=np.float32).reshape(2, 10, 10)
    vals[:, 0, 0] = -999.0
    _write_test_tif(img_path, vals, nodata=-999.0)

    assert iu_samp.extract_img_pxl_sample(
        str(img_path), pxl_n_sample=10, no_data_val=-999.0
    ).shape == (10, 2)

    in_msk = iu_samp.extract_img_pxl_vals_in_msk(
        str(img_path), [2, 1], str(msk_path), 1, no_data_val=-999.0
    )
    assert in_msk.shape == (16, 2)
    assert np.allclose(in_msk[:, 0], vals[1, 1:5, 1:5].ravel())

    rnd_pts = tmp_path / "rnd_pts.tif"
    iu_samp.assign_random_pxls(
        str(img_path),
        str(rnd_pts),
        n_pts=8,
        img_band=1,
        gdalformat="GTIFF",
        edge_pxl=1,
        use_no_data=True,
        rnd_seed=123,
    )
    with rasterio.open(rnd_pts) as src:
        rp = src.read(1)
        assert int((rp == 1).sum()) == 8
        assert int(rp[0, :].sum()) == 0

    rnd_img = tmp_path / "rnd_vals.tif"
    iu_samp.generate_random_pxl_vals_img(
        str(img_path), str(rnd_img), "GTIFF", low_val=5, up_val=15, rnd_seed=7
    )
    with rasterio.open(rnd_img) as src:
        ri = src.read(1)
        assert ri.dtype == np.int32
        assert ri.min() >= 5 and ri.max() < 15


def test_bands_stack_select_and_translate(tmp_path):
    """Test stack_img_bands, select_img_bands, and gdal_translate."""
    b1_p = tmp_path / "b1.tif"
    b2_p = tmp_path / "b2.tif"
    a1 = np.full((8, 8), 5.0, dtype=np.float32)
    a1[0, 0] = -99.0
    _write_test_tif(b1_p, a1)
    _write_test_tif(b2_p, np.full((8, 8), 10.0, dtype=np.float32))

    stk_p = tmp_path / "stacked.tif"
    iu_bands.stack_img_bands(
        [str(b1_p), str(b2_p)],
        ["B1", "B2"],
        str(stk_p),
        skip_value=-99.0,
        no_data_val=-999.0,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
    )
    with rasterio.open(stk_p) as src:
        assert src.count == 2
        assert src.descriptions == ("B1", "B2")
        sdata = src.read()
        assert np.allclose(sdata[:, 0, 0], [-999.0, -999.0])
        assert np.allclose(sdata[:, 1, 1], [5.0, 10.0])

    sel_p = tmp_path / "selected.tif"
    iu_bands.select_img_bands(
        str(stk_p), str(sel_p), "GTIFF", rsgispy.TYPE_32INT, [2]
    )
    with rasterio.open(sel_p) as src:
        assert src.count == 1
        assert src.dtypes == ("int32",)
        assert src.read(1)[1, 1] == 10

    trans_p = tmp_path / "translated.tif"
    iu_warp.gdal_translate(
        str(stk_p), str(trans_p), gdalformat="GTIFF", datatype=rsgispy.TYPE_64FLOAT
    )
    with rasterio.open(trans_p) as src:
        assert src.dtypes == ("float64", "float64")
        assert src.descriptions == ("B1", "B2")


def test_sharpen_low_res_bands(tmp_path):
    """Test sharpen_low_res_bands with SharpBandInfo."""
    rng = np.random.default_rng(11)
    pan = rng.uniform(50.0, 150.0, (16, 16)).astype(np.float32)
    low = (pan * 0.4 + 20.0).astype(np.float32)
    ign = np.full((16, 16), 3.0, dtype=np.float32)
    sh_in = tmp_path / "sh_in.tif"
    sh_out = tmp_path / "sh_out.tif"
    _write_test_tif(sh_in, np.stack([low, pan, ign], axis=0))
    binfo = [
        iu_bands.SharpBandInfo(band=1, status=rsgispy.SHARP_RES_LOW, name="Low"),
        iu_bands.SharpBandInfo(
            band=2, status=rsgispy.SHARP_RES_HIGH, name="High"
        ),
        iu_bands.SharpBandInfo(
            band=3, status=rsgispy.SHARP_RES_IGNORE, name="Ign"
        ),
    ]
    iu_bands.sharpen_low_res_bands(
        str(sh_in),
        str(sh_out),
        binfo,
        win_size=7,
        no_data_val=0,
        gdalformat="GTIFF",
    )
    with rasterio.open(sh_out) as src:
        out_s = src.read()
        assert src.descriptions == ("Low", "High", "Ign")
        assert np.allclose(out_s[0], low, atol=1e-3)
        assert np.allclose(out_s[1], pan)
        assert np.allclose(out_s[2], ign)


def test_spatial_pixel_geometry(tmp_path):
    """Test WGS84 pixel area/size and local/WGS84 pixel location rasters."""
    wgs_img = tmp_path / "wgs.tif"
    _write_test_tif(
        wgs_img,
        np.zeros((4, 4), dtype=np.float32),
        tl_x=0.0,
        tl_y=52.0,
        res=0.01,
        crs="EPSG:4326",
    )

    area_img = tmp_path / "area.tif"
    size_img = tmp_path / "size.tif"
    loc_img = tmp_path / "loc.tif"
    wgs_loc_img = tmp_path / "wgs_loc.tif"

    iu_spat.calc_wgs84_pixel_area(
        str(wgs_img), str(area_img), scale=10000, gdalformat="GTIFF"
    )
    iu_spat.calc_wgs84_pixel_size(
        str(wgs_img), str(size_img), gdalformat="GTIFF"
    )
    iu_spat.calc_pixel_locations(str(wgs_img), str(loc_img), gdalformat="GTIFF")

    with rasterio.open(size_img) as s_src, rasterio.open(area_img) as a_src:
        xm, ym = s_src.read(1), s_src.read(2)
        assert np.all(xm > 600.0) and np.all(ym > 1000.0)
        assert np.allclose(a_src.read(1), (xm * ym) / 10000.0)

    with rasterio.open(loc_img) as l_src:
        assert l_src.read(1)[0, 0] == pytest.approx(0.005)
        assert l_src.read(2)[0, 0] == pytest.approx(51.995)

    utm_img = tmp_path / "utm.tif"
    _write_test_tif(utm_img, np.zeros((4, 4), dtype=np.float32))
    iu_spat.calc_wgs84_pixel_locations(
        str(utm_img), str(wgs_loc_img), gdalformat="GTIFF"
    )
    with rasterio.open(wgs_loc_img) as wl_src:
        lons, lats = wl_src.read(1), wl_src.read(2)
        assert np.all((lons > -4.0) & (lons < -2.0))
        assert np.all((lats > 46.0) & (lats < 48.0))


def test_grid_scattered_pts(tmp_path):
    """Test grid_scattered_pts with invdistnn, nearest, and linear."""
    utm_img = tmp_path / "utm_grid.tif"
    _write_test_tif(utm_img, np.zeros((4, 4), dtype=np.float32))

    vec_path = tmp_path / "pts.gpkg"
    gdf = geopandas.GeoDataFrame(
        {"val": [10.0, 20.0, 30.0, 40.0]},
        geometry=[
            Point(500005.0, 5199995.0),
            Point(500035.0, 5199995.0),
            Point(500005.0, 5199965.0),
            Point(500035.0, 5199965.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(str(vec_path), layer="pts", driver="GPKG")

    grid_out = tmp_path / "grid_idw.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(grid_out),
        gdal_grid_alg="invdistnn:power=2.0:radius=100.0:max_points=4",
        gdalformat="GTIFF",
    )
    with rasterio.open(grid_out) as g_src:
        g_arr = g_src.read(1)
        assert g_arr[0, 0] == pytest.approx(10.0)
        assert g_arr[3, 3] == pytest.approx(40.0)

    lin_out = tmp_path / "grid_lin.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(lin_out),
        gdal_grid_alg="linear",
        gdalformat="GTIFF",
    )
    with rasterio.open(lin_out) as l_src:
        assert l_src.read(1)[0, 3] == pytest.approx(20.0)

    near_out = tmp_path / "grid_near.tif"
    iu_spat.grid_scattered_pts(
        str(vec_path),
        "pts",
        "val",
        str(utm_img),
        str(near_out),
        gdal_grid_alg="nearest:radius1=20.0:radius2=20.0",
        gdalformat="GTIFF",
    )
    with rasterio.open(near_out) as n_src:
        assert n_src.read(1)[0, 0] == pytest.approx(10.0)


def test_spectral_smoothing_and_outlier_mask(tmp_path):
    """Test spectral_smoothing (with calc_stats) and mask_outliners_data_values."""
    n_bands = 7
    wavelengths = np.arange(n_bands, dtype=np.float64)
    clean_spec = 2.0 * wavelengths + 1.0
    cube = np.broadcast_to(
        clean_spec[:, None, None], (n_bands, 6, 6)
    ).astype(np.float32).copy()

    in_img = tmp_path / "spec.tif"
    msk_img = tmp_path / "spec_msk.tif"
    _write_test_tif(in_img, cube, nodata=0.0)
    msk = np.zeros((6, 6), dtype=np.uint8)
    msk[1:5, 1:5] = 1
    _write_test_tif(msk_img, msk)

    smooth_out = tmp_path / "smooth.tif"
    iu_spec.spectral_smoothing(
        str(in_img),
        str(msk_img),
        1,
        str(smooth_out),
        win_len=5,
        polyorder=2,
        gdalformat="GTIFF",
        calc_stats=True,
    )
    with rasterio.open(smooth_out) as src:
        sm = src.read()
        assert np.allclose(sm[:, 2, 2], clean_spec, atol=1e-4)
        assert np.allclose(sm[:, 0, 0], 0.0)
        assert "STATISTICS_MEAN" in src.tags(1)

    outl_arr = cube.copy()
    outl_arr[3, 2, 2] = 500.0
    outl_in = tmp_path / "outl_in.tif"
    outl_out = tmp_path / "outl_out.tif"
    _write_test_tif(outl_in, outl_arr)
    iu_spec.mask_outliners_data_values(
        str(outl_in),
        str(msk_img),
        1,
        str(outl_out),
        lower=5,
        upper=85,
        gdalformat="GTIFF",
        calc_stats=False,
    )
    with rasterio.open(outl_out) as src:
        assert np.isnan(src.read()[3, 2, 2])


def test_polyfill_nan_data_values(tmp_path):
    """Test polyfill_nan_data_values reconstructs NaN spectral bands."""
    wavelengths = np.arange(7, dtype=np.float64)
    clean_spec = 2.0 * wavelengths + 1.0
    nan_arr = np.broadcast_to(
        clean_spec[:, None, None], (7, 6, 6)
    ).astype(np.float32).copy()
    nan_arr[3, 2, 2] = np.nan

    nan_in = tmp_path / "nan_in.tif"
    msk_img = tmp_path / "poly_msk.tif"
    poly_out = tmp_path / "poly_out.tif"
    _write_test_tif(nan_in, nan_arr)
    _write_test_tif(msk_img, np.ones((6, 6), dtype=np.uint8))

    iu_spec.polyfill_nan_data_values(
        str(nan_in),
        wavelengths.tolist(),
        str(msk_img),
        1,
        str(poly_out),
        polyorder=2,
        gdalformat="GTIFF",
        calc_stats=False,
    )
    with rasterio.open(poly_out) as src:
        assert src.read()[3, 2, 2] == pytest.approx(clean_spec[3], abs=1e-3)


def test_whiten_image(tmp_path):
    """Test whiten_image produces uncorrelated unit-variance valid spectra."""
    rng = np.random.default_rng(99)
    w_in_arr = rng.normal(0.0, 1.0, (3, 12, 12)).astype(np.float32)
    w_in_arr[1] = (
        w_in_arr[0] * 2.0
        + rng.normal(0.0, 0.5, (12, 12)).astype(np.float32)
    )
    w_in = tmp_path / "w_in.tif"
    w_msk = tmp_path / "w_msk.tif"
    w_out = tmp_path / "w_out.tif"
    _write_test_tif(w_in, w_in_arr)
    _write_test_tif(w_msk, np.ones((12, 12), dtype=np.uint8))

    iu_spec.whiten_image(
        str(w_in), str(w_msk), 1, str(w_out), gdalformat="GTIFF"
    )
    with rasterio.open(w_out) as src:
        cov = np.cov(src.read().reshape(3, -1))
        assert np.allclose(cov, np.eye(3), atol=1e-2)


def test_gdal_translate_bands_and_windows(tmp_path):
    """Test gdal_translate band selection, srcwin, projwin, projwin_srs, epo/eco."""
    arr = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    in_p = tmp_path / "win_in.tif"
    _write_test_tif(in_p, arr, tl_x=500000.0, tl_y=5200000.0, res=10.0)

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
    _write_test_tif(in_p, arr, res=10.0)
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
    _write_test_tif(in_p, dn, nodata=255)
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
    _write_test_tif(over_in, np.array([[10.0, 300.0]], dtype=np.float32))
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
    _write_test_tif(pal_in, np.array([[0, 1]], dtype=np.uint8))
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
    _write_test_tif(in_p, arr, nodata=-1.0)
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
        stats=True,
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
    _write_test_tif(mask_in, m_data)

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
    """Test gdal_warp CRS overrides, -tr, -ts, -te, -te_srs, -tap, and alias."""
    arr = np.arange(16 * 8, dtype=np.uint8).reshape(1, 8, 16)
    in_p = tmp_path / "gw_in.tif"
    _write_test_tif(in_p, arr, tl_x=500000.0, tl_y=5200000.0, res=10.0)

    # -tr + -tap + same CRS (no out_epsg required)
    tr_out = tmp_path / "gw_tr_tap.tif"
    iu_warp.gdalwarp(
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
    _write_test_tif(
        in_p, np.stack([b1, b2], axis=0), tl_x=500000.0, tl_y=5200000.0, nodata=-999.0
    )
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
    _write_test_tif(rgba_p, rgba_arr)
    with rasterio.open(rgba_p, "r+") as ds:
        ds.colorinterp = [ColorInterp.gray, ColorInterp.alpha]

    nosrc_out = tmp_path / "nosrca.tif"
    iu_warp.gdal_warp(str(rgba_p), str(nosrc_out), src_alpha=False)
    with rasterio.open(nosrc_out) as dst:
        assert dst.count == 2
        assert dst.read(2)[0, 0] == 0 and dst.read(1)[0, 0] == 50

    # Multi-input mosaic + meta_conflict_value + overwrite=False
    in_b = tmp_path / "in_b.tif"
    _write_test_tif(
        in_b,
        np.stack([b1 * 2.0, b2 * 2.0], axis=0),
        tl_x=500060.0,
        tl_y=5200000.0,
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
    _write_test_tif(
        upd_src,
        np.full((2, 2, 2), 99.0, dtype=np.float32),
        tl_x=500020.0,
        tl_y=5199980.0,
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
    _write_test_tif(in_p, arr, tl_x=500000.0, tl_y=5200000.0, res=10.0)

    vec_p = tmp_path / "cut.gpkg"
    gdf = geopandas.GeoDataFrame(
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
