"""
Tests for :mod:`rsgispy.imagecalc.spectralunmixing`.
"""

import importlib.util
import inspect
import math

import geopandas
import numpy as np
import pandas
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import rsgispy
import rsgispy.imagecalc.spectralunmixing as su
import rsgispy.zonalstats as zs

HAVE_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAVE_OPENPYXL = importlib.util.find_spec("openpyxl") is not None
HAVE_TABLES = importlib.util.find_spec("tables") is not None


def _write_raster(
    path,
    data,
    origin=(0.0, 100.0),
    res=10.0,
    nodata=None,
    descriptions=None,
):
    """Write a 3D ``(bands, height, width)`` GeoTIFF raster."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs="EPSG:32630",
        transform=from_origin(origin[0], origin[1], res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions is not None:
            for b_idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(b_idx, desc)


def _write_legacy_mtxt(path, em_arr):
    """Write a legacy RSGISLib ``.mtxt`` matrix file for conversion tests."""
    m, n = em_arr.shape
    vals = []
    for n_idx in range(n):
        for m_idx in range(m):
            vals.append(repr(float(em_arr[m_idx, n_idx])))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"m={m}\n")
        handle.write(f"n={n}\n")
        handle.write(",".join(vals) + "\n")


def test_exports_and_default_gdalformat():
    """Verify recommended/optional unmixing exports and GTIFF defaults."""
    assert hasattr(su, "read_endmembers_df")
    assert hasattr(su, "write_endmembers_df")
    assert hasattr(su, "convert_mtxt_to_df")
    assert not hasattr(su, "read_endmembers_mtxt")
    assert not hasattr(su, "write_endmembers_mtxt")

    funcs = [
        su.spec_unmix_ucls,
        su.spec_unmix_nnls,
        su.spec_unmix_fcls,
        su.spec_unmix_spts_ucls,
        su.spec_unmix_spts_nnls,
        su.spec_unmix_spts_fcls,
        su.spec_unmix_pymcr_nnls,
        su.spec_unmix_pymcr_fcls,
        su.exhcon_linear_spec_unmix,
        su.rescale_unmixing_results,
        su.predict_refl_linear_unmixing,
        su.calc_unmixing_rmse_residual_err,
        su.summarise_multi_endmember_linear_unmixing,
        su.calc_ppi,
    ]
    for fn in funcs:
        assert fn.__name__ in su.__all__
        sig = inspect.signature(fn)
        assert sig.parameters["gdalformat"].default == "GTIFF"


def test_df_roundtrip_formats_plot_and_equality(tmp_path):
    """Verify DataFrame write/read across user-chosen formats, plot, and equality."""
    em_df = pandas.DataFrame(
        {"Soil": [0.2, 0.4, 0.8], "Veg": [0.6, 0.2, 0.1]},
        index=pandas.Index(["blue", "green", "nir"], name="band"),
    )

    formats = [
        ("csv", "em.csv", None),
        ("tsv", "em.tsv", None),
        ("json", "em.json", None),
        ("pickle", "em.pkl", None),
        ("csv_override", "em_custom.dat", "csv"),
    ]
    if HAVE_PYARROW:
        formats.extend(
            [
                ("parquet", "em.parquet", None),
                ("feather", "em.feather", None),
            ]
        )
    if HAVE_OPENPYXL:
        formats.append(("excel", "em.xlsx", None))
    if HAVE_TABLES:
        formats.append(("hdf", "em.h5", None))

    for _, fname, fmt_override in formats:
        out_path = tmp_path / fname
        su.write_endmembers_df(em_df, out_path, file_format=fmt_override)
        assert out_path.exists()

        read_back = su.read_endmembers_df(out_path, file_format=fmt_override)
        assert list(read_back.columns) == ["Soil", "Veg"]
        assert list(read_back.index) == ["blue", "green", "nir"]
        np.testing.assert_allclose(read_back.to_numpy(), em_df.to_numpy(), atol=1e-9)
        assert su.are_endmembers_equal(em_df, read_back, flt_dif=1e-7)

    # Hand-crafted CSV without an index column should preserve class columns
    plain_csv = tmp_path / "plain.csv"
    plain_csv.write_text("Soil,Veg\n0.2,0.6\n0.4,0.2\n0.8,0.1\n", encoding="utf-8")
    plain_df = su.read_endmembers_df(plain_csv, gain=2.0)
    assert list(plain_df.columns) == ["Soil", "Veg"]
    assert list(plain_df.index) == ["band_1", "band_2", "band_3"]
    np.testing.assert_allclose(plain_df.to_numpy(), em_df.to_numpy() / 2.0)

    # Equality respects flt_dif and class column names
    em_df_close = em_df + 1e-5
    em_df_renamed = em_df.rename(columns={"Veg": "Forest"})
    assert su.are_endmembers_equal(em_df, em_df_close, flt_dif=1e-4)
    assert not su.are_endmembers_equal(em_df, em_df_close, flt_dif=1e-7)
    assert not su.are_endmembers_equal(em_df, em_df_renamed, check_names=True)
    assert su.are_endmembers_equal(em_df, em_df_renamed, check_names=False)

    # Plot works directly from DataFrame and from file path
    plot_from_df = tmp_path / "em_df.png"
    plot_from_csv = tmp_path / "em_csv.png"
    su.plot_endmembers(
        em_df,
        str(plot_from_df),
        plot_title="Test Endmembers (DF)",
        wavelengths=[480, 560, 660],
    )
    su.plot_endmembers(str(tmp_path / "em.csv"), str(plot_from_csv))
    assert plot_from_df.exists() and plot_from_df.stat().st_size > 0
    assert plot_from_csv.exists() and plot_from_csv.stat().st_size > 0


def test_convert_mtxt_to_df(tmp_path):
    """Verify legacy .mtxt conversion returns DataFrame and optionally writes to disk."""
    mtxt_path = tmp_path / "legacy.mtxt"
    out_csv = tmp_path / "converted.csv"
    em_arr = np.array([[0.2, 0.4, 0.8], [0.6, 0.2, 0.1]], dtype=np.float64)
    _write_legacy_mtxt(mtxt_path, em_arr)

    default_df = su.convert_mtxt_to_df(mtxt_path)
    assert default_df.shape == (3, 2)
    assert list(default_df.columns) == ["endmember_1", "endmember_2"]
    assert list(default_df.index) == ["band_1", "band_2", "band_3"]
    np.testing.assert_allclose(default_df.to_numpy().T, em_arr)

    named_df = su.convert_mtxt_to_df(
        mtxt_path,
        endmember_names=["Soil", "Veg"],
        band_names=["B2", "B3", "B8"],
        gain=2.0,
        out_file=out_csv,
    )
    assert list(named_df.columns) == ["Soil", "Veg"]
    assert list(named_df.index) == ["B2", "B3", "B8"]
    np.testing.assert_allclose(named_df.to_numpy().T, em_arr / 2.0)

    reread_df = su.read_endmembers_df(out_csv)
    assert su.are_endmembers_equal(named_df, reread_df, flt_dif=1e-9)

    with pytest.raises(rsgispy.RSGISPyException):
        su.convert_mtxt_to_df(mtxt_path, endmember_names=["OnlyOne"])
    with pytest.raises(rsgispy.RSGISPyException):
        su.convert_mtxt_to_df(mtxt_path, band_names=["B1", "B2"])


def test_extract_avg_endmembers_and_zonalstats_constants(tmp_path):
    """Verify extract_avg_endmembers returns a DataFrame with class columns."""
    assert zs.METHOD_POLYCONTAINSPIXELCENTER == 1
    assert zs.METHOD_POLYOVERLAPSPIXEL == 2

    img_path = tmp_path / "img.tif"
    vec_path = tmp_path / "rois.gpkg"
    out_csv = tmp_path / "extracted_em.csv"

    # 4x4 grid, origin (0, 40), res=10 -> x in [0, 40], y in [0, 40]
    arr = np.zeros((3, 4, 4), dtype=np.float32)
    arr[:, :2, :2] = np.array([0.2, 0.4, 0.8])[:, None, None]
    arr[:, :2, 2:] = np.array([0.4, 0.6, 0.4])[:, None, None]
    arr[:, 2:, 2:] = np.array([0.6, 0.2, 0.1])[:, None, None]
    _write_raster(
        img_path,
        arr,
        origin=(0.0, 40.0),
        res=10.0,
        descriptions=["Blue", "Green", "NIR"],
    )

    # Two polygons share class "Soil" (means [0.2,0.4,0.8] and [0.4,0.6,0.4] -> [0.3,0.5,0.6])
    gdf = geopandas.GeoDataFrame(
        {"landcover": ["Soil", "Soil", "Veg"]},
        geometry=[
            box(0.0, 20.0, 20.0, 40.0),
            box(20.0, 20.0, 40.0, 40.0),
            box(20.0, 0.0, 40.0, 20.0),
        ],
        crs="EPSG:32630",
    )
    gdf.to_file(vec_path, driver="GPKG", layer="rois")

    em_df = su.extract_avg_endmembers(
        str(img_path),
        str(vec_path),
        "rois",
        out_file=str(out_csv),
        class_col="landcover",
        pxl_in_poly_method=zs.METHOD_POLYCONTAINSPIXELCENTER,
    )

    assert isinstance(em_df, pandas.DataFrame)
    assert list(em_df.columns) == ["Soil", "Veg"]
    assert list(em_df.index) == ["Blue", "Green", "NIR"]
    np.testing.assert_allclose(em_df["Soil"].to_numpy(), [0.3, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(em_df["Veg"].to_numpy(), [0.6, 0.2, 0.1], atol=1e-6)
    assert out_csv.exists()
    assert su.are_endmembers_equal(em_df, su.read_endmembers_df(out_csv))


def test_exhcon_unmixing_and_downstream_metrics(tmp_path):
    """Verify exhaustive constrained unmixing, rescaling, prediction, error, and PPI."""
    img_path = tmp_path / "refl.tif"
    em_path = tmp_path / "em.csv"
    exh_path = tmp_path / "exh.tif"
    resc_path = tmp_path / "resc.tif"
    pred_path = tmp_path / "pred.tif"
    err_path = tmp_path / "err.tif"
    ppi_path = tmp_path / "ppi.tif"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pandas.DataFrame(
        {"Soil": em1, "Veg": em2},
        index=pandas.Index(["Blue", "Green", "NIR"], name="band"),
    )
    su.write_endmembers_df(em_df, str(em_path))

    # Build a 2x2 image: exact EM1, exact EM2, 50/50 mix (of unnormed), zeros
    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = em1
    arr[:, 0, 1] = em2
    arr[:, 1, 0] = 0.5 * em1 + 0.5 * em2
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    su.exhcon_linear_spec_unmix(
        str(img_path),
        str(exh_path),
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        endmembers=str(em_path),
        step_res=0.1,
        use_pbar=False,
    )
    with rasterio.open(exh_path) as src:
        out_exh = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "RMSE"]
    assert out_exh.shape == (3, 2, 2)
    np.testing.assert_allclose(out_exh[:, 0, 0], [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_exh[:, 0, 1], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_exh[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)
    assert abs((out_exh[0, 1, 0] + out_exh[1, 1, 0]) - 1.0) < 1e-5

    # Test rescale_unmixing_results with negative and >1 values
    raw_coef_path = tmp_path / "raw_coef.tif"
    raw_coef = np.array(
        [[[0.6, -0.2], [0.3, 0.0]], [[0.6, 0.8], [0.7, 0.0]]],
        dtype=np.float32,
    )
    _write_raster(raw_coef_path, raw_coef, origin=(0.0, 20.0), res=10.0)
    su.rescale_unmixing_results(
        str(raw_coef_path), str(resc_path), calc_stats=False, use_pbar=False
    )
    with rasterio.open(resc_path) as src:
        resc = src.read()
    np.testing.assert_allclose(resc[:, 0, 0], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(resc[:, 0, 1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(resc[:, 1, 1], [0.0, 0.0], atol=1e-6)

    # Exact abundance raster -> predicted reflectance (in-memory DataFrame) and zero error
    exact_coef = np.array(
        [[[1.0, 0.0], [0.5, 0.0]], [[0.0, 1.0], [0.5, 0.0]]],
        dtype=np.float32,
    )
    exact_coef_path = tmp_path / "exact_coef.tif"
    _write_raster(exact_coef_path, exact_coef, origin=(0.0, 20.0), res=10.0)

    su.predict_refl_linear_unmixing(
        str(exact_coef_path),
        em_df,
        str(pred_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(pred_path) as src:
        pred = src.read()
        assert list(src.descriptions) == ["Blue", "Green", "NIR"]
    np.testing.assert_allclose(pred, arr, atol=1e-6)

    su.calc_unmixing_rmse_residual_err(
        str(img_path),
        str(exact_coef_path),
        str(em_path),
        str(err_path),
        calc_stats=True,
        use_pbar=False,
    )
    with rasterio.open(err_path) as src:
        err = src.read()
        assert list(src.descriptions) == ["RMSE", "RMSPE", "Residual"]
    np.testing.assert_allclose(err, 0.0, atol=1e-5)

    # PPI test
    su.calc_ppi(
        str(img_path),
        str(ppi_path),
        niters=50,
        seed=42,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ppi_path) as src:
        ppi = src.read(1)
        assert src.dtypes[0] == "uint16"
    assert int(np.sum(ppi)) > 0


def test_summarise_multi_endmember_linear_unmixing(tmp_path):
    """Verify multi-model summary picks lower-RMSPE model using DataFrame/file columns."""
    refl_path = tmp_path / "refl.tif"
    em_a_path = tmp_path / "em_a.csv"
    unmix_a_path = tmp_path / "unmix_a.tif"
    unmix_b_path = tmp_path / "unmix_b.tif"
    out_unmix = tmp_path / "best_unmix.tif"
    out_ref = tmp_path / "best_ref.tif"

    em_veg = np.array([0.1, 0.3, 0.8], dtype=np.float64)
    em_soil = np.array([0.5, 0.4, 0.2], dtype=np.float64)
    em_water = np.array([0.2, 0.1, 0.05], dtype=np.float64)

    em_a_df = pandas.DataFrame({"Veg": em_veg, "Soil": em_soil})
    em_b_df = pandas.DataFrame({"Veg": em_veg, "Water": em_water})
    su.write_endmembers_df(em_a_df, str(em_a_path))

    # Left half: pure soil (perfectly explained by model A); right half: pure water (model B)
    refl = np.zeros((3, 2, 2), dtype=np.float32)
    refl[:, :, 0] = em_soil[:, None]
    refl[:, :, 1] = em_water[:, None]
    _write_raster(refl_path, refl, origin=(0.0, 20.0), res=10.0)

    coef_a = np.zeros((2, 2, 2), dtype=np.float32)
    coef_a[1, :, 0] = 1.0  # model A gets col 0 exact (Soil=1) and fails col 1 (0)
    _write_raster(unmix_a_path, coef_a, origin=(0.0, 20.0), res=10.0)

    coef_b = np.zeros((2, 2, 2), dtype=np.float32)
    coef_b[1, :, 1] = 1.0  # model B gets col 1 exact (Water=1) and fails col 0
    _write_raster(unmix_b_path, coef_b, origin=(0.0, 20.0), res=10.0)

    # Mix file path (class names inferred from CSV columns) and in-memory DataFrame
    info_a = su.ImageEndmemberInfo(str(unmix_a_path), str(em_a_path))
    info_b = su.ImageEndmemberInfo(str(unmix_b_path), em_b_df)
    assert info_a == su.ImageEndmemberInfo(str(unmix_a_path), str(em_a_path))
    assert info_b == su.ImageEndmemberInfo(str(unmix_b_path), em_b_df.copy())
    assert info_a != info_b

    su.summarise_multi_endmember_linear_unmixing(
        str(refl_path),
        [info_a, info_b],
        str(out_unmix),
        str(out_ref),
        tmp_dir=str(tmp_path / "tmp"),
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_ref) as src:
        ref = src.read(1)
    assert np.all(ref[:, 0] == 1)
    assert np.all(ref[:, 1] == 2)

    with rasterio.open(out_unmix) as src:
        comb = src.read()
        assert list(src.descriptions) == ["Veg", "Soil", "Water"]
    np.testing.assert_allclose(comb[:, 0, 0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(comb[:, 0, 1], [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.skipif(not su.HAVE_PYSPTOOLS, reason="pysptools not installed")
def test_pysptools_unmixers(tmp_path):
    """Verify PySptools UCLS, NNLS, and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pandas.DataFrame({"Soil": em1, "Veg": em2})
    su.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_ucls = tmp_path / "ucls.tif"
    su.spec_unmix_spts_ucls(
        str(img_path),
        str(msk_path),
        1,
        str(out_ucls),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_ucls) as src:
        assert list(src.descriptions) == ["Soil", "Veg"]
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-3)

    out_nnls = tmp_path / "nnls.tif"
    su.spec_unmix_spts_nnls(
        str(img_path),
        str(msk_path),
        1,
        str(out_nnls),
        em_df,
        weight=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    if su.HAVE_CVXOPT:
        out_fcls = tmp_path / "fcls.tif"
        su.spec_unmix_spts_fcls(
            str(img_path),
            str(msk_path),
            1,
            str(out_fcls),
            str(em_path),
            calc_stats=False,
            use_pbar=False,
        )
        with rasterio.open(out_fcls) as src:
            np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


@pytest.mark.skipif(not su.HAVE_PYMCR, reason="pymcr not installed")
def test_pymcr_unmixers(tmp_path):
    """Verify PyMCR NNLS and FCLS end-to-end when installed."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.json"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em_df = pandas.DataFrame({"Soil": em1, "Veg": em2})
    su.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.3 * em1 + 0.7 * em2
    arr[:, 0, 1] = em1
    arr[:, 1, 0] = em2
    arr[:, 1, 1] = 0.5 * em1 + 0.5 * em2
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "pymcr_nnls.tif"
    su.spec_unmix_pymcr_nnls(
        str(img_path),
        str(msk_path),
        1,
        str(out_nnls),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        assert list(src.descriptions) == ["Soil", "Veg"]
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)

    out_fcls = tmp_path / "pymcr_fcls.tif"
    su.spec_unmix_pymcr_fcls(
        str(img_path),
        str(msk_path),
        1,
        str(out_fcls),
        em_df,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        np.testing.assert_allclose(src.read()[:, 0, 0], [0.3, 0.7], atol=1e-2)


def test_recommended_numpy_ucls_unmixer(tmp_path):
    """Verify recommended pure-NumPy UCLS unmixing with DataFrame and CSV files."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    em2 = np.array([0.6, 0.2, 0.1], dtype=np.float64)
    em3 = np.array([0.3, 0.7, 0.2], dtype=np.float64)
    em_df = pandas.DataFrame({"Soil": em1, "Veg": em2, "Water": em3})
    su.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.2 * em1 + 0.5 * em2 + 0.3 * em3
    arr[:, 0, 1] = 0.8 * em1 + 0.2 * em2
    arr[:, 1, 0] = em3
    arr[:, 1, 1] = [0.0, 0.0, 0.0]
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.array([[[1, 1], [0, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_masked = tmp_path / "numpy_ucls.tif"
    su.spec_unmix_ucls(
        str(img_path),
        str(msk_path),
        1,
        str(out_masked),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_masked) as src:
        res = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Water"]
    np.testing.assert_allclose(res[:, 0, 0], [0.2, 0.5, 0.3], atol=1e-5)
    np.testing.assert_allclose(res[:, 0, 1], [0.8, 0.2, 0.0], atol=1e-5)
    np.testing.assert_allclose(res[:, 1, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(res[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)

    out_nomask = tmp_path / "numpy_ucls_nomask.tif"
    su.spec_unmix_ucls(
        str(img_path),
        None,
        1,
        str(out_nomask),
        em_df,
        weight=10.0,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nomask) as src:
        res_nm = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Water"]
    np.testing.assert_allclose(res_nm[:, 1, 0], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(res_nm[:, 1, 1], [0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_recommended_scipy_nnls_fcls_unmixers(tmp_path):
    """Verify recommended SciPy NNLS and FCLS unmixers using DataFrame and files."""
    img_path = tmp_path / "refl.tif"
    msk_path = tmp_path / "msk.tif"
    em_path = tmp_path / "em.csv"

    em1 = np.array([0.1, 0.2, 0.7, 0.3], dtype=np.float64)
    em2 = np.array([0.6, 0.3, 0.1, 0.2], dtype=np.float64)
    em3 = np.array([0.2, 0.6, 0.3, 0.7], dtype=np.float64)
    em_df = pandas.DataFrame({"Soil": em1, "Veg": em2, "Shadow": em3})
    su.write_endmembers_df(em_df, str(em_path))

    arr = np.zeros((4, 2, 2), dtype=np.float32)
    arr[:, 0, 0] = 0.25 * em1 + 0.45 * em2 + 0.30 * em3
    arr[:, 0, 1] = em1
    # Scaled sum != 1 pixel to test NNLS vs FCLS constraints
    arr[:, 1, 0] = 1.4 * em2 + 0.2 * em3
    # Perturbed out-of-simplex pixel
    arr[:, 1, 1] = np.clip(1.2 * em1 - 0.15 * em2 + 0.05 * em3, 0.01, None)
    _write_raster(img_path, arr, origin=(0.0, 20.0), res=10.0)

    msk = np.ones((1, 2, 2), dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    out_nnls = tmp_path / "scipy_nnls.tif"
    su.spec_unmix_nnls(
        str(img_path),
        str(msk_path),
        1,
        str(out_nnls),
        str(em_path),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_nnls) as src:
        nnls_arr = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Shadow"]
    assert np.all(nnls_arr >= -1e-7)
    np.testing.assert_allclose(nnls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(nnls_arr[:, 1, 0], [0.0, 1.4, 0.2], atol=1e-5)

    out_fcls = tmp_path / "scipy_fcls.tif"
    su.spec_unmix_fcls(
        str(img_path),
        str(msk_path),
        1,
        str(out_fcls),
        em_df,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls) as src:
        fcls_arr = src.read()
        assert list(src.descriptions) == ["Soil", "Veg", "Shadow"]
    assert np.all(fcls_arr >= -1e-7)
    np.testing.assert_allclose(np.sum(fcls_arr, axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 0], [0.25, 0.45, 0.30], atol=1e-5)
    np.testing.assert_allclose(fcls_arr[:, 0, 1], [1.0, 0.0, 0.0], atol=1e-5)

    # Scale-invariance check (reflectance scaled by 10000 with gain=1) using Parquet or JSON
    scaled_img = tmp_path / "refl_scaled.tif"
    ext = ".parquet" if HAVE_PYARROW else ".json"
    scaled_em = tmp_path / f"em_scaled{ext}"
    _write_raster(
        scaled_img, (arr * 10000.0).astype(np.float32), origin=(0.0, 20.0), res=10.0
    )
    su.write_endmembers_df(em_df * 10000.0, str(scaled_em))
    out_fcls_scaled = tmp_path / "scipy_fcls_scaled.tif"
    su.spec_unmix_fcls(
        str(scaled_img),
        None,
        1,
        str(out_fcls_scaled),
        str(scaled_em),
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(out_fcls_scaled) as src:
        fcls_scaled = src.read()
    np.testing.assert_allclose(fcls_scaled, fcls_arr, atol=1e-5)


@pytest.mark.skipif(not su.HAVE_SCIPY, reason="scipy not installed")
def test_batched_nnls_fcls_equivalence_and_edge_cases():
    """Verify batched active-set solvers match SciPy reference across m=1, m=5, and m=22."""
    import scipy.optimize

    rng = np.random.default_rng(123)

    # m=1 single endmember edge case
    em1 = rng.uniform(0.1, 0.9, size=(1, 4))
    px1 = rng.uniform(0.1, 0.9, size=(10, 4))
    out_fcls1 = su._scipy_fcls_solver(px1, em1)  # pylint: disable=protected-access
    np.testing.assert_allclose(out_fcls1, np.ones((10, 1)), atol=1e-12)
    out_nnls1 = su._scipy_nnls_solver(px1, em1)  # pylint: disable=protected-access
    ref_nnls1 = np.array([scipy.optimize.nnls(em1.T, p)[0] for p in px1])
    np.testing.assert_allclose(out_nnls1, ref_nnls1, atol=1e-10)

    # m=5 mixed interior and active-boundary pixels
    em5 = rng.uniform(0.05, 0.85, size=(5, 8))
    w_true = rng.dirichlet(np.ones(5) * 0.4, size=200)
    px5 = w_true @ em5 + rng.normal(0.0, 0.05, size=(200, 8))
    # Include unscaled / negative-coefficient stress cases
    px5[:20] = -0.3 * em5[0] + 1.5 * em5[2] + rng.normal(0.0, 0.02, size=(20, 8))

    out_nnls5 = su._scipy_nnls_solver(px5, em5)  # pylint: disable=protected-access
    ref_nnls5 = np.array([scipy.optimize.nnls(em5.T, p)[0] for p in px5])
    np.testing.assert_allclose(out_nnls5, ref_nnls5, atol=1e-8)

    out_fcls5 = su._scipy_fcls_solver(px5, em5)  # pylint: disable=protected-access
    ref_fcls5 = su._scipy_fcls_heinz_chang_loop(px5, em5)  # pylint: disable=protected-access
    assert np.all(out_fcls5 >= -1e-12)
    np.testing.assert_allclose(np.sum(out_fcls5, axis=1), 1.0, atol=1e-10)
    # Check residuals are at least as low as reference Heinz-Chang solver
    res_batched = np.sum((out_fcls5 @ em5 - px5) ** 2, axis=1)
    res_ref = np.sum((ref_fcls5 @ em5 - px5) ** 2, axis=1)
    assert np.all(res_batched <= res_ref + 1e-7)

    # m=22 fallback path (m > 20)
    em22 = rng.uniform(0.1, 0.9, size=(22, 25))
    px22 = rng.uniform(0.1, 0.9, size=(5, 25))
    out_nnls22 = su._scipy_nnls_solver(px22, em22)  # pylint: disable=protected-access
    assert out_nnls22.shape == (5, 22)
    assert np.all(out_nnls22 >= -1e-12)
    out_fcls22 = su._scipy_fcls_solver(px22, em22)  # pylint: disable=protected-access
    assert out_fcls22.shape == (5, 22)
    np.testing.assert_allclose(np.sum(out_fcls22, axis=1), 1.0, atol=1e-6)


def test_optional_dependency_friendly_errors(monkeypatch, tmp_path):
    """Verify optional unmixers raise RSGISPyException with pip install hints when missing."""
    em_df = pandas.DataFrame({"Soil": [0.2, 0.4], "Veg": [0.6, 0.2]})
    dummy_in = str(tmp_path / "in.tif")
    dummy_out = str(tmp_path / "out.tif")

    checks = [
        (su.spec_unmix_nnls, "HAVE_SCIPY", "scipy"),
        (su.spec_unmix_fcls, "HAVE_SCIPY", "scipy"),
        (su.spec_unmix_spts_ucls, "HAVE_PYSPTOOLS", "pysptools"),
        (su.spec_unmix_spts_nnls, "HAVE_PYSPTOOLS", "pysptools"),
        (su.spec_unmix_spts_fcls, "HAVE_PYSPTOOLS", "pysptools"),
        (su.spec_unmix_spts_fcls, "HAVE_CVXOPT", "cvxopt"),
        (su.spec_unmix_pymcr_nnls, "HAVE_PYMCR", "pyMCR"),
        (su.spec_unmix_pymcr_fcls, "HAVE_PYMCR", "pyMCR"),
    ]
    for fn, flag, pkg in checks:
        monkeypatch.setattr(su, flag, False)
        with pytest.raises(rsgispy.RSGISPyException, match="pip install"):
            fn(dummy_in, None, 1, dummy_out, em_df)
        with pytest.raises(rsgispy.RSGISPyException, match=pkg):
            fn(dummy_in, None, 1, dummy_out, em_df)
        monkeypatch.undo()


def test_spectral_angle_mapper(tmp_path):
    """Test spectral_angle_mapper accuracy, scale invariance, thresholding, and nodata."""
    em_veg = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    em_soil = np.array([3.0, 1.0, 1.0], dtype=np.float64)
    em_df = pandas.DataFrame({"Veg": em_veg, "Soil": em_soil})
    em_csv = tmp_path / "sam_em.csv"
    su.write_endmembers_df(em_df, str(em_csv))

    img_path = tmp_path / "sam_in.tif"
    msk_path = tmp_path / "sam_msk.tif"
    out_ang = tmp_path / "sam_ang.tif"
    out_cls = tmp_path / "sam_cls.tif"
    out_min = tmp_path / "sam_min.tif"

    # 2x3 image:
    # (0, 0): scaled Veg (3 * em_veg) -> exact angle 0 to Veg -> class 1
    # (0, 1): scaled Soil (0.5 * em_soil) -> exact angle 0 to Soil -> class 2
    # (0, 2): equal mix far from both -> large angle -> > max_angle=15 deg -> class 0
    # (1, 0): valid pure Veg but masked out -> nodata / class 0
    # (1, 1): nodata (-9999) -> nodata / class 0
    # (1, 2): zero spectrum -> nodata / class 0
    cube = np.zeros((3, 2, 3), dtype=np.float32)
    cube[:, 0, 0] = (3.0 * em_veg).astype(np.float32)
    cube[:, 0, 1] = (0.5 * em_soil).astype(np.float32)
    cube[:, 0, 2] = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    cube[:, 1, 0] = em_veg.astype(np.float32)
    cube[:, 1, 1] = -9999.0
    cube[:, 1, 2] = 0.0
    _write_raster(img_path, cube, origin=(0.0, 20.0), res=10.0, nodata=-9999.0)

    msk = np.array([[[1, 1, 1], [0, 1, 1]]], dtype=np.uint8)
    _write_raster(msk_path, msk, origin=(0.0, 20.0), res=10.0)

    su.spectral_angle_mapper(
        str(img_path),
        str(em_csv),
        str(out_ang),
        out_class_img=str(out_cls),
        out_min_angle_img=str(out_min),
        degrees=True,
        max_angle=15.0,
        valid_msk_img=str(msk_path),
        valid_msk_val=1,
        calc_stats=True,
        use_pbar=False,
    )

    with rasterio.open(out_ang) as src:
        ang = src.read()
        assert list(src.descriptions) == ["Veg", "Soil"]
    assert ang[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
    assert ang[1, 0, 1] == pytest.approx(0.0, abs=1e-3)
    expected_deg = math.degrees(
        math.acos(2.0 / math.sqrt(21.0))
    )  # [0,0,5] vs [1,4,2]
    assert ang[0, 0, 2] == pytest.approx(expected_deg, abs=1e-3)
    assert ang[0, 1, 0] == pytest.approx(-9999.0)
    assert ang[0, 1, 1] == pytest.approx(-9999.0)
    assert ang[0, 1, 2] == pytest.approx(-9999.0)

    with rasterio.open(out_cls) as src:
        cls = src.read(1)
    assert cls[0, 0] == 1
    assert cls[0, 1] == 2
    assert cls[0, 2] == 0  # exceeds max_angle=15 deg
    assert cls[1, 0] == 0  # masked out
    assert cls[1, 1] == 0  # nodata
    assert cls[1, 2] == 0  # zero norm

    with rasterio.open(out_min) as src:
        m_ang = src.read(1)
    assert m_ang[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert m_ang[0, 1] == pytest.approx(0.0, abs=1e-3)

    # Error handling: band mismatch and zero-norm endmember
    with pytest.raises(rsgispy.RSGISPyException):
        su.spectral_angle_mapper(
            str(img_path),
            pandas.DataFrame({"A": [1.0, 2.0]}),
            str(tmp_path / "bad_bands.tif"),
        )
    with pytest.raises(rsgispy.RSGISPyException):
        su.spectral_angle_mapper(
            str(img_path),
            pandas.DataFrame({"Zero": [0.0, 0.0, 0.0]}),
            str(tmp_path / "bad_zero.tif"),
        )


