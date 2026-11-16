"""
Tests for rsgispy.classification.clustersklearn unsupervised pixel
classification (whole-image clustering and sample-fit + image-predict modes).
"""

import inspect
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.classification as rcls
import rsgispy.classification.classdata as rcd
import rsgispy.classification.classsklearn as rcsk
import rsgispy.classification.clustersklearn as rclsk
import rsgispy.imageutils.colourtable as rct


def _write_raster(
    path: Path,
    arr: np.ndarray,
    origin_x: float = 500000.0,
    origin_y: float = 5600000.0,
    pxl_size: float = 10.0,
    crs: str = "EPSG:32630",
    nodata: float = -999.0,
) -> str:
    """Write a 2D or 3D GeoTIFF raster for unsupervised clustering tests."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
        write_arr = arr[None, :, :]
    else:
        count, height, width = arr.shape
        write_arr = arr

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(arr.dtype),
        crs=crs,
        transform=from_origin(origin_x, origin_y, pxl_size, pxl_size),
        nodata=nodata,
    ) as dst:
        dst.write(write_arr)
    return str(path)


def _create_cluster_scene(tmp_path: Path):
    """Build synthetic multi-band imagery with 3 well-separated clusters."""
    h, w = 24, 24
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    b1 = np.zeros((h, w), dtype=np.float32)
    b2 = np.zeros((h, w), dtype=np.float32)
    b3 = np.zeros((h, w), dtype=np.float32)

    # Cluster zone A (top rows 0..7): low b1, high b2, low b3
    b1[0:8, :] = 10.0 + 0.1 * xx[0:8, :]
    b2[0:8, :] = 90.0 + 0.1 * yy[0:8, :]
    b3[0:8, :] = 15.0 + 0.05 * xx[0:8, :]

    # Cluster zone B (middle rows 8..15): high b1, low b2, mid b3
    b1[8:16, :] = 90.0 + 0.1 * xx[8:16, :]
    b2[8:16, :] = 10.0 + 0.1 * yy[8:16, :]
    b3[8:16, :] = 50.0 + 0.05 * xx[8:16, :]

    # Cluster zone C (bottom rows 16..23): high b1, high b2, high b3
    b1[16:24, :] = 90.0 + 0.1 * xx[16:24, :]
    b2[16:24, :] = 90.0 + 0.1 * yy[16:24, :]
    b3[16:24, :] = 95.0 + 0.05 * xx[16:24, :]

    # Inject explicit nodata pixel inside zone A.
    b1[2, 2] = -999.0

    # Valid ROI mask (outer 1px border excluded = 0, interior = 1).
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[1:-1, 1:-1] = 1

    # Coarse 20m raster (12x12) for multi-resolution ImageBandInfo test.
    coarse = np.full((12, 12), 10.0, dtype=np.float32)
    coarse[4:8, :] = 50.0
    coarse[8:12, :] = 90.0

    img_3b = _write_raster(
        tmp_path / "scene_3b.tif", np.stack([b1, b2, b3]), nodata=-999.0
    )
    img_coarse = _write_raster(
        tmp_path / "coarse.tif", coarse, pxl_size=20.0, nodata=-999.0
    )
    msk_img = _write_raster(tmp_path / "mask.tif", mask, nodata=0.0)

    return {
        "img_3b": img_3b,
        "img_coarse": img_coarse,
        "msk_img": msk_img,
        "height": h,
        "width": w,
    }


def _assert_zones_separated(arr: np.ndarray, nodata_val: int) -> None:
    """Assert the three horizontal spectral zones map cleanly to 3 clusters."""
    zone_a = arr[3:6, 4:20]
    zone_b = arr[10:14, 4:20]
    zone_c = arr[18:22, 4:20]
    u_a = set(np.unique(zone_a).tolist()) - {nodata_val}
    u_b = set(np.unique(zone_b).tolist()) - {nodata_val}
    u_c = set(np.unique(zone_c).tolist()) - {nodata_val}
    assert len(u_a) == 1
    assert len(u_b) == 1
    assert len(u_c) == 1
    assert len(u_a | u_b | u_c) == 3


def test_clustersklearn_module_layout():
    """Verify clustersklearn exports and non-re-export from package init."""
    expected = {"img_pixel_cluster", "img_pixel_sample_cluster"}
    assert expected.issubset(set(rclsk.__all__))
    for name in expected:
        assert hasattr(rclsk, name)
        assert not hasattr(rcls, name)
        assert inspect.isfunction(getattr(rclsk, name))


def test_img_pixel_cluster_whole_image_and_transductive(tmp_path):
    """Test whole-image clustering with KMeans, DBSCAN, MeanShift, and configs."""
    sk_cluster = pytest.importorskip("sklearn.cluster")
    scene = _create_cluster_scene(tmp_path)

    # 1. Whole-image clustering with ImageBandInfo + MiniBatchKMeans + model save.
    out_kmeans = tmp_path / "cluster_full_kmeans.tif"
    out_model = tmp_path / "cluster_full.joblib"
    km = sk_cluster.MiniBatchKMeans(n_clusters=3, random_state=42)
    single_band_info = [rcd.ImageBandInfo(file_name=scene["img_3b"], name="s2")]
    fitted = rclsk.img_pixel_cluster(
        single_band_info,
        out_kmeans,
        clusterer=km,
        no_data_val=-999.0,
        block_size=8,
        rnd_seed=42,
        out_model_file=out_model,
        return_model=True,
        use_pbar=False,
    )
    assert out_kmeans.exists()
    assert out_model.exists()
    assert hasattr(fitted, "cluster_centers_")
    assert hasattr(km, "cluster_centers_")

    with rasterio.open(out_kmeans, "r") as src:
        arr = src.read(1)
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        assert rct.is_img_thematic(str(out_kmeans))
        assert arr[2, 2] == 0
        _assert_zones_separated(arr, 0)

    # 2. Transductive clusterer without predict() (DBSCAN) + noise mapping +
    # out_id_offset=0 (auto nodata avoids 0 collision).
    b1_noisy = np.zeros((16, 16), dtype=np.float32)
    b2_noisy = np.zeros((16, 16), dtype=np.float32)
    b1_noisy[0:8, :] = 10.0
    b2_noisy[0:8, :] = 10.0
    b1_noisy[8:16, :] = 80.0
    b2_noisy[8:16, :] = 80.0
    b1_noisy[4, 4] = 500.0
    b2_noisy[4, 4] = 500.0
    noisy_img = _write_raster(
        tmp_path / "noisy.tif", np.stack([b1_noisy, b2_noisy]), nodata=-999.0
    )

    out_dbscan = tmp_path / "cluster_dbscan.tif"
    dbscan = sk_cluster.DBSCAN(eps=5.0, min_samples=5)
    assert not hasattr(dbscan, "predict")
    res_none = rclsk.img_pixel_cluster(
        [rcd.ImageBandInfo(file_name=noisy_img, name="noisy")],
        out_dbscan,
        clusterer=dbscan,
        out_id_offset=0,
        use_pbar=False,
    )
    assert res_none is None
    with rasterio.open(out_dbscan, "r") as src:
        db_arr = src.read(1)
        assert src.nodata == 255
        # Single far outlier at (4, 4) becomes DBSCAN -1 -> mapped to nodata.
        assert db_arr[4, 4] == 255
        assert db_arr[1, 1] != db_arr[12, 12]
        assert db_arr[1, 1] in (0, 1)
        assert db_arr[12, 12] in (0, 1)

    # 3. MeanShift auto-bandwidth + multi-image ImageBandInfo + in_msk_img.
    out_ms = tmp_path / "cluster_full_ms.tif"
    band_info = [
        rcd.ImageBandInfo(
            file_name=scene["img_3b"],
            name="s2",
            bands=[1, 2],
        ),
        rcd.ImageBandInfo(
            file_name=scene["img_coarse"],
            name="coarse",
            bands=[1],
        ),
    ]
    rclsk.img_pixel_cluster(
        band_info,
        out_ms,
        clusterer=None,
        in_msk_img=scene["msk_img"],
        img_msk_val=1,
        use_mean_shift_est_band_width=True,
        mean_shift_quantile=0.25,
        rnd_seed=42,
        use_pbar=False,
    )
    with rasterio.open(out_ms, "r") as src:
        ms_arr = src.read(1)
        assert src.nodata == 0
        # Outer border excluded by mask and nodata pixel (2, 2) stay 0.
        assert np.all(ms_arr[0, :] == 0)
        assert ms_arr[2, 2] == 0
        _assert_zones_separated(ms_arr, 0)

    # 4. ClassificationConfig input object for whole-image clustering.
    cfg = rcd.ClassificationConfig(
        img_band_info=band_info,
        class_info={
            "zone_a": rcd.ClassInfoObj(out_id=1),
            "zone_b": rcd.ClassInfoObj(out_id=2),
            "zone_c": rcd.ClassInfoObj(out_id=3),
        },
    )
    out_cfg = tmp_path / "cluster_full_cfg.tif"
    rclsk.img_pixel_cluster(
        cfg,
        out_cfg,
        clusterer=sk_cluster.KMeans(n_clusters=3, n_init=10, random_state=42),
        rnd_seed=42,
        use_pbar=False,
    )
    with rasterio.open(out_cfg, "r") as src:
        cfg_arr = src.read(1)
        assert cfg_arr[2, 2] == 0
        _assert_zones_separated(cfg_arr, 0)


def test_img_pixel_sample_cluster_subset_and_predict(tmp_path):
    """Test sample-based clustering: fit on subset and predict whole image."""
    sk_cluster = pytest.importorskip("sklearn.cluster")
    sk_pipeline = pytest.importorskip("sklearn.pipeline")
    sk_preproc = pytest.importorskip("sklearn.preprocessing")

    scene = _create_cluster_scene(tmp_path)

    # 1. Fast single-image full-band unmasked ImageBandInfo path + block streaming.
    out_smpl = tmp_path / "cluster_sample_kmeans.tif"
    out_model = tmp_path / "sample_kmeans.joblib"
    km = sk_cluster.KMeans(n_clusters=3, n_init=10, random_state=42)
    fitted = rclsk.img_pixel_sample_cluster(
        [rcd.ImageBandInfo(file_name=scene["img_3b"], name="s2")],
        out_smpl,
        clusterer=km,
        n_img_smpl=5,
        max_samples=80,
        block_size=7,
        rnd_seed=42,
        out_model_file=out_model,
        return_model=True,
        use_pbar=False,
    )
    assert out_smpl.exists()
    assert out_model.exists()
    assert hasattr(km, "cluster_centers_")
    assert hasattr(fitted, "cluster_centers_")

    reloaded = rcsk.load_sklearn_classifier(out_model)
    assert reloaded.cluster_centers_.shape == (3, 3)

    with rasterio.open(out_smpl, "r") as src:
        arr = src.read(1)
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        assert rct.is_img_thematic(str(out_smpl))
        assert arr[2, 2] == 0
        _assert_zones_separated(arr, 0)

    # 2. Pipeline (StandardScaler + MiniBatchKMeans) with band subset & mask.
    out_pipe = tmp_path / "cluster_sample_pipe.tif"
    pipe = sk_pipeline.Pipeline(
        [
            ("scaler", sk_preproc.StandardScaler()),
            ("kmeans", sk_cluster.MiniBatchKMeans(n_clusters=3, random_state=42)),
        ]
    )
    rclsk.img_pixel_sample_cluster(
        [rcd.ImageBandInfo(file_name=scene["img_3b"], name="s2", bands=[1, 2])],
        out_pipe,
        clusterer=pipe,
        n_img_smpl=4,
        in_msk_img=scene["msk_img"],
        img_msk_val=1,
        datatype=rsgispy.TYPE_16UINT,
        out_no_data_val=0,
        rnd_seed=42,
        use_pbar=False,
    )
    with rasterio.open(out_pipe, "r") as src:
        pipe_arr = src.read(1)
        assert src.dtypes[0] == "uint16"
        assert src.nodata == 0
        assert np.all(pipe_arr[0, :] == 0)
        assert pipe_arr[2, 2] == 0
        _assert_zones_separated(pipe_arr, 0)

    # 3. MeanShift auto-bandwidth on sampled multi-resolution ImageBandInfo.
    out_ms = tmp_path / "cluster_sample_ms.tif"
    band_info = [
        rcd.ImageBandInfo(file_name=scene["img_3b"], name="s2", bands=[1, 2]),
        rcd.ImageBandInfo(file_name=scene["img_coarse"], name="c", bands=[1]),
    ]
    rclsk.img_pixel_sample_cluster(
        band_info,
        out_ms,
        clusterer=None,
        n_img_smpl=4,
        use_mean_shift_est_band_width=True,
        mean_shift_quantile=0.25,
        rnd_seed=42,
        use_pbar=False,
    )
    with rasterio.open(out_ms, "r") as src:
        ms_arr = src.read(1)
        assert ms_arr[2, 2] == 0
        _assert_zones_separated(ms_arr, 0)

    # 4. Saved ClassificationConfig JSON file path input.
    cfg = rcd.ClassificationConfig(
        img_band_info=band_info,
        class_info={
            "zone_a": rcd.ClassInfoObj(out_id=1),
            "zone_b": rcd.ClassInfoObj(out_id=2),
            "zone_c": rcd.ClassInfoObj(out_id=3),
        },
    )
    cfg_json = tmp_path / "cluster_cfg.json"
    rcd.write_cls_config(cfg_json, cfg)
    out_cfg_json = tmp_path / "cluster_sample_cfg.tif"
    rclsk.img_pixel_sample_cluster(
        cfg_json,
        out_cfg_json,
        clusterer=sk_cluster.KMeans(n_clusters=3, n_init=10, random_state=42),
        n_img_smpl=4,
        rnd_seed=42,
        use_pbar=False,
    )
    with rasterio.open(out_cfg_json, "r") as src:
        cfg_arr = src.read(1)
        assert cfg_arr[2, 2] == 0
        _assert_zones_separated(cfg_arr, 0)


def test_clustersklearn_validation_errors(tmp_path):
    """Verify clear RSGISPyException validation on invalid clustering inputs."""
    sk_cluster = pytest.importorskip("sklearn.cluster")
    scene = _create_cluster_scene(tmp_path)
    out_tif = tmp_path / "err.tif"
    valid_bands = [rcd.ImageBandInfo(file_name=scene["img_3b"], bands=[1, 2])]

    # DBSCAN lacks predict(), so sample clustering must fail cleanly.
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_sample_cluster(
            valid_bands,
            out_tif,
            clusterer=sk_cluster.DBSCAN(),
            n_img_smpl=5,
            use_pbar=False,
        )

    # Negative out_id_offset rejected to keep unsigned cluster IDs.
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_cluster(
            valid_bands,
            out_tif,
            clusterer=sk_cluster.KMeans(n_clusters=3),
            out_id_offset=-1,
            use_pbar=False,
        )

    # Invalid n_img_smpl / max_samples.
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_sample_cluster(
            valid_bands,
            out_tif,
            clusterer=sk_cluster.KMeans(n_clusters=3),
            n_img_smpl=0,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_sample_cluster(
            valid_bands,
            out_tif,
            clusterer=sk_cluster.KMeans(n_clusters=3),
            n_img_smpl=5,
            max_samples=0,
            use_pbar=False,
        )

    # Invalid img_band_info inputs: empty sequence, None, missing JSON config, wrong items.
    for bad_input in ([], None, tmp_path / "missing_cfg.json", [123]):
        with pytest.raises(rsgispy.RSGISPyException):
            rclsk.img_pixel_cluster(
                bad_input,  # type: ignore[arg-type]
                out_tif,
                clusterer=sk_cluster.KMeans(n_clusters=3),
                use_pbar=False,
            )

    # Passing clusterer alongside use_mean_shift_est_band_width=True.
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_cluster(
            valid_bands,
            out_tif,
            clusterer=sk_cluster.KMeans(n_clusters=3),
            use_mean_shift_est_band_width=True,
            use_pbar=False,
        )

    # All-nodata image raises RSGISPyException.
    all_nd = _write_raster(
        tmp_path / "all_nd.tif",
        np.full((2, 6, 6), -999.0, dtype=np.float32),
        nodata=-999.0,
    )
    with pytest.raises(rsgispy.RSGISPyException):
        rclsk.img_pixel_cluster(
            [rcd.ImageBandInfo(file_name=all_nd)],
            out_tif,
            clusterer=sk_cluster.KMeans(n_clusters=2),
            use_pbar=False,
        )
