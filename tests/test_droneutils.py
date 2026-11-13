"""Tests for rsgispy.droneutils."""

import datetime
import os
import sys
import types

import geopandas
import pandas
from PIL import Image

import rsgispy.droneutils


def _create_test_jpeg(path, dt_str=None):
    img = Image.new("RGB", (16, 12), color=(10, 20, 30))
    if dt_str is not None:
        exif = img.getexif()
        # EXIF tag 0x0132 = DateTime
        exif[0x0132] = dt_str
        img.save(path, format="JPEG", exif=exif.tobytes())
    else:
        img.save(path, format="JPEG")


def test_image_time_cluster_methods():
    t0 = datetime.datetime(2024, 6, 1, 10, 0, 0)
    t1 = datetime.datetime(2024, 6, 1, 10, 0, 30)
    t2 = datetime.datetime(2024, 6, 1, 10, 5, 0)

    c1 = rsgispy.droneutils.ImageTimeCluster(0, t0)
    assert c1.within_cluster(t1, time_split_secs=60)
    assert not c1.within_cluster(t2, time_split_secs=60)

    c1.add_image_to_cluster("img1.jpg", t1)
    assert c1.end_time == t1

    c2 = rsgispy.droneutils.ImageTimeCluster(
        1, datetime.datetime(2024, 6, 1, 10, 1, 0)
    )
    assert c1.does_cluster_overlap(c2, time_split_secs=60)
    c1.merge_cluster(c2)
    assert c1.end_time == datetime.datetime(2024, 6, 1, 10, 1, 0)
    assert "0:" in str(c1)
    assert "0:" in repr(c1)


def test_split_photos_by_time(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()

    # Deliberately interleaved timestamps to exercise clustering + merging
    img_a = src_dir / "a.jpg"
    img_b = src_dir / "b.jpg"
    img_c = src_dir / "c.jpg"
    img_d = src_dir / "d.jpg"

    _create_test_jpeg(img_a, "2024:06:01 10:00:00")
    _create_test_jpeg(img_b, "2024:06:01 10:01:20")
    _create_test_jpeg(img_c, "2024:06:01 10:00:40")
    _create_test_jpeg(img_d, "2024:06:01 10:10:00")

    rsgispy.droneutils.split_photos_by_time(
        [str(img_a), str(img_b), str(img_c), str(img_d)],
        str(out_dir),
        time_split_secs=60,
    )

    cluster_dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert cluster_dirs == ["20240601_100000", "20240601_101000"]
    assert sorted(p.name for p in (out_dir / "20240601_100000").iterdir()) == [
        "a.jpg",
        "b.jpg",
        "c.jpg",
    ]
    assert sorted(p.name for p in (out_dir / "20240601_101000").iterdir()) == [
        "d.jpg",
    ]


class _FakeExifToolHelper:
    def __init__(self, metadata_by_file, recorded_writes=None):
        self._meta = metadata_by_file
        self._writes = recorded_writes if recorded_writes is not None else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def get_metadata(self, img_path):
        return [self._meta[img_path]]

    def set_tags(self, img_path, tags=None, params=None):
        self._writes.append((img_path, tags, params))


def test_create_flightline_vec(tmp_path, monkeypatch):
    if not rsgispy.droneutils.HAVE_EXIFTOOL:
        return

    img1 = str(tmp_path / "img1.jpg")
    img2 = str(tmp_path / "img2.jpg")
    _create_test_jpeg(img1)
    _create_test_jpeg(img2)

    meta = {
        img1: {
            "EXIF:DateTimeOriginal": "2024:06:01 10:00:00",
            "Composite:GPSLatitude": 52.4,
            "Composite:GPSLongitude": -4.08,
            "XMP:RelativeAltitude": 50.0,
            "XMP:AbsoluteAltitude": 120.0,
            "XMP:GimbalRollDegree": 0.0,
            "XMP:GimbalPitchDegree": -90.0,
            "XMP:GimbalYawDegree": 45.0,
            "XMP:FlightPitchDegree": 1.0,
            "XMP:FlightRollDegree": 2.0,
            "XMP:FlightYawDegree": 90.0,
            "EXIF:FocalLength": 8.8,
            "EXIF:FocalLengthIn35mmFormat": 24.0,
            "EXIF:ImageWidth": 4000,
            "EXIF:ImageHeight": 3000,
            "EXIF:MaxApertureValue": 2.8,
            "EXIF:Model": "FC6310",
            "XMP:RigCameraIndex": 0,
        },
        img2: {
            "EXIF:DateTimeOriginal": "2024:06:01 10:00:05",
            "Composite:GPSLatitude": 52.401,
            "Composite:GPSLongitude": -4.081,
            "XMP:RelativeAltitude": 51.0,
            "XMP:AbsoluteAltitude": 121.0,
            "XMP:GimbalRollDegree": 0.1,
            "XMP:GimbalPitchDegree": -89.5,
            "XMP:GimbalYawDegree": 46.0,
            "EXIF:FocalLength": 8.8,
            "EXIF:FocalLengthIn35mmFormat": 24.0,
            "EXIF:ImageWidth": 4000,
            "EXIF:ImageHeight": 3000,
            "EXIF:MaxApertureValue": 2.8,
            "EXIF:Model": "FC6310",
            "XMP:SensorIndex": 1,
        },
    }

    fake_mod = types.ModuleType("exiftool")
    fake_mod.ExifToolHelper = lambda: _FakeExifToolHelper(meta)
    monkeypatch.setattr(rsgispy.droneutils, "exiftool", fake_mod)
    monkeypatch.setattr(
        rsgispy.tools.files,
        "sort_file_by_datetime",
        lambda imgs: list(imgs),
    )

    out_gpkg = str(tmp_path / "flight.gpkg")
    rsgispy.droneutils.create_flightline_vec(
        [img2, img1],
        out_gpkg,
        "pts",
        out_gpkg,
        "lines",
        out_format="GPKG",
    )

    pts_gdf = geopandas.read_file(out_gpkg, layer="pts")
    lines_gdf = geopandas.read_file(out_gpkg, layer="lines")
    assert len(pts_gdf) == 2
    assert list(pts_gdf["image_filename"]) == ["img1.jpg", "img2.jpg"]
    assert len(lines_gdf) == 1
    assert lines_gdf.iloc[0]["ID"] == 1


def test_create_flightline_extern_gps_and_define_tags(tmp_path, monkeypatch):
    if not rsgispy.droneutils.HAVE_EXIFTOOL:
        return

    img1 = tmp_path / "p1.jpg"
    img2 = tmp_path / "p2.jpg"
    _create_test_jpeg(img1)
    _create_test_jpeg(img2)

    gps_df = pandas.DataFrame(
        [
            {
                "image": "p1.jpg",
                "lat": 52.4,
                "lon": -4.08,
                "alt": 100.0,
                "roll": 1.0,
                "pitch": 2.0,
                "yaw": 180.0,
            },
            {
                "image": "p2.jpg",
                "lat": -12.3,
                "lon": 14.2,
                "alt": 102.0,
                "roll": 1.5,
                "pitch": 2.5,
                "yaw": 181.0,
            },
            {
                "image": "missing.jpg",
                "lat": 0.0,
                "lon": 0.0,
                "alt": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        ]
    )

    meta = {
        str(img1): {
            "EXIF:DateTimeOriginal": "2024:06:01 10:00:00",
            "EXIF:FocalLength": 5.0,
            "EXIF:FocalLengthIn35mmFormat": 20.0,
            "EXIF:ImageWidth": 1280,
            "EXIF:ImageHeight": 960,
            "EXIF:MaxApertureValue": 2.0,
            "EXIF:Model": "RedEdge",
            "XMP:RigCameraIndex": 1,
        },
        str(img2): {
            "EXIF:DateTimeOriginal": "2024:06:01 10:00:02",
            "EXIF:FocalLength": 5.0,
            "EXIF:FocalLengthIn35mmFormat": 20.0,
            "EXIF:ImageWidth": 1280,
            "EXIF:ImageHeight": 960,
            "EXIF:MaxApertureValue": 2.0,
            "EXIF:Model": "RedEdge",
            "XMP:RigCameraIndex": 1,
        },
    }

    writes = []
    fake_mod = types.ModuleType("exiftool")
    fake_mod.ExifToolHelper = lambda: _FakeExifToolHelper(meta, writes)
    monkeypatch.setattr(rsgispy.droneutils, "exiftool", fake_mod)

    pts_shp = str(tmp_path / "pts.geojson")
    line_shp = str(tmp_path / "lines.geojson")
    rsgispy.droneutils.create_flightline_extern_gps_vec(
        gps_df,
        str(tmp_path),
        pts_shp,
        "pts",
        line_shp,
        "lines",
        out_format="GeoJSON",
    )

    assert os.path.exists(pts_shp)
    assert os.path.exists(line_shp)
    pts_gdf = geopandas.read_file(pts_shp)
    assert len(pts_gdf) == 2
    assert list(pts_gdf["flight_yaw_degree"]) == [180.0, 181.0]

    rsgispy.droneutils.define_extern_gps_in_imgs(
        gps_df,
        str(tmp_path),
        flying_height=60.0,
        gimbal_roll=0.0,
        gimbal_pitch=-90.0,
        gimbal_yaw=10.0,
        overwrite=True,
    )
    assert len(writes) == 2
    assert writes[0][1]["EXIF:GPSLatitudeRef"] == "N"
    assert writes[0][1]["EXIF:GPSLongitudeRef"] == "W"
    assert writes[1][1]["EXIF:GPSLatitudeRef"] == "S"
    assert writes[1][1]["EXIF:GPSLongitudeRef"] == "E"
    assert writes[0][2] == ["-P", "-overwrite_original"]


def test_optional_exiftool_missing_droneutils():
    """Verify droneutils omits exiftool functions when exiftool is missing."""
    import importlib
    from unittest.mock import patch

    orig_droneutils = rsgispy.droneutils
    try:
        with patch.dict(sys.modules, {"exiftool": None}):
            sys.modules.pop("rsgispy.droneutils", None)
            drone_mod = importlib.import_module("rsgispy.droneutils")

        assert not drone_mod.HAVE_EXIFTOOL
        for fn_name in (
            "create_flightline_vec",
            "create_flightline_extern_gps_vec",
            "define_extern_gps_in_imgs",
        ):
            assert not hasattr(drone_mod, fn_name)
            assert fn_name not in drone_mod.__all__

        for sym_name in ("ImageTimeCluster", "split_photos_by_time"):
            assert hasattr(drone_mod, sym_name)
            assert sym_name in drone_mod.__all__
    finally:
        sys.modules["rsgispy.droneutils"] = orig_droneutils
        rsgispy.droneutils = orig_droneutils
