"""
Tests for rsgispy.tools.filetools.
"""

import datetime
import hashlib
import os
import tarfile
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import rsgispy
import rsgispy.tools.filetools as filetools


def test_basename_and_path_helpers(tmp_path):
    """Test get_file_basename, get_dir_name, split_path_all, and path validity checks."""
    assert filetools.get_file_basename("/a/b/my_img_20250101.tar.gz") == "my_img_20250101"
    assert filetools.get_file_basename("/a/b/my_img_20250101.tar.gz", rm_n_exts=1) == "my_img_20250101.tar"
    assert filetools.get_file_basename("/a/b/a! b@c-d_e.tif", check_valid=True) == "a_bc-d_e"
    assert filetools.get_file_basename("/a/b/scene_tile1_b2.tif", n_comps=2) == "scene_tile1"

    with pytest.raises(rsgispy.RSGISPyException):
        filetools.get_file_basename("/a/b/one_two.tif", n_comps=5)

    sub_dir = tmp_path / "sub_folder"
    sub_dir.mkdir()
    sample_file = sub_dir / "data.txt"
    sample_file.write_text("hello", encoding="utf-8")

    assert filetools.get_dir_name(str(sample_file)) == "sub_folder"
    assert filetools.get_dir_name(str(sub_dir)) == "sub_folder"
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.get_dir_name(str(tmp_path / "missing"))

    parts = filetools.split_path_all(str(sample_file))
    assert parts[-2:] == ["sub_folder", "data.txt"]

    assert filetools.is_path_valid(str(sample_file))
    assert not filetools.is_path_valid("")
    assert not filetools.is_path_valid("bad\x00path")
    assert filetools.is_path_sibling_creatable(str(sub_dir / "new.txt"))
    assert not filetools.is_path_sibling_creatable(str(tmp_path / "no_dir" / "new.txt"))
    assert filetools.does_path_exists_or_creatable(str(sample_file))
    assert filetools.does_path_exists_or_creatable(str(sub_dir / "future.txt"))


def test_delete_and_find_files(tmp_path):
    """Test delete_* and find_* functions and rm_files_size_gt."""
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    sub = d1 / "nested"
    sub.mkdir(parents=True)
    d2.mkdir()

    f1 = d1 / "scene_N15.tif"
    f1_hdr = d1 / "scene_N15.hdr"
    f2 = sub / "tile_01.tif"
    f3 = d2 / "extra.tif"

    f1.write_bytes(b"a" * 200)
    f1_hdr.write_bytes(b"hdr")
    f2.write_bytes(b"b" * 50)
    f3.write_bytes(b"c" * 300)

    assert filetools.find_file(str(d1), "*N15*.tif") == str(f1)
    assert filetools.find_file_none(str(d1), "*N15*.tif") == str(f1)
    assert filetools.find_file_none(str(d1), "*missing*.tif") is None
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.find_file(str(d1), "*missing*.tif")

    ext_dict = filetools.find_files_ext(str(d1), ".tif")
    assert set(ext_dict.keys()) == {"scene_N15.tif", "tile_01.tif"}

    multi_dict = filetools.find_files_mpaths_ext([str(d1), str(d2)], ".tif")
    assert set(multi_dict.keys()) == {"scene_N15.tif", "tile_01.tif", "extra.tif"}

    assert filetools.find_first_file(str(d1), "tile_*.tif") == str(f2)
    assert filetools.find_first_file(str(d1), "none_*.tif", rtn_except=False) is None
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.find_first_file(str(d1), "none_*.tif", rtn_except=True)

    sized = filetools.find_files_size_limits(str(d1), "*.tif", min_size=100, max_size=250)
    assert sized == [str(f1)]

    # rm_files_size_gt dry run (prints) then actual delete with basename
    filetools.rm_files_size_gt(str(d1), "*.tif", min_size=100, rm_file=False)
    assert f1.exists() and f1_hdr.exists()

    filetools.rm_files_size_gt(str(d1), "*.tif", min_size=100, rm_file=True, rm_use_basename=True)
    assert not f1.exists() and not f1_hdr.exists()
    assert filetools.delete_file_silent(str(f1))  # non-existent returns True


def test_file_sizes_mod_times_and_dirs(tmp_path):
    """Test size conversion, file size, mod times, hidden check, and dir listing."""
    assert filetools.convert_file_size_units(1, "kb", "bytes") == pytest.approx(1024.0)
    assert filetools.convert_file_size_units(1024**2, "bytes", "mb") == pytest.approx(1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.convert_file_size_units(10, "bad", "bytes")

    f_old = tmp_path / "old.txt"
    f_new = tmp_path / "new.txt"
    f_old.write_bytes(b"0" * 1024)
    f_new.write_bytes(b"1" * 2048)
    os.utime(f_old, (1_600_000_000, 1_600_000_000))
    os.utime(f_new, (1_750_000_000, 1_750_000_000))

    assert filetools.get_file_size(str(f_old), "kb") == pytest.approx(1.0)
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.get_file_size(str(tmp_path / "missing.txt"))

    mod_dt = filetools.get_file_mod_time(str(f_old))
    assert mod_dt.year == 2020

    cutoff = datetime.datetime.fromtimestamp(1_700_000_000)
    before_res = filetools.get_files_mod_time([str(f_old), str(f_new), str(tmp_path / "nope")], dt_before=cutoff)
    assert before_res == [str(f_old)]
    after_res = filetools.get_files_mod_time([str(f_old), str(f_new)], dt_after=cutoff)
    assert after_res == [str(f_new)]
    window_res = filetools.get_files_mod_time(
        [str(f_old), str(f_new)],
        dt_after=datetime.datetime.fromtimestamp(1_500_000_000),
        dt_before=datetime.datetime.fromtimestamp(1_800_000_000),
    )
    assert window_res == [str(f_old), str(f_new)]
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.get_files_mod_time([str(f_old)])

    # Directories & hidden
    vis1 = tmp_path / "alpha_data"
    vis2 = tmp_path / "beta_dir"
    hid = tmp_path / ".hidden_dir"
    vis1.mkdir()
    vis2.mkdir()
    hid.mkdir()

    assert filetools.file_is_hidden(str(hid))
    assert not filetools.file_is_hidden(str(vis1))
    assert set(filetools.get_dir_list(str(tmp_path))) == {str(vis1), str(vis2)}
    assert set(filetools.get_dir_list(str(tmp_path), inc_hidden=True)) == {str(vis1), str(vis2), str(hid)}
    assert filetools.get_dir_list_srch(str(tmp_path), "alpha") == [str(vis1)]


def test_file_locks_and_sorting(tmp_path):
    """Test get_file_lock, release_file_lock, clean_file_locks, and sorting functions."""
    target = tmp_path / "work.txt"
    target.write_text("data", encoding="utf-8")

    assert filetools.get_file_lock(str(target), sleep_period=0.01, wait_iters=1)
    assert not filetools.get_file_lock(str(target), sleep_period=0.01, wait_iters=1, use_except=False)
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.get_file_lock(str(target), sleep_period=0.01, wait_iters=1, use_except=True)

    filetools.release_file_lock(str(target))
    assert filetools.get_file_lock(str(target), sleep_period=0.01, wait_iters=1)

    # Backdate lock timestamp so clean_file_locks cleans it up
    lock_path = tmp_path / ".work.txt.lok"
    old_ts = (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    lock_path.write_text(f"{old_ts}\n", encoding="utf-8")
    filetools.clean_file_locks(str(tmp_path), timeout=60)
    assert not lock_path.exists()

    # Natural sort & sort by datetime
    files = [str(tmp_path / "img10.tif"), str(tmp_path / "img2.tif"), str(tmp_path / "img1.tif")]
    assert filetools.natural_sort_file_names(list(files)) == [
        str(tmp_path / "img1.tif"),
        str(tmp_path / "img2.tif"),
        str(tmp_path / "img10.tif"),
    ]

    f_a = tmp_path / "a.txt"
    f_b = tmp_path / "b.txt"
    f_a.write_text("a", encoding="utf-8")
    f_b.write_text("b", encoding="utf-8")
    os.utime(f_a, (1_700_000_010, 1_700_000_010))
    os.utime(f_b, (1_700_000_001, 1_700_000_001))
    assert filetools.sort_file_by_datetime([str(f_a), str(f_b)]) == [str(f_b), str(f_a)]


def test_sort_imgs_to_dirs_utm(tmp_path):
    """Test sort_imgs_to_dirs_utm using rasterio GeoTIFFs across UTM zones and non-UTM."""
    in_dir = tmp_path / "imgs"
    out_dir = tmp_path / "sorted"
    in_dir.mkdir()
    out_dir.mkdir()

    specs = [
        ("utm5n.tif", "EPSG:32605"),
        ("utm30n.tif", "EPSG:32630"),
        ("utm18s.tif", "EPSG:32718"),
        ("wgs84.tif", "EPSG:4326"),
    ]
    for fname, crs in specs:
        with rasterio.open(
            in_dir / fname,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=from_origin(500000.0, 5200000.0, 10.0, 10.0),
        ) as dst:
            dst.write(np.zeros((1, 4, 4), dtype=np.uint8))

    filetools.sort_imgs_to_dirs_utm(str(in_dir), "*.tif", str(out_dir))

    assert (out_dir / "utm05N" / "utm5n.tif").exists()
    assert (out_dir / "utm30N" / "utm30n.tif").exists()
    assert (out_dir / "utm18S" / "utm18s.tif").exists()
    # Non-UTM file stays in in_dir
    assert (in_dir / "wgs84.tif").exists()


def test_hashes(tmp_path):
    """Test all 12 hash functions against Python's hashlib."""
    payload = b"rsgispy-hash-test-payload" * 200
    f_path = tmp_path / "payload.bin"
    f_path.write_bytes(payload)

    pairs = [
        (filetools.create_sha1_hash, "sha1"),
        (filetools.create_sha224_hash, "sha224"),
        (filetools.create_sha256_hash, "sha256"),
        (filetools.create_sha384_hash, "sha384"),
        (filetools.create_sha512_hash, "sha512"),
        (filetools.create_md5_hash, "md5"),
        (filetools.create_blake2b_hash, "blake2b"),
        (filetools.create_blake2s_hash, "blake2s"),
        (filetools.create_sha3_224_hash, "sha3_224"),
        (filetools.create_sha3_256_hash, "sha3_256"),
        (filetools.create_sha3_384_hash, "sha3_384"),
        (filetools.create_sha3_512_hash, "sha3_512"),
    ]
    for fn, algo in pairs:
        expected = hashlib.new(algo, payload).hexdigest()
        assert fn(str(f_path), block_size=128) == expected


def test_archives_and_cmd_tool(tmp_path):
    """Test is_cmd_tool_avail, create_targz_arch, create_directory_archive, and untar/unzip helpers."""
    assert filetools.is_cmd_tool_avail("tar", ["tar", "--version"])
    assert not filetools.is_cmd_tool_avail("definitely_nonexistent_tool_xyz_123")
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.is_cmd_tool_avail("tar", "tar --version")  # type: ignore[arg-type]

    src_dir = tmp_path / "pkg"
    src_dir.mkdir()
    sample = src_dir / "hello.txt"
    sample.write_text("archive content", encoding="utf-8")

    # create_targz_arch + untar_gz_file
    tgz_path = tmp_path / "bundle.tar.gz"
    filetools.create_targz_arch(str(tgz_path), [str(sample)], base_path=str(src_dir))
    assert tgz_path.exists()

    out_tgz = tmp_path / "out_tgz"
    out_tgz.mkdir()
    extracted_tgz = filetools.untar_gz_file(str(tgz_path), str(out_tgz), gen_arch_dir=True)
    assert (out_tgz / "bundle" / "hello.txt").read_text(encoding="utf-8") == "archive content"
    assert extracted_tgz == str(out_tgz / "bundle")

    # .tar + untar_file
    tar_path = tmp_path / "plain.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(sample, arcname="hello.txt")
    out_tar = tmp_path / "out_tar"
    out_tar.mkdir()
    filetools.untar_file(str(tar_path), str(out_tar), gen_arch_dir=False)
    assert (out_tar / "hello.txt").exists()

    # .tar.bz2 + untar_bz_file
    tbz_path = tmp_path / "comp.tar.bz2"
    with tarfile.open(tbz_path, "w:bz2") as tf:
        tf.add(sample, arcname="hello.txt")
    out_bz = tmp_path / "out_bz"
    out_bz.mkdir()
    filetools.untar_bz_file(str(tbz_path), str(out_bz), gen_arch_dir=True)
    assert (out_bz / "comp" / "hello.txt").exists()

    # .zip + unzip_file (if unzip available, create via zipfile and extract)
    if filetools.is_cmd_tool_avail("unzip"):
        zip_path = tmp_path / "comp.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(sample, arcname="hello.txt")
        out_zip = tmp_path / "out_zip"
        out_zip.mkdir()
        filetools.unzip_file(str(zip_path), str(out_zip), gen_arch_dir=True)
        assert (out_zip / "comp" / "hello.txt").exists()

    # create_directory_archive with gztar
    arch_out = filetools.create_directory_archive(str(src_dir), str(tmp_path / "dir_arch"), "gztar")
    assert arch_out == str(tmp_path / "dir_arch.tar.gz")
    assert os.path.exists(arch_out)
    with pytest.raises(rsgispy.RSGISPyException):
        filetools.create_directory_archive(str(src_dir), str(tmp_path / "bad"), "unknown_fmt")
