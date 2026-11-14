"""
Comprehensive GeoParquet and ``out_compress`` tests across all vector-capable
modules in :mod:`rsgispy`.
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-lines

import os
import types

import geopandas
import numpy
import pyarrow.parquet as pq
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Point, box

import rsgispy
import rsgispy.dataaccess.openstreetmap as osm
import rsgispy.droneutils
import rsgispy.elevation.roughness as rrough
import rsgispy.imagecalc.calc_pt_win_smpls as rcptwin
import rsgispy.imageutils.imagelut as iu_lut
import rsgispy.imageutils.rasterise as iu_ras
import rsgispy.imageutils.spatial as iu_spat
import rsgispy.imageutils.subsetting as iu_sub
import rsgispy.imageutils.warp as iu_warp
import rsgispy.tools.checkdatasets as chk
import rsgispy.vectorattrs.columns as va_cols
import rsgispy.vectorattrs.io as va_io
import rsgispy.vectorattrs.spatial as va_spatial
import rsgispy.vectorgeoms.create as vg_create
import rsgispy.vectorgeoms.lines as vg_lines
import rsgispy.vectorgeoms.overlay as vg_overlay
import rsgispy.vectorgeoms.polygons as vg_polys
import rsgispy.vectorgeoms.spatial as vg_spatial
import rsgispy.vectorprocess as vp
import rsgispy.vectorstats.metrics as vs_metrics
import rsgispy.vectorutils.io as vu_io
import rsgispy.vectorutils.metadata as vu_meta
import rsgispy.vectorutils.selection as vu_sel
import rsgispy.zonalstats.point as zs_pt
import rsgispy.zonalstats.polygon as zs_poly


def _parquet_codec(path: str) -> str:
    """Return the Arrow Parquet compression codec string for column 0."""
    meta = pq.read_metadata(path)
    return str(meta.row_group(0).column(0).compression)


def _write_test_raster(
    path: str,
    arr: numpy.ndarray,
    epsg: int = 32630,
    tl_x: float = 500000.0,
    tl_y: float = 5600100.0,
    res: float = 10.0,
    nodata=None,
) -> str:
    """Write a single-band GeoTIFF for GeoParquet integration tests."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data = numpy.asarray(arr)
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=f"EPSG:{epsg}",
        transform=from_origin(tl_x, tl_y, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _sample_polys(epsg: int = 32630) -> geopandas.GeoDataFrame:
    """Return a three-polygon sample GeoDataFrame."""
    return geopandas.GeoDataFrame(
        {"cls": ["A", "B", "A"], "val": [10.0, 25.0, 40.0]},
        geometry=[
            box(500000.0, 5600000.0, 500040.0, 5600040.0),
            box(500040.0, 5600000.0, 500080.0, 5600040.0),
            box(500080.0, 5600040.0, 500120.0, 5600080.0),
        ],
        crs=f"EPSG:{epsg}",
    )


def test_vectorprocess_geoparquet_and_compression(tmp_path):
    """Test vectorprocess detection, compression codecs, GeoParquet 1.0/1.1 info, and bbox."""
    assert vp.is_geoparquet_file("polys.parquet") is True
    assert vp.is_geoparquet_file("polys.geoparquet") is True
    assert vp.is_geoparquet_file("polys.pq") is True
    assert vp.is_geoparquet_file("polys.parquet.gzip") is True
    assert vp.is_geoparquet_file("polys.parquet.sz") is True
    assert vp.is_geoparquet_file("polys.parquet.zst") is True
    assert vp.is_geoparquet_file("polys.gpkg") is False

    assert vp.is_geoparquet_format("PARQUET") is True
    assert vp.is_geoparquet_format("GeoParquet") is True
    assert vp.is_geoparquet_format("GPKG") is False
    assert vp.is_geoparquet_format(None) is False
    assert vp.check_format_name("PARQUET") == "Parquet"
    assert vp.get_file_vec_extension("Parquet") == "parquet"
    assert vp.get_file_vec_extension("GeoParquet") == "parquet"

    assert vp.infer_parquet_compression("a.parquet", None) is None
    assert vp.infer_parquet_compression("a.parquet.gzip", None) == "gzip"
    assert vp.infer_parquet_compression("a.parquet.gz", None) == "gzip"
    assert vp.infer_parquet_compression("a.parquet.sz", None) == "snappy"
    assert vp.infer_parquet_compression("a.parquet.zst", None) == "zstd"
    assert vp.infer_parquet_compression("a.parquet.br", None) == "brotli"
    assert vp.infer_parquet_compression("a.parquet.lz4", None) == "lz4"
    assert vp.infer_parquet_compression("a.parquet", False) is None
    assert vp.infer_parquet_compression("a.parquet", "none") is None
    assert vp.infer_parquet_compression("a.parquet", "uncompressed") is None
    assert vp.infer_parquet_compression("a.parquet", True) == "snappy"
    assert vp.infer_parquet_compression("a.parquet", "ZST") == "zstd"
    assert vp.infer_parquet_compression("a.parquet", "gz") == "gzip"
    assert vp.infer_parquet_compression("a.parquet", "sz") == "snappy"
    assert vp.infer_parquet_compression("a.parquet", "br") == "brotli"
    with pytest.raises(rsgispy.RSGISPyException):
        vp.infer_parquet_compression("a.parquet", "bad_codec")
    with pytest.raises(rsgispy.RSGISPyException):
        vp.infer_parquet_compression("a.parquet", 123)

    gdf = _sample_polys()
    # Test all compression variants when writing via write_vec_gdf
    cases = [
        ("plain.parquet", None, "UNCOMPRESSED"),
        ("off.parquet", False, "UNCOMPRESSED"),
        ("none.parquet", "none", "UNCOMPRESSED"),
        ("bool_true.parquet", True, "SNAPPY"),
        ("explicit_zstd.parquet", "zstd", "ZSTD"),
        ("alias_gz.parquet", "gz", "GZIP"),
        ("alias_sz.parquet", "sz", "SNAPPY"),
        ("alias_zst.parquet", "zst", "ZSTD"),
        ("alias_br.parquet", "br", "BROTLI"),
        ("explicit_lz4.parquet", "lz4", "LZ4"),
        ("ext_inf_gzip.parquet.gzip", None, "GZIP"),
        ("ext_inf_zst.parquet.zst", None, "ZSTD"),
        ("ext_inf_sz.parquet.sz", None, "SNAPPY"),
    ]
    for fname, comp_arg, expected_codec in cases:
        out_p = str(tmp_path / fname)
        vp.write_vec_gdf(gdf, out_p, out_format="PARQUET", out_compress=comp_arg)
        assert _parquet_codec(out_p) == expected_codec

    # GeoParquet 1.0 (default) read_vec_info & bbox filtering fallback
    pq10 = str(tmp_path / "plain.parquet")
    assert vp.list_vec_layers(pq10) == ["plain"]
    info10 = vp.read_vec_info(pq10)
    assert info10["driver"] == "Parquet"
    assert info10["crs"] == "EPSG:32630"
    assert info10["features"] == 3
    assert list(info10["fields"]) == ["cls", "val"]

    with pytest.raises(rsgispy.RSGISPyException):
        vp.read_vec_gdf(pq10, vec_lyr="wrong_layer")

    hit10 = vp.read_vec_gdf(pq10, bbox=(500005.0, 5600005.0, 500035.0, 5600035.0))
    assert len(hit10) == 1
    assert hit10.iloc[0]["cls"] == "A"

    miss10 = vp.read_vec_gdf(pq10, bbox=(501000.0, 5601000.0, 501100.0, 5601100.0))
    assert len(miss10) == 0

    # GeoParquet 1.1 with write_covering_bbox=True: covering bbox col excluded from fields
    pq11 = str(tmp_path / "gp11.geoparquet")
    gdf.to_parquet(pq11, write_covering_bbox=True, compression="zstd")
    info11 = vp.read_vec_info(pq11)
    assert list(info11["fields"]) == ["cls", "val"]
    hit11 = vp.read_vec_gdf(pq11, bbox=(500045.0, 5600005.0, 500075.0, 5600035.0))
    assert len(hit11) == 1
    assert hit11.iloc[0]["cls"] == "B"


def test_vectorutils_geoparquet(tmp_path):
    """Test vectorutils metadata, io, and selection with GeoParquet & compression."""
    gdf = _sample_polys()
    pq_in = str(tmp_path / "polys.parquet.zst")
    vp.write_vec_gdf(gdf, pq_in)

    assert vu_meta.check_format_name("GeoParquet") == "Parquet"
    assert vu_meta.get_file_vec_extension("PARQUET") == "parquet"
    assert vu_meta.get_vec_lyrs_lst(pq_in) == ["polys"]
    assert vu_meta.get_epsg_proj_from_vec(pq_in) == 32630
    assert "UTM" in vu_meta.get_wkt_proj_from_vec(pq_in)
    assert vu_meta.get_vec_lyr_bbox(pq_in) == pytest.approx(
        (500000.0, 500120.0, 5600000.0, 5600080.0)
    )
    assert vu_meta.get_vec_feat_count(pq_in) == 3
    assert vu_meta.get_vec_lyr_geom_type(pq_in) == rsgispy.GEOM_POLY
    assert vu_meta.get_vec_lyr_cols(pq_in) == ["cls", "val"]

    # copy & vector_translate (including append onto existing GeoParquet)
    pq_copy = str(tmp_path / "copy.parquet")
    vu_io.create_copy_vec_lyr(
        pq_in, None, pq_copy, out_format="PARQUET", replace=True, out_compress="gzip"
    )
    assert _parquet_codec(pq_copy) == "GZIP"
    assert vu_meta.get_vec_feat_count(pq_copy) == 3

    vu_io.vector_translate(
        pq_in,
        None,
        pq_copy,
        out_format="PARQUET",
        access_mode="append",
        out_compress="zstd",
    )
    assert _parquet_codec(pq_copy) == "ZSTD"
    assert vu_meta.get_vec_feat_count(pq_copy) == 6

    # Reprojection & merging
    pq_wgs = str(tmp_path / "wgs.parquet")
    vu_io.reproj_vec_lyr(
        pq_in, pq_wgs, epsg_code=4326, out_format="Parquet", out_compress="snappy"
    )
    assert vu_meta.get_epsg_proj_from_vec(pq_wgs) == 4326
    assert _parquet_codec(pq_wgs) == "SNAPPY"

    pq_merged = str(tmp_path / "merged.parquet")
    vu_io.merge_vec_files(
        [pq_in, pq_copy],
        pq_merged,
        out_format="PARQUET",
        use_pbar=False,
        out_compress="zstd",
    )
    assert vu_meta.get_vec_feat_count(pq_merged) == 9
    assert _parquet_codec(pq_merged) == "ZSTD"

    # Multi-layer guard for GeoParquet
    with pytest.raises(rsgispy.RSGISPyException):
        vu_io.merge_to_multi_lyr_vec(
            [pq_in, pq_copy],
            str(tmp_path / "multi_fail.parquet"),
            out_format="PARQUET",
            use_pbar=False,
        )

    # Split to GeoParquet chunks
    split_dir = str(tmp_path / "pq_splits")
    chunks = vu_io.split_vec_lyr(
        pq_copy,
        None,
        2,
        "Parquet",
        split_dir,
        "chunk_",
        "parquet",
        use_pbar=False,
        out_compress="gzip",
    )
    assert len(chunks) == 3
    assert all(_parquet_codec(c) == "GZIP" for c in chunks)

    # Selection utilities with GeoParquet
    pq_sub = str(tmp_path / "sub_cls.parquet")
    vu_sel.subset_by_attribute(
        pq_in,
        None,
        "cls",
        ["A"],
        pq_sub,
        out_format="PARQUET",
        out_compress="zstd",
    )
    assert vu_meta.get_vec_feat_count(pq_sub) == 2
    assert _parquet_codec(pq_sub) == "ZSTD"

    pq_bbox = str(tmp_path / "bbox_sel.parquet")
    assert vu_sel.spatial_select_bbox(
        pq_in,
        None,
        (500005.0, 500035.0, 5600005.0, 5600035.0),
        pq_bbox,
        out_format="PARQUET",
        out_compress="gzip",
    )
    assert vu_meta.get_vec_feat_count(pq_bbox) == 1
    assert _parquet_codec(pq_bbox) == "GZIP"

    pq_rsub = str(tmp_path / "rsub.parquet")
    vu_sel.vec_lyr_random_subset(
        pq_in,
        None,
        pq_rsub,
        None,
        n_smpl=2,
        out_format="PARQUET",
        rnd_seed=7,
        out_compress="snappy",
    )
    assert vu_meta.get_vec_feat_count(pq_rsub) == 2
    assert _parquet_codec(pq_rsub) == "SNAPPY"


def test_vectorgeoms_geoparquet(tmp_path):
    """Test vectorgeoms (overlay, spatial, polygons, lines, create) with GeoParquet."""
    pq_a = str(tmp_path / "a.parquet")
    pq_b = str(tmp_path / "b.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"cls": ["A", "B"]},
            geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
            crs="EPSG:32630",
        ),
        pq_a,
    )
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"roi": ["R1"]},
            geometry=[box(5.0, 0.0, 15.0, 10.0)],
            crs="EPSG:32630",
        ),
        pq_b,
    )

    pq_clip = str(tmp_path / "clip.parquet")
    vg_overlay.clip_vec_lyr(
        pq_a, None, pq_b, None, pq_clip, out_format="PARQUET", out_compress="zstd"
    )
    assert _parquet_codec(pq_clip) == "ZSTD"
    assert geopandas.read_parquet(pq_clip).area.sum() == pytest.approx(100.0)

    pq_inter = str(tmp_path / "inter.parquet")
    vg_overlay.vec_lyr_intersection_gp(
        pq_a, None, pq_b, None, pq_inter, out_format="PARQUET", out_compress="gzip"
    )
    assert _parquet_codec(pq_inter) == "GZIP"
    assert geopandas.read_parquet(pq_inter).area.sum() == pytest.approx(100.0)

    # Polygons + explode_vec_files guard
    pq_multi = str(tmp_path / "multi.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=[MultiPolygon([box(0, 0, 2, 2), box(4, 4, 6, 6)])],
            crs="EPSG:32630",
        ),
        pq_multi,
    )
    pq_exp_dir = str(tmp_path / "exp_pq")
    vg_polys.explode_vec_files(
        [pq_multi],
        pq_exp_dir,
        out_format="PARQUET",
        out_vec_ext="parquet",
        use_pbar=False,
        out_compress="zstd",
    )
    exp_out = os.path.join(pq_exp_dir, "multi.parquet")
    assert _parquet_codec(exp_out) == "ZSTD"
    assert len(geopandas.read_parquet(exp_out)) == 2

    gpkg_multi_lyr = str(tmp_path / "two_lyrs.gpkg")
    geopandas.GeoDataFrame(
        {"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:32630"
    ).to_file(gpkg_multi_lyr, layer="l1", driver="GPKG")
    geopandas.GeoDataFrame(
        {"id": [2]}, geometry=[Point(1, 1)], crs="EPSG:32630"
    ).to_file(gpkg_multi_lyr, layer="l2", driver="GPKG")
    with pytest.raises(rsgispy.RSGISPyException):
        vg_polys.explode_vec_files(
            [gpkg_multi_lyr], pq_exp_dir, out_format="PARQUET", use_pbar=False
        )

    pq_buf = str(tmp_path / "buf.parquet")
    vg_polys.buffer_vec_layer_gp(
        pq_a, None, 2.0, pq_buf, out_format="PARQUET", out_compress="snappy"
    )
    assert _parquet_codec(pq_buf) == "SNAPPY"
    assert geopandas.read_parquet(pq_buf).area.iloc[0] > 100.0

    # Spatial: alpha shape
    pts_pq = str(tmp_path / "hull_pts.parquet")
    vg_create.write_pts_to_vec(
        pts_pq,
        "pts",
        "PARQUET",
        32630,
        [0.0, 10.0, 10.0, 0.0, 5.0],
        [0.0, 0.0, 10.0, 10.0, 5.0],
        out_compress="zstd",
    )
    assert _parquet_codec(pts_pq) == "ZSTD"
    assert vg_spatial.bbox_intersects_vec_lyr(pts_pq, None, (2.0, 8.0, 2.0, 8.0)) is True

    hull_pq = str(tmp_path / "hull.parquet")
    created, _ = vg_spatial.create_alpha_shape(
        pts_pq,
        None,
        hull_pq,
        None,
        out_format="PARQUET",
        alpha_vals=[0.5, 1.0],
        out_compress="gzip",
    )
    assert created is True
    assert _parquet_codec(hull_pq) == "GZIP"

    # Lines
    lines_pq = str(tmp_path / "lines.parquet")
    vg_lines.create_lines_vec(
        lines_pq,
        "lines",
        "PARQUET",
        32630,
        [[(0.0, 0.0), (100.0, 0.0)]],
        out_compress="zstd",
    )
    assert _parquet_codec(lines_pq) == "ZSTD"
    orthg_pq = str(tmp_path / "orthg.parquet")
    vg_lines.create_orthg_lines(
        lines_pq,
        None,
        orthg_pq,
        None,
        pt_step=25.0,
        line_len=10.0,
        out_format="PARQUET",
        use_pbar=False,
        out_compress="gzip",
    )
    assert _parquet_codec(orthg_pq) == "GZIP"
    assert len(geopandas.read_parquet(orthg_pq)) == 3

    # Create: polygonise + existing file guard + define_grid multi-zone guard
    cls_tif = str(tmp_path / "cls.tif")
    arr = numpy.zeros((8, 8), dtype=numpy.uint8)
    arr[1:4, 1:4] = 1
    _write_test_raster(cls_tif, arr, nodata=0)

    poly_pq = str(tmp_path / "poly_ras.parquet")
    vg_create.polygonise_raster_to_vec_lyr(
        poly_pq, "cls", "PARQUET", cls_tif, mask_img=cls_tif, out_compress="zstd"
    )
    assert _parquet_codec(poly_pq) == "ZSTD"
    assert len(geopandas.read_parquet(poly_pq)) == 1

    with pytest.raises(rsgispy.RSGISPyException):
        vg_create.polygonise_raster_to_vec_lyr(
            poly_pq,
            "cls2",
            "PARQUET",
            cls_tif,
            replace_file=False,
            replace_lyr=False,
        )

    grid_pq = str(tmp_path / "grid.parquet")
    vg_create.define_grid(
        (500000.0, 500200.0, 5600000.0, 5600200.0),
        100.0,
        100.0,
        32630,
        grid_pq,
        "grid",
        out_format="PARQUET",
        out_compress="gzip",
    )
    assert _parquet_codec(grid_pq) == "GZIP"
    assert len(geopandas.read_parquet(grid_pq)) == 4

    with pytest.raises(rsgispy.RSGISPyException):
        vg_create.define_grid(
            (-0.5, 0.5, 52.0, 52.5),
            20000.0,
            20000.0,
            4326,
            str(tmp_path / "utm_grid_fail.parquet"),
            "tiles",
            out_format="PARQUET",
            utm_grid=True,
        )


def test_vectorattrs_vectorstats_zonalstats_geoparquet(tmp_path):
    """Test vectorattrs, vectorstats, and zonalstats reading/writing compressed GeoParquet."""
    polys_pq = str(tmp_path / "polys.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {
                "cls": ["Forest", "Water", "Forest"],
                "val": [10.0, 20.0, 30.0],
                "yr": [2024, 2025, 2026],
                "mo": [1, 6, 12],
                "dy": [5, 10, 15],
                "raw": ["A-1", "B 2!", "C 3"],
            },
            geometry=[
                box(500000.0, 5600060.0, 500020.0, 5600080.0),
                box(500020.0, 5600060.0, 500040.0, 5600080.0),
                box(500000.0, 5600040.0, 500020.0, 5600060.0),
            ],
            crs="EPSG:32630",
        ),
        polys_pq,
        out_compress="snappy",
    )
    assert _parquet_codec(polys_pq) == "SNAPPY"

    # vectorattrs.io in-place & copy operations + export_vec_attrs_to_parquet out_compress
    va_io.write_vec_column(
        polys_pq, None, "score", float, [1.0, 2.0, 3.0], out_compress="snappy"
    )
    assert _parquet_codec(polys_pq) == "SNAPPY"
    assert va_io.read_vec_column(polys_pq, None, "score") == [1.0, 2.0, 3.0]

    drop_pq = str(tmp_path / "drop.parquet")
    va_io.drop_vec_cols(
        polys_pq, None, ["raw"], drop_pq, out_format="PARQUET", out_compress="zstd"
    )
    assert _parquet_codec(drop_pq) == "ZSTD"
    assert "raw" not in vu_meta.get_vec_lyr_cols(drop_pq)

    sort_pq = str(tmp_path / "sort.parquet")
    va_io.sort_vec_lyr(
        drop_pq, None, sort_pq, None, "val", False, out_format="PARQUET", out_compress="gzip"
    )
    assert _parquet_codec(sort_pq) == "GZIP"
    assert va_io.read_vec_column(sort_pq, None, "val") == [30.0, 20.0, 10.0]

    attr_pq = str(tmp_path / "attrs_zstd.parquet")
    va_io.export_vec_attrs_to_parquet(
        polys_pq, None, attr_pq, gzip_output=True, out_compress="zstd"
    )
    assert _parquet_codec(attr_pq) == "ZSTD"

    # vectorattrs.columns & spatial
    fid_pq = str(tmp_path / "fid.parquet")
    va_cols.add_fid_col(
        polys_pq, None, fid_pq, None, out_format="PARQUET", out_compress="zstd"
    )
    assert _parquet_codec(fid_pq) == "ZSTD"
    assert va_io.read_vec_column(fid_pq, None, "fid") == [1, 2, 3]

    str_pq = str(tmp_path / "str.parquet")
    va_cols.add_string_col(
        fid_pq,
        None,
        "tag",
        str_pq,
        None,
        out_val="ok",
        out_format="PARQUET",
        out_compress="gzip",
    )
    assert _parquet_codec(str_pq) == "GZIP"
    assert va_io.read_vec_column(str_pq, None, "tag") == ["ok", "ok", "ok"]

    bbox_pq = str(tmp_path / "bbox.parquet")
    va_spatial.add_geom_bbox_cols(
        str_pq, None, bbox_pq, None, out_format="PARQUET", out_compress="zstd"
    )
    assert _parquet_codec(bbox_pq) == "ZSTD"
    assert "MinX" in vu_meta.get_vec_lyr_cols(bbox_pq)

    va_spatial.pop_bbox_cols(bbox_pq, None, out_compress="gzip")
    assert _parquet_codec(bbox_pq) == "GZIP"
    assert "xmin" in vu_meta.get_vec_lyr_cols(bbox_pq)

    # vectorstats.metrics
    area_pq = str(tmp_path / "area.parquet")
    vs_metrics.calc_vec_area(
        polys_pq, None, "area_m2", area_pq, None, out_format="PARQUET", out_compress="zstd"
    )
    assert _parquet_codec(area_pq) == "ZSTD"
    assert va_io.read_vec_column(area_pq, None, "area_m2") == [400.0, 400.0, 400.0]
    assert vs_metrics.count_feats_per_att_val(polys_pq, None, "cls") == {
        "Forest": 2,
        "Water": 1,
    }

    # zonalstats.polygon & point in-place GeoParquet updates with out_compress
    ras_tif = str(tmp_path / "zonal.tif")
    arr = numpy.arange(1, 65, dtype=numpy.float32).reshape(8, 8)
    _write_test_raster(ras_tif, arr, tl_x=500000.0, tl_y=5600080.0, res=10.0)

    zs_poly.calc_zonal_band_stats_file(
        polys_pq,
        None,
        ras_tif,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        mean_field="zmean",
        out_compress="zstd",
    )
    assert _parquet_codec(polys_pq) == "ZSTD"
    assert len(va_io.read_vec_column(polys_pq, None, "zmean")) == 3

    zs_poly.calc_zonal_poly_pts_band_stats_file(
        polys_pq, None, ras_tif, img_band=1, out_field="centval", out_compress="gzip"
    )
    assert _parquet_codec(polys_pq) == "GZIP"
    assert "centval" in vu_meta.get_vec_lyr_cols(polys_pq)

    zs_poly.calc_zonal_band_stats_test_poly_pts_file(
        polys_pq,
        None,
        ras_tif,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        mean_field="hybmean",
        out_compress="zstd",
    )
    assert _parquet_codec(polys_pq) == "ZSTD"

    zs_poly.calc_zonal_band_pxl_counts_file(
        polys_pq,
        None,
        ras_tif,
        img_band=1,
        vals_lut={1: "V1"},
        total_field="totpxls",
        out_compress="snappy",
    )
    assert _parquet_codec(polys_pq) == "SNAPPY"

    pts_pq = str(tmp_path / "pts_zonal.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(500005.0, 5600075.0), Point(500025.0, 5600055.0)],
            crs="EPSG:32630",
        ),
        pts_pq,
    )
    zs_pt.ext_point_band_values_file(
        pts_pq,
        None,
        ras_tif,
        img_band=1,
        min_thres=0.0,
        max_thres=100.0,
        out_no_data_val=-1.0,
        out_field="pval",
        out_compress="zstd",
    )
    assert _parquet_codec(pts_pq) == "ZSTD"
    assert va_io.read_vec_column(pts_pq, None, "pval") == pytest.approx([1.0, 19.0])


def test_imageutils_imagecalc_elevation_geoparquet(tmp_path):
    """Test imagelut, spatial, subsetting, warp cutline, rasterise, imagecalc, and roughness."""
    t1 = str(tmp_path / "t1.tif")
    t2 = str(tmp_path / "t2.tif")
    _write_test_raster(
        t1, numpy.full((8, 8), 10.0, dtype=numpy.float32), tl_x=500000.0, tl_y=5600080.0
    )
    _write_test_raster(
        t2, numpy.full((8, 8), 20.0, dtype=numpy.float32), tl_x=500080.0, tl_y=5600080.0
    )

    lut_pq = str(tmp_path / "img_lut.parquet")
    iu_lut.create_img_extent_lut(
        [t1, t2],
        lut_pq,
        "tiles",
        out_format="PARQUET",
        use_pbar=False,
        out_compress="zstd",
    )
    assert _parquet_codec(lut_pq) == "ZSTD"
    assert set(iu_lut.get_all_lut_imgs(lut_pq)) == {t1, t2}
    assert iu_lut.query_img_lut(
        (500010.0, 500050.0, 5600010.0, 5600050.0), lut_pq
    ) == [t1]
    assert iu_lut.query_img_lut(
        (501000.0, 501100.0, 5601000.0, 5601100.0), lut_pq
    ) == []
    assert (
        iu_lut.get_raster_lyr(
            (500010.0, 500050.0, 5600010.0, 5600050.0),
            lut_pq,
            None,
            str(tmp_path / "vrt_tmp"),
        )
        == t1
    )

    # Cross-CRS GeoParquet ROI query against GeoParquet LUT
    roi_wgs_pq = str(tmp_path / "roi_wgs.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=[box(500010.0, 5600010.0, 500050.0, 5600050.0)],
            crs="EPSG:32630",
        ).to_crs("EPSG:4326"),
        roi_wgs_pq,
    )
    assert iu_lut.query_file_lut(lut_pq, None, roi_wgs_pq) == [t1]

    # subsetting.subset + rasterise.rasterise_vec_lyr + spatial.grid_scattered_pts
    roi_utm_pq = str(tmp_path / "roi_utm.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"val": [5.0]},
            geometry=[box(500010.0, 5600020.0, 500050.0, 5600060.0)],
            crs="EPSG:32630",
        ),
        roi_utm_pq,
    )
    sub_tif = str(tmp_path / "sub.tif")
    iu_sub.subset(t1, roi_utm_pq, None, sub_tif, "GTIFF", rsgispy.TYPE_32FLOAT)
    with rasterio.open(sub_tif) as src:
        assert (src.width, src.height) == (4, 4)

    ras_tif = str(tmp_path / "ras_pq.tif")
    iu_ras.rasterise_vec_lyr(
        roi_utm_pq,
        None,
        t1,
        ras_tif,
        burn_val=7,
        calc_stats=False,
        use_pbar=False,
    )
    with rasterio.open(ras_tif) as src:
        assert 7 in numpy.unique(src.read(1))

    pts_pq = str(tmp_path / "grid_pts.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"val": [1.0, 2.0, 3.0, 4.0]},
            geometry=[
                Point(500005.0, 5600075.0),
                Point(500075.0, 5600075.0),
                Point(500005.0, 5600005.0),
                Point(500075.0, 5600005.0),
            ],
            crs="EPSG:32630",
        ),
        pts_pq,
    )
    idw_tif = str(tmp_path / "idw.tif")
    iu_spat.grid_scattered_pts(
        pts_pq,
        None,
        "val",
        t1,
        idw_tif,
        gdal_grid_alg="nearest:radius1=20.0:radius2=20.0",
        gdalformat="GTIFF",
    )
    with rasterio.open(idw_tif) as src:
        assert src.read(1)[0, 0] == pytest.approx(1.0)

    # gdal_warp GeoParquet cutline + guard against cutline_where/cutline_sql
    crop_tif = str(tmp_path / "warp_cut.tif")
    iu_warp.gdal_warp(
        t1, crop_tif, cutline=roi_utm_pq, crop_to_cutline=True, dst_nodata=-999.0
    )
    with rasterio.open(crop_tif) as src:
        assert (src.width, src.height) == (4, 4)

    with pytest.raises(rsgispy.RSGISPyException):
        iu_warp.gdal_warp(
            t1, crop_tif, cutline=roi_utm_pq, cutline_where="val = 5"
        )

    # calc_pt_smpl_img_vals & elevation.roughness with GeoParquet & out_compress
    yy, xx = numpy.mgrid[0:32, 0:32]
    dem_arr = (
        100.0 + 0.5 * xx + 2.0 * numpy.sin(2.0 * numpy.pi * xx / 6.0)
    ).astype(numpy.float32)
    dem_tif = str(tmp_path / "dem.tif")
    _write_test_raster(dem_tif, dem_arr, tl_x=500000.0, tl_y=5600320.0)

    dem_pts_pq = str(tmp_path / "dem_pts.parquet")
    vp.write_vec_gdf(
        geopandas.GeoDataFrame(
            {"id": [1], "xbox": [6], "ybox": [6], "angle": [0.0]},
            geometry=[Point(500160.0, 5600160.0)],
            crs="EPSG:32630",
        ),
        dem_pts_pq,
    )
    smpl_out_pq = str(tmp_path / "smpl_out.parquet")
    rcptwin.calc_pt_smpl_img_vals(
        dem_tif,
        dem_pts_pq,
        None,
        calc_objs=[rrough.CalcProfileRoughMetrics()],
        out_vec_file=smpl_out_pq,
        out_format="PARQUET",
        out_compress="zstd",
    )
    assert _parquet_codec(smpl_out_pq) == "ZSTD"
    assert "avg_ra" in geopandas.read_parquet(smpl_out_pq).columns

    rough_pq = str(tmp_path / "rough_out.parquet")
    rrough.calc_simple_roughness_profile_metrics(
        dem_tif,
        dem_pts_pq,
        None,
        rough_pq,
        out_format="PARQUET",
        detrend=True,
        out_compress="gzip",
    )
    assert _parquet_codec(rough_pq) == "GZIP"
    assert "avg_ra" in geopandas.read_parquet(rough_pq).columns


def test_dataaccess_droneutils_checkdatasets_geoparquet(tmp_path, monkeypatch):
    """Test openstreetmap, droneutils, and checkdatasets with GeoParquet & compression."""
    # OpenStreetMap GeoParquet export
    sample_gdf = geopandas.GeoDataFrame(
        {"addr:street": ["High St"], "geometry": [box(-4.09, 52.41, -4.08, 52.42)]},
        crs="EPSG:4326",
    )
    monkeypatch.setattr(osm, "OSMNX_AVAIL", True)
    monkeypatch.setattr(
        osm,
        "osmnx",
        types.SimpleNamespace(
            features=types.SimpleNamespace(features_from_bbox=lambda bbox, tags: sample_gdf)
        ),
        raising=False,
    )
    osm_pq = str(tmp_path / "osm.parquet")
    osm.get_osm_to_file(
        osm_pq,
        "buildings",
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags={"building": True},
        out_format="PARQUET",
        out_compress="zstd",
    )
    assert _parquet_codec(osm_pq) == "ZSTD"
    assert len(geopandas.read_parquet(osm_pq)) == 1

    osm_multi_pq = str(tmp_path / "osm_multi.parquet")
    osm.get_osm_multi_to_file(
        osm_multi_pq,
        "features",
        bbox=(-4.1, -4.07, 52.4, 52.43),
        osm_tags_lst=[{"building": True}, {"amenity": "cafe"}],
        out_format="PARQUET",
        out_compress="gzip",
    )
    assert _parquet_codec(osm_multi_pq) == "GZIP"
    assert len(geopandas.read_parquet(osm_multi_pq)) == 2

    # Droneutils flightline GeoParquet & single-layer guard
    if rsgispy.droneutils.HAVE_EXIFTOOL:
        img1 = str(tmp_path / "d1.jpg")
        img2 = str(tmp_path / "d2.jpg")
        for p in (img1, img2):
            with open(p, "wb") as fp:
                fp.write(b"fakejpeg")

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
                "XMP:FlightPitchDegree": 1.0,
                "XMP:FlightRollDegree": 2.0,
                "XMP:FlightYawDegree": 90.0,
                "EXIF:FocalLength": 8.8,
                "EXIF:FocalLengthIn35mmFormat": 24.0,
                "EXIF:ImageWidth": 4000,
                "EXIF:ImageHeight": 3000,
                "EXIF:MaxApertureValue": 2.8,
                "EXIF:Model": "FC6310",
                "XMP:SensorIndex": 1,
            },
        }

        class _FakeExif:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get_metadata(self, p):
                return [meta[p]]

        monkeypatch.setattr(
            rsgispy.droneutils,
            "exiftool",
            types.SimpleNamespace(ExifToolHelper=_FakeExif),
        )
        monkeypatch.setattr(
            rsgispy.tools.files, "sort_file_by_datetime", lambda imgs: list(imgs)
        )

        pts_pq = str(tmp_path / "flight_pts.parquet")
        lines_pq = str(tmp_path / "flight_lines.parquet")
        rsgispy.droneutils.create_flightline_vec(
            [img1, img2],
            pts_pq,
            "pts",
            lines_pq,
            "lines",
            out_format="PARQUET",
            out_compress="zstd",
        )
        assert _parquet_codec(pts_pq) == "ZSTD"
        assert _parquet_codec(lines_pq) == "ZSTD"
        assert len(geopandas.read_parquet(pts_pq)) == 2
        assert len(geopandas.read_parquet(lines_pq)) == 1

        with pytest.raises(rsgispy.RSGISPyException):
            rsgispy.droneutils.create_flightline_vec(
                [img1, img2],
                pts_pq,
                "pts",
                pts_pq,
                "lines",
                out_format="PARQUET",
            )

    # tools.checkdatasets.check_gdal_vector_file with GeoParquet
    ok, err = chk.check_gdal_vector_file(osm_pq, chk_proj=True, epsg_code=4326)
    assert ok is True
    assert err == ""

    comp_pq = str(tmp_path / "chk_comp.parquet.gzip")
    vp.write_vec_gdf(sample_gdf, comp_pq)
    ok_comp, err_comp = chk.check_gdal_vector_file(
        comp_pq, chk_proj=True, epsg_code=4326
    )
    assert ok_comp is True
    assert err_comp == ""

    ok_bad_epsg, err_bad_epsg = chk.check_gdal_vector_file(
        osm_pq, chk_proj=True, epsg_code=32630
    )
    assert ok_bad_epsg is False
    assert "does not match" in err_bad_epsg

    non_geo_pq = str(tmp_path / "non_geo.parquet")
    sample_gdf.drop(columns=["geometry"]).to_parquet(non_geo_pq)
    ok_non_geo, err_non_geo = chk.check_gdal_vector_file(non_geo_pq)
    assert ok_non_geo is False
    assert "geo" in err_non_geo.lower()
