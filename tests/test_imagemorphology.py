"""
Tests for rsgispy.imagemorphology.
"""


import numpy as np
import pytest
import rasterio

import rsgispy
import rsgispy.imagemorphology as rimorph

from conftest import write_raster


def _write_raster(path, arr, dtype="uint8", nodata=None):
    return write_raster(path, arr, dtype=dtype, nodata=nodata)


def test_create_circular_op_and_single_band_dilate_erode_gradient(tmp_path):
    """Test create_circular_op, image_dilate, image_erode, and image_gradient."""
    op_no_ext = tmp_path / "circ5"
    rimorph.create_circular_op(str(op_no_ext), op_size=5)
    op_file = tmp_path / "circ5.gmtxt"
    assert op_file.exists()
    content = op_file.read_text(encoding="utf-8").splitlines()
    assert content[:2] == ["m=5", "n=5"]
    assert content[2:] == [
        "0,1,1,1,0",
        "1,1,1,1,1",
        "1,1,1,1,1",
        "1,1,1,1,1",
        "0,1,1,1,0",
    ]

    with pytest.raises(rsgispy.RSGISPyException):
        rimorph.create_circular_op(str(op_file), op_size=4)

    # 9x9 binary image with single central bright pixel at (4, 4)
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 1
    in_path = tmp_path / "dot.tif"
    _write_raster(in_path, img, dtype="uint8")

    dil_file_out = tmp_path / "dil_circ.tif"
    rimorph.image_dilate(
        str(in_path),
        str(dil_file_out),
        str(op_file),
        use_op_file=True,
        op_size=5,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(dil_file_out) as dds:
        dil = dds.read(1)
        assert dds.dtypes[0] == "uint8"
        # Single pixel dilates into exact 5x5 circular footprint around (4, 4)
        expected_circ = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(dil[2:7, 2:7], expected_circ)
        assert dil[2, 2] == 0

    # Square 3x3 erosion on dilated image
    ero_out = tmp_path / "ero_sq.tif"
    rimorph.image_erode(
        str(dil_file_out),
        str(ero_out),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(ero_out) as eds:
        ero = eds.read(1)
        assert ero[4, 4] == 1
        assert ero[2, 3] == 0

    # Gradient = dilation - erosion using 3x3 square on 9x9 block of 5s inside 1s
    blk = np.ones((9, 9), dtype=np.float32)
    blk[3:6, 3:6] = 5.0
    blk_path = tmp_path / "blk.tif"
    _write_raster(blk_path, blk, dtype="float32")
    grad_out = tmp_path / "grad.tif"
    rimorph.image_gradient(
        str(blk_path),
        str(grad_out),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    with rasterio.open(grad_out) as gds:
        grad = gds.read(1)
        assert gds.dtypes[0] == "float32"
        # Centre (4, 4) surrounded entirely by 5.0 -> grad=0; boundary pixel (3, 3) -> 5 - 1 = 4
        assert grad[4, 4] == pytest.approx(0.0)
        assert grad[3, 3] == pytest.approx(4.0)


def test_combined_out_and_local_minima(tmp_path):
    """Test *_combined_out functions and image_local_minima variants."""
    # Two-band 7x7 raster
    b1 = np.full((7, 7), 10.0, dtype=np.float32)
    b2 = np.full((7, 7), 20.0, dtype=np.float32)
    b1[3, 3] = 50.0  # band 1 peak
    b2[3, 3] = 2.0   # band 2 valley
    in_path = tmp_path / "twoband.tif"
    _write_raster(in_path, np.stack([b1, b2]), dtype="float32")

    dil_comb = tmp_path / "dil_comb.tif"
    ero_comb = tmp_path / "ero_comb.tif"
    grad_comb = tmp_path / "grad_comb.tif"
    rimorph.image_dilate_combined_out(
        str(in_path),
        str(dil_comb),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    rimorph.image_erode_combined_out(
        str(in_path),
        str(ero_comb),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    rimorph.image_gradient_combined_out(
        str(in_path),
        str(grad_comb),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_32FLOAT,
        use_pbar=False,
    )
    with (
        rasterio.open(dil_comb) as dds,
        rasterio.open(ero_comb) as eds,
        rasterio.open(grad_comb) as gds,
    ):
        assert dds.count == 1
        assert eds.count == 1
        assert gds.count == 1
        assert dds.read(1)[3, 3] == pytest.approx(50.0)
        assert eds.read(1)[3, 3] == pytest.approx(2.0)
        assert gds.read(1)[3, 3] == pytest.approx(48.0)

    # Local minima test: two strict minima at (2, 2) and (5, 5)
    min_arr = np.full((8, 8), 10.0, dtype=np.float32)
    min_arr[2, 2] = 1.0
    min_arr[5, 5] = 2.0
    min_in = tmp_path / "min_in.tif"
    _write_raster(min_in, min_arr, dtype="float32")

    min_seq = tmp_path / "min_seq.tif"
    rimorph.image_local_minima(
        str(min_in),
        str(min_seq),
        sequencial_out=True,
        allow_equal=False,
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_16UINT,
        use_pbar=False,
    )
    with rasterio.open(min_seq) as mds:
        m = mds.read(1)
        assert mds.dtypes[0] == "uint16"
        # Border rows/cols see 0 padding so only interior minima at (2,2) and (5,5)
        assert m[2, 2] == 1
        assert m[5, 5] == 2
        assert int(np.sum(m > 0)) == 2

    # Combined local minima with allow_equal=False and binary output
    min_comb = tmp_path / "min_comb.tif"
    rimorph.image_local_minima_combined_out(
        str(in_path),
        str(min_comb),
        sequencial_out=False,
        allow_equal=False,
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(min_comb) as mcds:
        mc = mcds.read(1)
        assert mcds.count == 1
        assert mc[3, 3] == 1
        assert mc[4, 4] == 0


def test_opening_closing_and_top_hats(tmp_path):
    """Test image_opening, image_closing, image_black_top_hat, and image_white_top_hat."""
    # Background 5 with an isolated bright spike (20) at (3, 3) and dark pit (1) at (7, 7)
    arr = np.full((11, 11), 5, dtype=np.uint8)
    arr[3, 3] = 20
    arr[7, 7] = 1
    in_path = tmp_path / "spikes.tif"
    _write_raster(in_path, arr, dtype="uint8")

    open_out = tmp_path / "open.tif"
    open_tmp = tmp_path / "open_tmp.tif"
    rimorph.image_opening(
        str(in_path),
        str(open_out),
        op_size=3,
        tmp_img=str(open_tmp),
        niters=1,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    assert open_tmp.exists()
    with rasterio.open(open_out) as ods:
        opened = ods.read(1)
        # Bright spike at (3, 3) is removed by opening
        assert opened[3, 3] == 5

    close_out = tmp_path / "close.tif"
    rimorph.image_closing(
        str(in_path),
        str(close_out),
        op_size=3,
        niters=2,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(close_out) as cds:
        closed = cds.read(1)
        # Dark pit at (7, 7) is filled by closing
        assert closed[7, 7] == 5

    # White top-hat isolates bright spike: input - opening -> 20 - 5 = 15 at (3, 3)
    wth_out = tmp_path / "wth.tif"
    rimorph.image_white_top_hat(
        str(in_path),
        str(wth_out),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(wth_out) as wds:
        wth = wds.read(1)
        assert wth[3, 3] == 15
        assert wth[5, 5] == 0

    # Black top-hat isolates dark pit: closing - input -> 5 - 1 = 4 at (7, 7)
    bth_out = tmp_path / "bth.tif"
    rimorph.image_black_top_hat(
        str(in_path),
        str(bth_out),
        op_size=3,
        gdalformat="GTIFF",
        datatype=rsgispy.TYPE_8UINT,
        use_pbar=False,
    )
    with rasterio.open(bth_out) as bds:
        bth = bds.read(1)
        assert bth[7, 7] == 4
        assert bth[5, 5] == 0

    # Validation checks
    with pytest.raises(rsgispy.RSGISPyException):
        rimorph.image_dilate(
            str(tmp_path / "missing.tif"),
            str(open_out),
            op_size=3,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )
    with pytest.raises(rsgispy.RSGISPyException):
        rimorph.image_opening(
            str(in_path),
            str(open_out),
            op_size=3,
            niters=0,
            gdalformat="GTIFF",
            datatype=rsgispy.TYPE_8UINT,
            use_pbar=False,
        )


def test_imagemorphology_default_datatype_and_positional_gdalformat(tmp_path):
    """datatype=None keeps the input dtype (or picks the minima output type),
    no_data_val is honoured, and gdalformat cannot be passed positionally."""
    # Default datatype=None preserves input dtype and honours no_data_val
    arr_f32 = np.full((7, 7), 10.0, dtype=np.float32)
    arr_f32[3, 3] = 2.0
    in_f32 = tmp_path / "in_f32.tif"
    _write_raster(in_f32, arr_f32, dtype="float32")

    out_dil = tmp_path / "dil_default.tif"
    rimorph.image_dilate(str(in_f32), str(out_dil), no_data_val=-9999.0, use_pbar=False)
    with rasterio.open(out_dil) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == -9999.0

    # Local minima datatype=None resolves to uint8 (binary) or uint32 (sequential)
    out_min_bin = tmp_path / "min_bin.tif"
    rimorph.image_local_minima(
        str(in_f32), str(out_min_bin), sequencial_out=False, allow_equal=False, use_pbar=False
    )
    with rasterio.open(out_min_bin) as ds:
        assert ds.dtypes[0] == "uint8"

    out_min_seq = tmp_path / "min_seq_def.tif"
    rimorph.image_local_minima(
        str(in_f32), str(out_min_seq), sequencial_out=True, allow_equal=False, use_pbar=False
    )
    with rasterio.open(out_min_seq) as ds:
        assert ds.dtypes[0] == "uint32"

    # Passing gdalformat positionally must raise TypeError
    with pytest.raises(TypeError):
        rimorph.image_dilate(
            str(in_f32), str(tmp_path / "bad_pos.tif"), "", False, 3, "GTIFF"
        )


@pytest.mark.parametrize("sequencial_out", [False, True])
def test_local_minima_strict_finds_border_minima(tmp_path, sequencial_out):
    """allow_equal=False finds minima on the image border too."""
    arr = np.full((5, 5), 50, dtype=np.uint8)
    arr[0, 0] = 1  # corner minimum
    arr[2, 4] = 2  # edge minimum
    arr[2, 2] = 3  # interior minimum
    in_path = str(tmp_path / "in.tif")
    _write_raster(in_path, arr[np.newaxis])
    out_path = str(tmp_path / "minima.tif")
    rimorph.image_local_minima(
        in_path, out_path, sequencial_out=sequencial_out, allow_equal=False,
        calc_stats=False, use_pbar=False,
    )
    with rasterio.open(out_path) as ds:
        found = ds.read(1) > 0
    expected = np.zeros((5, 5), dtype=bool)
    expected[0, 0] = expected[2, 4] = expected[2, 2] = True
    np.testing.assert_array_equal(found, expected)

    if not sequencial_out:
        from rsgispy.imageprocess import MemorySink, RasterSource, TileGraph

        node = rimorph.image_local_minima_node(
            RasterSource(in_path), allow_equal=False
        )
        res = TileGraph(MemorySink(node, name="m"), block_size=3).run(use_pbar=False)
        np.testing.assert_array_equal(res["m"][0] > 0, expected)


def _morph_reference(arr, size, reducer, mode, fill=0.0):
    """Reference square-operator dilation / erosion with ``np.pad`` borders."""
    from numpy.lib.stride_tricks import sliding_window_view  # pylint: disable=import-outside-toplevel

    kwargs = {"constant_values": fill} if mode == "constant" else {}
    pad = np.pad(arr.astype(np.float64), size // 2, mode=mode, **kwargs)
    return reducer(sliding_window_view(pad, (size, size)), axis=(-2, -1))


@pytest.mark.parametrize(
    "mode, fill", [("edge", 0.0), ("reflect", 0.0), ("symmetric", 0.0), ("constant", 200.0)]
)
def test_morphology_border_mode(tmp_path, mode, fill):
    """border_mode / border_fill set how the operator is filled beyond the
    image edge for dilation, erosion, gradient, opening, closing and top-hats."""
    rng = np.random.default_rng(3)
    arr = rng.integers(1, 120, size=(9, 11)).astype(np.uint8)
    in_p = tmp_path / "bm_in.tif"
    _write_raster(in_p, arr)

    def _run(fn, tag, **kwargs):
        out_p = tmp_path / f"{tag}_{mode}.tif"
        fn(str(in_p), str(out_p), op_size=3, border_mode=mode, border_fill=fill,
           use_pbar=False, **kwargs)
        with rasterio.open(out_p) as src:
            return src.read(1).astype(np.float64)

    dil = _morph_reference(arr, 3, np.max, mode, fill)
    ero = _morph_reference(arr, 3, np.min, mode, fill)
    np.testing.assert_array_equal(_run(rimorph.image_dilate, "dil"), dil)
    np.testing.assert_array_equal(_run(rimorph.image_erode, "ero"), ero)
    np.testing.assert_array_equal(
        _run(rimorph.image_gradient, "grad"), np.minimum(dil - ero, 255)
    )
    np.testing.assert_array_equal(_run(rimorph.image_dilate_combined_out, "dilc"), dil)
    # Opening: the dilation pads the eroded image with the same border.
    opened = _morph_reference(ero, 3, np.max, mode, fill)
    np.testing.assert_array_equal(_run(rimorph.image_opening, "open"), opened)
    np.testing.assert_array_equal(
        _run(rimorph.image_white_top_hat, "wth"),
        np.clip(arr.astype(np.float64) - opened, 0, 255)
    )
    closed = _morph_reference(dil, 3, np.min, mode, fill)
    np.testing.assert_array_equal(_run(rimorph.image_closing, "close"), closed)
    np.testing.assert_array_equal(
        _run(rimorph.image_black_top_hat, "bth"),
        np.clip(closed - arr.astype(np.float64), 0, 255)
    )


def test_morphology_border_mode_default_is_edge_and_bad_mode_raises(tmp_path):
    """The default border_mode gives the same output as 'edge'; an unknown
    mode raises."""
    arr = np.arange(30, dtype=np.uint8).reshape(5, 6)
    in_p = tmp_path / "def_in.tif"
    _write_raster(in_p, arr)
    rimorph.image_erode(str(in_p), str(tmp_path / "a.tif"), use_pbar=False)
    rimorph.image_erode(str(in_p), str(tmp_path / "b.tif"), border_mode="edge", use_pbar=False)
    with rasterio.open(tmp_path / "a.tif") as a_ds, rasterio.open(tmp_path / "b.tif") as b_ds:
        np.testing.assert_array_equal(a_ds.read(), b_ds.read())
    with pytest.raises(rsgispy.RSGISPyException):
        rimorph.image_erode(
            str(in_p), str(tmp_path / "c.tif"), border_mode="wrap", use_pbar=False
        )


def test_morphology_edge_padding_and_no_nodata_bleed(tmp_path):
    arr = np.full((12, 12), 50.0, dtype=np.float32)
    arr[5, 5] = -9999.0  # nodata hole
    arr[6, 6] = np.nan
    in_img = write_raster(tmp_path / "morph_in.tif", arr, nodata=-9999.0, origin=(500000.0, 5800000.0))

    op_file = str(tmp_path / "op3.gmtxt")
    rimorph.create_circular_op(op_file, 3)

    out_erode = str(tmp_path / "erode.tif")
    rimorph.image_erode(
        in_img,
        out_erode,
        op_file,
        use_op_file=True,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )
    out_dilate = str(tmp_path / "dilate.tif")
    rimorph.image_dilate(
        in_img,
        out_dilate,
        op_file,
        use_op_file=True,
        no_data_val=-9999.0,
        calc_stats=False,
        use_pbar=False,
    )
    out_open = str(tmp_path / "open.tif")
    rimorph.image_opening(
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
