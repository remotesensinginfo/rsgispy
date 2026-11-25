"""
Worked example: techniques for processing large rasters with rsgispy
(memory estimates, block size and creation options, VRTs, custom block
processing, in-memory tile graphs and filter chains, tiling and mosaicking,
cloud optimised GeoTIFFs and tiled, multi-core segmentation).

This script is included step by step in the documentation guide
``docs/guides/large_rasters.rst`` and is run end-to-end by the test suite
(``tests/test_large_rasters_example.py``) on small synthetic data, so the
documented code is always tested.

Run it on your own data with, e.g.::

    python large_rasters.py --s2 s2_sref_10m.tif --red-band 4 --nir-band 8 \\
        --block-size 512 --tile-size 5000 --n-jobs 4 --out-dir outputs
"""

import argparse
import os
from typing import Any, Dict, Optional, Sequence


def run_workflow(
    s2_img: str,
    out_dir: str,
    red_band: int = 4,
    nir_band: int = 8,
    seg_bands: Sequence[int] = (2, 3, 4, 8, 11, 12),
    block_size: int = 512,
    tile_size: int = 5000,
    n_jobs: int = 1,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example on a surface reflectance image ``s2_img`` (no data
    value 0) with the red and NIR bands at ``red_band`` and ``nir_band``.
    ``block_size`` is the processing block size in pixels, ``tile_size`` the
    size of the tiles written to disk and segmented separately, and
    ``n_jobs`` the number of worker processes.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-memory]
    import numpy as np
    import rsgispy
    from rsgispy.imageutils.metadata import (
        get_img_band_count,
        get_img_size,
        get_rsgislib_datatype_from_img,
    )

    x_size, y_size = get_img_size(s2_img)
    n_bands = get_img_band_count(s2_img)
    in_dtype = np.dtype(
        rsgispy.get_numpy_datatype(get_rsgislib_datatype_from_img(s2_img))
    )
    mib = 2.0**20
    full_mib = x_size * y_size * n_bands * in_dtype.itemsize / mib
    # Blocks are read as float64 (8 bytes per value), one array per band.
    block_mib = block_size * block_size * n_bands * 8 / mib
    print(f"{x_size} x {y_size} pixels, {n_bands} bands ({in_dtype}): "
          f"{full_mib:.1f} MiB if read at once")
    print(f"One {block_size} x {block_size} block of all bands as float64: "
          f"{block_mib:.1f} MiB")
    # [end-step-1-memory]
    out["full_mib"] = full_mib
    out["block_mib"] = block_mib

    # [step-2-settings]
    import rsgispy.imageprocess
    import rsgispy.imageutils.envopts

    # Block size used by band_maths, the filters, zonal statistics, ...
    rsgispy.imageprocess.set_default_block_size(block_size)
    print(rsgispy.imageprocess.get_default_block_size())
    # GeoTIFF outputs are tiled and DEFLATE compressed (BigTIFF if needed) by
    # default; set_img_creation_opts / set_env_vars_*_gtiff_outs change this
    # and reset_img_creation_opts restores the default.
    rsgispy.imageutils.envopts.reset_img_creation_opts("GTIFF")
    print(rsgispy.imageutils.envopts.get_gdal_img_creation_opts("GTIFF"))
    # [end-step-2-settings]

    # [step-3-vrt]
    from rsgispy.imageutils.vrt import create_vrt_band_subset

    red_nir_vrt = os.path.join(out_dir, "red_nir.vrt")
    create_vrt_band_subset(
        s2_img, red_nir_vrt, img_bands=[red_band, nir_band],
        band_names=["Red", "NIR"],
    )
    # [end-step-3-vrt]
    out["red_nir_vrt"] = red_nir_vrt

    # [step-4-blocks]
    from contextlib import ExitStack
    from rsgispy.imageutils.stats import pop_img_stats

    band_defs = rsgispy.imageprocess.normalise_band_defs([
        rsgispy.imageprocess.BandDefn("red", red_nir_vrt, 1),
        rsgispy.imageprocess.BandDefn("nir", red_nir_vrt, 2),
    ])

    def calc_ndvi(block_vars: Dict[str, np.ndarray]) -> np.ndarray:
        red, nir = block_vars["red"], block_vars["nir"]
        valid = (red > 0) & (nir > 0)  # 0 is the input no data value
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
        return np.where(valid, ndvi, -9999.0)

    ndvi_img = os.path.join(out_dir, "ndvi.tif")
    with ExitStack() as stack:
        datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
        grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
        dst = rsgispy.imageprocess.open_output_dataset(
            stack, ndvi_img, "GTIFF", "float32", grid, count=1
        )
        dst.nodata = -9999.0
        dst.set_band_description(1, "NDVI")
        rsgispy.imageprocess.process_blocks(
            dst, datasets, band_defs, grid, calc_ndvi,
            out_np_dtype=np.dtype(np.float32),
            block_size=block_size,  # None = the default set in step 2
            use_pbar=use_pbar,
        )
    pop_img_stats(ndvi_img, no_data_val=-9999.0, use_pbar=use_pbar)
    # [end-step-4-blocks]
    out["ndvi_img"] = ndvi_img

    # [step-5-tilegraph]
    from rsgispy.imagefilter.chained import FilterStep, build_filter_chain_node
    from rsgispy.imageprocess import (
        BandMathOp,
        DiskSink,
        RasterSource,
        TileGraph,
    )

    red = RasterSource(s2_img, bands=red_band, name="red")
    nir = RasterSource(s2_img, bands=nir_band, name="nir")
    ndvi = BandMathOp(
        {"red": red, "nir": nir}, "(nir - red) / (nir + red)", nodata=0
    )
    # Median 3 x 3 then mean 5 x 5 of NIR; a 3-pixel halo is added for you.
    nir_smooth = build_filter_chain_node(
        nir, [FilterStep("median", filter_size=3),
              FilterStep("mean", filter_size=5)], no_data_val=0,
    )
    nir_highpass = BandMathOp(
        {"nir": nir, "smooth": nir_smooth}, "nir - smooth", nodata=0
    )
    tg_ndvi_img = os.path.join(out_dir, "tg_ndvi.tif")
    tg_hp_img = os.path.join(out_dir, "tg_nir_highpass.tif")
    TileGraph(
        [
            DiskSink(ndvi, tg_ndvi_img, nodata=0, band_names=["NDVI"],
                     calc_stats=True),
            DiskSink(nir_highpass, tg_hp_img, nodata=0,
                     band_names=["NIR_highpass"], calc_stats=True),
        ],
        block_size=block_size,
    ).run(use_pbar=use_pbar)
    # [end-step-5-tilegraph]
    out["tg_ndvi_img"] = tg_ndvi_img
    out["tg_hp_img"] = tg_hp_img

    # [step-6-filter-chain]
    from rsgispy.imagefilter.chained import apply_filter_chain

    ndvi_edges_img = os.path.join(out_dir, "ndvi_edges.tif")
    apply_filter_chain(
        ndvi_img,
        ndvi_edges_img,
        steps=[FilterStep("median", filter_size=3), FilterStep("sobel")],
        block_size=block_size,
        use_pbar=use_pbar,
    )
    # [end-step-6-filter-chain]
    out["ndvi_edges_img"] = ndvi_edges_img

    # [step-7-tiles]
    import glob
    from rsgispy.imagecalc.maths import BandDefn, band_maths
    from rsgispy.imageutils.tiling import create_tiles_multi_core
    from rsgispy.imageutils.vrt import create_mosaic_images_vrt

    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    create_tiles_multi_core(
        s2_img, os.path.join(tiles_dir, "s2"), tile_size, tile_size,
        n_cores=n_jobs, calc_stats=False, use_pbar=use_pbar,
    )
    tile_ndvi_imgs = []
    for tile_img in sorted(glob.glob(os.path.join(tiles_dir, "s2_x*y*.tif"))):
        tile_ndvi = tile_img.replace(".tif", "_ndvi.tif")
        band_maths(  # each tile could run as a separate job on a cluster
            tile_ndvi,
            "(red > 0) && (nir > 0) ? (nir - red) / (nir + red) : -9999",
            [BandDefn("red", tile_img, red_band),
             BandDefn("nir", tile_img, nir_band)],
            band_names=["NDVI"],
            use_pbar=False,
        )
        tile_ndvi_imgs.append(tile_ndvi)
    ndvi_mosaic_vrt = os.path.join(out_dir, "ndvi_tiles.vrt")
    create_mosaic_images_vrt(
        tile_ndvi_imgs, ndvi_mosaic_vrt, src_nodata=-9999.0,
        vrt_nodata=-9999.0,
    )
    print(f"{len(tile_ndvi_imgs)} tiles mosaicked into {ndvi_mosaic_vrt}")
    # [end-step-7-tiles]
    out["n_tiles"] = len(tile_ndvi_imgs)
    out["ndvi_mosaic_vrt"] = ndvi_mosaic_vrt

    # [step-8-cog]
    from rsgispy.imageutils.warp import gdal_translate
    from rsgispy.tools.checkdatasets import check_gdal_image_file

    ndvi_cog = os.path.join(out_dir, "ndvi_cog.tif")
    gdal_translate(
        ndvi_mosaic_vrt,
        ndvi_cog,
        options={"COMPRESS": "DEFLATE", "PREDICTOR": "YES",
                 "OVERVIEW_RESAMPLING": "AVERAGE", "BIGTIFF": "IF_SAFER"},
        gdalformat="COG",  # written via a temporary tiled GeoTIFF
        use_pbar=use_pbar,
    )
    print(check_gdal_image_file(ndvi_cog, n_bands=1, chk_proj=True,
                                read_img=True))
    # [end-step-8-cog]
    out["ndvi_cog"] = ndvi_cog

    # [step-9-segments]
    import rsgispy.segmentation
    from rsgispy.ml.stack import BandSpec
    from rsgispy.vector.create import polygonise_raster_tiled
    from rsgispy.zonalstats.labels import calc_label_zonal_stats

    segs_img = os.path.join(out_dir, "s2_segs.tif")
    rsgispy.segmentation.shepherd_segmentation(
        s2_img,
        segs_img,
        n_clusters=60,
        min_pixels=10,
        bands=list(seg_bands),
        tile_size=tile_size,  # segment tile by tile (bounded memory) ...
        overlap=32,  # ... with a halo, and stitch segments across seams
        n_jobs=n_jobs,  # tiles processed in parallel worker processes
        use_numba=True,  # compiled kernels if numba is installed
        no_data_val=0,
        use_pbar=use_pbar,
    )
    seg_stats = calc_label_zonal_stats(
        segs_img,
        BandSpec(s2_img, bands=tuple(seg_bands), no_data_val=0),
        stats=["mean", "stddev"],
        block_size=block_size,
        out_file=os.path.join(out_dir, "s2_seg_stats.parquet"),
        use_pbar=use_pbar,
    )
    print(f"{len(seg_stats)} segments")
    segs_vec = os.path.join(out_dir, "s2_segs.parquet")
    polygonise_raster_tiled(
        segs_img,
        segs_vec,  # GeoParquet (snappy compressed), streamed to disk
        pxl_val_fieldname="seg_id",
        no_data_val=0,
        dissolve_by_value=True,  # one (Multi)Polygon per segment ...
        sort_by_value=True,  # ... sorted by segment ID
        tile_size=min(tile_size, 2048),  # ~2 GB per worker for dense segments
        n_jobs=n_jobs,
        use_pbar=use_pbar,
    )
    # [end-step-9-segments]
    out["segs_img"] = segs_img
    out["seg_stats"] = seg_stats
    out["segs_vec"] = segs_vec
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command line interface for running the example on your own data."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True,
                        help="Surface reflectance image (no data value 0).")
    parser.add_argument("--red-band", type=int, default=4,
                        help="Red band number (Default: 4).")
    parser.add_argument("--nir-band", type=int, default=8,
                        help="NIR band number (Default: 8).")
    parser.add_argument("--seg-bands", type=int, nargs="+",
                        default=[2, 3, 4, 8, 11, 12],
                        help="Bands used for segmentation.")
    parser.add_argument("--block-size", type=int, default=512,
                        help="Processing block size in pixels (Default: 512).")
    parser.add_argument("--tile-size", type=int, default=5000,
                        help="Tile size in pixels (Default: 5000).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of worker processes (Default: 1).")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    args = parser.parse_args(argv)
    run_workflow(
        args.s2,
        args.out_dir,
        red_band=args.red_band,
        nir_band=args.nir_band,
        seg_bands=args.seg_bands,
        block_size=args.block_size,
        tile_size=args.tile_size,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
