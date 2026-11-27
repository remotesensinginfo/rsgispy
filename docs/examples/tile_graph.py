"""
Worked example: chaining operations in memory with a tile graph.

This script is included step by step in the documentation guide
``docs/guides/tile_graph.rst`` and is run end-to-end by the test suite
(``tests/test_tile_graph_example.py``) on small synthetic data, so the
documented code is always tested.

From a Sentinel-2-like surface reflectance stack it calculates NDVI and NDWI,
median filters them, adds their local standard deviation as a texture,
trains a small decision tree on labelled pixels, classifies the image and
smooths the classification with a majority filter, and, in the same pass,
writes the slope and hillshade of a DEM. It then shows filter chains, the
no-data convention, halos and block sizes, custom operations, statistics
sinks, parallel tiles and how to check a graph against the file-based
functions.

Run it on your own data with, e.g.::

    python tile_graph.py --s2 s2_sref_10m.tif --dem dem_10m.tif \\
        --train train_labels.tif --out-dir outputs \\
        --band-names 1=Blue 2=Green 3=Red 7=NIR --n-jobs 4

The DEM and the training labels (a raster of class IDs, 0 = unlabelled, e.g.
rasterised training polygons) must be on the image's pixel grid.
"""

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Tuple

# Class ID of the training image -> (name, colour) of the class map.
CLASSES = {
    1: ("Water", "#1f78b4"),
    2: ("Forest", "#33a02c"),
    3: ("Grass", "#b2df8a"),
    4: ("Bare", "#d9b38c"),
}


def run_workflow(
    s2_img: str,
    dem_img: str,
    train_img: str,
    out_dir: str,
    band_names: Optional[Mapping[int, str]] = None,
    classes: Optional[Mapping[int, Tuple[str, str]]] = None,
    block_size: Optional[int] = None,
    n_jobs: int = 1,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``band_names`` maps the band numbers of
    ``s2_img`` to the names used by the indices (``Blue``, ``Green``,
    ``Red``, ``NIR``; Default = bands 1-4 in that order). ``train_img`` holds
    the class ID of the training pixels and 0 elsewhere; ``classes`` maps
    each class ID to its name and colour (Default = ``CLASSES``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}
    if band_names is None:
        band_names = {1: "Blue", 2: "Green", 3: "Red", 4: "NIR"}
    if classes is None:
        classes = CLASSES

    # [step-1-sources]
    from rsgispy.imageprocess import RasterSource, infer_node_meta

    s2 = RasterSource(s2_img, name="s2")  # all bands, read as float64
    dem = RasterSource(dem_img, name="dem")
    print(infer_node_meta(s2))  # bands, band names, data type, no data
    # [end-step-1-sources]

    # [step-2-features]
    from rsgispy.imagecalc.calcindices import calc_indices_node
    from rsgispy.imagefilter.stats import median_filter_node, stddev_filter_node
    from rsgispy.imageprocess import SelectBands, StackBands

    indices = calc_indices_node(s2, band_names, ["NDVI", "NDWI"])
    indices_med = median_filter_node(indices, 3)  # remove isolated noise
    indices_sd = stddev_filter_node(indices, 5)  # local texture
    features = StackBands(  # 4 feature bands, with unique names
        [indices_med, indices_sd],
        band_names=["NDVI_med", "NDWI_med", "NDVI_sd", "NDWI_sd"],
    )
    ndvi = SelectBands(indices, 1)  # used again in later steps
    # [end-step-2-features]

    # [step-3-train]
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from rsgispy.imageprocess import SampleSink, TileGraph

    samples = TileGraph(
        SampleSink(features, RasterSource(train_img),  # 0 = unlabelled
                   max_per_class=2000, name="samples"),
        block_size=block_size,
    ).run(use_pbar=use_pbar)["samples"]
    feat_cols = list(infer_node_meta(features).band_names)
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(samples[feat_cols].to_numpy(), samples["out_cls_id"])
    # [end-step-3-train]
    out["samples"] = samples
    out["n_train"] = len(samples)

    # [step-4-graph]
    from rsgispy.classification.classpostprocess import majority_filter_node
    from rsgispy.elevation.illumination import hillshade_node
    from rsgispy.elevation.terrain import slope_node
    from rsgispy.imageprocess import DiskSink
    from rsgispy.ml.predict import predict_raster_node

    cls = predict_raster_node(features, clf, thematic=True)  # uint8, 0 = no data
    cls_smooth = majority_filter_node(cls, 3, dtype=np.uint8)
    slope = slope_node(dem)  # degrees
    hshade = hillshade_node(dem, azimuth=315.0, zenith=45.0)  # uint8

    cls_img = os.path.join(out_dir, "s2_cls.tif")
    feats_img = os.path.join(out_dir, "s2_features.tif")
    slope_img = os.path.join(out_dir, "slope.tif")
    hshade_img = os.path.join(out_dir, "hillshade.tif")
    graph = TileGraph(
        [
            # uint8 class map: thematic, with the class names and colours
            DiskSink(cls_smooth, cls_img, classes=classes),
            DiskSink(features, feats_img),  # float32, with the band names
            DiskSink(slope, slope_img),  # float32
            DiskSink(hshade, hshade_img),  # uint8
        ],
        block_size=block_size,
    )
    print(graph.describe())  # the grid, block size and every node
    graph.run(use_pbar=use_pbar, n_jobs=n_jobs)
    # [end-step-4-graph]
    out["describe"] = graph.describe()
    out.update(cls_img=cls_img, feats_img=feats_img, slope_img=slope_img,
               hshade_img=hshade_img)

    # [step-5-filter-chains]
    from rsgispy.imagefilter.chained import (
        FilterStep,
        apply_filter_chain,
        build_filter_chain_node,
    )

    # Unsharp mask: the input plus 1.5 x (input - its Gaussian smoothing).
    ndvi_sharp = build_filter_chain_node(ndvi, [
        FilterStep("gaussian", filter_size=5, params={"stddev": 1.0}),
        FilterStep("band_math", params={"expr": "input + 1.5 * (input - x)"}),
    ])
    # Difference of Gaussians: a named step is a variable for band_math.
    ndvi_dog = build_filter_chain_node(ndvi, [
        FilterStep("gaussian", filter_size=5, params={"stddev": 1.0},
                   name="fine"),
        FilterStep("gaussian", filter_size=9, params={"stddev": 2.0}),
        FilterStep("band_math", params={"expr": "fine - x"}),
    ])
    sharp_img = os.path.join(out_dir, "ndvi_sharp.tif")
    dog_img = os.path.join(out_dir, "ndvi_dog.tif")
    TileGraph([DiskSink(ndvi_sharp, sharp_img), DiskSink(ndvi_dog, dog_img)],
              block_size=block_size).run(use_pbar=use_pbar, n_jobs=n_jobs)

    # One image in, one out: apply_filter_chain builds and runs the graph.
    dem_edges_img = os.path.join(out_dir, "dem_edges.tif")
    apply_filter_chain(
        dem_img, dem_edges_img,
        steps=[FilterStep("median", filter_size=3), FilterStep("sobel")],
        block_size=block_size, use_pbar=use_pbar,
    )
    # [end-step-5-filter-chains]
    out.update(sharp_img=sharp_img, dog_img=dog_img, dem_edges_img=dem_edges_img)

    # [step-6-nodata]
    from rsgispy.imageprocess import MemorySink, SetNoData

    nir = RasterSource(s2_img, bands=[k for k, v in band_names.items()
                                      if v == "NIR"])
    nir_med = median_filter_node(nir, 5)  # the file's no data value (0)
    nir_med_nd = median_filter_node(nir, 5, no_data_val=0)  # explicit
    nir_med_all = median_filter_node(SetNoData(nir, None), 5)  # 0 is a value
    nd_res = TileGraph(
        [MemorySink(nir_med, name="meta"), MemorySink(nir_med_nd, name="explicit"),
         MemorySink(nir_med_all, name="ignored")],
        block_size=block_size,
    ).run(use_pbar=use_pbar)
    # [end-step-6-nodata]
    out["nodata"] = nd_res

    # [step-7-halos]
    # Median 3 x 3, mean 5 x 5 and Sobel: a halo of 1 + 2 + 1 = 4 pixels.
    ndvi_edges = build_filter_chain_node(ndvi, [
        FilterStep("median", filter_size=3),
        FilterStep("mean", filter_size=5),
        FilterStep("sobel"),
    ])
    by_block = {
        size: TileGraph(MemorySink(ndvi_edges, name="edges"), block_size=size)
        .run(use_pbar=False)["edges"]
        for size in (64, 256)
    }
    np.testing.assert_array_equal(by_block[64], by_block[256])
    # [end-step-7-halos]
    out["halo_graph"] = TileGraph(MemorySink(ndvi_edges)).describe()

    # [step-8-windowop]
    from numpy.lib.stride_tricks import sliding_window_view
    from rsgispy.imageprocess import WindowOp, same_bands_meta

    def veg_fraction(arr, ctx):
        """Fraction of valid pixels in a 5 x 5 window with NDVI > 0.5."""
        win = sliding_window_view(arr, (5, 5), axis=(1, 2))  # (C, H, W, 5, 5)
        valid = np.isfinite(win)
        if ctx.nodata is not None:
            valid &= win != ctx.nodata
        n_valid = valid.sum(axis=(-2, -1))
        n_veg = (valid & (win > 0.5)).sum(axis=(-2, -1))
        return np.where(n_valid > 0, n_veg / np.maximum(n_valid, 1), -1.0)

    veg_frac = WindowOp(ndvi, 2, veg_fraction, pass_ctx=True,
                        meta_fn=same_bands_meta(np.float32, nodata=-1.0))
    # [end-step-8-windowop]

    # [step-8-pointwiseop]
    from rsgispy.imageprocess import PointwiseOp

    def steep_vegetation(frac, slope_deg):
        """1 where mostly vegetated and steeper than 15 degrees, 255 = no data."""
        flag = ((frac > 0.5) & (slope_deg > 15.0)).astype(np.uint8)
        return np.where((frac < 0) | ~np.isfinite(slope_deg), 255, flag)

    steep_veg = PointwiseOp([veg_frac, slope], steep_vegetation,
                            meta_fn=same_bands_meta(np.uint8, nodata=255))
    steep_veg_img = os.path.join(out_dir, "steep_vegetation.tif")
    TileGraph(DiskSink(steep_veg, steep_veg_img),  # uint8, from meta_fn
              block_size=block_size).run(use_pbar=use_pbar, n_jobs=n_jobs)
    # [end-step-8-pointwiseop]
    out["steep_veg_img"] = steep_veg_img

    # [step-8-tilenode]
    from rsgispy.imageprocess import TileNode

    class GainOffset(TileNode):
        """Reusable node: ``value * gain + offset``, keeping no data."""

        def __init__(self, source, gain, offset=0.0, name=None):
            super().__init__(inputs=(source,), name=name)
            self.gain = gain
            self.offset = offset

        def infer_meta(self, inputs):
            return inputs[0].replace(dtype=np.dtype(np.float32))

        def evaluate(self, inputs, halo_y, halo_x, ctx, state=None):
            tile = inputs[0].crop(halo_y, halo_x)
            data = tile.data * self.gain + self.offset
            if ctx.nodata is not None:
                data = np.where(tile.data == ctx.nodata, ctx.nodata, data)
            return tile.with_data(data.astype(np.float32))

    ndvi_pct = GainOffset(ndvi, 100.0)  # NDVI x 100
    # [end-step-8-tilenode]
    out["ndvi_pct"] = TileGraph(MemorySink(ndvi_pct, name="pct"),
                                block_size=block_size).run(use_pbar=False)["pct"]

    # [step-9-stats]
    from rsgispy.imageprocess import StatsSink, ZonalStatsSink

    stats = TileGraph(
        [
            StatsSink(indices, bins=20, hist_range=(-1.0, 1.0), name="indices"),
            ZonalStatsSink(RasterSource(cls_img), indices, name="per_class"),
        ],
        block_size=block_size,
    ).run(use_pbar=use_pbar, n_jobs=n_jobs)
    print(stats["indices"].to_dataframe())  # count, min, max, mean, std
    print(stats["per_class"])  # NDVI and NDWI statistics per class
    # [end-step-9-stats]
    out["stats"] = stats

    # [step-9-two-pass]
    from rsgispy.imagecalc.rescale import img_norm_params_node, norm_params_from_stats

    # Pass 1 (above): StatsSink statistics of the indices.
    z_params = norm_params_from_stats(stats["indices"], method="zscore")
    # Pass 2: apply them in a node.
    indices_z = img_norm_params_node(indices, z_params)
    indices_z_img = os.path.join(out_dir, "indices_zscore.tif")
    TileGraph(DiskSink(indices_z, indices_z_img),
              block_size=block_size).run(use_pbar=use_pbar, n_jobs=n_jobs)
    # [end-step-9-two-pass]
    out["indices_z_img"] = indices_z_img

    # [step-10-parallel]
    chain = {"cls": cls_smooth, "slope": slope}
    results = {
        jobs: TileGraph([MemorySink(node, name=key) for key, node in chain.items()],
                        block_size=block_size).run(use_pbar=False, n_jobs=jobs)
        for jobs in (1, 2)
    }
    for key in chain:  # identical for any number of threads
        np.testing.assert_array_equal(results[1][key], results[2][key])
    try:
        import numba

        print("numba threading layer:", numba.threading_layer())
    except (ImportError, ValueError):  # no numba, or no parallel kernel run yet
        pass
    # [end-step-10-parallel]

    # [step-11-check]
    import rasterio
    from rsgispy.classification.classpostprocess import apply_majority_filter
    from rsgispy.imagecalc.calcindices import calc_indices
    from rsgispy.imagefilter.stats import apply_median_filter, apply_stddev_filter
    from rsgispy.ml.predict import predict_raster

    chk_dir = os.path.join(out_dir, "file_steps")
    os.makedirs(chk_dir, exist_ok=True)
    step_imgs = {k: os.path.join(chk_dir, f"{k}.tif")
                 for k in ("indices", "med", "sd", "cls", "cls_smooth")}
    calc_indices(s2_img, step_imgs["indices"], band_names, ["NDVI", "NDWI"],
                 use_pbar=use_pbar)
    apply_median_filter(step_imgs["indices"], step_imgs["med"], 3, use_pbar=use_pbar)
    apply_stddev_filter(step_imgs["indices"], step_imgs["sd"], 5, use_pbar=use_pbar)
    predict_raster(clf, [step_imgs["med"], step_imgs["sd"]], step_imgs["cls"],
                   thematic=True, use_pbar=use_pbar)
    apply_majority_filter(step_imgs["cls"], step_imgs["cls_smooth"], 3,
                          use_pbar=use_pbar)

    with rasterio.open(cls_img) as graph_ds, \
            rasterio.open(step_imgs["cls_smooth"]) as file_ds:
        np.testing.assert_array_equal(graph_ds.read(), file_ds.read())
    with rasterio.open(feats_img) as graph_ds, \
            rasterio.open(step_imgs["med"]) as med_ds, \
            rasterio.open(step_imgs["sd"]) as sd_ds:
        np.testing.assert_array_equal(
            graph_ds.read(), np.concatenate([med_ds.read(), sd_ds.read()]))
    # [end-step-11-check]
    out["step_imgs"] = step_imgs
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--dem", required=True, help="DEM on the same grid")
    parser.add_argument("--train", required=True,
                        help="Training labels (class IDs, 0 = unlabelled)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--band-names", nargs="+", default=None,
                        help="BAND=NAME entries, e.g. 1=Blue 2=Green 3=Red 7=NIR")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Tile size in pixels (Default = rsgispy default)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Threads used to evaluate the tiles")
    args = parser.parse_args()
    band_names = None
    if args.band_names:
        band_names = {}
        for entry in args.band_names:
            band, name = entry.split("=", 1)
            band_names[int(band)] = name
    run_workflow(args.s2, args.dem, args.train, args.out_dir,
                 band_names=band_names, block_size=args.block_size,
                 n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
