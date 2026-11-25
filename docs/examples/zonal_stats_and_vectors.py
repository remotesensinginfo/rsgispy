"""
Worked example: zonal statistics and vector analysis recipes with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/zonal_stats_and_vectors.rst`` and is run end-to-end by the test
suite (``tests/test_zonal_stats_and_vectors_example.py``) on small synthetic
data, so the documented code is always tested.

Run it on your own data with, e.g.::

    python zonal_stats_and_vectors.py --s2 s2_sref_10m.tif --fields fields.gpkg \\
        --plots survey_plots.gpkg --roi district.gpkg --out-dir outputs
"""

import argparse
import os
from typing import Any, Dict, Sequence


def run_workflow(
    s2_img: str,
    fields_vec: str,
    plots_vec: str,
    roi_vec: str,
    out_dir: str,
    s2_bands: Sequence[int] = (2, 3, 4, 8),
    s2_band_names: Sequence[str] = ("Blue", "Green", "Red", "NIR"),
    field_id_col: str = "field_id",
    hex_res: int = 8,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``s2_img`` is a Sentinel-2 surface reflectance
    stack (reflectance x 10000, no data value 0) and ``s2_bands`` the 1-based
    indices of its blue, green, red and NIR bands. ``fields_vec`` holds
    non-overlapping field polygons with a unique integer ``field_id_col``;
    ``plots_vec`` holds survey points (``crop`` column) and ``roi_vec`` a
    region of interest polygon. All layers share the image's projected CRS.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-ndvi]
    import rsgispy
    import rsgispy.imagecalc.calcindices

    red_band, nir_band = s2_bands[2], s2_bands[3]
    ndvi_img = os.path.join(out_dir, "s2_ndvi.tif")
    rsgispy.imagecalc.calcindices.calc_indices(
        s2_img,
        ndvi_img,
        band_names={red_band: "red", nir_band: "nir"},
        indices=["NDVI"],
        in_gain=10000,  # reflectance x 10000 -> 0..1
        no_data_val=0,  # input no data value
        out_no_data_val=-999,  # NDVI no data value
        use_pbar=use_pbar,
    )
    # [end-step-1-ndvi]
    out["ndvi_img"] = ndvi_img

    # [step-2-polygon-stats]
    import rsgispy.zonalstats.polygon

    fields_ndvi_vec = os.path.join(out_dir, "fields_ndvi.gpkg")
    rsgispy.zonalstats.polygon.calc_zonal_band_stats_file(
        fields_vec,
        None,  # layer name (None = the only/first layer)
        ndvi_img,
        img_band=1,
        min_thres=-1.0,  # only pixels with -1 <= value <= 1 are used,
        max_thres=1.0,  # so the -999 no data pixels are ignored
        out_no_data_val=-999,  # value for fields with no valid pixels
        mean_field="ndvi_mean",
        stddev_field="ndvi_std",
        min_field="ndvi_min",
        max_field="ndvi_max",
        count_field="ndvi_count",
        use_pbar=use_pbar,
        out_vec_file=fields_ndvi_vec,  # without this the input is updated
        out_vec_lyr="fields",
    )
    # [end-step-2-polygon-stats]
    out["fields_ndvi_vec"] = fields_ndvi_vec

    # [step-3-point-values]
    import rsgispy.zonalstats.point

    plots_vals_vec = os.path.join(out_dir, "plots_s2.gpkg")
    rsgispy.zonalstats.point.ext_point_band_values_file(
        plots_vec,
        None,
        s2_img,
        img_band=list(s2_bands),
        min_thres=1,  # 0 is the no data value
        max_thres=10000,
        out_no_data_val=-1,  # value for points outside the image or on no data
        out_field=list(s2_band_names),  # one column per band
        use_pbar=use_pbar,
        out_vec_file=plots_vals_vec,
        out_vec_lyr="plots",
    )
    # [end-step-3-point-values]
    out["plots_vals_vec"] = plots_vals_vec

    # [step-4-label-stats]
    import geopandas as gpd
    import rsgispy.imageutils.rasterise
    import rsgispy.zonalstats.labels
    from rsgispy.ml.stack import BandSpec

    # Burn the field IDs into a label raster on the image's pixel grid.
    fields_lbl_img = os.path.join(out_dir, "fields_labels.tif")
    rsgispy.imageutils.rasterise.rasterise_vec_lyr(
        fields_vec,
        None,
        ndvi_img,  # reference image: grid, extent and CRS
        fields_lbl_img,
        att_column=field_id_col,
        datatype=rsgispy.TYPE_32UINT,
        no_data_val=0,  # pixels outside every field
        use_pbar=use_pbar,
    )
    stats_df = rsgispy.zonalstats.labels.calc_label_zonal_stats(
        fields_lbl_img,
        [
            BandSpec(s2_img, name="s2", bands=tuple(s2_bands),
                     band_names=tuple(s2_band_names), no_data_val=0),
            BandSpec(ndvi_img, name="ndvi", band_names=("ndvi",),
                     no_data_val=-999),
        ],
        stats=["mean", "stddev", "median", "count"],
        id_col=field_id_col,
        out_file=os.path.join(out_dir, "fields_stats.parquet"),
        use_pbar=use_pbar,
    )
    fields_stats_vec = os.path.join(out_dir, "fields_stats.gpkg")
    rsgispy.zonalstats.labels.join_label_stats_to_vec(
        fields_vec,
        stats_df,
        id_col=field_id_col,
        out_vec_file=fields_stats_vec,
        out_vec_lyr="fields",
        overwrite=True,
    )

    # Check: the label and polygon methods give the same field means.
    poly_gdf = gpd.read_file(fields_ndvi_vec).set_index(field_id_col)
    lbl_df = stats_df.set_index(field_id_col)
    max_diff = (poly_gdf["ndvi_mean"] - lbl_df["ndvi_mean"]).abs().max()
    print(f"Max difference polygon vs label NDVI mean: {max_diff:.6f}")
    # [end-step-4-label-stats]
    out["fields_stats_vec"] = fields_stats_vec
    out["stats_df"] = stats_df
    out["max_diff"] = float(max_diff)

    # [step-5-hex-grid]
    import rsgispy.vector.attrs
    import rsgispy.vector.create
    import rsgispy.vector.stats

    # H3 hexagons covering the region of interest, in the ROI's CRS.
    hex_gdf = rsgispy.vector.create.create_hex_grid_polys(
        roi_vec, None, hex_res, out_vec_file=None, use_pbar=use_pbar
    )
    # An integer ID next to the H3 cell index (hex_id).
    hex_gdf = rsgispy.vector.attrs.add_fid_col(hex_gdf, None, out_col="hex_fid")
    hex_gdf = rsgispy.zonalstats.polygon.calc_zonal_band_stats(
        hex_gdf,
        ndvi_img,
        img_band=1,
        min_thres=-1.0,
        max_thres=1.0,
        out_no_data_val=-999,
        mean_field="ndvi_mean",
        count_field="ndvi_count",
        use_pbar=use_pbar,
    )
    hex_vec = os.path.join(out_dir, "hex_ndvi.gpkg")
    rsgispy.vector.stats.count_pt_intersects(
        hex_gdf,
        None,
        plots_vec,
        None,
        out_vec_file=hex_vec,
        out_vec_lyr="hex",
        out_count_col="n_plots",
    )
    # [end-step-5-hex-grid]
    out["hex_vec"] = hex_vec

    # [step-6-shape-metrics]
    fields_shape_vec = os.path.join(out_dir, "fields_shape.gpkg")
    rsgispy.vector.stats.calc_poly_shape_metrics(
        fields_stats_vec,
        "fields",
        metrics=["area", "perimeter", "compactness", "elongation", "solidity"],
        col_prefix="shp_",
        out_vec_file=fields_shape_vec,
        out_vec_lyr="fields",
        overwrite=True,
    )
    # [end-step-6-shape-metrics]
    out["fields_shape_vec"] = fields_shape_vec

    # [step-7-spatial-select]
    import rsgispy.vector.selection

    # Fields lying completely inside the region of interest.
    roi_fields_gdf = rsgispy.vector.selection.spatial_select(
        fields_shape_vec,
        "fields",
        roi_vec,
        None,
        out_vec_file=None,  # return a GeoDataFrame
        predicate="within",
    )
    print(f"{len(roi_fields_gdf)} fields within the region of interest")
    # [end-step-7-spatial-select]
    out["roi_fields_gdf"] = roi_fields_gdf

    # [step-8-train-test]
    train_vec = os.path.join(out_dir, "plots_train.gpkg")
    test_vec = os.path.join(out_dir, "plots_test.gpkg")
    rsgispy.vector.selection.create_train_test_smpls(
        plots_vals_vec,
        "plots",
        out_train_vec_file=train_vec,
        out_train_vec_lyr="plots",
        out_test_vec_file=test_vec,
        out_test_vec_lyr="plots",
        prop_test=0.3,  # 30 % of the points for testing
        rnd_seed=42,  # reproducible split
        del_exist_vec=True,
    )
    # [end-step-8-train-test]
    out["train_vec"] = train_vec
    out["test_vec"] = test_vec
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Sentinel-2 SREF stack")
    parser.add_argument("--fields", required=True, help="Field polygons")
    parser.add_argument("--plots", required=True, help="Survey points")
    parser.add_argument("--roi", required=True, help="Region of interest polygon")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--field-id-col", default="field_id",
                        help="Unique integer field ID column")
    parser.add_argument("--hex-res", type=int, default=8,
                        help="H3 resolution of the hexagon grid")
    args = parser.parse_args()
    run_workflow(args.s2, args.fields, args.plots, args.roi, args.out_dir,
                 field_id_col=args.field_id_col, hex_res=args.hex_res)


if __name__ == "__main__":
    main()
