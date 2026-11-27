"""
Worked example: hydrological analysis of a DEM with rsgispy.

This script is included step by step in the documentation guide
``docs/guides/hydrology_workflow.rst`` and is run end-to-end by the test
suite (``tests/test_hydrology_workflow_example.py``) on a small synthetic
DEM, so the documented code is always tested.

Run it on your own data with, e.g.::

    python hydrology_workflow.py --dem copernicus_glo30.tif --out-epsg 32630 \\
        --out-res 30 --gauges gauges.gpkg --gauge-id-field gauge_id \\
        --rivers mapped_rivers.gpkg --landcover landcover.tif \\
        --stream-threshold 500000 --out-dir hydro_out
"""

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Sequence


def run_workflow(
    dem_img: str,
    gauges_vec: str,
    out_dir: str,
    out_epsg: Optional[int] = None,
    out_res: Optional[float] = None,
    gauge_id_field: str = "gauge_id",
    rivers_vec: Optional[str] = None,
    landcover_img: Optional[str] = None,
    lc_names: Optional[Mapping[int, str]] = None,
    max_breach_depth: float = 5.0,
    max_breach_length: int = 20,
    stream_threshold: float = 50000.0,
    snap_dist: float = 50.0,
    flood_stages: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    make_map: bool = True,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Run the worked example. ``dem_img`` is a DEM in metres; if ``out_epsg``
    is given it is reprojected to that (projected, metric) CRS first.
    ``gauges_vec`` holds gauge/pour points with an integer ``gauge_id_field``,
    ``rivers_vec`` optional mapped river lines to burn into the DEM and
    ``landcover_img`` an optional land cover raster (``lc_names`` maps its
    values to names) summarised per catchment.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-prepare]
    import rasterio
    import rsgispy
    from rsgispy.imagecalc.stats import calc_basic_img_band_stats
    from rsgispy.imageutils.metadata import get_img_res
    from rsgispy.imageutils.warp import reproject_image

    if out_epsg is not None:
        # Elevation is continuous: use bilinear (or cubic) interpolation,
        # never nearest neighbour, which creates artificial steps and flats.
        dem_utm = os.path.join(out_dir, "dem_utm.tif")
        reproject_image(
            dem_img,
            dem_utm,
            out_epsg=out_epsg,
            interp_method=rsgispy.INTERP_BILINEAR,
            # Pixel size in metres; 'auto' keeps about the input number of
            # pixels ('image' would copy a geographic DEM's size in degrees).
            out_pxl_res=out_res if out_res is not None else "auto",
            use_pbar=use_pbar,
        )
        dem_img = dem_utm

    with rasterio.open(dem_img) as src:
        if not (src.crs and src.crs.is_projected and src.crs.linear_units == "metre"):
            raise rsgispy.RSGISPyException(f"{dem_img} is not in a metric CRS.")
        print(f"CRS: {src.crs.to_string()}, size: {src.width} x {src.height}, "
              f"nodata: {src.nodata}")
    x_res, y_res = get_img_res(dem_img, abs_vals=True)
    z_min, z_max, z_mean, _ = calc_basic_img_band_stats(dem_img, use_pbar=False)
    print(f"Pixel size: {x_res:.1f} x {y_res:.1f} m; elevation "
          f"{z_min[0]:.1f} to {z_max[0]:.1f} m (mean {z_mean[0]:.1f} m)")
    # [end-step-1-prepare]
    out["dem_img"] = dem_img

    # [step-2-sinks]
    import numpy as np
    import rsgispy.elevation.hydrology as hydro

    sinks_img = os.path.join(out_dir, "sinks.tif")
    sink_depth_img = os.path.join(out_dir, "sink_depth.tif")
    hydro.identify_sinks(dem_img, sinks_img, output_depth_img=sink_depth_img,
                         use_pbar=use_pbar)
    with rasterio.open(sinks_img) as src:
        sink_ids = src.read(1)  # 1..K sink IDs, 0 = not a sink, -1 = nodata
    with rasterio.open(sink_depth_img) as src:
        sink_depth = src.read(1)  # metres of fill needed to remove each sink
    n_sinks = int(sink_ids.max())
    print(f"{n_sinks} sinks covering {np.count_nonzero(sink_ids > 0)} pixels; "
          f"deepest {sink_depth.max():.2f} m")
    # [end-step-2-sinks]
    out["n_sinks"] = n_sinks

    # [step-3-condition]
    filled_img = os.path.join(out_dir, "dem_filled.tif")
    breached_img = os.path.join(out_dir, "dem_breached.tif")
    cond_dem = os.path.join(out_dir, "dem_cond.tif")

    # For comparison: fill everything, or breach everything (no limits).
    hydro.fill_depressions(dem_img, filled_img, use_pbar=use_pbar)
    hydro.breach_depressions(dem_img, breached_img, use_pbar=use_pbar)

    # The DEM used from here on: burn the mapped rivers (if any), breach
    # shallow, short obstructions (embankments, bridges) and fill the rest.
    hydro.condition_dem(
        dem_img,
        cond_dem,
        method="breach_and_fill",
        max_breach_depth=max_breach_depth,  # metres
        max_breach_length=max_breach_length,  # pixels
        streams_vec=rivers_vec,  # optional mapped river lines
        burn_depth=2.0,  # metres lowered along the burnt rivers
        use_pbar=use_pbar,
    )
    # [end-step-3-condition]
    out["cond_dem"] = cond_dem

    # [step-3b-check-condition]
    from rsgispy.imagecalc.maths import BandDefn, band_maths, calc_prop_true_exp
    from rsgispy.imageutils.metadata import get_img_no_data_value

    dem_nd = get_img_no_data_value(dem_img)
    changes = {}
    for name, cond_img in (("fill", filled_img), ("breach", breached_img),
                           ("condition_dem", cond_dem)):
        diff_img = os.path.join(out_dir, f"dem_diff_{name}.tif")
        band_defs = [BandDefn("raw", dem_img, 1), BandDefn("cond", cond_img, 1)]
        # Positive = raised (filled), negative = lowered (breached or burnt).
        band_maths(diff_img, f"raw == {dem_nd} ? {dem_nd} : cond - raw", band_defs,
                   datatype=rsgispy.TYPE_32FLOAT, no_data_val=dem_nd,
                   use_pbar=use_pbar)
        d_min, d_max, _, _ = calc_basic_img_band_stats(diff_img, use_pbar=False)
        prop = calc_prop_true_exp(
            f"raw != {dem_nd} && abs(cond - raw) > 0.001 ? 1 : 0", band_defs,
            use_pbar=False,
        )
        changes[name] = prop
        print(f"{name}: {100 * prop:.2f}% of pixels changed, "
              f"from {d_min[0]:.2f} to {d_max[0]:.2f} m")

    # Filling leaves flat lake surfaces. flow_direction routes across them
    # itself; resolve_flats shows where they are (> 0 = flat cell, counting
    # down towards the flat's outlet; 0 = not flat).
    flats_img = os.path.join(out_dir, "flats_gradient_filled.tif")
    hydro.resolve_flats(filled_img, flats_img, use_pbar=use_pbar)
    with rasterio.open(flats_img) as src:
        n_flat = int(np.count_nonzero(src.read(1) > 0))
    print(f"Filled DEM: {n_flat} flat pixels")
    # [end-step-3b-check-condition]
    out["changes"] = changes
    out["n_flat"] = n_flat

    # [step-4-flow-routing]
    fdir_d8 = os.path.join(out_dir, "flow_dir_d8.tif")
    facc_d8 = os.path.join(out_dir, "flow_acc_d8_cells.tif")
    hydro.flow_direction(cond_dem, fdir_d8, method="d8", use_pbar=use_pbar)
    # In cells; the units are recorded in the file, so extract_streams can
    # take its threshold in m2 or in cells.
    hydro.flow_accumulation(fdir_d8, facc_d8, out_units="cells", use_pbar=use_pbar)

    # Dispersive routing for wetness and erosion indices (Step 9).
    fdir_dinf = os.path.join(out_dir, "flow_dir_dinf.tif")
    facc_dinf = os.path.join(out_dir, "flow_acc_dinf_area.tif")
    hydro.flow_direction(cond_dem, fdir_dinf, method="dinf", use_pbar=use_pbar)
    hydro.flow_accumulation(fdir_dinf, facc_dinf, out_units="area",
                            use_pbar=use_pbar)
    fdir_mfd = os.path.join(out_dir, "flow_dir_mfd.tif")
    facc_mfd = os.path.join(out_dir, "flow_acc_mfd_area.tif")
    hydro.flow_direction(cond_dem, fdir_mfd, method="mfd", mfd_exponent=1.1,
                         use_pbar=use_pbar)
    hydro.flow_accumulation(fdir_mfd, facc_mfd, out_units="area", use_pbar=use_pbar)

    # Longest upstream flow path through every pixel (metres).
    flow_len_up = os.path.join(out_dir, "flow_length_upstream.tif")
    hydro.flow_length(fdir_d8, flow_len_up, direction="upstream", use_pbar=use_pbar)

    for name, facc_img, to_m2 in (("D8", facc_d8, x_res * y_res),
                                  ("Dinf", facc_dinf, 1.0), ("MFD", facc_mfd, 1.0)):
        _, a_max, _, _ = calc_basic_img_band_stats(facc_img, use_pbar=False)
        print(f"{name}: largest contributing area {a_max[0] * to_m2 / 1e6:.3f} km2")
    # [end-step-4-flow-routing]
    out["fdir_d8"] = fdir_d8
    out["facc_d8"] = facc_d8

    # [step-5-streams]
    streams_img = os.path.join(out_dir, "streams.tif")
    links_img = os.path.join(out_dir, "stream_links.tif")
    order_img = os.path.join(out_dir, "stream_strahler.tif")
    streams_vec = os.path.join(out_dir, "streams.gpkg")

    # A pixel is a channel when it drains more than stream_threshold m2.
    hydro.extract_streams(facc_d8, streams_img, threshold=stream_threshold,
                          threshold_units="area", use_pbar=use_pbar)
    hydro.stream_link(streams_img, fdir_d8, links_img, use_pbar=use_pbar)
    hydro.stream_order(streams_img, fdir_d8, order_img, method="strahler",
                       use_pbar=use_pbar)
    hydro.vectorise_streams(
        streams_img,
        fdir_d8,
        streams_vec,
        dem_img=cond_dem,  # adds elev_start, elev_end, drop_m and slope
        flow_acc_img=facc_d8,  # adds max_acc (here in cells)
        order_method="strahler",
        use_pbar=use_pbar,
    )
    # Overland distance from every pixel to the stream it drains to (metres).
    dist_to_stream = os.path.join(out_dir, "dist_to_stream.tif")
    hydro.flow_distance(fdir_d8, streams_img, dist_to_stream,
                        distance_type="horizontal", use_pbar=use_pbar)

    import geopandas as gpd

    streams_gdf = gpd.read_file(streams_vec)
    print(f"{len(streams_gdf)} reaches, {streams_gdf['length_m'].sum():.0f} m "
          f"of channel, maximum Strahler order {streams_gdf['order'].max()}")
    # [end-step-5-streams]
    out["streams_vec"] = streams_vec

    # [step-6-catchments]
    basins_img = os.path.join(out_dir, "basins.tif")
    subcatch_img = os.path.join(out_dir, "subcatchments.tif")
    hydro.delineate_basins(fdir_d8, basins_img, use_pbar=use_pbar)
    hydro.delineate_subcatchments(fdir_d8, links_img, subcatch_img,
                                  use_pbar=use_pbar)
    hydro.polygonise_watersheds(subcatch_img,
                                os.path.join(out_dir, "subcatchments.gpkg"),
                                dem_img=cond_dem, use_pbar=use_pbar)

    gauges_snap = os.path.join(out_dir, "gauges_snapped.gpkg")
    wsheds_img = os.path.join(out_dir, "watersheds.tif")
    wsheds_vec = os.path.join(out_dir, "watersheds.gpkg")
    # Move each gauge onto the stream pixel with the largest
    # accumulation within snap_dist metres.
    hydro.snap_pour_points(gauges_vec, facc_d8, gauges_snap,
                           snap_dist=snap_dist, streams_img=streams_img,
                           use_pbar=use_pbar)
    hydro.delineate_watersheds(fdir_d8, gauges_snap, wsheds_img,
                               id_field=gauge_id_field, use_pbar=use_pbar)
    hydro.polygonise_watersheds(wsheds_img, wsheds_vec, dem_img=cond_dem,
                                use_pbar=use_pbar)
    # Main channel upstream of each gauge (follows the largest
    # accumulation up to the headwater).
    hydro.trace_flow_path(fdir_d8, gauges_snap,
                          os.path.join(out_dir, "gauge_main_stems.gpkg"),
                          direction="upstream", flow_acc_img=facc_d8,
                          use_pbar=use_pbar)
    print(gpd.read_file(gauges_snap).drop(columns="geometry").to_string())
    print(gpd.read_file(wsheds_vec).drop(columns="geometry").round(1).to_string())
    # [end-step-6-catchments]
    out["gauges_snap"] = gauges_snap
    out["wsheds_img"] = wsheds_img
    out["wsheds_vec"] = wsheds_vec

    # [step-7-catchment-stats]
    from rsgispy.elevation.terrain import slope
    from rsgispy.imageutils.warp import resample_img_to_match
    from rsgispy.ml.stack import BandSpec
    from rsgispy.zonalstats.labels import calc_label_zonal_stats

    slope_img = os.path.join(out_dir, "slope_deg.tif")
    slope(cond_dem, slope_img, unit="degrees", use_pbar=use_pbar)
    band_specs = [BandSpec(slope_img, name="slope", band_names=("slope",)),
                  BandSpec(dist_to_stream, name="dist",
                           band_names=("dist_stream",))]
    if landcover_img is not None:
        # Categorical data: nearest neighbour onto the DEM grid.
        lc_grid = os.path.join(out_dir, "landcover_grid.tif")
        resample_img_to_match(cond_dem, landcover_img, lc_grid,
                              interp_method=rsgispy.INTERP_NEAREST_NEIGHBOUR,
                              use_pbar=use_pbar)
        band_specs.append(BandSpec(lc_grid, name="lc", band_names=("lc",),
                                   data_type=rsgispy.VAR_TYPE_CATEGORICAL,
                                   no_data_val=0))
    ws_stats = calc_label_zonal_stats(
        wsheds_img,
        band_specs,
        stats=["mean", "max"],
        id_col="watershed_id",
        use_pbar=use_pbar,
    )
    # Drop the nodata (-1) zone (the background, 0, is excluded by default).
    ws_stats = ws_stats[ws_stats["watershed_id"] > 0]
    if lc_names:
        ws_stats = ws_stats.rename(
            columns={f"lc_prop_{val}": f"prop_{name.lower()}"
                     for val, name in lc_names.items()}
        )
    wsheds_gdf = gpd.read_file(wsheds_vec).merge(ws_stats, on="watershed_id")
    wsheds_gdf.to_file(wsheds_vec)
    print(ws_stats.round(3).to_string())
    # [end-step-7-catchment-stats]
    out["ws_stats"] = ws_stats

    # [step-8-hand-flood]
    hand_img = os.path.join(out_dir, "hand.tif")
    hydro.calc_hand(cond_dem, fdir_d8, streams_img, hand_img, use_pbar=use_pbar)

    with rasterio.open(hand_img) as src:
        pxl_area = abs(src.res[0] * src.res[1])
        n_pxls = src.width * src.height
    flood_area = {}
    for stage in flood_stages:
        depth_img = os.path.join(out_dir, f"flood_depth_{stage:g}m.tif")
        hydro.flood_inundation_from_hand(hand_img, depth_img, water_depth=stage,
                                         use_pbar=use_pbar)
        prop_wet = calc_prop_true_exp("depth > 0 ? 1 : 0",
                                      [BandDefn("depth", depth_img, 1)],
                                      use_pbar=False)
        flood_area[stage] = prop_wet * n_pxls * pxl_area / 1e4  # hectares
        print(f"Stage {stage:g} m: {flood_area[stage]:.2f} ha inundated")
    # [end-step-8-hand-flood]
    out["hand_img"] = hand_img
    out["flood_area_ha"] = flood_area

    # [step-9-terrain-indices]
    sca_img = os.path.join(out_dir, "sca_dinf.tif")
    twi_img = os.path.join(out_dir, "twi.tif")
    spi_img = os.path.join(out_dir, "spi.tif")
    ls_img = os.path.join(out_dir, "ls_factor.tif")

    # Specific catchment area (m2/m): contributing area divided by the
    # contour width, taken as the pixel size.
    acc_nd = get_img_no_data_value(facc_dinf)
    band_maths(sca_img, f"acc == {acc_nd} ? {acc_nd} : acc / {x_res}",
               [BandDefn("acc", facc_dinf, 1)], datatype=rsgispy.TYPE_32FLOAT,
               no_data_val=acc_nd, use_pbar=use_pbar)
    hydro.calc_twi(sca_img, slope_img, twi_img, slope_units="degrees",
                   min_slope=0.001, use_pbar=use_pbar)
    hydro.calc_spi(sca_img, slope_img, spi_img, slope_units="degrees",
                   use_pbar=use_pbar)
    hydro.calc_ls_factor(sca_img, slope_img, ls_img, slope_units="degrees",
                         m=0.4, n=1.3, use_pbar=use_pbar)
    for name, idx_img in (("TWI", twi_img), ("SPI", spi_img), ("LS", ls_img)):
        i_min, i_max, i_mean, _ = calc_basic_img_band_stats(idx_img, use_pbar=False)
        print(f"{name}: {i_min[0]:.2f} to {i_max[0]:.2f} (mean {i_mean[0]:.2f})")
    # [end-step-9-terrain-indices]
    out["twi_img"] = twi_img

    if not make_map:
        return out

    # [step-10-map]
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    from rsgispy.elevation.illumination import hillshade
    from rsgispy.imageutils.metadata import get_img_bbox
    from rsgispy.mapping.raster import create_raster_img_map
    from rsgispy.mapping.vector import create_vec_lyr_map

    hshade_img = os.path.join(out_dir, "hillshade.tif")
    hillshade(cond_dem, hshade_img, azimuth=315.0, zenith=45.0, use_pbar=use_pbar)
    # Zoom on the gauged watersheds, with a 5% margin, within the DEM.
    dem_bbox = get_img_bbox(cond_dem)  # [MinX, MaxX, MinY, MaxY]
    ws_minx, ws_miny, ws_maxx, ws_maxy = wsheds_gdf.total_bounds
    margin = 0.05 * max(ws_maxx - ws_minx, ws_maxy - ws_miny)
    bbox = [max(ws_minx - margin, dem_bbox[0]), min(ws_maxx + margin, dem_bbox[1]),
            max(ws_miny - margin, dem_bbox[2]), min(ws_maxy + margin, dem_bbox[3])]

    fig, ax = plt.subplots(figsize=(7, 6))
    create_raster_img_map(ax, hshade_img, img_bands=[1],
                          img_stch=rsgispy.IMG_STRETCH_LINEAR, bbox=bbox,
                          show_scale_bar=False, show_map_axis=False)
    create_vec_lyr_map(
        ax,
        [gpd.read_file(os.path.join(out_dir, "subcatchments.gpkg")),
         wsheds_gdf, streams_gdf,
         gpd.read_file(os.path.join(out_dir, "gauge_main_stems.gpkg")),
         gpd.read_file(gauges_snap)],
        bbox,
        title_str="Streams, subcatchments and gauged watersheds",
        # Polygons with the fill colour "none" are drawn as outlines; lines
        # (the streams and main stems) are drawn in the line colour.
        vec_fill_clrs=["none", "none", "none", "none", "#e31a1c"],
        vec_line_clrs=["#666666", "#e31a1c", "#1f78b4", "#08306b", "#000000"],
        vec_line_widths=[0.5, 1.5, 1.0, 2.0, 0.5],
        vec_markersize=40,
        plot_zorders=[2, 3, 4, 5, 6],
        show_map_axis=False,
        show_north_arrow=True,
    )
    map_png = os.path.join(out_dir, "hydrology_map.png")
    fig.savefig(map_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    # [end-step-10-map]
    out["map_png"] = map_png
    return out


def main() -> None:
    """Command line interface for the worked example."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--dem", required=True, help="Input DEM (metres)")
    parser.add_argument("--gauges", required=True, help="Gauge/pour point vector")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--out-epsg", type=int, default=None,
                        help="Reproject the DEM to this projected EPSG code")
    parser.add_argument("--out-res", type=float, default=None,
                        help="Output pixel size (m) when reprojecting")
    parser.add_argument("--gauge-id-field", default="gauge_id",
                        help="Integer gauge ID column")
    parser.add_argument("--rivers", default=None, help="Mapped rivers to burn in")
    parser.add_argument("--landcover", default=None, help="Land cover raster")
    parser.add_argument("--stream-threshold", type=float, default=50000.0,
                        help="Channel initiation area (m2)")
    parser.add_argument("--snap-dist", type=float, default=50.0,
                        help="Gauge snapping distance (m)")
    args = parser.parse_args()
    run_workflow(args.dem, args.gauges, args.out_dir, out_epsg=args.out_epsg,
                 out_res=args.out_res, gauge_id_field=args.gauge_id_field,
                 rivers_vec=args.rivers, landcover_img=args.landcover,
                 stream_threshold=args.stream_threshold, snap_dist=args.snap_dist)


if __name__ == "__main__":
    main()
