"""
Worked example: making maps with rsgispy.mapping.

This script is included step by step in the documentation guide
``docs/guides/making_maps.rst`` and is run end-to-end by the test suite
(``tests/test_making_maps_example.py``) on small synthetic data, so the
documented code is always tested.

Run it on your own data with, e.g.::

    python making_maps.py --s2 s2_sref_10m.tif --cls land_cover.tif \\
        --score restoration_score.tif --regions admin_units.gpkg \\
        --rivers rivers.gpkg --overview countries.gpkg \\
        --classes cls_config.json --out-dir maps --basemap
"""

import argparse
import os
from typing import Any, Dict, Optional, Sequence


def run_workflow(
    s2_img: str,
    cls_img: str,
    score_img: str,
    regions_vec: str,
    rivers_vec: str,
    overview_vec: str,
    out_dir: str,
    classes_info: Any,
    rgb_bands: Sequence[int] = (3, 2, 1),
    fig_width: float = 6.7,
    tick_spacing: Optional[float] = None,
    dpi: int = 300,
    basemap: bool = False,
    wmts_url: Optional[str] = None,
    wmts_lyr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the worked example. ``s2_img`` is a surface reflectance stack,
    ``cls_img`` a land cover raster with a colour table, ``score_img`` a
    continuous single-band raster (e.g. a 0-1 score), ``regions_vec`` reporting
    polygons, ``rivers_vec`` a line layer and ``overview_vec`` a small-scale
    layer (e.g. country outlines) for the inset map. ``classes_info`` gives the
    class names (``{pixel value: name}``, a ``ClassificationConfig`` or its
    JSON file). Figures are written to ``out_dir``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, Any] = {}

    # [step-1-layout]
    import matplotlib.pyplot as plt
    import rsgispy.imageutils.metadata
    import rsgispy.mapping.layout as map_layout

    # (MinX, MaxX, MinY, MaxY) in the image CRS (projected, metres)
    bbox = rsgispy.imageutils.metadata.get_img_bbox(s2_img)
    img_epsg = bbox.epsg
    width_m = bbox[1] - bbox[0]
    if tick_spacing is None:
        tick_spacing = float(f"{width_m / 4:.0e}")  # about 4 ticks, rounded
    # Figure height that matches the map's aspect ratio for a given width
    # (6.7 inches is about the full text width of a journal page).
    fig_height = map_layout.calc_y_fig_size(bbox, fig_width)
    print(f"Map extent {width_m:.0f} m wide; figure {fig_width} x "
          f"{fig_height:.2f} in; ticks every {tick_spacing:g} m")

    # A zoom box: the central quarter of the image.
    x_mid = (bbox[0] + bbox[1]) / 2
    y_mid = (bbox[2] + bbox[3]) / 2
    zoom_bbox = (x_mid - width_m / 4, x_mid + width_m / 4,
                 y_mid - (bbox[3] - bbox[2]) / 4, y_mid + (bbox[3] - bbox[2]) / 4)
    # [end-step-1-layout]
    out["fig_height"] = fig_height

    # [step-2-rgb]
    import rsgispy
    import rsgispy.mapping.elements as map_elements
    import rsgispy.mapping.raster as map_raster

    stretches = [
        ("a) Linear (min-max)", rsgispy.IMG_STRETCH_LINEAR, {}),
        ("b) 2 standard deviations", rsgispy.IMG_STRETCH_STDEV,
         {"stch_n_stdevs": 2.0}),
        ("c) Cumulative 2-98 %", rsgispy.IMG_STRETCH_CUMULATIVE, {}),
        ("d) User 0-1500", rsgispy.IMG_STRETCH_USER,
         {"stch_min_max_vals": [{"min": 0, "max": 1500}] * 3}),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    for ax, (label, stretch, kwargs) in zip(axs.ravel(), stretches):
        map_raster.create_raster_img_map(
            ax, s2_img, img_bands=list(rgb_bands), img_stch=stretch,
            show_scale_bar=False, show_map_axis=False,
            no_data_out_clr=(1.0, 1.0, 1.0),  # no data drawn white
            **kwargs,
        )
        map_elements.add_axis_label(ax, label, lbl_font_size=8)
    fig.tight_layout()
    out["rgb_stretches"] = os.path.join(out_dir, "maps_rgb_stretches.png")
    fig.savefig(out["rgb_stretches"], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    map_raster.create_raster_img_map(
        ax, s2_img, img_bands=list(rgb_bands),
        img_stch=rsgispy.IMG_STRETCH_CUMULATIVE,
        title_str="Sentinel-2 true colour",
        scale_bar_loc="lower right",
        show_north_arrow=True,
        no_data_out_clr=(1.0, 1.0, 1.0),
    )
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    out["rgb_map"] = os.path.join(out_dir, "maps_rgb.png")
    fig.savefig(out["rgb_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # [end-step-2-rgb]

    # [step-3-continuous]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    map_raster.create_raster_cmap_img_map(
        ax, score_img, img_band=1,
        cmap_name="viridis",  # perceptually uniform, colour-blind safe
        norm_img_vals=True, norm_vmin=0.0, norm_vmax=1.0,  # fixed range
        title_str="Score", scale_bar_loc="lower right",
        show_colour_bar=True, cbar_label="Score (0-1)",
    )
    # The image artist is returned, so you can also add the colour bar
    # yourself (e.g. one bar for several panels): fig.colorbar(img).
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    out["score_map"] = os.path.join(out_dir, "maps_score.png")
    fig.savefig(out["score_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # [end-step-3-continuous]

    # [step-4-thematic]
    import rsgispy.mapping.legend as map_legend

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    map_raster.create_thematic_raster_map(
        ax, cls_img,
        title_str="Land cover",
        scale_bar_loc="lower right",
        show_legend=True,
        classes_info=classes_info,  # names; colours come from the colour table
        legend_title="Land cover",
        legend_position="bottom",  # companion axis below the map
        legend_kwargs={"only_present_in_img": True},
    )
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    out["cls_map"] = os.path.join(out_dir, "maps_land_cover.png")
    fig.savefig(out["cls_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # The same legend as a stand-alone image (e.g. for slides or a poster).
    out["cls_legend"] = os.path.join(out_dir, "maps_land_cover_legend.png")
    map_legend.save_categorical_legend_img(
        out["cls_legend"], classes_info=classes_info, input_img=cls_img,
        title="Land cover", ncols=4, fig_width=fig_width, dpi=dpi,
    )
    # [end-step-4-thematic]

    # [step-5-vectors]
    import geopandas as gpd
    import matplotlib.colors
    import rsgispy.mapping.vector as map_vector
    import rsgispy.zonalstats.polygon

    regions = gpd.read_file(regions_vec)
    rivers = gpd.read_file(rivers_vec)

    # Vector layers drawn over the image: outlines only for the regions.
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    map_raster.create_raster_img_map(
        ax, s2_img, img_bands=list(rgb_bands),
        img_stch=rsgispy.IMG_STRETCH_CUMULATIVE, show_scale_bar=False,
    )
    map_vector.create_vec_lyr_map(
        ax, [regions, rivers], bbox,
        # Polygons: fill and outline colours ("none": outlines only).
        # Lines (the rivers) are drawn in the line colour.
        vec_fill_clrs=["none", "none"],
        vec_line_clrs=["yellow", "deepskyblue"],
        vec_line_widths=[1.0, 2.0],
        plot_zorders=[2, 3],  # above the image (z-order 1)
        title_str="Regions and rivers", scale_bar_loc="lower right",
    )
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    out["vec_map"] = os.path.join(out_dir, "maps_vectors.png")
    fig.savefig(out["vec_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Choropleth: percentage of forest (class 2) in each region.
    regions = rsgispy.zonalstats.polygon.calc_zonal_band_pxl_counts(
        regions, cls_img, img_band=1, vals_lut={2: "forest_pxls"},
        total_field="total_pxls", use_pbar=False,
    )
    regions["forest_pct"] = 100.0 * regions["forest_pxls"] / regions["total_pxls"]
    print(regions[["name", "forest_pct"]])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    pct_norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    map_vector.create_choropleth_vec_lyr_map(
        ax, regions, "forest_pct", bbox,
        vec_fill_cmap="Greens", vec_var_norm=pct_norm, vec_line_clr="black",
        vec_line_width=0.5, title_str="Forest cover by region",
        scale_bar_loc="lower right",
        show_colour_bar=True, cbar_label="Forest (%)",
    )
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    out["choropleth_map"] = os.path.join(out_dir, "maps_choropleth.png")
    fig.savefig(out["choropleth_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # [end-step-5-vectors]
    out["regions"] = regions

    # [step-6-elements]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    map_raster.create_thematic_raster_map(
        ax, cls_img, show_scale_bar=False,  # added below with more control
    )
    map_vector.create_vec_lyr_map(
        ax, rivers, bbox, vec_line_clrs="navy", vec_line_widths=1.5,
        plot_zorders=2, show_scale_bar=False,
    )
    map_elements.add_scale_bar(
        ax, epsg_code=img_epsg, loc="lower right",
        style=map_elements.SCALEBAR_STYLE_ALTERNATING,
        units=map_elements.SCALEBAR_UNIT_METRES, n_segments=4,
    )
    map_elements.add_north_arrow(
        ax, loc="upper left", style=map_elements.NORTH_ARROW_STYLE_COMPASS,
        size=30,
    )
    # Outline (and label) the area shown in a zoomed-in panel.
    map_elements.draw_bboxes_to_axis(
        ax, [zoom_bbox], bbox_labels=["Zoom"], rect_clr="red", line_width=1.5,
        lbl_pos="above", lbl_font_size=9, lbl_font_clr="red",
        lbl_fill_clr="white",
    )
    map_layout.define_map_tick_spacing(ax, tick_spacing)
    # [end-step-6-elements]

    # [step-7-overview]
    overview = gpd.read_file(overview_vec)
    over_epsg = overview.crs.to_epsg()
    # Inset extent: +/- overview_buf (in the overview layer's units, here
    # degrees) around the centre of the map, reprojected from the map CRS.
    over_bbox, centre_pt = map_layout.get_overview_info(
        bbox, roi_epsg=img_epsg, overview_buf=5.0, out_epsg=over_epsg,
    )
    map_layout.add_overview_maps(
        ax, overview, over_bbox, overview_pt=centre_pt,
        over_size_x=0.28, over_size_y=0.28, loc="upper right",
        fill_clr="whitesmoke", line_clr="grey", pt_clr="red", pt_size=20,
    )
    # [end-step-7-overview]

    # [step-8-export]
    # Raster formats need a resolution: 300 dpi for print, 600+ for line art.
    out["final_map"] = os.path.join(out_dir, "maps_final.png")
    fig.savefig(out["final_map"], dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    # Vector formats (PDF/SVG) keep lines and text sharp at any size; the
    # raster layers inside are embedded at the dpi given here.
    out["final_map_pdf"] = os.path.join(out_dir, "maps_final.pdf")
    fig.savefig(out["final_map_pdf"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # A zoomed-in panel: the same functions with a sub-area bbox.
    fig_zoom, ax_zoom = plt.subplots(
        figsize=(fig_width / 2, map_layout.calc_y_fig_size(zoom_bbox, fig_width / 2))
    )
    map_raster.create_raster_img_map(
        ax_zoom, s2_img, img_bands=list(rgb_bands), bbox=zoom_bbox,
        img_stch=rsgispy.IMG_STRETCH_CUMULATIVE, scale_bar_loc="lower right",
        scale_bar_style=map_elements.SCALEBAR_STYLE_TICKS, show_map_axis=False,
    )
    out["zoom_map"] = os.path.join(out_dir, "maps_zoom.png")
    fig_zoom.savefig(out["zoom_map"], dpi=dpi, bbox_inches="tight")
    plt.close(fig_zoom)
    # [end-step-8-export]

    # [step-9-basemap]
    import rsgispy.mapping.basemaps as map_basemaps

    if basemap:  # needs the internet and the optional contextily package
        import contextily

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        map_vector.create_vec_lyr_map(
            ax, regions, bbox, vec_fill_clrs="none", vec_line_clrs="red",
            vec_line_widths=1.5, plot_zorders=2, scale_bar_loc="lower right",
        )
        map_basemaps.add_contextily_basemap(
            ax, epsg_crs=img_epsg,
            cx_src=contextily.providers.OpenStreetMap.Mapnik,
            cx_zoom_lvl="auto",  # or a fixed web map zoom level, e.g. 13
        )
        out["basemap_map"] = os.path.join(out_dir, "maps_basemap.png")
        fig.savefig(out["basemap_map"], dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if wmts_url is not None:  # an image from a WMTS service (e.g. a mosaic)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        map_basemaps.create_wmts_img_map(
            ax, wmts_url, wmts_lyr, bbox, bbox_epsg=img_epsg,
            title_str="WMTS imagery", scale_bar_loc="lower right",
            tmp_dir=out_dir,
        )
        out["wmts_map"] = os.path.join(out_dir, "maps_wmts.png")
        fig.savefig(out["wmts_map"], dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    # [end-step-9-basemap]
    return out


def main() -> None:
    """Command line interface for the worked example."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")  # write files only; no window needed
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--s2", required=True, help="Surface reflectance stack")
    parser.add_argument("--cls", required=True,
                        help="Land cover raster with a colour table")
    parser.add_argument("--score", required=True, help="Continuous raster")
    parser.add_argument("--regions", required=True, help="Reporting polygons")
    parser.add_argument("--rivers", required=True, help="Line layer")
    parser.add_argument("--overview", required=True,
                        help="Overview layer (e.g. country outlines)")
    parser.add_argument("--classes", default=None,
                        help="Classification config JSON with the class names")
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=[3, 2, 1],
                        help="Red, green and blue band numbers")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution")
    parser.add_argument("--basemap", action="store_true",
                        help="Also draw an OpenStreetMap basemap (internet)")
    parser.add_argument("--wmts-url", default=None, help="WMTS service URL")
    parser.add_argument("--wmts-lyr", default=None, help="WMTS layer name")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()
    run_workflow(args.s2, args.cls, args.score, args.regions, args.rivers,
                 args.overview, args.out_dir, args.classes,
                 rgb_bands=args.rgb_bands, dpi=args.dpi, basemap=args.basemap,
                 wmts_url=args.wmts_url, wmts_lyr=args.wmts_lyr)


if __name__ == "__main__":
    main()
