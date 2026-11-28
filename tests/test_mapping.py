"""
Tests for rsgispy.mapping submodules and rsgispy.tools.plotting using rasterio and geopandas.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import builtins
import shutil

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyproj
import pytest  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

import rsgispy  # noqa: E402
import rsgispy.mapping  # noqa: E402
import rsgispy.mapping.basemaps as map_basemaps  # noqa: E402
import rsgispy.mapping.elements as map_elements  # noqa: E402
import rsgispy.mapping.layout as map_layout  # noqa: E402
import rsgispy.mapping.legend as map_legend  # noqa: E402
import rsgispy.mapping.raster as map_raster  # noqa: E402
import rsgispy.mapping.vector as map_vector  # noqa: E402
import rsgispy.tools.plotting as plotting  # noqa: E402
from rsgispy import RSGISPyException
from rsgispy.mapping import _legend as map_legend_internal


def _write_continuous_raster(
    path,
    data,
    x_min=500000.0,
    y_max=5200000.0,
    res=10.0,
    crs="EPSG:32630",
    nodata=0.0,
):
    """Helper to write a 1-band or multi-band float32 GeoTIFF."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if data.ndim == 2:
        data_3d = data[np.newaxis, ...]
    else:
        data_3d = data
    count, height, width = data_3d.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=str(data_3d.dtype),
        crs=crs,
        transform=from_origin(x_min, y_max, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data_3d)


def _write_thematic_raster(
    path,
    data,
    clr_tab=None,
    x_min=500000.0,
    y_max=5200000.0,
    res=10.0,
    crs="EPSG:32630",
):
    """Helper to write a 1-band uint8 GeoTIFF with optional colour table."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(x_min, y_max, res, res),
        nodata=0,
    ) as dst:
        dst.write(data[np.newaxis, ...].astype(np.uint8))
        if clr_tab is not None:
            dst.write_colormap(1, clr_tab)


def test_calc_y_fig_size_and_extent_and_ticks():
    """Test calc_y_fig_size, define_axis_extent, and tick spacing."""
    bbox = (500000.0, 500200.0, 5199900.0, 5200000.0)
    assert map_layout.calc_y_fig_size(bbox, 10.0) == pytest.approx(5.0)

    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.calc_y_fig_size((10.0, 10.0, 0.0, 5.0), 10.0)

    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, bbox)
    assert ax.get_xlim() == pytest.approx((500000.0, 500200.0))
    assert ax.get_ylim() == pytest.approx((5199900.0, 5200000.0))

    map_layout.define_map_tick_spacing(ax, 50.0)
    xticks = ax.xaxis.get_majorticklocs()
    assert 500050.0 in xticks
    assert 500150.0 in xticks
    plt.close(fig)


def test_get_overview_info_and_add_overview_maps():
    """Test get_overview_info (including reprojection) and add_overview_maps."""
    bbox_wgs84 = (-4.2, -4.0, 52.3, 52.5)
    over_bbox, pt = map_layout.get_overview_info(
        bbox_wgs84, roi_epsg=4326, overview_buf=10.0, out_epsg=4326
    )
    assert pt == pytest.approx((-4.1, 52.4))
    assert over_bbox == pytest.approx((-14.1, 5.9, 42.4, 62.4))

    # Reproject UTM zone 30N bbox to WGS84
    utm_bbox = (420000.0, 430000.0, 5800000.0, 5810000.0)
    over_utm, pt_utm = map_layout.get_overview_info(
        utm_bbox, roi_epsg=32630, overview_buf=5.0, out_epsg=4326
    )
    assert -5.0 < pt_utm[0] < -3.0
    assert 52.0 < pt_utm[1] < 53.0
    assert over_utm[1] - over_utm[0] == pytest.approx(10.0)

    gdf = gpd.GeoDataFrame(
        {"name": ["area"]},
        geometry=[Polygon([(-20, 40), (10, 40), (10, 65), (-20, 65)])],
        crs="EPSG:4326",
    )
    fig, ax = plt.subplots()

    def _rel_bounds(inset_ax):
        return tuple(inset_ax.get_axes_locator()._bounds)  # pylint: disable=protected-access

    ax_over = map_layout.add_overview_maps(ax, gdf, over_bbox, overview_pt=pt)
    assert ax_over in ax.child_axes
    assert ax_over.get_xlim() == pytest.approx((over_bbox[0], over_bbox[1]))
    assert ax_over.get_anchor() == "NW"
    assert _rel_bounds(ax_over) == pytest.approx((0.0, 0.9, 0.1, 0.1))

    expected_positions = {
        "upper right": (0.8, 0.8),
        "top right": (0.8, 0.8),
        "lower left": (0.0, 0.0),
        "bottom left": (0.0, 0.0),
        "lower right": (0.8, 0.0),
        "upper center": (0.4, 0.8),
        "upper centre": (0.4, 0.8),
        "lower center": (0.4, 0.0),
        "center left": (0.0, 0.4),
        "left": (0.0, 0.4),
        "center right": (0.8, 0.4),
        "right": (0.8, 0.4),
        "center": (0.4, 0.4),
        "centre": (0.4, 0.4),
        1: (0.8, 0.8),
        4: (0.8, 0.0),
    }
    for loc_opt, (exp_x, exp_y) in expected_positions.items():
        ins = map_layout.add_overview_maps(
            ax, gdf, over_bbox, over_size_x=0.2, over_size_y=0.2, loc=loc_opt
        )
        rx, ry, rw, rh = _rel_bounds(ins)
        assert (rx, ry, rw, rh) == pytest.approx((exp_x, exp_y, 0.2, 0.2))

    ins_off = map_layout.add_overview_maps(
        ax,
        gdf,
        over_bbox,
        over_size_x=0.2,
        over_size_y=0.15,
        over_x_off=-0.02,
        over_y_off=0.03,
        loc="lower right",
    )
    assert _rel_bounds(ins_off) == pytest.approx((0.78, 0.03, 0.2, 0.15))

    ins_anchor_2 = map_layout.add_overview_maps(
        ax,
        gdf,
        over_bbox,
        over_size_x=0.2,
        over_size_y=0.2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    assert _rel_bounds(ins_anchor_2) == pytest.approx((1.02, 0.8, 0.2, 0.2))

    ins_anchor_4 = map_layout.add_overview_maps(
        ax,
        gdf,
        over_bbox,
        over_size_x=0.2,
        over_size_y=0.2,
        loc="lower right",
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
    )
    assert _rel_bounds(ins_anchor_4) == pytest.approx((0.75, 0.05, 0.2, 0.2))

    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.add_overview_maps(ax, gdf, over_bbox, loc="nowhere")
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.add_overview_maps(ax, gdf, over_bbox, loc=99)
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.add_overview_maps(ax, gdf, over_bbox, loc=True)
    with pytest.raises(rsgispy.RSGISPyException):
        map_layout.add_overview_maps(ax, gdf, over_bbox, bbox_to_anchor=(0.5,))

    plt.close(fig)


def test_draw_bboxes_and_add_axis_label():
    """Test draw_bboxes_to_axis placements/errors and add_axis_label."""
    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([0.0, 100.0])

    bboxes = [
        (10.0, 20.0, 10.0, 20.0),
        (30.0, 40.0, 30.0, 40.0),
        (50.0, 60.0, 50.0, 60.0),
        (70.0, 80.0, 70.0, 80.0),
        (20.0, 30.0, 70.0, 80.0),
    ]
    for pos in ("centre", "above", "below", "left", "right"):
        map_elements.draw_bboxes_to_axis(
            ax,
            [bboxes[0]],
            bbox_labels=[pos],
            lbl_pos=pos,
            lbl_fill_clr="white",
        )
    assert len(ax.patches) == 5

    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.draw_bboxes_to_axis(ax, bboxes[:2], bbox_labels=["only_one"])

    map_elements.add_axis_label(ax, "(a)", lbl_font_weight="bold")
    assert any(t.get_text() == "(a)" for t in ax.texts)
    plt.close(fig)


def test_create_vec_lyr_map_and_choropleth():
    """Test create_vec_lyr_map and create_choropleth_vec_lyr_map."""
    poly1 = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    poly2 = Polygon([(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0)])
    gdf1 = gpd.GeoDataFrame({"val": [1.0, 2.0]}, geometry=[poly1, poly2], crs="EPSG:32630")
    gdf2 = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(2.5, 2.5)], crs="EPSG:32630")
    bbox = (0.0, 10.0, 0.0, 5.0)

    fig, ax = plt.subplots()
    map_vector.create_vec_lyr_map(
        ax,
        [gdf1, gdf2],
        bbox=bbox,
        title_str="Vectors",
        vec_fill_clrs=["lightgrey", "red"],
        vec_line_clrs=["black", "black"],
        vec_line_widths=[0.5, 1.0],
        vec_fill_alphas=[0.8, 1.0],
        plot_zorders=[1, 2],
        sub_in_vec=True,
        use_grid=True,
    )
    assert ax.get_title() == "Vectors"
    assert any(isinstance(a, map_elements.MapScaleBar) for a in ax.artists)
    assert not any(isinstance(a, map_elements.MapNorthArrow) for a in ax.artists)
    plt.close(fig)

    # Error on mismatched layer list length
    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        map_vector.create_vec_lyr_map(ax, [gdf1, gdf2], bbox=bbox, vec_fill_clrs=["red"])
    plt.close(fig)

    # Choropleth in EPSG:4326 (tests great-circle scalebar calculation)
    gdf_wgs = gpd.GeoDataFrame(
        {"val": [10.0, 20.0]},
        geometry=[
            Polygon([(-4.2, 52.3), (-4.1, 52.3), (-4.1, 52.4), (-4.2, 52.4)]),
            Polygon([(-4.1, 52.3), (-4.0, 52.3), (-4.0, 52.4), (-4.1, 52.4)]),
        ],
        crs="EPSG:4326",
    )
    fig, ax = plt.subplots()
    map_vector.create_choropleth_vec_lyr_map(
        ax,
        gdf_wgs,
        vec_col="val",
        bbox=(-4.2, -4.0, 52.3, 52.4),
        title_str="Choropleth",
        show_map_axis=False,
    )
    assert ax.get_title() == "Choropleth"
    assert not ax.axison
    plt.close(fig)


def _rgba(clr):
    return np.asarray(matplotlib.colors.to_rgba(clr))


def test_create_vec_lyr_map_line_colours():
    """Lines are drawn in the line colour; polygons and points keep fill + edge."""
    lines = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[LineString([(0, 0), (5, 5)]), LineString([(5, 0), (10, 5)])],
        crs="EPSG:32630",
    )
    bbox = (0.0, 10.0, 0.0, 5.0)

    fig, ax = plt.subplots()
    map_vector.create_vec_lyr_map(
        ax, lines, bbox, vec_fill_clrs="none", vec_line_clrs="red",
        vec_line_widths=2.0, show_scale_bar=False,
    )
    line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(line_colls) == 1
    edge = line_colls[0].get_edgecolor()
    assert len(edge) > 0
    np.testing.assert_allclose(edge[0], _rgba("red"))
    assert line_colls[0].get_linewidth()[0] == pytest.approx(2.0)
    plt.close(fig)

    # A mixed layer is split by geometry type.
    mixed = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            LineString([(5, 0), (10, 5)]),
            Point(2.0, 2.0),
        ],
        crs="EPSG:32630",
    )
    fig, ax = plt.subplots()
    map_vector.create_vec_lyr_map(
        ax, mixed, bbox, vec_fill_clrs="yellow", vec_line_clrs="blue",
        vec_markersize=20, show_scale_bar=False,
    )
    lines_c = [c for c in ax.collections if isinstance(c, LineCollection)]
    others = [c for c in ax.collections if not isinstance(c, LineCollection)]
    assert len(lines_c) == 1 and len(others) == 2
    np.testing.assert_allclose(lines_c[0].get_edgecolor()[0], _rgba("blue"))
    for coll in others:  # polygon patches and point markers
        np.testing.assert_allclose(coll.get_facecolor()[0], _rgba("yellow"))
        np.testing.assert_allclose(coll.get_edgecolor()[0], _rgba("blue"))
    plt.close(fig)


def test_create_choropleth_vec_lyr_map_line_colours():
    """Choropleth lines are coloured by value, not drawn in the edge colour."""
    lines = gpd.GeoDataFrame(
        {"val": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (5, 5)]), LineString([(5, 0), (10, 5)])],
        crs="EPSG:32630",
    )
    fig, ax = plt.subplots()
    map_vector.create_choropleth_vec_lyr_map(
        ax, lines, "val", (0.0, 10.0, 0.0, 5.0), vec_fill_cmap="viridis",
        vec_line_clr="black", show_scale_bar=False,
    )
    fig.canvas.draw()
    (coll,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    cmap = matplotlib.colormaps["viridis"]
    np.testing.assert_allclose(coll.get_edgecolor(), [cmap(0.0), cmap(1.0)])
    plt.close(fig)

    polys = gpd.GeoDataFrame(
        {"val": [1.0, 2.0]},
        geometry=[
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
        ],
        crs="EPSG:32630",
    )
    fig, ax = plt.subplots()
    map_vector.create_choropleth_vec_lyr_map(
        ax, polys, "val", (0.0, 10.0, 0.0, 5.0), vec_fill_cmap="viridis",
        vec_line_clr="red", show_scale_bar=False,
    )
    fig.canvas.draw()
    (coll,) = ax.collections
    np.testing.assert_allclose(coll.get_edgecolor()[0], _rgba("red"))
    np.testing.assert_allclose(coll.get_facecolor(), [cmap(0.0), cmap(1.0)])
    plt.close(fig)


def test_add_overview_maps_lines_and_no_axis_labels():
    """The inset has no CRS axis labels and draws line layers in line_clr."""
    coast = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(-10, 50), (0, 55), (5, 58)])],
        crs="EPSG:4326",
    )
    fig, ax = plt.subplots()
    ax_over = map_layout.add_overview_maps(
        ax, coast, (-12.0, 8.0, 45.0, 60.0), overview_pt=(-4.0, 52.0),
        fill_clr="white", line_clr="green",
    )
    assert ax_over.get_xlabel() == ""
    assert ax_over.get_ylabel() == ""
    (coll,) = [c for c in ax_over.collections if isinstance(c, LineCollection)]
    np.testing.assert_allclose(coll.get_edgecolor()[0], _rgba("green"))
    plt.close(fig)


def test_plotting_imshow_and_stretches(tmp_path):
    """Test get_raster_mpl_imshow, stretch helpers, and create_raster_img_map."""
    arr = np.arange(1, 3 * 10 * 10 + 1, dtype=np.float32).reshape(3, 10, 10)
    arr[:, 0, 0] = 0.0  # nodata
    rgb_tif = tmp_path / "rgb.tif"
    _write_continuous_raster(str(rgb_tif), arr, nodata=0.0)

    # Full image and subset
    full_data, full_ext = plotting.get_raster_mpl_imshow(str(rgb_tif), bands=[1, 2, 3])
    assert full_data.shape == (10, 10, 3)
    assert full_ext == pytest.approx((500000.0, 500100.0, 5199900.0, 5200000.0))

    sub_data, sub_ext = plotting.get_raster_mpl_imshow(
        str(rgb_tif), bands=[1], bbox=(500020.0, 500060.0, 5199920.0, 5199980.0)
    )
    assert sub_data.shape == (6, 4)
    assert sub_ext == pytest.approx((500020.0, 500060.0, 5199920.0, 5199980.0))

    # Non-intersecting bbox and bad band errors
    with pytest.raises(rsgispy.RSGISPyException):
        plotting.get_raster_mpl_imshow(str(rgb_tif), bands=[1], bbox=(0.0, 10.0, 0.0, 10.0))
    with pytest.raises(rsgispy.RSGISPyException):
        plotting.get_raster_mpl_imshow(str(rgb_tif), bands=[9])
    with pytest.raises(rsgispy.RSGISPyException):
        plotting.get_raster_mpl_imshow(str(rgb_tif), bands=[1, 2])

    # Test stretches on 3D and 2D arrays
    for stretch_fn, extra in [
        (plotting.linear_stretch_np_arr, {}),
        (plotting.cumulative_stretch_np_arr, {"lower": 5, "upper": 95}),
        (plotting.stdev_stretch_np_arr, {"n_stdevs": 1.5}),
        (
            plotting.manual_stretch_np_arr,
            {
                "min_max_vals": [
                    {"min": 1, "max": 100},
                    {"min": 101, "max": 200},
                    {"min": 201, "max": 300},
                ]
            },
        ),
    ]:
        out3 = stretch_fn(full_data, no_data_val=0.0, no_data_clr=(0.0, 0.0, 0.0), **extra)
        assert out3.shape == full_data.shape
        assert np.all(out3 >= 0.0) and np.all(out3 <= 1.0)

    out2 = plotting.linear_stretch_np_arr(
        sub_data, no_data_val=0.0, out_int_type=True, out_gain=255, max_out_val=255
    )
    assert out2.dtype.kind in ("i", "u")

    # Constant band edge case (no division by zero)
    const_arr = np.full((4, 4), 5.0)
    const_out = plotting.linear_stretch_np_arr(const_arr)
    assert np.all(const_out == 0.0)

    # create_raster_img_map for all stretch modes
    for stretch_mode, extra_kw in [
        (rsgispy.IMG_STRETCH_LINEAR, {}),
        (rsgispy.IMG_STRETCH_STDEV, {"stch_n_stdevs": 2.0}),
        (rsgispy.IMG_STRETCH_CUMULATIVE, {}),
        (
            rsgispy.IMG_STRETCH_USER,
            {
                "stch_min_max_vals": [
                    {"min": 1, "max": 100},
                    {"min": 101, "max": 200},
                    {"min": 201, "max": 300},
                ]
            },
        ),
        (rsgispy.IMG_STRETCH_NONE, {}),
    ]:
        fig, ax = plt.subplots()
        map_raster.create_raster_img_map(
            ax,
            str(rgb_tif),
            img_bands=[1, 2, 3],
            img_stch=stretch_mode,
            title_str="Raster",
            **extra_kw,
        )
        assert ax.get_title() == "Raster"
        plt.close(fig)

    # Single-band USER stretch without min_max raises
    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        map_raster.create_raster_img_map(
            ax, str(rgb_tif), img_bands=[1], img_stch=rsgispy.IMG_STRETCH_USER
        )
    plt.close(fig)


def test_create_raster_cmap_img_map(tmp_path):
    """Test create_raster_cmap_img_map with linear and log normalisation."""
    arr = np.linspace(1.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    tif = tmp_path / "cmap.tif"
    _write_continuous_raster(str(tif), arr, nodata=None)

    fig, ax = plt.subplots()
    img = map_raster.create_raster_cmap_img_map(
        ax,
        str(tif),
        img_band=1,
        norm_img_vals=True,
        use_log_norm=True,
        vals_under_white=True,
        title_str="CMap",
        show_colour_bar=True,
        cbar_label="Value",
    )
    assert isinstance(img, matplotlib.image.AxesImage)
    assert isinstance(img.cmap, matplotlib.colors.Colormap)
    assert isinstance(img.norm, matplotlib.colors.LogNorm)
    assert img.colorbar is not None
    assert img.colorbar.ax.get_ylabel() == "Value"
    assert ax.get_title() == "CMap"
    plt.close(fig)


def test_thematic_raster_map(tmp_path):
    """Test get_thematic_raster_mpl_imshow and create_thematic_raster_map."""
    cls_arr = np.array(
        [
            [0, 1, 1, 2],
            [0, 1, 2, 2],
            [3, 3, 2, 0],
            [3, 0, 1, 2],
        ],
        dtype=np.uint8,
    )
    ctab = {
        0: (0, 0, 0, 0),
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (0, 0, 255, 255),
    }
    thmt_tif = tmp_path / "thmt.tif"
    _write_thematic_raster(str(thmt_tif), cls_arr, clr_tab=ctab)

    img_rgba, ext, patches = plotting.get_thematic_raster_mpl_imshow(
        str(thmt_tif),
        band=1,
        out_patches=True,
        cls_names_lut={1: "Forest", 2: "Grass"},
        alpha_lyr=True,
    )
    assert img_rgba.shape == (4, 4, 4)
    assert img_rgba[0, 0, 3] == 0  # class 0 transparent
    assert tuple(img_rgba[0, 1]) == (255, 0, 0, 255)
    assert ext == pytest.approx((500000.0, 500040.0, 5199960.0, 5200000.0))
    assert patches is not None
    assert patches[0].get_label() == "Forest"
    assert patches[1].get_label() == "Grass"

    fig, ax = plt.subplots()
    map_raster.create_thematic_raster_map(ax, str(thmt_tif), title_str="Thematic")
    assert ax.get_title() == "Thematic"
    plt.close(fig)

    # Missing colour table raises RSGISPyException
    no_ct_tif = tmp_path / "no_ct.tif"
    _write_thematic_raster(str(no_ct_tif), cls_arr, clr_tab=None)
    with pytest.raises(rsgispy.RSGISPyException):
        plotting.get_thematic_raster_mpl_imshow(str(no_ct_tif))


def test_optional_contextily_density_and_wmts(tmp_path):
    """Test add_contextily_basemap, create_vec_pt_density_map, and create_wmts_img_map."""
    # Mock contextily
    mock_cx = types.ModuleType("contextily")
    mock_cx.add_basemap = MagicMock()
    fig, ax = plt.subplots()
    with patch.dict(sys.modules, {"contextily": mock_cx}):
        map_basemaps.add_contextily_basemap(
            ax, epsg_crs=3857, cx_src="OpenTopoMap", cx_zoom_lvl=None
        )
    mock_cx.add_basemap.assert_called_once()
    plt.close(fig)

    # Mock mpl_scatter_density and axis.scatter_density
    mock_msd = types.ModuleType("mpl_scatter_density")
    gdf_pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(500010.0, 5199910.0), Point(500050.0, 5199950.0)],
        crs="EPSG:32630",
    )
    fig, ax = plt.subplots()
    ax.scatter_density = MagicMock(return_value="density_artist")
    with patch.dict(sys.modules, {"mpl_scatter_density": mock_msd}):
        density_artist = map_vector.create_vec_pt_density_map(
            ax,
            gdf_pts,
            bbox=(500000.0, 500100.0, 5199900.0, 5200000.0),
            use_log_norm=True,
            sub_in_vec=True,
            title_str="Density",
        )
    assert density_artist == "density_artist"
    ax.scatter_density.assert_called_once()
    sd_kwargs = ax.scatter_density.call_args.kwargs
    assert isinstance(sd_kwargs["cmap"], matplotlib.colors.Colormap)
    assert isinstance(sd_kwargs["norm"], matplotlib.colors.LogNorm)
    plt.close(fig)

    # Mock WMTS XML + GTiff creation via rasterio.shutil.copy
    sample_rgb = np.full((3, 8, 8), 128, dtype=np.uint8)
    sample_tif = tmp_path / "wmts_sample.tif"
    _write_continuous_raster(str(sample_tif), sample_rgb, crs="EPSG:32630", nodata=None)

    wmts_xml_template = (
        "<GDAL_WMTS>"
        "<DataWindow>"
        "<UpperLeftX>0</UpperLeftX>"
        "<UpperLeftY>0</UpperLeftY>"
        "<LowerRightX>0</LowerRightX>"
        "<LowerRightY>0</LowerRightY>"
        "</DataWindow>"
        "</GDAL_WMTS>"
    )

    def _fake_rio_copy(src_path, dst_path, driver=None, **_kwargs):
        if driver == "WMTS":
            with open(dst_path, "w", encoding="utf-8") as fobj:
                fobj.write(wmts_xml_template)
        else:
            with open(sample_tif, "rb") as fin, open(dst_path, "wb") as fout:
                fout.write(fin.read())

    fig, ax = plt.subplots()
    with patch("rasterio.shutil.copy", side_effect=_fake_rio_copy):
        map_basemaps.create_wmts_img_map(
            ax,
            wmts_url="https://example.com/wmts",
            wmts_lyr="visual",
            bbox=(500000.0, 500080.0, 5199920.0, 5200000.0),
            bbox_epsg=32630,
            wmts_epsg=32630,
            wmts_zoom_level=10,
            title_str="WMTS",
        )
    assert ax.get_title() == "WMTS"
    assert not any(os.path.exists(f"tmp_{x}") for x in ())
    plt.close(fig)


def test_scale_bar_styles_units_and_rendering(tmp_path):
    """Test add_scale_bar styles, m/km unit enforcement, EPSG:4326, and rendering."""
    # 1) Small UTM extent -> auto selects metres ('m')
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (500000.0, 500400.0, 5200000.0, 5200400.0))
    sb_small = map_elements.add_scale_bar(ax, epsg_code=32630)
    assert isinstance(sb_small, map_elements.MapScaleBar)
    assert sb_small.units == map_elements.SCALEBAR_UNIT_METRES
    assert sb_small.style == map_elements.SCALEBAR_STYLE_SOLID
    assert sb_small.label_text.endswith(" m")
    out_small = tmp_path / "sb_solid.png"
    fig.savefig(out_small, dpi=72, bbox_inches="tight")
    assert out_small.exists() and out_small.stat().st_size > 0
    plt.close(fig)

    # 2) Large UTM extent -> auto selects kilometres ('km')
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (500000.0, 520000.0, 5200000.0, 5220000.0))
    sb_large = map_elements.add_scale_bar(
        ax,
        epsg_code=32630,
        style=map_elements.SCALEBAR_STYLE_ALTERNATING,
        loc="bottom left",
    )
    assert sb_large.units == map_elements.SCALEBAR_UNIT_KILOMETRES
    assert sb_large.style == map_elements.SCALEBAR_STYLE_ALTERNATING
    assert sb_large.loc_str == "lower left"
    assert sb_large.label_text.endswith(" km")
    out_alt = tmp_path / "sb_alt.png"
    fig.savefig(out_alt, dpi=72, bbox_inches="tight")
    assert out_alt.exists() and out_alt.stat().st_size > 0
    plt.close(fig)

    # 3) Ticks style and explicit 'm' unit override on large extent
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (500000.0, 504000.0, 5200000.0, 5204000.0))
    sb_ticks = map_elements.add_scale_bar(
        ax,
        epsg_code=32630,
        style=map_elements.SCALEBAR_STYLE_TICKS,
        units=map_elements.SCALEBAR_UNIT_METRES,
        loc="lower right",
    )
    assert sb_ticks.units == "m"
    assert sb_ticks.style == map_elements.SCALEBAR_STYLE_TICKS
    assert sb_ticks.bar_length_metres == pytest.approx(sb_ticks.bar_length_value)
    out_ticks = tmp_path / "sb_ticks.png"
    fig.savefig(out_ticks, dpi=72, bbox_inches="tight")
    assert out_ticks.exists() and out_ticks.stat().st_size > 0
    plt.close(fig)

    # 4) EPSG:4326 geographic extent (~20 km wide at 52N) -> kilometres, positive bar width
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (-4.3, -4.0, 52.3, 52.5))
    sb_wgs = map_elements.add_scale_bar(
        ax, epsg_code=4326, units=map_elements.SCALEBAR_UNIT_KILOMETRES
    )
    assert sb_wgs.units == "km"
    assert 1.0 <= sb_wgs.bar_length_value <= 10.0
    fig.canvas.draw()
    plt.close(fig)

    # 5) Sub-metre map still uses 'm' (never cm/mm)
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (0.0, 1.0, 0.0, 1.0))
    sb_tiny = map_elements.add_scale_bar(ax, epsg_code=32630)
    assert sb_tiny.units == "m"
    plt.close(fig)

    # 6) Invalid units, styles, and locations raise RSGISPyException
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (0.0, 100.0, 0.0, 100.0))
    for bad_unit in ("cm", "mm", "ft", "mi", "deg"):
        with pytest.raises(rsgispy.RSGISPyException):
            map_elements.add_scale_bar(ax, units=bad_unit)
    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.add_scale_bar(ax, style="dashed")
    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.add_scale_bar(ax, loc="nowhere")
    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.add_scale_bar(ax, length_fraction=1.5)
    plt.close(fig)


def test_north_arrow_styles_and_map_integration(tmp_path):
    """Test add_north_arrow styles, rotation, errors, and create_*_map integration."""
    styles = (
        map_elements.NORTH_ARROW_STYLE_TRIANGLE,
        map_elements.NORTH_ARROW_STYLE_ARROW,
        map_elements.NORTH_ARROW_STYLE_COMPASS,
        map_elements.NORTH_ARROW_STYLE_SIMPLE,
    )
    for style in styles:
        fig, ax = plt.subplots()
        map_layout.define_axis_extent(ax, (500000.0, 510000.0, 5200000.0, 5210000.0))
        na = map_elements.add_north_arrow(
            ax,
            style=style,
            loc="upper left",
            rotation=15.0 if style == map_elements.NORTH_ARROW_STYLE_ARROW else 0.0,
        )
        assert isinstance(na, map_elements.MapNorthArrow)
        assert na.style == style
        out_png = tmp_path / f"na_{style}.png"
        fig.savefig(out_png, dpi=72, bbox_inches="tight")
        assert out_png.exists() and out_png.stat().st_size > 0
        plt.close(fig)

    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.add_north_arrow(ax, style="unknown_style")
    with pytest.raises(rsgispy.RSGISPyException):
        map_elements.add_north_arrow(ax, size=0.0)
    plt.close(fig)

    # Verify integration on create_raster_img_map & create_vec_lyr_map
    arr = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
    tif = tmp_path / "map_int.tif"
    _write_continuous_raster(str(tif), arr, res=100.0, nodata=0.0)

    # Default: scale bar enabled, north arrow NOT added
    fig, ax = plt.subplots()
    map_raster.create_raster_img_map(
        ax,
        str(tif),
        img_bands=[1],
        img_stch=rsgispy.IMG_STRETCH_LINEAR,
    )
    assert any(isinstance(a, map_elements.MapScaleBar) for a in ax.artists)
    assert not any(isinstance(a, map_elements.MapNorthArrow) for a in ax.artists)
    plt.close(fig)

    # Enabled with custom styles and units
    fig, ax = plt.subplots()
    map_raster.create_raster_img_map(
        ax,
        str(tif),
        img_bands=[1],
        img_stch=rsgispy.IMG_STRETCH_LINEAR,
        scale_bar_style=map_elements.SCALEBAR_STYLE_TICKS,
        scale_bar_units=map_elements.SCALEBAR_UNIT_KILOMETRES,
        show_north_arrow=True,
        north_arrow_style=map_elements.NORTH_ARROW_STYLE_COMPASS,
        north_arrow_loc="lower left",
    )
    sbs = [a for a in ax.artists if isinstance(a, map_elements.MapScaleBar)]
    nas = [a for a in ax.artists if isinstance(a, map_elements.MapNorthArrow)]
    assert len(sbs) == 1
    assert sbs[0].style == map_elements.SCALEBAR_STYLE_TICKS
    assert sbs[0].units == map_elements.SCALEBAR_UNIT_KILOMETRES
    assert len(nas) == 1
    assert nas[0].style == map_elements.NORTH_ARROW_STYLE_COMPASS
    out_full = tmp_path / "full_map.png"
    fig.savefig(out_full, dpi=72, bbox_inches="tight")
    assert out_full.exists() and out_full.stat().st_size > 0
    plt.close(fig)


def test_categorical_map_legends(tmp_path):
    """Test categorical/classification legends: inputs, layouts, and placements."""
    from rsgispy.classification.classdata import (
        ClassificationConfig,
        ClassInfoObj,
        ImageBandInfo,
    )

    thm_img = tmp_path / "thematic_lgd.tif"
    data = np.array(
        [
            [0, 1, 1, 2],
            [1, 1, 2, 2],
            [2, 3, 3, 3],
            [3, 3, 1, 0],
        ],
        dtype=np.uint8,
    )
    clr_tab = {
        0: (0, 0, 0, 0),
        1: (34, 139, 34, 255),
        2: (218, 165, 32, 255),
        3: (30, 144, 255, 255),
        4: (200, 50, 50, 255),  # Absent from image pixel array
    }
    _write_thematic_raster(str(thm_img), data, clr_tab=clr_tab)

    # 1. Auto layout adapts column count to class label lengths
    r_short, c_short = map_legend.calc_categorical_legend_layout(
        labels=["Water", "Forest", "Grass", "Urban", "Sand", "Rock"],
        avail_width_in=6.5,
    )
    r_long, c_long = map_legend.calc_categorical_legend_layout(
        labels=[
            "Temperate Broadleaf and Mixed Evergreen Forest",
            "Intensified Irrigated Arable Cropland and Pasture",
            "High Density Impervious Urban Commercial Settlement",
            "Inland Freshwater Rivers Lakes and Reservoirs",
        ],
        avail_width_in=6.5,
    )
    assert c_short >= 2 and r_short * c_short >= 6
    assert c_long == 1 and r_long == 4

    with pytest.raises(rsgispy.RSGISPyException):
        map_legend.calc_categorical_legend_layout(["A", "B", "C"], ncols=1, nrows=2)

    # 2. {pixel_val: class_name} dict + embedded raster colour table + overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    map_raster.create_thematic_raster_map(
        ax=ax,
        input_img=str(thm_img),
        title_str="Land Cover",
        show_legend=True,
        classes_info={1: "Forest", 2: "Cropland", 3: "Water", 4: "Urban"},
        legend_title="Classes",
        legend_kwargs={"only_present_in_img": True, "show_pxl_vals": True},
    )
    # Companion axis was added (figure now has 2 axes)
    assert len(fig.axes) == 2
    lgd_comp = fig.axes[1].get_legend()
    assert lgd_comp is not None
    assert [t.get_text() for t in lgd_comp.get_texts()] == [
        "1: Forest",
        "2: Cropland",
        "3: Water",
    ]
    out_comp = tmp_path / "map_with_companion_legend.png"
    fig.savefig(out_comp, dpi=100, bbox_inches="tight")
    assert out_comp.exists() and out_comp.stat().st_size > 0
    plt.close(fig)

    # 3. Dedicated legend axes positioned by user in multi-axis figures
    #    (no companion axis created; layout adapts to axis geometry)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    ax_m1 = fig.add_subplot(gs[0, 0])
    ax_leg_side = fig.add_subplot(gs[0, 1])
    ax_leg_bot = fig.add_subplot(gs[1, :])
    map_raster.create_thematic_raster_map(ax=ax_m1, input_img=str(thm_img))
    lgd_side = map_legend.add_categorical_legend_axis(
        ax=ax_leg_side,
        classes_info={
            1: ("Forest", "#228b22"),
            2: ("Crop", "#daa520"),
            3: ("Water", "#1e90ff"),
        },
        loc="upper left",
        title="Legend",
    )
    lgd_bot = map_legend.add_categorical_legend_axis(
        ax=ax_leg_bot,
        classes_info={
            1: ("Forest", "#228b22"),
            2: ("Crop", "#daa520"),
            3: ("Water", "#1e90ff"),
            4: ("Urban", "#d62728"),
        },
        swatch_shape="square",
        title="Bottom Legend",
    )
    assert len(fig.axes) == 3
    assert not ax_leg_side.axison
    assert not ax_leg_bot.axison
    assert lgd_side is ax_leg_side.get_legend()
    assert lgd_bot is ax_leg_bot.get_legend()
    assert lgd_side.get_title().get_text() == "Legend"
    assert lgd_side._ncols == 1  # pylint: disable=protected-access
    assert lgd_bot._ncols >= 2  # pylint: disable=protected-access
    plt.close(fig)

    # create_thematic_raster_map writing legend into explicit legend_ax
    fig, (ax_m2, ax_l2) = plt.subplots(1, 2, figsize=(8, 4))
    map_raster.create_thematic_raster_map(
        ax=ax_m2,
        input_img=str(thm_img),
        show_legend=True,
        classes_info={1: "Forest", 2: "Crop", 3: "Water"},
        legend_title="Classes",
        legend_ax=ax_l2,
    )
    assert len(fig.axes) == 2
    assert not ax_l2.axison
    assert ax_l2.get_legend() is not None
    plt.close(fig)

    # 4. Existing axis overlay=True and dedicated axis overlay=False
    fig, (ax_map, ax_leg) = plt.subplots(1, 2, figsize=(8, 4))
    map_raster.create_thematic_raster_map(ax=ax_map, input_img=str(thm_img))
    lgd_ov = map_legend.add_categorical_legend(
        ax=ax_map,
        classes_info={1: "Forest", 2: "Crop", 3: "Water"},
        input_img=str(thm_img),
        loc="bottom left",
        overlay=True,
    )
    assert lgd_ov in ax_map.get_children()
    lgd_panel = map_legend.add_categorical_legend(
        ax=ax_leg,
        classes_info={1: "Forest", 2: "Crop", 3: "Water"},
        colour_lut={1: "#228b22", 2: "#daa520", 3: "#1e90ff"},
        overlay=False,
        nrows=2,
        sort_by="name",
        swatch_shape="circle",
    )
    assert not ax_leg.axison
    assert [t.get_text() for t in lgd_panel.get_texts()] == ["Crop", "Forest", "Water"]
    plt.close(fig)

    # 5. ClassificationConfig + JSON config path + standalone saved legend image
    cfg = ClassificationConfig(
        img_band_info=[ImageBandInfo(str(thm_img), name="lc", bands=[1])],
        class_info={
            "Forest": ClassInfoObj(out_id=1, name="Forest", colour="#228b22"),
            "Cropland": ClassInfoObj(out_id=2, name="Cropland", colour="#daa520"),
            "Water": ClassInfoObj(out_id=3, name="Water", red=30, green=144, blue=255),
        },
    )
    cfg_json = tmp_path / "cls_cfg.json"
    cfg.save(cfg_json)

    handles_cfg = map_legend.create_categorical_legend_patches(cfg, max_label_width=6)
    assert len(handles_cfg) == 3

    out_lgd_png = tmp_path / "standalone_legend.png"
    r_saved, c_saved = map_legend.save_categorical_legend_img(
        output_img=out_lgd_png,
        classes_info=cfg_json,
        title="Land Cover Classification",
        swatch_edge_colour="#111111",
        dpi=120,
        transparent=True,
    )
    assert out_lgd_png.exists() and out_lgd_png.stat().st_size > 0
    assert r_saved * c_saved >= 3


def test_mapping_british_english_params(tmp_path):
    """Test renamed British English parameters on elements, legend, and raster maps."""
    fig, ax = plt.subplots()
    map_layout.define_axis_extent(ax, (500000.0, 510000.0, 5200000.0, 5210000.0))
    sb = map_elements.add_scale_bar(
        ax,
        epsg_code=32630,
        style=map_elements.SCALEBAR_STYLE_ALTERNATING,
        colour="navy",
        alt_colour="lightgrey",
        box_colour="ivory",
    )
    assert sb.colour == "navy"
    assert sb.alt_colour == "lightgrey"

    na = map_elements.add_north_arrow(
        ax,
        style=map_elements.NORTH_ARROW_STYLE_COMPASS,
        colour="darkgreen",
        alt_colour="beige",
        box_colour="white",
    )
    assert na.colour == "darkgreen"
    assert na.alt_colour == "beige"

    handles = map_legend.create_categorical_legend_patches(
        classes_info={1: "Forest", 2: "Water"},
        colour_lut={1: "#228b22", 2: "#1e90ff"},
        swatch_edge_colour="#000000",
    )
    assert len(handles) == 2
    plt.close(fig)

    arr = np.array([[1.0, 2.0], [-99.0, 4.0]], dtype=np.float32)
    tif = tmp_path / "nd.tif"
    _write_continuous_raster(str(tif), arr, nodata=None)
    fig2, ax2 = plt.subplots()
    map_raster.create_raster_img_map(
        ax2,
        str(tif),
        img_bands=[1],
        img_stch=rsgispy.IMG_STRETCH_LINEAR,
        no_data_val=-99.0,
        no_data_out_clr=0.0,
    )
    assert len(ax2.images) == 1
    plt.close(fig2)


def _write_raster(
    path: str,
    arr: np.ndarray,
    *,
    origin=(500000.0, 5200000.0),
    res=(10.0, 10.0),
    crs: str = "EPSG:32630",
    nodata=None,
    tiled: bool = False,
    blocksize: int | None = None,
) -> str:
    """Write a 2D or 3D NumPy array to a GeoTIFF."""
    if arr.ndim == 2:
        data = arr[np.newaxis, :, :]
    else:
        data = arr
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], res[0], res[1]),
        "nodata": nodata,
    }
    if tiled:
        profile["tiled"] = True
        if blocksize is not None:
            profile["blockxsize"] = blocksize
            profile["blockysize"] = blocksize
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_elements_geodesic_scale_and_axis_label_trans_axes():
    """_compute_metres_per_map_unit uses geodesic distance; add_axis_label uses transAxes."""
    # EPSG:3857 at ~60 deg N: cos(60) = 0.5 -> ~0.5 m per Web Mercator metre
    t = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = t.transform(0.0, 60.0)
    m_3857 = map_elements._compute_metres_per_map_unit(
        3857, (x0 - 1000.0, x0 + 1000.0, y0 - 1000.0, y0 + 1000.0)
    )
    assert m_3857 is not None
    assert 0.49 < m_3857 < 0.51

    # EPSG:4326 at equator: ~111319 m/deg
    m_4326_eq = map_elements._compute_metres_per_map_unit(
        4326, (-0.5, 0.5, -0.5, 0.5)
    )
    assert m_4326_eq is not None
    assert 111000.0 < m_4326_eq < 111600.0

    # EPSG:32630 (UTM): ~1.0 m/unit
    m_utm = map_elements._compute_metres_per_map_unit(
        32630, (500000.0, 501000.0, 5700000.0, 5701000.0)
    )
    assert m_utm is not None
    assert 0.99 < m_utm < 1.01

    # Invalid EPSG falls back cleanly to 1.0
    assert map_elements._compute_metres_per_map_unit(
        99999999, (0, 1, 0, 1)
    ) == pytest.approx(1.0)

    # add_axis_label uses ax.transAxes at (0.0, 1.0)
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(100, 200)
        ax.set_ylim(300, 400)
        map_elements.add_axis_label(ax, "(a)")
        txt = ax.texts[-1]
        assert txt.get_position() == (0.0, 1.0)
        assert txt.get_transform() == ax.transAxes
    finally:
        plt.close(fig)


def test_legend_normalise_rgba_scan_present_and_resolve_loc(tmp_path):
    """scale_255 flag, tiled _scan_present_values, and resolve_mpl_location."""
    assert map_legend_internal._normalise_rgba(
        (1, 0, 0, 1), scale_255=False
    ) == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert map_legend_internal._normalise_rgba(
        (1, 0, 0, 1), scale_255=True
    ) == pytest.approx((1.0 / 255.0, 0.0, 0.0, 1.0 / 255.0))

    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[0:16, 0:16] = 1
    arr[16:32, 16:32] = 2
    tif = _write_raster(
        str(tmp_path / "thm.tif"), arr, tiled=True, blocksize=16
    )
    # Top-left bbox sees only 1
    present_tl = map_legend_internal._scan_present_values(
        tif, band=1, bbox=(500000.0, 500160.0, 5199840.0, 5200000.0)
    )
    assert present_tl == {1}
    # Disjoint bbox returns empty set cleanly
    present_none = map_legend_internal._scan_present_values(
        tif, band=1, bbox=(600000.0, 600160.0, 5199840.0, 5200000.0)
    )
    assert present_none == set()

    # resolve_mpl_location in add_categorical_legend
    fig, ax = plt.subplots()
    try:
        leg = map_legend.add_categorical_legend(
            ax,
            classes_info={1: "Forest", 2: "Water"},
            colour_lut={1: "green", 2: "blue"},
            overlay=True,
            loc="top left",
        )
        assert leg is not None
        with pytest.raises(RSGISPyException):
            map_legend.add_categorical_legend(
                ax,
                classes_info={1: "Forest"},
                colour_lut={1: "green"},
                overlay=True,
                loc="nowhere",
            )
    finally:
        plt.close(fig)


def test_raster_cmap_nodata_mask_and_vals_under_white(tmp_path):
    """create_raster_cmap_img_map masks nodata before auto-norm and honours vals_under_white."""
    arr = np.array(
        [[-9999.0, 10.0], [20.0, 30.0]], dtype=np.float32
    )
    tif = _write_raster(str(tmp_path / "cmap.tif"), arr, nodata=-9999.0)

    fig, ax = plt.subplots()
    try:
        img = map_raster.create_raster_cmap_img_map(
            ax=ax,
            input_img=tif,
            img_band=1,
            norm_img_vals=True,
            show_scale_bar=False,
        )
        cmap, norm = img.cmap, img.norm
        assert norm is not None
        assert norm.vmin == pytest.approx(10.0)
        assert norm.vmax == pytest.approx(30.0)
        assert cmap is not None
    finally:
        plt.close(fig)

    fig2, ax2 = plt.subplots()
    try:
        img2 = map_raster.create_raster_cmap_img_map(
            ax=ax2,
            input_img=tif,
            img_band=1,
            norm_img_vals=False,
            vals_under_white=True,
            show_scale_bar=False,
        )
        cmap2, norm2 = img2.cmap, img2.norm
        # Without norm_img_vals the image uses Matplotlib's linear autoscaling.
        assert type(norm2) is matplotlib.colors.Normalize
        assert tuple(cmap2.get_under()) == pytest.approx((1.0, 1.0, 1.0, 1.0))
    finally:
        plt.close(fig2)


def test_vector_subset_exact_and_optional_dep_guards(monkeypatch, tmp_path):
    """_subset_vec_bbox has no +/-4 buffer; optional deps guarded; WMTS XML & 1-band."""
    pts = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[Point(0.5, 0.5), Point(3.0, 0.5)],
        crs="EPSG:4326",
    )
    sub = map_vector._subset_vec_bbox(pts, (0.0, 1.0, 0.0, 1.0))
    assert list(sub["id"]) == [1]

    # Guard missing mpl_scatter_density and contextily
    real_import = builtins.__import__

    def _block_opt(name, *args, **kwargs):
        if name in ("mpl_scatter_density", "contextily"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_opt)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(RSGISPyException, match="mpl-scatter-density"):
            map_vector.create_vec_pt_density_map(
                ax, pts, bbox=(0.0, 1.0, 0.0, 1.0)
            )
        with pytest.raises(RSGISPyException, match="contextily"):
            map_basemaps.add_contextily_basemap(
                ax, epsg_crs=3857, cx_src="dummy"
            )
    finally:
        plt.close(fig)
    monkeypatch.setattr(builtins, "__import__", real_import)

    # _update_wmts_xml replaces existing <ZoomLevel> instead of duplicating
    in_xml = tmp_path / "wmts_in.xml"
    out_xml = tmp_path / "wmts_out.xml"
    in_xml.write_text(
        "<GDAL_WMTS>"
        "<DataWindow>"
        "<UpperLeftX>0</UpperLeftX><UpperLeftY>10</UpperLeftY>"
        "<LowerRightX>10</LowerRightX><LowerRightY>0</LowerRightY>"
        "</DataWindow>"
        "<ZoomLevel>5</ZoomLevel>"
        "</GDAL_WMTS>",
        encoding="utf-8",
    )
    map_basemaps._update_wmts_xml(
        str(in_xml), str(out_xml), (1.0, 2.0, 3.0, 4.0), wmts_zoom_level=12
    )
    xml_out = out_xml.read_text(encoding="utf-8")
    assert xml_out.count("<ZoomLevel>") == 1
    assert "<ZoomLevel>12</ZoomLevel>" in xml_out

    # create_wmts_img_map handles a 1-band raster cleanly
    tif1 = _write_raster(
        str(tmp_path / "oneband.tif"),
        np.arange(16, dtype=np.uint8).reshape((4, 4)),
        crs="EPSG:3857",
    )

    def _fake_rio_copy(src, dst, **kwargs):
        if kwargs.get("driver") == "WMTS":
            with open(dst, "w", encoding="utf-8") as xml_f:
                xml_f.write(
                    "<GDAL_WMTS>"
                    "<DataWindow>"
                    "<UpperLeftX>0</UpperLeftX><UpperLeftY>10</UpperLeftY>"
                    "<LowerRightX>10</LowerRightX><LowerRightY>0</LowerRightY>"
                    "</DataWindow>"
                    "</GDAL_WMTS>"
                )
        else:
            shutil.copyfile(tif1, dst)

    monkeypatch.setattr(rasterio.shutil, "copy", _fake_rio_copy)
    fig3, ax3 = plt.subplots()
    try:
        map_basemaps.create_wmts_img_map(
            ax=ax3,
            wmts_url="https://example.org/wmts",
            wmts_lyr="lyr",
            bbox=(500000.0, 500040.0, 5199960.0, 5200000.0),
            bbox_epsg=3857,
            wmts_epsg=3857,
            show_scale_bar=False,
        )
        assert len(ax3.images) == 1
        assert ax3.images[0].get_cmap().name == "gray"
    finally:
        plt.close(fig3)
