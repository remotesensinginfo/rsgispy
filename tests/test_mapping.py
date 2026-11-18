"""
Tests for rsgispy.tools.mapping and rsgispy.tools.plotting using rasterio and geopandas.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402

import rsgispy  # noqa: E402
import rsgispy.tools.mapping as mapping  # noqa: E402
import rsgispy.tools.plotting as plotting  # noqa: E402


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
    """Test calc_y_fig_size, define_axis_extent, and define_map_tick_spacing."""
    bbox = (500000.0, 500200.0, 5199900.0, 5200000.0)
    assert mapping.calc_y_fig_size(bbox, 10.0) == pytest.approx(5.0)

    with pytest.raises(rsgispy.RSGISPyException):
        mapping.calc_y_fig_size((10.0, 10.0, 0.0, 5.0), 10.0)

    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, bbox)
    assert ax.get_xlim() == pytest.approx((500000.0, 500200.0))
    assert ax.get_ylim() == pytest.approx((5199900.0, 5200000.0))

    mapping.define_map_tick_spacing(ax, 50.0)
    xticks = ax.xaxis.get_majorticklocs()
    assert 500050.0 in xticks
    assert 500150.0 in xticks
    plt.close(fig)


def test_get_overview_info_and_add_overview_maps():
    """Test get_overview_info (including reprojection) and add_overview_maps."""
    bbox_wgs84 = (-4.2, -4.0, 52.3, 52.5)
    over_bbox, pt = mapping.get_overview_info(
        bbox_wgs84, roi_epsg=4326, overview_buf=10.0, out_epsg=4326
    )
    assert pt == pytest.approx((-4.1, 52.4))
    assert over_bbox == pytest.approx((-14.1, 5.9, 42.4, 62.4))

    # Reproject UTM zone 30N bbox to WGS84
    utm_bbox = (420000.0, 430000.0, 5800000.0, 5810000.0)
    over_utm, pt_utm = mapping.get_overview_info(
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
    ax_over = mapping.add_overview_maps(ax, gdf, over_bbox, overview_pt=pt)
    assert ax_over in ax.child_axes
    assert ax_over.get_xlim() == pytest.approx((over_bbox[0], over_bbox[1]))
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
        mapping.draw_bboxes_to_axis(
            ax,
            [bboxes[0]],
            bbox_labels=[pos],
            lbl_pos=pos,
            lbl_fill_clr="white",
        )
    assert len(ax.patches) == 5

    with pytest.raises(rsgispy.RSGISPyException):
        mapping.draw_bboxes_to_axis(ax, bboxes[:2], bbox_labels=["only_one"])

    mapping.add_axis_label(ax, "(a)", lbl_font_weight="bold")
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
    mapping.create_vec_lyr_map(
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
    assert any(isinstance(a, mapping.MapScaleBar) for a in ax.artists)
    assert not any(isinstance(a, mapping.MapNorthArrow) for a in ax.artists)
    plt.close(fig)

    # Error on mismatched layer list length
    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.create_vec_lyr_map(ax, [gdf1, gdf2], bbox=bbox, vec_fill_clrs=["red"])
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
    mapping.create_choropleth_vec_lyr_map(
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
        mapping.create_raster_img_map(
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
        mapping.create_raster_img_map(
            ax, str(rgb_tif), img_bands=[1], img_stch=rsgispy.IMG_STRETCH_USER
        )
    plt.close(fig)


def test_create_raster_cmap_img_map(tmp_path):
    """Test create_raster_cmap_img_map with linear and log normalisation."""
    arr = np.linspace(1.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    tif = tmp_path / "cmap.tif"
    _write_continuous_raster(str(tif), arr, nodata=None)

    fig, ax = plt.subplots()
    cmap, norm = mapping.create_raster_cmap_img_map(
        ax,
        str(tif),
        img_band=1,
        norm_img_vals=True,
        use_log_norm=True,
        vals_under_white=True,
        title_str="CMap",
    )
    assert isinstance(cmap, matplotlib.colors.Colormap)
    assert isinstance(norm, matplotlib.colors.LogNorm)
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
    mapping.create_thematic_raster_map(ax, str(thmt_tif), title_str="Thematic")
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
        mapping.add_contextily_basemap(ax, epsg_crs=3857, cx_src="OpenTopoMap", cx_zoom_lvl=None)
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
    ax.scatter_density = MagicMock()
    with patch.dict(sys.modules, {"mpl_scatter_density": mock_msd}):
        cmap, norm = mapping.create_vec_pt_density_map(
            ax,
            gdf_pts,
            bbox=(500000.0, 500100.0, 5199900.0, 5200000.0),
            use_log_norm=True,
            sub_in_vec=True,
            title_str="Density",
        )
    assert isinstance(cmap, matplotlib.colors.Colormap)
    assert isinstance(norm, matplotlib.colors.LogNorm)
    ax.scatter_density.assert_called_once()
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
        mapping.create_wmts_img_map(
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
    mapping.define_axis_extent(ax, (500000.0, 500400.0, 5200000.0, 5200400.0))
    sb_small = mapping.add_scale_bar(ax, epsg_code=32630)
    assert isinstance(sb_small, mapping.MapScaleBar)
    assert sb_small.units == mapping.SCALEBAR_UNIT_METRES
    assert sb_small.style == mapping.SCALEBAR_STYLE_SOLID
    assert sb_small.label_text.endswith(" m")
    out_small = tmp_path / "sb_solid.png"
    fig.savefig(out_small, dpi=72, bbox_inches="tight")
    assert out_small.exists() and out_small.stat().st_size > 0
    plt.close(fig)

    # 2) Large UTM extent -> auto selects kilometres ('km')
    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, (500000.0, 520000.0, 5200000.0, 5220000.0))
    sb_large = mapping.add_scale_bar(
        ax,
        epsg_code=32630,
        style=mapping.SCALEBAR_STYLE_ALTERNATING,
        loc="bottom left",
    )
    assert sb_large.units == mapping.SCALEBAR_UNIT_KILOMETRES
    assert sb_large.style == mapping.SCALEBAR_STYLE_ALTERNATING
    assert sb_large.loc_str == "lower left"
    assert sb_large.label_text.endswith(" km")
    out_alt = tmp_path / "sb_alt.png"
    fig.savefig(out_alt, dpi=72, bbox_inches="tight")
    assert out_alt.exists() and out_alt.stat().st_size > 0
    plt.close(fig)

    # 3) Ticks style and explicit 'm' unit override on large extent
    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, (500000.0, 504000.0, 5200000.0, 5204000.0))
    sb_ticks = mapping.add_scale_bar(
        ax,
        epsg_code=32630,
        style=mapping.SCALEBAR_STYLE_TICKS,
        units=mapping.SCALEBAR_UNIT_METRES,
        loc="lower right",
    )
    assert sb_ticks.units == "m"
    assert sb_ticks.style == mapping.SCALEBAR_STYLE_TICKS
    assert sb_ticks.bar_length_metres == pytest.approx(sb_ticks.bar_length_value)
    out_ticks = tmp_path / "sb_ticks.png"
    fig.savefig(out_ticks, dpi=72, bbox_inches="tight")
    assert out_ticks.exists() and out_ticks.stat().st_size > 0
    plt.close(fig)

    # 4) EPSG:4326 geographic extent (~20 km wide at 52N) -> kilometres, positive bar width
    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, (-4.3, -4.0, 52.3, 52.5))
    sb_wgs = mapping.add_scale_bar(ax, epsg_code=4326, units=mapping.SCALEBAR_UNIT_KILOMETRES)
    assert sb_wgs.units == "km"
    assert 1.0 <= sb_wgs.bar_length_value <= 10.0
    fig.canvas.draw()
    plt.close(fig)

    # 5) Sub-metre map still uses 'm' (never cm/mm)
    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, (0.0, 1.0, 0.0, 1.0))
    sb_tiny = mapping.add_scale_bar(ax, epsg_code=32630)
    assert sb_tiny.units == "m"
    plt.close(fig)

    # 6) Invalid units, styles, and locations raise RSGISPyException
    fig, ax = plt.subplots()
    mapping.define_axis_extent(ax, (0.0, 100.0, 0.0, 100.0))
    for bad_unit in ("cm", "mm", "ft", "mi", "deg"):
        with pytest.raises(rsgispy.RSGISPyException):
            mapping.add_scale_bar(ax, units=bad_unit)
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.add_scale_bar(ax, style="dashed")
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.add_scale_bar(ax, loc="nowhere")
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.add_scale_bar(ax, length_fraction=1.5)
    plt.close(fig)


def test_north_arrow_styles_and_map_integration(tmp_path):
    """Test add_north_arrow styles, rotation, errors, and create_*_map integration."""
    styles = (
        mapping.NORTH_ARROW_STYLE_TRIANGLE,
        mapping.NORTH_ARROW_STYLE_ARROW,
        mapping.NORTH_ARROW_STYLE_COMPASS,
        mapping.NORTH_ARROW_STYLE_SIMPLE,
    )
    for style in styles:
        fig, ax = plt.subplots()
        mapping.define_axis_extent(ax, (500000.0, 510000.0, 5200000.0, 5210000.0))
        na = mapping.add_north_arrow(
            ax,
            style=style,
            loc="upper left",
            rotation=15.0 if style == mapping.NORTH_ARROW_STYLE_ARROW else 0.0,
        )
        assert isinstance(na, mapping.MapNorthArrow)
        assert na.style == style
        out_png = tmp_path / f"na_{style}.png"
        fig.savefig(out_png, dpi=72, bbox_inches="tight")
        assert out_png.exists() and out_png.stat().st_size > 0
        plt.close(fig)

    fig, ax = plt.subplots()
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.add_north_arrow(ax, style="unknown_style")
    with pytest.raises(rsgispy.RSGISPyException):
        mapping.add_north_arrow(ax, size=0.0)
    plt.close(fig)

    # Verify integration on create_raster_img_map & create_vec_lyr_map
    arr = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
    tif = tmp_path / "map_int.tif"
    _write_continuous_raster(str(tif), arr, res=100.0, nodata=0.0)

    # Default: scale bar enabled, north arrow NOT added
    fig, ax = plt.subplots()
    mapping.create_raster_img_map(
        ax,
        str(tif),
        img_bands=[1],
        img_stch=rsgispy.IMG_STRETCH_LINEAR,
    )
    assert any(isinstance(a, mapping.MapScaleBar) for a in ax.artists)
    assert not any(isinstance(a, mapping.MapNorthArrow) for a in ax.artists)
    plt.close(fig)

    # Enabled with custom styles and units
    fig, ax = plt.subplots()
    mapping.create_raster_img_map(
        ax,
        str(tif),
        img_bands=[1],
        img_stch=rsgispy.IMG_STRETCH_LINEAR,
        scale_bar_style=mapping.SCALEBAR_STYLE_TICKS,
        scale_bar_units=mapping.SCALEBAR_UNIT_KILOMETRES,
        show_north_arrow=True,
        north_arrow_style=mapping.NORTH_ARROW_STYLE_COMPASS,
        north_arrow_loc="lower left",
    )
    sbs = [a for a in ax.artists if isinstance(a, mapping.MapScaleBar)]
    nas = [a for a in ax.artists if isinstance(a, mapping.MapNorthArrow)]
    assert len(sbs) == 1
    assert sbs[0].style == mapping.SCALEBAR_STYLE_TICKS
    assert sbs[0].units == mapping.SCALEBAR_UNIT_KILOMETRES
    assert len(nas) == 1
    assert nas[0].style == mapping.NORTH_ARROW_STYLE_COMPASS
    out_full = tmp_path / "full_map.png"
    fig.savefig(out_full, dpi=72, bbox_inches="tight")
    assert out_full.exists() and out_full.stat().st_size > 0
    plt.close(fig)


def test_categorical_map_legends(tmp_path):
    """Test categorical/classification legends: inputs, layouts, and placements."""
    from rsgispy.classification.classdata import (
        ClassInfoObj,
        ClassificationConfig,
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
    r_short, c_short = mapping.calc_categorical_legend_layout(
        labels=["Water", "Forest", "Grass", "Urban", "Sand", "Rock"],
        avail_width_in=6.5,
    )
    r_long, c_long = mapping.calc_categorical_legend_layout(
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
        mapping.calc_categorical_legend_layout(["A", "B", "C"], ncols=1, nrows=2)

    # 2. {pixel_val: class_name} dict + embedded raster colour table + overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    mapping.create_thematic_raster_map(
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

    # 3. Companion axes in all 4 positions + explicit ncols/nrows
    for pos in ("right", "left", "bottom", "top"):
        fig, ax = plt.subplots(figsize=(6, 5))
        mapping.create_thematic_raster_map(ax=ax, input_img=str(thm_img))
        leg_ax, lgd = mapping.add_categorical_legend_axis(
            ax=ax,
            classes_info={
                1: ("Forest", "#228b22"),
                2: ("Crop", "#daa520"),
                3: ("Water", "#1e90ff"),
            },
            position=pos,
            ncols=2 if pos in ("bottom", "top") else None,
            swatch_shape="square" if pos == "bottom" else "rectangle",
            title="Legend",
        )
        assert leg_ax in fig.axes
        assert lgd.get_title().get_text() == "Legend"
        plt.close(fig)

    # 4. Existing axis overlay=True and dedicated axis overlay=False
    fig, (ax_map, ax_leg) = plt.subplots(1, 2, figsize=(8, 4))
    mapping.create_thematic_raster_map(ax=ax_map, input_img=str(thm_img))
    lgd_ov = mapping.add_categorical_legend(
        ax=ax_map,
        classes_info={1: "Forest", 2: "Crop", 3: "Water"},
        input_img=str(thm_img),
        loc="bottom left",
        overlay=True,
    )
    assert lgd_ov in ax_map.get_children()
    lgd_panel = mapping.add_categorical_legend(
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

    handles_cfg = mapping.create_categorical_legend_patches(cfg, max_label_width=6)
    assert len(handles_cfg) == 3

    out_lgd_png = tmp_path / "standalone_legend.png"
    r_saved, c_saved = mapping.save_categorical_legend_img(
        out_img_file=out_lgd_png,
        classes_info=cfg_json,
        title="Land Cover Classification",
        dpi=120,
        transparent=True,
    )
    assert out_lgd_png.exists() and out_lgd_png.stat().st_size > 0
    assert r_saved * c_saved >= 3
