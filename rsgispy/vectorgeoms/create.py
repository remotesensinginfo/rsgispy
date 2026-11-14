"""
Vector geometry creation, grid generation, and raster-to-vector conversion
utilities for :mod:`rsgispy.vectorgeoms.create` using ``geopandas``,
``pyogrio``, ``shapely``, ``rasterio``, and ``h3``.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas
import numpy as np
import pandas
import pyogrio
import rasterio
import rasterio.features
import rasterio.transform
import shapely
from shapely.geometry import LineString, Point, Polygon, box, shape

import rsgispy
from rsgispy.imageprocess import (
    _INT_DTYPES,
    ensure_file_exists,
    iter_dataset_windows,
    validate_band_index,
)
from rsgispy.imageutils.masking import gen_valid_mask
from rsgispy.imageutils.metadata import get_img_bbox, get_img_no_data_value
import rsgispy.tools.files
from rsgispy.tools.geometry import (
    get_bbox_geojson_poly,
    get_bbox_grid,
    reproj_bbox,
    unwrap_wgs84_bbox,
)
from rsgispy.tools.projection import get_deg_coord_as_str
import rsgispy.tools.utm
from rsgispy.vectorprocess import (
    CRSLike,
    build_atts_dataframe,
    check_format_name,
    get_file_vec_extension,
    parse_crs_input,
    prepare_out_vec_file,
    read_vec_gdf,
    resolve_out_lyr_name,
    write_vec_gdf,
)

__all__ = [
    "polygonise_raster_to_vec_lyr",
    "vectorise_pxls_to_pts",
    "extract_img_footprint",
    "create_poly_vec_for_lst_bboxs",
    "define_grid",
    "create_poly_vec_bboxs",
    "write_pts_to_vec",
    "create_bboxs_for_pts",
    "create_wgs84_vec_grid",
    "create_vec_for_imgs",
    "create_hex_grid_bbox",
    "create_hex_grid_poly",
    "create_hex_grid_polys",
    "create_random_pts_in_radius",
    "create_random_pts_in_bbox",
]


def _get_h3() -> Any:
    """Import ``h3`` lazily and raise :class:`rsgispy.RSGISPyException` if absent."""
    try:
        import h3  # pylint: disable=import-outside-toplevel

        return h3
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "The 'h3' package is required for hexagon grid functions."
        ) from err


def polygonise_raster_to_vec_lyr(
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str,
    input_img: str,
    img_band: int = 1,
    mask_img: Optional[str] = None,
    mask_band: int = 1,
    replace_file: bool = True,
    replace_lyr: bool = True,
    pxl_val_fieldname: str = "PXLVAL",
    use_8_conn: bool = False,
) -> None:
    """
    Polygonise connected pixel regions in ``input_img`` into a vector layer
    using ``rasterio.features.shapes``.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (e.g., ``'GPKG'``).
    :param input_img: Path to input raster file.
    :param img_band: 1-based band index in ``input_img`` to polygonise
                     (Default = ``1``).
    :param mask_img: Optional path to mask raster where non-zero pixels are
                     polygonised (Default = ``None``).
    :param mask_band: 1-based band index in ``mask_img`` (Default = ``1``).
    :param replace_file: If ``True``, delete an existing ``out_vec_file``
                         before writing (Default = ``True``).
    :param replace_lyr: If ``True``, allow overwriting an existing layer of the
                        same name when ``replace_file=False`` (Default = ``True``).
    :param pxl_val_fieldname: Attribute column name holding raster pixel values
                              (Default = ``'PXLVAL'``).
    :param use_8_conn: If ``True``, use 8-connected pixel grouping; otherwise
                       4-connected (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.polygonise_raster_to_vec_lyr(
            "classes.gpkg", "classes", "GPKG", "classes.tif", img_band=1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if (not replace_file) and (not replace_lyr) and os.path.exists(out_vec_file):
        existing_lyrs = [str(row[0]) for row in pyogrio.list_layers(out_vec_file)]
        if out_vec_lyr in existing_lyrs:
            raise rsgispy.RSGISPyException(
                f"Layer '{out_vec_lyr}' already exists in '{out_vec_file}' "
                "and replace_lyr=False."
            )

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        band_arr = src.read(img_band)
        src_crs = src.crs
        src_transform = src.transform
        is_int = src.dtypes[img_band - 1] in _INT_DTYPES

    mask_arr: Optional[np.ndarray] = None
    if mask_img is not None:
        ensure_file_exists(mask_img, label="Mask image")
        with rasterio.open(mask_img, "r") as msk_src:
            validate_band_index(mask_band, msk_src.count, mask_img)
            mask_arr = msk_src.read(mask_band).astype(bool)

    connectivity = 8 if use_8_conn else 4
    geoms: List[Any] = []
    vals: List[Union[int, float]] = []
    for geom_dict, pxl_val in rasterio.features.shapes(
        band_arr, mask=mask_arr, connectivity=connectivity, transform=src_transform
    ):
        geoms.append(shape(geom_dict))
        vals.append(int(pxl_val) if is_int else float(pxl_val))

    if geoms:
        gdf = geopandas.GeoDataFrame(
            {pxl_val_fieldname: vals}, geometry=geoms, crs=src_crs
        )
    else:
        col_dtype = np.int64 if is_int else np.float64
        gdf = geopandas.GeoDataFrame(
            {pxl_val_fieldname: pandas.Series(dtype=col_dtype)},
            geometry=geopandas.GeoSeries([], crs=src_crs),
            crs=src_crs,
        )

    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=replace_file,
    )


def vectorise_pxls_to_pts(
    input_img: str,
    img_band: int,
    img_msk_val: Union[int, float],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    out_epsg_code: Optional[int] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Create a point vector layer at the centre of every pixel in ``input_img``
    whose value equals ``img_msk_val``.

    :param input_img: Input raster image path.
    :param img_band: 1-based band index to inspect.
    :param img_msk_val: Pixel value identifying pixels to vectorise as points.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_epsg_code: Optional EPSG code overriding the input image CRS.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar across raster
                     windows (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.vectorise_pxls_to_pts(
            "mask.tif", 1, 1, "mask_pts.gpkg", "mask_pts"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)

    with rasterio.open(input_img, "r") as src:
        validate_band_index(img_band, src.count, input_img)
        if out_epsg_code is not None:
            out_crs: Any = parse_crs_input(int(out_epsg_code))
        elif src.crs is not None:
            out_crs = src.crs
        else:
            raise rsgispy.RSGISPyException(
                "Input image has no CRS defined; supply out_epsg_code."
            )

        pts_x: List[float] = []
        pts_y: List[float] = []
        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            arr = src.read(img_band, window=win)
            rows, cols = np.where(arr == img_msk_val)
            if len(rows) == 0:
                continue
            xs, ys = rasterio.transform.xy(
                src.transform,
                rows + int(win.row_off),
                cols + int(win.col_off),
                offset="center",
            )
            pts_x.extend(np.asarray(xs, dtype=float).tolist())
            pts_y.extend(np.asarray(ys, dtype=float).tolist())

    if pts_x:
        gdf = geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(pts_x, pts_y), crs=out_crs
        )
        write_vec_gdf(
            gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
            out_format=out_format,
            overwrite=del_exist_vec,
        )


def extract_img_footprint(
    input_img: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    no_data_val: Optional[float] = None,
    tmp_dir: Optional[str] = None,
    reproj_to: Optional[CRSLike] = None,
    use_pbar: bool = True,
) -> None:
    """
    Extract the valid-data polygon footprint of ``input_img`` and annotate each
    polygon with a ``FileName`` column containing the raster basename.

    :param input_img: Input raster image path.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param no_data_val: Input no-data pixel value (read from image header if
                        ``None``).
    :param tmp_dir: Optional working directory for intermediate valid mask
                    raster; uses a temporary directory if ``None``.
    :param reproj_to: Optional target CRS (e.g., ``'EPSG:4326'`` or ``4326``)
                      for the output footprint layer.
    :param use_pbar: If ``True``, display a tqdm progress bar during mask
                     generation (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.extract_img_footprint(
            "scene.tif", "footprint.gpkg", "footprint", no_data_val=0.0
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ensure_file_exists(input_img)
    if not rsgispy.tools.files.does_path_exists_or_creatable(out_vec_file):
        raise rsgispy.RSGISPyException(
            f"Output file path is not creatable: {out_vec_file}"
        )

    eff_nd = (
        no_data_val
        if no_data_val is not None
        else get_img_no_data_value(input_img)
    )
    if eff_nd is None:
        raise rsgispy.RSGISPyException(
            "A no-data value was not specified and is unset on the image."
        )

    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)

    with tempfile.TemporaryDirectory() as auto_tmp:
        work_dir = tmp_dir if tmp_dir is not None else auto_tmp
        os.makedirs(work_dir, exist_ok=True)
        in_base = os.path.splitext(os.path.basename(input_img))[0]
        valid_out_img = os.path.join(work_dir, f"{in_base}_valid_msk.tif")

        gen_valid_mask(
            input_imgs=input_img,
            output_img=valid_out_img,
            gdalformat="GTIFF",
            no_data_val=float(eff_nd),
            use_pbar=use_pbar,
        )

        with rasterio.open(valid_out_img, "r") as msk_src:
            msk_arr = msk_src.read(1)
            msk_bool = msk_arr == 1
            geoms = [
                shape(geom_dict)
                for geom_dict, _ in rasterio.features.shapes(
                    msk_arr,
                    mask=msk_bool,
                    connectivity=4,
                    transform=msk_src.transform,
                )
            ]
            src_crs = msk_src.crs

        if tmp_dir is not None and os.path.exists(valid_out_img):
            os.remove(valid_out_img)

    gdf = geopandas.GeoDataFrame(
        {"FileName": [os.path.basename(input_img)] * len(geoms)},
        geometry=geoms,
        crs=src_crs,
    )
    if reproj_to is not None and len(gdf) > 0:
        gdf = gdf.to_crs(parse_crs_input(reproj_to))

    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_lyr,
        out_format=out_format,
        overwrite=True,
    )


def create_poly_vec_for_lst_bboxs(
    csv_file: str,
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str,
    epsg_code: CRSLike,
    min_x_col: int = 0,
    max_x_col: int = 1,
    min_y_col: int = 2,
    max_y_col: int = 3,
    ignore_rows: int = 0,
    del_exist_vec: bool = False,
) -> None:
    """
    Create a polygon vector layer from bounding boxes stored one per row in a
    CSV file.

    :param csv_file: Input CSV file path.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param min_x_col: 0-based CSV column index for minimum X (Default = ``0``).
    :param max_x_col: 0-based CSV column index for maximum X (Default = ``1``).
    :param min_y_col: 0-based CSV column index for minimum Y (Default = ``2``).
    :param max_y_col: 0-based CSV column index for maximum Y (Default = ``3``).
    :param ignore_rows: Number of header rows to skip (Default = ``0``).
    :param del_exist_vec: If ``True``, delete existing ``out_vec_file`` first
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_poly_vec_for_lst_bboxs(
            "bboxes.csv", "bboxes.gpkg", "bboxes", "GPKG", 4326, ignore_rows=1
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ensure_file_exists(csv_file, label="CSV file")
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)

    bboxes: List[Tuple[float, float, float, float]] = []
    with open(csv_file, "r", encoding="utf-8") as handle:
        for row_idx, raw_line in enumerate(handle):
            if row_idx < ignore_rows:
                continue
            line = raw_line.strip()
            if not line:
                continue
            comps = line.split(",")
            bboxes.append(
                (
                    float(comps[min_x_col]),
                    float(comps[max_x_col]),
                    float(comps[min_y_col]),
                    float(comps[max_y_col]),
                )
            )

    create_poly_vec_bboxs(
        vec_file=out_vec_file,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=epsg_code,
        bboxs=bboxes,
        overwrite=del_exist_vec,
    )


def create_poly_vec_bboxs(
    vec_file: str,
    vec_lyr: str,
    out_format: str,
    epsg_code: CRSLike,
    bboxs: Sequence[Sequence[float]],
    atts: Optional[Dict[str, Sequence[Any]]] = None,
    att_types: Optional[Dict[str, Sequence[Any]]] = None,
    overwrite: bool = True,
) -> None:
    """
    Create a polygon vector layer from a sequence of ``(MinX, MaxX, MinY, MaxY)``
    bounding boxes with optional attributes.

    :param vec_file: Output vector file path.
    :param vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param bboxs: Sequence of ``(MinX, MaxX, MinY, MaxY)`` bounding boxes.
    :param atts: Optional mapping ``{col_name: values_sequence}``.
    :param att_types: Optional mapping ``{'names': [...], 'types': [...]}``.
    :param overwrite: If ``True``, replace an existing output file; set
                      ``False`` when adding multiple layers to a GeoPackage
                      (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_poly_vec_bboxs(
            "grid.gpkg",
            "grid",
            "GPKG",
            32630,
            [(500000.0, 501000.0, 5600000.0, 5601000.0)],
            atts={"tile_id": [1]},
            att_types={"names": ["tile_id"], "types": [int]},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    crs_obj = parse_crs_input(epsg_code)
    att_cols = build_atts_dataframe(
        len(bboxs), atts=atts, att_types=att_types, length_label="bboxs"
    )
    geoms = [
        box(float(b[0]), float(b[2]), float(b[1]), float(b[3])) for b in bboxs
    ]
    gdf = geopandas.GeoDataFrame(att_cols, geometry=geoms, crs=crs_obj)
    write_vec_gdf(
        gdf,
        out_vec_file=vec_file,
        out_vec_lyr=vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
    )


def define_grid(
    bbox: Sequence[float],
    x_size: float,
    y_size: float,
    in_epsg_code: int,
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str = "GPKG",
    out_epsg_code: Optional[int] = None,
    utm_grid: bool = False,
    utm_hemi: bool = False,
) -> None:
    """
    Create a regular polygon grid covering ``bbox``, optionally reprojected to
    ``out_epsg_code`` or split across UTM zones when ``utm_grid=True``.

    When ``utm_grid=True`` spans multiple UTM zones, separate layers named
    ``{out_vec_lyr}_utm{zone}{n|s}`` are written into a multi-layer format such
    as ``GPKG``.

    :param bbox: Input bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param x_size: Grid cell width in output CRS units.
    :param y_size: Grid cell height in output CRS units.
    :param in_epsg_code: EPSG integer code for ``bbox``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name (or prefix for multi-UTM grids).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_epsg_code: Optional output EPSG integer code.
    :param utm_grid: If ``True``, output grid tiles in UTM zone projections
                     with ``x_size``/``y_size`` in metres (Default = ``False``).
    :param utm_hemi: If ``True``, distinguish Southern hemisphere UTM zones
                     (``utmXXs``); otherwise always use Northern hemisphere
                     (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.define_grid(
            (-4.2, -3.8, 52.2, 52.5),
            10000.0,
            10000.0,
            4326,
            "utm_grid.gpkg",
            "tiles",
            utm_grid=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    in_bbox: Tuple[float, float, float, float] = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )

    if not utm_grid:
        target_epsg = (
            int(out_epsg_code)
            if out_epsg_code is not None
            else int(in_epsg_code)
        )
        proj_bbox = (
            reproj_bbox(in_bbox, int(in_epsg_code), target_epsg)
            if target_epsg != int(in_epsg_code)
            else in_bbox
        )
        grid_bboxes = get_bbox_grid(proj_bbox, x_size, y_size)
        create_poly_vec_bboxs(
            vec_file=out_vec_file,
            vec_lyr=out_vec_lyr,
            out_format=out_format,
            epsg_code=target_epsg,
            bboxs=grid_bboxes,
            overwrite=True,
        )
        return

    wgs84_bbox = (
        reproj_bbox(in_bbox, int(in_epsg_code), 4326)
        if int(in_epsg_code) != 4326
        else in_bbox
    )
    split_wgs84 = unwrap_wgs84_bbox(wgs84_bbox)

    if len(split_wgs84) == 1 and (-180.0 <= split_wgs84[0][0] <= 180.0) and (
        -180.0 <= split_wgs84[0][1] <= 180.0
    ):
        wb = split_wgs84[0]
        utm_tl = rsgispy.tools.utm.from_latlon(wb[3], wb[0])
        utm_tr = rsgispy.tools.utm.from_latlon(wb[3], wb[1])
        utm_br = rsgispy.tools.utm.from_latlon(wb[2], wb[1])
        utm_bl = rsgispy.tools.utm.from_latlon(wb[2], wb[0])
        if utm_tl[2] == utm_tr[2] == utm_br[2] == utm_bl[2]:
            hemi = "S" if (utm_hemi and wb[3] < 0) else "N"
            utm_epsg = int(rsgispy.tools.utm.epsg_for_utm(utm_tl[2], hemi))
            define_grid(
                bbox=in_bbox,
                x_size=x_size,
                y_size=y_size,
                in_epsg_code=in_epsg_code,
                out_vec_file=out_vec_file,
                out_vec_lyr=out_vec_lyr,
                out_format=out_format,
                out_epsg_code=utm_epsg,
                utm_grid=False,
                utm_hemi=False,
            )
            return

    utm_zone_bboxs: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for sub_wb in split_wgs84:
        utm_zone_bboxs.extend(rsgispy.tools.utm.split_wgs84_bbox_utm_zones(sub_wb))

    first = True
    for zone_num, zone_bbox in utm_zone_bboxs:
        hemi = "S" if (utm_hemi and zone_bbox[3] < 0) else "N"
        utm_epsg = int(rsgispy.tools.utm.epsg_for_utm(zone_num, hemi))
        utm_bbox = reproj_bbox(zone_bbox, 4326, utm_epsg)
        grid_bboxes = get_bbox_grid(utm_bbox, x_size, y_size)
        utm_lyr = f"{out_vec_lyr}_utm{zone_num}{hemi.lower()}"
        create_poly_vec_bboxs(
            vec_file=out_vec_file,
            vec_lyr=utm_lyr,
            out_format=out_format,
            epsg_code=utm_epsg,
            bboxs=grid_bboxes,
            overwrite=first,
        )
        first = False


def write_pts_to_vec(
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str,
    epsg_code: CRSLike,
    pts_x: Sequence[float],
    pts_y: Sequence[float],
    atts: Optional[Dict[str, Sequence[Any]]] = None,
    att_types: Optional[Dict[str, Sequence[Any]]] = None,
    replace: bool = True,
) -> None:
    """
    Write X/Y point coordinates and optional attributes to a vector layer.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver.
    :param epsg_code: EPSG integer code or CRS defining coordinate projection.
    :param pts_x: Sequence of point X coordinates.
    :param pts_y: Sequence of point Y coordinates.
    :param atts: Optional mapping ``{col_name: values_sequence}``.
    :param att_types: Optional mapping ``{'names': [...], 'types': [...]}``.
    :param replace: If ``True``, delete an existing ``out_vec_file`` before
                    writing (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.write_pts_to_vec(
            "pts.gpkg",
            "pts",
            "GPKG",
            32630,
            [500100.0, 500200.0],
            [5600100.0, 5600200.0],
            atts={"id": [1, 2]},
            att_types={"names": ["id"], "types": [int]},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if len(pts_x) != len(pts_y):
        raise rsgispy.RSGISPyException(
            "The X and Y coordinate sequences are not the same length."
        )

    crs_obj = parse_crs_input(epsg_code)
    att_cols = build_atts_dataframe(
        len(pts_x), atts=atts, att_types=att_types, length_label="pts_x"
    )
    gdf = geopandas.GeoDataFrame(
        att_cols,
        geometry=geopandas.points_from_xy(pts_x, pts_y),
        crs=crs_obj,
    )
    write_vec_gdf(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=replace,
    )


def create_bboxs_for_pts(
    vec_file: str,
    vec_lyr: Optional[str],
    bbox_width: float,
    bbox_height: float,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    epsg_code: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Create rectangular polygon bounding boxes centred on each point in an input
    point vector layer.

    :param vec_file: Input point vector file path.
    :param vec_lyr: Input point vector layer name.
    :param bbox_width: Output rectangle width in CRS units.
    :param bbox_height: Output rectangle height in CRS units.
    :param out_vec_file: Output polygon vector file path.
    :param out_vec_lyr: Output polygon vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete existing ``out_vec_file`` first
                          (Default = ``False``).
    :param epsg_code: Optional EPSG code overriding the input layer CRS.
    :param use_pbar: If ``True``, display a tqdm progress bar across points
                     (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_bboxs_for_pts(
            "pts.gpkg", "pts", 100.0, 100.0, "pt_boxes.gpkg", "pt_boxes"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, epsg=epsg_code)
    if gdf.crs is None:
        raise rsgispy.RSGISPyException(
            "Input layer has no CRS defined; supply epsg_code."
        )

    h_width = bbox_width / 2.0
    h_height = bbox_height / 2.0
    pbar_cls = rsgispy.get_tqdm()
    out_bboxs: List[Tuple[float, float, float, float]] = []

    for geom in pbar_cls(gdf.geometry, disable=not use_pbar):
        if geom is None or geom.is_empty:
            continue
        pt_x = float(geom.x)
        pt_y = float(geom.y)
        out_bboxs.append(
            (pt_x - h_width, pt_x + h_width, pt_y - h_height, pt_y + h_height)
        )

    create_poly_vec_bboxs(
        vec_file=out_vec_file,
        vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        epsg_code=gdf.crs,
        bboxs=out_bboxs,
        overwrite=False,
    )


def create_wgs84_vec_grid(
    out_vec_file: str,
    out_vec_lyr: str,
    out_format: str,
    grid_x: float,
    grid_y: float,
    bbox: Sequence[float],
    overlap: Optional[float] = None,
    tile_names_col: str = "tile_names",
    tile_name_prefix: str = "",
) -> None:
    """
    Create a regular WGS84 (``EPSG:4326``) grid of polygon tiles named by
    their top-left degree coordinate.

    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver.
    :param grid_x: Cell width in longitude degrees.
    :param grid_y: Cell height in latitude degrees.
    :param bbox: Extent ``(MinX, MaxX, MinY, MaxY)`` in degrees.
    :param overlap: Optional overlap buffer added around each tile in degrees.
    :param tile_names_col: Attribute column name for formatted tile names
                           (Default = ``'tile_names'``).
    :param tile_name_prefix: Prefix prepended to every tile name
                             (Default = ``''``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_wgs84_vec_grid(
            "wgs84_grid.gpkg", "grid", "GPKG", 1.0, 1.0, (-5.0, -3.0, 51.0, 53.0)
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    min_x = float(bbox[0])
    max_x = float(bbox[1])
    min_y = float(bbox[2])
    max_y = float(bbox[3])
    cell_x = float(grid_x)
    cell_y = float(grid_y)

    n_x_cells = int(np.floor((max_x - min_x) / cell_x))
    x_remain = (max_x - min_x) - (cell_x * n_x_cells)
    if x_remain < 1e-9:
        x_remain = 0.0

    n_y_cells = int(np.floor((max_y - min_y) / cell_y))
    y_remain = (max_y - min_y) - (cell_y * n_y_cells)
    if y_remain < 1e-9:
        y_remain = 0.0

    x_spans: List[Tuple[float, float]] = [
        (min_x + j * cell_x, min_x + (j + 1) * cell_x)
        for j in range(n_x_cells)
    ]
    if x_remain > 0:
        x_spans.append((min_x + n_x_cells * cell_x, max_x))

    y_spans: List[Tuple[float, float]] = [
        (max_y - (i + 1) * cell_y, max_y - i * cell_y)
        for i in range(n_y_cells)
    ]
    if y_remain > 0:
        y_spans.append((min_y, max_y - n_y_cells * cell_y))

    ovl = float(overlap) if overlap is not None else 0.0
    bboxes: List[Tuple[float, float, float, float]] = []
    tile_names: List[str] = []

    for c_min_y, c_max_y in y_spans:
        for c_min_x, c_max_x in x_spans:
            coord_str = get_deg_coord_as_str(lat=c_max_y, lon=c_min_x, n_chars=4)
            tile_names.append(f"{tile_name_prefix}{coord_str}")
            bx0 = max(-180.0, c_min_x - ovl)
            bx1 = min(180.0, c_max_x + ovl)
            by0 = max(-90.0, c_min_y - ovl)
            by1 = min(90.0, c_max_y + ovl)
            bboxes.append((bx0, bx1, by0, by1))

    create_poly_vec_bboxs(
        vec_file=out_vec_file,
        vec_lyr=out_vec_lyr,
        out_format=out_format,
        epsg_code=4326,
        bboxs=bboxes,
        atts={tile_names_col: tile_names},
        att_types={"names": [tile_names_col], "types": [str]},
        overwrite=True,
    )


def create_vec_for_imgs(
    input_imgs: Sequence[str],
    output_dir: str,
    out_format: str = "GeoJSON",
    geometry_type: int = rsgispy.GEOM_PT,
    out_name_replace: Optional[Dict[str, str]] = None,
    out_file_ext: Optional[str] = None,
    del_exist_vec: bool = False,
    use_pbar: bool = True,
) -> List[str]:
    """
    Create a single-feature geometry vector file (point at centre, diagonal
    line from top-left to bottom-right, or bounding-box polygon) for each raster
    in ``input_imgs``.

    :param input_imgs: Sequence of input raster image file paths.
    :param output_dir: Output directory path (created if absent).
    :param out_format: Output vector format driver (Default = ``'GeoJSON'``).
    :param geometry_type: One of ``rsgispy.GEOM_PT`` (default),
                          ``rsgispy.GEOM_LINE``, or ``rsgispy.GEOM_POLY``.
    :param out_name_replace: Optional mapping ``{old_sub: new_sub}`` applied to
                             each input basename before naming outputs.
    :param out_file_ext: Output file extension without dot (derived from
                         ``out_format`` when ``None``).
    :param del_exist_vec: If ``True``, overwrite existing output vector files
                          (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar across images
                     (Default = ``True``).
    :return: List of created output vector file paths.

    .. code:: python

        import rsgispy
        import rsgispy.vectorgeoms.create

        out_vecs = rsgispy.vectorgeoms.create.create_vec_for_imgs(
            ["img1.tif", "img2.tif"],
            "out_vecs",
            geometry_type=rsgispy.GEOM_POLY,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if geometry_type not in (rsgispy.GEOM_PT, rsgispy.GEOM_LINE, rsgispy.GEOM_POLY):
        raise rsgispy.RSGISPyException(
            "geometry_type must be rsgispy.GEOM_PT, rsgispy.GEOM_LINE, "
            "or rsgispy.GEOM_POLY."
        )

    fmt = check_format_name(out_format)
    ext = (
        out_file_ext
        if out_file_ext is not None
        else get_file_vec_extension(fmt)
    )
    os.makedirs(output_dir, exist_ok=True)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    for img in pbar_cls(input_imgs, disable=not use_pbar):
        ensure_file_exists(img)
        basename = rsgispy.tools.files.get_file_basename(img, check_valid=True)
        if out_name_replace is not None:
            for old_s, new_s in out_name_replace.items():
                basename = basename.replace(old_s, new_s)

        out_vec_file = os.path.join(output_dir, f"{basename}.{ext}")
        prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)

        with rasterio.open(img, "r") as src:
            img_crs = src.crs
        img_bbox = get_img_bbox(img)

        if geometry_type == rsgispy.GEOM_PT:
            geom: Any = Point(
                img_bbox[0] + ((img_bbox[1] - img_bbox[0]) / 2.0),
                img_bbox[2] + ((img_bbox[3] - img_bbox[2]) / 2.0),
            )
        elif geometry_type == rsgispy.GEOM_LINE:
            geom = LineString(
                [(img_bbox[0], img_bbox[3]), (img_bbox[1], img_bbox[2])]
            )
        else:
            geom = box(img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3])

        gdf = geopandas.GeoDataFrame(geometry=[geom], crs=img_crs)
        write_vec_gdf(
            gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=basename,
            out_format=fmt,
            overwrite=del_exist_vec,
        )
        out_files.append(out_vec_file)

    return out_files


def _h3_ids_to_gdf(hex_ids: Sequence[Any], h3_mod: Any) -> Optional[geopandas.GeoDataFrame]:
    """Convert H3 cell identifiers to a WGS84 Polygon GeoDataFrame."""
    ids = list(hex_ids)
    if not ids:
        return None
    polys = [
        Polygon([(lon, lat) for lat, lon in h3_mod.cell_to_boundary(hid)])
        for hid in ids
    ]
    return geopandas.GeoDataFrame(
        {"hex_id": ids}, geometry=polys, crs="EPSG:4326"
    )


def create_hex_grid_bbox(
    bbox: Sequence[float],
    bbox_epsg: int,
    hex_scale: int,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
) -> None:
    """
    Create an H3 hexagon grid covering ``bbox`` and export it in ``bbox_epsg``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_epsg: EPSG integer code for ``bbox`` and the output layer.
    :param hex_scale: H3 resolution level (integer ``0..15``; higher means
                      smaller hexagons).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_hex_grid_bbox(
            (-4.2, -3.9, 52.3, 52.5), 4326, 7, "hex_bbox.gpkg", "hex"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    h3_mod = _get_h3()
    in_bbox: Tuple[float, float, float, float] = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )
    bbox_wgs84 = (
        reproj_bbox(in_bbox, int(bbox_epsg), 4326)
        if int(bbox_epsg) != 4326
        else in_bbox
    )
    bbox_geojson = get_bbox_geojson_poly(bbox_wgs84)
    hex_ids = h3_mod.h3shape_to_cells(
        h3_mod.geo_to_h3shape(bbox_geojson), int(hex_scale)
    )

    h3_gdf = _h3_ids_to_gdf(hex_ids, h3_mod)
    if h3_gdf is None:
        return
    if int(bbox_epsg) != 4326:
        h3_gdf = h3_gdf.to_crs(epsg=int(bbox_epsg))

    write_vec_gdf(
        h3_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=True,
    )


def create_hex_grid_poly(
    poly_series: Any,
    hex_scale: int,
) -> Optional[geopandas.GeoDataFrame]:
    """
    Create an H3 hexagon GeoDataFrame covering a single WGS84
    (``EPSG:4326``) polygon row or geometry.

    :param poly_series: GeoSeries row (having ``.geometry``) or Shapely
                        Polygon/MultiPolygon in ``EPSG:4326``.
    :param hex_scale: H3 resolution level (integer ``0..15``).
    :return: GeoDataFrame with ``hex_id`` and ``geometry`` in ``EPSG:4326``,
             or ``None`` if no hexagon centres fall inside the polygon.

    .. code:: python

        import geopandas
        import rsgispy.vectorgeoms.create

        gdf = geopandas.read_file("roi_wgs84.gpkg")
        hex_gdf = rsgispy.vectorgeoms.create.create_hex_grid_poly(gdf.iloc[0], 7)
    """
    h3_mod = _get_h3()
    geom = (
        poly_series.geometry
        if hasattr(poly_series, "geometry")
        else poly_series
    )
    if geom is None or geom.is_empty:
        return None
    if geom.has_z:
        geom = shapely.force_2d(geom)

    hex_ids = h3_mod.h3shape_to_cells(
        h3_mod.geo_to_h3shape(geom.__geo_interface__), int(hex_scale)
    )
    return _h3_ids_to_gdf(hex_ids, h3_mod)


def create_hex_grid_polys(
    vec_in_file: str,
    vec_in_lyr: Optional[str],
    hex_scale: int,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    use_pbar: bool = True,
) -> None:
    """
    Create an H3 hexagon grid covering all polygons in ``vec_in_file`` and
    export the result in the input layer's coordinate reference system.

    :param vec_in_file: Input polygon vector file path.
    :param vec_in_lyr: Input vector layer name.
    :param hex_scale: H3 resolution level (integer ``0..15``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar across polygons
                     (Default = ``True``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_hex_grid_polys(
            "polys.gpkg", "polys", 7, "hex_polys.gpkg", "hex_polys"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    polys_gdf = read_vec_gdf(vec_in_file, vec_lyr=vec_in_lyr)
    orig_crs = polys_gdf.crs
    reproj_poly = orig_crs is not None and (
        orig_crs != parse_crs_input(4326)
    )
    if reproj_poly:
        polys_gdf = polys_gdf.to_crs(epsg=4326)

    pbar_cls = rsgispy.get_tqdm()
    hex_parts: List[geopandas.GeoDataFrame] = []
    for idx in pbar_cls(range(len(polys_gdf)), disable=not use_pbar):
        hex_tmp = create_hex_grid_poly(polys_gdf.iloc[idx], hex_scale)
        if hex_tmp is not None and len(hex_tmp) > 0:
            hex_parts.append(hex_tmp)

    if not hex_parts:
        return

    hex_gdf = geopandas.GeoDataFrame(
        pandas.concat(hex_parts, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    if reproj_poly and orig_crs is not None:
        hex_gdf = hex_gdf.to_crs(orig_crs)

    write_vec_gdf(
        hex_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=True,
    )


def create_random_pts_in_radius(
    centre_x: float,
    centre_y: float,
    radius: float,
    n_pts: int,
    epsg_code: CRSLike,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    n_pts_multi_bbox: float = 3.0,
) -> None:
    """
    Generate ``n_pts`` uniformly distributed random points inside a circle of
    ``radius`` around ``(centre_x, centre_y)``, recording ``x_coords``,
    ``y_coords``, ``dist``, and clockwise north-referenced azimuth ``angle``.

    :param centre_x: X coordinate of circle centre (projected CRS).
    :param centre_y: Y coordinate of circle centre (projected CRS).
    :param radius: Circle radius in CRS units.
    :param n_pts: Number of points to generate.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param n_pts_multi_bbox: Oversampling factor within the enclosing square
                             (Default = ``3.0``).

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_random_pts_in_radius(
            500000.0, 5600000.0, 250.0, 50, 32630, "circle_pts.gpkg", "pts",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rng = np.random.default_rng(rnd_seed)
    n_cand = int(n_pts * n_pts_multi_bbox)
    x_cand = rng.uniform(centre_x - radius, centre_x + radius, n_cand)
    y_cand = rng.uniform(centre_y - radius, centre_y + radius, n_cand)

    dist = np.hypot(x_cand - centre_x, y_cand - centre_y)
    in_mask = dist <= radius
    x_in = x_cand[in_mask]
    y_in = y_cand[in_mask]

    if len(x_in) < n_pts:
        raise rsgispy.RSGISPyException(
            "An insufficient number of points fell within the radius; "
            "increase n_pts_multi_bbox."
        )

    sel_idx = rng.choice(len(x_in), size=n_pts, replace=False)
    x_sel = x_in[sel_idx]
    y_sel = y_in[sel_idx]

    dx = x_sel - centre_x
    dy = y_sel - centre_y
    angles = (90.0 - np.rad2deg(np.arctan2(dy, dx))) % 360.0

    data_gdf = geopandas.GeoDataFrame(
        {
            "x_coords": x_sel,
            "y_coords": y_sel,
            "dist": np.hypot(dx, dy),
            "angle": angles,
        },
        geometry=geopandas.points_from_xy(x_sel, y_sel),
        crs=parse_crs_input(epsg_code),
    )
    write_vec_gdf(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=True,
    )


def create_random_pts_in_bbox(
    bbox: Sequence[float],
    n_pts: int,
    epsg_code: CRSLike,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
) -> None:
    """
    Generate ``n_pts`` uniformly distributed random points within
    ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param n_pts: Number of random points to generate.
    :param epsg_code: EPSG integer code or CRS defining output projection.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.

    .. code:: python

        import rsgispy.vectorgeoms.create

        rsgispy.vectorgeoms.create.create_random_pts_in_bbox(
            (500000.0, 501000.0, 5600000.0, 5601000.0),
            100,
            32630,
            "bbox_pts.gpkg",
            "pts",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rng = np.random.default_rng(rnd_seed)
    x_coords = rng.uniform(float(bbox[0]), float(bbox[1]), n_pts)
    y_coords = rng.uniform(float(bbox[2]), float(bbox[3]), n_pts)

    data_gdf = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(x=x_coords, y=y_coords),
        crs=parse_crs_input(epsg_code),
    )
    write_vec_gdf(
        data_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=True,
    )
