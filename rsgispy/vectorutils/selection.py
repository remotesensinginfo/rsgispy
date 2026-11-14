"""
Spatial and attribute selection, deduplication, point matching, and random
subsetting utilities for :mod:`rsgispy.vectorutils` using ``geopandas``,
``rasterio``, and ``shapely``.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas
import numpy as np
import pandas
import rasterio
import rasterio.features
import rasterio.windows
from shapely.geometry import box

import rsgispy
from rsgispy.imageprocess import (
    bbox_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
)
from rsgispy.tools.geometry import (
    calc_pt_distance,
    do_bboxes_intersect,
    reproj_bbox,
)
import rsgispy.vectorgeoms.spatial
from rsgispy.vectorprocess import (
    parse_crs_input,
    read_vec_gdf,
    resolve_out_lyr_name,
    resolve_vec_layer,
    to_python_scalar,
    validate_columns_exist,
    write_vec_gdf,
)
from rsgispy.vectorutils.metadata import (
    get_vec_feat_count,
    get_vec_lyr_bbox,
    get_wkt_proj_from_vec,
)

__all__ = [
    "does_vmsk_img_intersect",
    "spatial_select",
    "spatial_select_bbox",
    "subset_vec_lyr_to_feat_bboxs",
    "get_att_lst_select_feats",
    "get_att_lst_select_bbox_feats",
    "subset_by_attribute",
    "select_feats_str_search",
    "drop_rows_by_attribute",
    "drop_rows_by_multi_attributes",
    "rm_attrib_duplicates",
    "match_closest_vec_pts",
    "split_vec_lyr_random_subset",
    "vec_lyr_random_subset",
    "create_train_test_smpls",
    "create_train_test_valid_smpls",
    "create_n_random_subsets",
]


def _resolve_crs_for_bbox(
    vec_file: str,
    vec_lyr: Optional[str],
    epsg_override: Optional[int],
) -> Any:
    """Resolve CRS from override EPSG or vector layer header metadata."""
    if epsg_override is not None:
        return parse_crs_input(int(epsg_override))
    wkt = get_wkt_proj_from_vec(vec_file, vec_lyr=vec_lyr)
    return parse_crs_input(wkt) if wkt else None


def _reproj_bbox_to_gdf_crs(
    bbox: Sequence[float],
    bbox_epsg: Optional[int],
    target_crs: Any,
) -> Tuple[float, float, float, float]:
    """Convert ``bbox`` to ``(MinX, MaxX, MinY, MaxY)`` in ``target_crs``."""
    bbox_tuple: Tuple[float, float, float, float] = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )
    if bbox_epsg is not None and target_crs is not None:
        src_crs = parse_crs_input(int(bbox_epsg))
        if src_crs != target_crs:
            return reproj_bbox(bbox_tuple, src_crs, target_crs)
    return bbox_tuple


def does_vmsk_img_intersect(
    input_vmsk_img: str,
    vec_roi_file: str,
    vec_roi_lyr: Optional[str] = None,
    tmp_dir: str = ".",
    vec_epsg: Optional[int] = None,
    use_pbar: bool = True,
) -> bool:
    """
    Test whether valid pixels (value ``1`` in band 1) of ``input_vmsk_img``
    intersect any geometry in ``vec_roi_file``.

    First tests bounding-box overlap, then streams over image windows to
    rasterise intersecting ROI geometries in memory and test pixel overlap.

    :param input_vmsk_img: Path to binary valid-data mask raster (``1`` = valid).
    :param vec_roi_file: Path to region-of-interest vector file.
    :param vec_roi_lyr: Layer name inside ``vec_roi_file`` (or ``None``).
    :param tmp_dir: Retained for API compatibility (unused in-memory check).
    :param vec_epsg: Optional EPSG code overriding the ROI vector layer CRS.
    :param use_pbar: If ``True``, display a tqdm progress bar across raster
                     windows (Default = ``True``).
    :return: ``True`` if at least one valid pixel overlaps the ROI geometries.

    .. code:: python

        import rsgispy.vectorutils.selection

        hits = rsgispy.vectorutils.selection.does_vmsk_img_intersect(
            "valid_mask.tif", "roi.gpkg", "roi"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,unused-argument
    ensure_file_exists(input_vmsk_img, label="Valid mask image")
    roi_lyr = resolve_vec_layer(vec_roi_file, vec_roi_lyr)
    vec_crs = _resolve_crs_for_bbox(vec_roi_file, roi_lyr, vec_epsg)
    vec_bbox = get_vec_lyr_bbox(
        vec_roi_file, vec_lyr=roi_lyr, compute_if_exp=True
    )

    with rasterio.open(input_vmsk_img, "r") as src:
        img_crs = src.crs
        img_bbox = bbox_from_dataset(src)
        if img_crs is not None and vec_crs is not None and img_crs != vec_crs:
            img_bbox_cmp = reproj_bbox(img_bbox, img_crs, vec_crs)
        else:
            img_bbox_cmp = img_bbox

        if not do_bboxes_intersect(img_bbox_cmp, vec_bbox):
            return False

        roi_gdf = read_vec_gdf(vec_roi_file, vec_lyr=roi_lyr, epsg=vec_epsg)
        roi_gdf = roi_gdf[
            roi_gdf.geometry.notna() & (~roi_gdf.geometry.is_empty)
        ]
        if len(roi_gdf) == 0:
            return False

        if (
            img_crs is not None
            and roi_gdf.crs is not None
            and roi_gdf.crs != img_crs
        ):
            roi_gdf = roi_gdf.to_crs(img_crs)

        roi_sindex = roi_gdf.sindex
        roi_geoms = roi_gdf.geometry.to_numpy()

        for win in iter_dataset_windows(
            src.height, src.width, use_pbar=use_pbar
        ):
            vmsk_arr = src.read(1, window=win)
            if not np.any(vmsk_arr == 1):
                continue

            min_x, min_y, max_x, max_y = rasterio.windows.bounds(
                win, src.transform
            )
            win_box = box(min_x, min_y, max_x, max_y)
            cand_idxs = roi_sindex.query(win_box, predicate="intersects")
            if len(cand_idxs) == 0:
                continue

            win_transform = src.window_transform(win)
            shapes = [(roi_geoms[idx], 1) for idx in cand_idxs]
            roi_mask = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(int(win.height), int(win.width)),
                transform=win_transform,
                fill=0,
                dtype=np.uint8,
            )
            if np.any((vmsk_arr == 1) & (roi_mask == 1)):
                return True

    return False


def spatial_select(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_roi_file: str,
    vec_roi_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    predicate: str = "intersects",
    vec_epsg: Optional[int] = None,
    vec_roi_epsg: Optional[int] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Select features from ``vec_file`` that spatially match features in
    ``vec_roi_file`` using ``geopandas.sjoin``.

    :param vec_file: Input vector file path to select features from.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param vec_roi_file: Region-of-interest vector file path.
    :param vec_roi_lyr: Layer name inside ``vec_roi_file``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param predicate: Spatial join predicate supported by GeoPandas
                      (Default = ``'intersects'``).
    :param vec_epsg: Optional EPSG code override for ``vec_file``.
    :param vec_roi_epsg: Optional EPSG code override for ``vec_roi_file``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.spatial_select(
            "buildings.gpkg",
            "buildings",
            "roi.gpkg",
            "roi",
            "buildings_sel.gpkg",
            "buildings_sel",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, epsg=vec_epsg)
    roi_gdf = read_vec_gdf(
        vec_roi_file, vec_lyr=vec_roi_lyr, epsg=vec_roi_epsg
    )

    if (
        in_gdf.crs is not None
        and roi_gdf.crs is not None
        and in_gdf.crs != roi_gdf.crs
    ):
        roi_gdf = roi_gdf.to_crs(in_gdf.crs)

    tmp_id_col = "_rsgispy_sel_idx_"
    work_gdf = in_gdf[["geometry"]].copy()
    work_gdf[tmp_id_col] = np.arange(len(work_gdf), dtype=np.int64)

    joined = geopandas.sjoin(
        work_gdf,
        roi_gdf[["geometry"]],
        how="inner",
        predicate=predicate,
    )
    sel_ids = joined[tmp_id_col].unique()
    if len(sel_ids) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features intersect the ROI."
        )

    out_gdf = in_gdf.iloc[np.sort(sel_ids)].copy()
    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def _select_gdf_by_bbox(
    gdf: geopandas.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
) -> geopandas.GeoDataFrame:
    """Return subset of ``gdf`` whose geometries intersect ``bbox``."""
    bbox_poly = box(bbox[0], bbox[2], bbox[1], bbox[3])
    cand_idxs = gdf.sindex.query(bbox_poly, predicate="intersects")
    if len(cand_idxs) == 0:
        return gdf.iloc[0:0].copy()
    return gdf.iloc[np.sort(cand_idxs)].copy()


def spatial_select_bbox(
    vec_file: str,
    vec_lyr: Optional[str],
    bbox: Sequence[float],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    bbox_epsg: Optional[int] = None,
    del_exist_vec: bool = False,
) -> bool:
    """
    Select features intersecting a bounding box ``(MinX, MaxX, MinY, MaxY)``
    and write them to ``out_vec_file``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param bbox_epsg: Optional EPSG code of ``bbox`` if it differs from the
                      input vector layer CRS.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :return: ``True`` if at least one feature intersected ``bbox``, else
             ``False``.

    .. code:: python

        import rsgispy.vectorutils.selection

        found = rsgispy.vectorutils.selection.spatial_select_bbox(
            "polys.gpkg",
            "polys",
            (500000.0, 501000.0, 5600000.0, 5601000.0),
            "polys_sub.gpkg",
            "polys_sub",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    target_bbox = _reproj_bbox_to_gdf_crs(bbox, bbox_epsg, gdf.crs)
    sel_gdf = _select_gdf_by_bbox(gdf, target_bbox)

    write_vec_gdf(
        sel_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    return len(sel_gdf) > 0


def subset_vec_lyr_to_feat_bboxs(
    vec_file_bbox: str,
    vec_lyr_bbox: Optional[str],
    vec_file_tosub: str,
    vec_lyr_tosub: Optional[str],
    out_lyr_name: str,
    out_file_base: str,
    out_file_end: str = "gpkg",
    out_format: str = "GPKG",
    use_pbar: bool = True,
) -> List[str]:
    """
    Subset a vector layer once for each feature bounding box in a reference
    vector layer, writing numbered output files.

    :param vec_file_bbox: Vector file defining feature bounding boxes.
    :param vec_lyr_bbox: Layer name in ``vec_file_bbox``.
    :param vec_file_tosub: Vector file to subset.
    :param vec_lyr_tosub: Layer name in ``vec_file_tosub``.
    :param out_lyr_name: Layer name written inside every output vector file.
    :param out_file_base: Output filename prefix; numeric indices ``0..n-1``
                          are appended before ``.{out_file_end}``.
    :param out_file_end: Output file extension without dot (Default = ``'gpkg'``).
    :param out_format: Output vector driver name (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar across bounding
                     boxes (Default = ``True``).
    :return: List of created output file paths.

    .. code:: python

        import rsgispy.vectorutils.selection

        out_files = rsgispy.vectorutils.selection.subset_vec_lyr_to_feat_bboxs(
            "tiles.gpkg", "tiles", "roads.gpkg", "roads", "roads", "tile_"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    bboxes = rsgispy.vectorgeoms.spatial.get_geoms_as_bboxs(vec_file_bbox, vec_lyr_bbox)
    base_gdf = read_vec_gdf(vec_file_tosub, vec_lyr=vec_lyr_tosub)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    for idx, feat_bbox in enumerate(pbar_cls(bboxes, disable=not use_pbar)):
        sel_gdf = _select_gdf_by_bbox(base_gdf, feat_bbox)
        out_vec_file = f"{out_file_base}{idx}.{out_file_end}"
        write_vec_gdf(
            sel_gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=out_lyr_name,
            out_format=out_format,
            overwrite=True,
        )
        out_files.append(out_vec_file)
    return out_files


def get_att_lst_select_feats(
    vec_file: str,
    vec_lyr: Optional[str],
    att_names: Sequence[str],
    vec_sel_file: str,
    vec_sel_lyr: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Extract attribute dictionaries for features in ``vec_file`` that intersect
    features in ``vec_sel_file``.

    :param vec_file: Vector file supplying attributes.
    :param vec_lyr: Layer name in ``vec_file``.
    :param att_names: Sequence of attribute column names to return.
    :param vec_sel_file: Vector file defining the spatial selection region.
    :param vec_sel_lyr: Layer name in ``vec_sel_file``.
    :return: List of ``{col_name: value}`` dictionaries.

    .. code:: python

        import rsgispy.vectorutils.selection

        atts = rsgispy.vectorutils.selection.get_att_lst_select_feats(
            "plots.gpkg", "plots", ["plot_id", "species"], "roi.gpkg", "roi"
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(in_gdf, att_names)
    sel_gdf = read_vec_gdf(vec_sel_file, vec_lyr=vec_sel_lyr)

    if (
        in_gdf.crs is not None
        and sel_gdf.crs is not None
        and in_gdf.crs != sel_gdf.crs
    ):
        sel_gdf = sel_gdf.to_crs(in_gdf.crs)

    cols_to_read = list(att_names)
    joined = geopandas.sjoin(
        in_gdf[cols_to_read + ["geometry"]],
        sel_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )

    return [
        {col: to_python_scalar(row[col]) for col in cols_to_read}
        for _, row in joined.iterrows()
    ]


def get_att_lst_select_bbox_feats(
    vec_file: str,
    vec_lyr: Optional[str],
    att_names: Sequence[str],
    bbox: Sequence[float],
    bbox_epsg: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract attribute dictionaries for features in ``vec_file`` that intersect
    ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param vec_file: Vector file supplying attributes.
    :param vec_lyr: Layer name in ``vec_file``.
    :param att_names: Sequence of attribute column names to return.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_epsg: Optional EPSG code of ``bbox`` if it differs from
                      ``vec_file``.
    :return: List of ``{col_name: value}`` dictionaries.

    .. code:: python

        import rsgispy.vectorutils.selection

        atts = rsgispy.vectorutils.selection.get_att_lst_select_bbox_feats(
            "plots.gpkg", "plots", ["plot_id"], (0.0, 50.0, 0.0, 50.0)
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(in_gdf, att_names)
    target_bbox = _reproj_bbox_to_gdf_crs(bbox, bbox_epsg, in_gdf.crs)
    sel_gdf = _select_gdf_by_bbox(in_gdf, target_bbox)

    cols_to_read = list(att_names)
    return [
        {col: to_python_scalar(row[col]) for col in cols_to_read}
        for _, row in sel_gdf.iterrows()
    ]


def subset_by_attribute(
    vec_file: str,
    vec_lyr: Optional[str],
    sub_col: str,
    sub_vals: Sequence[Any],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    match_type: str = "equals",
    del_exist_vec: bool = False,
) -> None:
    """
    Subset features in a vector layer whose values in ``sub_col`` match any
    item in ``sub_vals``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param sub_col: Attribute column name used to filter features.
    :param sub_vals: Values (or regex patterns when ``match_type`` is
                     ``'contains'`` or ``'start'``) to select.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param match_type: Match strategy: ``'equals'`` (default), ``'contains'``,
                       or ``'start'``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.subset_by_attribute(
            "landcover.gpkg",
            "landcover",
            "class_name",
            ["Forest", "Water"],
            "subset.gpkg",
            "subset",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    match_key = match_type.lower()
    if match_key not in ("equals", "contains", "start"):
        raise rsgispy.RSGISPyException(
            "match_type must be either 'equals', 'contains' or 'start'."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(base_gdf, [sub_col])

    mask = pandas.Series(False, index=base_gdf.index)
    col_series = base_gdf[sub_col]
    for val in sub_vals:
        if match_key == "equals":
            mask = mask | (col_series == val)
        elif match_key == "contains":
            mask = mask | col_series.astype(str).str.contains(
                str(val), na=False
            )
        else:
            mask = mask | col_series.astype(str).str.match(str(val), na=False)

    out_gdf = base_gdf[mask].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features were selected."
        )

    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def select_feats_str_search(
    vec_file: str,
    vec_lyr: Optional[str],
    select_col: str,
    select_val: str,
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Select features where string column ``select_col`` contains ``select_val``
    (case-sensitive substring/regex search).

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param select_col: Attribute column name to search.
    :param select_val: Substring or regular expression to match.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.select_feats_str_search(
            "waterways.gpkg", "waterways", "name", "River", "rivers.gpkg", "rivers"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(data_gdf, [select_col])
    col_vals = data_gdf[select_col].fillna("").astype(str)
    sel_gdf = data_gdf[col_vals.str.contains(select_val, na=False)].copy()

    if len(sel_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features were selected."
        )

    write_vec_gdf(
        sel_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def drop_rows_by_attribute(
    vec_file: str,
    vec_lyr: Optional[str],
    sub_col: str,
    sub_vals: Sequence[Any],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Drop features whose value in ``sub_col`` matches any value in ``sub_vals``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param sub_col: Column name to inspect.
    :param sub_vals: Values to exclude.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.drop_rows_by_attribute(
            "samples.gpkg", "samples", "class", ["Cloud", "Shadow"],
            "clean.gpkg", "clean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(base_gdf, [sub_col])

    out_gdf = base_gdf[~base_gdf[sub_col].isin(list(sub_vals))].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features remained."
        )

    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def drop_rows_by_multi_attributes(
    vec_file: str,
    vec_lyr: Optional[str],
    drop_atts: Dict[str, Union[float, int, str]],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Drop features matching any column/value pair in ``drop_atts``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param drop_atts: Mapping of ``{column_name: value_to_drop}``.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.drop_rows_by_multi_attributes(
            "samples.gpkg", "samples", {"cloud": 1, "quality": "bad"},
            "clean.gpkg", "clean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    validate_columns_exist(base_gdf, list(drop_atts.keys()))

    keep_mask = pandas.Series(True, index=base_gdf.index)
    for col, drop_val in drop_atts.items():
        keep_mask = keep_mask & (base_gdf[col] != drop_val)

    out_gdf = base_gdf[keep_mask].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features remained."
        )

    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def rm_attrib_duplicates(
    vec_file: str,
    vec_lyr: Optional[str],
    dup_col: Union[str, Sequence[str]],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    keep_rows: str = "First",
    del_exist_vec: bool = False,
) -> None:
    """
    Remove rows with duplicate values in ``dup_col`` from a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param dup_col: Column name or sequence of column names defining
                    duplicates.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param keep_rows: Duplicate handling option: ``'First'`` (default),
                      ``'Last'``, or ``'RemoveAll'``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.rm_attrib_duplicates(
            "points.gpkg", "points", "site_id", "points_unq.gpkg", "points_unq"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    keep_map: Dict[str, Any] = {
        "First": "first",
        "Last": "last",
        "RemoveAll": False,
    }
    if keep_rows not in keep_map:
        raise rsgispy.RSGISPyException(
            "keep_rows option must be one of: 'First', 'Last', 'RemoveAll'."
        )

    data_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)
    cols_to_check = (
        [dup_col] if isinstance(dup_col, str) else list(dup_col)
    )
    validate_columns_exist(data_gdf, cols_to_check)

    dedup_gdf = data_gdf.drop_duplicates(
        subset=dup_col, keep=keep_map[keep_rows], ignore_index=False
    )
    write_vec_gdf(
        dedup_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def match_closest_vec_pts(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_match_file: str,
    vec_match_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GeoJSON",
    tolerance: Optional[float] = None,
    cp_match_atts: bool = False,
    out_x_col: str = "x_match",
    out_y_col: str = "y_match",
    out_dist_col: str = "dist_match",
    out_att_prefix: str = "match_",
    use_pbar: bool = True,
    del_exist_vec: bool = False,
) -> bool:
    """
    Find the closest point in ``vec_match_file`` for each point in
    ``vec_base_file``, optionally filtering matches exceeding ``tolerance`` and
    copying attributes from the matched layer.

    :param vec_base_file: Input base point vector file path.
    :param vec_base_lyr: Input base point layer name.
    :param vec_match_file: Input target point vector file path.
    :param vec_match_lyr: Input target point layer name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GeoJSON'``).
    :param tolerance: Optional maximum distance threshold; matches with
                      distance ``>= tolerance`` are excluded.
    :param cp_match_atts: If ``True``, copy attribute columns from
                          ``vec_match_file`` prefixed by ``out_att_prefix``
                          (Default = ``False``).
    :param out_x_col: Output column name for matched X coordinate
                      (Default = ``'x_match'``).
    :param out_y_col: Output column name for matched Y coordinate
                      (Default = ``'y_match'``).
    :param out_dist_col: Output column name for Euclidean distance
                         (Default = ``'dist_match'``).
    :param out_att_prefix: Column name prefix when ``cp_match_atts=True``
                           (Default = ``'match_'``).
    :param use_pbar: If ``True``, display a tqdm progress bar across base points
                     (Default = ``True``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :return: ``True`` if at least one point matched and was written, else
             ``False``.

    .. code:: python

        import rsgispy.vectorutils.selection

        matched = rsgispy.vectorutils.selection.match_closest_vec_pts(
            "obs.gpkg", "obs", "ref.gpkg", "ref", "matched.gpkg", "matched",
            out_format="GPKG", tolerance=50.0, cp_match_atts=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    base_gdf = read_vec_gdf(vec_base_file, vec_lyr=vec_base_lyr)
    match_gdf = read_vec_gdf(vec_match_file, vec_lyr=vec_match_lyr)
    if len(base_gdf) == 0 or len(match_gdf) == 0:
        return False

    base_xs = base_gdf.geometry.x.to_numpy(dtype=float)
    base_ys = base_gdf.geometry.y.to_numpy(dtype=float)
    match_xs = match_gdf.geometry.x.to_numpy(dtype=float)
    match_ys = match_gdf.geometry.y.to_numpy(dtype=float)

    out_x: List[float] = []
    out_y: List[float] = []
    out_dists: List[float] = []
    match_cols = [c for c in match_gdf.columns if c != "geometry"]
    match_atts: Dict[str, List[Any]] = {
        col: [] for col in match_cols
    } if cp_match_atts else {}

    pbar_cls = rsgispy.get_tqdm()
    for base_x, base_y in pbar_cls(
        zip(base_xs, base_ys), total=len(base_xs), disable=not use_pbar
    ):
        dists = np.hypot(match_xs - base_x, match_ys - base_y)
        best_idx = int(np.argmin(dists))
        join_x = float(match_xs[best_idx])
        join_y = float(match_ys[best_idx])
        dist_val = calc_pt_distance(float(base_x), float(base_y), join_x, join_y)

        out_x.append(join_x)
        out_y.append(join_y)
        out_dists.append(dist_val)

        if cp_match_atts:
            row = match_gdf.iloc[best_idx]
            for col in match_cols:
                match_atts[col].append(to_python_scalar(row[col]))

    out_gdf = base_gdf.copy()
    out_gdf[out_x_col] = out_x
    out_gdf[out_y_col] = out_y
    out_gdf[out_dist_col] = out_dists
    if cp_match_atts:
        for col in match_cols:
            out_gdf[f"{out_att_prefix}{col}"] = match_atts[col]

    if tolerance is not None:
        out_gdf = out_gdf[out_gdf[out_dist_col] < tolerance].copy()

    if len(out_gdf) == 0:
        return False

    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    return True


def split_vec_lyr_random_subset(
    vec_file: str,
    vec_lyr: Optional[str],
    out_rmain_vec_file: str,
    out_rmain_vec_lyr: Optional[str],
    out_smpl_vec_file: str,
    out_smpl_vec_lyr: Optional[str],
    n_smpl: int,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Split a vector layer into a random sample of ``n_smpl`` features and a
    complementary remainder layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_rmain_vec_file: Output file path for remaining features.
    :param out_rmain_vec_lyr: Output layer name for remaining features.
    :param out_smpl_vec_file: Output file path for sampled features.
    :param out_smpl_vec_lyr: Output layer name for sampled features.
    :param n_smpl: Number of features to randomly sample.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.split_vec_lyr_random_subset(
            "pts.gpkg", "pts", "remain.gpkg", "remain",
            "sample.gpkg", "sample", n_smpl=50, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    n_feats = get_vec_feat_count(vec_file, lyr_name)
    if n_smpl < 1 or n_smpl >= n_feats:
        raise rsgispy.RSGISPyException(
            "n_smpl must be >= 1 and strictly less than the number of features."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    smpl_gdf = base_gdf.sample(n=n_smpl, random_state=rnd_seed)
    remain_gdf = base_gdf.drop(smpl_gdf.index)

    write_vec_gdf(
        remain_gdf,
        out_vec_file=out_rmain_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_rmain_vec_file, out_rmain_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    write_vec_gdf(
        smpl_gdf,
        out_vec_file=out_smpl_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_smpl_vec_file, out_smpl_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def vec_lyr_random_subset(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    n_smpl: int,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Write a random subset of ``n_smpl`` features from an input vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param n_smpl: Number of features to randomly select.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.vec_lyr_random_subset(
            "pts.gpkg", "pts", "sub.gpkg", "sub", n_smpl=25, rnd_seed=42
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    n_feats = get_vec_feat_count(vec_file, lyr_name)
    if n_smpl < 1 or n_smpl > n_feats:
        raise rsgispy.RSGISPyException(
            "n_smpl must be between 1 and the total feature count."
        )

    base_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    smpl_gdf = base_gdf.sample(n=n_smpl, random_state=rnd_seed)
    write_vec_gdf(
        smpl_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def create_train_test_smpls(
    vec_file: str,
    vec_lyr: Optional[str],
    out_train_vec_file: str,
    out_train_vec_lyr: Optional[str],
    out_test_vec_file: str,
    out_test_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    prop_test: float = 0.2,
    tmp_dir: Optional[str] = None,
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Split a vector dataset into disjoint training and testing subsets, adding a
    1-based ``fid`` identifier column to preserve original feature provenance.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_train_vec_file: Output file path for training features.
    :param out_train_vec_lyr: Output layer name for training features.
    :param out_test_vec_file: Output file path for testing features.
    :param out_test_vec_lyr: Output layer name for testing features.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param prop_test: Fraction of features assigned to testing
                      (Default = ``0.2``).
    :param tmp_dir: Retained for API compatibility (unused in-memory split).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.create_train_test_smpls(
            "samples.gpkg", "samples",
            "train.gpkg", "train",
            "test.gpkg", "test",
            prop_test=0.25,
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    if prop_test <= 0.0 or prop_test >= 1.0:
        raise rsgispy.RSGISPyException(
            "prop_test must be strictly between 0 and 1."
        )

    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    base_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    base_gdf["fid"] = np.arange(1, len(base_gdf) + 1, 1, dtype=int)

    n_test = int(len(base_gdf) * prop_test)
    if n_test < 1 or n_test >= len(base_gdf):
        raise rsgispy.RSGISPyException(
            "Dataset too small for the requested prop_test split."
        )

    test_gdf = base_gdf.sample(n=n_test, random_state=rnd_seed)
    train_gdf = base_gdf.drop(test_gdf.index)

    write_vec_gdf(
        train_gdf,
        out_vec_file=out_train_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_train_vec_file, out_train_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    write_vec_gdf(
        test_gdf,
        out_vec_file=out_test_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_test_vec_file, out_test_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def create_train_test_valid_smpls(
    vec_file: str,
    vec_lyr: Optional[str],
    out_train_vec_file: str,
    out_train_vec_lyr: Optional[str],
    out_test_vec_file: str,
    out_test_vec_lyr: Optional[str],
    out_valid_vec_file: str,
    out_valid_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    prop_test: float = 0.2,
    prop_valid: float = 0.2,
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
) -> None:
    """
    Split a vector dataset into mutually disjoint training, testing, and
    validation subsets.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_train_vec_file: Output file path for training features.
    :param out_train_vec_lyr: Output layer name for training features.
    :param out_test_vec_file: Output file path for testing features.
    :param out_test_vec_lyr: Output layer name for testing features.
    :param out_valid_vec_file: Output file path for validation features.
    :param out_valid_vec_lyr: Output layer name for validation features.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param prop_test: Fraction of features assigned to testing
                      (Default = ``0.2``).
    :param prop_valid: Fraction of features assigned to validation
                       (Default = ``0.2``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorutils.selection

        rsgispy.vectorutils.selection.create_train_test_valid_smpls(
            "samples.gpkg", "samples",
            "train.gpkg", "train",
            "test.gpkg", "test",
            "valid.gpkg", "valid",
            prop_test=0.2, prop_valid=0.2, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if (
        prop_test <= 0.0
        or prop_valid <= 0.0
        or (prop_test + prop_valid) >= 1.0
    ):
        raise rsgispy.RSGISPyException(
            "prop_test and prop_valid must be > 0 and sum to < 1."
        )

    lyr_name = resolve_vec_layer(vec_file, vec_lyr)
    smpl_pts_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name)
    n_smpls = len(smpl_pts_gdf)
    n_test = int(n_smpls * prop_test)
    n_valid = int(n_smpls * prop_valid)
    n_train = n_smpls - (n_test + n_valid)
    if n_test < 1 or n_valid < 1 or n_train < 1:
        raise rsgispy.RSGISPyException(
            "Dataset too small to allocate disjoint train/test/valid splits."
        )

    shuffled_gdf = smpl_pts_gdf.sample(
        frac=1.0, random_state=rnd_seed
    ).reset_index(drop=True)

    test_gdf = shuffled_gdf.iloc[:n_test].copy()
    valid_gdf = shuffled_gdf.iloc[n_test : n_test + n_valid].copy()
    train_gdf = shuffled_gdf.iloc[n_test + n_valid :].copy()

    write_vec_gdf(
        test_gdf,
        out_vec_file=out_test_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_test_vec_file, out_test_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    write_vec_gdf(
        valid_gdf,
        out_vec_file=out_valid_vec_file,
        out_vec_lyr=resolve_out_lyr_name(
            out_valid_vec_file, out_valid_vec_lyr
        ),
        out_format=out_format,
        overwrite=del_exist_vec,
    )
    write_vec_gdf(
        train_gdf,
        out_vec_file=out_train_vec_file,
        out_vec_lyr=resolve_out_lyr_name(
            out_train_vec_file, out_train_vec_lyr
        ),
        out_format=out_format,
        overwrite=del_exist_vec,
    )


def create_n_random_subsets(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_dir: str,
    out_vec_base: str,
    out_vec_ext: str,
    out_format: str = "GPKG",
    n_subs: int = 10,
    smpl_frac: float = 0.5,
    n_smpl: Optional[int] = None,
    replacement: bool = False,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> List[str]:
    """
    Create ``n_subs`` random (optionally bootstrapped) subsets of features from
    a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name.
    :param out_vec_dir: Output directory path (created if absent).
    :param out_vec_base: Base prefix for output filenames and layer names.
    :param out_vec_ext: Output file extension without leading dot (e.g., ``'gpkg'``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param n_subs: Number of subsets to generate (Default = ``10``).
    :param smpl_frac: Fraction of features per subset when ``n_smpl`` is
                      ``None`` (Default = ``0.5``).
    :param n_smpl: Optional exact feature count per subset (overrides
                   ``smpl_frac`` when supplied).
    :param replacement: If ``True``, sample with replacement
                        (Default = ``False``).
    :param rnd_seed: Optional base random seed (incremented per subset for
                     distinct reproducible samples).
    :param use_pbar: If ``True``, display a tqdm progress bar across subsets
                     (Default = ``True``).
    :return: List of created output vector file paths.

    .. code:: python

        import rsgispy.vectorutils.selection

        sub_files = rsgispy.vectorutils.selection.create_n_random_subsets(
            "pts.gpkg", "pts", "bootstraps", "boot", "gpkg",
            n_subs=5, smpl_frac=0.5, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_subs < 1:
        raise rsgispy.RSGISPyException("n_subs must be >= 1.")
    os.makedirs(out_vec_dir, exist_ok=True)
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    for idx in pbar_cls(range(n_subs), disable=not use_pbar):
        seed_i = None if rnd_seed is None else int(rnd_seed) + idx
        if n_smpl is not None:
            sub_gdf = base_gdf.sample(
                n=n_smpl, frac=None, replace=replacement, random_state=seed_i
            )
        else:
            sub_gdf = base_gdf.sample(
                n=None,
                frac=smpl_frac,
                replace=replacement,
                random_state=seed_i,
            )

        out_lyr = f"{out_vec_base}_{idx}"
        out_file = os.path.join(out_vec_dir, f"{out_lyr}.{out_vec_ext}")
        write_vec_gdf(
            sub_gdf,
            out_vec_file=out_file,
            out_vec_lyr=out_lyr,
            out_format=out_format,
            overwrite=True,
        )
        out_files.append(out_file)
    return out_files
