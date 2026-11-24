"""
Spatial and attribute selection, deduplication, point matching, and random
subsetting utilities for :mod:`rsgispy.vector` using ``geopandas``,
``rasterio``, and ``shapely``.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
from shapely.geometry import box

try:
    from scipy.spatial import cKDTree  # pylint: disable=import-error

    HAVE_SCIPY = True
except ImportError:
    cKDTree = None  # type: ignore[assignment,misc]
    HAVE_SCIPY = False

import rsgispy
from rsgispy.imageprocess import (
    bbox_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
)
from rsgispy.tools.geometry import (
    do_bboxes_intersect,
    reproj_bbox,
)
import rsgispy.vector._spatial as _vec_spatial
from rsgispy.vectorprocess import (
    VectorInput,
    align_gdf_crs,
    ensure_matching_crs,
    finish_vec_output,
    parse_crs_input,
    read_vec_gdf,
    resolve_out_lyr_name,
    resolve_vec_layer,
    to_python_scalar,
    validate_columns_exist,
    write_vec_gdf,
)
from rsgispy.vector._io_meta import (
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
    vec_file: VectorInput,
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
) -> rsgispy.BBox:
    """Convert ``bbox`` to a :class:`rsgispy.BBox` in ``target_crs``."""
    eff_epsg = bbox_epsg
    if eff_epsg is None and isinstance(bbox, rsgispy.BBox):
        eff_epsg = bbox.epsg
    bbox_obj = rsgispy.BBox(
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
        epsg=eff_epsg,
    )
    if eff_epsg is not None and target_crs is not None:
        src_crs = parse_crs_input(int(eff_epsg))
        if src_crs != target_crs:
            return reproj_bbox(bbox_obj, src_crs, target_crs)
    return bbox_obj


def does_vmsk_img_intersect(
    input_vmsk_img: str,
    vec_roi_file: VectorInput,
    vec_roi_lyr: Optional[str] = None,
    tmp_dir: str = ".",
    vec_epsg: Optional[int] = None,
    *,
    use_pbar: bool = True,
) -> bool:
    """
    Test whether valid pixels (value ``1`` in band 1) of ``input_vmsk_img``
    intersect any geometry in ``vec_roi_file``.

    First tests bounding-box overlap, then streams over image windows to
    rasterise intersecting ROI geometries in memory and test pixel overlap.

    :param input_vmsk_img: Path to binary valid-data mask raster (``1`` = valid).
    :param vec_roi_file: Region-of-interest vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_roi_lyr: Layer name inside ``vec_roi_file`` (or ``None``).
    :param tmp_dir: Retained for API compatibility (unused in-memory check).
    :param vec_epsg: Optional EPSG code overriding the ROI vector layer CRS.
    :param use_pbar: If ``True``, display a tqdm progress bar across raster
                     windows (Default = ``True``).
    :return: ``True`` if at least one valid pixel overlaps the ROI geometries.

    .. code:: python

        import rsgispy.vector.selection

        hits = rsgispy.vector.selection.does_vmsk_img_intersect(
            "valid_mask.tif", "roi.gpkg", "roi"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,unused-argument
    ensure_file_exists(input_vmsk_img, label="Valid mask image")
    roi_lyr = resolve_vec_layer(vec_roi_file, vec_roi_lyr)
    if get_vec_feat_count(vec_roi_file, roi_lyr) == 0:
        return False

    vec_crs = _resolve_crs_for_bbox(vec_roi_file, roi_lyr, vec_epsg)
    try:
        vec_bbox = get_vec_lyr_bbox(
            vec_roi_file, vec_lyr=roi_lyr, compute_if_exp=True
        )
    except rsgispy.RSGISPyException:
        roi_chk = read_vec_gdf(
            vec_roi_file, vec_lyr=roi_lyr, epsg=vec_epsg, copy=False
        )
        roi_chk = roi_chk[
            roi_chk.geometry.notna() & (~roi_chk.geometry.is_empty)
        ]
        if len(roi_chk) == 0:
            return False
        raise

    with rasterio.open(input_vmsk_img, "r") as src:
        img_crs = src.crs
        img_bbox = bbox_from_dataset(src)
        if img_crs is not None and vec_crs is not None and img_crs != vec_crs:
            img_bbox_cmp = reproj_bbox(img_bbox, img_crs, vec_crs)
        else:
            img_bbox_cmp = img_bbox

        if not do_bboxes_intersect(img_bbox_cmp, vec_bbox):
            return False

        roi_gdf = read_vec_gdf(
            vec_roi_file, vec_lyr=roi_lyr, epsg=vec_epsg, copy=False
        )
        roi_gdf = roi_gdf[
            roi_gdf.geometry.notna() & (~roi_gdf.geometry.is_empty)
        ]
        if len(roi_gdf) == 0:
            return False

        roi_gdf = align_gdf_crs(roi_gdf, img_crs)

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
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_roi_file: VectorInput,
    vec_roi_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    predicate: str = "intersects",
    vec_epsg: Optional[int] = None,
    vec_roi_epsg: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Select features from ``vec_file`` that spatially match features in
    ``vec_roi_file`` using ``geopandas.sjoin``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     to select features from.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param vec_roi_file: Region-of-interest vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_roi_lyr: Layer name inside ``vec_roi_file``.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param predicate: Spatial join predicate supported by GeoPandas
                      (Default = ``'intersects'``).
    :param vec_epsg: Optional EPSG code override for ``vec_file``.
    :param vec_roi_epsg: Optional EPSG code override for ``vec_roi_file``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Selected :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.spatial_select(
            "buildings.gpkg",
            "buildings",
            "roi.gpkg",
            "roi",
            "buildings_sel.gpkg",
            "buildings_sel",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    in_gdf = read_vec_gdf(
        vec_file, vec_lyr=vec_lyr, epsg=vec_epsg, copy=False
    )
    roi_gdf = read_vec_gdf(
        vec_roi_file, vec_lyr=vec_roi_lyr, epsg=vec_roi_epsg, copy=False
    )

    roi_gdf = ensure_matching_crs(in_gdf, roi_gdf)

    tmp_id_col = "_rsgispy_sel_idx_"
    work_gdf = gpd.GeoDataFrame(
        {tmp_id_col: np.arange(len(in_gdf), dtype=np.int64)},
        geometry=in_gdf.geometry.values,
        crs=in_gdf.crs,
    )
    roi_geom_gdf = gpd.GeoDataFrame(
        geometry=roi_gdf.geometry.values, crs=roi_gdf.crs
    )

    joined = gpd.sjoin(
        work_gdf,
        roi_geom_gdf,
        how="inner",
        predicate=predicate,
    )
    sel_ids = joined[tmp_id_col].unique()
    if len(sel_ids) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features intersect the ROI."
        )

    out_gdf = in_gdf.iloc[np.sort(sel_ids)].copy()
    return finish_vec_output(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def _select_gdf_by_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Return subset of ``gdf`` whose geometries intersect ``bbox``."""
    bbox_poly = box(bbox[0], bbox[2], bbox[1], bbox[3])
    cand_idxs = gdf.sindex.query(bbox_poly, predicate="intersects")
    if len(cand_idxs) == 0:
        return gdf.iloc[0:0].copy()
    return gdf.iloc[np.sort(cand_idxs)].copy()


def spatial_select_bbox(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    bbox: Sequence[float],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    bbox_epsg: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[bool, gpd.GeoDataFrame]:
    """
    Select features intersecting a bounding box ``(MinX, MaxX, MinY, MaxY)``
    and write them to ``out_vec_file`` (or return the subset GeoDataFrame when
    ``out_vec_file`` is ``None``).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name. If ``None``, reads the first layer.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param bbox_epsg: Optional EPSG code of ``bbox`` if it differs from the
                      input vector layer CRS.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Selected :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``; otherwise ``True`` if at least one feature intersected
             ``bbox``, else ``False``.

    .. code:: python

        import rsgispy.vector.selection

        found = rsgispy.vector.selection.spatial_select_bbox(
            "polys.gpkg",
            "polys",
            (500000.0, 501000.0, 5600000.0, 5601000.0),
            "polys_sub.gpkg",
            "polys_sub",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    target_bbox = _reproj_bbox_to_gdf_crs(bbox, bbox_epsg, gdf.crs)
    sel_gdf = _select_gdf_by_bbox(gdf, target_bbox)

    if out_vec_file is None:
        return sel_gdf
    write_vec_gdf(
        sel_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )
    return len(sel_gdf) > 0


def subset_vec_lyr_to_feat_bboxs(
    vec_file_bbox: VectorInput,
    vec_lyr_bbox: Optional[str],
    vec_file_tosub: VectorInput,
    vec_lyr_tosub: Optional[str],
    out_lyr_name: Optional[str] = None,
    out_file_base: Optional[str] = None,
    out_file_end: str = "gpkg",
    *,
    out_format: str = "GPKG",
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[List[str], List[gpd.GeoDataFrame]]:
    """
    Subset a vector layer once for each feature bounding box in a reference
    vector layer, writing numbered output files (or returning a list of
    :class:`geopandas.GeoDataFrame` objects when ``out_file_base`` is
    ``None``).

    :param vec_file_bbox: Vector file path or :class:`geopandas.GeoDataFrame`
                          defining feature bounding boxes.
    :param vec_lyr_bbox: Layer name in ``vec_file_bbox``.
    :param vec_file_tosub: Vector file path or :class:`geopandas.GeoDataFrame`
                           to subset.
    :param vec_lyr_tosub: Layer name in ``vec_file_tosub``.
    :param out_lyr_name: Layer name written inside every output vector file.
    :param out_file_base: Output filename prefix; numeric indices ``0..n-1``
                          are appended before ``.{out_file_end}``. If
                          ``None``, return a list of GeoDataFrames in memory.
    :param out_file_end: Output file extension without dot (Default = ``'gpkg'``).
    :param out_format: Output vector driver name (Default = ``'GPKG'``).
    :param use_pbar: If ``True``, display a tqdm progress bar across bounding
                     boxes (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: List of created output file paths, or list of subset
             :class:`geopandas.GeoDataFrame` objects when ``out_file_base`` is
             ``None``.

    .. code:: python

        import rsgispy.vector.selection

        out_files = rsgispy.vector.selection.subset_vec_lyr_to_feat_bboxs(
            "tiles.gpkg", "tiles", "roads.gpkg", "roads", "roads", "tile_"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    bbox_gdf = read_vec_gdf(vec_file_bbox, vec_lyr=vec_lyr_bbox, copy=False)
    base_gdf = read_vec_gdf(vec_file_tosub, vec_lyr=vec_lyr_tosub, copy=False)
    bbox_gdf = ensure_matching_crs(base_gdf, bbox_gdf)
    bboxes = _vec_spatial.get_geoms_as_bboxs(bbox_gdf, None)

    pbar_cls = rsgispy.get_tqdm()
    if out_file_base is None:
        return [
            _select_gdf_by_bbox(base_gdf, feat_bbox)
            for feat_bbox in pbar_cls(bboxes, disable=not use_pbar)
        ]

    out_files: List[str] = []
    for idx, feat_bbox in enumerate(pbar_cls(bboxes, disable=not use_pbar)):
        sel_gdf = _select_gdf_by_bbox(base_gdf, feat_bbox)
        out_vec_file = f"{out_file_base}{idx}.{out_file_end}"
        write_vec_gdf(
            sel_gdf,
            out_vec_file=out_vec_file,
            out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_lyr_name),
            out_format=out_format,
            overwrite=True,
            out_compress=out_compress,
        )
        out_files.append(out_vec_file)
    return out_files


def _rows_to_att_dicts(
    df: pd.DataFrame, cols: Sequence[str]
) -> List[Dict[str, Any]]:
    """Extract selected columns as a list of ``{col: python_scalar}`` dicts without iterrows."""
    col_list = list(cols)
    if not col_list:
        return [{} for _ in range(len(df))]
    col_vals = [
        [to_python_scalar(v) for v in df[col].tolist()] for col in col_list
    ]
    return [dict(zip(col_list, row_vals)) for row_vals in zip(*col_vals)]


def get_att_lst_select_feats(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    att_names: Sequence[str],
    vec_sel_file: VectorInput,
    vec_sel_lyr: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Extract attribute dictionaries for features in ``vec_file`` that intersect
    features in ``vec_sel_file``.

    :param vec_file: Vector file path or :class:`geopandas.GeoDataFrame`
                     supplying attributes.
    :param vec_lyr: Layer name in ``vec_file``.
    :param att_names: Sequence of attribute column names to return.
    :param vec_sel_file: Vector file path or :class:`geopandas.GeoDataFrame`
                         defining the spatial selection region.
    :param vec_sel_lyr: Layer name in ``vec_sel_file``.
    :return: List of ``{col_name: value}`` dictionaries.

    .. code:: python

        import rsgispy.vector.selection

        atts = rsgispy.vector.selection.get_att_lst_select_feats(
            "plots.gpkg", "plots", ["plot_id", "species"], "roi.gpkg", "roi"
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(in_gdf, att_names)
    sel_gdf = read_vec_gdf(vec_sel_file, vec_lyr=vec_sel_lyr, copy=False)

    sel_gdf = ensure_matching_crs(in_gdf, sel_gdf)

    left_geom = gpd.GeoDataFrame(
        geometry=in_gdf.geometry.values,
        index=np.arange(len(in_gdf)),
        crs=in_gdf.crs,
    )
    right_geom = gpd.GeoDataFrame(
        geometry=sel_gdf.geometry.values, crs=sel_gdf.crs
    )
    joined = gpd.sjoin(
        left_geom,
        right_geom,
        how="inner",
        predicate="intersects",
    )
    sel_pos = np.sort(joined.index.unique().to_numpy())
    return _rows_to_att_dicts(in_gdf.iloc[sel_pos], att_names)


def get_att_lst_select_bbox_feats(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    att_names: Sequence[str],
    bbox: Sequence[float],
    bbox_epsg: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract attribute dictionaries for features in ``vec_file`` that intersect
    ``bbox`` ``(MinX, MaxX, MinY, MaxY)``.

    :param vec_file: Vector file path or :class:`geopandas.GeoDataFrame`
                     supplying attributes.
    :param vec_lyr: Layer name in ``vec_file``.
    :param att_names: Sequence of attribute column names to return.
    :param bbox: Bounding box ``(MinX, MaxX, MinY, MaxY)``.
    :param bbox_epsg: Optional EPSG code of ``bbox`` if it differs from
                      ``vec_file``.
    :return: List of ``{col_name: value}`` dictionaries.

    .. code:: python

        import rsgispy.vector.selection

        atts = rsgispy.vector.selection.get_att_lst_select_bbox_feats(
            "plots.gpkg", "plots", ["plot_id"], (0.0, 50.0, 0.0, 50.0)
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(in_gdf, att_names)
    target_bbox = _reproj_bbox_to_gdf_crs(bbox, bbox_epsg, in_gdf.crs)
    sel_gdf = _select_gdf_by_bbox(in_gdf, target_bbox)

    return _rows_to_att_dicts(sel_gdf, att_names)


def subset_by_attribute(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    sub_col: str,
    sub_vals: Sequence[Any],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    match_type: str = "equals",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Subset features in a vector layer whose values in ``sub_col`` match any
    item in ``sub_vals``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param sub_col: Attribute column name used to filter features.
    :param sub_vals: Values (or substrings/prefixes when ``match_type`` is
                     ``'contains'`` or ``'start'``) to select.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param match_type: Match strategy: ``'equals'`` (default), ``'contains'``,
                       or ``'start'``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Subset :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.subset_by_attribute(
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

    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(base_gdf, [sub_col])

    mask = pd.Series(False, index=base_gdf.index)
    col_series = base_gdf[sub_col]
    if match_key == "equals":
        for val in sub_vals:
            mask = mask | (col_series == val).fillna(False)
    else:
        not_na = col_series.notna()
        str_series = col_series.astype(str)
        for val in sub_vals:
            if match_key == "contains":
                mask = mask | (
                    not_na
                    & str_series.str.contains(str(val), na=False, regex=False)
                )
            else:
                mask = mask | (
                    not_na & str_series.str.startswith(str(val), na=False)
                )

    out_gdf = base_gdf[mask].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features were selected."
        )

    return finish_vec_output(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def select_feats_str_search(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    select_col: str,
    select_val: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Select features where string column ``select_col`` contains ``select_val``
    (case-sensitive substring/regex search).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param select_col: Attribute column name to search.
    :param select_val: Substring or regular expression to match.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Selected :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.select_feats_str_search(
            "waterways.gpkg", "waterways", "name", "River", "rivers.gpkg", "rivers"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    data_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(data_gdf, [select_col])
    col_vals = data_gdf[select_col].fillna("").astype(str)
    sel_gdf = data_gdf[col_vals.str.contains(select_val, na=False)].copy()

    if len(sel_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features were selected."
        )

    return finish_vec_output(
        sel_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def drop_rows_by_attribute(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    sub_col: str,
    sub_vals: Sequence[Any],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Drop features whose value in ``sub_col`` matches any value in ``sub_vals``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param sub_col: Column name to inspect.
    :param sub_vals: Values to exclude.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Filtered :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.drop_rows_by_attribute(
            "samples.gpkg", "samples", "class", ["Cloud", "Shadow"],
            "clean.gpkg", "clean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(base_gdf, [sub_col])

    out_gdf = base_gdf[~base_gdf[sub_col].isin(list(sub_vals))].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features remained."
        )

    return finish_vec_output(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def drop_rows_by_multi_attributes(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    drop_atts: Dict[str, Union[float, int, str]],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Drop features matching any column/value pair in ``drop_atts``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param drop_atts: Mapping of ``{column_name: value_to_drop}``.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Filtered :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.drop_rows_by_multi_attributes(
            "samples.gpkg", "samples", {"cloud": 1, "quality": "bad"},
            "clean.gpkg", "clean",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    validate_columns_exist(base_gdf, list(drop_atts.keys()))

    keep_mask = pd.Series(True, index=base_gdf.index)
    for col, drop_val in drop_atts.items():
        keep_mask = keep_mask & (~base_gdf[col].isin([drop_val]))

    out_gdf = base_gdf[keep_mask].copy()
    if len(out_gdf) == 0:
        raise rsgispy.RSGISPyException(
            "No output file created as no features remained."
        )

    return finish_vec_output(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def rm_attrib_duplicates(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    dup_col: Union[str, Sequence[str]],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    keep_rows: str = "First",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Remove rows with duplicate values in ``dup_col`` from a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param dup_col: Column name or sequence of column names defining
                    duplicates.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param keep_rows: Duplicate handling option: ``'First'`` (default),
                      ``'Last'``, or ``'RemoveAll'``.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Deduplicated :class:`geopandas.GeoDataFrame` when ``out_vec_file``
             is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.rm_attrib_duplicates(
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

    data_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)
    cols_to_check = (
        [dup_col] if isinstance(dup_col, str) else list(dup_col)
    )
    validate_columns_exist(data_gdf, cols_to_check)

    dedup_gdf = data_gdf.drop_duplicates(
        subset=dup_col, keep=keep_map[keep_rows], ignore_index=False
    ).copy()
    return finish_vec_output(
        dedup_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def match_closest_vec_pts(
    vec_base_file: VectorInput,
    vec_base_lyr: Optional[str],
    vec_match_file: VectorInput,
    vec_match_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GeoJSON",
    tolerance: Optional[float] = None,
    cp_match_atts: bool = False,
    out_x_col: str = "x_match",
    out_y_col: str = "y_match",
    out_dist_col: str = "dist_match",
    out_att_prefix: str = "match_",
    use_pbar: bool = True,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[bool, gpd.GeoDataFrame]:
    """
    Find the closest point in ``vec_match_file`` for each point in
    ``vec_base_file``, optionally filtering matches exceeding ``tolerance`` and
    copying attributes from the matched layer.

    :param vec_base_file: Input base point vector file path or
                          :class:`geopandas.GeoDataFrame`.
    :param vec_base_lyr: Input base point layer name.
    :param vec_match_file: Input target point vector file path or
                           :class:`geopandas.GeoDataFrame`.
    :param vec_match_lyr: Input target point layer name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
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
    :param use_pbar: Retained for API compatibility (vectorised KD-tree query).
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Matched :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``; otherwise ``True`` if at least one point matched and was
             written, else ``False``.

    .. code:: python

        import rsgispy.vector.selection

        matched = rsgispy.vector.selection.match_closest_vec_pts(
            "obs.gpkg", "obs", "ref.gpkg", "ref", "matched.gpkg", "matched",
            out_format="GPKG", tolerance=50.0, cp_match_atts=True,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,unused-argument
    base_gdf = read_vec_gdf(vec_base_file, vec_lyr=vec_base_lyr)
    match_gdf = read_vec_gdf(
        vec_match_file, vec_lyr=vec_match_lyr, copy=False
    )
    match_gdf = ensure_matching_crs(base_gdf, match_gdf)

    base_gdf = base_gdf[
        base_gdf.geometry.notna() & (~base_gdf.geometry.is_empty)
    ].copy()
    match_gdf = match_gdf[
        match_gdf.geometry.notna() & (~match_gdf.geometry.is_empty)
    ]
    if len(base_gdf) == 0 or len(match_gdf) == 0:
        if out_vec_file is None:
            return base_gdf.iloc[0:0].copy()
        return False

    base_xs = base_gdf.geometry.x.to_numpy(dtype=float)
    base_ys = base_gdf.geometry.y.to_numpy(dtype=float)
    match_xs = match_gdf.geometry.x.to_numpy(dtype=float)
    match_ys = match_gdf.geometry.y.to_numpy(dtype=float)

    if HAVE_SCIPY and cKDTree is not None:
        tree = cKDTree(np.column_stack((match_xs, match_ys)))
        dists, best_idxs = tree.query(np.column_stack((base_xs, base_ys)), k=1)
        best_idxs = np.asarray(best_idxs, dtype=np.intp)
        dists = np.asarray(dists, dtype=float)
    else:
        n_base = base_xs.shape[0]
        best_idxs = np.empty(n_base, dtype=np.intp)
        dists = np.empty(n_base, dtype=float)
        chunk_size = 1024
        for start in range(0, n_base, chunk_size):
            end = min(start + chunk_size, n_base)
            dx = base_xs[start:end, np.newaxis] - match_xs[np.newaxis, :]
            dy = base_ys[start:end, np.newaxis] - match_ys[np.newaxis, :]
            sq_dists = dx * dx + dy * dy
            chunk_best = np.argmin(sq_dists, axis=1)
            best_idxs[start:end] = chunk_best
            dists[start:end] = np.sqrt(
                sq_dists[np.arange(end - start), chunk_best]
            )

    out_gdf = base_gdf.copy()
    out_gdf[out_x_col] = match_xs[best_idxs]
    out_gdf[out_y_col] = match_ys[best_idxs]
    out_gdf[out_dist_col] = dists
    if cp_match_atts:
        match_geom_col = match_gdf.geometry.name
        match_cols = [c for c in match_gdf.columns if c != match_geom_col]
        matched_sub = match_gdf.iloc[best_idxs]
        for col in match_cols:
            out_gdf[f"{out_att_prefix}{col}"] = matched_sub[col].values

    if tolerance is not None:
        out_gdf = out_gdf[out_gdf[out_dist_col] < tolerance].copy()

    if len(out_gdf) == 0:
        if out_vec_file is None:
            return out_gdf
        return False

    if out_vec_file is None:
        return out_gdf

    write_vec_gdf(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )
    return True


def split_vec_lyr_random_subset(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_rmain_vec_file: Optional[Union[str, os.PathLike]],
    out_rmain_vec_lyr: Optional[str],
    out_smpl_vec_file: Optional[Union[str, os.PathLike]],
    out_smpl_vec_lyr: Optional[str],
    n_smpl: int,
    *,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[
    Union[
        gpd.GeoDataFrame,
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ]
]:
    """
    Split a vector layer into a random sample of ``n_smpl`` features and a
    complementary remainder layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_rmain_vec_file: Output file path for remaining features, or
                               ``None`` to return the GeoDataFrame in memory.
    :param out_rmain_vec_lyr: Output layer name for remaining features.
    :param out_smpl_vec_file: Output file path for sampled features, or
                              ``None`` to return the GeoDataFrame in memory.
    :param out_smpl_vec_lyr: Output layer name for sampled features.
    :param n_smpl: Number of features to randomly sample.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: ``(remain_gdf, smpl_gdf)`` when both output paths are ``None``,
             the single un-written :class:`geopandas.GeoDataFrame` if only one
             path is ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.split_vec_lyr_random_subset(
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

    base_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, copy=False)
    rng = np.random.default_rng(rnd_seed)
    perm = rng.permutation(len(base_gdf))
    smpl_gdf = base_gdf.iloc[perm[:n_smpl]].copy()
    remain_gdf = base_gdf.iloc[perm[n_smpl:]].copy()

    if out_rmain_vec_file is not None:
        write_vec_gdf(
            remain_gdf,
            out_vec_file=out_rmain_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_rmain_vec_file, out_rmain_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
    if out_smpl_vec_file is not None:
        write_vec_gdf(
            smpl_gdf,
            out_vec_file=out_smpl_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_smpl_vec_file, out_smpl_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )

    if out_rmain_vec_file is None and out_smpl_vec_file is None:
        return remain_gdf, smpl_gdf
    if out_rmain_vec_file is None:
        return remain_gdf
    if out_smpl_vec_file is None:
        return smpl_gdf
    return None


def vec_lyr_random_subset(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    n_smpl: int = 10,
    *,
    out_format: str = "GPKG",
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a random subset of ``n_smpl`` features from an input vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param n_smpl: Number of features to randomly select (Default = ``10``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete ``out_vec_file`` if it exists
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Sampled :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.vec_lyr_random_subset(
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

    base_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, copy=False)
    smpl_gdf = base_gdf.sample(n=n_smpl, random_state=rnd_seed).copy()
    return finish_vec_output(
        smpl_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=resolve_out_lyr_name(out_vec_file, out_vec_lyr),
        out_format=out_format,
        overwrite=del_exist_vec,
        out_compress=out_compress,
    )


def create_train_test_smpls(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_train_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_train_vec_lyr: Optional[str] = None,
    out_test_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_test_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    prop_test: float = 0.2,
    tmp_dir: Optional[str] = None,
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[
    Union[
        gpd.GeoDataFrame,
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    ]
]:
    """
    Split a vector dataset into disjoint training and testing subsets, adding a
    1-based ``fid`` identifier column to preserve original feature provenance.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_train_vec_file: Output file path for training features, or
                               ``None`` to return in memory.
    :param out_train_vec_lyr: Output layer name for training features.
    :param out_test_vec_file: Output file path for testing features, or
                              ``None`` to return in memory.
    :param out_test_vec_lyr: Output layer name for testing features.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param prop_test: Fraction of features assigned to testing
                      (Default = ``0.2``).
    :param tmp_dir: Retained for API compatibility (unused in-memory split).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: ``(train_gdf, test_gdf)`` when both output paths are ``None``,
             the single un-written :class:`geopandas.GeoDataFrame` when one is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.create_train_test_smpls(
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

    rng = np.random.default_rng(rnd_seed)
    perm = rng.permutation(len(base_gdf))
    test_gdf = base_gdf.iloc[perm[:n_test]].copy()
    train_gdf = base_gdf.iloc[perm[n_test:]].copy()

    if out_train_vec_file is not None:
        write_vec_gdf(
            train_gdf,
            out_vec_file=out_train_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_train_vec_file, out_train_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
    if out_test_vec_file is not None:
        write_vec_gdf(
            test_gdf,
            out_vec_file=out_test_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_test_vec_file, out_test_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )

    if out_train_vec_file is None and out_test_vec_file is None:
        return train_gdf, test_gdf
    if out_train_vec_file is None:
        return train_gdf
    if out_test_vec_file is None:
        return test_gdf
    return None


def create_train_test_valid_smpls(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_train_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_train_vec_lyr: Optional[str] = None,
    out_test_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_test_vec_lyr: Optional[str] = None,
    out_valid_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    prop_test: float = 0.2,
    prop_valid: float = 0.2,
    rnd_seed: Optional[int] = None,
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[
    Tuple[
        Optional[gpd.GeoDataFrame],
        Optional[gpd.GeoDataFrame],
        Optional[gpd.GeoDataFrame],
    ]
]:
    """
    Split a vector dataset into mutually disjoint training, testing, and
    validation subsets.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_train_vec_file: Output file path for training features, or
                               ``None`` to return in memory.
    :param out_train_vec_lyr: Output layer name for training features.
    :param out_test_vec_file: Output file path for testing features, or
                              ``None`` to return in memory.
    :param out_test_vec_lyr: Output layer name for testing features.
    :param out_valid_vec_file: Output file path for validation features, or
                               ``None`` to return in memory.
    :param out_valid_vec_lyr: Output layer name for validation features.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param prop_test: Fraction of features assigned to testing
                      (Default = ``0.2``).
    :param prop_valid: Fraction of features assigned to validation
                       (Default = ``0.2``).
    :param rnd_seed: Optional random seed for reproducibility.
    :param del_exist_vec: If ``True``, delete existing output files first
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: ``(train_gdf, test_gdf, valid_gdf)`` when any output path is
             ``None`` (with ``None`` in any slot that was written to disk), or
             ``None`` when all three outputs are written to disk.

    .. code:: python

        import rsgispy.vector.selection

        rsgispy.vector.selection.create_train_test_valid_smpls(
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
    smpl_pts_gdf = read_vec_gdf(vec_file, vec_lyr=lyr_name, copy=False)
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
    valid_gdf = shuffled_gdf.iloc[n_test:n_test + n_valid].copy()
    train_gdf = shuffled_gdf.iloc[n_test + n_valid:].copy()

    if out_test_vec_file is not None:
        write_vec_gdf(
            test_gdf,
            out_vec_file=out_test_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_test_vec_file, out_test_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
    if out_valid_vec_file is not None:
        write_vec_gdf(
            valid_gdf,
            out_vec_file=out_valid_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_valid_vec_file, out_valid_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )
    if out_train_vec_file is not None:
        write_vec_gdf(
            train_gdf,
            out_vec_file=out_train_vec_file,
            out_vec_lyr=resolve_out_lyr_name(
                out_train_vec_file, out_train_vec_lyr
            ),
            out_format=out_format,
            overwrite=del_exist_vec,
            out_compress=out_compress,
        )

    if (
        out_train_vec_file is not None
        and out_test_vec_file is not None
        and out_valid_vec_file is not None
    ):
        return None
    return (
        train_gdf if out_train_vec_file is None else None,
        test_gdf if out_test_vec_file is None else None,
        valid_gdf if out_valid_vec_file is None else None,
    )


def create_n_random_subsets(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    out_vec_dir: Optional[Union[str, os.PathLike]] = None,
    out_vec_base: Optional[str] = None,
    out_vec_ext: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    n_subs: int = 10,
    smpl_frac: float = 0.5,
    n_smpl: Optional[int] = None,
    replacement: bool = False,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Union[List[str], List[gpd.GeoDataFrame]]:
    """
    Create ``n_subs`` random (optionally bootstrapped) subsets of features from
    a vector layer.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name.
    :param out_vec_dir: Output directory path (created if absent), or ``None``
                        to return a list of :class:`geopandas.GeoDataFrame`
                        objects in memory.
    :param out_vec_base: Base prefix for output filenames and layer names
                         (required when ``out_vec_dir`` is provided).
    :param out_vec_ext: Output file extension without leading dot (e.g.,
                        ``'gpkg'``; required when ``out_vec_dir`` is provided).
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
    :param out_compress: Optional GeoParquet compression setting.
    :return: List of created output vector file paths, or list of sampled
             :class:`geopandas.GeoDataFrame` objects when ``out_vec_dir`` is
             ``None``.

    .. code:: python

        import rsgispy.vector.selection

        sub_files = rsgispy.vector.selection.create_n_random_subsets(
            "pts.gpkg", "pts", "bootstraps", "boot", "gpkg",
            n_subs=5, smpl_frac=0.5, rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if n_subs < 1:
        raise rsgispy.RSGISPyException("n_subs must be >= 1.")
    if out_vec_dir is not None:
        if out_vec_base is None or out_vec_ext is None:
            raise rsgispy.RSGISPyException(
                "out_vec_base and out_vec_ext must be provided when "
                "out_vec_dir is specified."
            )
        os.makedirs(out_vec_dir, exist_ok=True)
    base_gdf = read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=False)

    pbar_cls = rsgispy.get_tqdm()
    out_files: List[str] = []
    out_gdfs: List[gpd.GeoDataFrame] = []
    for idx in pbar_cls(range(n_subs), disable=not use_pbar):
        seed_i = None if rnd_seed is None else int(rnd_seed) + idx
        if n_smpl is not None:
            sub_gdf = base_gdf.sample(
                n=n_smpl, frac=None, replace=replacement, random_state=seed_i
            ).copy()
        else:
            sub_gdf = base_gdf.sample(
                n=None,
                frac=smpl_frac,
                replace=replacement,
                random_state=seed_i,
            ).copy()

        if out_vec_dir is None:
            out_gdfs.append(sub_gdf)
        else:
            out_lyr = f"{out_vec_base}_{idx}"
            out_file = os.path.join(
                os.fspath(out_vec_dir), f"{out_lyr}.{out_vec_ext}"
            )
            write_vec_gdf(
                sub_gdf,
                out_vec_file=out_file,
                out_vec_lyr=out_lyr,
                out_format=out_format,
                overwrite=True,
                out_compress=out_compress,
            )
            out_files.append(out_file)
    if out_vec_dir is None:
        return out_gdfs
    return out_files
