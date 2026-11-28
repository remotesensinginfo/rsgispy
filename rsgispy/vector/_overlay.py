"""
The vector.geoms module provides vector layer overlay operations
(intersection, difference, symmetric difference, identity, union) and
region-of-interest clipping functions using ``geopandas``.
"""

import os
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

import rsgispy
import rsgispy.tools.strings
from rsgispy.vectorprocess import (
    VectorInput,
    check_format_name,
    ensure_matching_crs,
    finish_vec_output,
    prepare_optional_out_vec,
    read_vec_gdf,
    resolve_out_lyr_name,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "vec_lyr_intersection_gp",
    "vec_lyr_difference_gp",
    "vec_lyr_sym_difference_gp",
    "vec_lyr_identity_gp",
    "vec_lyr_union_gp",
    "clip_vec_lyr",
    "clip_and_merge_with_roi",
    "clip_vec_into_sets",
]


def _overlay_vec_lyrs(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    del_exist_vec: bool,
    how: str,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Shared implementation for ``geopandas.overlay`` vector layer operations."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_path = prepare_optional_out_vec(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_path, out_vec_lyr)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    over_data_gdf = read_vec_gdf(vec_over_file, vec_over_lyr)
    over_data_gdf = ensure_matching_crs(data_gdf, over_data_gdf)
    res_gdf = gpd.overlay(data_gdf, over_data_gdf, how=how)

    return finish_vec_output(
        res_gdf,
        out_path,
        out_vec_lyr=out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )


def vec_lyr_intersection_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a geometric intersection between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Result :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_intersection_gp(
            "a.gpkg", "a", "b.gpkg", "b", "inter.gpkg", "inter"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="intersection",
        out_compress=out_compress,
    )


def vec_lyr_difference_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a geometric difference between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path or :class:`geopandas.GeoDataFrame`
                          to subtract.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Result :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_difference_gp(
            "a.gpkg", "a", "b.gpkg", "b", "diff.gpkg", "diff"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="difference",
        out_compress=out_compress,
    )


def vec_lyr_sym_difference_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a symmetric difference between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Result :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_sym_difference_gp(
            "a.gpkg", "a", "b.gpkg", "b", "symdiff.gpkg", "symdiff"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="symmetric_difference",
        out_compress=out_compress,
    )


def vec_lyr_identity_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a geometric identity overlay between ``vec_file`` and
    ``vec_over_file`` using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Result :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_identity_gp(
            "a.gpkg", "a", "b.gpkg", "b", "identity.gpkg", "identity"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="identity",
        out_compress=out_compress,
    )


def vec_lyr_union_gp(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_over_file: VectorInput,
    vec_over_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Perform a geometric union between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Result :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.vec_lyr_union_gp(
            "a.gpkg", "a", "b.gpkg", "b", "union.gpkg", "union"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="union",
        out_compress=out_compress,
    )


def clip_vec_lyr(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_roi_file: VectorInput,
    vec_roi_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Clip a vector layer to the region of interest defined by ``vec_roi_file``
    using ``geopandas.clip``.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_roi_lyr: ROI polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Clipped :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.clip_vec_lyr(
            "roads.gpkg", "roads", "roi.gpkg", "roi", "clipped.gpkg", "roads"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr, copy=False)
    roi_gdf = ensure_matching_crs(base_gdf, roi_gdf)
    clipped_gdf = gpd.clip(base_gdf, roi_gdf, keep_geom_type=True)
    if out_vec_file is None:
        return clipped_gdf
    write_vec_gdf(
        clipped_gdf,
        out_vec_file,
        out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def clip_and_merge_with_roi(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_roi_file: VectorInput,
    vec_roi_lyr: Optional[str],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    out_format: str = "GPKG",
    dissolve_lyr: bool = False,
    explode_rslt_lyr: bool = False,
    ref_col_name: str = "ref_bkgrd",
    roi_rgn_val: int = 0,
    data_rgn_val: int = 1,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Clip the polygon vector layer (``vec_file``) using the ROI polygon layer
    (``vec_roi_file``) and merge the clipped features with the remaining
    background ROI area, tagging each polygon in ``ref_col_name``.

    :param vec_file: Input polygon vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_roi_lyr: ROI polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param dissolve_lyr: Dissolve clipped features prior to merging with the
                         background ROI (Default = ``False``).
    :param explode_rslt_lyr: Explode multi-part geometries in the merged result
                             (Default = ``False``).
    :param ref_col_name: Output column tagging ROI vs input polygons
                         (Default = ``'ref_bkgrd'``).
    :param roi_rgn_val: Attribute value for background ROI polygons (Default = ``0``).
    :param data_rgn_val: Attribute value for input data polygons (Default = ``1``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Merged :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.clip_and_merge_with_roi(
            "polys.gpkg", "polys", "roi.gpkg", "roi", "merged.gpkg", "merged"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr)
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    roi_gdf = ensure_matching_crs(data_gdf, roi_gdf)
    roi_gdf[ref_col_name] = roi_rgn_val
    data_gdf[ref_col_name] = data_rgn_val

    clipped_data_gdf = data_gdf.clip(roi_gdf, keep_geom_type=True)

    if dissolve_lyr and not clipped_data_gdf.empty:
        clipped_data_gdf = clipped_data_gdf.dissolve(ref_col_name, as_index=False)
        clipped_data_gdf[ref_col_name] = data_rgn_val

    roi_diff_gdf = gpd.overlay(roi_gdf, clipped_data_gdf, how="difference")
    geom_col = clipped_data_gdf.geometry.name
    if roi_diff_gdf.geometry.name != geom_col:
        roi_diff_gdf = roi_diff_gdf.rename_geometry(geom_col)
    out_crs = data_gdf.crs if data_gdf.crs is not None else roi_gdf.crs
    out_data_gdf = gpd.GeoDataFrame(
        pd.concat([clipped_data_gdf, roi_diff_gdf], ignore_index=True),
        geometry=geom_col,
        crs=out_crs,
    )

    if explode_rslt_lyr:
        out_data_gdf = out_data_gdf.explode(index_parts=False)

    if out_vec_file is None:
        return out_data_gdf

    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)
    write_vec_gdf(
        out_data_gdf,
        out_vec_file,
        out_lyr,
        out_format=out_format,
        out_compress=out_compress,
    )
    return None


def clip_vec_into_sets(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    vec_roi_file: VectorInput,
    vec_roi_lyr: Optional[str],
    roi_att_col: str,
    out_vec_dir: Union[str, os.PathLike],
    out_vec_base_pre: str = "",
    out_vec_base_post: str = "",
    out_vec_ext: str = "gpkg",
    *,
    out_format: str = "GPKG",
    out_name_lower: bool = False,
    use_pbar: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> None:
    """
    Clip a vector dataset into separate output files, one for each unique value
    of ``roi_att_col`` in ``vec_roi_file``. Supports both standard OGR/GDAL
    vector formats and GeoParquet inputs/outputs.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`
                     to clip.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path or
                         :class:`geopandas.GeoDataFrame`.
    :param vec_roi_lyr: ROI vector layer name (or ``None``).
    :param roi_att_col: Column in ``vec_roi_file`` defining each subset.
    :param out_vec_dir: Output directory (created if missing).
    :param out_vec_base_pre: Prefix prepended to output file/layer names.
    :param out_vec_base_post: Suffix appended to output file/layer names.
    :param out_vec_ext: Output file extension without leading dot
                        (Default = ``'gpkg'``).
    :param out_format: Output vector format driver or ``'PARQUET'``
                       (Default = ``'GPKG'``).
    :param out_name_lower: Convert output filenames/layer names to lowercase
                           (Default = ``False``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :param out_compress: Optional GeoParquet compression setting.

    .. code:: python

        import rsgispy.vector.geoms

        rsgispy.vector.geoms.clip_vec_into_sets(
            "pts.gpkg", "pts", "regions.gpkg", "regions", "region_name",
            out_vec_dir="out_sets"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    out_dir = os.fspath(out_vec_dir)
    os.makedirs(out_dir, exist_ok=True)

    data_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr, copy=False)
    roi_gdf = ensure_matching_crs(data_gdf, roi_gdf)

    validate_columns_exist(roi_gdf, [roi_att_col])
    unq_roi_vals = roi_gdf[roi_att_col].unique()

    pre = f"{out_vec_base_pre}_" if out_vec_base_pre != "" else ""
    post = f"_{out_vec_base_post}" if out_vec_base_post != "" else ""

    used_names: set = set()
    pbar_cls = rsgispy.get_tqdm()
    for roi_val in pbar_cls(
        unq_roi_vals, total=len(unq_roi_vals), disable=not use_pbar
    ):
        is_na = pd.isna(roi_val)
        if is_na:
            roi_sub_gdf = roi_gdf[roi_gdf[roi_att_col].isna()]
            val_label = "null"
        else:
            roi_sub_gdf = roi_gdf[roi_gdf[roi_att_col] == roi_val]
            val_label = str(roi_val)

        roi_data_gdf = data_gdf.clip(roi_sub_gdf, keep_geom_type=True)

        out_vec_lyr = f"{pre}{val_label}{post}"
        out_vec_lyr = rsgispy.tools.strings.check_str(
            out_vec_lyr,
            rm_non_ascii=True,
            rm_dashs=False,
            rm_spaces=True,
            rm_punc=True,
        )
        if out_name_lower:
            out_vec_lyr = out_vec_lyr.lower()
        if not out_vec_lyr:
            out_vec_lyr = "group"
        base_lyr = out_vec_lyr
        suffix_idx = 1
        while out_vec_lyr in used_names:
            out_vec_lyr = f"{base_lyr}_{suffix_idx}"
            suffix_idx += 1
        used_names.add(out_vec_lyr)

        out_vec_file = os.path.join(out_dir, f"{out_vec_lyr}.{out_vec_ext}")
        write_vec_gdf(
            roi_data_gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=check_format_name(out_format),
            out_compress=out_compress,
        )
