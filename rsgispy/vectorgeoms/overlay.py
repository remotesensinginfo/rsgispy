"""
The vectorgeoms.overlay module provides vector layer overlay operations
(intersection, difference, symmetric difference, identity, union) and
region-of-interest clipping functions using ``geopandas``.
"""

import os
from typing import Optional

import geopandas
import pandas

import rsgispy
import rsgispy.tools.strings
from rsgispy.vectorprocess import (
    check_format_name,
    prepare_out_vec_file,
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
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str,
    del_exist_vec: bool,
    how: str,
) -> None:
    """Shared implementation for ``geopandas.overlay`` vector layer operations."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)

    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    over_data_gdf = read_vec_gdf(vec_over_file, vec_over_lyr)
    res_gdf = geopandas.overlay(data_gdf, over_data_gdf, how=how)

    if (res_gdf is not None) and len(res_gdf) > 0:
        write_vec_gdf(res_gdf, out_vec_file, out_lyr, out_format=out_format)


def vec_lyr_intersection_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Perform a geometric intersection between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.vec_lyr_intersection_gp(
            "a.gpkg", "a", "b.gpkg", "b", "inter.gpkg", "inter"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="intersection",
    )


def vec_lyr_difference_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Perform a geometric difference between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path to subtract.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.vec_lyr_difference_gp(
            "a.gpkg", "a", "b.gpkg", "b", "diff.gpkg", "diff"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="difference",
    )


def vec_lyr_sym_difference_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Perform a symmetric difference between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.vec_lyr_sym_difference_gp(
            "a.gpkg", "a", "b.gpkg", "b", "symdiff.gpkg", "symdiff"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="symmetric_difference",
    )


def vec_lyr_identity_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Perform a geometric identity overlay between ``vec_file`` and
    ``vec_over_file`` using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.vec_lyr_identity_gp(
            "a.gpkg", "a", "b.gpkg", "b", "identity.gpkg", "identity"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="identity",
    )


def vec_lyr_union_gp(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_over_file: str,
    vec_over_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    del_exist_vec: bool = False,
) -> None:
    """
    Perform a geometric union between ``vec_file`` and ``vec_over_file``
    using ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_over_file: Overlay vector file path.
    :param vec_over_lyr: Overlay vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.vec_lyr_union_gp(
            "a.gpkg", "a", "b.gpkg", "b", "union.gpkg", "union"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _overlay_vec_lyrs(
        vec_file,
        vec_lyr,
        vec_over_file,
        vec_over_lyr,
        out_vec_file,
        out_vec_lyr,
        out_format,
        del_exist_vec,
        how="union",
    )


def clip_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_roi_file: str,
    vec_roi_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
) -> None:
    """
    Clip a vector layer to the region of interest defined by ``vec_roi_file``
    using ``geopandas.clip``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path.
    :param vec_roi_lyr: ROI polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.clip_vec_lyr(
            "roads.gpkg", "roads", "roi.gpkg", "roi", "clipped.gpkg", "roads"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr)
    clipped_gdf = geopandas.clip(base_gdf, roi_gdf, keep_geom_type=True)
    write_vec_gdf(clipped_gdf, out_vec_file, out_vec_lyr, out_format=out_format)


def clip_and_merge_with_roi(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_roi_file: str,
    vec_roi_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str] = None,
    out_format: str = "GPKG",
    dissolve_lyr: bool = False,
    explode_rslt_lyr: bool = False,
    ref_col_name: str = "ref_bkgrd",
    roi_rgn_val: int = 0,
    data_rgn_val: int = 1,
) -> None:
    """
    Clip the polygon vector layer (``vec_file``) using the ROI polygon layer
    (``vec_roi_file``) and merge the clipped features with the remaining
    background ROI area, tagging each polygon in ``ref_col_name``.

    :param vec_file: Input polygon vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path.
    :param vec_roi_lyr: ROI polygon vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
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

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.clip_and_merge_with_roi(
            "polys.gpkg", "polys", "roi.gpkg", "roi", "merged.gpkg", "merged"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr)
    roi_gdf[ref_col_name] = roi_rgn_val
    data_gdf = read_vec_gdf(vec_file, vec_lyr)
    data_gdf[ref_col_name] = data_rgn_val

    clipped_data_gdf = data_gdf.clip(roi_gdf, keep_geom_type=True)

    if dissolve_lyr and not clipped_data_gdf.empty:
        clipped_data_gdf = clipped_data_gdf.dissolve(ref_col_name, as_index=False)
        clipped_data_gdf[ref_col_name] = data_rgn_val

    roi_diff_gdf = geopandas.overlay(roi_gdf, clipped_data_gdf, how="difference")
    out_data_gdf = geopandas.GeoDataFrame(
        pandas.concat([clipped_data_gdf, roi_diff_gdf], ignore_index=True),
        crs=roi_gdf.crs,
    )

    if explode_rslt_lyr:
        out_data_gdf = out_data_gdf.explode(index_parts=False)

    out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)
    write_vec_gdf(out_data_gdf, out_vec_file, out_lyr, out_format=out_format)


def clip_vec_into_sets(
    vec_file: str,
    vec_lyr: Optional[str],
    vec_roi_file: str,
    vec_roi_lyr: Optional[str],
    roi_att_col: str,
    out_vec_dir: str,
    out_vec_base_pre: str = "",
    out_vec_base_post: str = "",
    out_vec_ext: str = "gpkg",
    out_format: str = "GPKG",
    out_name_lower: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Clip a vector dataset into separate output files, one for each unique value
    of ``roi_att_col`` in ``vec_roi_file``. Supports both standard OGR/GDAL
    vector formats and GeoParquet inputs/outputs.

    :param vec_file: Input vector file path to clip.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param vec_roi_file: ROI polygon vector file path.
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

    .. code:: python

        import rsgispy.vectorgeoms.overlay

        rsgispy.vectorgeoms.overlay.clip_vec_into_sets(
            "pts.gpkg", "pts", "regions.gpkg", "regions", "region_name",
            out_vec_dir="out_sets"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    os.makedirs(out_vec_dir, exist_ok=True)

    if "parquet" in os.path.basename(vec_file).lower():
        data_gdf = geopandas.read_parquet(vec_file)
    else:
        data_gdf = read_vec_gdf(vec_file, vec_lyr)

    if "parquet" in os.path.basename(vec_roi_file).lower():
        roi_gdf = geopandas.read_parquet(vec_roi_file)
    else:
        roi_gdf = read_vec_gdf(vec_roi_file, vec_roi_lyr)

    validate_columns_exist(roi_gdf, [roi_att_col])
    unq_roi_vals = roi_gdf[roi_att_col].unique()

    pre = f"{out_vec_base_pre}_" if out_vec_base_pre != "" else ""
    post = f"_{out_vec_base_post}" if out_vec_base_post != "" else ""

    pbar_cls = rsgispy.get_tqdm()
    for roi_val in pbar_cls(
        unq_roi_vals, total=len(unq_roi_vals), disable=not use_pbar
    ):
        roi_sub_gdf = roi_gdf[roi_gdf[roi_att_col] == roi_val]
        roi_data_gdf = data_gdf.clip(roi_sub_gdf, keep_geom_type=True)

        out_vec_lyr = f"{pre}{roi_val}{post}"
        out_vec_lyr = rsgispy.tools.strings.check_str(
            out_vec_lyr,
            rm_non_ascii=True,
            rm_dashs=False,
            rm_spaces=True,
            rm_punc=True,
        )
        if out_name_lower:
            out_vec_lyr = out_vec_lyr.lower()

        out_vec_file = os.path.join(out_vec_dir, f"{out_vec_lyr}.{out_vec_ext}")

        if out_format.upper() == "PARQUET":
            out_compress = None
            if "gzip" in out_vec_ext:
                out_compress = "gzip"
            elif "sz" in out_vec_ext:
                out_compress = "snappy"
            roi_data_gdf.to_parquet(out_vec_file, compression=out_compress)
        else:
            write_vec_gdf(
                roi_data_gdf,
                out_vec_file,
                out_vec_lyr,
                out_format=check_format_name(out_format),
            )
