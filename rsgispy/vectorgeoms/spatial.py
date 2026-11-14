"""
The vectorgeoms.spatial module provides geometry point extraction, alpha-shape
hulls, bounding-box queries, R-tree spatial indexing, and layer-to-layer
spatial predicate checks using ``geopandas``, ``shapely``, and ``rtree``.
"""

from typing import List, Optional, Sequence, Tuple, Union

import geopandas
import pyogrio
import shapely.geometry
import shapely.ops
from shapely.geometry.base import BaseGeometry

import rsgispy
from rsgispy.vectorprocess import (
    extract_geom_pts,
    prepare_out_vec_file,
    read_vec_gdf,
    resolve_out_lyr_name,
    write_vec_gdf,
)

__all__ = [
    "get_geom_pts",
    "get_vec_lyr_as_pts",
    "create_alpha_shape",
    "get_geoms_as_bboxs",
    "bbox_intersects_vec_lyr",
    "create_rtree_index",
    "bbox_intersects_index",
    "vec_intersects_vec",
    "vec_overlaps_vec",
    "vec_within_vec",
    "vec_contains_vec",
    "vec_touches_vec",
    "vec_crosses_vec",
]


def get_geom_pts(
    geom: Optional[BaseGeometry],
    pts_lst: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """
    Recursively extract all vertex coordinates ``(x, y)`` from a Shapely
    geometry.

    :param geom: Input Shapely geometry from which points are extracted.
    :param pts_lst: Optional destination list to append extracted points into;
                    a new list is created if ``None``.
    :return: List of ``(x, y)`` coordinate tuples.

    .. code:: python

        from shapely.geometry import box
        import rsgispy.vectorgeoms.spatial

        pts = rsgispy.vectorgeoms.spatial.get_geom_pts(box(0.0, 0.0, 10.0, 10.0))
    """
    return extract_geom_pts(geom, pts_lst)


def get_vec_lyr_as_pts(
    vec_file: str,
    vec_lyr: Optional[str],
    use_pbar: bool = True,
) -> List[Tuple[float, float]]:
    """
    Extract all vertex coordinates from every feature in a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: List of ``(x, y)`` coordinate tuples.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        pts = rsgispy.vectorgeoms.spatial.get_vec_lyr_as_pts("polys.gpkg", "polys")
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
    pts_lst: List[Tuple[float, float]] = []
    pbar_cls = rsgispy.get_tqdm()
    for geom in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        if geom is not None and not geom.is_empty:
            extract_geom_pts(geom, pts_lst)
    return pts_lst


def create_alpha_shape(
    vec_file: str,
    vec_lyr: Optional[str],
    out_vec_file: str,
    out_vec_lyr: Optional[str],
    out_format: str = "GPKG",
    alpha_val: Optional[float] = None,
    alpha_vals: Optional[Sequence[float]] = None,
    max_iter: int = 10000,
    del_exist_vec: bool = False,
) -> Tuple[bool, Optional[float]]:
    """
    Calculate a concave hull (alpha shape) polygon around all vertices of the
    input vector layer using the ``alphashape`` package and write the result
    via ``geopandas``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param out_vec_file: Output vector file path.
    :param out_vec_lyr: Output vector layer name.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param alpha_val: Alpha parameter controlling hull tightness. If ``None``
                      and ``alpha_vals`` is ``None``, it is estimated
                      automatically using ``alphashape.optimizealpha``.
    :param alpha_vals: Optional sequence of candidate alpha values tested in
                       order; the first producing a valid ``Polygon`` or
                       ``MultiPolygon`` is used.
    :param max_iter: Maximum iterations when estimating alpha automatically
                     (Default = ``10000``).
    :param del_exist_vec: If ``True``, overwrite existing ``out_vec_file``
                          (Default = ``False``).
    :return: Tuple ``(vec_output, alpha_val)`` where ``vec_output`` indicates
             whether an output polygon was written.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        created, used_alpha = rsgispy.vectorgeoms.spatial.create_alpha_shape(
            "pts.gpkg", "pts", "hull.gpkg", "hull", alpha_vals=[10.0, 5.0, 2.0]
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    try:
        import alphashape  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "The 'alphashape' package is required for create_alpha_shape."
        ) from err

    prepare_out_vec_file(out_vec_file, del_exist_vec=del_exist_vec)
    pts = get_vec_lyr_as_pts(vec_file, vec_lyr, use_pbar=False)
    if len(pts) < 3:
        raise rsgispy.RSGISPyException(
            "At least three points are required to compute an alpha shape."
        )

    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ran_x = max_x - min_x
    ran_y = max_y - min_y
    if ran_x == 0 or ran_y == 0:
        raise rsgispy.RSGISPyException(
            "Input points have zero extent along X or Y; cannot normalise "
            "coordinates for alpha shape."
        )

    norm_pts = [((pt[0] - min_x) / ran_x, (pt[1] - min_y) / ran_y) for pt in pts]

    alpha_shape_geom = None
    if alpha_vals is not None:
        for alpha_test_val in alpha_vals:
            cand_shape = alphashape.alphashape(norm_pts, alpha=alpha_test_val)
            if cand_shape.geom_type in ("MultiPolygon", "Polygon"):
                alpha_val = alpha_test_val
                alpha_shape_geom = cand_shape
                break
    else:
        if alpha_val is None:
            alpha_val = alphashape.optimizealpha(norm_pts, max_iterations=max_iter)
        alpha_shape_geom = alphashape.alphashape(norm_pts, alpha=alpha_val)

    vec_output = False
    if (
        alpha_shape_geom is not None
        and not alpha_shape_geom.is_empty
        and alpha_shape_geom.geom_type in ("Polygon", "MultiPolygon")
    ):
        out_alpha_shape = shapely.ops.transform(
            lambda x, y, z=None: ((x * ran_x) + min_x, (y * ran_y) + min_y),
            alpha_shape_geom,
        )
        info_kw = {"layer": vec_lyr} if vec_lyr is not None else {}
        crs_info = pyogrio.read_info(vec_file, **info_kw).get("crs")
        out_gdf = geopandas.GeoDataFrame(
            {"geometry": [out_alpha_shape]}, crs=crs_info
        )
        out_lyr = resolve_out_lyr_name(out_vec_file, out_vec_lyr)
        write_vec_gdf(out_gdf, out_vec_file, out_lyr, out_format=out_format)
        vec_output = True

    return vec_output, alpha_val


def get_geoms_as_bboxs(
    vec_file: str,
    vec_lyr: Optional[str],
) -> List[Tuple[float, float, float, float]]:
    """
    Return the bounding box ``(minx, maxx, miny, maxy)`` of each feature in a
    vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :return: List of ``(minx, maxx, miny, maxy)`` bounding-box tuples.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        bboxes = rsgispy.vectorgeoms.spatial.get_geoms_as_bboxs("polys.gpkg", "polys")
    """
    base_gdf = read_vec_gdf(vec_file, vec_lyr)
    geom_bounds = base_gdf.geometry.bounds
    return [
        (float(x0), float(x1), float(y0), float(y1))
        for x0, x1, y0, y1 in zip(
            geom_bounds["minx"],
            geom_bounds["maxx"],
            geom_bounds["miny"],
            geom_bounds["maxy"],
        )
    ]


def bbox_intersects_vec_lyr(
    vec_file: str,
    vec_lyr: Optional[str],
    bbox: Union[Tuple[float, float, float, float], Sequence[float]],
) -> bool:
    """
    Test whether any feature in a vector layer intersects ``bbox``.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param bbox: Bounding box ``(xMin, xMax, yMin, yMax)`` in the layer's CRS.
    :return: ``True`` if at least one geometry intersects ``bbox``.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        hits = rsgispy.vectorgeoms.spatial.bbox_intersects_vec_lyr(
            "polys.gpkg", "polys", (0.0, 20.0, 0.0, 20.0)
        )
    """
    in_gdf = read_vec_gdf(vec_file, vec_lyr)
    geoms = in_gdf.geometry[in_gdf.geometry.notna()]
    if len(geoms) == 0:
        return False
    poly_bbox = shapely.geometry.box(bbox[0], bbox[2], bbox[1], bbox[3])
    cand_idxs = geoms.sindex.query(poly_bbox, predicate="intersects")
    return len(cand_idxs) > 0


def create_rtree_index(
    vec_file: str,
    vec_lyr: Optional[str],
    use_pbar: bool = True,
):
    """
    Build an ``rtree`` spatial index and corresponding Shapely geometry list for
    a vector layer.

    :param vec_file: Input vector file path.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: Tuple ``(idx_obj, geom_lst)`` where ``idx_obj`` is an
             ``rtree.index.Index`` and ``geom_lst`` contains Shapely geometries.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        rt_idx, geoms = rsgispy.vectorgeoms.spatial.create_rtree_index(
            "polys.gpkg", "polys"
        )
    """
    try:
        import rtree  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise rsgispy.RSGISPyException(
            "The 'rtree' package is required for create_rtree_index."
        ) from err

    in_gdf = read_vec_gdf(vec_file, vec_lyr)
    idx_obj = rtree.index.Index(interleaved=False)
    geom_lst: List[BaseGeometry] = []
    pbar_cls = rsgispy.get_tqdm()

    for geom_obj in pbar_cls(
        in_gdf.geometry, total=len(in_gdf), disable=not use_pbar
    ):
        if geom_obj is not None and not geom_obj.is_empty:
            xmin, ymin, xmax, ymax = geom_obj.bounds
            idx_obj.insert(len(geom_lst), (xmin, xmax, ymin, ymax))
            geom_lst.append(geom_obj)

    return idx_obj, geom_lst


def bbox_intersects_index(
    rt_idx,
    geom_lst: Sequence[BaseGeometry],
    bbox: Union[Tuple[float, float, float, float], Sequence[float]],
) -> bool:
    """
    Test whether ``bbox`` intersects any geometry referenced by ``rt_idx`` and
    ``geom_lst`` (created via :func:`create_rtree_index`).

    :param rt_idx: ``rtree.index.Index`` built with ``interleaved=False``.
    :param geom_lst: Sequence of Shapely geometries indexed by ``rt_idx``.
    :param bbox: Bounding box ``(xMin, xMax, yMin, yMax)``.
    :return: ``True`` if at least one geometry intersects ``bbox``.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        rt_idx, geoms = rsgispy.vectorgeoms.spatial.create_rtree_index(
            "polys.gpkg", "polys"
        )
        hits = rsgispy.vectorgeoms.spatial.bbox_intersects_index(
            rt_idx, geoms, (0.0, 20.0, 0.0, 20.0)
        )
    """
    poly_bbox = shapely.geometry.box(bbox[0], bbox[2], bbox[1], bbox[3])
    for geom_idx in rt_idx.intersection(bbox):
        if poly_bbox.intersects(geom_lst[geom_idx]):
            return True
    return False


def _check_vec_spatial_predicate(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    predicate: str,
    require_all_comp: bool = False,
    use_pbar: bool = True,
) -> bool:
    """
    Evaluate a spatial predicate between two vector layers using ``geopandas``
    spatial indexing.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    base_gdf = read_vec_gdf(vec_base_file, vec_base_lyr)
    comp_gdf = read_vec_gdf(vec_comp_file, vec_comp_lyr)

    base_geoms = base_gdf.geometry[base_gdf.geometry.notna()]
    comp_geoms = comp_gdf.geometry[comp_gdf.geometry.notna()]

    if len(base_geoms) == 0 or len(comp_geoms) == 0:
        return False

    base_sindex = base_geoms.sindex
    base_arr = base_geoms.to_numpy()
    pbar_cls = rsgispy.get_tqdm()
    iterator = pbar_cls(
        comp_geoms,
        total=len(comp_geoms),
        disable=not use_pbar,
    )

    if require_all_comp:
        for comp_geom in iterator:
            cand_idxs = base_sindex.query(comp_geom, predicate="intersects")
            matched = False
            for idx in cand_idxs:
                base_geom = base_arr[idx]
                if predicate == "contains":
                    ok = base_geom.contains(comp_geom)
                elif predicate == "within":
                    ok = comp_geom.within(base_geom)
                else:
                    ok = getattr(comp_geom, predicate)(base_geom)
                if ok:
                    matched = True
                    break
            if not matched:
                return False
        return True

    for comp_geom in iterator:
        cand_idxs = base_sindex.query(comp_geom, predicate="intersects")
        for idx in cand_idxs:
            if getattr(comp_geom, predicate)(base_arr[idx]):
                return True
    return False


def vec_intersects_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether any geometry in ``vec_comp_file`` intersects any geometry in
    ``vec_base_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if an intersection exists.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_intersects_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="intersects",
        require_all_comp=False,
        use_pbar=use_pbar,
    )


def vec_overlaps_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether any geometry in ``vec_comp_file`` overlaps any geometry in
    ``vec_base_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if an overlap exists.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_overlaps_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="overlaps",
        require_all_comp=False,
        use_pbar=use_pbar,
    )


def vec_within_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether every geometry in ``vec_comp_file`` lies within at least one
    geometry in ``vec_base_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if all comparison geometries are within base geometries.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_within_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="within",
        require_all_comp=True,
        use_pbar=use_pbar,
    )


def vec_contains_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether ``vec_base_file`` contains every geometry in ``vec_comp_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if every comparison geometry is contained by a base
             geometry.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_contains_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="contains",
        require_all_comp=True,
        use_pbar=use_pbar,
    )


def vec_touches_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether any geometry in ``vec_comp_file`` touches any geometry in
    ``vec_base_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if a touching pair exists.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_touches_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="touches",
        require_all_comp=False,
        use_pbar=use_pbar,
    )


def vec_crosses_vec(
    vec_base_file: str,
    vec_base_lyr: Optional[str],
    vec_comp_file: str,
    vec_comp_lyr: Optional[str],
    use_pbar: bool = True,
) -> bool:
    """
    Test whether any geometry in ``vec_comp_file`` crosses any geometry in
    ``vec_base_file``.

    :param vec_base_file: Base vector file path.
    :param vec_base_lyr: Base vector layer name (or ``None``).
    :param vec_comp_file: Comparison vector file path.
    :param vec_comp_lyr: Comparison vector layer name (or ``None``).
    :param use_pbar: Show progress bar (Default = ``True``).
    :return: ``True`` if a crossing pair exists.

    .. code:: python

        import rsgispy.vectorgeoms.spatial

        res = rsgispy.vectorgeoms.spatial.vec_crosses_vec(
            "base.gpkg", "base", "comp.gpkg", "comp"
        )
    """
    return _check_vec_spatial_predicate(
        vec_base_file,
        vec_base_lyr,
        vec_comp_file,
        vec_comp_lyr,
        predicate="crosses",
        require_all_comp=False,
        use_pbar=use_pbar,
    )
