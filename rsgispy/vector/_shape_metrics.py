"""
Vectorised polygon shape / geometry metrics for :mod:`rsgispy.vector.stats`
using ``shapely`` 2 array functions.
"""

import math
import os
from typing import Any, Dict, Optional, Sequence, Union
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import rsgispy
from rsgispy.vectorprocess import VectorInput, finish_vec_output, read_vec_gdf

__all__ = [
    "SHAPE_METRICS",
    "calc_shape_metrics_geoseries",
    "calc_poly_shape_metrics",
]

SHAPE_METRICS = (
    "area",
    "perimeter",
    "n_parts",
    "n_holes",
    "hole_area",
    "n_vertices",
    "compactness",
    "shape_index",
    "fractal_dim",
    "convexity",
    "solidity",
    "rectangularity",
    "mrr_length",
    "mrr_width",
    "elongation",
    "orientation",
    "roundness",
    "equiv_diameter",
)
"""Names of the polygon shape metrics supported by :func:`calc_shape_metrics_geoseries`."""

_POLY_TYPE_IDS = (3, 6)  # Polygon, MultiPolygon


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise ``num / den`` returning NaN where ``den`` is 0 or not finite."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(den) & (den != 0.0) & np.isfinite(num)
    np.divide(num, den, out=out, where=ok)
    return out


def _part_sums(
    geoms: np.ndarray, func: Any, n_geoms: int
) -> np.ndarray:
    """Sum ``func(part)`` over the polygon parts of each (Multi)Polygon."""
    parts, idx = shapely.get_parts(geoms, return_index=True)
    if parts.size == 0:
        return np.zeros(n_geoms, dtype=np.float64)
    vals = np.asarray(func(parts), dtype=np.float64)
    return np.bincount(idx, weights=vals, minlength=n_geoms)


def _mrr_dims(mrr: np.ndarray) -> Dict[str, np.ndarray]:
    """Length, width and long-axis orientation of minimum rotated rectangles."""
    # pylint: disable=too-many-locals
    n_geoms = mrr.size
    length = np.full(n_geoms, np.nan, dtype=np.float64)
    width = np.full(n_geoms, np.nan, dtype=np.float64)
    orient = np.full(n_geoms, np.nan, dtype=np.float64)
    is_poly = shapely.get_type_id(mrr) == 3
    is_poly &= ~shapely.is_empty(mrr)
    if not np.any(is_poly):
        return {"length": length, "width": width, "orientation": orient}
    rings = shapely.get_exterior_ring(mrr[is_poly])
    coords, cidx = shapely.get_coordinates(rings, return_index=True)
    counts = np.bincount(cidx, minlength=rings.size)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    good = counts >= 4
    p0 = coords[starts[good]]
    p1 = coords[starts[good] + 1]
    p2 = coords[starts[good] + 2]
    e1 = p1 - p0
    e2 = p2 - p1
    l1 = np.hypot(e1[:, 0], e1[:, 1])
    l2 = np.hypot(e2[:, 0], e2[:, 1])
    long_is_1 = l1 >= l2
    long_vec = np.where(long_is_1[:, None], e1, e2)
    ang = np.degrees(np.arctan2(long_vec[:, 1], long_vec[:, 0])) % 180.0
    ang = np.where(np.isclose(ang, 180.0), 0.0, ang)
    tgt = np.flatnonzero(is_poly)[good]
    length[tgt] = np.maximum(l1, l2)
    width[tgt] = np.minimum(l1, l2)
    orient[tgt] = ang
    return {"length": length, "width": width, "orientation": orient}


def calc_shape_metrics_geoseries(
    geoms: Union[gpd.GeoSeries, Sequence[Any], np.ndarray],
    metrics: Optional[Sequence[str]] = None,
    col_prefix: str = "",
    make_valid: bool = False,
) -> pd.DataFrame:
    """
    Calculate polygon shape metrics for an array of geometries in a single
    vectorised pass (no per-feature Python loops), returning one row per
    geometry. Lengths and areas are in the units of the geometries' CRS, so
    use a projected CRS (see :func:`calc_poly_shape_metrics`).

    Metrics (``A`` = area, ``P`` = perimeter including hole boundaries, ``CH``
    = convex hull, ``MRR`` = minimum rotated (oriented) rectangle, ``MBC`` =
    minimum bounding circle):

    * ``area`` — ``A``.
    * ``perimeter`` — ``P``.
    * ``n_parts`` — number of polygon parts.
    * ``n_holes`` — number of interior rings (holes) over all parts.
    * ``hole_area`` — total area of the holes.
    * ``n_vertices`` — number of coordinates (ring closing vertices included).
    * ``compactness`` — Polsby–Popper ``4πA / P²`` in ``(0, 1]`` (1 = circle).
    * ``shape_index`` — ``P / (2√(πA))``, ``≥ 1`` (1 = circle).
    * ``fractal_dim`` — ``2 ln(P / 4) / ln(A)``, approximately ``[1, 2]``
      (1 for a square, slightly below 1 for a circle, towards 2 for highly
      convoluted boundaries); NaN when ``A ≤ 1`` (in CRS units).
    * ``convexity`` — ``P_CH / P`` in ``(0, 1]`` for single-part polygons
      (1 = convex); can exceed 1 for widely separated multi-part polygons.
    * ``solidity`` — ``A / A_CH`` in ``(0, 1]`` (1 = convex).
    * ``rectangularity`` — ``A / A_MRR`` in ``(0, 1]`` (1 = rectangle).
    * ``mrr_length``, ``mrr_width`` — long and short side of the MRR.
    * ``elongation`` — ``mrr_length / mrr_width``, ``≥ 1`` (1 = square/circle).
    * ``orientation`` — direction of the MRR long axis in degrees
      anticlockwise from the x axis (east), in ``[0, 180)`` (arbitrary for
      shapes with ``elongation`` close to 1).
    * ``roundness`` — ``A / (π r²)`` where ``r`` is the MBC radius, in
      ``(0, 1]`` (1 = circle).
    * ``equiv_diameter`` — diameter of the circle with the same area,
      ``2√(A / π)``.

    Missing, empty, invalid (unless ``make_valid=True``) and non-polygonal
    geometries give NaN for every metric; ratios with a zero denominator give
    NaN.

    :param geoms: :class:`geopandas.GeoSeries` or array of shapely geometries.
    :param metrics: Metrics to compute (Default = all of :data:`SHAPE_METRICS`).
    :param col_prefix: Prefix added to the output column names (Default = ``''``).
    :param make_valid: If ``True``, repair invalid geometries with
                       :func:`shapely.make_valid` before computing metrics
                       (Default = ``False``).
    :return: :class:`pandas.DataFrame` (index matching ``geoms`` when it is a
             GeoSeries) with one column per metric.

    .. code:: python

        import geopandas as gpd
        import rsgispy.vector.stats

        gdf = gpd.read_file("segments.gpkg")
        shape_df = rsgispy.vector.stats.calc_shape_metrics_geoseries(
            gdf.geometry, metrics=["area", "compactness", "elongation"]
        )
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    sel = list(SHAPE_METRICS) if metrics is None else [str(m) for m in metrics]
    bad = [m for m in sel if m not in SHAPE_METRICS]
    if bad:
        raise rsgispy.RSGISPyException(
            f"Unknown shape metric(s) {bad}; supported: {list(SHAPE_METRICS)}."
        )
    index = geoms.index if isinstance(geoms, (gpd.GeoSeries, pd.Series)) else None
    garr = np.asarray(
        geoms.values if isinstance(geoms, (gpd.GeoSeries, pd.Series)) else geoms,
        dtype=object,
    )
    n_geoms = garr.size
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if make_valid:
            garr = shapely.make_valid(garr)
            # make_valid may return collections; keep only the polygonal parts.
            is_coll = shapely.get_type_id(garr) == 7
            if np.any(is_coll):
                coll_idx = np.flatnonzero(is_coll)
                parts, pidx = shapely.get_parts(garr[coll_idx], return_index=True)
                keep = np.isin(shapely.get_type_id(parts), _POLY_TYPE_IDS)
                for loc, g_i in enumerate(coll_idx):
                    sel_parts = parts[keep & (pidx == loc)]
                    garr[g_i] = (
                        shapely.union_all(sel_parts)
                        if sel_parts.size
                        else shapely.Polygon()
                    )
        type_ids = shapely.get_type_id(garr)
        usable = np.isin(type_ids, _POLY_TYPE_IDS)
        usable &= ~shapely.is_empty(garr)
        usable &= shapely.is_valid(garr)
        g_use = np.where(usable, garr, None)

        cols: Dict[str, np.ndarray] = {}
        area = shapely.area(g_use)
        perim = shapely.length(g_use)
        need = set(sel)
        need_hull = need & {"convexity", "solidity"}
        need_mrr = need & {
            "rectangularity",
            "mrr_length",
            "mrr_width",
            "elongation",
            "orientation",
        }
        hull = shapely.convex_hull(g_use) if need_hull else None
        mrr = shapely.oriented_envelope(g_use) if need_mrr else None
        mrr_dims = _mrr_dims(mrr) if mrr is not None else None

        for metric in sel:
            if metric == "area":
                vals = area
            elif metric == "perimeter":
                vals = perim
            elif metric == "n_parts":
                vals = shapely.get_num_geometries(g_use).astype(np.float64)
            elif metric == "n_holes":
                vals = _part_sums(g_use, shapely.get_num_interior_rings, n_geoms)
            elif metric == "hole_area":
                filled = _part_sums(
                    g_use,
                    lambda p: shapely.area(
                        shapely.polygons(shapely.get_exterior_ring(p))
                    ),
                    n_geoms,
                )
                vals = np.maximum(filled - np.nan_to_num(area), 0.0)
            elif metric == "n_vertices":
                vals = shapely.get_num_coordinates(g_use).astype(np.float64)
            elif metric == "compactness":
                vals = _safe_div(4.0 * math.pi * area, perim * perim)
            elif metric == "shape_index":
                vals = _safe_div(perim, 2.0 * np.sqrt(math.pi * area))
            elif metric == "fractal_dim":
                ok_a = np.nan_to_num(area) > 1.0
                ok_p = np.nan_to_num(perim) > 0.0
                log_a = np.full(n_geoms, np.nan, dtype=np.float64)
                log_p = np.full(n_geoms, np.nan, dtype=np.float64)
                log_a[ok_a] = np.log(area[ok_a])
                log_p[ok_p] = np.log(perim[ok_p] / 4.0)
                vals = _safe_div(2.0 * log_p, log_a)
            elif metric == "convexity":
                vals = _safe_div(shapely.length(hull), perim)
            elif metric == "solidity":
                vals = _safe_div(area, shapely.area(hull))
            elif metric == "rectangularity":
                vals = _safe_div(area, shapely.area(mrr))
            elif metric == "mrr_length":
                vals = mrr_dims["length"]  # type: ignore[index]
            elif metric == "mrr_width":
                vals = mrr_dims["width"]  # type: ignore[index]
            elif metric == "elongation":
                vals = _safe_div(
                    mrr_dims["length"], mrr_dims["width"]  # type: ignore[index]
                )
            elif metric == "orientation":
                vals = mrr_dims["orientation"]  # type: ignore[index]
            elif metric == "roundness":
                rad = shapely.minimum_bounding_radius(g_use)
                vals = _safe_div(area, math.pi * rad * rad)
            else:  # equiv_diameter
                vals = 2.0 * np.sqrt(np.maximum(area, 0.0) / math.pi)
            vals = np.asarray(vals, dtype=np.float64).copy()
            vals[~usable] = np.nan
            cols[f"{col_prefix}{metric}"] = vals

    return pd.DataFrame(cols, index=index)


def calc_poly_shape_metrics(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    metrics: Optional[Sequence[str]] = None,
    col_prefix: str = "",
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    proj_epsg: Optional[Union[int, str]] = None,
    allow_geographic: bool = False,
    make_valid: bool = False,
    *,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
    overwrite: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calculate polygon shape metrics (see :func:`calc_shape_metrics_geoseries`
    for the list, formulas and ranges) for every feature of a polygon layer
    and add them as columns.

    Areas and lengths are in the units of the layer's CRS. For a layer in a
    geographic (latitude/longitude) CRS this raises an exception unless
    ``proj_epsg`` is given (the metrics are then calculated on a reprojected
    copy; the output geometry is unchanged) or ``allow_geographic=True``
    (metrics in degrees, which is rarely meaningful).

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param metrics: Metrics to compute (Default = all).
    :param col_prefix: Prefix for the output column names (Default = ``''``).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param proj_epsg: Optional EPSG code of a projected CRS to calculate the
                      metrics in, or ``'utm'`` to use the layer's estimated
                      UTM zone.
    :param allow_geographic: If ``True``, allow calculation in a geographic
                             CRS (Default = ``False``).
    :param make_valid: If ``True``, repair invalid geometries before
                       calculating metrics (Default = ``False``; invalid
                       geometries give NaN).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :param overwrite: If ``True``, overwrite an existing output file.
    :return: :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.stats

        rsgispy.vector.stats.calc_poly_shape_metrics(
            "segments.gpkg",
            "segments",
            metrics=["area", "compactness", "solidity", "elongation"],
            out_vec_file="segments_shape.gpkg",
            out_vec_lyr="segments",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    gdf = read_vec_gdf(vec_file, vec_lyr)
    geoms = gdf.geometry
    if proj_epsg is not None:
        if gdf.crs is None:
            raise rsgispy.RSGISPyException(
                "The vector layer has no CRS, so it cannot be reprojected for "
                "the shape metrics."
            )
        tgt_crs: Any = (
            gdf.estimate_utm_crs()
            if str(proj_epsg).strip().lower() == "utm"
            else f"EPSG:{int(proj_epsg)}"
        )
        geoms = geoms.to_crs(tgt_crs)
        if geoms.crs is not None and geoms.crs.is_geographic:
            raise rsgispy.RSGISPyException(
                f"proj_epsg ({proj_epsg}) is a geographic CRS; a projected CRS "
                "is required."
            )
    elif gdf.crs is not None and gdf.crs.is_geographic and not allow_geographic:
        raise rsgispy.RSGISPyException(
            "The vector layer is in a geographic CRS (degrees); provide "
            "proj_epsg (an EPSG code or 'utm') to calculate the shape metrics "
            "in a projected CRS, or set allow_geographic=True."
        )
    shape_df = calc_shape_metrics_geoseries(
        geoms, metrics=metrics, col_prefix=col_prefix, make_valid=make_valid
    )
    clash = [c for c in shape_df.columns if c in gdf.columns]
    if clash:
        raise rsgispy.RSGISPyException(
            f"Output column(s) {clash} already exist; use col_prefix."
        )
    for col in shape_df.columns:
        gdf[col] = shape_df[col].to_numpy()
    return finish_vec_output(
        gdf,
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )
