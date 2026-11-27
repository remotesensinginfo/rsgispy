"""
Spatial thinning (rarefaction) of point samples for :mod:`rsgispy.vector.geoms`:
minimum-distance thinning (randomised spThin-style or deterministic greedy)
and one-point-per-raster-cell thinning, using ``shapely.STRtree`` and
``rasterio`` (no ``scipy`` dependency).
"""

import os
import warnings
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely

import rsgispy
from rsgispy.vectorprocess import (
    VectorInput,
    read_vec_gdf,
    validate_columns_exist,
    write_vec_gdf,
)

__all__ = [
    "thin_pts_min_dist",
    "thin_pts_by_raster_grid",
]

_THIN_MIN_DIST_METHODS = ("random", "greedy")
_THIN_GRID_KEEP = ("random", "first")


def _get_pts_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Return an ``(n, 2)`` array of point coordinates for ``gdf``; non-point
    geometries are represented by their centroid. Raises for missing or empty
    geometries.
    """
    geoms = gdf.geometry.values
    if len(geoms) == 0:
        return np.empty((0, 2), dtype=np.float64)
    missing = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    if np.any(missing):
        raise rsgispy.RSGISPyException(
            f"{int(np.sum(missing))} feature(s) have a missing or empty geometry; "
            "remove them before thinning."
        )
    is_pt = shapely.get_type_id(geoms) == 0
    pts = np.where(is_pt, geoms, shapely.centroid(geoms))
    return shapely.get_coordinates(pts)[:, :2].astype(np.float64)


def _build_neighbours(xy: np.ndarray, min_dist: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a CSR-style neighbour list of points closer than ``min_dist`` to each
    other (self pairs excluded). Returns ``(indptr, indices)``: the neighbours of
    point ``i`` are ``indices[indptr[i]:indptr[i + 1]]``.
    """
    n_pts = xy.shape[0]
    pts = shapely.points(xy)
    tree = shapely.STRtree(pts)
    src_idx, nbr_idx = tree.query(pts, predicate="dwithin", distance=min_dist)
    keep = src_idx != nbr_idx
    src_idx = src_idx[keep]
    nbr_idx = nbr_idx[keep]
    # 'dwithin' includes pairs exactly min_dist apart; those are allowed.
    pair_dist = np.hypot(
        xy[src_idx, 0] - xy[nbr_idx, 0], xy[src_idx, 1] - xy[nbr_idx, 1]
    )
    keep = pair_dist < min_dist
    src_idx = src_idx[keep]
    nbr_idx = nbr_idx[keep]
    order = np.lexsort((nbr_idx, src_idx))
    src_idx = src_idx[order]
    nbr_idx = nbr_idx[order].astype(np.intp)
    counts = np.bincount(src_idx, minlength=n_pts)
    indptr = np.zeros(n_pts + 1, dtype=np.intp)
    np.cumsum(counts, out=indptr[1:])
    return indptr, nbr_idx


def _thin_random_rep(
    indptr: np.ndarray, indices: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One randomised greedy pass: visit points in random order, accepting a
    point unless an already-accepted point is closer than ``min_dist``."""
    n_pts = indptr.shape[0] - 1
    blocked = np.zeros(n_pts, dtype=bool)
    accepted = np.zeros(n_pts, dtype=bool)
    for pt_idx in rng.permutation(n_pts):
        if blocked[pt_idx]:
            continue
        accepted[pt_idx] = True
        blocked[indices[indptr[pt_idx] : indptr[pt_idx + 1]]] = True
    return np.flatnonzero(accepted)


def _thin_greedy(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Deterministic greedy thinning: repeatedly remove the point with the most
    remaining neighbours (lowest index on ties) until no conflicts remain."""
    n_pts = indptr.shape[0] - 1
    degree = np.diff(indptr).astype(np.int64)
    active = np.ones(n_pts, dtype=bool)
    while n_pts > 0:
        rm_idx = int(np.argmax(degree))
        if degree[rm_idx] <= 0:
            break
        active[rm_idx] = False
        degree[rm_idx] = 0
        nbrs = indices[indptr[rm_idx] : indptr[rm_idx + 1]]
        nbrs = nbrs[active[nbrs]]
        degree[nbrs] -= 1
    return np.flatnonzero(active)


def _finish_thin_output(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf: gpd.GeoDataFrame,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str],
    out_format: str,
    out_compress: Optional[Union[str, bool]],
) -> Optional[gpd.GeoDataFrame]:
    if out_vec_file is None:
        return gdf
    if len(gdf) > 0:
        write_vec_gdf(
            gdf,
            out_vec_file,
            out_vec_lyr,
            out_format=out_format,
            out_compress=out_compress,
        )
    else:
        warnings.warn(
            "No features were retained; no output file was written.",
            UserWarning,
            stacklevel=3,
        )
    return None


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def thin_pts_min_dist(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    min_dist: float,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    n_reps: int = 10,
    method: str = "random",
    rnd_seed: Optional[int] = None,
    allow_geographic: bool = False,
    return_reps: bool = False,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
    use_pbar: bool = False,
) -> Union[
    Optional[gpd.GeoDataFrame],
    Tuple[Optional[gpd.GeoDataFrame], List[np.ndarray]],
]:
    """
    Spatially thin (rarefy) point features so that no two retained points are
    closer than ``min_dist`` to each other. Typically used to reduce the
    clustering of species occurrence records before species distribution
    modelling.

    Methods:

    * ``"random"`` (Default) — the randomised greedy algorithm of ``spThin``
      (Aiello-Lammens et al. 2015): visit the points in a random order and keep
      each point unless an already-kept point is closer than ``min_dist``. This
      is repeated ``n_reps`` times and the replicate keeping the most points is
      returned (the first such replicate on ties).
    * ``"greedy"`` — deterministic: repeatedly remove the point with the most
      neighbours closer than ``min_dist`` (lowest row position on ties) until no
      pair is closer than ``min_dist``. This usually keeps at least as many
      points as one random replicate. ``n_reps`` and ``rnd_seed`` are ignored.

    Points exactly ``min_dist`` apart may both be kept. Non-point geometries are
    represented by their centroid. Distances are Euclidean in the layer's CRS
    units, so a geographic CRS (degrees) is rejected unless
    ``allow_geographic=True``; reproject to a projected CRS first (e.g. with
    :func:`rsgispy.vector.io.reproj_vec_lyr`). All attributes are kept, and the
    retained rows stay in their input order.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param min_dist: Minimum distance between retained points (CRS units, > 0).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param n_reps: Number of random replicates for ``method="random"``
                   (Default = ``10``).
    :param method: ``"random"`` (Default) or ``"greedy"``.
    :param rnd_seed: Optional random seed for reproducibility.
    :param allow_geographic: Allow a geographic CRS, treating ``min_dist`` as
                             degrees (Default = ``False``).
    :param return_reps: Also return the retained row positions (0-based, into
                        the input layer) of every replicate, e.g. to build
                        replicate presence sets for an ensemble
                        (Default = ``False``).
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :param use_pbar: Display a tqdm progress bar over the replicates
                     (Default = ``False``).
    :return: Thinned :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``. With ``return_reps=True``, a tuple
             ``(result, reps)`` where ``reps`` is a list of sorted row-position
             arrays (one per replicate; one array for ``"greedy"``).

    .. code:: python

        import rsgispy.vector.geoms

        thin_gdf = rsgispy.vector.geoms.thin_pts_min_dist(
            "presences.gpkg", "presences", 1000.0, n_reps=20, rnd_seed=42
        )
    """
    if not np.isfinite(min_dist) or float(min_dist) <= 0.0:
        raise rsgispy.RSGISPyException(f"min_dist must be > 0, got {min_dist!r}.")
    mthd = str(method).strip().lower()
    if mthd not in _THIN_MIN_DIST_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported method '{method}'; use one of {_THIN_MIN_DIST_METHODS}."
        )
    if int(n_reps) < 1:
        raise rsgispy.RSGISPyException(f"n_reps must be >= 1, got {n_reps!r}.")

    pts_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    if pts_gdf.crs is not None and pts_gdf.crs.is_geographic and not allow_geographic:
        raise rsgispy.RSGISPyException(
            "The input layer has a geographic CRS (degrees), so min_dist would be "
            "in degrees. Reproject to a projected CRS (e.g. with "
            "rsgispy.vector.io.reproj_vec_lyr) or pass allow_geographic=True."
        )

    xy = _get_pts_xy(pts_gdf)
    n_pts = xy.shape[0]
    reps: List[np.ndarray] = []
    if n_pts == 0:
        reps = [np.empty(0, dtype=np.intp)]
    else:
        indptr, indices = _build_neighbours(xy, float(min_dist))
        if mthd == "greedy":
            reps = [_thin_greedy(indptr, indices)]
        else:
            rng = np.random.default_rng(rnd_seed)
            pbar = rsgispy.get_tqdm()
            for _ in pbar(range(int(n_reps)), disable=not use_pbar):
                reps.append(_thin_random_rep(indptr, indices, rng))

    best_rep = max(range(len(reps)), key=lambda i: (reps[i].size, -i))
    thin_gdf = pts_gdf.iloc[reps[best_rep]].copy()
    result = _finish_thin_output(
        thin_gdf, out_vec_file, out_vec_lyr, out_format, out_compress
    )
    if return_reps:
        return result, reps
    return result


def _assign_grid_cells(
    xy: np.ndarray,
    ref_img: Union[str, os.PathLike],
    pts_crs: Optional[pyproj.CRS],
    cell_mult: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(inside, cell_row, cell_col)`` for each point on the
    ``cell_mult``-aggregated pixel grid of ``ref_img``."""
    with rasterio.open(os.fspath(ref_img)) as src:
        transform = src.transform
        n_rows, n_cols = src.height, src.width
        img_crs = src.crs

    if (
        xy.shape[0] > 0
        and pts_crs is not None
        and img_crs is not None
        and not pyproj.CRS.from_user_input(pts_crs).equals(
            pyproj.CRS.from_user_input(img_crs)
        )
    ):
        pts_img = gpd.GeoSeries(shapely.points(xy), crs=pts_crs).to_crs(img_crs)
        xy = shapely.get_coordinates(pts_img.values)[:, :2]

    inv = ~transform
    cols_f = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
    rows_f = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
    cols = np.floor(np.asarray(cols_f, dtype=np.float64))
    rows = np.floor(np.asarray(rows_f, dtype=np.float64))
    inside = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & (rows >= 0)
        & (rows < n_rows)
        & (cols >= 0)
        & (cols < n_cols)
    )
    rows = np.where(inside, rows, -1).astype(np.int64) // int(cell_mult)
    cols = np.where(inside, cols, -1).astype(np.int64) // int(cell_mult)
    return inside, rows, cols


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def thin_pts_by_raster_grid(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    ref_img: Union[str, os.PathLike],
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    cell_mult: int = 1,
    keep: str = "random",
    rnd_seed: Optional[int] = None,
    prefer_col: Optional[str] = None,
    prefer_max: bool = True,
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Spatially thin (rarefy) point features so that at most one point is kept
    per pixel of ``ref_img`` (or per ``cell_mult`` × ``cell_mult`` block of
    pixels). No grid vector layer is needed: cells are computed from the raster
    geotransform. This is the "one record per pixel" step recommended before
    species distribution modelling, where several records in one pixel of the
    environmental rasters are duplicates as far as the model is concerned.

    Which point is kept in each cell:

    * ``prefer_col`` given — the point with the largest (``prefer_max=True``)
      or smallest value in that column, e.g. the most recent or most precise
      record; missing values rank last and ties are broken by ``keep``.
    * otherwise ``keep="random"`` (Default; seeded by ``rnd_seed``) or
      ``keep="first"`` (first in input order).

    Points falling outside the raster extent are dropped with a warning.
    Non-point geometries are represented by their centroid. Points are
    reprojected to the raster CRS to find their cell, but the output keeps the
    input geometries, attributes and row order.

    :param vec_file: Input vector file path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Input vector layer name (or ``None``).
    :param ref_img: Reference raster defining the pixel grid (e.g. one of the
                    environmental covariate rasters).
    :param out_vec_file: Output vector file path, or ``None`` to return an
                         in-memory :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output vector layer name.
    :param cell_mult: Thinning cell size as a multiple of the pixel size
                      (Default = ``1``, one point per pixel).
    :param keep: ``"random"`` (Default) or ``"first"``.
    :param rnd_seed: Optional random seed used when ``keep="random"``.
    :param prefer_col: Optional column used to choose the point kept per cell.
    :param prefer_max: Keep the largest ``prefer_col`` value when ``True``
                       (Default), otherwise the smallest.
    :param out_format: Output vector format driver (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: Thinned :class:`geopandas.GeoDataFrame` when ``out_vec_file`` is
             ``None``, otherwise ``None``.

    .. code:: python

        import rsgispy.vector.geoms

        thin_gdf = rsgispy.vector.geoms.thin_pts_by_raster_grid(
            "presences.gpkg", "presences", "elevation.tif", rnd_seed=42
        )
    """
    keep_mode = str(keep).strip().lower()
    if keep_mode not in _THIN_GRID_KEEP:
        raise rsgispy.RSGISPyException(
            f"Unsupported keep '{keep}'; use one of {_THIN_GRID_KEEP}."
        )
    if int(cell_mult) != cell_mult or int(cell_mult) < 1:
        raise rsgispy.RSGISPyException(
            f"cell_mult must be an integer >= 1, got {cell_mult!r}."
        )
    if not os.path.exists(os.fspath(ref_img)):
        raise rsgispy.RSGISPyException(f"Reference image '{ref_img}' does not exist.")

    pts_gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    if prefer_col is not None:
        validate_columns_exist(pts_gdf, [prefer_col])

    xy = _get_pts_xy(pts_gdf)
    inside, rows, cols = _assign_grid_cells(xy, ref_img, pts_gdf.crs, int(cell_mult))
    n_outside = int(np.sum(~inside))
    if n_outside > 0:
        warnings.warn(
            f"{n_outside} point(s) fall outside the extent of '{ref_img}' and "
            "were dropped.",
            UserWarning,
            stacklevel=2,
        )

    n_pts = xy.shape[0]
    if keep_mode == "random":
        order = np.random.default_rng(rnd_seed).permutation(n_pts)
    else:
        order = np.arange(n_pts)
    order = order[inside[order]]

    cand_df = pd.DataFrame({"pos": order, "row": rows[order], "col": cols[order]})
    if prefer_col is not None:
        cand_df["val"] = pd.to_numeric(
            pts_gdf[prefer_col].iloc[order], errors="raise"
        ).to_numpy()
        cand_df = cand_df.sort_values(
            "val", ascending=not prefer_max, kind="stable", na_position="last"
        )
    kept_pos = np.sort(
        cand_df.drop_duplicates(subset=["row", "col"], keep="first")["pos"].to_numpy()
    )

    thin_gdf = pts_gdf.iloc[kept_pos].copy()
    return _finish_thin_output(
        thin_gdf, out_vec_file, out_vec_lyr, out_format, out_compress
    )
