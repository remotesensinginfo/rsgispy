"""
Global spatial autocorrelation of point attributes for
:mod:`rsgispy.vector.stats`: Moran's I with distance-band or k-nearest
neighbour weights and a permutation test, and a Moran's I correlogram over
distance classes. Neighbour searches use ``shapely.STRtree`` (``scipy`` is
used for k-nearest neighbours when installed, otherwise a chunked brute-force
search), so no PySAL dependency is needed.
"""

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import shapely

import rsgispy
from rsgispy.vectorprocess import VectorInput, read_vec_gdf, validate_columns_exist

try:
    import scipy.spatial  # pylint: disable=import-error

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

__all__ = [
    "calc_morans_i",
    "calc_correlogram",
    "calc_morans_i_arrays",
    "calc_correlogram_arrays",
]

_WEIGHT_TYPES = ("distance_band", "knn")


def _read_xy_vals(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    value_col: str,
    allow_geographic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    gdf = read_vec_gdf(vec_file, vec_lyr, copy=False)
    validate_columns_exist(gdf, [value_col])
    if gdf.crs is not None and gdf.crs.is_geographic and not allow_geographic:
        raise rsgispy.RSGISPyException(
            "The layer has a geographic CRS (degrees); reproject it to a projected "
            "CRS or pass allow_geographic=True."
        )
    geoms = gdf.geometry.values
    if np.any(shapely.is_missing(geoms) | shapely.is_empty(geoms)):
        raise rsgispy.RSGISPyException("Some features have no geometry.")
    pts = np.where(shapely.get_type_id(geoms) == 0, geoms, shapely.centroid(geoms))
    xy = shapely.get_coordinates(pts)[:, :2].astype(np.float64)
    vals = pd.to_numeric(gdf[value_col], errors="coerce").to_numpy(dtype=np.float64)
    good = np.isfinite(vals)
    if not np.all(good):
        warnings.warn(
            f"{int(np.sum(~good))} feature(s) with a missing '{value_col}' value "
            "were ignored.",
            UserWarning,
            stacklevel=3,
        )
    return xy[good], vals[good]


def _check_xy_vals(xy: Any, vals: Any) -> Tuple[np.ndarray, np.ndarray]:
    xy_arr = np.asarray(xy, dtype=np.float64)
    v_arr = np.asarray(vals, dtype=np.float64).ravel()
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2 or xy_arr.shape[0] != v_arr.size:
        raise rsgispy.RSGISPyException(
            f"xy must have shape (n, 2) matching vals; got {xy_arr.shape} and "
            f"{v_arr.size} values."
        )
    if not (np.all(np.isfinite(xy_arr)) and np.all(np.isfinite(v_arr))):
        raise rsgispy.RSGISPyException("xy and vals must be finite.")
    if v_arr.size < 3:
        raise rsgispy.RSGISPyException("At least three points are needed.")
    if np.allclose(v_arr, v_arr[0]):
        raise rsgispy.RSGISPyException("All values are equal; Moran's I is undefined.")
    return xy_arr, v_arr


def _pairs_within(
    xy: np.ndarray, max_dist: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ordered pairs ``(i, j)``, ``i != j``, with distance ``<= max_dist``."""
    pts = shapely.points(xy)
    src, nbr = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=max_dist)
    keep = src != nbr
    src, nbr = src[keep], nbr[keep]
    dist = np.hypot(xy[src, 0] - xy[nbr, 0], xy[src, 1] - xy[nbr, 1])
    keep = dist <= max_dist
    return src[keep], nbr[keep], dist[keep]


def _max_nn_dist(xy: np.ndarray) -> float:
    """Largest nearest-neighbour distance: the smallest band giving every point
    at least one neighbour."""
    pts = shapely.points(xy)
    idx, dist = shapely.STRtree(pts).query_nearest(
        pts, return_distance=True, exclusive=True
    )
    src = idx[0]
    nn_dist = np.full(xy.shape[0], np.inf)
    np.minimum.at(nn_dist, src, dist)
    return float(np.max(nn_dist))


def _knn_pairs(xy: np.ndarray, k_nn: int) -> Tuple[np.ndarray, np.ndarray]:
    n_pts = xy.shape[0]
    if k_nn >= n_pts:
        raise rsgispy.RSGISPyException(f"k must be < the number of points ({n_pts}).")
    if _HAVE_SCIPY:
        _, idx = scipy.spatial.cKDTree(xy).query(xy, k=k_nn + 1)
        # Remove each point from its own list; with duplicate locations the
        # point may not come first, so drop it wherever it is (or the last).
        nbrs = np.array(
            [row[row != pt_i][:k_nn] for pt_i, row in enumerate(idx)], dtype=np.intp
        )
    else:
        nbrs = np.empty((n_pts, k_nn), dtype=np.intp)
        for start in range(0, n_pts, 1024):
            stop = min(start + 1024, n_pts)
            d2 = np.sum((xy[start:stop, None, :] - xy[None, :, :]) ** 2, axis=2)
            d2[np.arange(stop - start), np.arange(start, stop)] = np.inf
            part = np.argpartition(d2, k_nn - 1, axis=1)[:, :k_nn]
            nbrs[start:stop] = part
    src = np.repeat(np.arange(n_pts), k_nn)
    return src, nbrs.ravel().astype(np.intp)


def _morans_i_from_pairs(
    z_vals: np.ndarray, src: np.ndarray, nbr: np.ndarray, wgt: np.ndarray
) -> float:
    n_pts = z_vals.size
    s0 = float(np.sum(wgt))
    if s0 <= 0.0:
        return float("nan")
    lag = np.bincount(src, weights=wgt * z_vals[nbr], minlength=n_pts)
    return float(n_pts / s0 * np.dot(z_vals, lag) / np.dot(z_vals, z_vals))


def _row_standardise(src: np.ndarray, n_pts: int) -> Tuple[np.ndarray, int]:
    counts = np.bincount(src, minlength=n_pts)
    wgt = 1.0 / counts[src]
    return wgt, int(np.sum(counts == 0))


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _moran_test(
    vals: np.ndarray,
    src: np.ndarray,
    nbr: np.ndarray,
    wgt: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    z_vals = vals - vals.mean()
    obs = _morans_i_from_pairs(z_vals, src, nbr, wgt)
    expected = -1.0 / (vals.size - 1)
    out = {
        "I": obs,
        "expected_I": expected,
        "z_score": float("nan"),
        "p_value": float("nan"),
        "n": float(vals.size),
        "n_pairs": float(src.size),
    }
    if n_perms > 0 and np.isfinite(obs):
        perm_i = np.array(
            [
                _morans_i_from_pairs(rng.permutation(z_vals), src, nbr, wgt)
                for _ in range(int(n_perms))
            ]
        )
        if obs >= expected:
            n_extreme = int(np.sum(perm_i >= obs))
        else:
            n_extreme = int(np.sum(perm_i <= obs))
        out["p_value"] = (n_extreme + 1.0) / (n_perms + 1.0)
        perm_sd = float(np.std(perm_i))
        if perm_sd > 0:
            out["z_score"] = (obs - float(np.mean(perm_i))) / perm_sd
    return out


# pylint: disable=too-many-arguments,too-many-locals
def calc_morans_i_arrays(
    xy: Union[np.ndarray, Sequence[Sequence[float]]],
    vals: Union[np.ndarray, Sequence[float]],
    *,
    weights: str = "distance_band",
    dist: Optional[float] = None,
    k: int = 8,
    n_perms: int = 999,
    rnd_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate global Moran's I from coordinate and value arrays; see
    :func:`calc_morans_i` for the method and the returned dictionary.

    :param xy: ``(n, 2)`` point coordinates.
    :param vals: ``n`` values.
    :param weights: ``"distance_band"`` (Default) or ``"knn"``.
    :param dist: Band distance for ``"distance_band"`` (Default: the largest
                 nearest-neighbour distance).
    :param k: Number of neighbours for ``"knn"`` (Default = ``8``).
    :param n_perms: Number of permutations (Default = ``999``; ``0`` for none).
    :param rnd_seed: Optional random seed for the permutations.
    :return: Dictionary of results.
    """
    xy_arr, v_arr = _check_xy_vals(xy, vals)
    w_type = str(weights).strip().lower()
    if w_type not in _WEIGHT_TYPES:
        raise rsgispy.RSGISPyException(
            f"Unsupported weights '{weights}'; use one of {_WEIGHT_TYPES}."
        )
    if int(n_perms) < 0:
        raise rsgispy.RSGISPyException("n_perms must be >= 0.")
    n_pts = v_arr.size
    if w_type == "knn":
        if int(k) < 1:
            raise rsgispy.RSGISPyException(f"k must be >= 1, got {k!r}.")
        src, nbr = _knn_pairs(xy_arr, int(k))
        band = float("nan")
    else:
        band = _max_nn_dist(xy_arr) if dist is None else float(dist)
        if not np.isfinite(band) or band <= 0.0:
            raise rsgispy.RSGISPyException(f"dist must be > 0, got {dist!r}.")
        src, nbr, _ = _pairs_within(xy_arr, band)
    wgt, n_islands = _row_standardise(src, n_pts)
    if n_islands:
        warnings.warn(
            f"{n_islands} point(s) have no neighbours (islands) and contribute "
            "nothing to Moran's I; consider a larger dist.",
            UserWarning,
            stacklevel=2,
        )
    out = _moran_test(
        v_arr, src, nbr, wgt, int(n_perms), np.random.default_rng(rnd_seed)
    )
    out["n_islands"] = float(n_islands)
    out["dist"] = band
    return out


# pylint: disable=too-many-arguments
def calc_morans_i(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    value_col: str,
    *,
    weights: str = "distance_band",
    dist: Optional[float] = None,
    k: int = 8,
    n_perms: int = 999,
    rnd_seed: Optional[int] = None,
    allow_geographic: bool = False,
) -> Dict[str, float]:
    """
    Calculate global Moran's I, a measure of spatial autocorrelation: whether
    nearby features have similar (``I > 0``), dissimilar (``I < 0``) or
    unrelated (``I`` close to ``E[I] = -1 / (n - 1)``) values of ``value_col``.
    Typical uses are checking survey data before modelling, and checking model
    residuals for remaining spatial structure.

    Neighbours are features within ``dist`` of each other
    (``weights="distance_band"``; default ``dist`` is the largest
    nearest-neighbour distance, so every feature has at least one neighbour)
    or each feature's ``k`` nearest features (``weights="knn"``). Weights are
    row-standardised. Non-point geometries are represented by their centroid,
    and features with a missing value are ignored with a warning. Features with
    no neighbours ("islands") are reported with a warning.

    Significance is assessed by permutation: the values are shuffled over the
    locations ``n_perms`` times. ``p_value`` is the one-sided pseudo p-value
    ``(n_extreme + 1) / (n_perms + 1)`` in the direction of the observed ``I``,
    and ``z_score`` is relative to the permutation distribution.

    :param vec_file: Vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Layer name (or ``None``).
    :param value_col: Numeric column to test.
    :param weights: ``"distance_band"`` (Default) or ``"knn"``.
    :param dist: Band distance in CRS units for ``"distance_band"``.
    :param k: Number of neighbours for ``"knn"`` (Default = ``8``).
    :param n_perms: Number of permutations (Default = ``999``; ``0`` for no test).
    :param rnd_seed: Optional random seed for the permutations.
    :param allow_geographic: Allow a geographic CRS, with distances in degrees
                             (Default = ``False``).
    :return: Dictionary with ``I``, ``expected_I``, ``z_score``, ``p_value``,
             ``n``, ``n_pairs`` (number of neighbour pairs), ``n_islands`` and
             ``dist`` (the band distance used; ``NaN`` for ``"knn"``).

    .. code:: python

        import rsgispy.vector.stats

        res = rsgispy.vector.stats.calc_morans_i(
            "plots.gpkg", "plots", "biomass", dist=500.0, rnd_seed=42
        )
        print(res["I"], res["p_value"])
    """
    xy, vals = _read_xy_vals(vec_file, vec_lyr, value_col, allow_geographic)
    return calc_morans_i_arrays(
        xy, vals, weights=weights, dist=dist, k=k, n_perms=n_perms, rnd_seed=rnd_seed
    )


# pylint: disable=too-many-arguments,too-many-locals
def calc_correlogram_arrays(
    xy: Union[np.ndarray, Sequence[Sequence[float]]],
    vals: Union[np.ndarray, Sequence[float]],
    *,
    dist_classes: Optional[Sequence[float]] = None,
    n_classes: int = 10,
    max_dist: Optional[float] = None,
    n_perms: int = 99,
    rnd_seed: Optional[int] = None,
    plot_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate a Moran's I correlogram from coordinate and value arrays; see
    :func:`calc_correlogram` for the method and the returned table.
    """
    xy_arr, v_arr = _check_xy_vals(xy, vals)
    if dist_classes is not None:
        edges = np.asarray(dist_classes, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise rsgispy.RSGISPyException(
                "dist_classes must be at least two increasing distances."
            )
    else:
        if int(n_classes) < 1:
            raise rsgispy.RSGISPyException("n_classes must be >= 1.")
        if max_dist is None:
            span = xy_arr.max(axis=0) - xy_arr.min(axis=0)
            max_dist = 0.5 * float(np.hypot(*span))
        if not np.isfinite(float(max_dist)) or float(max_dist) <= 0.0:
            raise rsgispy.RSGISPyException(f"max_dist must be > 0, got {max_dist!r}.")
        edges = np.linspace(0.0, float(max_dist), int(n_classes) + 1)

    src_all, nbr_all, dist_all = _pairs_within(xy_arr, float(edges[-1]))
    rng = np.random.default_rng(rnd_seed)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_cls = (dist_all > low) & (dist_all <= high)
        src, nbr = src_all[in_cls], nbr_all[in_cls]
        row: Dict[str, float] = {
            "dist_lower": float(low),
            "dist_upper": float(high),
            "dist_mid": float(0.5 * (low + high)),
            "n_pairs": float(src.size),
            "morans_i": float("nan"),
            "expected_i": -1.0 / (v_arr.size - 1),
            "p_value": float("nan"),
        }
        if src.size > 0:
            wgt, _ = _row_standardise(src, v_arr.size)
            res = _moran_test(v_arr, src, nbr, wgt, int(n_perms), rng)
            row["morans_i"] = res["I"]
            row["p_value"] = res["p_value"]
        rows.append(row)
    corr_df = pd.DataFrame(rows)

    if plot_file is not None:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        from rsgispy.tools._ml_explain import (  # pylint: disable=import-outside-toplevel
            finalise_plot,
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(corr_df["expected_i"].iloc[0], color="0.5", linestyle="--")
        sig = corr_df["p_value"] <= 0.05
        ax.plot(corr_df["dist_mid"], corr_df["morans_i"], color="#1f77b4", marker="o",
                markerfacecolor="white")
        ax.plot(corr_df.loc[sig, "dist_mid"], corr_df.loc[sig, "morans_i"], "o",
                color="#1f77b4", label="p <= 0.05")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Moran's I")
        ax.legend(loc="upper right")
        finalise_plot(fig, out_file=plot_file)
    return corr_df


# pylint: disable=too-many-arguments
def calc_correlogram(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    value_col: str,
    *,
    dist_classes: Optional[Sequence[float]] = None,
    n_classes: int = 10,
    max_dist: Optional[float] = None,
    n_perms: int = 99,
    rnd_seed: Optional[int] = None,
    plot_file: Optional[str] = None,
    allow_geographic: bool = False,
) -> pd.DataFrame:
    """
    Calculate a Moran's I correlogram: Moran's I of ``value_col`` for pairs of
    features in successive distance classes (``lower < d <= upper``), with
    row-standardised binary weights within each class and a permutation
    p-value per class (see :func:`calc_morans_i`). The distance at which I
    falls to its expectation is the range of spatial autocorrelation, a guide
    for thinning distances and spatial cross-validation block sizes.

    :param vec_file: Vector path or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Layer name (or ``None``).
    :param value_col: Numeric column to test.
    :param dist_classes: Optional class edges (increasing distances, e.g.
                         ``[0, 500, 1000, 2000]``).
    :param n_classes: Number of equal-width classes when ``dist_classes`` is
                      not given (Default = ``10``).
    :param max_dist: Largest distance when ``dist_classes`` is not given
                     (Default: half the diagonal of the layer's extent).
    :param n_perms: Permutations per class (Default = ``99``; ``0`` for none).
    :param rnd_seed: Optional random seed for the permutations.
    :param plot_file: Optional output path for a correlogram plot (filled
                      markers: ``p <= 0.05``).
    :param allow_geographic: Allow a geographic CRS (Default = ``False``).
    :return: :class:`pandas.DataFrame` with one row per class: ``dist_lower``,
             ``dist_upper``, ``dist_mid``, ``n_pairs``, ``morans_i``,
             ``expected_i`` and ``p_value`` (``NaN`` for classes without pairs).

    .. code:: python

        import rsgispy.vector.stats

        corr_df = rsgispy.vector.stats.calc_correlogram(
            "plots.gpkg", "plots", "biomass", n_classes=8, max_dist=5000.0,
            plot_file="correlogram.png",
        )
    """
    xy, vals = _read_xy_vals(vec_file, vec_lyr, value_col, allow_geographic)
    return calc_correlogram_arrays(
        xy,
        vals,
        dist_classes=dist_classes,
        n_classes=n_classes,
        max_dist=max_dist,
        n_perms=n_perms,
        rnd_seed=rnd_seed,
        plot_file=plot_file,
    )
