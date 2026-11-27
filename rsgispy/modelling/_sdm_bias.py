"""
Sampling-bias grids for :mod:`rsgispy.modelling.species_distribution`:
building a bias (sampling-effort) raster from records or features, drawing
background / pseudo-absence points with probability proportional to a bias
raster, and inverse-bias sample weights.
"""

import os
import tempfile
from contextlib import ExitStack
from typing import Iterator, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy.imagecalc.distance import calc_dist_to_img_vals
from rsgispy.imageprocess import (
    IntersectionGrid,
    get_default_block_size,
    open_output_dataset,
)
from rsgispy.imageutils.rasterise import rasterise_vec_lyr
from rsgispy.imageutils.stats import pop_img_stats
from rsgispy.modelling._helpers import read_model_vector, write_model_vector
from rsgispy.vectorprocess import VectorInput, validate_columns_exist
from rsgispy.zonalstats.point import ext_point_band_values

__all__ = [
    "create_bias_grid",
    "calc_bias_weights",
    "sample_bias_weighted_pts",
]

_BIAS_METHODS = ("kde", "count", "distance")
_BIAS_NO_DATA = 0.0


def _iter_windows(
    n_rows: int, n_cols: int, block_size: Optional[int]
) -> Iterator[Window]:
    """Yield row-major processing windows covering an ``n_rows`` × ``n_cols`` grid."""
    blk = int(block_size) if block_size is not None else get_default_block_size()
    if blk < 1:
        raise rsgispy.RSGISPyException(f"block_size must be >= 1, got {block_size!r}.")
    for row_off in range(0, n_rows, blk):
        for col_off in range(0, n_cols, blk):
            yield Window(
                col_off,
                row_off,
                min(blk, n_cols - col_off),
                min(blk, n_rows - row_off),
            )


def _check_north_up(transform: rasterio.Affine, img: str) -> None:
    if transform.b != 0.0 or transform.d != 0.0:
        raise rsgispy.RSGISPyException(
            f"'{img}' has a rotated geotransform, which is not supported; "
            "warp it to a north-up grid first."
        )


def _window_centres(
    transform: rasterio.Affine, win: Window
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-centre x (per column) and y (per row) coordinates of ``win``."""
    cols = np.arange(win.col_off, win.col_off + win.width, dtype=np.float64) + 0.5
    rows = np.arange(win.row_off, win.row_off + win.height, dtype=np.float64) + 0.5
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e
    return xs, ys


def _read_feats_in_crs(
    vec_file: VectorInput,
    vec_lyr: Optional[str],
    crs: Optional[rasterio.crs.CRS],
) -> gpd.GeoDataFrame:
    gdf = read_model_vector(vec_file, vec_lyr=vec_lyr)
    gdf = gdf[~(gdf.geometry.is_empty | gdf.geometry.isna())]
    if gdf.empty:
        raise rsgispy.RSGISPyException("The sample/feature layer has no geometries.")
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def _pts_xy_weights(
    gdf: gpd.GeoDataFrame, weight_col: Optional[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    geoms = gdf.geometry
    pts = geoms.where(geoms.geom_type == "Point", geoms.centroid)
    p_x = pts.x.to_numpy(dtype=np.float64)
    p_y = pts.y.to_numpy(dtype=np.float64)
    if weight_col is None:
        wgt = np.ones(p_x.size, dtype=np.float64)
    else:
        validate_columns_exist(gdf, [weight_col])
        wgt = gdf[weight_col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(wgt)) or np.any(wgt < 0):
            raise rsgispy.RSGISPyException(
                f"weight_col '{weight_col}' must contain finite values >= 0."
            )
    return p_x, p_y, wgt


def _default_kde_bandwidth(p_x: np.ndarray, p_y: np.ndarray) -> float:
    """Scott's rule for two dimensions: ``n^(-1/6)`` × mean coordinate SD."""
    if p_x.size < 2:
        raise rsgispy.RSGISPyException(
            "At least two records are needed to estimate a KDE bandwidth; "
            "pass bandwidth explicitly."
        )
    sd_mean = 0.5 * (float(np.std(p_x, ddof=1)) + float(np.std(p_y, ddof=1)))
    if not np.isfinite(sd_mean) or sd_mean <= 0.0:
        raise rsgispy.RSGISPyException(
            "Cannot estimate a KDE bandwidth (all records are at one location); "
            "pass bandwidth explicitly."
        )
    return sd_mean * float(p_x.size) ** (-1.0 / 6.0)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _kde_block(
    xs: np.ndarray,
    ys: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    wgt: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Unnormalised Gaussian kernel density on the pixel centres of a block.
    The 2D Gaussian kernel is separable, so the block is ``Gyᵀ · diag(w) · Gx``."""
    reach = 4.0 * bandwidth
    sel = (
        (p_x >= xs.min() - reach)
        & (p_x <= xs.max() + reach)
        & (p_y >= ys.min() - reach)
        & (p_y <= ys.max() + reach)
        & (wgt > 0)
    )
    if not np.any(sel):
        return np.zeros((ys.size, xs.size), dtype=np.float64)
    d_x = (xs[None, :] - p_x[sel, None]) / bandwidth
    d_y = (ys[None, :] - p_y[sel, None]) / bandwidth
    # Truncate each axis at 4 bandwidths per pixel so the result does not depend
    # on the processing block size.
    g_x = np.where(np.abs(d_x) <= 4.0, np.exp(-0.5 * d_x**2), 0.0)
    g_y = np.where(np.abs(d_y) <= 4.0, np.exp(-0.5 * d_y**2), 0.0)
    return (g_y * wgt[sel, None]).T @ g_x


def _count_block(
    win: Window,
    transform: rasterio.Affine,
    p_x: np.ndarray,
    p_y: np.ndarray,
    wgt: np.ndarray,
) -> np.ndarray:
    cols = np.floor((p_x - transform.c) / transform.a).astype(np.int64) - int(
        win.col_off
    )
    rows = np.floor((p_y - transform.f) / transform.e).astype(np.int64) - int(
        win.row_off
    )
    inside = (rows >= 0) & (rows < win.height) & (cols >= 0) & (cols < win.width)
    out = np.zeros((int(win.height), int(win.width)), dtype=np.float64)
    np.add.at(out, (rows[inside], cols[inside]), wgt[inside])
    return out


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
def create_bias_grid(
    smpls_vec_file: VectorInput,
    smpls_vec_lyr: Optional[str],
    in_msk_img: str,
    img_msk_val: int,
    output_img: str,
    *,
    method: str = "kde",
    bandwidth: Optional[float] = None,
    dist_decay: Optional[float] = None,
    min_weight: float = 1e-3,
    weight_col: Optional[str] = None,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    use_pbar: bool = True,
) -> None:
    """
    Create a sampling-bias (sampling-effort) raster: values proportional to the
    probability that a location would have been surveyed. Use it to draw
    bias-weighted background points (``bias_img`` in
    :func:`gen_pseudo_absences_smpls`), as a covariate held constant at
    prediction (``fixed_var_vals`` in the ``pred_*`` functions), or for
    inverse-bias sample weights (:func:`calc_bias_weights`).

    Methods:

    * ``"kde"`` (Default) — Gaussian kernel density of records, typically
      "target-group" records: occurrences of related species collected with
      the same methods, which show where people have looked. ``bandwidth`` is
      the kernel standard deviation in CRS units (Default: Scott's rule,
      ``n^(-1/6)`` × the mean of the x and y standard deviations). Records
      more than 4 bandwidths from a pixel in x or y are ignored for that
      pixel.
    * ``"count"`` — number of records (or sum of ``weight_col``) per pixel,
      without smoothing (use ``"kde"`` for a smoothed surface).
    * ``"distance"`` — effort decaying with distance to features such as roads,
      paths or settlements (``smpls_vec_file`` holds those features):
      ``exp(-d / dist_decay)``, with ``d`` the distance to the nearest feature
      pixel. Features are rasterised onto the mask grid, so features outside
      the mask raster's extent are ignored. Needs ``scipy`` (the ``stats``
      extra), used by :func:`rsgispy.imagecalc.distance.calc_dist_to_img_vals`.

    The output is ``float32`` on the grid of ``in_msk_img``, ``0`` (no data)
    outside ``in_msk_img == img_msk_val``, and scaled inside it to
    ``min_weight + (1 - min_weight) × value / max(value)``, so every valid
    pixel keeps a small chance of being sampled.

    :param smpls_vec_file: Records (``"kde"``, ``"count"``) or features
                           (``"distance"``): vector path or
                           :class:`geopandas.GeoDataFrame`. Reprojected to the
                           mask CRS if needed; non-point geometries use their
                           centroid for ``"kde"`` and ``"count"``.
    :param smpls_vec_lyr: Vector layer name (or ``None``).
    :param in_msk_img: Mask raster defining the grid and the study area.
    :param img_msk_val: Value of the study-area pixels in ``in_msk_img``.
    :param output_img: Output bias raster path.
    :param method: ``"kde"`` (Default), ``"count"`` or ``"distance"``.
    :param bandwidth: KDE bandwidth (kernel SD) in CRS units.
    :param dist_decay: Distance (CRS units) over which effort falls by a factor
                       of ``e`` (required for ``"distance"``).
    :param min_weight: Smallest output value inside the mask, in ``(0, 1]``
                       (Default = ``1e-3``).
    :param weight_col: Optional column of record weights (``"kde"``, ``"count"``).
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param calc_stats: Calculate statistics and pyramids (Default = ``True``).
    :param block_size: Processing block size in pixels (Default = ``None``:
                       :func:`rsgispy.imageprocess.get_default_block_size`).
    :param tmp_dir: Directory for the intermediate rasters of ``"distance"``
                    (Default = ``None``: a system temporary directory).
    :param use_pbar: Show progress bars (Default = ``True``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        sdm.create_bias_grid(
            "target_group_records.gpkg", "records", "valid_mask.tif", 1,
            "bias_grid.tif", method="kde", bandwidth=2000.0,
        )
    """
    mthd = str(method).strip().lower()
    if mthd not in _BIAS_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported method '{method}'; use one of {_BIAS_METHODS}."
        )
    if not 0.0 < float(min_weight) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"min_weight must be in (0, 1], got {min_weight!r}."
        )
    if bandwidth is not None and (
        not np.isfinite(float(bandwidth)) or float(bandwidth) <= 0.0
    ):
        raise rsgispy.RSGISPyException(f"bandwidth must be > 0, got {bandwidth!r}.")
    if mthd == "distance" and (
        dist_decay is None
        or not np.isfinite(float(dist_decay))
        or float(dist_decay) <= 0.0
    ):
        raise rsgispy.RSGISPyException(
            "dist_decay must be given and > 0 for method='distance'."
        )
    if not os.path.exists(in_msk_img):
        raise rsgispy.RSGISPyException(f"Mask image '{in_msk_img}' does not exist.")

    with rasterio.open(in_msk_img) as msk_src:
        transform = msk_src.transform
        msk_crs = msk_src.crs
        n_rows, n_cols = msk_src.height, msk_src.width
    _check_north_up(transform, in_msk_img)

    feats_gdf = _read_feats_in_crs(smpls_vec_file, smpls_vec_lyr, msk_crs)

    with ExitStack() as stack:
        dist_src = None
        if mthd == "distance":
            tmp_base = stack.enter_context(tempfile.TemporaryDirectory(dir=tmp_dir))
            feat_img = os.path.join(tmp_base, "bias_feats.tif")
            dist_img = os.path.join(tmp_base, "bias_dist.tif")
            rasterise_vec_lyr(
                feats_gdf,
                None,
                in_msk_img,
                feat_img,
                burn_val=1,
                gdalformat="GTIFF",
                datatype=rsgispy.TYPE_8UINT,
                no_data_val=0.0,
                calc_stats=False,
                use_pbar=False,
            )
            with rasterio.open(feat_img) as feat_src:
                if not np.any(feat_src.read(1) == 1):
                    raise rsgispy.RSGISPyException(
                        "No features fall within the extent of the mask image."
                    )
            calc_dist_to_img_vals(
                feat_img,
                dist_img,
                pxl_vals=1,
                unit_geo=True,
                gdalformat="GTIFF",
                out_no_data_val=-1.0,
                use_pbar=use_pbar,
            )
            dist_src = stack.enter_context(rasterio.open(dist_img))
            p_x = p_y = wgt = np.empty(0)
        else:
            p_x, p_y, wgt = _pts_xy_weights(feats_gdf, weight_col)
            if mthd == "kde" and bandwidth is None:
                bandwidth = _default_kde_bandwidth(p_x, p_y)

        msk_src = stack.enter_context(rasterio.open(in_msk_img))
        grid = IntersectionGrid.from_dataset(msk_src)
        dst = open_output_dataset(
            stack=stack,
            output_img=os.fspath(output_img),
            gdalformat=gdalformat,
            out_rio_dtype="float32",
            grid=grid,
            count=1,
        )
        dst.nodata = _BIAS_NO_DATA

        windows = list(_iter_windows(n_rows, n_cols, block_size))
        raw_max = 0.0
        any_valid = False
        pbar = rsgispy.get_tqdm()
        for win in pbar(windows, disable=not use_pbar, desc="Bias grid"):
            valid = msk_src.read(1, window=win) == img_msk_val
            raw = np.zeros((int(win.height), int(win.width)), dtype=np.float64)
            if np.any(valid):
                any_valid = True
                if mthd == "kde":
                    xs, ys = _window_centres(transform, win)
                    raw = _kde_block(xs, ys, p_x, p_y, wgt, float(bandwidth))
                elif mthd == "count":
                    raw = _count_block(win, transform, p_x, p_y, wgt)
                else:
                    dist = dist_src.read(1, window=win).astype(np.float64)
                    raw = np.where(dist >= 0.0, np.exp(-dist / float(dist_decay)), 0.0)
                    raw[~np.isfinite(raw)] = 0.0
                raw_max = max(raw_max, float(np.max(raw[valid], initial=0.0)))
            raw[~valid] = _BIAS_NO_DATA
            dst.write(raw.astype(np.float32), 1, window=win)

    if not any_valid:
        raise rsgispy.RSGISPyException(
            f"No pixels in '{in_msk_img}' have the value {img_msk_val}."
        )
    if raw_max <= 0.0:
        raise rsgispy.RSGISPyException(
            "The bias surface is zero everywhere inside the mask (no records near "
            "the study area); check the layer, CRS and bandwidth."
        )

    min_w = float(min_weight)
    with rasterio.open(in_msk_img) as msk_src, rasterio.open(
        os.fspath(output_img), "r+"
    ) as dst:
        for win in _iter_windows(n_rows, n_cols, block_size):
            valid = msk_src.read(1, window=win) == img_msk_val
            vals = dst.read(1, window=win).astype(np.float64)
            out = np.full(vals.shape, _BIAS_NO_DATA, dtype=np.float64)
            out[valid] = min_w + (1.0 - min_w) * vals[valid] / raw_max
            dst.write(out.astype(np.float32), 1, window=win)

    if calc_stats:
        pop_img_stats(
            os.fspath(output_img),
            no_data_val=_BIAS_NO_DATA,
            calc_pyramids=True,
            use_pbar=use_pbar,
        )


def _check_same_grid(bias_img: str, in_msk_img: str) -> None:
    with rasterio.open(bias_img) as b_src, rasterio.open(in_msk_img) as m_src:
        same = (
            b_src.width == m_src.width
            and b_src.height == m_src.height
            and b_src.transform.almost_equals(m_src.transform)
            and (b_src.crs == m_src.crs or b_src.crs is None or m_src.crs is None)
        )
    if not same:
        raise rsgispy.RSGISPyException(
            f"The bias image '{bias_img}' is not on the same grid (size, transform "
            f"and CRS) as the mask '{in_msk_img}'. Resample it first, e.g. with "
            "rsgispy.imageutils.warp.resample_img_to_match, or create it with "
            "create_bias_grid using the same mask."
        )


def _bias_block_weights(
    msk_src: rasterio.io.DatasetReader,
    bias_src: rasterio.io.DatasetReader,
    bias_band: int,
    img_msk_val: int,
    win: Window,
) -> np.ndarray:
    """Flat per-pixel sampling weights of ``win`` (0 where not sampleable)."""
    valid = msk_src.read(1, window=win) == img_msk_val
    vals = bias_src.read(bias_band, window=win).astype(np.float64)
    nd_val = bias_src.nodatavals[bias_band - 1]
    good = valid & np.isfinite(vals) & (vals > 0.0)
    if nd_val is not None and not np.isnan(nd_val):
        good &= vals != float(nd_val)
    return np.where(good, vals, 0.0).ravel()


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def sample_bias_weighted_pts(
    in_msk_img: str,
    img_msk_val: int,
    bias_img: str,
    n_pts: int,
    *,
    bias_img_band: int = 1,
    rnd_seed: Optional[Union[int, np.random.Generator]] = None,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> gpd.GeoDataFrame:
    """
    Draw ``n_pts`` random points inside ``in_msk_img == img_msk_val`` with
    probability proportional to the values of ``bias_img`` (which must be on
    the same grid). Zero, negative, non-finite and no-data bias values are
    never sampled.

    Two passes over the rasters, block by block: the first sums the weights
    of each block; the number of points per block is then drawn from a
    multinomial distribution over the block sums, and the second pass draws
    pixels within each block in proportion to their weights. A pixel can be
    drawn more than once; each point is placed at a uniformly random position
    inside its pixel, so points are distinct.

    :return: Point :class:`geopandas.GeoDataFrame` in the mask CRS.
    """
    if int(n_pts) < 1:
        raise rsgispy.RSGISPyException(f"n_pts must be >= 1, got {n_pts!r}.")
    if not os.path.exists(bias_img):
        raise rsgispy.RSGISPyException(f"Bias image '{bias_img}' does not exist.")
    _check_same_grid(bias_img, in_msk_img)
    rng = (
        rnd_seed
        if isinstance(rnd_seed, np.random.Generator)
        else np.random.default_rng(rnd_seed)
    )

    with rasterio.open(in_msk_img) as msk_src, rasterio.open(bias_img) as bias_src:
        if not 1 <= int(bias_img_band) <= bias_src.count:
            raise rsgispy.RSGISPyException(
                f"bias_img_band {bias_img_band} is outside 1..{bias_src.count}."
            )
        transform = msk_src.transform
        windows = list(_iter_windows(msk_src.height, msk_src.width, block_size))
        pbar = rsgispy.get_tqdm()
        block_sums = np.array(
            [
                _bias_block_weights(
                    msk_src, bias_src, int(bias_img_band), img_msk_val, win
                ).sum()
                for win in pbar(windows, disable=not use_pbar, desc="Bias weights")
            ],
            dtype=np.float64,
        )
        total = float(block_sums.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise rsgispy.RSGISPyException(
                "All bias weights inside the mask are zero, negative or no data."
            )
        blk_counts = rng.multinomial(int(n_pts), block_sums / total)

        xs_out = []
        ys_out = []
        for win, cnt in zip(windows, blk_counts):
            if cnt == 0:
                continue
            weights = _bias_block_weights(
                msk_src, bias_src, int(bias_img_band), img_msk_val, win
            )
            pix = rng.choice(weights.size, size=int(cnt), p=weights / weights.sum())
            rows = pix // int(win.width) + int(win.row_off) + rng.random(int(cnt))
            cols = pix % int(win.width) + int(win.col_off) + rng.random(int(cnt))
            p_x = transform.c + cols * transform.a + rows * transform.b
            p_y = transform.f + cols * transform.d + rows * transform.e
            xs_out.append(p_x)
            ys_out.append(p_y)
        crs = msk_src.crs

    p_x = np.concatenate(xs_out)
    p_y = np.concatenate(ys_out)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(p_x, p_y), crs=crs)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_bias_weights(
    smpls_vec_file: VectorInput,
    smpls_vec_lyr: Optional[str],
    bias_img: str,
    out_vec_file: Optional[Union[str, os.PathLike]] = None,
    out_vec_lyr: Optional[str] = None,
    *,
    bias_img_band: int = 1,
    presence_only: bool = True,
    cls_col: str = "clsid",
    out_col: str = "weight",
    out_format: str = "GPKG",
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Add an inverse-bias sample weight column: ``1 / bias`` at each sample,
    scaled to a mean of 1, so records from heavily surveyed places count less.
    Pass the column to the fitting functions with ``sample_weight_col``.

    With ``presence_only=True`` (Default) only the presences
    (``cls_col == 1``) are weighted and the other rows get ``1``; if
    ``cls_col`` is not in the layer, every row is treated as a presence. With
    ``presence_only=False`` every row is weighted and the mean is taken over
    all rows.

    :param smpls_vec_file: Samples: vector path or :class:`geopandas.GeoDataFrame`.
    :param smpls_vec_lyr: Vector layer name (or ``None``).
    :param bias_img: Bias raster, e.g. from :func:`create_bias_grid`.
    :param out_vec_file: Output vector path, or ``None`` to return a
                         :class:`geopandas.GeoDataFrame`.
    :param out_vec_lyr: Output layer name.
    :param bias_img_band: Band of ``bias_img`` (Default = ``1``).
    :param presence_only: Weight only the presences (Default = ``True``).
    :param cls_col: Class column, ``1`` = presence (Default = ``'clsid'``).
    :param out_col: Output weight column (Default = ``'weight'``).
    :param out_format: Output vector format (Default = ``'GPKG'``).
    :param out_compress: Optional GeoParquet compression setting.
    :return: GeoDataFrame with ``out_col`` when ``out_vec_file`` is ``None``,
             otherwise ``None``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        train_gdf = sdm.calc_bias_weights(train_gdf, None, "bias_grid.tif")
    """
    if not os.path.exists(bias_img):
        raise rsgispy.RSGISPyException(f"Bias image '{bias_img}' does not exist.")
    smpls_gdf = read_model_vector(smpls_vec_file, vec_lyr=smpls_vec_lyr)
    if smpls_gdf.empty:
        raise rsgispy.RSGISPyException("The sample layer is empty.")
    tmp_col = "__rsgispy_bias__"
    with rasterio.open(bias_img) as src:
        crs = src.crs
    pts_gdf = smpls_gdf[[smpls_gdf.geometry.name]].reset_index(drop=True)
    if pts_gdf.crs is not None and crs is not None and pts_gdf.crs != crs:
        pts_gdf = pts_gdf.to_crs(crs)
    pts_gdf = ext_point_band_values(
        vec_gdf=pts_gdf,
        input_img=bias_img,
        img_band=int(bias_img_band),
        min_thres=-np.inf,
        max_thres=np.inf,
        out_no_data_val=np.nan,
        out_field=tmp_col,
        reproj_vec=False,
        use_pbar=False,
    )
    bias_vals = pts_gdf[tmp_col].to_numpy(dtype=np.float64)

    if presence_only and cls_col in smpls_gdf.columns:
        wgt_rows = smpls_gdf[cls_col].to_numpy() == 1
    else:
        wgt_rows = np.ones(len(smpls_gdf), dtype=bool)
    if not np.any(wgt_rows):
        raise rsgispy.RSGISPyException(f"No rows have {cls_col} == 1 to weight.")
    bad = wgt_rows & ~(np.isfinite(bias_vals) & (bias_vals > 0.0))
    if np.any(bad):
        raise rsgispy.RSGISPyException(
            f"{int(np.sum(bad))} sample(s) fall outside the bias image or on zero / "
            "no-data bias values; remove them or use a bias grid covering the "
            "samples."
        )
    weights = np.ones(len(smpls_gdf), dtype=np.float64)
    inv = 1.0 / bias_vals[wgt_rows]
    weights[wgt_rows] = inv / inv.mean()

    out_gdf = smpls_gdf.copy()
    out_gdf[out_col] = weights
    return write_model_vector(
        out_gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=True,
        out_compress=out_compress,
    )
