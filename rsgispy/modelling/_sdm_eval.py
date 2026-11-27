"""
Evaluation metrics for :mod:`rsgispy.modelling.species_distribution`: the
continuous Boyce index for presence-only evaluation (from score arrays or a
prediction raster) and a single dictionary of threshold-dependent and
threshold-independent metrics.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy.ml.cv import _iter_cv_splits
from rsgispy.modelling._helpers import (
    calc_binary_roc_auc,
    calc_binary_threshold,
    check_sklearn_available,
    extract_features_and_target,
    finalise_plot,
    fit_clean_estimator,
    get_positive_class_probs,
    read_model_vector,
)
from rsgispy.tools.accuracy import accuracy_scores_binary
from rsgispy.vector._stats_autocorr import calc_correlogram_arrays
from rsgispy.vectorprocess import VectorInput
from rsgispy.zonalstats.point import ext_point_band_values

__all__ = [
    "calc_boyce_index",
    "calc_boyce_index_from_raster",
    "calc_sdm_eval_metrics",
    "calc_null_model_auc",
    "calc_residual_autocorrelation",
]

_BOYCE_METHODS = ("spearman", "pearson")
_THRESHOLD_METHODS = ("max_tss", "sens_eq_spec")


def _as_1d_float(vals: Any, name: str) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise rsgispy.RSGISPyException(f"{name} has no finite values.")
    return arr


def _check_boyce_args(n_bins: int, window_width: float, method: str) -> str:
    if int(n_bins) < 2:
        raise rsgispy.RSGISPyException(f"n_bins must be >= 2, got {n_bins!r}.")
    if not 0.0 < float(window_width) <= 1.0:
        raise rsgispy.RSGISPyException(
            f"window_width must be in (0, 1], got {window_width!r}."
        )
    mthd = str(method).strip().lower()
    if mthd not in _BOYCE_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported method '{method}'; use one of {_BOYCE_METHODS}."
        )
    return mthd


def _boyce_windows(
    v_min: float, v_max: float, n_bins: int, window_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges of ``n_bins`` overlapping moving windows of width
    ``window_width`` × range, evenly spaced from ``v_min`` to ``v_max``."""
    width = float(window_width) * (v_max - v_min)
    lows = np.linspace(v_min, v_max - width, int(n_bins))
    return lows, lows + width


def _window_counts(vals: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Number of ``vals`` in each closed window ``[low, high]``."""
    srt = np.sort(vals)
    return np.searchsorted(srt, highs, side="right") - np.searchsorted(
        srt, lows, side="left"
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _boyce_from_counts(
    pres_cnts: np.ndarray,
    bg_cnts: np.ndarray,
    n_pres: int,
    n_bg: int,
    mids: np.ndarray,
    method: str,
    rm_duplicates: bool,
) -> Dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        pe_ratio = (pres_cnts / float(n_pres)) / (bg_cnts / float(n_bg))
    pe_ratio = np.where(bg_cnts > 0, pe_ratio, np.nan)

    keep = np.isfinite(pe_ratio)
    if rm_duplicates:
        # As ecospat.boyce: drop windows whose P/E equals the previous window's,
        # so long flat runs (e.g. no presences) do not inflate the correlation.
        fin_idx = np.flatnonzero(keep)
        dup = np.zeros(pe_ratio.size, dtype=bool)
        if fin_idx.size > 1:
            same = np.isclose(pe_ratio[fin_idx[1:]], pe_ratio[fin_idx[:-1]])
            dup[fin_idx[1:][same]] = True
        keep &= ~dup

    boyce = np.nan
    if np.count_nonzero(keep) >= 2:
        x_vals = pd.Series(mids[keep])
        y_vals = pd.Series(pe_ratio[keep])
        if method == "spearman":
            x_vals = x_vals.rank()
            y_vals = y_vals.rank()
        if x_vals.std() > 0 and y_vals.std() > 0:
            boyce = float(np.corrcoef(x_vals, y_vals)[0, 1])
    return {"boyce": boyce, "bin_mids": mids, "pe_ratio": pe_ratio}


def _plot_boyce(result: Dict[str, Any], plot_file: str) -> None:
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["bin_mids"], result["pe_ratio"], color="#1f77b4", marker=".")
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted suitability")
    ax.set_ylabel("Predicted / expected ratio")
    ax.set_title(f"Continuous Boyce index = {result['boyce']:.3f}")
    finalise_plot(fig, out_file=plot_file)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calc_boyce_index(
    pres_scores: Union[np.ndarray, Sequence[float]],
    bg_scores: Union[np.ndarray, Sequence[float]],
    *,
    n_bins: int = 101,
    window_width: float = 0.1,
    method: str = "spearman",
    rm_duplicates: bool = True,
    plot_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the continuous Boyce index (Hirzel et al. 2006), the usual
    accuracy measure for presence-only data: it needs only predictions at
    presences and at background (or pseudo-absence) locations, not true
    absences.

    The range of the background scores is covered by ``n_bins`` overlapping
    moving windows, each ``window_width`` × the range wide. For each window the
    predicted/expected (P/E) ratio is the fraction of presences in the window
    divided by the fraction of background points in it. A good model has P/E
    rising with the predicted value; the index is the Spearman (default) or
    Pearson correlation between P/E and the window midpoints, from ``-1``
    (worse than random) through ``0`` (random) to ``1``. Windows with no
    background points give ``NaN`` P/E and are left out, as are (with
    ``rm_duplicates=True``, as in ``ecospat.boyce``) windows whose P/E equals
    the previous window's.

    :param pres_scores: Predicted values at the presences.
    :param bg_scores: Predicted values at background points (or at every valid
                      pixel; see :func:`calc_boyce_index_from_raster`).
    :param n_bins: Number of moving windows (Default = ``101``).
    :param window_width: Window width as a fraction of the score range, in
                         ``(0, 1]`` (Default = ``0.1``).
    :param method: ``"spearman"`` (Default) or ``"pearson"``.
    :param rm_duplicates: Drop windows repeating the previous P/E value
                          (Default = ``True``).
    :param plot_file: Optional output path for a plot of P/E against the
                      predicted value.
    :return: ``{"boyce": float, "bin_mids": array, "pe_ratio": array}``;
             ``boyce`` is ``NaN`` if fewer than two windows are usable.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        res = sdm.calc_boyce_index(pres_probs, bg_probs)
        print(res["boyce"])
    """
    mthd = _check_boyce_args(n_bins, window_width, method)
    pres = _as_1d_float(pres_scores, "pres_scores")
    bkg = _as_1d_float(bg_scores, "bg_scores")
    v_min = float(min(bkg.min(), pres.min()))
    v_max = float(max(bkg.max(), pres.max()))
    if v_max <= v_min:
        raise rsgispy.RSGISPyException("All scores are equal; Boyce index undefined.")
    lows, highs = _boyce_windows(v_min, v_max, n_bins, window_width)
    result = _boyce_from_counts(
        _window_counts(pres, lows, highs),
        _window_counts(bkg, lows, highs),
        pres.size,
        bkg.size,
        (lows + highs) / 2.0,
        mthd,
        rm_duplicates,
    )
    if plot_file is not None:
        _plot_boyce(result, plot_file)
    return result


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_boyce_index_from_raster(
    pred_img: str,
    pres_vec_file: VectorInput,
    pres_vec_lyr: Optional[str] = None,
    *,
    img_band: int = 1,
    n_bins: int = 101,
    window_width: float = 0.1,
    method: str = "spearman",
    rm_duplicates: bool = True,
    plot_file: Optional[str] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Calculate the continuous Boyce index (see :func:`calc_boyce_index`) of a
    prediction raster, using every valid pixel as the background distribution
    and the raster values at the presence points. The raster is read in blocks
    (two passes: range, then exact window counts), so it is never loaded whole.
    Pixels equal to the raster's no-data value or non-finite are ignored, as
    are presences falling on them or outside the raster.

    :param pred_img: Prediction raster (e.g. from
                     :func:`pred_ensemble_sklearn_mdls_prob`).
    :param pres_vec_file: Presence points: path or :class:`geopandas.GeoDataFrame`
                          (reprojected to the raster CRS if needed).
    :param pres_vec_lyr: Presence layer name (or ``None``).
    :param img_band: Raster band (Default = ``1``).
    :param n_bins: Number of moving windows (Default = ``101``).
    :param window_width: Window width as a fraction of the range (Default = ``0.1``).
    :param method: ``"spearman"`` (Default) or ``"pearson"``.
    :param rm_duplicates: Drop windows repeating the previous P/E value
                          (Default = ``True``).
    :param plot_file: Optional output path for the P/E plot.
    :param use_pbar: Show progress bars (Default = ``True``).
    :return: As :func:`calc_boyce_index`.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        res = sdm.calc_boyce_index_from_raster("sdm_prob.tif", "presences.gpkg", "pres")
    """
    mthd = _check_boyce_args(n_bins, window_width, method)
    if not os.path.exists(pred_img):
        raise rsgispy.RSGISPyException(f"Prediction image '{pred_img}' does not exist.")

    pres_gdf = read_model_vector(pres_vec_file, vec_lyr=pres_vec_lyr)
    with rasterio.open(pred_img) as src:
        img_crs = src.crs
        nd_val = src.nodatavals[int(img_band) - 1]
    pts = pres_gdf[[pres_gdf.geometry.name]].reset_index(drop=True)
    if pts.crs is not None and img_crs is not None and pts.crs != img_crs:
        pts = pts.to_crs(img_crs)
    pts = ext_point_band_values(
        vec_gdf=pts,
        input_img=pred_img,
        img_band=int(img_band),
        min_thres=-np.inf,
        max_thres=np.inf,
        out_no_data_val=np.nan,
        out_field="__score__",
        reproj_vec=False,
        use_pbar=False,
    )
    pres = pts["__score__"].to_numpy(dtype=np.float64)
    if nd_val is not None and not np.isnan(nd_val):
        pres = pres[pres != float(nd_val)]
    pres = _as_1d_float(pres, "presence scores (inside the raster)")

    def _valid_block(src: Any, win: Any) -> np.ndarray:
        vals = src.read(int(img_band), window=win).astype(np.float64).ravel()
        good = np.isfinite(vals)
        if nd_val is not None and not np.isnan(nd_val):
            good &= vals != float(nd_val)
        return vals[good]

    pbar = rsgispy.get_tqdm()
    with rasterio.open(pred_img) as src:
        windows = [win for _, win in src.block_windows(int(img_band))]
        v_min, v_max, n_bg = np.inf, -np.inf, 0
        for win in pbar(windows, disable=not use_pbar, desc="Boyce: range"):
            vals = _valid_block(src, win)
            if vals.size:
                v_min = min(v_min, float(vals.min()))
                v_max = max(v_max, float(vals.max()))
                n_bg += vals.size
        if n_bg == 0:
            raise rsgispy.RSGISPyException(f"'{pred_img}' has no valid pixels.")
        v_min = min(v_min, float(pres.min()))
        v_max = max(v_max, float(pres.max()))
        if v_max <= v_min:
            raise rsgispy.RSGISPyException(
                "All scores are equal; Boyce index undefined."
            )
        lows, highs = _boyce_windows(v_min, v_max, n_bins, window_width)
        bg_cnts = np.zeros(lows.size, dtype=np.int64)
        for win in pbar(windows, disable=not use_pbar, desc="Boyce: counts"):
            vals = _valid_block(src, win)
            if vals.size:
                bg_cnts += _window_counts(vals, lows, highs)

    result = _boyce_from_counts(
        _window_counts(pres, lows, highs),
        bg_cnts,
        pres.size,
        n_bg,
        (lows + highs) / 2.0,
        mthd,
        rm_duplicates,
    )
    if plot_file is not None:
        _plot_boyce(result, plot_file)
    return result


# pylint: disable=too-many-locals
def calc_sdm_eval_metrics(
    y_true: Union[np.ndarray, Sequence[Any]],
    scores: Union[np.ndarray, Sequence[float]],
    *,
    threshold: Optional[float] = None,
    threshold_method: str = "max_tss",
    bg_scores: Optional[Union[str, np.ndarray, Sequence[float]]] = None,
) -> Dict[str, float]:
    """
    Calculate the usual species distribution model accuracy measures in one
    dictionary, from labelled samples and continuous scores (e.g. presence
    probabilities).

    Threshold-independent: ``auc`` (ROC AUC) and, when ``bg_scores`` is given,
    ``boyce`` (continuous Boyce index, :func:`calc_boyce_index`). The scores
    are then turned into presence/absence at ``threshold`` (default: chosen
    from these samples with :func:`calc_binary_threshold` and
    ``threshold_method``; a sample is present when ``score >= threshold``) to
    give ``tss``, ``sensitivity``, ``specificity``, ``omission_rate``
    (``1 - sensitivity``), ``commission_rate`` (``1 - specificity``),
    ``kappa``, ``accuracy``, ``balanced_accuracy`` and ``prevalence``, from
    :func:`rsgispy.tools.accuracy.accuracy_scores_binary`. ``n_presence`` and
    ``n_absence`` are also returned.

    Choosing the threshold on the same samples that are scored makes the
    threshold-dependent values slightly optimistic; pass a ``threshold`` chosen
    on other samples to avoid that.

    :param y_true: Binary labels: ``1`` (or ``True``) for presence, ``0``
                   (or ``-1``, ``False``) for absence. Both classes needed.
    :param scores: Continuous scores, higher for presence.
    :param threshold: Optional fixed threshold.
    :param threshold_method: ``"max_tss"`` (Default) or ``"sens_eq_spec"``.
    :param bg_scores: Optional background scores for the Boyce index, or
                      ``"absences"`` to use the scores of the absence rows.
    :return: Dictionary of metrics (all ``float``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        metrics = sdm.calc_sdm_eval_metrics(test_y, test_probs, bg_scores="absences")
        print(metrics["auc"], metrics["tss"], metrics["boyce"])
    """
    y_arr = np.asarray(y_true).ravel()
    s_arr = np.asarray(scores, dtype=np.float64).ravel()
    if y_arr.size != s_arr.size:
        raise rsgispy.RSGISPyException(
            f"y_true ({y_arr.size}) and scores ({s_arr.size}) differ in length."
        )
    if not np.all(np.isfinite(s_arr)):
        raise rsgispy.RSGISPyException("scores must all be finite.")
    if str(threshold_method) not in _THRESHOLD_METHODS:
        raise rsgispy.RSGISPyException(
            f"Unsupported threshold_method '{threshold_method}'; use one of "
            f"{_THRESHOLD_METHODS}."
        )
    y_bin = (y_arr == 1).astype(np.uint8)

    _, _, auc_val = calc_binary_roc_auc(y_bin, s_arr)
    thres = (
        float(threshold)
        if threshold is not None
        else float(calc_binary_threshold(y_bin, s_arr, method=threshold_method))
    )
    y_pred = (s_arr >= thres).astype(np.uint8)
    names, vals = accuracy_scores_binary(y_bin, y_pred)
    acc = dict(zip(names.tolist(), vals.tolist()))

    out: Dict[str, float] = {
        "auc": float(auc_val),
        "threshold": thres,
        "tss": float(acc["Hanssen-Kuiper skill"]),
        "sensitivity": float(acc["Sensitivity"]),
        "specificity": float(acc["Specificity"]),
        "omission_rate": 1.0 - float(acc["Sensitivity"]),
        "commission_rate": 1.0 - float(acc["Specificity"]),
        "kappa": float(acc["Cohen Kappa"]),
        "accuracy": float(acc["Accuracy"]),
        "balanced_accuracy": float(acc["Balanced Accuracy"]),
        "prevalence": float(acc["Prevalence"]),
        "n_presence": float(np.count_nonzero(y_bin == 1)),
        "n_absence": float(np.count_nonzero(y_bin == 0)),
    }
    if bg_scores is not None:
        if isinstance(bg_scores, str):
            if bg_scores != "absences":
                raise rsgispy.RSGISPyException(
                    "bg_scores must be an array of scores or 'absences'."
                )
            bkg = s_arr[y_bin == 0]
        else:
            bkg = np.asarray(bg_scores, dtype=np.float64)
        out["boyce"] = float(calc_boyce_index(s_arr[y_bin == 1], bkg)["boyce"])
    return out


def _sample_coords(gdf: Any) -> np.ndarray:
    geoms = gdf.geometry
    pts = geoms.where(geoms.geom_type == "Point", geoms.centroid)
    return np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()]).astype(np.float64)


def _valid_rows(x_df: pd.DataFrame, no_data_val: Optional[float]) -> np.ndarray:
    vals = x_df.to_numpy(dtype=np.float64)
    good = np.all(np.isfinite(vals), axis=1)
    if no_data_val is not None:
        good &= ~np.any(vals == float(no_data_val), axis=1)
    return good


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-statements,too-many-branches
def calc_null_model_auc(
    est_cls_obj: Any,
    bg_vec_file: VectorInput,
    bg_vec_lyr: Optional[str],
    test_vec_file: Optional[VectorInput],
    test_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    n_pres: int,
    *,
    n_reps: int = 99,
    observed_auc: Optional[float] = None,
    cls_col: str = "clsid",
    cv: Optional[Any] = None,
    no_data_val: Optional[float] = -9999.0,
    rnd_seed: Optional[int] = None,
    plot_file: Optional[str] = None,
    use_pbar: bool = True,
) -> Dict[str, Any]:
    """
    Test whether a model's AUC is better than expected by chance with a null
    model (Raes & ter Steege 2007): ``n_reps`` times, draw ``n_pres`` random
    "presences" from the background points, fit a clone of ``est_cls_obj`` to
    them against the rest of the background, and calculate its AUC. The null
    AUCs show how high an AUC the model reaches with no real signal, given the
    number of presences and the study area; with few presences or a flexible
    model they can be far above 0.5.

    How each null AUC is calculated should match how ``observed_auc`` was:

    * ``test_vec_file`` given — on the test layer: as many of its absences as
      it has presences (``cls_col == 1``) are drawn at random as "presences",
      and the AUC is calculated against its other absences;
    * ``cv`` given (a splitter as in :func:`fit_kfold_sklearn_mdls`, e.g.
      :class:`rsgispy.ml.cv.SpatialBlockKFold`; coordinates from the
      background geometries) — the mean out-of-fold AUC;
    * neither — the training AUC, as in Raes & ter Steege (2007).

    :param est_cls_obj: ``scikit-learn`` classifier template (not modified).
    :param bg_vec_file: Background points with one column per variable (e.g.
                        pseudo-absences after :func:`extract_env_var_data`).
                        Rows with ``cls_col == 1``, if the column exists, and
                        rows with missing or ``no_data_val`` values are left out.
    :param bg_vec_lyr: Background layer name (or ``None``).
    :param test_vec_file: Optional test layer with ``cls_col`` (or ``None``).
    :param test_vec_lyr: Test layer name (or ``None``).
    :param analysis_vars: Predictor columns.
    :param n_pres: Number of random presences per replicate (use the number of
                   real training presences).
    :param n_reps: Number of null replicates (Default = ``99``).
    :param observed_auc: Optional AUC of the real model to test.
    :param cls_col: Class column (Default = ``'clsid'``).
    :param cv: Optional cross-validation splitter or integer.
    :param no_data_val: Value treated as missing (Default = ``-9999.0``).
    :param rnd_seed: Optional random seed.
    :param plot_file: Optional histogram of the null AUCs (observed AUC marked).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :return: Dictionary with ``null_aucs`` (array), ``null_mean``,
             ``null_sd``, ``null_p95`` (95th percentile: the AUC a model must
             beat for significance at 5 %), ``mode`` (``"test"``, ``"cv"`` or
             ``"train"``) and, when ``observed_auc`` is given, ``observed_auc``
             and the one-sided ``p_value``
             ``(1 + #(null >= observed)) / (n_reps + 1)``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        res = sdm.calc_null_model_auc(
            clf, abs_gdf, None, test_gdf, None, analysis_vars, n_pres=34,
            observed_auc=0.93, n_reps=99, rnd_seed=42,
        )
        print(res["null_p95"], res["p_value"])
    """
    check_sklearn_available("calc_null_model_auc")
    if int(n_reps) < 1:
        raise rsgispy.RSGISPyException("n_reps must be >= 1.")
    if test_vec_file is not None and cv is not None:
        raise rsgispy.RSGISPyException("Give either test_vec_file or cv, not both.")

    var_list = list(analysis_vars)
    bg_gdf = read_model_vector(bg_vec_file, vec_lyr=bg_vec_lyr)
    if cls_col in bg_gdf.columns:
        bg_gdf = bg_gdf[bg_gdf[cls_col].to_numpy() != 1]
    bg_x, _ = extract_features_and_target(bg_gdf, var_list, cls_col=None)
    good = _valid_rows(bg_x, no_data_val)
    bg_gdf = bg_gdf[good]
    bg_x = bg_x[good].reset_index(drop=True)
    n_bg = len(bg_x)
    if not 2 <= int(n_pres) < n_bg - 1:
        raise rsgispy.RSGISPyException(
            f"n_pres must be >= 2 and smaller than the number of valid background "
            f"points ({n_bg}) minus 1, got {n_pres}."
        )

    test_x = None
    n_test_pres = 0
    coords: Optional[np.ndarray] = None
    if test_vec_file is not None:
        test_gdf = read_model_vector(test_vec_file, vec_lyr=test_vec_lyr)
        t_x, t_y = extract_features_and_target(test_gdf, var_list, cls_col=cls_col)
        n_test_pres = int(np.sum(t_y == 1))
        test_x = t_x[(t_y != 1) & _valid_rows(t_x, no_data_val)].reset_index(drop=True)
        if n_test_pres < 1 or n_test_pres >= len(test_x):
            raise rsgispy.RSGISPyException(
                "The test layer needs at least one presence and more absences "
                "than presences."
            )
        mode = "test"
    elif cv is not None:
        mode = "cv"
        coords = _sample_coords(bg_gdf)
    else:
        mode = "train"

    rng = np.random.default_rng(rnd_seed)
    null_aucs = np.full(int(n_reps), np.nan)
    pbar = rsgispy.get_tqdm()
    for rep_i in pbar(range(int(n_reps)), disable=not use_pbar, desc="Null models"):
        y_arr = np.zeros(n_bg, dtype=np.uint8)
        y_arr[rng.choice(n_bg, int(n_pres), replace=False)] = 1
        rep_seed = int(rng.integers(0, 2**31 - 1))
        if mode == "cv":
            fold_aucs = []
            for tr_idx, val_idx in _iter_cv_splits(cv, bg_x, y_arr, coords, None):
                if np.unique(y_arr[tr_idx]).size < 2 or np.unique(y_arr[val_idx]).size < 2:
                    continue
                mdl = fit_clean_estimator(
                    est_cls_obj, bg_x.iloc[tr_idx], y_arr[tr_idx],
                    sync_input=False, rnd_seed=rep_seed,
                )
                probs = get_positive_class_probs(mdl, bg_x.iloc[val_idx], pos_label=1)
                fold_aucs.append(calc_binary_roc_auc(y_arr[val_idx], probs)[2])
            if fold_aucs:
                null_aucs[rep_i] = float(np.mean(fold_aucs))
            continue
        mdl = fit_clean_estimator(
            est_cls_obj, bg_x, y_arr, sync_input=False, rnd_seed=rep_seed
        )
        if mode == "test" and test_x is not None:
            y_test = np.zeros(len(test_x), dtype=np.uint8)
            y_test[rng.choice(len(test_x), n_test_pres, replace=False)] = 1
            probs = get_positive_class_probs(mdl, test_x, pos_label=1)
            null_aucs[rep_i] = calc_binary_roc_auc(y_test, probs)[2]
        else:
            probs = get_positive_class_probs(mdl, bg_x, pos_label=1)
            null_aucs[rep_i] = calc_binary_roc_auc(y_arr, probs)[2]

    fin = null_aucs[np.isfinite(null_aucs)]
    if fin.size == 0:
        raise rsgispy.RSGISPyException(
            "No null AUC could be calculated (every cv fold lacked a class)."
        )
    out: Dict[str, Any] = {
        "null_aucs": null_aucs,
        "null_mean": float(np.mean(fin)),
        "null_sd": float(np.std(fin)),
        "null_p95": float(np.percentile(fin, 95)),
        "mode": mode,
    }
    if observed_auc is not None:
        out["observed_auc"] = float(observed_auc)
        out["p_value"] = float((1 + np.sum(fin >= float(observed_auc))) / (fin.size + 1))

    if plot_file is not None:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fin, bins=20, color="#9ecae1", edgecolor="#3182bd")
        ax.axvline(out["null_p95"], color="0.4", linestyle="--", label="Null 95th percentile")
        if observed_auc is not None:
            ax.axvline(float(observed_auc), color="#d7301f", label="Observed AUC")
        ax.set_xlabel(f"Null model AUC ({mode})")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left")
        finalise_plot(fig, out_file=plot_file)
    return out


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def calc_residual_autocorrelation(
    est_cls_obj: Any,
    smpls_vec_file: VectorInput,
    smpls_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    *,
    cls_col: str = "clsid",
    dist_classes: Optional[Sequence[float]] = None,
    n_classes: int = 10,
    max_dist: Optional[float] = None,
    n_perms: int = 99,
    rnd_seed: Optional[int] = None,
    plot_file: Optional[str] = None,
    allow_geographic: bool = False,
) -> pd.DataFrame:
    """
    Check a fitted model's residuals for spatial autocorrelation: the Moran's I
    correlogram (:func:`rsgispy.vector.stats.calc_correlogram`) of the
    residuals ``y - p``, where ``y`` is the class (``1`` presence, ``0``
    absence) and ``p`` the predicted presence probability. Significantly
    positive Moran's I at short distances means nearby samples share
    unexplained variation, e.g. a missing variable, clustered records or
    sampling bias: thin the samples further, add variables, or rely on spatial
    cross-validation for the accuracy.

    :param est_cls_obj: Fitted ``scikit-learn`` classifier.
    :param smpls_vec_file: Samples with ``analysis_vars`` and ``cls_col``
                           (e.g. the training or test layer).
    :param smpls_vec_lyr: Layer name (or ``None``).
    :param analysis_vars: Predictor columns, in training order.
    :param cls_col: Class column (Default = ``'clsid'``).
    :param dist_classes: Optional distance class edges.
    :param n_classes: Number of equal-width classes (Default = ``10``).
    :param max_dist: Largest distance (Default: half the layer's diagonal).
    :param n_perms: Permutations per class (Default = ``99``).
    :param rnd_seed: Optional random seed.
    :param plot_file: Optional correlogram plot path.
    :param allow_geographic: Allow a geographic CRS (Default = ``False``).
    :return: Correlogram :class:`pandas.DataFrame` (see
             :func:`rsgispy.vector.stats.calc_correlogram`).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        corr_df = sdm.calc_residual_autocorrelation(
            mdl, train_gdf, None, analysis_vars, max_dist=5000.0,
            plot_file="plots/residual_correlogram.png",
        )
    """
    smpls_gdf = read_model_vector(smpls_vec_file, vec_lyr=smpls_vec_lyr)
    if (
        smpls_gdf.crs is not None
        and smpls_gdf.crs.is_geographic
        and not allow_geographic
    ):
        raise rsgispy.RSGISPyException(
            "The samples have a geographic CRS (degrees); reproject them or pass "
            "allow_geographic=True."
        )
    x_df, y_arr = extract_features_and_target(smpls_gdf, analysis_vars, cls_col=cls_col)
    resid = (np.asarray(y_arr) == 1).astype(np.float64) - np.asarray(
        get_positive_class_probs(est_cls_obj, x_df, pos_label=1), dtype=np.float64
    )
    return calc_correlogram_arrays(
        _sample_coords(smpls_gdf),
        resid,
        dist_classes=dist_classes,
        n_classes=n_classes,
        max_dist=max_dist,
        n_perms=n_perms,
        rnd_seed=rnd_seed,
        plot_file=plot_file,
    )
