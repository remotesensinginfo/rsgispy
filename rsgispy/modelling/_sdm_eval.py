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
from rsgispy.modelling._helpers import (
    calc_binary_roc_auc,
    calc_binary_threshold,
    finalise_plot,
    read_model_vector,
)
from rsgispy.tools.accuracy import accuracy_scores_binary
from rsgispy.vectorprocess import VectorInput
from rsgispy.zonalstats.point import ext_point_band_values

__all__ = [
    "calc_boyce_index",
    "calc_boyce_index_from_raster",
    "calc_sdm_eval_metrics",
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
