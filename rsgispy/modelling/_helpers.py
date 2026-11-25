"""
Shared internal and reusable modelling helpers for :mod:`rsgispy.modelling`.

Provides environmental covariate descriptors (:class:`EnvVarInfo`), vector and
table preparation helpers, isolated ``scikit-learn`` estimator cloning/fitting
utilities, safe matplotlib figure lifecycle management, and a streaming
:mod:`rsgispy.imageprocess` block-prediction engine shared across single-model
and ensemble raster inference functions.

The user-facing helpers (:class:`EnvVarInfo`, :func:`get_positive_class_probs`,
:func:`calc_single_class_predictions`, :func:`calc_binary_roc_auc`,
:func:`calc_binary_threshold`, :func:`plot_roc_curve` and
:func:`predict_raster_from_env_vars`) are public through
:mod:`rsgispy.modelling.species_distribution`; import them from there.
"""

from contextlib import ExitStack
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import (
    BandDefn,
    cast_result_block,
    compute_spatial_intersection,
    extract_roi_feature_matrix,
    iter_image_blocks,
    open_input_datasets,
    open_output_dataset,
)
from rsgispy.imageutils.stats import pop_img_stats, pop_thmt_img_stats
from rsgispy.tools._ml_explain import finalise_plot
from rsgispy.tools._sklearn import (
    HAVE_SKLEARN,
    check_sklearn_available as _check_sklearn_available,
    clone_sklearn_estimator as clone_estimator,
    fit_clean_sklearn_estimator as fit_clean_estimator,
    sklearn,
    sync_sklearn_estimator_state as sync_estimator_state,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.vectorprocess import (
    VectorInput,
    finish_vec_output,
    read_vec_gdf,
    validate_columns_exist,
)

__all__ = [
    "HAVE_SKLEARN",
    "EnvVarInfo",
    "build_env_block_matrix",
    "calc_binary_roc_auc",
    "calc_binary_threshold",
    "calc_single_class_predictions",
    "check_sklearn_available",
    "clone_estimator",
    "env_vars_to_band_defs",
    "extract_features_and_target",
    "finalise_plot",
    "fit_clean_estimator",
    "get_positive_class_probs",
    "lookup_env_var",
    "normalise_array_with_info",
    "plot_roc_curve",
    "predict_raster_from_env_vars",
    "read_model_vector",
    "resolve_analysis_vars",
    "run_env_block_pipeline",
    "sync_estimator_state",
    "write_model_vector",
]

_MASK_BAND_KEY = "__rsgispy_modelling_mask__"


class EnvVarInfo:
    """
    Descriptor storing metadata and normalisation coefficients for an
    environmental covariate band.

    :param name: Variable/column name for the covariate.
    :param file: Path to the raster file representing the covariate.
    :param band: 1-based band index within ``file`` (Default = ``1``).
    :param data_type: Variable type constant (``rsgispy.VAR_TYPE_CONTINUOUS``,
                      ``rsgispy.VAR_TYPE_CATEGORICAL``, or
                      ``rsgispy.VAR_TYPE_UNDEFINED``).
    :param min_vld_val: Optional inclusive lower valid threshold.
    :param max_vld_val: Optional inclusive upper valid threshold.
    :param norm_mean: Optional mean value for z-score normalisation.
    :param norm_std: Optional standard deviation for z-score normalisation.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        name: Optional[str] = None,
        file: Optional[str] = None,
        band: int = 1,
        data_type: int = rsgispy.VAR_TYPE_UNDEFINED,
        min_vld_val: Optional[float] = None,
        max_vld_val: Optional[float] = None,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
    ):
        self.name = name
        self.file = file
        self.band = int(band)
        self.data_type = int(data_type)
        self.min_vld_val = min_vld_val
        self.max_vld_val = max_vld_val
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def set_norm_mean(self, norm_mean: float) -> None:
        """Set the normalisation mean value."""
        self.norm_mean = float(norm_mean)

    def set_norm_std(self, norm_std: float) -> None:
        """Set the normalisation standard deviation value."""
        self.norm_std = float(norm_std)

    def set_norm_mean_std(self, norm_mean: float, norm_std: float) -> None:
        """Set both normalisation mean and standard deviation values."""
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)

    def to_band_spec(self) -> Any:
        """Convert this environmental variable into a :class:`~rsgispy.ml.stack.BandSpec`."""
        # pylint: disable=import-outside-toplevel
        from rsgispy.ml.stack import BandSpec

        return BandSpec.from_env_var_info(self)

    def _format_desc(self) -> str:
        type_str = "Unknown"
        if self.data_type == rsgispy.VAR_TYPE_CONTINUOUS:
            type_str = "Continuous"
        elif self.data_type == rsgispy.VAR_TYPE_CATEGORICAL:
            type_str = "Categorical"
        return (
            f"{self.name}: {self.file} (Band: {self.band}) (Type: {type_str}) "
            f"(Range: {self.min_vld_val}, {self.max_vld_val}) "
            f"(Normalisation: (x - {self.norm_mean}) / {self.norm_std})"
        )

    def __str__(self) -> str:
        return self._format_desc()

    def __repr__(self) -> str:
        return self._format_desc()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvVarInfo):
            return NotImplemented
        return (
            self.name == other.name
            and self.file == other.file
            and self.band == other.band
            and self.data_type == other.data_type
            and self.min_vld_val == other.min_vld_val
            and self.max_vld_val == other.max_vld_val
            and self.norm_mean == other.norm_mean
            and self.norm_std == other.norm_std
        )


def check_sklearn_available(feature_name: str) -> None:
    """Verify ``scikit-learn`` is installed before running model routines."""
    _check_sklearn_available(feature_name, extra="modelling")


def lookup_env_var(
    env_vars: Mapping[str, EnvVarInfo],
    var_name: str,
) -> EnvVarInfo:
    """
    Look up an :class:`EnvVarInfo` in ``env_vars`` by dictionary key or by
    :attr:`EnvVarInfo.name`.
    """
    if var_name in env_vars:
        return env_vars[var_name]
    matches = [info for info in env_vars.values() if info.name == var_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise rsgispy.RSGISPyException(
            f"Variable '{var_name}' matches multiple entries in env_vars."
        )
    raise rsgispy.RSGISPyException(
        f"Variable '{var_name}' was not found in env_vars."
    )


def resolve_analysis_vars(
    env_vars: Union[Sequence[str], Mapping[str, EnvVarInfo]],
    analysis_vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Resolve and validate an ordered list of analysis variable column names
    from ``env_vars`` and optional ``analysis_vars``.
    """
    if analysis_vars is not None:
        raw_vars = [str(v) for v in analysis_vars]
    elif isinstance(env_vars, Mapping):
        raw_vars = [
            info.name if info.name is not None else str(key)
            for key, info in env_vars.items()
        ]
    else:
        raw_vars = [str(v) for v in env_vars]

    if not raw_vars:
        raise rsgispy.RSGISPyException(
            "analysis_vars must contain at least one variable name."
        )
    if isinstance(env_vars, Mapping):
        resolved: List[str] = []
        for var_name in raw_vars:
            info = lookup_env_var(env_vars, var_name)
            resolved.append(info.name if info.name is not None else var_name)
        return resolved
    return raw_vars


def read_model_vector(
    vec_file: VectorInput,
    vec_lyr: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load vector data from file path or :class:`geopandas.GeoDataFrame`."""
    return read_vec_gdf(vec_file, vec_lyr=vec_lyr, copy=True)


def write_model_vector(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    gdf: gpd.GeoDataFrame,
    out_vec_file: Optional[Union[str, os.PathLike]],
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = "GPKG",
    overwrite: bool = True,
    out_compress: Optional[Union[str, bool]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Write a modelling :class:`geopandas.GeoDataFrame` or return in memory."""
    return finish_vec_output(
        gdf,
        out_vec_file=out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )


def extract_features_and_target(
    gdf: gpd.GeoDataFrame,
    analysis_vars: Sequence[str],
    cls_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Validate columns and return a clean ``(X_df, y_arr)`` pair from ``gdf``.
    """
    var_list = list(analysis_vars)
    required = list(var_list)
    if cls_col is not None:
        required.append(cls_col)
    validate_columns_exist(gdf, required)

    x_df = gdf[var_list].copy()
    y_arr = gdf[cls_col].to_numpy() if cls_col is not None else None
    return x_df, y_arr


def normalise_array_with_info(
    values: Union[np.ndarray, pd.Series],
    info: EnvVarInfo,
) -> Union[np.ndarray, pd.Series]:
    """
    Apply z-score normalisation ``(x - norm_mean) / norm_std`` when ``info`` is
    continuous and normalisation parameters are defined, guarding zero std.
    """
    if (
        info.data_type == rsgispy.VAR_TYPE_CONTINUOUS
        and info.norm_mean is not None
        and info.norm_std is not None
    ):
        std_val = float(info.norm_std)
        if std_val == 0.0 or not np.isfinite(std_val):
            return values - float(info.norm_mean)
        return (values - float(info.norm_mean)) / std_val
    return values


def get_positive_class_probs(
    est_cls_obj: Any,
    x_data: Union[pd.DataFrame, np.ndarray],
    pos_label: Any = 1,
) -> np.ndarray:
    """
    Return the probability of the positive (presence) class predicted by a
    fitted ``scikit-learn`` classifier for each row of ``x_data``.

    The column of ``est_cls_obj.predict_proba(x_data)`` is chosen by looking up
    ``pos_label`` in ``est_cls_obj.classes_``, so the result does not depend on
    the order of the classes (indexing ``predict_proba(...)[:, 1]`` is only
    correct when the classes are exactly ``[0, 1]``). Special cases:

    * a 1D ``predict_proba`` result is returned unchanged;
    * a model fitted on a single class other than ``pos_label`` (one
      probability column) gives zeros, as it can never predict ``pos_label``;
    * if the estimator has no ``classes_`` attribute, or ``pos_label`` is not
      one of its classes, the last column is used (for scikit-learn's sorted
      classes, the largest label).

    :param est_cls_obj: Fitted ``scikit-learn`` classifier with ``predict_proba``.
    :param x_data: Feature table, one row per sample, with the columns in the
                   order used for training (a :class:`pandas.DataFrame` with
                   the training column names, or a 2D array).
    :param pos_label: Label of the positive (presence) class (Default = ``1``).
    :return: 1D ``float64`` array of positive class probabilities, one per row
             of ``x_data``.

    .. code:: python

        import geopandas as gpd
        import numpy as np
        import rsgispy.modelling.species_distribution as sdm

        test_gdf = gpd.read_file("sdm_test.gpkg", layer="test")
        # Median presence probability of an ensemble from fit_kfold_sklearn_mdls.
        test_prob = np.median(
            [sdm.get_positive_class_probs(mdl, test_gdf[analysis_vars]) for mdl in mdls],
            axis=0,
        )
    """
    probs = np.asarray(est_cls_obj.predict_proba(x_data), dtype=np.float64)
    if probs.ndim == 1:
        return probs
    classes = getattr(est_cls_obj, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        if pos_label in class_list:
            pos_idx = class_list.index(pos_label)
            return probs[:, pos_idx]
        if probs.shape[1] == 1:
            return np.zeros(probs.shape[0], dtype=np.float64)
    return probs[:, -1]


def calc_single_class_predictions(
    est_cls_obj: Any,
    x_data: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Predict presence (``1``) or absence (``0``) with a fitted single-class
    (one-class or outlier detection) ``scikit-learn`` estimator, such as
    ``IsolationForest`` or ``OneClassSVM``.

    These estimators return ``+1`` for samples similar to the training data
    (inliers, here presences) and ``-1`` for outliers. This function maps
    ``-1`` to ``0`` so the result can be compared directly with ``0``/``1``
    presence/absence labels, e.g. with
    :func:`rsgispy.tools.accuracy.accuracy_scores_binary`.

    :param est_cls_obj: Fitted single-class estimator with ``predict``.
    :param x_data: Feature table, one row per sample, with the columns in the
                   order used for training.
    :return: 1D ``uint8`` array of ``0`` (absence) and ``1`` (presence).

    .. code:: python

        from sklearn.ensemble import IsolationForest
        import rsgispy.modelling.species_distribution as sdm
        from rsgispy.tools.accuracy import accuracy_scores_binary

        oc_mdl, test_acc, test_auc = sdm.fit_sklearn_sgl_cls_mdl(
            IsolationForest(random_state=42), pres_train_gdf, None,
            test_gdf, None, analysis_vars, return_model=True,
        )
        test_pred = sdm.calc_single_class_predictions(oc_mdl, test_gdf[analysis_vars])
        names, scores = accuracy_scores_binary(test_gdf["clsid"].to_numpy(), test_pred)
    """
    pred = np.asarray(est_cls_obj.predict(x_data), dtype=np.int16)
    pred = np.where(pred < 0, 0, pred)
    return pred.astype(np.uint8)


def _check_label_score_arrays(
    y_true: Union[np.ndarray, Sequence[Any]],
    scores: Union[np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return 1D ``(y_true, scores)`` arrays of equal length."""
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(scores, dtype=np.float64)
    if y_arr.ndim != 1 or s_arr.ndim != 1:
        raise rsgispy.RSGISPyException(
            "y_true and scores must be 1D arrays (one value per sample)."
        )
    if y_arr.size != s_arr.size:
        raise rsgispy.RSGISPyException(
            f"y_true ({y_arr.size} values) and scores ({s_arr.size} values) "
            "must have the same length."
        )
    if y_arr.size == 0:
        raise rsgispy.RSGISPyException("y_true and scores are empty.")
    if not np.all(np.isfinite(s_arr)):
        raise rsgispy.RSGISPyException(
            "scores contain NaN or infinite values; remove those samples first."
        )
    n_labels = np.unique(y_arr).size
    if n_labels > 2:
        raise rsgispy.RSGISPyException(
            "y_true must be binary (e.g. 1 = presence, 0 = absence) but has "
            f"{n_labels} distinct values."
        )
    return y_arr, s_arr


def calc_binary_roc_auc(
    y_true: Union[np.ndarray, Sequence[Any]],
    scores: Union[np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the receiver operating characteristic (ROC) curve and the area
    under it (AUC) for binary presence/absence labels and continuous scores,
    using ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.auc``.

    The AUC is the probability that a randomly chosen presence gets a higher
    score than a randomly chosen absence: ``0.5`` is no better than random,
    ``1`` is perfect. The curve can be plotted with :func:`plot_roc_curve` and
    a presence/absence threshold chosen with :func:`calc_binary_threshold`.

    :param y_true: 1D binary labels with ``1`` (or ``True``) for presence and
                   ``0`` (or ``-1``, ``False``) for absence.
    :param scores: 1D continuous scores, higher for presence (e.g. presence
                   probabilities from :func:`get_positive_class_probs`, or the
                   ``decision_function`` of a single-class model).
    :return: Tuple ``(fpr, tpr, roc_auc)``: the false positive rates
             (1 − specificity) and true positive rates (sensitivity) of the
             ROC curve for decreasing thresholds, and the AUC as a ``float``.
             If ``y_true`` contains only one class the curve is undefined:
             ``scikit-learn`` warns and the AUC is ``nan``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        test_y = test_gdf["clsid"].to_numpy()
        test_prob = sdm.get_positive_class_probs(rf_mdl, test_gdf[analysis_vars])
        fpr, tpr, test_auc = sdm.calc_binary_roc_auc(test_y, test_prob)
        sdm.plot_roc_curve(fpr, tpr, test_auc, "roc_curve.png")
    """
    check_sklearn_available("calc_binary_roc_auc")
    y_arr, s_arr = _check_label_score_arrays(y_true, scores)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_arr, s_arr)
    roc_auc = float(sklearn.metrics.auc(fpr, tpr))
    return fpr, tpr, roc_auc


_THRESHOLD_METHODS = ("max_tss", "sens_eq_spec")


def calc_binary_threshold(
    y_true: Union[np.ndarray, Sequence[Any]],
    scores: Union[np.ndarray, Sequence[float]],
    *,
    method: str = "max_tss",
) -> float:
    """
    Choose the threshold that converts continuous scores (e.g. presence
    probabilities) into presence/absence, from labelled samples. A sample is
    predicted present when ``score >= threshold``.

    Methods:

    * ``"max_tss"`` — maximise the true skill statistic (TSS = sensitivity +
      specificity − 1, also called Youden's J or the Hanssen-Kuiper skill
      score). This is the point of the ROC curve furthest above the diagonal.
      Unlike kappa, the TSS does not depend on prevalence.
    * ``"sens_eq_spec"`` — the threshold at which sensitivity and specificity
      are equal (as nearly as the samples allow), balancing omission and
      commission errors.

    The candidate thresholds are the distinct values of ``scores`` (the
    thresholds of ``sklearn.metrics.roc_curve``), so the result is always one
    of the scores; when several thresholds are equally good the highest is
    returned. Choose the threshold on one set of samples and report the
    accuracy on another, otherwise the accuracy is slightly optimistic.

    :param y_true: 1D binary labels with ``1`` (or ``True``) for presence and
                   ``0`` (or ``-1``, ``False``) for absence. Both classes must
                   be present.
    :param scores: 1D continuous scores, higher for presence.
    :param method: ``"max_tss"`` (Default) or ``"sens_eq_spec"``.
    :return: The threshold as a ``float``.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm
        from rsgispy.tools.accuracy import accuracy_scores_binary

        prob_thres = sdm.calc_binary_threshold(test_y, test_prob, method="max_tss")
        names, scores = accuracy_scores_binary(test_y, (test_prob >= prob_thres) * 1)
    """
    check_sklearn_available("calc_binary_threshold")
    if method not in _THRESHOLD_METHODS:
        raise rsgispy.RSGISPyException(
            f"method must be one of {_THRESHOLD_METHODS}, got {method!r}."
        )
    y_arr, s_arr = _check_label_score_arrays(y_true, scores)
    if np.unique(y_arr).size != 2:
        raise rsgispy.RSGISPyException(
            "y_true must contain both presences and absences to choose a threshold."
        )
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_arr, s_arr, drop_intermediate=False
    )
    # The first ROC point is a threshold above every score (nothing predicted
    # present); skip it so the threshold is always one of the scores.
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    if method == "max_tss":
        best_idx = int(np.argmax(tpr - fpr))
    else:
        # Sensitivity is tpr and specificity is 1 - fpr.
        best_idx = int(np.argmin(np.abs(tpr + fpr - 1.0)))
    return float(thresholds[best_idx])


def plot_roc_curve(
    fpr: Union[np.ndarray, Sequence[float]],
    tpr: Union[np.ndarray, Sequence[float]],
    roc_auc: float,
    out_file: Optional[Union[str, os.PathLike]] = None,
    *,
    label: str = "ROC Curve",
    title: str = "Receiver Operating Characteristic",
    dpi: int = 150,
) -> None:
    """
    Plot a binary ROC curve (true positive rate against false positive rate),
    with the AUC in the legend and the 1:1 line of a random model, and save it
    to ``out_file``. The Matplotlib figure is always closed afterwards.

    :param fpr: False positive rates (1 − specificity), e.g. from
                :func:`calc_binary_roc_auc`.
    :param tpr: True positive rates (sensitivity), the same length as ``fpr``.
    :param roc_auc: Area under the curve, shown in the legend.
    :param out_file: Output image file (format from the extension, e.g.
                     ``.png``); parent directories are created. If ``None``
                     the plot is displayed when running in a Jupyter notebook
                     and is otherwise discarded.
    :param label: Legend label of the curve (Default = ``'ROC Curve'``).
    :param title: Plot title (Default = ``'Receiver Operating Characteristic'``).
    :param dpi: Resolution of the saved image (Default = ``150``).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        fpr, tpr, test_auc = sdm.calc_binary_roc_auc(test_y, test_prob)
        sdm.plot_roc_curve(fpr, tpr, test_auc, "plots/roc_curve.png", label="Ensemble")
    """
    # pylint: disable=too-many-arguments
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    tpr_arr = np.asarray(tpr, dtype=np.float64)
    if fpr_arr.ndim != 1 or fpr_arr.shape != tpr_arr.shape:
        raise rsgispy.RSGISPyException(
            "fpr and tpr must be 1D arrays of the same length."
        )
    fig, ax = plt.subplots()
    try:
        ax.plot(fpr_arr, tpr_arr, label=f"{label} (AUC = {roc_auc:.3f})")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
    finally:
        finalise_plot(fig, out_file=out_file, dpi=dpi)


def env_vars_to_band_defs(
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    mask_img: Optional[str] = None,
    mask_band: int = 1,
) -> Tuple[List[BandDefn], Dict[str, str]]:
    """
    Build validated :class:`rsgispy.imageprocess.BandDefn` objects for
    ``analysis_vars`` and optional ``mask_img``, returning ``(band_defs, var_keys)``
    where ``var_keys`` maps each variable name to its internal block key.
    """
    band_defs: List[BandDefn] = []
    if mask_img is not None:
        band_defs.append(
            BandDefn(
                band_name=_MASK_BAND_KEY,
                input_img=os.fspath(mask_img),
                img_band=int(mask_band),
            )
        )

    var_keys: Dict[str, str] = {}
    for idx, var_name in enumerate(analysis_vars):
        info = lookup_env_var(env_vars, var_name)
        if info.file is None:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{var_name}' has file=None."
            )
        b_key = f"__env_{idx}__"
        var_keys[var_name] = b_key
        band_defs.append(
            BandDefn(
                band_name=b_key,
                input_img=os.fspath(info.file),
                img_band=int(info.band),
            )
        )

    return band_defs, var_keys


def build_env_block_matrix(
    block_vars: Mapping[str, np.ndarray],
    var_keys: Mapping[str, str],
    analysis_vars: Sequence[str],
    env_vars: Mapping[str, EnvVarInfo],
    datasets: Mapping[str, Any],
    img_msk_val: int,
    normalise_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble valid, finite environmental covariate rows for pixels matching
    ``img_msk_val`` inside a raster processing block, enforcing optional
    :attr:`EnvVarInfo.min_vld_val` and :attr:`EnvVarInfo.max_vld_val` bounds.

    Returns ``(valid_df_matrix, flat_valid_indices)`` where
    ``valid_df_matrix`` has shape ``(n_valid_pxls, n_vars)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    msk_flat = block_vars[_MASK_BAND_KEY].ravel()
    roi_idx = np.flatnonzero(msk_flat == img_msk_val)
    if roi_idx.size == 0:
        return np.empty((0, len(analysis_vars)), dtype=np.float64), roi_idx

    band_arrays: List[Tuple[np.ndarray, Optional[float]]] = []
    infos: List[EnvVarInfo] = []
    for var_name in analysis_vars:
        info = lookup_env_var(env_vars, var_name)
        if info.file is None:
            raise rsgispy.RSGISPyException(
                f"Environmental variable '{var_name}' has file=None."
            )
        ds = datasets[os.fspath(info.file)]
        nd_val = ds.nodatavals[info.band - 1]
        band_arrays.append((block_vars[var_keys[var_name]], nd_val))
        infos.append(info)

    env_mat, valid_idx = extract_roi_feature_matrix(band_arrays, roi_idx)
    if env_mat.shape[0] > 0:
        in_range = np.ones(env_mat.shape[0], dtype=bool)
        for col_idx, info in enumerate(infos):
            col_vals = env_mat[:, col_idx]
            if info.min_vld_val is not None:
                in_range &= col_vals >= float(info.min_vld_val)
            if info.max_vld_val is not None:
                in_range &= col_vals <= float(info.max_vld_val)
        if not np.all(in_range):
            env_mat = env_mat[in_range]
            valid_idx = valid_idx[in_range]

    if normalise_data and env_mat.shape[0] > 0:
        for col_idx, info in enumerate(infos):
            env_mat[:, col_idx] = np.asarray(
                normalise_array_with_info(env_mat[:, col_idx], info),
                dtype=np.float64,
            )
    return env_mat, valid_idx


def run_env_block_pipeline(
    in_msk_img: str,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: str,
    block_eval_fn: Callable[
        [
            List[str],
            Dict[str, str],
            Dict[str, np.ndarray],
            Mapping[str, Any],
            int,
            int,
        ],
        np.ndarray,
    ],
    gdalformat: str = "GTIFF",
    out_rio_dtype: str = "float32",
    calc_stats: bool = True,
    thematic: bool = False,
    no_data_val: Optional[float] = -1.0,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> None:
    """
    Execute a shared streaming :mod:`rsgispy.imageprocess` block pipeline
    across a mask raster and environmental covariate layers, writing output
    blocks and optionally populating image statistics and overviews.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    resolved_vars = resolve_analysis_vars(env_vars, analysis_vars)
    band_defs, var_keys = env_vars_to_band_defs(
        env_vars=env_vars,
        analysis_vars=resolved_vars,
        mask_img=in_msk_img,
        mask_band=1,
    )
    out_np_dtype = np.dtype(out_rio_dtype)
    out_path = ensure_parent_dir(output_img)

    with ExitStack() as stack:
        datasets = open_input_datasets(stack, band_defs)
        grid = compute_spatial_intersection(datasets)
        dst = open_output_dataset(
            stack=stack,
            output_img=out_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
        )
        if no_data_val is not None:
            dst.nodata = (
                int(no_data_val)
                if np.issubdtype(out_np_dtype, np.integer)
                else float(no_data_val)
            )

        for out_win, block_vars in iter_image_blocks(
            datasets=datasets,
            norm_defs=band_defs,
            grid=grid,
            block_size=block_size,
            read_dtype=np.float64,
            use_pbar=use_pbar,
        ):
            win_h = int(out_win.height)
            win_w = int(out_win.width)
            raw_block = block_eval_fn(
                resolved_vars, var_keys, block_vars, datasets, win_h, win_w
            )
            out_arr = cast_result_block(raw_block, win_h, win_w, out_np_dtype)
            dst.write(out_arr, 1, window=out_win)

    if calc_stats:
        if thematic:
            ignore_zero = no_data_val is None or float(no_data_val) == 0.0
            pop_thmt_img_stats(
                os.fspath(output_img),
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=ignore_zero,
                use_pbar=use_pbar,
            )
        else:
            pop_img_stats(
                os.fspath(output_img),
                no_data_val=float(no_data_val) if no_data_val is not None else None,
                calc_pyramids=True,
                use_pbar=use_pbar,
            )


def _resolve_pred_rio_dtype(
    datatype: Optional[int],
    out_rio_dtype: Optional[str],
    thematic: bool,
) -> str:
    """Resolve the output rasterio dtype from ``datatype`` / ``out_rio_dtype``."""
    if datatype is not None:
        rio_dtype = rsgispy.get_rasterio_datatype(int(datatype))
        if out_rio_dtype is not None and np.dtype(out_rio_dtype) != np.dtype(rio_dtype):
            raise rsgispy.RSGISPyException(
                f"datatype ({rio_dtype}) and out_rio_dtype ({out_rio_dtype}) "
                "disagree; pass only one of them."
            )
        return rio_dtype
    if out_rio_dtype is not None:
        return str(np.dtype(out_rio_dtype))
    return "uint8" if thematic else "float32"


def _check_value_fits_dtype(value: float, rio_dtype: str, arg_name: str) -> None:
    """Raise if ``value`` cannot be stored exactly in integer ``rio_dtype``."""
    np_dtype = np.dtype(rio_dtype)
    if not np.issubdtype(np_dtype, np.integer):
        return
    info = np.iinfo(np_dtype)
    val = float(value)
    if not np.isfinite(val) or val != np.floor(val) or not info.min <= val <= info.max:
        raise rsgispy.RSGISPyException(
            f"{arg_name}={value!r} cannot be stored in a {rio_dtype} raster; "
            f"use an integer between {info.min} and {info.max}."
        )


def predict_raster_from_env_vars(
    in_msk_img: Union[str, os.PathLike],
    img_msk_val: int,
    env_vars: Mapping[str, EnvVarInfo],
    analysis_vars: Sequence[str],
    output_img: Union[str, os.PathLike],
    predict_fn: Callable[[pd.DataFrame], Union[np.ndarray, pd.Series]],
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = -1.0,
    fill_val: Optional[float] = None,
    normalise_data: bool = False,
    thematic: bool = False,
    calc_stats: bool = True,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    out_rio_dtype: Optional[str] = None,
) -> None:
    """
    Apply any prediction function to the environmental variables of every
    valid pixel and write the result as a single-band raster. This is the
    engine behind the ``pred_*`` functions of this module; use it directly to
    apply a model or rule that they do not cover, e.g. a mean instead of a
    median ensemble, a regression model, a model wrapped in a pipeline, or a
    hand-written suitability index.

    The rasters are read block by block. For each block, the pixels where
    ``in_msk_img == img_msk_val`` and every variable is finite, not the
    raster's no data value and within the ``min_vld_val`` / ``max_vld_val``
    range of its :class:`EnvVarInfo` are collected into a
    :class:`pandas.DataFrame` with one column per variable, named and ordered
    as ``analysis_vars``. ``predict_fn`` is called with that table and must
    return one value per row. All other pixels get ``fill_val``.

    :param in_msk_img: Mask raster defining the area to predict (e.g. from
                       :func:`create_finite_mask`).
    :param img_msk_val: Value of the pixels in ``in_msk_img`` to predict.
    :param env_vars: Environmental variable descriptors (:class:`EnvVarInfo`),
                     keyed by variable name. All rasters must be on the same
                     grid as ``in_msk_img``.
    :param analysis_vars: Variable names, in the order the model expects
                          (the order used for training).
    :param output_img: Output raster path; parent directories are created.
    :param predict_fn: Function taking a :class:`pandas.DataFrame` of variable
                       values (one row per valid pixel) and returning a 1D
                       array or Series of the same length, e.g.
                       ``lambda df: sdm.get_positive_class_probs(mdl, df)``.
    :param gdalformat: Output GDAL format (Default = ``'GTIFF'``).
    :param datatype: Output data type as an ``rsgispy.TYPE_*`` constant.
                     Default: :data:`rsgispy.TYPE_8UINT` if ``thematic`` is
                     ``True``, otherwise :data:`rsgispy.TYPE_32FLOAT`.
                     Predictions are cast to this type (truncated for integer
                     types).
    :param no_data_val: No data value written to the output raster
                        (Default = ``-1.0``, which suits probabilities); use
                        ``None`` for none. For integer outputs it must fit the
                        data type (e.g. ``0`` or ``255`` for ``uint8``).
    :param fill_val: Value of the pixels outside the mask or with invalid
                     variables (Default = ``None``: ``no_data_val``, or ``0``
                     if that is ``None``).
    :param normalise_data: Apply the z-score normalisation stored in
                           ``env_vars`` (see :func:`pop_normalise_coeffs`)
                           before calling ``predict_fn`` (Default = ``False``).
    :param thematic: Treat the output as a thematic (class) raster: statistics
                     are calculated with a colour table
                     (Default = ``False``).
    :param calc_stats: Calculate image statistics and pyramids for the output
                       (Default = ``True``).
    :param block_size: Processing block size in pixels (Default = ``None``:
                       :func:`rsgispy.imageprocess.get_default_block_size`).
    :param use_pbar: Show a progress bar (Default = ``True``).
    :param out_rio_dtype: Output data type as a rasterio/NumPy name (e.g.
                          ``'float32'``), an alternative to ``datatype`` kept
                          for compatibility. Pass only one of the two.

    .. code:: python

        import numpy as np
        import rsgispy.modelling.species_distribution as sdm

        # Mean (rather than median) presence probability of an ensemble.
        def ens_mean_prob(env_df):
            return np.mean(
                [sdm.get_positive_class_probs(mdl, env_df) for mdl in mdls], axis=0
            )

        sdm.predict_raster_from_env_vars(
            "valid_mask.tif", 1, env_vars, analysis_vars, "sdm_mean_prob.tif",
            ens_mean_prob, no_data_val=-1.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    rio_dtype = _resolve_pred_rio_dtype(datatype, out_rio_dtype, thematic)
    if fill_val is None:
        fill_val = no_data_val if no_data_val is not None else 0
    if no_data_val is not None:
        _check_value_fits_dtype(no_data_val, rio_dtype, "no_data_val")
    _check_value_fits_dtype(fill_val, rio_dtype, "fill_val")
    fill_val = float(fill_val)

    def _eval_block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        resolved_vars: List[str],
        var_keys: Dict[str, str],
        block_vars: Dict[str, np.ndarray],
        datasets: Mapping[str, Any],
        win_h: int,
        win_w: int,
    ) -> np.ndarray:
        out_flat = np.full(win_h * win_w, fill_val, dtype=np.float64)
        env_mat, valid_idx = build_env_block_matrix(
            block_vars=block_vars,
            var_keys=var_keys,
            analysis_vars=resolved_vars,
            env_vars=env_vars,
            datasets=datasets,
            img_msk_val=img_msk_val,
            normalise_data=normalise_data,
        )
        if valid_idx.size > 0:
            env_pxl_df = pd.DataFrame(data=env_mat, columns=resolved_vars)
            preds = np.asarray(predict_fn(env_pxl_df), dtype=np.float64).ravel()
            if preds.size != valid_idx.size:
                raise rsgispy.RSGISPyException(
                    f"predict_fn returned {preds.size} values for "
                    f"{valid_idx.size} pixels; it must return one value per row."
                )
            out_flat[valid_idx] = preds
        return out_flat.reshape((win_h, win_w))

    run_env_block_pipeline(
        in_msk_img=os.fspath(in_msk_img),
        env_vars=env_vars,
        analysis_vars=analysis_vars,
        output_img=os.fspath(output_img),
        block_eval_fn=_eval_block,
        gdalformat=gdalformat,
        out_rio_dtype=rio_dtype,
        calc_stats=calc_stats,
        thematic=thematic,
        no_data_val=no_data_val,
        block_size=block_size,
        use_pbar=use_pbar,
    )
