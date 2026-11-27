"""
MaxEnt species distribution model for
:mod:`rsgispy.modelling.species_distribution`, implemented on
``scikit-learn`` without any MaxEnt package.

MaxEnt is equivalent to an L1-regularised inhomogeneous Poisson point-process
model, which can be fitted as an "infinitely weighted" logistic regression of
the presences against background points (Fithian & Hastie 2013, *Methods in
Ecology and Evolution* 4: 424-432; Renner & Warton 2013, *Biometrics* 69:
274-281). The formulation, feature classes, default feature classes and
regularisation follow ``maxnet`` (Phillips et al. 2017, "Opening the black
box: an open-source release of Maxent", *Ecography* 40: 887-893):

* features — linear, quadratic and product terms of the min-max scaled
  variables, forward and reverse hinges, thresholds, and one-hot categories;
* weights — presences ``1``, background ``bg_weight`` (``100``);
* per-feature L1 penalty — ``beta_multiplier`` × ``maxnet``'s class
  regularisation (interpolated over the number of presences) × the feature's
  standard deviation over the presences, with ``maxnet``'s floors;
* output — the fitted intercept is replaced by the normalising constant that
  makes the raw output sum to 1 over the background; cloglog output is
  ``1 - exp(-exp(link + H))`` with ``H`` the entropy of the raw output.
"""


import inspect
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.ml.cv import _iter_cv_splits
from rsgispy.modelling._helpers import (
    calc_binary_roc_auc,
    check_sklearn_available,
    read_model_vector,
)
from rsgispy.vectorprocess import VectorInput, validate_columns_exist

try:  # pragma: no cover - exercised implicitly
    from sklearn.base import (  # pylint: disable=import-error
        BaseEstimator,
        ClassifierMixin,
        TransformerMixin,
    )
    from sklearn.exceptions import ConvergenceWarning  # pylint: disable=import-error
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False

    # pylint: disable=too-few-public-methods
    class BaseEstimator:  # type: ignore[no-redef]
        """Placeholder when scikit-learn is not installed."""

    class ClassifierMixin:  # type: ignore[no-redef]
        """Placeholder when scikit-learn is not installed."""

    class TransformerMixin:  # type: ignore[no-redef]
        """Placeholder when scikit-learn is not installed."""

    class ConvergenceWarning(UserWarning):  # type: ignore[no-redef]
        """Placeholder when scikit-learn is not installed."""

    LogisticRegression = None  # type: ignore[assignment,misc]

__all__ = [
    "MaxentFeatureTransformer",
    "MaxentClassifier",
    "tune_maxent_beta",
    "maxent_default_feature_types",
    "maxent_feature_penalties",
]

_FEATURE_CODES = "lqpht"
_MIN_PRESENCES = 5
_LIBLINEAR_INTERCEPT_SCALING = 1000.0

# maxnet regularisation tables: (number of presences, class regularisation).
_REG_LINEAR = ((0.0, 10.0, 30.0, 100.0), (1.0, 1.0, 0.2, 0.05))
_REG_QUADRATIC = ((0.0, 10.0, 17.0, 30.0, 100.0), (1.3, 0.8, 0.5, 0.25, 0.05))
_REG_PRODUCT = ((0.0, 10.0, 17.0, 30.0, 100.0), (2.6, 1.6, 0.9, 0.55, 0.05))
_REG_HINGE = ((0.0, 1.0), (0.5, 0.5))
_REG_THRESHOLD = ((0.0, 100.0), (2.0, 1.0))
_REG_CATEGORICAL = ((0.0, 10.0, 17.0), (0.65, 0.5, 0.25))


def maxent_default_feature_types(n_presences: int) -> str:
    """
    Return ``maxnet``'s default feature classes for ``n_presences`` presences:
    ``"l"`` (linear) below 10, ``"lq"`` (plus quadratic) for 10-14, ``"lqh"``
    (plus hinge) for 15-79 and ``"lqph"`` (plus product) from 80 (Phillips et
    al. 2017, ``maxnet::default.features``).

    :param n_presences: Number of presence records.
    :return: Feature class codes.
    """
    n_pres = int(n_presences)
    if n_pres < 10:
        return "l"
    if n_pres < 15:
        return "lq"
    if n_pres < 80:
        return "lqh"
    return "lqph"


def _parse_feature_types(feature_types: str, n_presences: Optional[int]) -> str:
    ftypes = str(feature_types).strip().lower()
    if ftypes == "auto":
        if n_presences is None:
            raise rsgispy.RSGISPyException(
                "feature_types='auto' needs the class labels (y) to count presences."
            )
        return maxent_default_feature_types(n_presences)
    bad = sorted(set(ftypes) - set(_FEATURE_CODES))
    if not ftypes or bad:
        raise rsgispy.RSGISPyException(
            f"Invalid feature_types '{feature_types}': use 'auto' or a combination "
            f"of {list(_FEATURE_CODES)} (linear, quadratic, product, hinge, "
            "threshold)."
        )
    return "".join(c for c in _FEATURE_CODES if c in ftypes)


def _as_frame(x_data: Any, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if isinstance(x_data, pd.DataFrame):
        return x_data
    arr = np.asarray(x_data)
    if arr.ndim != 2:
        raise rsgispy.RSGISPyException(f"X must be 2D, got shape {arr.shape}.")
    cols = list(names) if names is not None else [f"x{i}" for i in range(arr.shape[1])]
    if len(cols) != arr.shape[1]:
        raise rsgispy.RSGISPyException(
            f"X has {arr.shape[1]} columns but {len(cols)} were expected."
        )
    return pd.DataFrame(arr, columns=cols)


def _fmt(val: float) -> str:
    return f"{val:.6g}"


def _fmt_cat(val: Any) -> str:
    if np.issubdtype(np.asarray(val).dtype, np.number):
        return _fmt(float(val))
    return str(val)


class MaxentFeatureTransformer(  # pylint: disable=too-many-ancestors
    TransformerMixin, BaseEstimator
):
    """
    Build MaxEnt features from environmental variables (as ``maxnet``).

    Continuous variables are min-max scaled with the range of the data passed
    to :meth:`fit` (presences and background), and give:

    * ``"l"`` linear — the scaled value ``s``;
    * ``"q"`` quadratic — ``s²``;
    * ``"p"`` product — ``s_a × s_b`` for every pair of variables;
    * ``"h"`` hinge — for knots ``k`` evenly spaced from the variable's
      minimum to maximum (``n_hinge_knots`` of them), forward hinges
      ``max(0, (x - k) / (max - k))`` (all knots but the last) and reverse
      hinges ``max(0, (k - x) / (k - min))`` (all knots but the first);
    * ``"t"`` threshold — ``x >= k`` for ``n_threshold_knots`` knots evenly
      spaced strictly inside the range.

    Categorical variables become one-hot indicator features (all classes);
    categories not seen in :meth:`fit` give all-zero indicators. With
    ``clamp=True`` continuous values are clamped to the fitted range before the
    features are built, as Maxent does when predicting.

    :param feature_types: ``"auto"`` (Default; from the number of presences, see
                          :func:`maxent_default_feature_types`) or a
                          combination of ``"l"``, ``"q"``, ``"p"``, ``"h"`` and
                          ``"t"``.
    :param n_hinge_knots: Number of hinge knots per variable (Default = ``50``).
    :param n_threshold_knots: Number of threshold knots per variable
                              (Default = ``50``).
    :param clamp: Clamp values to the fitted range in :meth:`transform`
                  (Default = ``True``).
    :param categorical_vars: Optional names (or column positions) of the
                             categorical variables.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        ftr = sdm.MaxentFeatureTransformer(feature_types="lqh").fit(x_df, y)
        feats = ftr.transform(x_df)
        names = ftr.get_feature_names_out()
    """

    # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
    def __init__(
        self,
        feature_types: str = "auto",
        n_hinge_knots: int = 50,
        n_threshold_knots: int = 50,
        clamp: bool = True,
        categorical_vars: Optional[Sequence[Union[str, int]]] = None,
    ) -> None:
        self.feature_types = feature_types
        self.n_hinge_knots = n_hinge_knots
        self.n_threshold_knots = n_threshold_knots
        self.clamp = clamp
        self.categorical_vars = categorical_vars

    def _resolve_cat_cols(self, x_df: pd.DataFrame) -> List[str]:
        if self.categorical_vars is None:
            return []
        cols = []
        for var in self.categorical_vars:
            if isinstance(var, (int, np.integer)) and var not in x_df.columns:
                cols.append(str(x_df.columns[int(var)]))
            elif var in x_df.columns:
                cols.append(var)
            else:
                raise rsgispy.RSGISPyException(
                    f"categorical variable {var!r} is not a column of X."
                )
        return cols

    def fit(self, X: Any, y: Any = None) -> "MaxentFeatureTransformer":  # noqa: N803
        """Learn variable ranges, knots and categories from ``X`` (presences
        and background); ``y`` (1 = presence) is needed for ``"auto"``."""
        # pylint: disable=invalid-name
        x_df = _as_frame(X)
        n_pres = None if y is None else int(np.sum(np.asarray(y) == 1))
        self.feature_types_ = _parse_feature_types(self.feature_types, n_pres)
        if int(self.n_hinge_knots) < 2 or int(self.n_threshold_knots) < 1:
            raise rsgispy.RSGISPyException(
                "n_hinge_knots must be >= 2 and n_threshold_knots >= 1."
            )
        self.feature_names_in_ = np.array([str(c) for c in x_df.columns], dtype=object)
        self.n_features_in_ = x_df.shape[1]
        self.cat_cols_ = self._resolve_cat_cols(x_df)
        self.cont_cols_ = [c for c in x_df.columns if c not in self.cat_cols_]
        cont = x_df[self.cont_cols_].to_numpy(dtype=np.float64)
        if cont.size and not np.all(np.isfinite(cont)):
            raise rsgispy.RSGISPyException("X contains missing or non-finite values.")
        self.mins_ = cont.min(axis=0) if cont.size else np.empty(0)
        self.maxs_ = cont.max(axis=0) if cont.size else np.empty(0)
        self.categories_ = {
            col: np.unique(x_df[col].to_numpy()) for col in self.cat_cols_
        }
        self.hinge_knots_ = [
            np.linspace(lo, hi, int(self.n_hinge_knots))
            for lo, hi in zip(self.mins_, self.maxs_)
        ]
        n_thr = int(self.n_threshold_knots)
        self.threshold_knots_ = [
            np.linspace(lo, hi, n_thr + 2)[1 : n_thr + 1]
            for lo, hi in zip(self.mins_, self.maxs_)
        ]
        self._build_feature_index()
        return self

    def _build_feature_index(self) -> None:
        names: List[str] = []
        classes: List[str] = []
        ftypes = self.feature_types_
        cont = [str(c) for c in self.cont_cols_]
        if "l" in ftypes:
            names += cont
            classes += ["l"] * len(cont)
        if "q" in ftypes:
            names += [f"{c}^2" for c in cont]
            classes += ["q"] * len(cont)
        if "p" in ftypes:
            for i_a, c_a in enumerate(cont):
                for c_b in cont[i_a + 1 :]:
                    names.append(f"{c_a}*{c_b}")
                    classes.append("p")
        if "h" in ftypes:
            for col, knots in zip(cont, self.hinge_knots_):
                names += [f"hinge_fwd({col},{_fmt(k)})" for k in knots[:-1]]
                names += [f"hinge_rev({col},{_fmt(k)})" for k in knots[1:]]
                classes += ["h"] * (2 * (knots.size - 1))
        if "t" in ftypes:
            for col, knots in zip(cont, self.threshold_knots_):
                names += [f"thr({col},{_fmt(k)})" for k in knots]
                classes += ["t"] * knots.size
        for col in self.cat_cols_:
            names += [f"cat({col}={_fmt_cat(v)})" for v in self.categories_[col]]
            classes += ["c"] * self.categories_[col].size
        self.feature_names_out_ = np.array(names, dtype=object)
        self.feature_classes_ = np.array(classes, dtype=object)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Readable feature names, e.g. ``temp``, ``temp^2``, ``temp*precip``,
        ``hinge_fwd(temp,12.3)``, ``thr(temp,12.3)``, ``cat(lc=3)``."""
        _ = input_features
        return self.feature_names_out_.copy()

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        """Return the ``(n_samples, n_features)`` feature matrix."""
        # pylint: disable=invalid-name,too-many-locals
        x_df = _as_frame(
            X,
            names=None if isinstance(X, pd.DataFrame) else list(self.feature_names_in_),
        )
        col_names = [str(k) for k in x_df.columns]
        missing = [c for c in self.feature_names_in_ if c not in col_names]
        if missing:
            raise rsgispy.RSGISPyException(f"X is missing column(s) {missing}.")
        x_df.columns = [str(c) for c in x_df.columns]
        cont = x_df[[str(c) for c in self.cont_cols_]].to_numpy(dtype=np.float64)
        if self.clamp and cont.size:
            cont = np.clip(cont, self.mins_, self.maxs_)
        span = np.where(self.maxs_ > self.mins_, self.maxs_ - self.mins_, 1.0)
        scaled = (cont - self.mins_) / span if cont.size else cont
        blocks: List[np.ndarray] = []
        ftypes = self.feature_types_
        n_rows = len(x_df)
        if "l" in ftypes:
            blocks.append(scaled)
        if "q" in ftypes:
            blocks.append(scaled**2)
        if "p" in ftypes:
            n_cont = scaled.shape[1]
            prods = [
                scaled[:, i_a] * scaled[:, i_b]
                for i_a in range(n_cont)
                for i_b in range(i_a + 1, n_cont)
            ]
            blocks.append(np.column_stack(prods) if prods else np.empty((n_rows, 0)))
        if "h" in ftypes:
            for col_i, knots in enumerate(self.hinge_knots_):
                vals = cont[:, col_i : col_i + 1]
                lo, hi = self.mins_[col_i], self.maxs_[col_i]
                fwd_k, rev_k = knots[:-1], knots[1:]
                fwd_den = np.where(hi - fwd_k > 0, hi - fwd_k, 1.0)
                rev_den = np.where(rev_k - lo > 0, rev_k - lo, 1.0)
                blocks.append(np.maximum(0.0, (vals - fwd_k) / fwd_den))
                blocks.append(np.maximum(0.0, (rev_k - vals) / rev_den))
        if "t" in ftypes:
            for col_i, knots in enumerate(self.threshold_knots_):
                blocks.append((cont[:, col_i : col_i + 1] >= knots).astype(np.float64))
        for col in self.cat_cols_:
            vals = x_df[str(col)].to_numpy()
            blocks.append((vals[:, None] == self.categories_[col][None, :]).astype(np.float64))
        if not blocks:
            return np.empty((n_rows, 0))
        return np.hstack(blocks).astype(np.float64)


def _interp_reg(table: Tuple[Tuple[float, ...], Tuple[float, ...]], n_pres: int) -> float:
    return float(np.interp(float(n_pres), table[0], table[1]))


def maxent_feature_penalties(
    feats: np.ndarray,
    y_arr: np.ndarray,
    feature_classes: Sequence[str],
) -> np.ndarray:
    """
    Per-feature L1 penalties of ``maxnet`` (before ``beta_multiplier``): the
    largest of ``0.001`` × the feature's range, a hinge floor, a threshold
    floor (``1`` for thresholds constant over the presences), and the
    feature's standard deviation over the presences × its class
    regularisation / ``sqrt(n_presences)``. Features constant over all the data
    get ``1`` (their coefficient is always ``0``).

    :param feats: Feature matrix of presences and background.
    :param y_arr: Labels (``1`` presence).
    :param feature_classes: Class code of each feature column.
    :return: Array of penalties.
    """
    pres = feats[np.asarray(y_arr) == 1]
    n_pres = pres.shape[0]
    classes = np.asarray(feature_classes)
    lqp_table = _REG_LINEAR
    if np.any(classes == "q"):
        lqp_table = _REG_QUADRATIC
    if np.any(classes == "p"):
        lqp_table = _REG_PRODUCT
    tables = {
        "l": lqp_table,
        "q": lqp_table,
        "p": lqp_table,
        "h": _REG_HINGE,
        "t": _REG_THRESHOLD,
        "c": _REG_CATEGORICAL,
    }
    class_reg = np.array([_interp_reg(tables[c], n_pres) for c in classes]) / np.sqrt(
        n_pres
    )
    sd_pres = pres.std(axis=0, ddof=1) if n_pres > 1 else np.zeros(feats.shape[1])
    f_range = feats.max(axis=0) - feats.min(axis=0)
    hinge_floor = np.where(
        classes == "h", np.maximum(sd_pres, 1.0 / np.sqrt(n_pres)) * 0.5 / np.sqrt(n_pres), 0.0
    )
    pres_sum = pres.sum(axis=0)
    thr_floor = np.where(
        (classes == "t") & ((pres_sum == 0) | (pres_sum == n_pres)), 1.0, 0.0
    )
    pen = np.maximum.reduce(
        [0.001 * f_range, hinge_floor, thr_floor, sd_pres * class_reg]
    )
    return np.where(f_range > 0, pen, 1.0)


def _l1_logistic(c_val: float, solver: str, max_iter: int, tol: float, rnd: Any) -> Any:
    """``LogisticRegression`` with an L1 penalty, for old and new scikit-learn.
    ``liblinear`` penalises the intercept like a coefficient; a large
    ``intercept_scaling`` makes that penalty negligible."""
    params = inspect.signature(LogisticRegression.__init__).parameters
    kwargs: Dict[str, Any] = {
        "C": c_val,
        "solver": solver,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": rnd,
        "fit_intercept": True,
    }
    if solver == "liblinear":
        kwargs["intercept_scaling"] = _LIBLINEAR_INTERCEPT_SCALING
    if "l1_ratio" in params and params["penalty"].default == "deprecated":
        kwargs["l1_ratio"] = 1.0
    else:
        kwargs["penalty"] = "l1"
    return LogisticRegression(**kwargs)


class MaxentClassifier(ClassifierMixin, BaseEstimator):
    """
    MaxEnt species distribution model (the ``maxnet`` formulation; see the
    module description), usable wherever this module takes a ``scikit-learn``
    classifier: :func:`fit_sklearn_mdl`, :func:`fit_kfold_sklearn_mdls`
    (including ``cv``), the ``pred_*`` functions, SHAP, response curves and
    jackknife importance.

    Fit it with presences (``y = 1``) and many background points (``y = 0``;
    around 10,000, ideally bias-weighted, see :func:`gen_pseudo_absences_smpls`).
    Do not equalise the classes: the background describes the environment
    available, not absences. Background points get weight ``bg_weight``
    (approximating MaxEnt's Poisson point-process limit), multiplied by any
    ``sample_weight`` (e.g. from :func:`calc_bias_weights`).

    Each feature ``j`` is penalised by ``beta_multiplier × λ_j`` with ``λ_j``
    from :func:`maxent_feature_penalties`. ``scikit-learn`` has one L1 strength
    for all coefficients, so the weighted lasso is fitted by dividing feature
    ``j`` by ``λ_j`` (the L1 penalty ``|w'_j|`` on the rescaled feature equals
    ``λ_j |w_j|`` on the original), with ``C = 1 / n_presences`` (``maxnet``'s
    final ``glmnet`` penalty), then dividing the coefficients by ``λ_j``.

    Outputs: :meth:`decision_function` is the link ``η`` (with the normalising
    constant, so ``exp(η)`` sums to 1 over the training background);
    :meth:`predict_raw` is ``exp(η)``; :meth:`predict_proba` column 1 is the
    ``"cloglog"`` (Default) ``1 - exp(-exp(η + H))`` or ``"logistic"``
    ``1 / (1 + exp(-(η + H)))`` output, ``H`` being the entropy of the raw
    output over the background. :meth:`predict` thresholds it at ``threshold``
    (prefer a threshold from :func:`calc_binary_threshold`).

    :param feature_types: ``"auto"`` (Default) or feature class codes
                          (see :class:`MaxentFeatureTransformer`).
    :param beta_multiplier: Regularisation multiplier (Default = ``1.0``);
                            larger values give smoother models with fewer
                            features.
    :param n_hinge_knots: Hinge knots per variable (Default = ``50``).
    :param n_threshold_knots: Threshold knots per variable (Default = ``50``).
    :param clamp: Clamp variables to the training range when predicting
                  (Default = ``True``).
    :param categorical_vars: Optional categorical variable names or positions.
    :param bg_weight: Weight of each background point (Default = ``100.0``).
    :param output: ``"cloglog"`` (Default) or ``"logistic"``.
    :param threshold: Threshold of :meth:`predict` (Default = ``0.5``).
    :param solver: ``LogisticRegression`` solver supporting L1
                   (Default = ``"liblinear"``, with a large
                   ``intercept_scaling`` so the intercept is effectively not
                   penalised). The heavily weighted background makes the
                   problem badly conditioned: ``"saga"`` converges very slowly
                   and can stop far from the optimum.
    :param max_iter: Maximum solver iterations (Default = ``10000``).
    :param tol: Solver tolerance (Default = ``1e-8``).
    :param random_state: Optional solver random state.

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm

        mdl = sdm.MaxentClassifier(beta_multiplier=1.5)
        mdl, train_acc, test_acc, auc = sdm.fit_sklearn_mdl(
            mdl, train_gdf, None, test_gdf, None, analysis_vars, return_model=True
        )
        print(mdl.selected_features_)
    """

    # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        feature_types: str = "auto",
        beta_multiplier: float = 1.0,
        n_hinge_knots: int = 50,
        n_threshold_knots: int = 50,
        clamp: bool = True,
        categorical_vars: Optional[Sequence[Union[str, int]]] = None,
        bg_weight: float = 100.0,
        output: str = "cloglog",
        threshold: float = 0.5,
        solver: str = "liblinear",
        max_iter: int = 10000,
        tol: float = 1e-8,
        random_state: Optional[int] = None,
    ) -> None:
        self.feature_types = feature_types
        self.beta_multiplier = beta_multiplier
        self.n_hinge_knots = n_hinge_knots
        self.n_threshold_knots = n_threshold_knots
        self.clamp = clamp
        self.categorical_vars = categorical_vars
        self.bg_weight = bg_weight
        self.output = output
        self.threshold = threshold
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "MaxentClassifier":  # noqa: N803
        """Fit the model to presences (``y = 1``) and background (``y = 0``)."""
        # pylint: disable=invalid-name,too-many-locals
        check_sklearn_available("MaxentClassifier")
        if str(self.output) not in ("cloglog", "logistic"):
            raise rsgispy.RSGISPyException(
                f"output must be 'cloglog' or 'logistic', got {self.output!r}."
            )
        if float(self.beta_multiplier) <= 0.0 or float(self.bg_weight) <= 0.0:
            raise rsgispy.RSGISPyException("beta_multiplier and bg_weight must be > 0.")
        y_arr = np.asarray(y).ravel()
        uniq = set(np.unique(y_arr).tolist())
        if not uniq <= {0, 1}:
            raise rsgispy.RSGISPyException(
                f"y must be binary (1 = presence, 0 = background), got {sorted(uniq)}."
            )
        y_arr = (y_arr == 1).astype(np.int64)
        n_pres = int(y_arr.sum())
        if n_pres < _MIN_PRESENCES:
            raise rsgispy.RSGISPyException(
                f"MaxEnt needs at least {_MIN_PRESENCES} presences, got {n_pres}."
            )
        if n_pres == y_arr.size:
            raise rsgispy.RSGISPyException("No background points (y = 0) were given.")
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array([str(c) for c in X.columns], dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        self.transformer_ = MaxentFeatureTransformer(
            feature_types=self.feature_types,
            n_hinge_knots=self.n_hinge_knots,
            n_threshold_knots=self.n_threshold_knots,
            clamp=self.clamp,
            categorical_vars=self.categorical_vars,
        ).fit(X, y_arr)
        self.n_features_in_ = self.transformer_.n_features_in_
        feats = self.transformer_.transform(X)
        penalties = float(self.beta_multiplier) * maxent_feature_penalties(
            feats, y_arr, self.transformer_.feature_classes_
        )
        weights = np.where(y_arr == 1, 1.0, float(self.bg_weight))
        if sample_weight is not None:
            s_wgt = np.asarray(sample_weight, dtype=np.float64).ravel()
            if s_wgt.size != y_arr.size or np.any(s_wgt < 0) or not np.all(np.isfinite(s_wgt)):
                raise rsgispy.RSGISPyException(
                    "sample_weight must be finite, >= 0 and one per sample."
                )
            weights = weights * s_wgt

        lr_mdl = _l1_logistic(
            1.0 / n_pres, self.solver, self.max_iter, self.tol, self.random_state
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lr_mdl.fit(feats / penalties, y_arr, sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            warnings.warn(
                "MaxentClassifier did not converge; increase max_iter or "
                "beta_multiplier.",
                UserWarning,
                stacklevel=2,
            )
        self.coef_ = np.asarray(lr_mdl.coef_, dtype=np.float64).ravel() / penalties
        self.penalties_ = penalties
        bg_link = feats[y_arr == 0] @ self.coef_
        link_max = float(np.max(bg_link))
        self.alpha_ = -(link_max + float(np.log(np.sum(np.exp(bg_link - link_max)))))
        raw_bg = np.exp(bg_link + self.alpha_)
        self.entropy_ = float(-np.sum(raw_bg * np.log(np.maximum(raw_bg, 1e-300))))
        self.intercept_ = np.array([self.alpha_])
        self.classes_ = np.array([0, 1])
        nz_idx = np.flatnonzero(self.coef_ != 0.0)
        self.selected_features_ = pd.DataFrame(
            {
                "feature": self.transformer_.feature_names_out_[nz_idx],
                "feature_class": self.transformer_.feature_classes_[nz_idx],
                "coef": self.coef_[nz_idx],
            }
        ).reset_index(drop=True)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise rsgispy.RSGISPyException("MaxentClassifier is not fitted yet.")

    def decision_function(self, X: Any) -> np.ndarray:  # noqa: N803
        """Link ``η``: ``exp(η)`` is the raw output (sums to 1 over the
        training background)."""
        # pylint: disable=invalid-name
        self._check_fitted()
        return self.transformer_.transform(X) @ self.coef_ + self.alpha_

    def predict_raw(self, X: Any) -> np.ndarray:  # noqa: N803
        """Raw (relative intensity) output ``exp(η)``."""
        # pylint: disable=invalid-name
        return np.exp(self.decision_function(X))

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        """``(n, 2)`` array; column 1 is the cloglog or logistic output."""
        # pylint: disable=invalid-name
        link = self.decision_function(X) + self.entropy_
        if self.output == "logistic":
            prob = 1.0 / (1.0 + np.exp(-link))
        else:
            prob = -np.expm1(-np.exp(np.minimum(link, 700.0)))
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """``1`` where the :meth:`predict_proba` output is ``>= threshold``."""
        # pylint: disable=invalid-name
        return (self.predict_proba(X)[:, 1] >= float(self.threshold)).astype(np.int64)


def _maxent_aicc(mdl: MaxentClassifier, x_df: pd.DataFrame, y_arr: np.ndarray) -> Tuple[float, int]:
    """AICc (Warren & Seifert 2011) from the raw output at the presences."""
    n_pres = int(np.sum(y_arr == 1))
    n_par = int(np.count_nonzero(mdl.coef_))
    log_lik = float(np.sum(mdl.decision_function(x_df[y_arr == 1])))
    if n_par >= n_pres - 1:
        return float("inf"), n_par
    return 2 * n_par - 2 * log_lik + 2 * n_par * (n_par + 1) / (n_pres - n_par - 1), n_par


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def tune_maxent_beta(
    train_vec_file: VectorInput,
    train_vec_lyr: Optional[str],
    analysis_vars: Sequence[str],
    *,
    beta_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 4.0),
    feature_types: Sequence[str] = ("auto",),
    cv: Optional[Any] = 5,
    metric: str = "auc",
    cls_col: str = "clsid",
    maxent_params: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Choose MaxEnt's ``beta_multiplier`` and feature classes by a grid search.

    * ``metric="auc"`` (Default) — mean out-of-fold AUC over the folds of
      ``cv`` (any splitter accepted by :func:`fit_kfold_sklearn_mdls`; an
      integer means that many :class:`rsgispy.ml.cv.SpatialBlockKFold`
      folds, using the sample coordinates). Folds lacking a class are skipped.
    * ``metric="aicc"`` — AICc (Warren & Seifert 2011) of a model fitted to all
      the samples: the number of non-zero coefficients as the parameter count
      and the log-likelihood of the presences under the raw output.

    :param train_vec_file: Presence + background samples with ``analysis_vars``
                           and ``cls_col``: path or GeoDataFrame.
    :param train_vec_lyr: Layer name (or ``None``).
    :param analysis_vars: Predictor columns.
    :param beta_multipliers: Values to try (Default = ``(0.5, 1, 2, 3, 4)``).
    :param feature_types: Feature class settings to try (Default = ``("auto",)``).
    :param cv: Cross-validation splitter or integer (``"auc"``; Default = ``5``).
    :param metric: ``"auc"`` (Default) or ``"aicc"``.
    :param cls_col: Class column (Default = ``'clsid'``).
    :param maxent_params: Other :class:`MaxentClassifier` parameters.
    :param use_pbar: Show a progress bar (Default = ``True``).
    :return: ``(results_df, best_params)``: one row per combination
             (``beta_multiplier``, ``feature_types``, ``auc_mean`` and
             ``auc_sd`` or ``aicc``, and ``n_coefs`` of a model fitted to all
             samples), and the best ``{"beta_multiplier", "feature_types"}``
             (highest AUC or lowest AICc).

    .. code:: python

        import rsgispy.modelling.species_distribution as sdm
        from rsgispy.ml.cv import SpatialBlockKFold

        res_df, best = sdm.tune_maxent_beta(
            train_gdf, None, analysis_vars, feature_types=("lq", "lqh"),
            cv=SpatialBlockKFold(n_splits=5, block_size=1000.0, stratify=True),
        )
        mdl = sdm.MaxentClassifier(**best)
    """
    check_sklearn_available("tune_maxent_beta")
    mtr = str(metric).strip().lower()
    if mtr not in ("auc", "aicc"):
        raise rsgispy.RSGISPyException(f"metric must be 'auc' or 'aicc', got {metric!r}.")
    if mtr == "auc" and cv is None:
        raise rsgispy.RSGISPyException("metric='auc' needs cv.")
    train_gdf = read_model_vector(train_vec_file, vec_lyr=train_vec_lyr)
    var_list = list(analysis_vars)
    validate_columns_exist(train_gdf, var_list + [cls_col])
    x_df = train_gdf[var_list].reset_index(drop=True)
    y_arr = (train_gdf[cls_col].to_numpy() == 1).astype(np.int64)
    splits = None
    if mtr == "auc":
        geoms = train_gdf.geometry
        pts = geoms.where(geoms.geom_type == "Point", geoms.centroid)
        coords = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])
        cv_obj = int(cv) if isinstance(cv, (int, np.integer)) else cv
        splits = _iter_cv_splits(cv_obj, x_df, y_arr, coords, None)

    base_params = dict(maxent_params or {})
    combos = [(float(b), str(f)) for f in feature_types for b in beta_multipliers]
    rows = []
    pbar = rsgispy.get_tqdm()
    for beta, ftypes in pbar(combos, disable=not use_pbar, desc="MaxEnt tuning"):
        params = {**base_params, "beta_multiplier": beta, "feature_types": ftypes}
        row: Dict[str, Any] = {"beta_multiplier": beta, "feature_types": ftypes}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            full_mdl = MaxentClassifier(**params).fit(x_df, y_arr)
            if mtr == "aicc":
                row["aicc"], row["n_coefs"] = _maxent_aicc(full_mdl, x_df, y_arr)
            else:
                row["n_coefs"] = int(np.count_nonzero(full_mdl.coef_))
                aucs = []
                for tr_idx, val_idx in splits or []:
                    if (
                        np.unique(y_arr[tr_idx]).size < 2
                        or np.unique(y_arr[val_idx]).size < 2
                        or np.sum(y_arr[tr_idx]) < _MIN_PRESENCES
                    ):
                        continue
                    mdl = MaxentClassifier(**params).fit(
                        x_df.iloc[tr_idx], y_arr[tr_idx]
                    )
                    probs = mdl.predict_proba(x_df.iloc[val_idx])[:, 1]
                    aucs.append(calc_binary_roc_auc(y_arr[val_idx], probs)[2])
                row["auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
                row["auc_sd"] = float(np.std(aucs)) if aucs else float("nan")
        rows.append(row)
    res_df = pd.DataFrame(rows)
    if mtr == "auc":
        if res_df["auc_mean"].isna().all():
            raise rsgispy.RSGISPyException(
                "No cv fold had both classes and enough presences."
            )
        best_row = res_df.loc[res_df["auc_mean"].idxmax()]
    else:
        best_row = res_df.loc[res_df["aicc"].idxmin()]
    best = {
        "beta_multiplier": float(best_row["beta_multiplier"]),
        "feature_types": str(best_row["feature_types"]),
    }
    return res_df, best
