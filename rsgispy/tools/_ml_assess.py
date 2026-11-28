"""
Shared internal helpers for the repeated k-fold and jackknife (leave-one-out)
assessment of estimators, reused by :mod:`rsgispy.regression.regsklearn` and
:mod:`rsgispy.classification.classsklearn`: input checks, split generation,
fold metric summaries, jackknife coefficient statistics and CSV output.
"""

import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.tools._sklearn import sklearn
from rsgispy.tools.strings import check_str

__all__ = [
    "FOLD_ID_COLS",
    "prep_assess_x",
    "prep_assess_groups",
    "take_rows",
    "group_kfold_splits",
    "repeated_kfold_splits",
    "jackknife_splits",
    "summarise_fold_metrics",
    "final_estimator",
    "coef_names",
    "jackknife_coef_stats",
    "file_stems",
    "write_named_csvs",
]

FOLD_ID_COLS = ("repeat", "fold", "n_samples")
"""Identifier columns of the per-fold metric tables (not summarised)."""

SplitIter = Iterator[Tuple[int, int, np.ndarray, np.ndarray]]


def prep_assess_x(x: Any) -> Any:
    """
    Normalise the predictors: a ``pandas.Series`` becomes a one-column
    DataFrame, a DataFrame is kept (so feature names reach the estimator) and
    anything else becomes a 2D NumPy array.
    """
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise rsgispy.RSGISPyException(
            f"x must be 1D or 2D: got an array with {x.ndim} dimensions."
        )
    return x


def prep_assess_groups(groups: Optional[Any], n_smpls: int) -> Optional[np.ndarray]:
    """Check ``groups`` is 1D with one value per sample (or ``None``)."""
    if groups is None:
        return None
    grp_arr = np.asarray(groups)
    if grp_arr.ndim != 1 or grp_arr.shape[0] != n_smpls:
        raise rsgispy.RSGISPyException(
            "groups must be 1D with one value per sample "
            f"({n_smpls}): got shape {grp_arr.shape}."
        )
    return grp_arr


def take_rows(x: Any, idx: np.ndarray) -> Any:
    """Select rows of a NumPy array or DataFrame by position."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[idx]
    return x[idx]


def group_kfold_splits(
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    rnd_seed: Optional[int],
) -> SplitIter:
    """
    Yield ``(repeat, fold, train_idx, test_idx)`` for a k-fold over groups,
    repeated with a new random order of the groups each time. The shuffled
    groups are split into ``n_splits`` folds with (nearly) the same number of
    groups, so a group is never in both the train and test samples.
    """
    uniq_grps, grp_codes = np.unique(groups, return_inverse=True)
    grp_codes = grp_codes.reshape(-1)
    n_grps = uniq_grps.shape[0]
    if n_grps < n_splits:
        raise rsgispy.RSGISPyException(
            f"There are {n_grps} groups, fewer than n_splits ({n_splits})."
        )
    rng = np.random.default_rng(rnd_seed)
    for i_rep in range(n_repeats):
        grp_fold = np.empty(n_grps, dtype=np.int64)
        for i_fold, fold_grps in enumerate(
            np.array_split(rng.permutation(n_grps), n_splits)
        ):
            grp_fold[fold_grps] = i_fold
        smpl_fold = grp_fold[grp_codes]
        for i_fold in range(n_splits):
            yield (
                i_rep,
                i_fold,
                np.flatnonzero(smpl_fold != i_fold),
                np.flatnonzero(smpl_fold == i_fold),
            )


def repeated_kfold_splits(
    n_smpls: int,
    n_splits: int,
    n_repeats: int,
    groups: Optional[np.ndarray] = None,
    rnd_seed: Optional[int] = None,
    strat_y: Optional[np.ndarray] = None,
) -> Tuple[SplitIter, int, int]:
    """
    Check ``n_splits`` / ``n_repeats`` and return ``(splits, n_splits,
    n_repeats)``, with ``splits`` yielding ``(repeat, fold, train_idx,
    test_idx)``. With ``groups`` the splits are :func:`group_kfold_splits`;
    otherwise scikit-learn's ``RepeatedStratifiedKFold`` over ``strat_y``
    when it is given, else ``RepeatedKFold``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if int(n_splits) < 2:
        raise rsgispy.RSGISPyException(f"n_splits must be >= 2: got {n_splits}.")
    if int(n_repeats) < 1:
        raise rsgispy.RSGISPyException(
            f"n_repeats must be >= 1: got {n_repeats}."
        )
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    if groups is not None:
        return (
            group_kfold_splits(groups, n_splits, n_repeats, rnd_seed),
            n_splits,
            n_repeats,
        )
    dummy_x = np.zeros((n_smpls, 1))
    if strat_y is not None:
        rkf = sklearn.model_selection.RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=rnd_seed
        )
        raw_splits = rkf.split(dummy_x, strat_y)
    else:
        rkf = sklearn.model_selection.RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=rnd_seed
        )
        raw_splits = rkf.split(dummy_x)
    splits = (
        (i_split // n_splits, i_split % n_splits, train_idx, test_idx)
        for i_split, (train_idx, test_idx) in enumerate(raw_splits)
    )
    return splits, n_splits, n_repeats


def jackknife_splits(
    n_smpls: int, groups: Optional[np.ndarray]
) -> Tuple[Iterator[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Leave-one-out (or leave-one-group-out with ``groups``) ``(train_idx,
    test_idx)`` splits and the number of left-out units (at least 3).
    """
    if groups is None:
        splitter = sklearn.model_selection.LeaveOneOut()
        n_units = n_smpls
    else:
        splitter = sklearn.model_selection.LeaveOneGroupOut()
        n_units = int(np.unique(groups).shape[0])
    if n_units < 3:
        raise rsgispy.RSGISPyException(
            f"The jackknife needs at least 3 samples (or groups): got {n_units}."
        )
    return splitter.split(np.zeros((n_smpls, 1)), groups=groups), n_units


def summarise_fold_metrics(m_df: pd.DataFrame) -> pd.DataFrame:
    """
    The mean, median, standard deviation and 5th / 95th percentiles of each
    metric column of a per-fold metric table (``nan`` ignored), indexed by
    metric.
    """
    vals = m_df.drop(
        columns=[c for c in FOLD_ID_COLS if c in m_df.columns]
    ).apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "p05": vals.quantile(0.05),
            "p95": vals.quantile(0.95),
        }
    )


def final_estimator(fitted: Any) -> Any:
    """
    The last fitted step of a pipeline (or the target regressor of a
    ``TransformedTargetRegressor``), or the estimator itself.
    """
    est = fitted
    while True:
        if hasattr(est, "steps") and est.steps:
            est = est.steps[-1][1]
        elif hasattr(est, "regressor_"):
            est = est.regressor_
        else:
            return est


def coef_names(fitted: Any, x: Any, n_coefs: int) -> List[str]:
    """Names of the coefficients (intercept first)."""
    feat_names = None
    if isinstance(x, pd.DataFrame) and x.shape[1] == n_coefs:
        feat_names = [str(c) for c in x.columns]
    elif hasattr(fitted, "steps") and len(fitted.steps) > 1:
        try:
            out_names = list(fitted[:-1].get_feature_names_out())
            if len(out_names) == n_coefs:
                feat_names = [str(c) for c in out_names]
        except (AttributeError, ValueError, TypeError):
            feat_names = None
    if feat_names is None:
        feat_names = [f"x{i}" for i in range(n_coefs)]
    return ["intercept", *feat_names]


def jackknife_coef_stats(
    full_coefs: np.ndarray,
    loo_coefs: np.ndarray,
    coef_labels: Sequence[str],
    row_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Jackknife bias, standard error and 95% interval of each coefficient
    (Efron and Tibshirani, 1993). ``full_coefs`` is (rows, coefs) and
    ``loo_coefs`` is (left-out units, rows, coefs); a row is a regression
    target or a class of a linear classifier.
    """
    import scipy.stats  # pylint: disable=import-outside-toplevel

    n_units = loo_coefs.shape[0]
    jack_mean = loo_coefs.mean(axis=0)
    bias = (n_units - 1) * (jack_mean - full_coefs)
    std_err = np.sqrt(
        (n_units - 1) / n_units * ((loo_coefs - jack_mean) ** 2).sum(axis=0)
    )
    jack_est = full_coefs - bias
    t_crit = float(scipy.stats.t.ppf(0.975, n_units - 1))
    out = {}
    for i_row, name in enumerate(row_names):
        out[name] = pd.DataFrame(
            {
                "estimate": full_coefs[i_row],
                "jack_mean": jack_mean[i_row],
                "bias": bias[i_row],
                "jack_estimate": jack_est[i_row],
                "std_err": std_err[i_row],
                "ci_lower": jack_est[i_row] - t_crit * std_err[i_row],
                "ci_upper": jack_est[i_row] + t_crit * std_err[i_row],
            },
            index=pd.Index(list(coef_labels), name="coef"),
        )
    return out


def file_stems(names: Sequence[str], default: str = "target") -> List[str]:
    """File-name safe, unique versions of target (or class) names."""
    stems = [
        check_str(
            str(name),
            rm_non_ascii=True,
            rm_dashs=True,
            rm_spaces=True,
            rm_punc=True,
        ).lower()
        or default
        for name in names
    ]
    if len(set(stems)) != len(stems):
        raise rsgispy.RSGISPyException(
            f"The names {list(names)} give the same file name "
            f"({stems}): give distinct names."
        )
    return stems


def write_named_csvs(
    out_dir: Union[str, os.PathLike],
    out_prefix: str,
    tables: Mapping[str, Mapping[str, pd.DataFrame]],
    names: Sequence[str],
    index_tables: Sequence[str] = (),
    default_stem: str = "target",
) -> None:
    """
    Write ``tables[kind][name]`` to
    ``<out_dir>/<out_prefix>_<kind>_<name stem>.csv``. The tables named in
    ``index_tables`` are written with their index.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stems = file_stems(names, default=default_stem)
    for kind, per_name in tables.items():
        for name, stem in zip(names, stems, strict=True):
            if name not in per_name:
                continue
            out_file = os.path.join(out_dir, f"{out_prefix}_{kind}_{stem}.csv")
            per_name[name].to_csv(out_file, index=kind in index_tables)
