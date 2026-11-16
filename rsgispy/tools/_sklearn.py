"""
Shared internal ``scikit-learn`` estimator cloning, fitting, and validation
helpers reused across :mod:`rsgispy.classification` and
:mod:`rsgispy.modelling`.
"""

from typing import Any, Optional

import numpy as np

import rsgispy

try:
    import sklearn.base  # pylint: disable=import-error
    import sklearn.cluster  # pylint: disable=import-error
    import sklearn.ensemble  # pylint: disable=import-error
    import sklearn.exceptions  # pylint: disable=import-error
    import sklearn.feature_selection  # pylint: disable=import-error
    import sklearn.inspection  # pylint: disable=import-error
    import sklearn.metrics  # pylint: disable=import-error
    import sklearn.model_selection  # pylint: disable=import-error
    import sklearn.utils.validation  # pylint: disable=import-error

    HAVE_SKLEARN = True
except ImportError:
    sklearn = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    HAVE_SKLEARN = False

__all__ = [
    "HAVE_SKLEARN",
    "sklearn",
    "check_sklearn_available",
    "clone_sklearn_estimator",
    "sync_sklearn_estimator_state",
    "fit_clean_sklearn_estimator",
    "check_sklearn_estimator_fitted",
    "is_sklearn_search_cv",
]


def check_sklearn_available(
    feature_name: str,
    extra: str = "stats",
) -> None:
    """Verify ``scikit-learn`` is installed before running estimator routines."""
    rsgispy.check_optional_dep(
        HAVE_SKLEARN, feature_name, "scikit-learn", extra
    )


def clone_sklearn_estimator(
    est_obj: Any,
    random_state: Optional[int] = None,
    extra: str = "stats",
) -> Any:
    """
    Create a fresh, unfitted ``scikit-learn`` estimator clone so no fitted state
    or ``warm_start`` attributes leak across runs.
    """
    check_sklearn_available("clone_sklearn_estimator", extra=extra)
    cloned = sklearn.base.clone(est_obj, safe=True)
    if random_state is not None:
        params = cloned.get_params(deep=False)
        if "random_state" in params:
            cloned.set_params(random_state=int(random_state))
    return cloned


def sync_sklearn_estimator_state(target_est: Any, fitted_est: Any) -> None:
    """
    Update ``target_est`` internal attributes with ``fitted_est`` in place.
    """
    if type(target_est) is type(fitted_est):  # pylint: disable=unidiomatic-typecheck
        target_est.__dict__.clear()
        target_est.__dict__.update(fitted_est.__dict__)


def fit_clean_sklearn_estimator(
    est_obj: Any,
    train_x: Any,
    train_y: Optional[np.ndarray] = None,
    sync_input: bool = True,
    random_state: Optional[int] = None,
    extra: str = "stats",
) -> Any:
    """
    Clone ``est_obj`` into a fresh unfitted estimator, fit on ``train_x``
    (and ``train_y`` when provided), and optionally update ``est_obj`` in place.
    """
    clean_est = clone_sklearn_estimator(
        est_obj, random_state=random_state, extra=extra
    )
    if train_y is None:
        clean_est.fit(train_x)
    else:
        clean_est.fit(train_x, train_y)
    if sync_input:
        sync_sklearn_estimator_state(est_obj, clean_est)
    return clean_est


def check_sklearn_estimator_fitted(
    estimator: Any,
    label: str = "estimator",
) -> None:
    """
    Validate that ``estimator`` exposes ``predict()`` and has been fitted.
    """
    check_sklearn_available("check_sklearn_estimator_fitted")
    if not hasattr(estimator, "predict"):
        raise rsgispy.RSGISPyException(
            f"{label} must provide a predict() method."
        )
    try:
        sklearn.utils.validation.check_is_fitted(estimator)
    except sklearn.exceptions.NotFittedError as err:
        raise rsgispy.RSGISPyException(
            f"{label} must be fitted before applying to data."
        ) from err
    except Exception:  # pylint: disable=broad-exception-caught
        # Non-BaseEstimator classifiers (e.g. custom duck-typed estimators)
        # may not follow sklearn trailing-underscore convention.
        pass


def is_sklearn_search_cv(estimator: Any) -> bool:
    """
    Return ``True`` if ``estimator`` is a ``scikit-learn`` SearchCV object.
    """
    check_sklearn_available("is_sklearn_search_cv")
    return isinstance(
        estimator,
        (
            sklearn.model_selection.GridSearchCV,
            sklearn.model_selection.RandomizedSearchCV,
        ),
    ) or (
        hasattr(estimator, "cv")
        and hasattr(estimator, "refit")
        and (
            hasattr(estimator, "param_grid")
            or hasattr(estimator, "param_distributions")
        )
    )
