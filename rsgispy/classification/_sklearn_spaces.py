"""
Built-in standard hyperparameter search spaces for commonly used
``scikit-learn`` classifiers, delegating to
:mod:`rsgispy.tools._sklearn_spaces`.
"""

from typing import Any, Callable, Dict, Tuple, Union

from rsgispy.tools._sklearn_spaces import (
    SUPPORTED_SKLEARN_CLASSIFIERS,
    build_sklearn_param_space as _build_sklearn_param_space,
    resolve_estimator_and_prefix as _resolve_estimator_and_prefix,
)

_SpaceType = Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]


def resolve_classifier_and_prefix(sk_classifier: Any) -> Tuple[str, str]:
    """
    Resolve ``sk_classifier`` to ``(canonical_name, param_prefix)``,
    unwrapping nested ``Pipeline`` estimators when provided.
    """
    return _resolve_estimator_and_prefix(sk_classifier, task="classification")


def build_sklearn_param_space(
    sk_classifier: Any,
    method: str,
    extended: bool = False,
) -> _SpaceType:
    """Build standard Optuna or Grid/Random CV search space for ``sk_classifier``."""
    return _build_sklearn_param_space(
        sk_estimator=sk_classifier,
        method=method,
        extended=extended,
        task="classification",
    )


__all__ = [
    "SUPPORTED_SKLEARN_CLASSIFIERS",
    "resolve_classifier_and_prefix",
    "build_sklearn_param_space",
]
