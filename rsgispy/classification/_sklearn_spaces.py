"""
Built-in standard hyperparameter search spaces for commonly used
``scikit-learn`` classifiers, supporting Optuna Bayesian optimisation,
Grid Search, and Random Search (including ``Pipeline``-wrapped estimators).
"""

import copy
from typing import Any, Callable, Dict, Mapping, Tuple, Union

import rsgispy

_SpaceType = Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]


def _rf_et_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for RandomForestClassifier / ExtraTreesClassifier."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 25},
            "max_depth": {"type": "int", "low": 4, "high": 32},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        }
    return {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 25},
        "max_depth": {"type": "int", "low": 3, "high": 40},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.3, 0.5, 0.8, None],
        },
        "criterion": {
            "type": "categorical",
            "choices": ["gini", "entropy", "log_loss"],
        },
        "bootstrap": {"type": "categorical", "choices": [True, False]},
        "class_weight": {
            "type": "categorical",
            "choices": [None, "balanced", "balanced_subsample"],
        },
        "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.02},
    }


def _rf_et_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for RandomForest / ExtraTrees."""
    if not extended:
        return {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
    return {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [6, 12, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, None],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
    }


def _gb_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for GradientBoostingClassifier."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 400, "step": 25},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 15},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        }
    return {
        "n_estimators": {"type": "int", "low": 50, "high": 800, "step": 25},
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.3,
            "log": True,
        },
        "max_depth": {"type": "int", "low": 2, "high": 10},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 25},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.5, None],
        },
        "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.02},
    }


def _gb_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for GradientBoostingClassifier."""
    if not extended:
        return {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        }
    return {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "max_features": ["sqrt", "log2", None],
    }


def _hgb_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for HistGradientBoostingClassifier."""
    if not extended:
        return {
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_iter": {"type": "int", "low": 50, "high": 500, "step": 25},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 127},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "l2_regularization": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },
        }
    return {
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.3,
            "log": True,
        },
        "max_iter": {"type": "int", "low": 50, "high": 1000, "step": 25},
        "max_leaf_nodes": {"type": "int", "low": 15, "high": 255},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
        "l2_regularization": {
            "type": "float",
            "low": 1e-5,
            "high": 20.0,
            "log": True,
        },
        "max_bins": {"type": "int", "low": 64, "high": 255, "step": 64},
        "class_weight": {
            "type": "categorical",
            "choices": [None, "balanced"],
        },
    }


def _hgb_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for HistGradientBoostingClassifier."""
    if not extended:
        return {
            "learning_rate": [0.05, 0.1, 0.2],
            "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 40],
        }
    return {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 400, 600],
        "max_leaf_nodes": [15, 31, 63, 127],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [10, 20, 40, 80],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }


def _sample_svc_extended_optuna(trial: Any) -> Dict[str, Any]:
    """Conditional Optuna trial sampler for SVC."""
    kernel = trial.suggest_categorical(
        "kernel", ["rbf", "poly", "sigmoid", "linear"]
    )
    params: Dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-2, 1000.0, log=True),
        "kernel": kernel,
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }
    if kernel in ("rbf", "poly", "sigmoid"):
        gamma_type = trial.suggest_categorical(
            "gamma_type", ["heuristic", "numeric"]
        )
        if gamma_type == "heuristic":
            params["gamma"] = trial.suggest_categorical(
                "gamma_mode", ["scale", "auto"]
            )
        else:
            params["gamma"] = trial.suggest_float(
                "gamma_val", 1e-4, 1.0, log=True
            )
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    if kernel in ("poly", "sigmoid"):
        params["coef0"] = trial.suggest_float("coef0", 0.0, 2.0)
    return params


def _svc_optuna(extended: bool) -> _SpaceType:
    """Return Optuna space for SVC."""
    if not extended:
        return {
            "C": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
            "kernel": {
                "type": "categorical",
                "choices": ["rbf", "linear"],
            },
            "gamma": {
                "type": "categorical",
                "choices": ["scale", "auto"],
            },
            "class_weight": {
                "type": "categorical",
                "choices": [None, "balanced"],
            },
        }
    return _sample_svc_extended_optuna


def _svc_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for SVC."""
    if not extended:
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
    return {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "poly", "linear"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "degree": [2, 3, 4],
        "class_weight": [None, "balanced"],
    }


def _sample_mlp_extended_optuna(trial: Any) -> Dict[str, Any]:
    """Conditional Optuna trial sampler for MLPClassifier multi-layer grids."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_int("hidden_width", 32, 256, step=32)
    return {
        "hidden_layer_sizes": tuple([width] * n_layers),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "tanh", "logistic"]
        ),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 5e-2, log=True
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", ["constant", "adaptive"]
        ),
        "early_stopping": trial.suggest_categorical(
            "early_stopping", [False, True]
        ),
    }


def _mlp_optuna(extended: bool) -> _SpaceType:
    """Return Optuna space for MLPClassifier."""
    if not extended:
        return {
            "hidden_layer_sizes": {
                "type": "int",
                "low": 32,
                "high": 256,
                "step": 32,
            },
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh"],
            },
            "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            },
        }
    return _sample_mlp_extended_optuna


def _mlp_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for MLPClassifier."""
    if not extended:
        return {
            "hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-2],
        }
    return {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (256,),
            (128, 64),
            (256, 128, 64),
        ],
        "activation": ["relu", "tanh", "logistic"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "learning_rate_init": [5e-4, 1e-3, 5e-3, 1e-2],
        "learning_rate": ["constant", "adaptive"],
    }


def _knn_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for KNeighborsClassifier."""
    if not extended:
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 25, "step": 2},
            "weights": {
                "type": "categorical",
                "choices": ["uniform", "distance"],
            },
            "p": {"type": "int", "low": 1, "high": 2},
        }
    return {
        "n_neighbors": {"type": "int", "low": 1, "high": 51, "step": 2},
        "weights": {
            "type": "categorical",
            "choices": ["uniform", "distance"],
        },
        "p": {"type": "int", "low": 1, "high": 3},
        "leaf_size": {"type": "int", "low": 15, "high": 60, "step": 5},
    }


def _knn_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for KNeighborsClassifier."""
    if not extended:
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }
    return {
        "n_neighbors": [3, 5, 7, 11, 15, 21, 31],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "leaf_size": [20, 30, 50],
    }


def _logreg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for LogisticRegression."""
    if not extended:
        return {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
            "class_weight": {
                "type": "categorical",
                "choices": [None, "balanced"],
            },
        }
    return {
        "C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "solver": {
            "type": "categorical",
            "choices": ["lbfgs", "newton-cg", "saga"],
        },
        "fit_intercept": {"type": "categorical", "choices": [True, False]},
        "class_weight": {
            "type": "categorical",
            "choices": [None, "balanced"],
        },
        "tol": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    }


def _logreg_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for LogisticRegression."""
    if not extended:
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": [None, "balanced"],
        }
    return {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "newton-cg", "saga"],
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"],
        "tol": [1e-4, 1e-3, 1e-2],
    }


def _dt_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for DecisionTreeClassifier."""
    if not extended:
        return {
            "criterion": {
                "type": "categorical",
                "choices": ["gini", "entropy"],
            },
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 15},
        }
    return {
        "criterion": {
            "type": "categorical",
            "choices": ["gini", "entropy", "log_loss"],
        },
        "splitter": {"type": "categorical", "choices": ["best", "random"]},
        "max_depth": {"type": "int", "low": 2, "high": 40},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 25},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.5, None],
        },
        "class_weight": {
            "type": "categorical",
            "choices": [None, "balanced"],
        },
        "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.05},
    }


def _dt_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for DecisionTreeClassifier."""
    if not extended:
        return {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        }
    return {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_depth": [4, 8, 16, 24, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }


def _ada_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for AdaBoostClassifier."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 400, "step": 25},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 2.0,
                "log": True,
            },
        }
    return {
        "n_estimators": {"type": "int", "low": 50, "high": 800, "step": 25},
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 2.0,
            "log": True,
        },
    }


def _ada_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for AdaBoostClassifier."""
    if not extended:
        return {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
        }
    return {
        "n_estimators": [50, 100, 200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5],
    }


_CLASSIFIER_SPACES: Dict[
    str, Tuple[Callable[[bool], _SpaceType], Callable[[bool], Dict[str, Any]]]
] = {
    "RandomForestClassifier": (_rf_et_optuna, _rf_et_cv),
    "ExtraTreesClassifier": (_rf_et_optuna, _rf_et_cv),
    "GradientBoostingClassifier": (_gb_optuna, _gb_cv),
    "HistGradientBoostingClassifier": (_hgb_optuna, _hgb_cv),
    "SVC": (_svc_optuna, _svc_cv),
    "MLPClassifier": (_mlp_optuna, _mlp_cv),
    "KNeighborsClassifier": (_knn_optuna, _knn_cv),
    "LogisticRegression": (_logreg_optuna, _logreg_cv),
    "DecisionTreeClassifier": (_dt_optuna, _dt_cv),
    "AdaBoostClassifier": (_ada_optuna, _ada_cv),
}

_ALIASES: Dict[str, str] = {
    "randomforestclassifier": "RandomForestClassifier",
    "randomforest": "RandomForestClassifier",
    "rf": "RandomForestClassifier",
    "extratreesclassifier": "ExtraTreesClassifier",
    "extratrees": "ExtraTreesClassifier",
    "et": "ExtraTreesClassifier",
    "gradientboostingclassifier": "GradientBoostingClassifier",
    "gradientboosting": "GradientBoostingClassifier",
    "gb": "GradientBoostingClassifier",
    "gbc": "GradientBoostingClassifier",
    "histgradientboostingclassifier": "HistGradientBoostingClassifier",
    "histgradientboosting": "HistGradientBoostingClassifier",
    "hgb": "HistGradientBoostingClassifier",
    "hgbc": "HistGradientBoostingClassifier",
    "svc": "SVC",
    "svm": "SVC",
    "mlpclassifier": "MLPClassifier",
    "mlp": "MLPClassifier",
    "neuralnet": "MLPClassifier",
    "neuralnetwork": "MLPClassifier",
    "kneighborsclassifier": "KNeighborsClassifier",
    "kneighbors": "KNeighborsClassifier",
    "knn": "KNeighborsClassifier",
    "logisticregression": "LogisticRegression",
    "logreg": "LogisticRegression",
    "logistic": "LogisticRegression",
    "decisiontreeclassifier": "DecisionTreeClassifier",
    "decisiontree": "DecisionTreeClassifier",
    "dt": "DecisionTreeClassifier",
    "adaboostclassifier": "AdaBoostClassifier",
    "adaboost": "AdaBoostClassifier",
    "ada": "AdaBoostClassifier",
}

SUPPORTED_SKLEARN_CLASSIFIERS = tuple(_CLASSIFIER_SPACES.keys())


def _normalise_key(name: str) -> str:
    """Normalise classifier name/alias to lowercase alphanumeric key."""
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_classifier_and_prefix(sk_classifier: Any) -> Tuple[str, str]:
    """
    Resolve ``sk_classifier`` to ``(canonical_name, param_prefix)``,
    unwrapping nested ``Pipeline`` estimators when provided.
    """
    prefix_parts = []
    target = sk_classifier
    while (
        not isinstance(target, (str, bytes, type))
        and hasattr(target, "steps")
        and isinstance(target.steps, (list, tuple))
        and target.steps
    ):
        step_name, target = target.steps[-1]
        prefix_parts.append(str(step_name))

    if isinstance(target, str):
        raw_name = target
    elif isinstance(target, type):
        raw_name = target.__name__
    else:
        raw_name = type(target).__name__

    canonical = _ALIASES.get(_normalise_key(raw_name))
    if canonical is None:
        supported = ", ".join(SUPPORTED_SKLEARN_CLASSIFIERS)
        raise rsgispy.RSGISPyException(
            f"No built-in parameter space is defined for classifier "
            f"'{raw_name}'. Supported classifiers: {supported} (or a Pipeline "
            "whose final step is one of these). Pass an explicit param_space."
        )
    prefix = f"{'__'.join(prefix_parts)}__" if prefix_parts else ""
    return canonical, prefix


def build_sklearn_param_space(
    sk_classifier: Any,
    method: str,
    extended: bool = False,
) -> _SpaceType:
    """Build standard Optuna or Grid/Random CV search space for ``sk_classifier``."""
    canonical, prefix = resolve_classifier_and_prefix(sk_classifier)
    optuna_fn, cv_fn = _CLASSIFIER_SPACES[canonical]
    raw_space = optuna_fn(extended) if method == "optuna" else cv_fn(extended)

    if callable(raw_space):
        if not prefix:
            return raw_space

        def _prefixed_sampler(trial: Any) -> Dict[str, Any]:
            return {
                f"{prefix}{key}": val for key, val in raw_space(trial).items()
            }

        return _prefixed_sampler

    copied: Mapping[str, Any] = copy.deepcopy(raw_space)
    if not prefix:
        return dict(copied)
    return {f"{prefix}{key}": val for key, val in copied.items()}
