"""
Built-in standard hyperparameter search spaces for commonly used
``scikit-learn`` classifiers and regressors, supporting Optuna Bayesian
optimisation, Grid Search, and Random Search (including ``Pipeline``,
``MultiOutputRegressor``, and ``TransformedTargetRegressor`` wrappers).
"""

import copy
from typing import Any, Callable, Dict, Mapping, Tuple, Union

import rsgispy

_SpaceType = Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]


def _rf_et_cls_optuna(extended: bool) -> Dict[str, Any]:
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


def _rf_et_cls_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for RandomForest / ExtraTrees classifiers."""
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


def _rf_et_reg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for RandomForestRegressor / ExtraTreesRegressor."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 25},
            "max_depth": {"type": "int", "low": 4, "high": 32},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", 0.5, 1.0],
            },
        }
    return {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 25},
        "max_depth": {"type": "int", "low": 3, "high": 40},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0],
        },
        "criterion": {
            "type": "categorical",
            "choices": ["squared_error", "friedman_mse", "absolute_error"],
        },
        "bootstrap": {"type": "categorical", "choices": [True, False]},
        "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.02},
    }


def _rf_et_reg_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for RandomForest / ExtraTrees regressors."""
    if not extended:
        return {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 1.0],
        }
    return {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [6, 12, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 1.0],
        "criterion": ["squared_error", "friedman_mse"],
    }


def _gb_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for GradientBoosting estimators."""
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


def _gb_reg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for GradientBoostingRegressor."""
    space = _gb_optuna(extended)
    if extended:
        space["loss"] = {
            "type": "categorical",
            "choices": ["squared_error", "absolute_error", "huber"],
        }
    return space


def _gb_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for GradientBoosting estimators."""
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


def _hgb_cls_optuna(extended: bool) -> Dict[str, Any]:
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


def _hgb_reg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for HistGradientBoostingRegressor."""
    space = _hgb_cls_optuna(extended)
    space.pop("class_weight", None)
    if extended:
        space["loss"] = {
            "type": "categorical",
            "choices": ["squared_error", "absolute_error"],
        }
    return space


def _hgb_cls_cv(extended: bool) -> Dict[str, Any]:
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


def _hgb_reg_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for HistGradientBoostingRegressor."""
    space = _hgb_cls_cv(extended)
    space.pop("class_weight", None)
    return space


def _sample_svm_extended_optuna(trial: Any, is_cls: bool) -> Dict[str, Any]:
    """Conditional Optuna trial sampler for SVC / SVR."""
    kernel = trial.suggest_categorical(
        "kernel", ["rbf", "poly", "sigmoid", "linear"]
    )
    params: Dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-2, 1000.0, log=True),
        "kernel": kernel,
    }
    if is_cls:
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )
    else:
        params["epsilon"] = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)

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
    return lambda trial: _sample_svm_extended_optuna(trial, is_cls=True)


def _svr_optuna(extended: bool) -> _SpaceType:
    """Return Optuna space for SVR."""
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
            "epsilon": {
                "type": "float",
                "low": 1e-3,
                "high": 1.0,
                "log": True,
            },
        }
    return lambda trial: _sample_svm_extended_optuna(trial, is_cls=False)


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


def _svr_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for SVR."""
    if not extended:
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.01, 0.1, 0.5],
        }
    return {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "poly", "linear"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "degree": [2, 3, 4],
        "epsilon": [0.005, 0.01, 0.1, 0.5],
    }


def _sample_mlp_extended_optuna(trial: Any) -> Dict[str, Any]:
    """Conditional Optuna trial sampler for MLPClassifier / MLPRegressor."""
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
    """Return Optuna space for MLP estimators."""
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
    """Return Grid/Random CV space for MLP estimators."""
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
    """Return Optuna space for KNeighborsClassifier / KNeighborsRegressor."""
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
    """Return Grid/Random CV space for KNeighbors estimators."""
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


def _linear_alpha_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for Ridge / Lasso."""
    if not extended:
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
        }
    return {
        "alpha": {"type": "float", "low": 1e-5, "high": 1000.0, "log": True},
        "fit_intercept": {"type": "categorical", "choices": [True, False]},
        "tol": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    }


def _linear_alpha_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for Ridge / Lasso."""
    if not extended:
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
        }
    return {
        "alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 500.0],
        "fit_intercept": [True, False],
        "tol": [1e-4, 1e-3, 1e-2],
    }


def _elasticnet_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for ElasticNet."""
    space = _linear_alpha_optuna(extended)
    space["l1_ratio"] = {"type": "float", "low": 0.05, "high": 0.95}
    return space


def _elasticnet_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for ElasticNet."""
    space = _linear_alpha_cv(extended)
    space["l1_ratio"] = [0.1, 0.3, 0.5, 0.7, 0.9]
    return space


def _dt_cls_optuna(extended: bool) -> Dict[str, Any]:
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


def _dt_cls_cv(extended: bool) -> Dict[str, Any]:
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


def _dt_reg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for DecisionTreeRegressor."""
    if not extended:
        return {
            "criterion": {
                "type": "categorical",
                "choices": ["squared_error", "friedman_mse"],
            },
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 15},
        }
    return {
        "criterion": {
            "type": "categorical",
            "choices": ["squared_error", "friedman_mse", "absolute_error"],
        },
        "splitter": {"type": "categorical", "choices": ["best", "random"]},
        "max_depth": {"type": "int", "low": 2, "high": 40},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 25},
        "max_features": {
            "type": "categorical",
            "choices": ["sqrt", "log2", 0.5, None],
        },
        "ccp_alpha": {"type": "float", "low": 0.0, "high": 0.05},
    }


def _dt_reg_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for DecisionTreeRegressor."""
    if not extended:
        return {
            "criterion": ["squared_error", "friedman_mse"],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        }
    return {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": [4, 8, 16, 24, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }


def _ada_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for AdaBoost estimators."""
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


def _ada_reg_optuna(extended: bool) -> Dict[str, Any]:
    """Return Optuna space for AdaBoostRegressor."""
    space = _ada_optuna(extended)
    if extended:
        space["loss"] = {
            "type": "categorical",
            "choices": ["linear", "square", "exponential"],
        }
    return space


def _ada_cv(extended: bool) -> Dict[str, Any]:
    """Return Grid/Random CV space for AdaBoost estimators."""
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
    "RandomForestClassifier": (_rf_et_cls_optuna, _rf_et_cls_cv),
    "ExtraTreesClassifier": (_rf_et_cls_optuna, _rf_et_cls_cv),
    "GradientBoostingClassifier": (_gb_optuna, _gb_cv),
    "HistGradientBoostingClassifier": (_hgb_cls_optuna, _hgb_cls_cv),
    "SVC": (_svc_optuna, _svc_cv),
    "MLPClassifier": (_mlp_optuna, _mlp_cv),
    "KNeighborsClassifier": (_knn_optuna, _knn_cv),
    "LogisticRegression": (_logreg_optuna, _logreg_cv),
    "DecisionTreeClassifier": (_dt_cls_optuna, _dt_cls_cv),
    "AdaBoostClassifier": (_ada_optuna, _ada_cv),
}

_REGRESSOR_SPACES: Dict[
    str, Tuple[Callable[[bool], _SpaceType], Callable[[bool], Dict[str, Any]]]
] = {
    "RandomForestRegressor": (_rf_et_reg_optuna, _rf_et_reg_cv),
    "ExtraTreesRegressor": (_rf_et_reg_optuna, _rf_et_reg_cv),
    "GradientBoostingRegressor": (_gb_reg_optuna, _gb_cv),
    "HistGradientBoostingRegressor": (_hgb_reg_optuna, _hgb_reg_cv),
    "SVR": (_svr_optuna, _svr_cv),
    "MLPRegressor": (_mlp_optuna, _mlp_cv),
    "KNeighborsRegressor": (_knn_optuna, _knn_cv),
    "Ridge": (_linear_alpha_optuna, _linear_alpha_cv),
    "Lasso": (_linear_alpha_optuna, _linear_alpha_cv),
    "ElasticNet": (_elasticnet_optuna, _elasticnet_cv),
    "DecisionTreeRegressor": (_dt_reg_optuna, _dt_reg_cv),
    "AdaBoostRegressor": (_ada_reg_optuna, _ada_cv),
}

_CLS_ALIASES: Dict[str, str] = {
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

_REG_ALIASES: Dict[str, str] = {
    "randomforestregressor": "RandomForestRegressor",
    "randomforest": "RandomForestRegressor",
    "rf": "RandomForestRegressor",
    "rfr": "RandomForestRegressor",
    "extratreesregressor": "ExtraTreesRegressor",
    "extratrees": "ExtraTreesRegressor",
    "et": "ExtraTreesRegressor",
    "etr": "ExtraTreesRegressor",
    "gradientboostingregressor": "GradientBoostingRegressor",
    "gradientboosting": "GradientBoostingRegressor",
    "gb": "GradientBoostingRegressor",
    "gbr": "GradientBoostingRegressor",
    "histgradientboostingregressor": "HistGradientBoostingRegressor",
    "histgradientboosting": "HistGradientBoostingRegressor",
    "hgb": "HistGradientBoostingRegressor",
    "hgbr": "HistGradientBoostingRegressor",
    "svr": "SVR",
    "svm": "SVR",
    "mlpregressor": "MLPRegressor",
    "mlp": "MLPRegressor",
    "neuralnet": "MLPRegressor",
    "neuralnetwork": "MLPRegressor",
    "kneighborsregressor": "KNeighborsRegressor",
    "kneighbors": "KNeighborsRegressor",
    "knn": "KNeighborsRegressor",
    "knr": "KNeighborsRegressor",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "enet": "ElasticNet",
    "decisiontreeregressor": "DecisionTreeRegressor",
    "decisiontree": "DecisionTreeRegressor",
    "dt": "DecisionTreeRegressor",
    "dtr": "DecisionTreeRegressor",
    "adaboostregressor": "AdaBoostRegressor",
    "adaboost": "AdaBoostRegressor",
    "ada": "AdaBoostRegressor",
    "adar": "AdaBoostRegressor",
}

SUPPORTED_SKLEARN_CLASSIFIERS = tuple(_CLASSIFIER_SPACES.keys())
SUPPORTED_SKLEARN_REGRESSORS = tuple(_REGRESSOR_SPACES.keys())


def _normalise_key(name: str) -> str:
    """Normalise estimator name/alias to lowercase alphanumeric key."""
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_estimator_and_prefix(
    sk_estimator: Any,
    task: str = "classification",
) -> Tuple[str, str]:
    """
    Resolve ``sk_estimator`` to ``(canonical_name, param_prefix)``,
    unwrapping nested ``Pipeline``, ``MultiOutputRegressor``, and
    ``TransformedTargetRegressor`` wrappers when provided.
    """
    prefix_parts = []
    target = sk_estimator
    while not isinstance(target, (str, bytes, type)):
        if (
            hasattr(target, "steps")
            and isinstance(target.steps, (list, tuple))
            and target.steps
        ):
            step_name, target = target.steps[-1]
            prefix_parts.append(str(step_name))
        elif (
            type(target).__name__
            in ("MultiOutputRegressor", "MultiOutputClassifier")
            and hasattr(target, "estimator")
            and target.estimator is not None
        ):
            target = target.estimator
            prefix_parts.append("estimator")
        elif hasattr(target, "regressor") and target.regressor is not None:
            target = target.regressor
            prefix_parts.append("regressor")
        else:
            break

    if isinstance(target, str):
        raw_name = target
    elif isinstance(target, type):
        raw_name = target.__name__
    else:
        raw_name = type(target).__name__

    if task == "regression":
        aliases = _REG_ALIASES
        supported = ", ".join(SUPPORTED_SKLEARN_REGRESSORS)
        role_label = "regressor"
    else:
        aliases = _CLS_ALIASES
        supported = ", ".join(SUPPORTED_SKLEARN_CLASSIFIERS)
        role_label = "classifier"

    canonical = aliases.get(_normalise_key(raw_name))
    if canonical is None:
        raise rsgispy.RSGISPyException(
            f"No built-in parameter space is defined for {role_label} "
            f"'{raw_name}'. Supported {role_label}s: {supported} (or a "
            "Pipeline / multi-output wrapper around one of these). Pass an "
            "explicit param_space."
        )
    prefix = f"{'__'.join(prefix_parts)}__" if prefix_parts else ""
    return canonical, prefix


def build_sklearn_param_space(
    sk_estimator: Any,
    method: str,
    extended: bool = False,
    task: str = "classification",
) -> _SpaceType:
    """Build standard Optuna or Grid/Random CV search space for ``sk_estimator``."""
    canonical, prefix = resolve_estimator_and_prefix(sk_estimator, task=task)
    spaces = _REGRESSOR_SPACES if task == "regression" else _CLASSIFIER_SPACES
    optuna_fn, cv_fn = spaces[canonical]
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
