"""
Shared Gradient Boosted Decision Tree (GBDT) GPU detection/configuration,
built-in Optuna hyperparameter spaces, parameter merging, and model
serialisation helpers reused across :mod:`rsgispy.classification` and
:mod:`rsgispy.regression`.
"""

import inspect
import os
import shutil
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import warnings

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.tools.files import ensure_parent_dir

try:
    import xgboost as xgb  # pylint: disable=import-error

    HAVE_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAVE_XGBOOST = False

try:
    import lightgbm as lgb  # pylint: disable=import-error

    HAVE_LIGHTGBM = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    HAVE_LIGHTGBM = False

try:
    import catboost as cb  # pylint: disable=import-error
    from catboost.utils import (  # pylint: disable=import-error
        get_gpu_device_count as _cb_get_gpu_device_count,
    )

    HAVE_CATBOOST = True
except ImportError:
    cb = None  # type: ignore[assignment]
    _cb_get_gpu_device_count = None  # type: ignore[assignment]
    HAVE_CATBOOST = False

try:
    import joblib  # pylint: disable=import-error

    HAVE_JOBLIB = True
except ImportError:
    joblib = None  # type: ignore[assignment]
    HAVE_JOBLIB = False

OptunaSpaceType = Union[Dict[str, Any], Callable[[Any], Mapping[str, Any]]]


def validate_gpu_device_id(gpu_device_id: Optional[int]) -> Optional[int]:
    """Validate and coerce an optional non-negative GPU device index."""
    if gpu_device_id is None:
        return None
    if isinstance(gpu_device_id, bool) or not isinstance(
        gpu_device_id, (int, np.integer)
    ):
        raise rsgispy.RSGISPyException(
            f"gpu_device_id must be a non-negative integer or None, "
            f"got {gpu_device_id!r}."
        )
    dev_idx = int(gpu_device_id)
    if dev_idx < 0:
        raise rsgispy.RSGISPyException(
            f"gpu_device_id must be non-negative, got {gpu_device_id}."
        )
    return dev_idx


def resolve_gbdt_gpu(
    backend_name: str,
    gpu_check_fn: Callable[[Optional[int]], bool],
    use_gpu: bool,
    gpu_device_id: Optional[int],
) -> Tuple[bool, Optional[int]]:
    """
    Validate GPU device selection and resolve whether GPU execution is active,
    falling back to CPU with a ``UserWarning`` when unavailable.
    """
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not use_gpu:
        return False, None
    if gpu_check_fn(dev_idx):
        return True, dev_idx
    warnings.warn(
        f"GPU requested for {backend_name} (use_gpu=True), but no compatible "
        f"GPU device or GPU-enabled build was detected. Falling back to CPU.",
        UserWarning,
        stacklevel=3,
    )
    return False, None


# ---------------------------------------------------------------------------
# XGBoost shared helpers
# ---------------------------------------------------------------------------


def is_xgboost_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """Check whether CUDA GPU execution is supported and available for XGBoost."""
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_XGBOOST:
        return False
    try:
        build_info = xgb.build_info()
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    if not bool(build_info.get("USE_CUDA", False)):
        return False
    device = "cuda" if dev_idx is None else f"cuda:{dev_idx}"
    probe_x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]],
        dtype=np.float32,
    )
    probe_y = np.array([0, 1, 0, 1], dtype=np.int32)
    try:
        xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device=device,
            verbosity=0,
        ).fit(probe_x, probe_y)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def get_xgboost_optuna_param_space(
    extended: bool = False,
    binary: bool = True,
    task: str = "classification",
) -> Dict[str, Any]:
    """Return built-in Optuna hyperparameter search space for XGBoost."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 25},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "gamma": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },
        }

    space: Dict[str, Any] = {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 25},
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.3,
            "log": True,
        },
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "min_child_weight": {
            "type": "float",
            "low": 0.5,
            "high": 20.0,
            "log": True,
        },
        "gamma": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
        "colsample_bylevel": {"type": "float", "low": 0.4, "high": 1.0},
        "colsample_bynode": {"type": "float", "low": 0.4, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-5, "high": 20.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-5, "high": 20.0, "log": True},
        "max_delta_step": {"type": "int", "low": 0, "high": 10},
        "max_bin": {"type": "int", "low": 64, "high": 512, "step": 64},
        "grow_policy": {
            "type": "categorical",
            "choices": ["depthwise", "lossguide"],
        },
        "max_leaves": {"type": "int", "low": 0, "high": 128},
    }
    if task == "classification" and binary:
        space["scale_pos_weight"] = {
            "type": "float",
            "low": 0.5,
            "high": 10.0,
            "log": True,
        }
    return space


def apply_xgb_gpu_params(
    params: Dict[str, Any],
    gpu_device_id: Optional[int],
) -> Dict[str, Any]:
    """Configure XGBoost parameters for CUDA GPU histogram training."""
    out = dict(params)
    out.pop("gpu_id", None)
    out["tree_method"] = "hist"
    out["device"] = (
        "cuda" if gpu_device_id is None else f"cuda:{int(gpu_device_id)}"
    )
    return out


def prepare_xgb_predict_estimator(
    estimator: Any,
    gpu_active: bool,
    gpu_device_id: Optional[int],
) -> Tuple[Any, Dict[str, Any], Optional[Callable[[], None]]]:
    """Temporarily configure XGBoost inference device and restore afterwards."""
    if not hasattr(estimator, "get_params") or not hasattr(
        estimator, "set_params"
    ):
        return estimator, {}, None
    orig_device = estimator.get_params().get("device", None)
    target_device: Optional[str] = None
    if gpu_active:
        target_device = (
            "cuda" if gpu_device_id is None else f"cuda:{int(gpu_device_id)}"
        )
    elif isinstance(orig_device, str) and orig_device.strip().lower().startswith(
        ("cuda", "gpu")
    ):
        target_device = "cpu"

    if target_device is not None and orig_device != target_device:
        estimator.set_params(device=target_device)

        def _restore() -> None:
            estimator.set_params(device=orig_device)

        return estimator, {}, _restore
    return estimator, {}, None


def fit_xgb_estimator(
    est: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """Fit ``XGBClassifier`` or ``XGBRegressor`` with optional early stopping."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if valid_x is not None and valid_y is not None and (
        early_stopping_rounds is not None
        or getattr(est, "early_stopping_rounds", None) is not None
    ):
        est.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            verbose=False,
        )
    else:
        est.fit(train_x, train_y)
    return est


def save_xgb_or_cb_model(
    model: Any,
    out_file: Union[str, os.PathLike],
    have_pkg: bool,
    pkg_name: str,
    extra: str,
    func_name: str,
) -> None:
    """Save XGBoost or CatBoost model using joblib or native ``save_model()``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(have_pkg, func_name, pkg_name, extra)
    out_path = ensure_parent_dir(out_file)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".joblib", ".pkl"):
        rsgispy.check_optional_dep(HAVE_JOBLIB, func_name, "joblib", extra)
        joblib.dump(model, out_path)
    else:
        model.save_model(out_path)


def load_xgb_or_cb_model(
    in_file: Union[str, os.PathLike],
    model_factory: Callable[[], Any],
    have_pkg: bool,
    pkg_name: str,
    extra: str,
    func_name: str,
    label: str,
) -> Any:
    """Load XGBoost or CatBoost model from joblib or native file format."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(have_pkg, func_name, pkg_name, extra)
    in_path = os.fspath(in_file)
    ensure_file_exists(in_path, label=label)
    ext = os.path.splitext(in_path)[1].lower()
    if ext in (".joblib", ".pkl"):
        rsgispy.check_optional_dep(HAVE_JOBLIB, func_name, "joblib", extra)
        return joblib.load(in_path)
    mdl = model_factory()
    mdl.load_model(in_path)
    return mdl


# ---------------------------------------------------------------------------
# LightGBM shared helpers
# ---------------------------------------------------------------------------


def detect_lightgbm_gpu_device(
    gpu_device_id: Optional[int] = None,
) -> Optional[str]:
    """Detect operational LightGBM GPU backend (``'cuda'`` or ``'gpu'``)."""
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_LIGHTGBM:
        return None
    has_nvidia = (
        shutil.which("nvidia-smi") is not None
        or os.path.exists("/dev/nvidia0")
        or bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    )
    candidates = ("cuda", "gpu") if has_nvidia else ("gpu",)
    probe_x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]],
        dtype=np.float32,
    )
    probe_y = np.array([0, 1, 0, 1], dtype=np.int32)
    for dev_name in candidates:
        kwargs: Dict[str, Any] = {
            "n_estimators": 1,
            "min_child_samples": 1,
            "min_data_in_bin": 1,
            "device": dev_name,
            "verbose": -1,
        }
        if dev_idx is not None:
            kwargs["gpu_device_id"] = dev_idx
        try:
            lgb.LGBMClassifier(**kwargs).fit(probe_x, probe_y)
            return dev_name
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return None


def is_lightgbm_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """Return True if CUDA or OpenCL GPU training is available for LightGBM."""
    return detect_lightgbm_gpu_device(gpu_device_id) is not None


def get_lightgbm_optuna_param_space(
    extended: bool = False,
    task: str = "classification",
) -> Dict[str, Any]:
    """Return built-in Optuna hyperparameter search space for LightGBM."""
    if not extended:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 25},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "min_child_samples": {"type": "int", "low": 5, "high": 50},
            "min_split_gain": {
                "type": "float",
                "low": 1e-4,
                "high": 1.0,
                "log": True,
            },
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "subsample_freq": 1,
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },
        }

    space: Dict[str, Any] = {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 25},
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.3,
            "log": True,
        },
        "num_leaves": {"type": "int", "low": 8, "high": 255},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "min_child_samples": {"type": "int", "low": 3, "high": 100},
        "min_child_weight": {
            "type": "float",
            "low": 1e-4,
            "high": 10.0,
            "log": True,
        },
        "min_split_gain": {
            "type": "float",
            "low": 1e-4,
            "high": 5.0,
            "log": True,
        },
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "subsample_freq": {"type": "int", "low": 1, "high": 7},
        "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-5, "high": 20.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-5, "high": 20.0, "log": True},
        "max_bin": {"type": "int", "low": 63, "high": 511, "step": 64},
        "subsample_for_bin": {
            "type": "int",
            "low": 20000,
            "high": 300000,
            "step": 20000,
        },
        "path_smooth": {"type": "float", "low": 0.0, "high": 10.0},
        "extra_trees": {"type": "categorical", "choices": [False, True]},
    }
    if task == "classification":
        space["class_weight"] = {
            "type": "categorical",
            "choices": [None, "balanced"],
        }
    return space


def apply_lgb_gpu_params(
    params: Dict[str, Any],
    gpu_device_id: Optional[int],
) -> Dict[str, Any]:
    """Configure LightGBM parameters for CUDA or OpenCL GPU training."""
    out = dict(params)
    out.pop("device_type", None)
    device = detect_lightgbm_gpu_device(gpu_device_id) or "gpu"
    out["device"] = device
    if gpu_device_id is not None:
        out["gpu_device_id"] = int(gpu_device_id)
    if device == "gpu" and "max_bin" in out and out["max_bin"] is not None:
        out["max_bin"] = min(int(out["max_bin"]), 255)
    return out


def fit_lgb_estimator(
    est: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """Fit ``LGBMClassifier`` or ``LGBMRegressor`` with optional early stopping."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if valid_x is not None and valid_y is not None and early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(int(early_stopping_rounds), verbose=False)]
        fit_sig = inspect.signature(est.fit).parameters
        if "eval_X" in fit_sig and "eval_y" in fit_sig:
            est.fit(
                train_x,
                train_y,
                eval_X=valid_x,
                eval_y=valid_y,
                callbacks=callbacks,
            )
        else:
            est.fit(
                train_x,
                train_y,
                eval_set=[(valid_x, valid_y)],
                callbacks=callbacks,
            )
    else:
        est.fit(train_x, train_y)
    return est


def save_joblib_model(
    model: Any,
    out_file: Union[str, os.PathLike],
    have_pkg: bool,
    pkg_name: str,
    extra: str,
    func_name: str,
) -> None:
    """Save a joblib-serialised GBDT model after verifying package dependencies."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(have_pkg, func_name, pkg_name, extra)
    rsgispy.check_optional_dep(HAVE_JOBLIB, func_name, "joblib", extra)
    out_path = ensure_parent_dir(out_file)
    joblib.dump(model, out_path)


def load_joblib_model(
    in_file: Union[str, os.PathLike],
    have_pkg: bool,
    pkg_name: str,
    extra: str,
    func_name: str,
    label: str,
) -> Any:
    """Load a joblib-serialised GBDT model after verifying dependencies."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    rsgispy.check_optional_dep(have_pkg, func_name, pkg_name, extra)
    rsgispy.check_optional_dep(HAVE_JOBLIB, func_name, "joblib", extra)
    in_path = os.fspath(in_file)
    ensure_file_exists(in_path, label=label)
    return joblib.load(in_path)


# ---------------------------------------------------------------------------
# CatBoost shared helpers
# ---------------------------------------------------------------------------


def is_catboost_gpu_available(gpu_device_id: Optional[int] = None) -> bool:
    """Check whether CUDA GPU execution is supported and available for CatBoost."""
    dev_idx = validate_gpu_device_id(gpu_device_id)
    if not HAVE_CATBOOST or _cb_get_gpu_device_count is None:
        return False
    try:
        n_gpus = int(_cb_get_gpu_device_count())
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    if dev_idx is not None:
        return n_gpus > dev_idx
    return n_gpus > 0


def sample_extended_cb_optuna_params(
    trial: Any,
    include_class_weights: bool = True,
) -> Dict[str, Any]:
    """Sample extended CatBoost parameters respecting conditional constraints."""
    params: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 50, 1000, step=25),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.3, log=True
        ),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-3, 10.0, log=True
        ),
        "rsm": trial.suggest_float("rsm", 0.4, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 10
        ),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "model_size_reg": trial.suggest_float("model_size_reg", 0.0, 2.0),
        "score_function": trial.suggest_categorical(
            "score_function", ["Cosine", "L2"]
        ),
    }
    if include_class_weights:
        params["auto_class_weights"] = trial.suggest_categorical(
            "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
        )

    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
    )
    params["bootstrap_type"] = bootstrap_type
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 2.0
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params["grow_policy"] = grow_policy
    if grow_policy == "SymmetricTree":
        params["boosting_type"] = trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        )
    elif grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 128)

    return params


def get_catboost_optuna_param_space(
    extended: bool = False,
    task: str = "classification",
) -> OptunaSpaceType:
    """Return built-in Optuna hyperparameter search space for CatBoost."""
    if extended:
        include_cw = task == "classification"
        return lambda trial: sample_extended_cb_optuna_params(
            trial, include_class_weights=include_cw
        )

    return {
        "iterations": {"type": "int", "low": 50, "high": 500, "step": 25},
        "learning_rate": {
            "type": "float",
            "low": 0.01,
            "high": 0.3,
            "log": True,
        },
        "depth": {"type": "int", "low": 3, "high": 8},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
        "random_strength": {
            "type": "float",
            "low": 1e-3,
            "high": 10.0,
            "log": True,
        },
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
        "rsm": {"type": "float", "low": 0.5, "high": 1.0},
        "border_count": {"type": "int", "low": 32, "high": 255},
    }


def apply_cb_gpu_params(
    params: Dict[str, Any],
    gpu_device_id: Optional[int],
) -> Dict[str, Any]:
    """Configure CatBoost parameters for GPU training and strip CPU-only options."""
    out = dict(params)
    out["task_type"] = "GPU"
    if gpu_device_id is not None:
        out["devices"] = str(int(gpu_device_id))
    out.pop("rsm", None)
    if out.get("bootstrap_type") == "MVS":
        out["bootstrap_type"] = "Bernoulli"
    return out


def prepare_cb_predict_estimator(
    estimator: Any,
    gpu_active: bool,
    _gpu_device_id: Optional[int],
) -> Tuple[Any, Dict[str, Any], Optional[Callable[[], None]]]:
    """Pass ``task_type='GPU'`` via predict kwargs without mutating fitted models."""
    if gpu_active:
        return estimator, {"task_type": "GPU"}, None
    return estimator, {}, None


def fit_cb_estimator(
    est: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: Optional[pd.DataFrame],
    valid_y: Optional[np.ndarray],
    early_stopping_rounds: Optional[int],
) -> Any:
    """Fit ``CatBoostClassifier`` or ``CatBoostRegressor`` with optional early stopping."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if valid_x is not None and valid_y is not None and early_stopping_rounds is not None:
        est.fit(
            train_x,
            train_y,
            eval_set=(valid_x, valid_y),
            early_stopping_rounds=int(early_stopping_rounds),
            verbose=False,
        )
    else:
        est.fit(train_x, train_y)
    return est


# ---------------------------------------------------------------------------
# Generic parameter resolution & merging
# ---------------------------------------------------------------------------


def merge_gbdt_params(
    backend_name: str,
    task_defaults: Dict[str, Any],
    seed_param: str,
    jobs_param: str,
    apply_gpu_params: Callable[[Dict[str, Any], Optional[int]], Dict[str, Any]],
    base_estimator: Optional[Any],
    explicit_params: Optional[Mapping[str, Any]],
    rnd_seed: Optional[int],
    n_jobs: Optional[int],
    early_stopping_rounds: Optional[int],
    has_valid: bool,
    gpu_active: bool = False,
    gpu_device_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build clean GBDT estimator parameter dictionary with task defaults."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    merged: Dict[str, Any] = dict(task_defaults)

    if base_estimator is not None and hasattr(base_estimator, "get_params"):
        base_params = {
            k: v
            for k, v in base_estimator.get_params(deep=False).items()
            if v is not None
        }
        merged.update(base_params)
        for k, v in task_defaults.items():
            if k not in base_params or base_params[k] is None:
                merged[k] = v

    if explicit_params:
        merged.update(dict(explicit_params))

    if rnd_seed is not None:
        if seed_param == "random_seed":
            merged.pop("random_state", None)
        merged[seed_param] = int(rnd_seed)

    if n_jobs is not None:
        if jobs_param == "thread_count":
            merged.pop("n_jobs", None)
        merged[jobs_param] = int(n_jobs)

    if not has_valid:
        merged.pop("early_stopping_rounds", None)
    elif early_stopping_rounds is not None and backend_name == "XGBoost":
        merged["early_stopping_rounds"] = int(early_stopping_rounds)

    if gpu_active:
        merged = apply_gpu_params(merged, gpu_device_id)

    return merged


def resolve_gbdt_optuna_space(
    default_space_fn: Callable[[], OptunaSpaceType],
    extended_space_fn: Callable[[], OptunaSpaceType],
    param_space: Optional[Union[str, OptunaSpaceType]],
    extended_param_space: bool,
    fixed_params: Optional[Mapping[str, Any]],
) -> Tuple[OptunaSpaceType, str]:
    """Resolve effective GBDT Optuna trial search space and mode name."""
    use_extended = bool(extended_param_space)
    custom_space: Optional[Any] = None

    if isinstance(param_space, str):
        preset = param_space.strip().lower()
        if preset in ("default", "key", "standard"):
            if extended_param_space:
                raise rsgispy.RSGISPyException(
                    "Cannot combine param_space='default' with "
                    "extended_param_space=True."
                )
            use_extended = False
        elif preset in ("extended", "full", "large"):
            use_extended = True
        else:
            raise rsgispy.RSGISPyException(
                f"Unsupported param_space preset '{param_space}'. Expected "
                "'default' or 'extended', or pass a dict or callable."
            )
    elif param_space is not None:
        if extended_param_space:
            raise rsgispy.RSGISPyException(
                "Provide either a custom param_space or "
                "extended_param_space=True, not both."
            )
        custom_space = param_space

    if custom_space is not None:
        eff_space: OptunaSpaceType = custom_space
        space_mode = "custom"
    elif use_extended:
        eff_space = extended_space_fn()
        space_mode = "extended"
    else:
        eff_space = default_space_fn()
        space_mode = "default"

    if not callable(eff_space) and fixed_params:
        trial_space: OptunaSpaceType = {
            k: v for k, v in eff_space.items() if k not in fixed_params
        }
    else:
        trial_space = eff_space

    return trial_space, space_mode
