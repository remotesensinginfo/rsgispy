"""
The imagecalibration.cloudmasking module provides Sentinel-2 cloud and cloud
shadow masking using the CloudSEN12 / CloudSEN12+ deep learning models
(Aybar et al., 2022, 2024) with ``torch``, ``rasterio`` and ``numpy``.

Models are available for both Top-of-Atmosphere (TOA, Level-1C) reflectance
and Surface Reflectance (SREF, Level-2A) inputs and can be run on the CPU or
a GPU (CUDA or Apple MPS). Model weights are downloaded from the Hugging Face
repository ``isp-uv-es/cloudsen12_models`` and cached locally, or can be
provided from a local directory.

The output cloud mask is a single band ``uint8`` thematic image with the
following pixel values:

* ``0`` — No data
* ``1`` — Clear
* ``2`` — Thick cloud
* ``3`` — Thin cloud
* ``4`` — Cloud shadow

.. note::

    The CloudSEN12 pre-trained model weights are released by their authors
    under the Creative Commons Attribution Non-Commercial 4.0 licence
    (CC BY-NC 4.0). Please check the licence before use and cite:

    * Aybar, C., et al. (2022). CloudSEN12, a global dataset for semantic
      understanding of cloud and cloud shadow in Sentinel-2. Scientific Data,
      9, 782.
    * Aybar, C., et al. (2024). CloudSEN12+: The largest dataset of expert-labeled
      pixels for cloud and cloud shadow detection in Sentinel-2. Data in Brief,
      56, 110852.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

import rsgispy
from rsgispy.imageprocess import (
    build_valid_cube_mask,
    create_output_from_dataset,
    ensure_file_exists,
    iter_dataset_windows,
    normalise_creation_options,
    on_output_close,
    pop_stats_on_close,
    read_window_with_halo,
    validate_band_index,
)

__all__ = [
    "CloudSEN12ModelInfo",
    "CLOUDSEN12_MODELS",
    "CLOUDSEN12_S2_L1C_BANDS",
    "CLOUDSEN12_S2_L2A_BANDS",
    "CLOUDSEN12_NO_DATA",
    "CLOUDSEN12_CLEAR",
    "CLOUDSEN12_THICK_CLOUD",
    "CLOUDSEN12_THIN_CLOUD",
    "CLOUDSEN12_CLOUD_SHADOW",
    "get_cloudsen12_weights_dir",
    "download_cloudsen12_weights",
    "get_cloudsen12_weights",
    "get_torch_device",
    "apply_cloudsen12_s2_toa_cloud_mask",
    "apply_cloudsen12_s2_sref_cloud_mask",
]

CLOUDSEN12_WEIGHTS_URL = (
    "https://huggingface.co/isp-uv-es/cloudsen12_models/resolve/main/{}"
)
"""URL template used to download CloudSEN12 model weight files."""

CLOUDSEN12_WEIGHTS_DIR_ENV = "RSGISPY_CLOUDSEN12_WEIGHTS_DIR"
"""Environment variable which can be used to override the weights cache dir."""

CLOUDSEN12_S2_L1C_BANDS: Tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)
"""Sentinel-2 Level-1C (TOA) band order expected by the CloudSEN12 models."""

CLOUDSEN12_S2_L2A_BANDS: Tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
"""Sentinel-2 Level-2A (SREF) band order expected by the CloudSEN12 L2A model."""

CLOUDSEN12_NO_DATA = 0
"""Output pixel value for no-data pixels."""
CLOUDSEN12_CLEAR = 1
"""Output pixel value for clear pixels."""
CLOUDSEN12_THICK_CLOUD = 2
"""Output pixel value for thick cloud pixels."""
CLOUDSEN12_THIN_CLOUD = 3
"""Output pixel value for thin cloud pixels."""
CLOUDSEN12_CLOUD_SHADOW = 4
"""Output pixel value for cloud shadow pixels."""

_CLOUDSEN12_CLR_TAB: Dict[int, Tuple[int, int, int, int]] = {
    CLOUDSEN12_NO_DATA: (0, 0, 0, 0),
    CLOUDSEN12_CLEAR: (139, 64, 0, 255),
    CLOUDSEN12_THICK_CLOUD: (220, 220, 220, 255),
    CLOUDSEN12_THIN_CLOUD: (180, 180, 180, 255),
    CLOUDSEN12_CLOUD_SHADOW: (60, 60, 60, 255),
}

_MODEL_DIVISOR = 32


@dataclass(frozen=True)
class CloudSEN12ModelInfo:
    """
    Description of a CloudSEN12 cloud masking model.

    :param name: Model name (key within :data:`CLOUDSEN12_MODELS`).
    :param weights_file: File name of the model weights on Hugging Face.
    :param bands: Sentinel-2 bands (in order) the model expects as input.
    :param product: Input product type: ``'toa'`` (Level-1C) or ``'sref'``
                    (Level-2A).
    :param model_type: ``'weights'`` (a ``segmentation_models_pytorch`` UNet
                       state dict) or ``'jit'`` (a TorchScript model).
    """

    name: str
    weights_file: str
    bands: Tuple[str, ...]
    product: str
    model_type: str


CLOUDSEN12_MODELS: Dict[str, CloudSEN12ModelInfo] = {
    "UNetMobV2_V1": CloudSEN12ModelInfo(
        name="UNetMobV2_V1",
        weights_file="UNetMobV2_V1.pt",
        bands=CLOUDSEN12_S2_L1C_BANDS,
        product="toa",
        model_type="weights",
    ),
    "UNetMobV2_V2": CloudSEN12ModelInfo(
        name="UNetMobV2_V2",
        weights_file="UNetMobV2_V2.pt",
        bands=CLOUDSEN12_S2_L1C_BANDS,
        product="toa",
        model_type="weights",
    ),
    "cloudsen12l2a": CloudSEN12ModelInfo(
        name="cloudsen12l2a",
        weights_file="cloudsen12-s2-l2a-1.0.0.pt",
        bands=CLOUDSEN12_S2_L2A_BANDS,
        product="sref",
        model_type="jit",
    ),
}
"""Registry of the supported CloudSEN12 / CloudSEN12+ Sentinel-2 models."""


def _get_model_info(model_name: str) -> CloudSEN12ModelInfo:
    """Look up a model in :data:`CLOUDSEN12_MODELS` or raise."""
    if model_name not in CLOUDSEN12_MODELS:
        raise rsgispy.RSGISPyException(
            f"Unknown CloudSEN12 model '{model_name}'. Available models: "
            f"{sorted(CLOUDSEN12_MODELS)}."
        )
    return CLOUDSEN12_MODELS[model_name]


def _import_torch(feature_name: str) -> Any:
    """Import ``torch`` with an optional dependency guard."""
    try:
        import torch  # pylint: disable=import-outside-toplevel,import-error
    except ImportError as exc:
        rsgispy.check_optional_dep(
            False, feature_name, "torch", "cloudmask", error_detail=str(exc)
        )
        raise
    return torch


def get_cloudsen12_weights_dir() -> str:
    """
    Get the default local directory used to cache the CloudSEN12 model weights.

    The directory is (in order of precedence) the value of the
    ``RSGISPY_CLOUDSEN12_WEIGHTS_DIR`` environment variable,
    ``$XDG_CACHE_HOME/rsgispy/cloudsen12`` or ``~/.cache/rsgispy/cloudsen12``.
    The directory is not created by this function.

    :return: Path to the weights cache directory.

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        print(cldmsk.get_cloudsen12_weights_dir())
    """
    env_dir = os.environ.get(CLOUDSEN12_WEIGHTS_DIR_ENV)
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(cache_root, "rsgispy", "cloudsen12")


def download_cloudsen12_weights(
    model_name: str = "UNetMobV2_V2",
    out_dir: Optional[str] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
    timeout: float = 60.0,
) -> str:
    """
    Download the weights for a CloudSEN12 model from Hugging Face
    (``isp-uv-es/cloudsen12_models``) into a local directory. If the weights
    file already exists it is not downloaded again unless ``overwrite=True``.

    :param model_name: Name of the model (a key within
                       :data:`CLOUDSEN12_MODELS`; Default = ``'UNetMobV2_V2'``).
    :param out_dir: Directory in which to save the weights. If ``None``, the
                    default cache directory is used (see
                    :func:`get_cloudsen12_weights_dir`).
    :param overwrite: If ``True``, re-download the weights even if a local
                      copy exists (Default = ``False``).
    :param use_pbar: If ``True``, display a tqdm progress bar (Default = ``True``).
    :param timeout: HTTP request timeout in seconds (Default = ``60.0``).
    :return: Path to the local weights file.

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        weights_file = cldmsk.download_cloudsen12_weights("UNetMobV2_V2")
    """
    model_info = _get_model_info(model_name)
    eff_out_dir = (
        get_cloudsen12_weights_dir()
        if out_dir is None
        else os.path.abspath(os.path.expanduser(out_dir))
    )
    os.makedirs(eff_out_dir, exist_ok=True)
    out_file = os.path.join(eff_out_dir, model_info.weights_file)
    if os.path.isfile(out_file) and not overwrite:
        return out_file

    try:
        import requests  # pylint: disable=import-outside-toplevel,import-error
    except ImportError as exc:
        rsgispy.check_optional_dep(
            False,
            "download_cloudsen12_weights",
            "requests",
            "cloudmask",
            error_detail=str(exc),
        )
        raise
    # pylint: disable=import-outside-toplevel
    from rsgispy.tools._download_common import stream_http_download

    url = CLOUDSEN12_WEIGHTS_URL.format(model_info.weights_file)
    if use_pbar:
        print(f"Downloading CloudSEN12 '{model_name}' weights from: {url}")
    stream_http_download(
        requests.Session(),
        url,
        out_file,
        timeout=timeout,
        use_pbar=use_pbar,
    )
    return out_file


def get_cloudsen12_weights(
    model_name: str = "UNetMobV2_V2",
    weights_dir: Optional[str] = None,
    download: bool = True,
    use_pbar: bool = True,
) -> str:
    """
    Get the path to the local weights file for a CloudSEN12 model, downloading
    the weights if they are not available locally and ``download=True``.

    :param model_name: Name of the model (a key within
                       :data:`CLOUDSEN12_MODELS`; Default = ``'UNetMobV2_V2'``).
    :param weights_dir: Local directory containing (or to download) the model
                        weights, or a direct path to the weights file. If
                        ``None``, the default cache directory is used (see
                        :func:`get_cloudsen12_weights_dir`).
    :param download: If ``True``, download the weights when they are not
                     found locally; otherwise raise an exception
                     (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar when downloading
                     (Default = ``True``).
    :return: Path to the local weights file.

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        weights_file = cldmsk.get_cloudsen12_weights(
            "UNetMobV2_V2", weights_dir="./cloudsen12_weights"
        )
    """
    model_info = _get_model_info(model_name)
    if weights_dir is not None and os.path.isfile(weights_dir):
        return os.path.abspath(weights_dir)

    eff_dir = (
        get_cloudsen12_weights_dir()
        if weights_dir is None
        else os.path.abspath(os.path.expanduser(weights_dir))
    )
    weights_file = os.path.join(eff_dir, model_info.weights_file)
    if os.path.isfile(weights_file):
        return weights_file
    if not download:
        raise rsgispy.RSGISPyException(
            f"The CloudSEN12 '{model_name}' weights file "
            f"('{model_info.weights_file}') was not found in '{eff_dir}' and "
            "download=False."
        )
    return download_cloudsen12_weights(
        model_name=model_name, out_dir=eff_dir, use_pbar=use_pbar
    )


def get_torch_device(device: str = "cpu") -> Any:
    """
    Resolve a device specification into a ``torch.device``, checking that the
    requested device is available.

    :param device: Device specification: ``'cpu'``, ``'cuda'``,
                   ``'cuda:<index>'`` (NVIDIA GPU), ``'mps'`` (Apple Silicon
                   GPU) or ``'auto'`` to select CUDA, then MPS, then CPU
                   depending on availability (Default = ``'cpu'``).
    :return: ``torch.device`` instance.

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        device = cldmsk.get_torch_device("auto")
    """
    torch = _import_torch("get_torch_device")
    dev_str = str(device).strip().lower()
    has_mps = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    if dev_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if dev_str == "cpu":
        return torch.device("cpu")
    if dev_str == "gpu" or dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise rsgispy.RSGISPyException(
                f"Device '{device}' was requested but CUDA is not available."
            )
        tdev = torch.device("cuda" if dev_str == "gpu" else dev_str)
        if tdev.index is not None and tdev.index >= torch.cuda.device_count():
            raise rsgispy.RSGISPyException(
                f"CUDA device index {tdev.index} is not available "
                f"({torch.cuda.device_count()} device(s) found)."
            )
        return tdev
    if dev_str == "mps":
        if not has_mps:
            raise rsgispy.RSGISPyException(
                "Device 'mps' was requested but Apple MPS is not available."
            )
        return torch.device("mps")
    raise rsgispy.RSGISPyException(
        f"Unknown device '{device}'; expected 'cpu', 'cuda', 'cuda:<n>', "
        "'mps' or 'auto'."
    )


def _load_cloudsen12_model(
    model_info: CloudSEN12ModelInfo, weights_file: str, device: Any
) -> Any:
    """Load a CloudSEN12 model onto ``device`` in evaluation mode."""
    torch = _import_torch("CloudSEN12 cloud masking")
    if model_info.model_type == "jit":
        model = torch.jit.load(weights_file, map_location=device)
    else:
        try:
            # pylint: disable=import-outside-toplevel,import-error
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            rsgispy.check_optional_dep(
                False,
                "CloudSEN12 cloud masking",
                "segmentation-models-pytorch",
                "cloudmask",
                error_detail=str(exc),
            )
            raise
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=len(model_info.bands),
            classes=4,
        )
        weights = torch.load(weights_file, map_location=device, weights_only=True)
        if isinstance(weights, Mapping) and "state_dict" in weights:
            weights = {
                k.removeprefix("model."): v
                for k, v in weights["state_dict"].items()
            }
        model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model


def _build_torch_predictor(
    model: Any, device: Any
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a torch model as ``f((C, H, W) float32) -> (H, W) uint8`` in ``[0, 3]``."""
    torch = _import_torch("CloudSEN12 cloud masking")

    def _predict(arr: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(arr[np.newaxis])).to(
                device
            )
            pred = model(tensor)
            if pred.ndim == 4:
                pred = torch.argmax(pred, dim=1)
            return pred[0].to("cpu").numpy().astype(np.uint8)

    return _predict


def _pad_to_divisor(
    arr: np.ndarray, divisor: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Symmetric-pad the spatial dims of ``(C, H, W)`` to a multiple of ``divisor``."""
    height, width = int(arr.shape[1]), int(arr.shape[2])
    pad_h = (divisor * math.ceil(height / divisor)) - height
    pad_w = (divisor * math.ceil(width / divisor)) - width
    if pad_h == 0 and pad_w == 0:
        return arr, (height, width)
    padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
    return padded, (height, width)


def _run_cloud_mask_inference(
    input_img: str,
    output_img: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    img_bands: Sequence[int],
    scale_factor: float,
    add_offset: float,
    no_data_val: Optional[float],
    tile_size: int,
    tile_overlap: int,
    gdalformat: str,
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Dict[str, Any]],
) -> None:
    """Run tiled cloud mask inference over ``input_img`` writing ``output_img``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(input_img, "r"))
        for bidx in img_bands:
            validate_band_index(int(bidx), src.count, input_img)
        eff_nd = no_data_val if no_data_val is not None else src.nodata

        dst = create_output_from_dataset(
            stack=stack,
            src=src,
            output_img=output_img,
            gdalformat=gdalformat,
            out_rio_dtype="uint8",
            count=1,
            creation_opts=creation_opts,
        )
        dst.nodata = CLOUDSEN12_NO_DATA
        dst.set_band_description(1, "CloudSEN12")
        # Statistics, then the class colour table, once the writer has closed
        # (for a COG output, before the temporary GeoTIFF is converted).
        pop_stats_on_close(
            dst,
            calc_stats,
            no_data_val=CLOUDSEN12_NO_DATA,
            thematic=True,
            use_pbar=use_pbar,
        )
        on_output_close(dst, _write_cloud_mask_colour_table)

        band_list = [int(b) for b in img_bands]
        for win in iter_dataset_windows(
            src.height, src.width, block_size=tile_size, use_pbar=use_pbar
        ):
            win_h, win_w = int(win.height), int(win.width)
            blk = read_window_with_halo(
                src,
                band_list,
                win,
                pad=tile_overlap,
                out_dtype=np.float32,
                pad_mode="symmetric",
            )
            rows = slice(tile_overlap, tile_overlap + win_h)
            cols = slice(tile_overlap, tile_overlap + win_w)
            core = blk[:, rows, cols]
            valid = build_valid_cube_mask(core, no_data_val=eff_nd)
            out_arr = np.full((win_h, win_w), CLOUDSEN12_NO_DATA, dtype=np.uint8)
            if np.any(valid):
                blk_valid = build_valid_cube_mask(blk, no_data_val=eff_nd)
                refl = (blk + np.float32(add_offset)) / np.float32(scale_factor)
                refl[:, ~blk_valid] = 0.0
                refl_pad, _ = _pad_to_divisor(
                    refl.astype(np.float32, copy=False), _MODEL_DIVISOR
                )
                pred = np.asarray(predict_fn(refl_pad))
                pred_core = pred[rows, cols]
                out_arr[valid] = (pred_core[valid].astype(np.uint8) + 1).clip(
                    CLOUDSEN12_CLEAR, CLOUDSEN12_CLOUD_SHADOW
                )
            dst.write(out_arr, 1, window=win)


def _write_cloud_mask_colour_table(output_img: str) -> None:
    """Write the CloudSEN12 class colour table onto band 1 of ``output_img``."""
    try:
        with rasterio.open(output_img, "r+") as dst:
            dst.write_colormap(1, _CLOUDSEN12_CLR_TAB)
            dst.colorinterp = [ColorInterp.palette]
    except Exception:  # pylint: disable=broad-exception-caught
        # Some formats do not support colour tables; this is non-fatal.
        pass


def _apply_cloudsen12_cloud_mask(
    input_img: str,
    output_img: str,
    product: str,
    model_name: str,
    img_bands: Optional[Sequence[int]],
    weights_dir: Optional[str],
    download_weights: bool,
    device: str,
    scale_factor: float,
    add_offset: float,
    no_data_val: Optional[float],
    tile_size: int,
    tile_overlap: int,
    gdalformat: str,
    calc_stats: bool,
    use_pbar: bool,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]],
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    """Shared implementation for the TOA and SREF CloudSEN12 functions."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img)
    model_info = _get_model_info(model_name)
    if model_info.product != product:
        valid_names = sorted(
            n for n, m in CLOUDSEN12_MODELS.items() if m.product == product
        )
        raise rsgispy.RSGISPyException(
            f"Model '{model_name}' is for '{model_info.product}' input data, not "
            f"'{product}'. Use one of: {valid_names}."
        )
    sf_val = float(scale_factor)
    if not math.isfinite(sf_val) or sf_val <= 0.0:
        raise rsgispy.RSGISPyException(
            f"scale_factor must be finite and > 0, got {scale_factor!r}."
        )
    off_val = float(add_offset)
    if not math.isfinite(off_val):
        raise rsgispy.RSGISPyException(
            f"add_offset must be finite, got {add_offset!r}."
        )
    tile_sz = int(tile_size)
    if tile_sz < _MODEL_DIVISOR:
        raise rsgispy.RSGISPyException(
            f"tile_size must be >= {_MODEL_DIVISOR}, got {tile_size!r}."
        )
    overlap = int(tile_overlap)
    if overlap < 0:
        raise rsgispy.RSGISPyException(
            f"tile_overlap must be >= 0, got {tile_overlap!r}."
        )

    n_model_bands = len(model_info.bands)
    if img_bands is None:
        with rasterio.open(input_img, "r") as src:
            n_img_bands = src.count
        if n_img_bands != n_model_bands:
            raise rsgispy.RSGISPyException(
                f"Model '{model_name}' requires {n_model_bands} bands "
                f"({', '.join(model_info.bands)}) but '{input_img}' has "
                f"{n_img_bands} bands; use img_bands to select/order the bands."
            )
        band_idxs = list(range(1, n_model_bands + 1))
    else:
        band_idxs = [int(b) for b in img_bands]
        if len(band_idxs) != n_model_bands:
            raise rsgispy.RSGISPyException(
                f"Model '{model_name}' requires {n_model_bands} bands "
                f"({', '.join(model_info.bands)}) but {len(band_idxs)} were "
                "specified in img_bands."
            )

    norm_opts = normalise_creation_options(creation_opts)

    if predict_fn is None:
        weights_file = get_cloudsen12_weights(
            model_name=model_name,
            weights_dir=weights_dir,
            download=download_weights,
            use_pbar=use_pbar,
        )
        torch_device = get_torch_device(device)
        if use_pbar:
            print(
                f"Running CloudSEN12 '{model_name}' cloud masking on "
                f"device '{torch_device}'."
            )
        model = _load_cloudsen12_model(model_info, weights_file, torch_device)
        predict_fn = _build_torch_predictor(model, torch_device)

    _run_cloud_mask_inference(
        input_img=input_img,
        output_img=output_img,
        predict_fn=predict_fn,
        img_bands=band_idxs,
        scale_factor=sf_val,
        add_offset=off_val,
        no_data_val=no_data_val,
        tile_size=tile_sz,
        tile_overlap=overlap,
        gdalformat=gdalformat,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=norm_opts,
    )


def apply_cloudsen12_s2_toa_cloud_mask(
    input_img: str,
    output_img: str,
    model_name: str = "UNetMobV2_V2",
    img_bands: Optional[Sequence[int]] = None,
    weights_dir: Optional[str] = None,
    download_weights: bool = True,
    device: str = "cpu",
    scale_factor: float = 10000.0,
    add_offset: float = 0.0,
    no_data_val: Optional[float] = 0.0,
    tile_size: int = 512,
    tile_overlap: int = 32,
    *,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Apply a CloudSEN12 / CloudSEN12+ cloud and cloud shadow masking model to a
    Sentinel-2 Top-of-Atmosphere (TOA, Level-1C) reflectance image.

    The input image must contain the 13 Sentinel-2 bands (resampled to a
    common resolution, typically 10 m) in the order
    ``B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12``, or
    ``img_bands`` must be provided to select those bands (in that order) from
    the input image. Pixel values are converted to reflectance (``0–1``) as
    ``(pxl + add_offset) / scale_factor``.

    The image is processed in tiles of ``tile_size`` pixels, with each tile
    expanded by ``tile_overlap`` pixels of context on every side to avoid tile
    edge artefacts. The output is a ``uint8`` thematic image where
    ``0`` = no data, ``1`` = clear, ``2`` = thick cloud, ``3`` = thin cloud and
    ``4`` = cloud shadow.

    :param input_img: Path to the input Sentinel-2 TOA reflectance image.
    :param output_img: Path to the output cloud mask image.
    :param model_name: CloudSEN12 TOA model name: ``'UNetMobV2_V2'`` (trained
                       on CloudSEN12+) or ``'UNetMobV2_V1'``
                       (Default = ``'UNetMobV2_V2'``).
    :param img_bands: Optional sequence of 13 1-based band indices within
                      ``input_img`` giving the bands in the order expected by
                      the model. If ``None``, the input image must have exactly
                      13 bands in the expected order.
    :param weights_dir: Local directory containing (or into which to download)
                        the model weights, or a path to the weights file. If
                        ``None``, the default cache directory is used (see
                        :func:`get_cloudsen12_weights_dir`).
    :param download_weights: If ``True``, download the model weights if they
                             are not available locally (Default = ``True``).
    :param device: Device used to run the model: ``'cpu'``, ``'cuda'``,
                   ``'cuda:<n>'``, ``'mps'`` or ``'auto'`` (Default = ``'cpu'``).
    :param scale_factor: Divisor converting input pixel values to reflectance
                         (Default = ``10000.0``).
    :param add_offset: Offset added to the input pixel values before scaling,
                       e.g. ``-1000`` for raw ESA L1C data with processing
                       baseline >= 04.00 (Default = ``0.0``).
    :param no_data_val: Input no-data value; pixels where all bands equal this
                        value are output as no data. If ``None``, the input
                        image header no-data value is used (Default = ``0.0``).
    :param tile_size: Tile size in pixels for processing (Default = ``512``).
    :param tile_overlap: Number of context pixels added around each tile
                         (Default = ``32``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars and progress
                     messages (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        cldmsk.apply_cloudsen12_s2_toa_cloud_mask(
            "s2_l1c_toa_10m.tif",
            "s2_l1c_clouds.tif",
            model_name="UNetMobV2_V2",
            device="cuda",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _apply_cloudsen12_cloud_mask(
        input_img=input_img,
        output_img=output_img,
        product="toa",
        model_name=model_name,
        img_bands=img_bands,
        weights_dir=weights_dir,
        download_weights=download_weights,
        device=device,
        scale_factor=scale_factor,
        add_offset=add_offset,
        no_data_val=no_data_val,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        gdalformat=gdalformat,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )


def apply_cloudsen12_s2_sref_cloud_mask(
    input_img: str,
    output_img: str,
    model_name: str = "cloudsen12l2a",
    img_bands: Optional[Sequence[int]] = None,
    weights_dir: Optional[str] = None,
    download_weights: bool = True,
    device: str = "cpu",
    scale_factor: float = 10000.0,
    add_offset: float = 0.0,
    no_data_val: Optional[float] = 0.0,
    tile_size: int = 512,
    tile_overlap: int = 32,
    *,
    gdalformat: str = "GTIFF",
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> None:
    """
    Apply the CloudSEN12 cloud and cloud shadow masking model to a Sentinel-2
    Surface Reflectance (SREF, Level-2A) image.

    The input image must contain the 12 Sentinel-2 L2A bands (resampled to a
    common resolution, typically 10 m) in the order
    ``B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12``, or
    ``img_bands`` must be provided to select those bands (in that order) from
    the input image. Pixel values are converted to reflectance (``0–1``) as
    ``(pxl + add_offset) / scale_factor``.

    .. note::

        The 10-band stack produced by
        :func:`rsgispy.imagecalibration.sensorlvl2data.create_stacked_sen2_esa_l2a_img`
        does not include bands B01 and B09, so cannot be used directly as input.

    The image is processed in tiles of ``tile_size`` pixels, with each tile
    expanded by ``tile_overlap`` pixels of context on every side to avoid tile
    edge artefacts. The output is a ``uint8`` thematic image where
    ``0`` = no data, ``1`` = clear, ``2`` = thick cloud, ``3`` = thin cloud and
    ``4`` = cloud shadow.

    :param input_img: Path to the input Sentinel-2 surface reflectance image.
    :param output_img: Path to the output cloud mask image.
    :param model_name: CloudSEN12 SREF model name
                       (Default = ``'cloudsen12l2a'``).
    :param img_bands: Optional sequence of 12 1-based band indices within
                      ``input_img`` giving the bands in the order expected by
                      the model. If ``None``, the input image must have exactly
                      12 bands in the expected order.
    :param weights_dir: Local directory containing (or into which to download)
                        the model weights, or a path to the weights file. If
                        ``None``, the default cache directory is used (see
                        :func:`get_cloudsen12_weights_dir`).
    :param download_weights: If ``True``, download the model weights if they
                             are not available locally (Default = ``True``).
    :param device: Device used to run the model: ``'cpu'``, ``'cuda'``,
                   ``'cuda:<n>'``, ``'mps'`` or ``'auto'`` (Default = ``'cpu'``).
    :param scale_factor: Divisor converting input pixel values to reflectance
                         (Default = ``10000.0``).
    :param add_offset: Offset added to the input pixel values before scaling,
                       e.g. ``-1000`` for raw ESA L2A data with processing
                       baseline >= 04.00 (Default = ``0.0``).
    :param no_data_val: Input no-data value; pixels where all bands equal this
                        value are output as no data. If ``None``, the input
                        image header no-data value is used (Default = ``0.0``).
    :param tile_size: Tile size in pixels for processing (Default = ``512``).
    :param tile_overlap: Number of context pixels added around each tile
                         (Default = ``32``).
    :param gdalformat: Output GDAL driver format (Default = ``'GTIFF'``).
    :param calc_stats: If ``True``, populate image statistics and pyramids
                       (Default = ``True``).
    :param use_pbar: If ``True``, display tqdm progress bars and progress
                     messages (Default = ``True``).
    :param creation_opts: Optional GDAL creation options (dict or sequence of
                          ``'KEY=VALUE'``).

    .. code:: python

        import rsgispy.imagecalibration.cloudmasking as cldmsk

        cldmsk.apply_cloudsen12_s2_sref_cloud_mask(
            "s2_l2a_sref_10m.tif",
            "s2_l2a_clouds.tif",
            device="auto",
            weights_dir="./cloudsen12_weights",
            gdalformat="GTIFF",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    _apply_cloudsen12_cloud_mask(
        input_img=input_img,
        output_img=output_img,
        product="sref",
        model_name=model_name,
        img_bands=img_bands,
        weights_dir=weights_dir,
        download_weights=download_weights,
        device=device,
        scale_factor=scale_factor,
        add_offset=add_offset,
        no_data_val=no_data_val,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        gdalformat=gdalformat,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
        creation_opts=creation_opts,
    )
