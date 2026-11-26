"""
Golden reference cases for the morphology (``rsgispy.imagemorphology``),
classification post-processing (``rsgispy.classification.classpostprocess``)
and raster prediction (``rsgispy.ml.predict``) file functions.

The reference outputs in ``tests/data/golden/`` (``morph_*.npz``,
``postproc_*.npz`` and ``predict_*.npz``) were generated with the block-loop
implementations, before these functions were rebuilt on the tile graph; the
test ``tests/test_golden_morph_class_ml.py`` checks that the current code
still produces them (integers exactly, floats within 1 ulp) for several
block sizes, including ones that do not divide the image.

Regenerate (only when an output change is intended) with::

    python tests/golden_morph_class_ml.py
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "golden"
#: Block sizes the references are checked at (16 and 23 do not divide the
#: 37 x 45 test image; 256 holds it in one block).
BLOCK_SIZES = (16, 23, 256)
HEIGHT, WIDTH = 37, 45
GROUPS = ("morph", "postproc", "predict")

CaseFunc = Callable[[Dict[str, Any], Path], List[str]]


# --------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------


def _write(
    path: Path,
    data: np.ndarray,
    nodata: Any = None,
    descriptions: Any = None,
    colormap: Any = None,
) -> str:
    arr = data if data.ndim == 3 else data[np.newaxis]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(arr.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(500000.0, 6000000.0, 10.0, 10.0),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if descriptions is not None:
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, desc)
        if colormap is not None:
            dst.write_colormap(1, colormap)
    return str(path)


def _classes(rng: np.random.Generator, n_cls: int) -> np.ndarray:
    """Blocky class map with salt-and-pepper noise (classes ``1..n_cls``)."""
    coarse = rng.integers(1, n_cls + 1, size=(7, 8))
    full = np.kron(coarse, np.ones((6, 6), dtype=np.int64))[:HEIGHT, :WIDTH]
    noise = rng.random((HEIGHT, WIDTH)) < 0.15
    full[noise] = rng.integers(1, n_cls + 1, size=int(noise.sum()))
    return full


def make_inputs(tmp_dir: Path) -> Dict[str, Any]:
    """Write the synthetic input rasters and return their paths (and data)."""
    # pylint: disable=too-many-locals
    rng = np.random.default_rng(20261126)
    shape = (HEIGHT, WIDTH)
    inp: Dict[str, Any] = {}

    u8 = rng.integers(0, 60, size=shape).astype(np.uint8)
    u8[rng.random(shape) < 0.05] = 0
    inp["u8"] = _write(tmp_dir / "u8.tif", u8, nodata=0, descriptions=["grey"])

    i16 = rng.integers(-200, 1000, size=(2, *shape)).astype(np.int16)
    i16[rng.random((2, *shape)) < 0.05] = -1
    inp["i16"] = _write(tmp_dir / "i16.tif", i16, nodata=-1)

    f32 = (rng.normal(100.0, 40.0, size=(3, *shape))).astype(np.float32)
    f32[rng.random((3, *shape)) < 0.04] = np.nan
    f32[rng.random((3, *shape)) < 0.04] = -9999.0
    f32[:, 20:24, 30:34] = -9999.0
    inp["f32"] = _write(
        tmp_dir / "f32.tif", f32, nodata=-9999.0, descriptions=["r", "g", "b"]
    )

    f32n = (rng.random(shape) * 50.0).astype(np.float32)
    f32n[rng.random(shape) < 0.06] = np.nan
    inp["f32nan"] = _write(tmp_dir / "f32nan.tif", f32n, nodata=float("nan"))

    cmap = {
        0: (0, 0, 0, 0),
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (0, 0, 255, 255),
        4: (255, 255, 0, 255),
        5: (0, 255, 255, 255),
    }
    cls = _classes(rng, 5).astype(np.uint8)
    cls[rng.random(shape) < 0.04] = 0
    cls[30:, :6] = 0
    inp["cls"] = _write(
        tmp_dir / "cls.tif", cls, nodata=0, descriptions=["classes"], colormap=cmap
    )
    cls2 = np.stack([_classes(rng, 4), _classes(rng, 6)]).astype(np.int16)
    inp["cls2"] = _write(tmp_dir / "cls2.tif", cls2, descriptions=["a", "b"])

    # Feature rasters for prediction.
    feat_a = (rng.random((3, *shape)) * 100.0).astype(np.float32)
    feat_a[rng.random((3, *shape)) < 0.03] = -9999.0
    feat_a[1, rng.random(shape) < 0.02] = np.nan
    inp["feat_a"] = _write(tmp_dir / "feat_a.tif", feat_a, nodata=-9999.0)
    feat_b = rng.integers(0, 50, size=shape).astype(np.int16)
    feat_b[rng.random(shape) < 0.03] = -1
    inp["feat_b"] = _write(tmp_dir / "feat_b.tif", feat_b, nodata=-1)
    msk = np.ones(shape, dtype=np.uint8)
    msk[:24, :24] = 0  # first tiles (block 16 / 23) have no ROI pixels
    msk[rng.random(shape) < 0.05] = 0
    inp["mask"] = _write(tmp_dir / "mask.tif", msk, nodata=0)

    # pylint: disable=import-outside-toplevel
    import rsgispy.imagemorphology as rimorph

    rimorph.create_circular_op(str(tmp_dir / "circ5"), op_size=5)
    inp["circ5"] = str(tmp_dir / "circ5.gmtxt")
    inp["models"] = _make_models(rng)
    return inp


def _feature_stack(inp: Dict[str, Any]) -> Any:
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.stack import BandSpec, ImageFeatureStack

    return ImageFeatureStack(
        [
            BandSpec(inp["feat_a"], name="a"),
            BandSpec(inp["feat_b"], name="b", min_vld_val=1.0, max_vld_val=45.0),
        ]
    )


class _FixedLinear:
    """Deterministic point regressor ``X @ coef + intercept``."""

    def __init__(self, coef: Any, intercept: float) -> None:
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)

    def predict(self, x_in: Any) -> np.ndarray:
        return np.asarray(x_in, dtype=np.float64) @ self.coef + self.intercept


def _make_models(rng: np.random.Generator) -> Dict[str, Any]:
    """Small fixed-seed models trained on synthetic samples."""
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from rsgispy.ml.scaling import FeatureScaler

    n_s = 300
    x_tr = np.column_stack(
        [rng.random((n_s, 3)) * 100.0, rng.integers(1, 46, size=n_s).astype(float)]
    )
    y_cls = 1 + (x_tr[:, 0] > 50).astype(int) + 2 * (x_tr[:, 3] > 22).astype(int)
    y_bin = ((x_tr[:, 1] + x_tr[:, 2]) > 100.0).astype(int)
    y_reg = 0.3 * x_tr[:, 0] - 0.2 * x_tr[:, 1] + 0.5 * x_tr[:, 3]
    y_reg2 = np.column_stack([y_reg, x_tr[:, 2] - x_tr[:, 0]])

    scaler = FeatureScaler(method="standard_scaler").fit(x_tr)
    x_sc = scaler.transform(x_tr)
    names = ["a_b1", "a_b2", "a_b3", "b_b1"]
    return {
        "clf": DecisionTreeClassifier(max_depth=4, random_state=0).fit(x_tr, y_cls),
        "bin": DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_tr, y_bin),
        "rf": RandomForestRegressor(
            n_estimators=5, max_depth=5, random_state=0
        ).fit(x_tr, y_reg),
        "rf2": RandomForestRegressor(
            n_estimators=4, max_depth=4, random_state=1
        ).fit(x_tr, y_reg2),
        "scaler": scaler,
        "clf_sc": DecisionTreeClassifier(max_depth=4, random_state=0).fit(x_sc, y_cls),
        "df_reg": DecisionTreeRegressor(max_depth=5, random_state=0).fit(
            pd.DataFrame(x_tr, columns=names), y_reg
        ),
        "lin": _FixedLinear([0.3, -0.2, 0.01, 0.5], 1.5),
        "q_lo": DecisionTreeRegressor(max_depth=3, random_state=0).fit(
            x_tr, y_reg - 5.0
        ),
        "q_hi": DecisionTreeRegressor(max_depth=3, random_state=0).fit(
            x_tr, y_reg + 5.0
        ),
        "residuals": np.abs(rng.normal(0.0, 3.0, size=50)),
    }


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


def _morph_cases() -> Dict[str, CaseFunc]:
    # pylint: disable=import-outside-toplevel
    import rsgispy
    import rsgispy.imagemorphology as rm

    def _simple(func: Any, key: str, **kw: Any) -> CaseFunc:
        def _run(inp: Dict[str, Any], out: Path) -> List[str]:
            args = dict(kw)
            if args.pop("circ", False):
                args.update(morph_op_file=inp["circ5"], use_op_file=True)
            path = str(out / "out.tif")
            func(inp[key], path, use_pbar=False, **args)
            return [path]

        return _run

    def _with_tmp(func: Any, key: str, **kw: Any) -> CaseFunc:
        def _run(inp: Dict[str, Any], out: Path) -> List[str]:
            args = dict(kw)
            if args.pop("circ", False):
                args.update(morph_op_file=inp["circ5"], use_op_file=True)
            path = str(out / "out.tif")
            tmp = str(out / "tmp.tif")
            func(inp[key], path, tmp_img=tmp, use_pbar=False, **args)
            return [path, tmp]

        return _run

    return {
        "morph_dilate_u8": _simple(rm.image_dilate, "u8"),
        "morph_dilate_f32nan": _simple(rm.image_dilate, "f32nan", op_size=5),
        "morph_dilate_f32_to_u8": _simple(
            rm.image_dilate, "f32", datatype=rsgispy.TYPE_8UINT
        ),
        "morph_erode_i16_circ": _simple(rm.image_erode, "i16", circ=True),
        "morph_erode_f32_op7": _simple(rm.image_erode, "f32", op_size=7),
        "morph_erode_u8_nd5": _simple(rm.image_erode, "u8", no_data_val=5),
        "morph_gradient_f32": _simple(rm.image_gradient, "f32", op_size=5),
        "morph_gradient_i16_nostats": _simple(
            rm.image_gradient, "i16", calc_stats=False
        ),
        "morph_dilate_comb_i16": _simple(rm.image_dilate_combined_out, "i16"),
        "morph_erode_comb_f32": _simple(rm.image_erode_combined_out, "f32", circ=True),
        "morph_gradient_comb_f32": _simple(
            rm.image_gradient_combined_out, "f32", datatype=rsgispy.TYPE_32FLOAT
        ),
        "morph_localmin_u8": _simple(rm.image_local_minima, "u8"),
        "morph_localmin_f32_strict": _simple(
            rm.image_local_minima, "f32", allow_equal=False, op_size=5
        ),
        "morph_localmin_seq_i16": _simple(
            rm.image_local_minima, "i16", sequencial_out=True
        ),
        "morph_localmin_comb_f32": _simple(rm.image_local_minima_combined_out, "f32"),
        "morph_localmin_comb_seq_i16": _simple(
            rm.image_local_minima_combined_out, "i16", sequencial_out=True, circ=True
        ),
        "morph_opening_u8": _simple(rm.image_opening, "u8"),
        "morph_opening_i16_tmp": _with_tmp(rm.image_opening, "i16", circ=True, niters=2),
        "morph_opening_f32": _simple(rm.image_opening, "f32", op_size=5),
        "morph_closing_f32nan": _simple(rm.image_closing, "f32nan", niters=2),
        "morph_closing_i16_to_u8": _simple(
            rm.image_closing, "i16", datatype=rsgispy.TYPE_8UINT
        ),
        "morph_closing_u8_tmp": _with_tmp(rm.image_closing, "u8", op_size=5),
        "morph_whitetophat_u8": _simple(rm.image_white_top_hat, "u8"),
        "morph_whitetophat_i16": _with_tmp(rm.image_white_top_hat, "i16", circ=True),
        "morph_blacktophat_f32": _with_tmp(rm.image_black_top_hat, "f32"),
        "morph_blacktophat_f32nan_i16": _simple(
            rm.image_black_top_hat, "f32nan", datatype=rsgispy.TYPE_16INT
        ),
    }


def _postproc_cases() -> Dict[str, CaseFunc]:
    # pylint: disable=import-outside-toplevel
    import rsgispy
    import rsgispy.classification.classpostprocess as cp

    def _maj(key: str, **kw: Any) -> CaseFunc:
        def _run(inp: Dict[str, Any], out: Path) -> List[str]:
            path = str(out / "out.tif")
            cp.apply_majority_filter(inp[key], path, use_pbar=False, **kw)
            return [path]

        return _run

    def _pipe(key: str, **kw: Any) -> CaseFunc:
        def _run(inp: Dict[str, Any], out: Path) -> List[str]:
            path = str(out / "out.tif")
            cp.apply_postprocess_pipeline(inp[key], path, use_pbar=False, **kw)
            return [path]

        return _run

    return {
        "postproc_majority_cls3": _maj("cls"),
        "postproc_majority_cls5_notie": _maj(
            "cls", filter_size=5, keep_centre_on_tie=False
        ),
        "postproc_majority_cls2_band2": _maj("cls2", img_band=2),
        "postproc_majority_dtype_nd": _maj(
            "cls", datatype=rsgispy.TYPE_16UINT, no_data_val=3, filter_size=5
        ),
        "postproc_majority_nocolour_nostats": _maj(
            "cls", copy_colour_table=False, calc_stats=False, filter_size=7
        ),
        "postproc_majority_float": _maj("f32nan", datatype=rsgispy.TYPE_32FLOAT),
        "postproc_pipeline_maj": _pipe("cls", majority_sizes=(3, 5)),
        "postproc_pipeline_maj_cls2": _pipe(
            "cls2", majority_sizes=(5, 3), img_band=1, keep_centre_on_tie=False
        ),
        "postproc_pipeline_maj_sieve": _pipe(
            "cls", majority_sizes=(3,), sieve_size_threshold=4, sieve_connectivity=8
        ),
        "postproc_pipeline_maj_sieve_tiled": _pipe(
            "cls", majority_sizes=(3,), sieve_size_threshold=6, in_memory=False
        ),
    }


def _predict_cases() -> Dict[str, CaseFunc]:
    # pylint: disable=import-outside-toplevel
    import rsgispy
    from rsgispy.ml import predict as mp

    def _case(func: Any, model: str, extra_out: Tuple[str, ...] = (), **kw: Any) -> CaseFunc:
        def _run(inp: Dict[str, Any], out: Path) -> List[str]:
            models = inp["models"]
            args = dict(kw)
            if args.pop("use_mask", False):
                args["in_msk_img"] = inp["mask"]
            if args.pop("use_scaler", False):
                args["scaler"] = models["scaler"]
            if args.pop("use_residuals", False):
                args["conformal_residuals"] = models["residuals"]
            est = (
                {0.1: models["q_lo"], 0.9: models["q_hi"]}
                if model == "q_map"
                else models[model]
            )
            path = str(out / "out.tif")
            paths = [path]
            for key in extra_out:
                args[key] = str(out / f"{key}.tif")
                paths.append(args[key])
            func(est, _feature_stack(inp), path, use_pbar=False, **args)
            return paths

        return _run

    return {
        "predict_cls_thematic": _case(mp.predict_raster, "clf", thematic=True),
        "predict_cls_mask_lut": _case(
            mp.predict_raster,
            "clf",
            use_mask=True,
            lut={1: 10, 2: 20, 3: 30, 4: 40},
            thematic=True,
            band_names=["landcover"],
        ),
        "predict_cls_float": _case(mp.predict_raster, "clf"),
        "predict_threshold": _case(mp.predict_raster, "bin", threshold=0.5),
        "predict_reg": _case(mp.predict_raster, "rf", no_data_val=-1.0),
        "predict_cls_scaler": _case(
            mp.predict_raster, "clf_sc", use_scaler=True, thematic=True,
            datatype=rsgispy.TYPE_16UINT,
        ),
        "predict_df_reg_mask": _case(
            mp.predict_raster, "df_reg", use_mask=True, band_names=["agb"]
        ),
        "predict_multi_mask": _case(mp.predict_raster, "rf2", use_mask=True),
        "predict_proba_all": _case(
            mp.predict_proba_raster,
            "clf",
            extra_out=("out_class_img", "out_entropy_img"),
            lut={1: 11, 2: 12, 3: 13, 4: 14},
        ),
        "predict_proba_pct_u8_mask": _case(
            mp.predict_proba_raster,
            "clf",
            use_mask=True,
            scale_to_pct=True,
            datatype=rsgispy.TYPE_8UINT,
            class_names=["w", "x", "y", "z"],
            extra_out=("out_class_img",),
        ),
        "predict_quant_ensemble": _case(
            mp.predict_quantiles_raster, "rf", quantiles=(0.1, 0.5, 0.9)
        ),
        "predict_quant_conformal_mask": _case(
            mp.predict_quantiles_raster, "lin", use_residuals=True, alpha=0.2,
            use_mask=True,
        ),
        "predict_quant_mapping": _case(
            mp.predict_quantiles_raster, "q_map", no_data_val=-1.0
        ),
    }


def all_cases() -> Dict[str, CaseFunc]:
    """All golden cases by name (the prefix names the golden file)."""
    cases: Dict[str, CaseFunc] = {}
    cases.update(_morph_cases())
    cases.update(_postproc_cases())
    cases.update(_predict_cases())
    return cases


# --------------------------------------------------------------------------
# Reading outputs
# --------------------------------------------------------------------------

_STAT_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
)


def read_output(path: str) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Return ``(pixels, metadata, colour_table)`` of an output raster."""
    with rasterio.open(path) as src:
        data = src.read()
        meta: Dict[str, Any] = {
            "dtype": src.dtypes[0],
            "count": int(src.count),
            "nodata": None if src.nodata is None else repr(float(src.nodata)),
            "descriptions": [d or "" for d in src.descriptions],
            "layer_type": [src.tags(b).get("LAYER_TYPE", "") for b in src.indexes],
            "stats": [
                [src.tags(b).get(k, "") for k in _STAT_KEYS] for b in src.indexes
            ],
            "overviews": [len(src.overviews(b)) for b in src.indexes],
        }
        try:
            cmap = src.colormap(1)
        except ValueError:
            cmap = None
        ctab = (
            np.array([cmap.get(i, (0, 0, 0, 0)) for i in range(256)], dtype=np.uint8)
            if cmap
            else np.zeros((0, 4), dtype=np.uint8)
        )
    return data, meta, ctab


@contextmanager
def block_size_env(block_size: int) -> Iterator[None]:
    """Temporarily set the package default block size."""
    old = os.environ.get("RSGISPY_BLOCK_SIZE")
    os.environ["RSGISPY_BLOCK_SIZE"] = str(int(block_size))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RSGISPY_BLOCK_SIZE", None)
        else:
            os.environ["RSGISPY_BLOCK_SIZE"] = old


def run_case(
    name: str, inp: Dict[str, Any], work_dir: Path, block_size: int
) -> List[Tuple[np.ndarray, Dict[str, Any], np.ndarray]]:
    """Run one case at ``block_size`` and read its outputs."""
    out_dir = work_dir / f"{name}_{block_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with block_size_env(block_size):
        paths = all_cases()[name](inp, out_dir)
    return [read_output(p) for p in paths]


def stats_close(stats_a: Any, stats_b: Any, rtol: float = 1e-9) -> bool:
    """
    Compare the statistics tags of two outputs. The statistics are
    accumulated block by block, so their last digits depend on the block size.
    """
    if len(stats_a) != len(stats_b):
        return False
    for band_a, band_b in zip(stats_a, stats_b):
        for val_a, val_b in zip(band_a, band_b):
            if (val_a == "") != (val_b == ""):
                return False
            if val_a and not np.isclose(float(val_a), float(val_b), rtol=rtol, atol=0.0):
                return False
    return True


def meta_equal(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
    """Compare output metadata (statistics within a relative tolerance)."""
    keys = set(meta_a) | set(meta_b)
    for key in keys:
        if key == "stats":
            if not stats_close(meta_a.get(key, []), meta_b.get(key, [])):
                return False
        elif meta_a.get(key) != meta_b.get(key):
            return False
    return True


def _same(res_a: List[Any], res_b: List[Any]) -> bool:
    for (d_a, m_a, c_a), (d_b, m_b, c_b) in zip(res_a, res_b):
        if d_a.dtype != d_b.dtype or not np.array_equal(d_a, d_b, equal_nan=True):
            return False
        if not meta_equal(m_a, m_b) or not np.array_equal(c_a, c_b):
            return False
    return len(res_a) == len(res_b)


def generate(out_dir: Path = GOLDEN_DIR) -> None:
    """Run every case at every block size and write the golden ``.npz`` files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, Any]] = {g: {} for g in GROUPS}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        inp = make_inputs(work)
        for name in all_cases():
            group = name.split("_", 1)[0]
            per_bs = {bs: run_case(name, inp, work, bs) for bs in BLOCK_SIZES}
            ref_bs = BLOCK_SIZES[0]
            # Store one reference when all block sizes agree, otherwise one
            # per block size.
            if all(_same(per_bs[ref_bs], per_bs[bs]) for bs in BLOCK_SIZES):
                variants = {"all": per_bs[ref_bs]}
            else:
                variants = {str(bs): per_bs[bs] for bs in BLOCK_SIZES}
                print(f"{name}: output depends on the block size")
            for tag, results in variants.items():
                for idx, (data, meta, ctab) in enumerate(results):
                    key = f"{name}__{tag}__{idx}"
                    payload[group][f"{key}__data"] = data
                    payload[group][f"{key}__meta"] = np.array(json.dumps(meta))
                    payload[group][f"{key}__ctab"] = ctab
    for group, arrays in payload.items():
        np.savez_compressed(out_dir / f"{group}_golden.npz", **arrays)


def load_golden(group: str) -> Dict[str, np.ndarray]:
    """Load the golden arrays of ``group`` (``'morph'``, ``'postproc'``, ``'predict'``)."""
    with np.load(GOLDEN_DIR / f"{group}_golden.npz", allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    generate()
    for grp in GROUPS:
        fpath = GOLDEN_DIR / f"{grp}_golden.npz"
        print(f"{fpath}: {fpath.stat().st_size} bytes")
