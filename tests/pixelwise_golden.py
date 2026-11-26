"""
Golden-output cases for the pixel-wise file-based functions (tile-graph
phase 3: indices, value transforms, masks, change detection, multi-date
percentiles and spectral transforms).

Each case writes small deterministic synthetic inputs (multi-band, integer
and float types, no data, image edges), runs one file-based function and
returns its output rasters. The reference outputs were generated with the
implementation before the functions became tile-graph wrappers and are
stored in ``tests/data/golden/pixelwise_<case>.npz``; the test
``tests/test_pixelwise_golden.py`` checks that the current code reproduces
them (integers exactly, floats within 1 ulp) for several block sizes,
including sizes that do not divide the image.

Regenerate (only when an output change is intended)::

    python tests/pixelwise_golden.py
"""

# pylint: disable=too-many-locals,missing-function-docstring

import json
import math
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import rsgispy
from rsgispy.imageprocess import reset_default_block_size, set_default_block_size

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden")
PREFIX = "pixelwise_"
#: Block sizes: one larger than the image (a single block), and two that do
#: not divide the image dimensions.
BLOCK_SIZES = (256, 16, 23)
HEIGHT = 45
WIDTH = 53
ORIGIN = (500000.0, 5000000.0)
RES = 10.0


def pattern(n_bands: int, seed: int, low: float, high: float,
            height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """Deterministic pseudo-random ``(n_bands, height, width)`` values in [low, high)."""
    b_idx, r_idx, c_idx = np.meshgrid(
        np.arange(n_bands, dtype=np.uint64),
        np.arange(height, dtype=np.uint64),
        np.arange(width, dtype=np.uint64),
        indexing="ij",
    )
    key = (
        r_idx * np.uint64(7919)
        + c_idx * np.uint64(104729)
        + b_idx * np.uint64(15485863)
        + np.uint64(seed) * np.uint64(2654435761)
    )
    key = (key * np.uint64(2654435761)) % np.uint64(2**32)
    key = (key ^ (key >> np.uint64(13))) * np.uint64(1274126177) % np.uint64(2**32)
    frac = key.astype(np.float64) / float(2**32)
    return low + frac * (high - low)


def write_img(path: str, arr: np.ndarray, nodata: Optional[float] = None,
              names: Optional[Sequence[str]] = None,
              origin=ORIGIN) -> str:
    arr3 = arr[np.newaxis] if arr.ndim == 2 else arr
    profile = {
        "driver": "GTiff",
        "height": arr3.shape[1],
        "width": arr3.shape[2],
        "count": arr3.shape[0],
        "dtype": str(arr3.dtype),
        "crs": "EPSG:32630",
        "transform": from_origin(origin[0], origin[1], RES, RES),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr3)
        if names:
            for idx, name in enumerate(names, start=1):
                dst.set_band_description(idx, name)
    return path


def _s2_cube(tmp: str, dtype=np.uint16, nodata: Optional[float] = 0.0) -> str:
    """Six-band (blue, green, red, nir, swir1, swir2) reflectance x 10000."""
    arr = pattern(6, 1, 50.0, 6000.0)
    arr[3] = pattern(1, 2, 1500.0, 9000.0)[0]  # NIR brighter
    out = np.rint(arr).astype(dtype)
    if nodata is not None:
        out[:, :3, :5] = nodata  # no-data corner on the image edge
        out[2, 20, 7:12] = nodata  # no data in one band only
    out[:, 30, 30] = 0 if nodata is None else out[:, 30, 30]
    return write_img(os.path.join(tmp, "s2.tif"), out, nodata=nodata,
                     names=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])


def _float_cube(tmp: str, name: str, n_bands: int, seed: int,
                nodata: Optional[float] = None, low: float = 0.01,
                high: float = 0.6) -> str:
    arr = pattern(n_bands, seed, low, high).astype(np.float32)
    arr[:, 0, :4] = np.nan
    arr[0, 10, 10] = np.inf
    arr[:, 44, 50:] = -np.inf if nodata is None else nodata
    return write_img(os.path.join(tmp, name), arr, nodata=nodata)


def _class_img(tmp: str, name: str = "cls.tif", nodata: Optional[float] = 0) -> str:
    arr = np.floor(pattern(1, 7, 0.0, 6.0)).astype(np.uint8)
    return write_img(os.path.join(tmp, name), arr, nodata=nodata)


def _mask_img(tmp: str, name: str = "msk.tif", origin=ORIGIN, height=HEIGHT,
              width=WIDTH) -> str:
    arr = (pattern(1, 11, 0.0, 1.0, height, width) > 0.2).astype(np.uint8)
    return write_img(os.path.join(tmp, name), arr, origin=origin)


def _endmembers(n_bands: int = 6) -> pd.DataFrame:
    vals = {
        "veg": [0.04, 0.08, 0.05, 0.45, 0.25, 0.12][:n_bands],
        "soil": [0.12, 0.18, 0.24, 0.30, 0.38, 0.33][:n_bands],
        "water": [0.06, 0.05, 0.03, 0.02, 0.01, 0.008][:n_bands],
    }
    return pd.DataFrame(vals, index=[f"b{i + 1}" for i in range(n_bands)])


# --------------------------------------------------------------------------
# Cases: each returns {output_label: output_path}.
# --------------------------------------------------------------------------


def case_calc_indices_float(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.calcindices as ic_idx

    img = _s2_cube(tmp)
    out = os.path.join(tmp, "idx.tif")
    ic_idx.calc_indices(
        img, out,
        band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir1", 6: "swir2"},
        indices=["NDVI", "EVI", "NDWI", "MNDWI", "SAVI", "Brightness_Scaled", "NBR"],
        in_gain=10000.0,
        index_params={"SAVI": {"l": 0.4}},
        use_pbar=False,
    )
    return {"out": out}


def case_calc_indices_int(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.calcindices as ic_idx

    img = _s2_cube(tmp, dtype=np.int16, nodata=-1)
    out = os.path.join(tmp, "idx_int.tif")
    ic_idx.calc_indices(
        img, out,
        band_names={3: "red", 4: "nir", 2: "green", 5: "swir1"},
        indices=["NDVI", "GNDVI", "NDMI"],
        in_gain=[10000.0, 10000.0, 9000.0, 10000.0, 10000.0, 10000.0],
        in_offset={"nir": 100.0, 3: 50.0, "green": 0.0, "swir1": 0.0},
        out_gain=[1000.0, 500.0, 1000.0],
        out_offset={"NDVI": 1.0, "GNDVI": 0.0, "NDMI": 0.0},
        datatype=rsgispy.TYPE_16INT,
        out_no_data_val=-32768,
        calc_stats=False,
        use_pbar=False,
    )
    return {"out": out}


def case_calc_sar_indices_db(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.calcindices as ic_idx

    arr = pattern(2, 3, -25.0, -2.0).astype(np.float32)
    arr[1] -= 6.0
    arr[:, 5, 5:9] = np.nan
    arr[0, 40, 40] = 0.0
    img = write_img(os.path.join(tmp, "s1.tif"), arr, nodata=float("nan"))
    out = os.path.join(tmp, "sar_idx.tif")
    ic_idx.calc_sar_indices(
        img, out,
        band_names={1: "VV", 2: "VH"},
        indices=["RVI_VV_VH", "CR_VH_VV_dB", "DPSVI", "Mean_VV_VH_dB", "SDWI"],
        in_units="dB",
        use_pbar=False,
    )
    return {"out": out}


def case_calc_sar_indices_power_int(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.calcindices as ic_idx

    arr = np.rint(pattern(2, 4, 0.0, 5000.0)).astype(np.uint16)
    arr[:, 12, :] = 0
    img = write_img(os.path.join(tmp, "palsar.tif"), arr, nodata=0)
    out = os.path.join(tmp, "sar_idx_int.tif")
    ic_idx.calc_sar_indices(
        img, out,
        band_names={1: "HH", 2: "HV"},
        indices=["RFDI", "CR_HV_HH_dB", "Span_HH_HV"],
        in_units="power",
        in_gain=10000.0,
        out_gain=[1000.0, 100.0, 1000.0],
        index_params={"DPSVI": {"vv_max": 1.0}},
        datatype=rsgispy.TYPE_16INT,
        use_pbar=False,
    )
    return {"out": out}


def case_rescale_img_pxl_vals(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.rescale as ic_resc

    img = _s2_cube(tmp)
    outs = {}
    objs = [
        ic_resc.ImageBandRescale(band=b, in_min=0.0, in_max=5000.0 + 100 * b,
                                 out_min=0.0, out_max=1.0)
        for b in range(1, 7)
    ]
    outs["float"] = os.path.join(tmp, "resc.tif")
    ic_resc.rescale_img_pxl_vals(img, outs["float"], objs, use_pbar=False)
    objs8 = [
        ic_resc.ImageBandRescale(band=b, in_min=6000.0, in_max=100.0,
                                 no_data_val=0.0 if b != 2 else 1234.0,
                                 out_min=1.0, out_max=254.0, out_no_data=0)
        for b in range(1, 7)
    ]
    outs["uint8"] = os.path.join(tmp, "resc8.tif")
    ic_resc.rescale_img_pxl_vals(img, outs["uint8"], objs8, trim_to_limits=False,
                                 datatype=rsgispy.TYPE_8UINT, use_pbar=False)
    objs_kea = [
        ic_resc.ImageBandRescale(band=b, in_min=0.0, in_max=6000.0, out_min=0.0,
                                 out_max=100.0, out_no_data=-(b * 10.0))
        for b in range(1, 7)
    ]
    outs["kea_nd"] = os.path.join(tmp, "resc.kea")
    ic_resc.rescale_img_pxl_vals(img, outs["kea_nd"], objs_kea, gdalformat="KEA",
                                 use_pbar=False)
    return outs


def case_calc_img_rescale(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.rescale as ic_resc

    img = _s2_cube(tmp)
    other = write_img(
        os.path.join(tmp, "other.tif"),
        np.rint(pattern(2, 9, 0.0, 300.0, HEIGHT + 4, WIDTH + 3)).astype(np.int16),
        origin=(ORIGIN[0] - 2 * RES, ORIGIN[1] + 1 * RES),
    )
    out = os.path.join(tmp, "calc_resc.tif")
    ic_resc.calc_img_rescale([img, other], out, c_no_data_val=0.0, c_gain=10000.0,
                             n_no_data_val=-1.0, n_gain=255.0, n_offset=1.0,
                             datatype=rsgispy.TYPE_16INT, use_pbar=False)
    return {"out": out}


def case_apply_img_norm_params(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.rescale as ic_resc

    img = _s2_cube(tmp)
    outs = {}
    for method in ("linear", "zscore", "l2", "stdev"):
        params = ic_resc.calc_img_norm_params(img, method=method, bands=[1, 3, 4],
                                              use_pbar=False)
        outs[method] = os.path.join(tmp, f"norm_{method}.tif")
        ic_resc.apply_img_norm_params(img, outs[method], norm_params=params,
                                      out_no_data_val=-99.0, use_pbar=False)
    params = ic_resc.calc_img_norm_params(img, method="linear", out_min=0.0,
                                          out_max=255.0, use_pbar=False)
    outs["linear_u8"] = os.path.join(tmp, "norm_u8.tif")
    ic_resc.apply_img_norm_params(img, outs["linear_u8"], norm_params=params,
                                  trim_to_limits=True, datatype=rsgispy.TYPE_8UINT,
                                  use_pbar=False)
    return outs


def case_recode_int_raster(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.encoding as ic_enc

    img = _class_img(tmp)
    outs = {"keep": os.path.join(tmp, "recode_keep.tif"),
            "drop": os.path.join(tmp, "recode_drop.tif")}
    ic_enc.recode_int_raster(img, outs["keep"], {1: 10, 2: 20, 0: 99}, use_pbar=False)
    ic_enc.recode_int_raster(img, outs["drop"], {3: 7, 5: 1}, keep_vals_not_in_dict=False,
                             datatype=rsgispy.TYPE_8UINT, use_pbar=False)
    return outs


def case_one_hot_category_encoding(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.encoding as ic_enc

    img = _class_img(tmp)
    outs = {"auto": os.path.join(tmp, "onehot.tif"),
            "dict": os.path.join(tmp, "onehot_dict.tif")}
    ic_enc.one_hot_category_encoding(img, outs["auto"], no_data_val=0, use_pbar=False)
    ic_enc.one_hot_category_encoding(img, outs["dict"], recode_dict={1: 2, 4: 1, 5: 2},
                                     use_pbar=False)
    return outs


def case_mask_img(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    img = _s2_cube(tmp)
    msk = _mask_img(tmp, origin=(ORIGIN[0] + 3 * RES, ORIGIN[1] + 2 * RES),
                    height=HEIGHT + 5, width=WIDTH)
    outs = {"int": os.path.join(tmp, "masked.tif"),
            "float": os.path.join(tmp, "masked_f.tif")}
    iu_msk.mask_img(img, msk, outs["int"], 0, [0], use_pbar=False)
    iu_msk.mask_img(img, msk, outs["float"], -5.5, 1, datatype=rsgispy.TYPE_32FLOAT,
                    no_data_val=-5.5, use_pbar=False)
    return outs


def case_gen_finite_mask(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    img = _float_cube(tmp, "fl.tif", 3, 5)
    out = os.path.join(tmp, "finite.tif")
    iu_msk.gen_finite_mask(img, out, use_pbar=False)
    return {"out": out}


def case_gen_valid_mask(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    img1 = _s2_cube(tmp)
    img2 = _float_cube(tmp, "fl.tif", 2, 6, nodata=-1.0)
    out = os.path.join(tmp, "valid.tif")
    iu_msk.gen_valid_mask([img1, img2], out, use_pbar=False)
    out2 = os.path.join(tmp, "valid_nan.tif")
    iu_msk.gen_valid_mask(img2, out2, no_data_val=None, use_pbar=False)
    return {"out": out, "nan": out2}


def case_gen_img_edge_mask(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    img = _class_img(tmp)
    out = os.path.join(tmp, "edge.tif")
    iu_msk.gen_img_edge_mask(img, out, 4, use_pbar=False)
    return {"out": out}


def case_mask_all_band_zero_vals(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    arr = np.rint(pattern(3, 12, -50.0, 200.0)).astype(np.int16)
    arr[:, 4, 4] = 0
    img = write_img(os.path.join(tmp, "z.tif"), arr, nodata=-50, names=["a", "", "c"])
    out = os.path.join(tmp, "zero.tif")
    iu_msk.mask_all_band_zero_vals(img, out, out_val=7, use_pbar=False)
    return {"out": out}


def case_create_valid_mask(tmp: str) -> Dict[str, str]:
    import rsgispy.imageutils.masking as iu_msk

    img1 = _s2_cube(tmp)
    img2 = _float_cube(tmp, "fl.tif", 3, 8)
    infos = [iu_msk.ImageBandInfo(img1, "s2", [1, 4]), iu_msk.ImageBandInfo(img2, "fl")]
    out = os.path.join(tmp, "cvalid.tif")
    iu_msk.create_valid_mask(infos, out, no_data_val=0, use_pbar=False)
    return {"out": out}


def _change_pair(tmp: str):
    base = np.rint(pattern(3, 21, 0.0, 3000.0)).astype(np.uint16)
    chng = np.rint(pattern(3, 22, 0.0, 3000.0)).astype(np.uint16)
    base[:, 0:2, :] = 0
    chng[:, :, 50:] = 0
    chng[1, 20, 20] = 0
    return (write_img(os.path.join(tmp, "t1.tif"), base),
            write_img(os.path.join(tmp, "t2.tif"), chng))


def case_image_difference(tmp: str) -> Dict[str, str]:
    import rsgispy.changedetect.imgchngmthds as icm

    base, chng = _change_pair(tmp)
    msk = _mask_img(tmp)
    outs = {k: os.path.join(tmp, f"diff_{k}.tif") for k in ("plain", "std", "msk")}
    icm.image_difference(base, chng, outs["plain"], use_pbar=False)
    icm.image_difference(base, chng, outs["std"], apply_std=True, img_base_bands=[3, 1],
                         img_chng_bands=[1, 2], no_data_val=-999.0, use_pbar=False)
    icm.image_difference(base, chng, outs["msk"], vld_msk_img=msk,
                         datatype=rsgispy.TYPE_16INT, no_data_val=-1, use_pbar=False)
    return outs


def case_image_ratio(tmp: str) -> Dict[str, str]:
    import rsgispy.changedetect.imgchngmthds as icm

    base, chng = _change_pair(tmp)
    outs = {k: os.path.join(tmp, f"ratio_{k}.tif") for k in ("plain", "std")}
    icm.image_ratio(base, chng, outs["plain"], use_pbar=False)
    icm.image_ratio(base, chng, outs["std"], apply_std=True, calc_stats=False,
                    use_pbar=False)
    return outs


def case_calc_imgs_pxl_percentiles(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.percentiles as ic_pct

    imgs = []
    for idx in range(3):
        arr = np.rint(pattern(2, 30 + idx, 0.0, 1000.0)).astype(np.uint16)
        arr[:, idx * 5:idx * 5 + 6, 3] = 0
        arr[:, :, 0] = 0
        imgs.append(write_img(os.path.join(tmp, f"p{idx}.tif"), arr))
    out = os.path.join(tmp, "pct.tif")
    ic_pct.calc_imgs_pxl_percentiles(imgs, out, [10.0, 50.0, 92.5], use_pbar=False)
    return {"out": out}


def case_calc_img_band_pxl_percentiles(tmp: str) -> Dict[str, str]:
    import rsgispy.imagecalc.percentiles as ic_pct

    imgs = []
    for idx in range(4):
        imgs.append(_float_cube(tmp, f"f{idx}.tif", 2, 40 + idx))
    with rasterio.open(imgs[0], "r+") as dst:
        dst.set_band_description(1, "Blue")
    out = os.path.join(tmp, "bpct.tif")
    ic_pct.calc_img_band_pxl_percentiles(imgs, out, [25.0, 75.0], no_data_val=None,
                                         use_pbar=False)
    return {"out": out}


def case_calc_tasseled_cap(tmp: str) -> Dict[str, str]:
    import rsgispy.imagespectral.transforms as st

    img = _s2_cube(tmp)
    msk = _mask_img(tmp)
    outs = {"preset": os.path.join(tmp, "tc.tif"), "custom": os.path.join(tmp, "tc2.tif")}
    st.calc_tasseled_cap(img, outs["preset"], sensor="sentinel2_msi", gain=10000.0,
                         use_pbar=False)
    st.calc_tasseled_cap(img, outs["custom"], coeffs=[[0.5, 0.5], [0.3, -0.7]],
                         bands=[3, 4], valid_msk_img=msk, component_names=["a", "b"],
                         datatype=rsgispy.TYPE_16INT, out_no_data_val=-1, use_pbar=False)
    return outs


def case_calc_pca_mnf(tmp: str) -> Dict[str, str]:
    import rsgispy.imagespectral.transforms as st

    img = _s2_cube(tmp)
    msk = _mask_img(tmp)
    outs = {k: os.path.join(tmp, f"{k}.tif") for k in ("pca", "mnf", "white")}
    st.calc_pca(img, outs["pca"], n_components=3, standardise=True, use_pbar=False)
    st.calc_mnf(img, outs["mnf"], n_components=2, valid_msk_img=msk, use_pbar=False)
    st.whiten_image(img, outs["white"], msk, use_pbar=False)
    return outs


def case_spectral_angle_mapper(tmp: str) -> Dict[str, str]:
    import rsgispy.imagespectral.classification as sc

    img = _s2_cube(tmp)
    msk = _mask_img(tmp)
    outs = {k: os.path.join(tmp, f"sam_{k}.tif") for k in ("angles", "cls", "min")}
    sc.spectral_angle_mapper(img, outs["angles"], _endmembers(), out_class_img=outs["cls"],
                             out_min_angle_img=outs["min"], img_gain=10000.0,
                             degrees=True, max_angle=20.0, valid_msk_img=msk,
                             use_pbar=False)
    return outs


def case_spectral_correlation_mapper(tmp: str) -> Dict[str, str]:
    import rsgispy.imagespectral.classification as sc

    img = _s2_cube(tmp)
    outs = {k: os.path.join(tmp, f"scm_{k}.tif") for k in ("corr", "cls")}
    sc.spectral_correlation_mapper(img, outs["corr"], _endmembers(),
                                   out_class_img=outs["cls"], min_corr=0.3,
                                   no_data_val=0, use_pbar=False)
    return outs


def _mix_cube(tmp: str) -> str:
    """Six-band image mixing the endmembers (x 10000) plus noise and no data."""
    fracs = pattern(3, 60, 0.0, 1.0)
    fracs /= fracs.sum(axis=0, keepdims=True)
    ems = _endmembers().to_numpy().T  # (3, 6)
    refl = np.einsum("mhw,mb->bhw", fracs, ems) + pattern(6, 61, -0.01, 0.01)
    arr = np.rint(np.clip(refl, 0.0, None) * 10000.0).astype(np.uint16)
    arr[:, :3, :5] = 0
    return write_img(os.path.join(tmp, "s2.tif"), arr, nodata=0)


def _unmix(tmp: str, fn_name: str, **kwargs) -> Dict[str, str]:
    import rsgispy.imagespectral.unmixing as su

    img = _mix_cube(tmp)
    msk = _mask_img(tmp)
    out = os.path.join(tmp, f"{fn_name}.tif")
    getattr(su, fn_name)(img, out, _endmembers() * 10000.0, valid_msk_img=msk, gain=10000.0,
                         use_pbar=False, **kwargs)
    return {"out": out}


def case_spec_unmix_ucls(tmp: str) -> Dict[str, str]:
    outs = _unmix(tmp, "spec_unmix_ucls")
    import rsgispy.imagespectral.unmixing as su

    out_w = os.path.join(tmp, "ucls_w.tif")
    su.spec_unmix_ucls(os.path.join(tmp, "s2.tif"), out_w, _endmembers() * 10000.0, gain=10000.0,
                       weight=0.5, out_no_data_val=float("nan"), use_pbar=False)
    outs["weight"] = out_w
    return outs


def case_spec_unmix_nnls(tmp: str) -> Dict[str, str]:
    return _unmix(tmp, "spec_unmix_nnls")


def case_spec_unmix_fcls(tmp: str) -> Dict[str, str]:
    return _unmix(tmp, "spec_unmix_fcls")


def case_predict_refl_linear_unmixing(tmp: str) -> Dict[str, str]:
    import rsgispy.imagespectral.unmixing as su

    coefs = pattern(3, 50, 0.0, 1.0).astype(np.float32)
    coefs[:, 7, 7:20] = -9999.0
    coefs[1, 30, 3] = np.nan
    img = write_img(os.path.join(tmp, "coefs.tif"), coefs, nodata=-9999.0)
    out = os.path.join(tmp, "pred.tif")
    su.predict_refl_linear_unmixing(img, out, _endmembers(), use_pbar=False)
    return {"out": out}


CASES: Dict[str, Callable[[str], Dict[str, str]]] = {
    name[len("case_"):]: fn
    for name, fn in sorted(globals().items())
    if name.startswith("case_") and callable(fn)
}


# --------------------------------------------------------------------------
# Reading outputs, generating and comparing.
# --------------------------------------------------------------------------


def _tags_float_round(tags: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in sorted(tags.items())}


def read_output(path: str) -> Dict[str, Any]:
    """Return ``{'data': array, 'meta': dict}`` for a raster output."""
    with rasterio.open(path) as ds:
        data = ds.read()
        meta: Dict[str, Any] = {
            "driver": ds.driver,
            "dtype": ds.dtypes[0],
            "count": ds.count,
            "shape": [ds.height, ds.width],
            "transform": list(ds.transform)[:6],
            "crs": ds.crs.to_string() if ds.crs else None,
            "nodatavals": [None if v is None else repr(float(v)) for v in ds.nodatavals],
            "descriptions": list(ds.descriptions),
            "band_tags": [_tags_float_round(ds.tags(b)) for b in range(1, ds.count + 1)],
            "overviews": [ds.overviews(b) for b in range(1, ds.count + 1)],
        }
        try:
            cmap = ds.colormap(1)
            meta["colormap_len"] = len(cmap)
        except ValueError:
            meta["colormap_len"] = 0
    return {"data": data, "meta": meta}


def run_case(name: str, block_size: int) -> Dict[str, Dict[str, Any]]:
    """Run case ``name`` with ``block_size`` and read its outputs."""
    set_default_block_size(block_size)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = CASES[name](tmp)
            return {label: read_output(path) for label, path in paths.items()}
    finally:
        reset_default_block_size()


def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{PREFIX}{name}.npz")


def _same(arr_a: np.ndarray, arr_b: np.ndarray) -> bool:
    return arr_a.dtype == arr_b.dtype and arr_a.shape == arr_b.shape and bool(
        np.array_equal(arr_a, arr_b, equal_nan=np.issubdtype(arr_a.dtype, np.floating))
    )


def generate(name: str) -> List[str]:
    """Write the golden file of case ``name``; return labels that vary by block size."""
    per_bs = {bs: run_case(name, bs) for bs in BLOCK_SIZES}
    ref = per_bs[BLOCK_SIZES[0]]
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    varying = []
    for label, out in ref.items():
        meta[label] = out["meta"]
        if all(_same(out["data"], per_bs[bs][label]["data"]) for bs in BLOCK_SIZES[1:]):
            arrays[f"{label}__all"] = out["data"]
        else:
            varying.append(label)
            for bs in BLOCK_SIZES:
                arrays[f"{label}__bs{bs}"] = per_bs[bs][label]["data"]
    arrays["__meta__"] = np.array(json.dumps(meta, sort_keys=True))
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    np.savez_compressed(golden_path(name), **arrays)
    return varying


def load_golden(name: str):
    with np.load(golden_path(name), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(str(arrays.pop("__meta__")))
    return arrays, meta


def assert_close_ulp(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    """Integers exactly equal; floats equal within 1 ulp (NaN matching NaN)."""
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    if not np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        return
    nan_a = np.isnan(actual)
    nan_e = np.isnan(expected)
    np.testing.assert_array_equal(nan_a, nan_e, err_msg=f"{label}: NaN positions")
    fin = ~nan_e
    np.testing.assert_array_max_ulp(actual[fin], expected[fin], maxulp=1)


def assert_meta_equal(actual: Dict[str, Any], expected: Dict[str, Any], label: str) -> None:
    exp = json.loads(json.dumps(expected))
    act = json.loads(json.dumps(actual))
    for key in exp:
        if key == "band_tags":
            for b_idx, (t_a, t_e) in enumerate(zip(act[key], exp[key])):
                assert set(t_a) == set(t_e), f"{label}: band {b_idx + 1} tag keys"
                for tag, val in t_e.items():
                    try:
                        f_e = float(val)
                        f_a = float(t_a[tag])
                    except ValueError:
                        assert t_a[tag] == val, f"{label}: tag {tag}"
                        continue
                    assert (math.isnan(f_e) and math.isnan(f_a)) or math.isclose(
                        f_a, f_e, rel_tol=1e-6, abs_tol=1e-9
                    ), f"{label}: band {b_idx + 1} tag {tag}: {t_a[tag]} != {val}"
        else:
            assert act[key] == exp[key], f"{label}: {key}: {act[key]} != {exp[key]}"


if __name__ == "__main__":
    os.environ.setdefault("RSGISPY_QUIET", "1")
    names = sys.argv[1:] or list(CASES)
    for case_name in names:
        var = generate(case_name)
        size = os.path.getsize(golden_path(case_name))
        print(f"{case_name}: {size / 1024:.1f} KiB" + (f" (varies by block size: {var})" if var else ""))
