"""
Shared internal raster/band resolution, vector pixel extraction, sample table
I/O, splitting, and block-reading helpers reused across
:mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`.
"""

from contextlib import ExitStack
import math
import os
import secrets
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject as rio_reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    compute_spatial_intersection,
    ensure_file_exists,
    extract_roi_feature_matrix,
    grids_compatible,
    validate_band_index,
)
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.tools.tables import (
    read_dataframe_table,
    resolve_table_format,
    write_dataframe_table,
)
from rsgispy.vectorprocess import infer_parquet_compression
from rsgispy.zonalstats.point import ext_point_band_values_file
from rsgispy.zonalstats.polygon import export_poly_pxls_to_df

SamplesInput = Union[pd.DataFrame, str, os.PathLike]


def resolve_rng(rnd_seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy Generator from an optional integer seed."""
    seed = secrets.randbits(128) if rnd_seed is None else abs(int(rnd_seed))
    return np.random.default_rng(seed)


def normalise_img_band_info_list(
    img_band_info: Sequence[Union[ImageBandInfo, Mapping[str, Any]]],
) -> List[ImageBandInfo]:
    """Validate and copy a sequence of :class:`ImageBandInfo` objects."""
    if not img_band_info or isinstance(img_band_info, (str, bytes)):
        raise rsgispy.RSGISPyException(
            "img_band_info must be a non-empty sequence of ImageBandInfo objects."
        )

    out_list: List[ImageBandInfo] = []
    for item in img_band_info:
        if isinstance(item, ImageBandInfo):
            obj = ImageBandInfo(
                file_name=item.file_name,
                name=item.name,
                bands=item.bands,
                band_names=item.band_names,
            )
        elif isinstance(item, Mapping):
            obj = ImageBandInfo.from_dict(item)
        else:
            raise rsgispy.RSGISPyException(
                "Each item in img_band_info must be an ImageBandInfo instance."
            )
        if not obj.file_name:
            raise rsgispy.RSGISPyException(
                "Each ImageBandInfo object must specify a valid 'file_name'."
            )
        out_list.append(obj)
    return out_list


def resolve_img_band_info(
    img_band_info: Sequence[Union[ImageBandInfo, Mapping[str, Any]]],
) -> List[Tuple[str, List[int], List[str]]]:
    """
    Validate input raster paths and resolve 1-based band lists and unique
    output feature column names for each :class:`ImageBandInfo`.
    """
    norm_list = normalise_img_band_info_list(img_band_info)
    resolved: List[Tuple[str, List[int], List[str]]] = []
    seen_names: set = set()
    seen_lower_names: set = set()

    for img_idx, info in enumerate(norm_list):
        if info.file_name is None:
            raise rsgispy.RSGISPyException(
                f"ImageBandInfo at index {img_idx} is missing a file_name."
            )
        ensure_file_exists(info.file_name)
        with rasterio.open(info.file_name, "r") as src:
            n_bands = int(src.count)

        if info.bands is None:
            bands = list(range(1, n_bands + 1))
        else:
            bands = [int(b) for b in info.bands]
            if not bands:
                raise rsgispy.RSGISPyException(
                    f"ImageBandInfo for '{info.file_name}' specified an empty "
                    "band list."
                )
            for b_idx in bands:
                validate_band_index(b_idx, n_bands, info.file_name)

        if info.band_names is not None:
            if len(info.band_names) != len(bands):
                raise rsgispy.RSGISPyException(
                    f"ImageBandInfo for '{info.file_name}' has "
                    f"{len(info.band_names)} band_names for {len(bands)} bands."
                )
            col_names = list(info.band_names)
        else:
            if info.name and info.name.strip():
                prefix = info.name.strip()
            else:
                stem = os.path.splitext(os.path.basename(info.file_name))[0]
                prefix = stem if stem else f"img{img_idx + 1}"
            col_names = [f"{prefix}_b{b_idx}" for b_idx in bands]

        for col in col_names:
            if col in seen_names or col.lower() in seen_lower_names:
                raise rsgispy.RSGISPyException(
                    f"Duplicate feature column name '{col}' across "
                    "img_band_info. Specify unique ImageBandInfo names or "
                    "band_names."
                )
            seen_names.add(col)
            seen_lower_names.add(col.lower())

        resolved.append((info.file_name, bands, col_names))

    return resolved


def infer_band_names_from_img_info(
    img_band_info: Sequence[ImageBandInfo],
) -> Optional[List[str]]:
    """
    Resolve feature column names from ``ImageBandInfo`` objects, avoiding
    opening raster files when band indices and names are already explicit.
    """
    all_have_files = all(
        info.file_name is not None and os.path.exists(info.file_name)
        for info in img_band_info
    )
    if all_have_files:
        resolved = resolve_img_band_info(img_band_info)
        return [col for _, _, cols in resolved for col in cols]

    names: List[str] = []
    for idx, info in enumerate(img_band_info):
        if info.band_names is not None:
            names.extend(info.band_names)
        elif info.bands is not None:
            if info.name and info.name.strip():
                prefix = info.name.strip()
            elif info.file_name:
                prefix = os.path.splitext(os.path.basename(info.file_name))[0]
            else:
                prefix = f"img{idx + 1}"
            names.extend(f"{prefix}_b{int(b)}" for b in info.bands)
        else:
            return None
    return names


def resolve_apply_band_specs_from_seq(
    img_band_info: Optional[Sequence[ImageBandInfo]],
    cfg_img_band_info: Optional[Sequence[ImageBandInfo]] = None,
    cfg_label: str = "config",
) -> List[Tuple[str, int, str]]:
    """
    Resolve ordered ``(img_path, band_1based, col_name)`` specs from either
    an explicit ``img_band_info`` sequence or a fallback config sequence.
    """
    info_seq = (
        img_band_info
        if img_band_info is not None
        else cfg_img_band_info
    )
    if not info_seq:
        raise rsgispy.RSGISPyException(
            f"img_band_info must be provided either inside {cfg_label} "
            "or via the img_band_info parameter."
        )
    resolved = resolve_img_band_info(info_seq)
    band_specs: List[Tuple[str, int, str]] = []
    for img_path, bands, col_names in resolved:
        for b_idx, col_name in zip(bands, col_names):
            band_specs.append((img_path, int(b_idx), col_name))
    return band_specs


def write_ml_samples(
    samples_df: pd.DataFrame,
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
) -> None:
    """
    Write a sample :class:`pandas.DataFrame` to disk in Parquet (using Snappy
    compression by default) or another pandas-supported table format.
    """
    path_str = os.fspath(out_file)
    fmt = resolve_table_format(path_str, out_format=out_format)
    eff_compress: Optional[Union[str, bool]]
    if fmt == "parquet":
        if out_compress is None:
            inferred = infer_parquet_compression(path_str, out_compress=None)
            eff_compress = inferred if inferred is not None else "snappy"
        else:
            eff_compress = out_compress
    elif out_compress == "snappy":
        eff_compress = None
    else:
        eff_compress = out_compress

    write_dataframe_table(
        samples_df,
        path_str,
        out_format=fmt,
        out_compress=eff_compress,
    )


def read_ml_samples(
    in_file: Union[str, os.PathLike],
    in_format: Optional[str] = None,
) -> pd.DataFrame:
    """Read a sample table file into a :class:`pandas.DataFrame`."""
    return read_dataframe_table(os.fspath(in_file), in_format=in_format)


def load_samples_df(samples: SamplesInput, label: str) -> pd.DataFrame:
    """Load sample DataFrame from an in-memory DataFrame or table file path."""
    if isinstance(samples, pd.DataFrame):
        return samples
    if isinstance(samples, (str, os.PathLike)):
        return read_ml_samples(samples)
    raise rsgispy.RSGISPyException(
        f"{label} must be a pandas DataFrame or sample table file path."
    )


def resolve_feature_cols(
    cfg_img_band_info: Optional[Sequence[ImageBandInfo]],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
    excluded_cols: Sequence[str],
) -> List[str]:
    """Determine and validate ordered feature column names for sample tables."""
    if feature_cols is not None:
        cols = [str(col) for col in feature_cols]
    elif cfg_img_band_info is not None:
        inferred = infer_band_names_from_img_info(cfg_img_band_info)
        if inferred is not None and all(col in df.columns for col in inferred):
            cols = inferred
        else:
            excluded = set(excluded_cols) | {"x", "y"}
            cols = [
                str(col)
                for col in df.columns
                if col not in excluded
                and pd.api.types.is_numeric_dtype(df[col])
            ]
    else:
        excluded = set(excluded_cols) | {"x", "y"}
        cols = [
            str(col)
            for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not cols:
        raise rsgispy.RSGISPyException(
            "Could not resolve any feature columns from training data."
        )
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Feature column(s) {missing} missing from sample DataFrame."
        )
    return cols


def resolve_estimator_feature_cols(
    estimator: Any,
    band_specs: Sequence[Tuple[str, int, str]],
) -> Tuple[bool, List[str], List[Tuple[str, int, str]]]:
    """
    Determine whether ``estimator`` expects named DataFrame columns and align
    ``band_specs`` to match ``estimator``'s trained feature order (including
    when the model was trained on a subset or reordered list of features).
    """
    specs_list = list(band_specs)
    col_names = [spec[2] for spec in specs_list]
    feat_names = getattr(estimator, "feature_names_in_", None)
    if feat_names is None:
        feat_names = getattr(estimator, "feature_names_", None)
    if feat_names is not None:
        est_cols = [str(c) for c in feat_names]
        spec_by_col = {spec[2]: spec for spec in specs_list}
        if all(col in spec_by_col for col in est_cols):
            aligned_specs = [spec_by_col[col] for col in est_cols]
            return True, est_cols, aligned_specs
        if len(est_cols) == len(col_names):
            return True, est_cols, specs_list
        missing = [col for col in est_cols if col not in spec_by_col]
        raise rsgispy.RSGISPyException(
            f"Estimator was trained on {len(est_cols)} feature(s) {est_cols}, "
            f"which do not match the {len(col_names)} configured image band(s) "
            f"{col_names} (missing: {missing})."
        )
    return False, col_names, specs_list


def open_apply_datasets(
    stack: ExitStack,
    msk_path: str,
    band_specs: Sequence[Tuple[str, int, str]],
) -> Tuple[DatasetReader, Dict[str, DatasetReader], IntersectionGrid, bool]:
    """Open mask and feature datasets and compute spatial intersection."""
    msk_ds = stack.enter_context(rasterio.open(msk_path, "r"))
    datasets: Dict[str, DatasetReader] = {msk_path: msk_ds}
    for img_path, band_idx, _ in band_specs:
        if img_path not in datasets:
            datasets[img_path] = stack.enter_context(
                rasterio.open(img_path, "r")
            )
        if band_idx < 1 or band_idx > datasets[img_path].count:
            raise rsgispy.RSGISPyException(
                f"Band {band_idx} is out of range for image '{img_path}'."
            )

    all_compatible = all(grids_compatible(msk_ds, ds) for ds in datasets.values())
    if all_compatible:
        grid = compute_spatial_intersection(datasets)
    else:
        grid = IntersectionGrid.from_dataset(msk_ds, offsets={msk_path: (0, 0)})
    return msk_ds, datasets, grid, all_compatible


def read_feature_block(
    band_specs: Sequence[Tuple[str, int, str]],
    datasets: Mapping[str, DatasetReader],
    ref_ds: DatasetReader,
    grid: IntersectionGrid,
    out_win: Window,
    all_compatible: bool,
    ref_path: str,
    explicit_no_data: Optional[float] = None,
) -> List[Tuple[np.ndarray, Optional[float]]]:
    """Read aligned or on-the-fly reprojected feature band arrays for a block."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    col_off = int(out_win.col_off)
    row_off = int(out_win.row_off)
    win_w = int(out_win.width)
    win_h = int(out_win.height)
    out_arrays: List[Tuple[np.ndarray, Optional[float]]] = []

    exp_nd: Optional[float] = (
        float(explicit_no_data) if explicit_no_data is not None else None
    )
    win_xfm = None
    for img_path, band_idx, _ in band_specs:
        src_ds = datasets[img_path]
        if exp_nd is not None:
            nd_val: Optional[float] = exp_nd
        else:
            raw_nd = src_ds.nodatavals[band_idx - 1]
            nd_val = None
            if raw_nd is not None and not (
                isinstance(raw_nd, float) and np.isnan(raw_nd)
            ):
                nd_val = float(raw_nd)

        if all_compatible:
            base_col, base_row = grid.offsets[img_path]
            in_win = Window(
                base_col + col_off, base_row + row_off, win_w, win_h
            )
            arr = src_ds.read(band_idx, window=in_win, out_dtype=np.float64)
        elif grids_compatible(ref_ds, src_ds):
            base_col, base_row = grid.offsets.get(
                img_path,
                (
                    int(
                        round(
                            (
                                ~src_ds.transform
                                @ (grid.transform.c, grid.transform.f)
                            )[0]
                        )
                    ),
                    int(
                        round(
                            (
                                ~src_ds.transform
                                @ (grid.transform.c, grid.transform.f)
                            )[1]
                        )
                    ),
                ),
            )
            in_win = Window(
                base_col + col_off, base_row + row_off, win_w, win_h
            )
            arr = src_ds.read(band_idx, window=in_win, out_dtype=np.float64)
        else:
            if win_xfm is None:
                base_ref_col, base_ref_row = grid.offsets.get(
                    ref_path, (0, 0)
                )
                ref_win = Window(
                    base_ref_col + col_off,
                    base_ref_row + row_off,
                    win_w,
                    win_h,
                )
                win_xfm = window_transform(ref_win, ref_ds.transform)
            arr = np.full((win_h, win_w), np.nan, dtype=np.float64)
            reproj_nd = (
                None
                if (nd_val is not None and np.isnan(nd_val))
                else nd_val
            )
            rio_reproject(
                source=rasterio.band(src_ds, band_idx),
                destination=arr,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                src_nodata=reproj_nd,
                dst_transform=win_xfm,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            nd_val = None

        out_arrays.append((arr, nd_val))
    return out_arrays


def _grids_pixel_aligned(img_a: str, img_b: str) -> bool:
    """Return True if two rasters share CRS, resolution, and pixel grid alignment."""
    with rasterio.open(img_a, "r") as ds_a, rasterio.open(img_b, "r") as ds_b:
        if ds_a.crs != ds_b.crs:
            return False
        ta, tb = ds_a.transform, ds_b.transform
        if not (
            math.isclose(ta.a, tb.a, rel_tol=1e-6, abs_tol=1e-9)
            and math.isclose(ta.e, tb.e, rel_tol=1e-6, abs_tol=1e-9)
            and math.isclose(ta.b, tb.b, abs_tol=1e-9)
            and math.isclose(ta.d, tb.d, abs_tol=1e-9)
        ):
            return False
        if abs(ta.a) == 0.0 or abs(ta.e) == 0.0:
            return False
        dx_px = (tb.c - ta.c) / ta.a
        dy_px = (tb.f - ta.f) / ta.e
        return math.isclose(
            dx_px, round(dx_px), abs_tol=1e-4
        ) and math.isclose(dy_px, round(dy_px), abs_tol=1e-4)


def extract_points_samples(
    pts_gdf: gpd.GeoDataFrame,
    resolved_imgs: List[Tuple[str, List[int], List[str]]],
    attr_cols: Optional[Sequence[str]] = None,
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Extract band values at point samples using ``ext_point_band_values_file``,
    optionally preserving vector attribute columns ``attr_cols``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    work_gdf = pts_gdf.copy()
    eff_min = -math.inf if min_thres is None else float(min_thres)
    eff_max = math.inf if max_thres is None else float(max_thres)
    eff_nd = math.nan if out_no_data_val is None else float(out_no_data_val)

    keep_attrs = [str(c) for c in attr_cols] if attr_cols else []
    missing_attrs = [c for c in keep_attrs if c not in work_gdf.columns]
    if missing_attrs:
        raise rsgispy.RSGISPyException(
            f"Attribute column(s) {missing_attrs} not found in point vector layer."
        )

    tmp_col_map: Dict[str, str] = {}
    ordered_cols: List[str] = []
    col_counter = 0

    for img_path, bands, col_names in resolved_imgs:
        tmp_fields: List[str] = []
        for col in col_names:
            tmp_name = f"__rsgis_f_{col_counter}__"
            col_counter += 1
            tmp_fields.append(tmp_name)
            tmp_col_map[tmp_name] = col
            ordered_cols.append(col)

        updated = ext_point_band_values_file(
            vec_file=work_gdf,
            vec_lyr=None,
            input_img=img_path,
            img_band=bands,
            min_thres=eff_min,
            max_thres=eff_max,
            out_no_data_val=eff_nd,
            out_field=tmp_fields,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
            out_vec_file=None,
        )
        if updated is None:
            raise rsgispy.RSGISPyException(
                f"Point pixel extraction returned no data for '{img_path}'."
            )
        work_gdf = updated

    data: Dict[str, Any] = {}
    for col in keep_attrs:
        data[col] = work_gdf[col].to_numpy()

    if include_coords:
        first_img_path = resolved_imgs[0][0]
        coord_geom = work_gdf.geometry
        with rasterio.open(first_img_path, "r") as src:
            if src.crs is not None and reproj_vec:
                img_crs = pyproj.CRS.from_user_input(src.crs)
                if work_gdf.crs is not None:
                    vec_crs = pyproj.CRS.from_user_input(work_gdf.crs)
                elif vec_def_epsg is not None:
                    vec_crs = pyproj.CRS.from_epsg(int(vec_def_epsg))
                else:
                    vec_crs = None
                if vec_crs is not None and vec_crs != img_crs:
                    coord_geom = coord_geom.set_crs(
                        vec_crs, allow_override=True
                    ).to_crs(img_crs)
        data["x"] = coord_geom.x.to_numpy(dtype=np.float64)
        data["y"] = coord_geom.y.to_numpy(dtype=np.float64)

    for tmp_name, target_col in tmp_col_map.items():
        data[target_col] = work_gdf[tmp_name].to_numpy()

    out_df = pd.DataFrame(data)
    if mask_nodata and len(out_df) > 0:
        valid = np.ones(len(out_df), dtype=bool)
        for col in ordered_cols:
            vals = out_df[col].to_numpy(dtype=np.float64)
            valid &= np.isfinite(vals)
            if not math.isnan(eff_nd):
                valid &= vals != eff_nd
        out_df = out_df.loc[valid].reset_index(drop=True)
    return out_df


def extract_polygons_samples(
    poly_gdf: gpd.GeoDataFrame,
    resolved_imgs: List[Tuple[str, List[int], List[str]]],
    attr_cols: Optional[Sequence[str]] = None,
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Extract polygon pixel samples using ``export_poly_pxls_to_df``,
    optionally preserving vector attribute columns ``attr_cols``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    keep_attrs = [str(c) for c in attr_cols] if attr_cols else []
    missing_attrs = [c for c in keep_attrs if c not in poly_gdf.columns]
    if missing_attrs:
        raise rsgispy.RSGISPyException(
            f"Attribute column(s) {missing_attrs} not found in polygon vector layer."
        )

    multi_img = len(resolved_imgs) > 1
    first_img_path, first_bands, first_cols = resolved_imgs[0]

    base_df = export_poly_pxls_to_df(
        vec_gdf=poly_gdf,
        input_img=first_img_path,
        img_bands=first_bands,
        out_file=None,
        band_names=first_cols,
        attr_cols=keep_attrs if keep_attrs else None,
        mask_nodata=mask_nodata,
        all_touched=all_touched,
        include_coords=include_coords or multi_img,
        reproj_vec=reproj_vec,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
    )
    if base_df is None:
        raise rsgispy.RSGISPyException(
            f"Polygon pixel extraction returned no data for '{first_img_path}'."
        )

    all_feature_cols = list(first_cols)
    if multi_img and len(base_df) > 0:
        with rasterio.open(first_img_path, "r") as src0:
            base_crs = src0.crs

        for img_path, bands, col_names in resolved_imgs[1:]:
            all_feature_cols.extend(col_names)
            if _grids_pixel_aligned(first_img_path, img_path):
                other_df = export_poly_pxls_to_df(
                    vec_gdf=poly_gdf,
                    input_img=img_path,
                    img_bands=bands,
                    out_file=None,
                    band_names=col_names,
                    mask_nodata=mask_nodata,
                    all_touched=all_touched,
                    include_coords=True,
                    reproj_vec=reproj_vec,
                    vec_def_epsg=vec_def_epsg,
                    use_pbar=use_pbar,
                )
                if other_df is None:
                    raise rsgispy.RSGISPyException(
                        f"Polygon pixel extraction returned no data for '{img_path}'."
                    )
                base_df["__xk__"] = np.round(base_df["x"].to_numpy(), 6)
                base_df["__yk__"] = np.round(base_df["y"].to_numpy(), 6)
                other_df["__xk__"] = np.round(other_df["x"].to_numpy(), 6)
                other_df["__yk__"] = np.round(other_df["y"].to_numpy(), 6)
                join_cols = ["poly_fid", "__xk__", "__yk__", *col_names]
                base_df = base_df.merge(
                    other_df[join_cols],
                    on=["poly_fid", "__xk__", "__yk__"],
                    how="inner",
                )
                base_df = base_df.drop(columns=["__xk__", "__yk__"])
            else:
                pts_gdf = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(
                        base_df["x"].to_numpy(), base_df["y"].to_numpy()
                    ),
                    crs=base_crs,
                )
                sampled_df = extract_points_samples(
                    pts_gdf=pts_gdf,
                    resolved_imgs=[(img_path, bands, col_names)],
                    attr_cols=None,
                    min_thres=min_thres,
                    max_thres=max_thres,
                    out_no_data_val=out_no_data_val,
                    mask_nodata=False,
                    include_coords=False,
                    reproj_vec=True,
                    vec_def_epsg=None,
                    use_pbar=use_pbar,
                )
                for col in col_names:
                    base_df[col] = sampled_df[col].to_numpy()
    elif multi_img:
        for _, _, col_names in resolved_imgs[1:]:
            all_feature_cols.extend(col_names)
            for col in col_names:
                base_df[col] = pd.Series(dtype="float64")

    if len(base_df) > 0 and (
        mask_nodata or min_thres is not None or max_thres is not None
    ):
        eff_min = -math.inf if min_thres is None else float(min_thres)
        eff_max = math.inf if max_thres is None else float(max_thres)
        eff_nd = math.nan if out_no_data_val is None else float(out_no_data_val)
        valid = np.ones(len(base_df), dtype=bool)
        for col in all_feature_cols:
            vals = base_df[col].to_numpy(dtype=np.float64)
            if mask_nodata:
                valid &= np.isfinite(vals)
                if not math.isnan(eff_nd):
                    valid &= vals != eff_nd
            valid &= (vals >= eff_min) & (vals <= eff_max)
        base_df = base_df.loc[valid].reset_index(drop=True)

    keep_cols: List[str] = list(keep_attrs)
    if include_coords:
        keep_cols.extend(["x", "y"])
    keep_cols.extend(all_feature_cols)
    return base_df[keep_cols].reset_index(drop=True)


def extract_vector_samples_from_gdf(
    vec_gdf: gpd.GeoDataFrame,
    resolved_imgs: List[Tuple[str, List[int], List[str]]],
    attr_cols: Optional[Sequence[str]] = None,
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    use_pbar: bool = True,
    label: str = "training vector",
) -> pd.DataFrame:
    """Dispatch Point vs Polygon/MultiPolygon pixel sample extraction."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    geom_types = set(vec_gdf.geom_type.dropna().unique())
    if len(vec_gdf) == 0 or not geom_types:
        raise rsgispy.RSGISPyException(
            f"Vector layer for {label} contains no valid geometries."
        )

    if geom_types.issubset({"Point"}):
        return extract_points_samples(
            pts_gdf=vec_gdf,
            resolved_imgs=resolved_imgs,
            attr_cols=attr_cols,
            min_thres=min_thres,
            max_thres=max_thres,
            out_no_data_val=out_no_data_val,
            mask_nodata=mask_nodata,
            include_coords=include_coords,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
        )
    if geom_types.issubset({"Polygon", "MultiPolygon"}):
        return extract_polygons_samples(
            poly_gdf=vec_gdf,
            resolved_imgs=resolved_imgs,
            attr_cols=attr_cols,
            min_thres=min_thres,
            max_thres=max_thres,
            out_no_data_val=out_no_data_val,
            mask_nodata=mask_nodata,
            all_touched=all_touched,
            include_coords=include_coords,
            reproj_vec=reproj_vec,
            vec_def_epsg=vec_def_epsg,
            use_pbar=use_pbar,
        )
    raise rsgispy.RSGISPyException(
        f"Unsupported geometry type(s) for {label}: "
        f"{sorted(geom_types)}. Expected Point or Polygon/MultiPolygon."
    )


def resolve_split_proportions(
    train_prop: Optional[float],
    test_prop: float,
    valid_prop: float,
) -> Tuple[float, float, float]:
    """Validate and resolve ``(train_prop, test_prop, valid_prop)``."""
    test_p = float(test_prop)
    valid_p = float(valid_prop)
    if test_p < 0.0 or valid_p < 0.0:
        raise rsgispy.RSGISPyException(
            "test_prop and valid_prop must be non-negative."
        )
    if train_prop is None:
        train_p = 1.0 - test_p - valid_p
    else:
        train_p = float(train_prop)
    if train_p < 0.0:
        raise rsgispy.RSGISPyException(
            f"Resolved train_prop ({train_p}) must be non-negative."
        )
    total = train_p + test_p + valid_p
    if total <= 0.0 or total > 1.0 + 1e-9:
        raise rsgispy.RSGISPyException(
            "Split proportions train_prop + test_prop + valid_prop must be "
            f"in (0, 1], got {total}."
        )
    return train_p, test_p, valid_p


def allocate_split_counts(
    n_total: int,
    train_p: float,
    test_p: float,
    valid_p: float,
    c_train: Optional[int] = None,
    c_test: Optional[int] = None,
    c_valid: Optional[int] = None,
    group_label: str = "samples",
) -> Tuple[int, int, int]:
    """Compute non-overlapping ``(n_train, n_test, n_valid)`` counts."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if any(c is not None for c in (c_train, c_test, c_valid)):
        n_tr = (
            c_train
            if c_train is not None
            else int(round(n_total * train_p))
        )
        n_te = (
            c_test
            if c_test is not None
            else int(round(n_total * test_p))
        )
        n_va = (
            c_valid
            if c_valid is not None
            else int(round(n_total * valid_p))
        )
        if n_tr + n_te + n_va > n_total:
            raise rsgispy.RSGISPyException(
                f"Requested split counts ({n_tr}, {n_te}, {n_va}) exceed "
                f"available {group_label} ({n_total})."
            )
        return n_tr, n_te, n_va

    if n_total == 0:
        return 0, 0, 0
    if n_total == 1:
        return (1, 0, 0) if train_p > 0.0 else (0, 1, 0)

    n_te = int(round(n_total * test_p)) if test_p > 0.0 else 0
    n_va = int(round(n_total * valid_p)) if valid_p > 0.0 else 0

    if n_te + n_va > n_total:
        excess = (n_te + n_va) - n_total
        reduce_va = min(n_va, excess)
        n_va -= reduce_va
        n_te = max(0, n_te - (excess - reduce_va))

    if train_p + test_p + valid_p >= 1.0 - 1e-6:
        n_tr = n_total - n_te - n_va
        if train_p > 0.0 and n_tr == 0:
            if n_va > 1 or (n_va == 1 and valid_p <= test_p):
                n_va -= 1
                n_tr += 1
            elif n_te > 0:
                n_te -= 1
                n_tr += 1
    else:
        n_tr = min(int(round(n_total * train_p)), n_total - n_te - n_va)

    return n_tr, n_te, n_va


def read_valid_roi_block(
    msk_ds: DatasetReader,
    msk_path: str,
    img_msk_val: int,
    band_specs: Sequence[Tuple[str, int, str]],
    datasets: Mapping[str, DatasetReader],
    grid: IntersectionGrid,
    out_win: Window,
    all_compatible: bool,
    use_df_input: bool,
    df_cols: Sequence[str],
    base_msk_col: int,
    base_msk_row: int,
) -> Tuple[np.ndarray, Optional[Union[pd.DataFrame, np.ndarray]]]:
    """Read mask + feature block and return valid ROI indices and feature matrix."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    win_h = int(out_win.height)
    win_w = int(out_win.width)
    msk_win = Window(
        base_msk_col + int(out_win.col_off),
        base_msk_row + int(out_win.row_off),
        win_w,
        win_h,
    )
    msk_arr = msk_ds.read(1, window=msk_win)
    roi_idx = np.flatnonzero(msk_arr.ravel() == img_msk_val)
    if roi_idx.size == 0:
        return np.empty(0, dtype=np.intp), None

    feat_arrays = read_feature_block(
        band_specs=band_specs,
        datasets=datasets,
        ref_ds=msk_ds,
        grid=grid,
        out_win=out_win,
        all_compatible=all_compatible,
        ref_path=msk_path,
    )
    valid_feats, valid_idx = extract_roi_feature_matrix(feat_arrays, roi_idx)
    if valid_idx.size == 0:
        return valid_idx, None
    x_valid: Union[pd.DataFrame, np.ndarray] = (
        pd.DataFrame(valid_feats, columns=list(df_cols))
        if use_df_input
        else valid_feats
    )
    return valid_idx, x_valid


def resolve_bootstrap_sample_size(
    n_total: int,
    sample_frac: float,
    sample_n_smps: Optional[int],
    bootstrap_n: int,
) -> int:
    """Validate bootstrap parameters and return sample size per iteration."""
    if bootstrap_n < 1:
        raise rsgispy.RSGISPyException("bootstrap_n must be at least 1.")
    if sample_n_smps is None:
        if sample_frac <= 0.0 or sample_frac > 1.0:
            raise rsgispy.RSGISPyException(
                "sample_frac must be within (0.0, 1.0]."
            )
        return max(1, int(n_total * sample_frac))
    if sample_n_smps < 1:
        raise rsgispy.RSGISPyException("sample_n_smps must be at least 1.")
    return min(int(sample_n_smps), n_total)


_BOOTSTRAP_PERCENTILES: Tuple[float, ...] = (
    0.5,
    2.5,
    5.0,
    50.0,
    95.0,
    97.5,
    99.5,
)


def summarise_bootstrap_interval(values: Sequence[float]) -> Dict[str, Any]:
    """Compute mean, median, and 90th/95th/99th percentile bounds."""
    arr = np.asarray(values, dtype=float)
    inters = np.percentile(arr, _BOOTSTRAP_PERCENTILES)
    return {
        "mean": float(np.mean(arr)),
        "median": float(inters[3]),
        "90th": [float(inters[2]), float(inters[4])],
        "95th": [float(inters[1]), float(inters[5])],
        "99th": [float(inters[0]), float(inters[6])],
    }
