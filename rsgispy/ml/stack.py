"""
Unified raster feature stack descriptor (:class:`ImageFeatureStack` and
:class:`BandSpec`) replacing separate ``ImageBandInfo`` / ``EnvVarInfo``
pipelines across :mod:`rsgispy.classification`, :mod:`rsgispy.regression`, and
:mod:`rsgispy.modelling`.

Provides built-in valid-mask generation, coordinate- and polygon-ID-preserving
sample extraction, on-the-fly tiled feature reading with optional
:class:`~rsgispy.ml.scaling.FeatureScaler` scaling, and JSON serialisation.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.io import DatasetReader
from rasterio.windows import Window

import rsgispy
from rsgispy.imageprocess import (
    IntersectionGrid,
    build_valid_mask,
    compute_spatial_intersection,
    ensure_file_exists,
    extract_roi_feature_matrix,
    get_default_block_size,
    grids_compatible,
    iter_dataset_windows,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageutils._helpers import normalise_creation_options
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.ml.scaling import FeatureScaler
from rsgispy.tools._ml_data import read_feature_block
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.vectorprocess import VectorInput, read_vec_gdf

__all__ = [
    "BandSpec",
    "ImageFeatureStack",
]


@dataclass(frozen=True)
class BandSpec:
    """
    Immutable descriptor for one input raster file (and selected bands) inside
    an :class:`ImageFeatureStack`.

    Unifies :class:`~rsgispy.imageutils.masking.ImageBandInfo` and
    :class:`~rsgispy.modelling._helper.EnvVarInfo`.

    :param file_name: Path to the raster file.
    :param name: Optional logical image/variable prefix.
    :param bands: Optional tuple of 1-based band indices (all bands when ``None``).
    :param band_names: Optional explicit feature names matching ``bands``.
    :param data_type: Optional variable type constant
                      (``rsgispy.VAR_TYPE_CONTINUOUS``,
                      ``rsgispy.VAR_TYPE_CATEGORICAL``, or
                      ``rsgispy.VAR_TYPE_UNDEFINED``).
    :param min_vld_val: Optional inclusive lower valid data threshold.
    :param max_vld_val: Optional inclusive upper valid data threshold.
    :param no_data_val: Optional explicit no-data override (uses raster header
                        nodata when ``None``).
    """

    file_name: str
    name: Optional[str] = None
    bands: Optional[Tuple[int, ...]] = None
    band_names: Optional[Tuple[str, ...]] = None
    data_type: int = rsgispy.VAR_TYPE_UNDEFINED
    min_vld_val: Optional[float] = None
    max_vld_val: Optional[float] = None
    no_data_val: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.file_name:
            raise rsgispy.RSGISPyException(
                "BandSpec requires a non-empty 'file_name'."
            )
        object.__setattr__(self, "file_name", os.fspath(self.file_name))
        if self.bands is not None:
            b_tuple = tuple(int(b) for b in self.bands)
            if not b_tuple:
                raise rsgispy.RSGISPyException(
                    "BandSpec 'bands' must be non-empty when provided."
                )
            object.__setattr__(self, "bands", b_tuple)
        if self.band_names is not None:
            n_tuple = tuple(str(n) for n in self.band_names)
            if any(not n.strip() for n in n_tuple):
                raise rsgispy.RSGISPyException(
                    "BandSpec 'band_names' must contain non-empty strings."
                )
            if self.bands is not None and len(n_tuple) != len(self.bands):
                raise rsgispy.RSGISPyException(
                    f"Length of band_names ({len(n_tuple)}) must match bands "
                    f"({len(self.bands)})."
                )
            object.__setattr__(self, "band_names", n_tuple)

    def to_image_band_info(self) -> ImageBandInfo:
        """Convert to a legacy :class:`~rsgispy.imageutils.masking.ImageBandInfo`."""
        return ImageBandInfo(
            file_name=self.file_name,
            name=self.name,
            bands=list(self.bands) if self.bands is not None else None,
            band_names=list(self.band_names) if self.band_names is not None else None,
        )

    @classmethod
    def from_image_band_info(cls, info: ImageBandInfo) -> "BandSpec":
        """Create a :class:`BandSpec` from an :class:`ImageBandInfo`."""
        if not info.file_name:
            raise rsgispy.RSGISPyException(
                "ImageBandInfo must have a non-empty file_name."
            )
        return cls(
            file_name=info.file_name,
            name=info.name,
            bands=tuple(info.bands) if info.bands is not None else None,
            band_names=tuple(info.band_names) if info.band_names is not None else None,
        )

    @classmethod
    def from_env_var_info(cls, env_info: Any) -> "BandSpec":
        """Create a :class:`BandSpec` from a :class:`~rsgispy.modelling._helper.EnvVarInfo`."""
        if not env_info.file:
            raise rsgispy.RSGISPyException(
                "EnvVarInfo must define a non-empty 'file' path."
            )
        var_name = env_info.name or os.path.splitext(os.path.basename(env_info.file))[0]
        return cls(
            file_name=env_info.file,
            name=var_name,
            bands=(int(env_info.band),),
            band_names=(str(var_name),),
            data_type=int(env_info.data_type),
            min_vld_val=env_info.min_vld_val,
            max_vld_val=env_info.max_vld_val,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "file_name": self.file_name,
            "name": self.name,
            "bands": list(self.bands) if self.bands is not None else None,
            "band_names": list(self.band_names) if self.band_names is not None else None,
            "data_type": int(self.data_type),
            "min_vld_val": self.min_vld_val,
            "max_vld_val": self.max_vld_val,
            "no_data_val": self.no_data_val,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BandSpec":
        """Construct a :class:`BandSpec` from a dictionary."""
        return cls(
            file_name=data["file_name"],
            name=data.get("name"),
            bands=tuple(data["bands"]) if data.get("bands") is not None else None,
            band_names=(
                tuple(data["band_names"])
                if data.get("band_names") is not None
                else None
            ),
            data_type=int(data.get("data_type", rsgispy.VAR_TYPE_UNDEFINED)),
            min_vld_val=data.get("min_vld_val"),
            max_vld_val=data.get("max_vld_val"),
            no_data_val=data.get("no_data_val"),
        )


def _coerce_band_spec(item: Any) -> BandSpec:
    """Coerce a :class:`BandSpec`, :class:`ImageBandInfo`, ``EnvVarInfo``, path, or dict."""
    if isinstance(item, BandSpec):
        return item
    if isinstance(item, ImageBandInfo):
        return BandSpec.from_image_band_info(item)
    if isinstance(item, (str, os.PathLike)):
        return BandSpec(file_name=os.fspath(item))
    if isinstance(item, Mapping):
        return BandSpec.from_dict(item)
    if hasattr(item, "file") and hasattr(item, "band") and hasattr(item, "data_type"):
        return BandSpec.from_env_var_info(item)
    raise rsgispy.RSGISPyException(
        f"Unsupported feature stack item type: {type(item).__name__}."
    )


class ImageFeatureStack:
    """
    Unified multi-raster feature stack combining band definitions, valid-mask
    generation, coordinate-preserving vector sample extraction, optional
    :class:`~rsgispy.ml.scaling.FeatureScaler` scaling, and tiled block reading.

    :param specs: Non-empty sequence of :class:`BandSpec`,
                  :class:`~rsgispy.imageutils.masking.ImageBandInfo`,
                  :class:`~rsgispy.modelling._helper.EnvVarInfo`, file paths,
                  or dictionaries.
    :param scaler: Optional :class:`~rsgispy.ml.scaling.FeatureScaler` applied
                   on the fly during tiled prediction or block iteration.

    .. code:: python

        from rsgispy.ml.stack import BandSpec, ImageFeatureStack
        from rsgispy.ml.scaling import FeatureScaler

        stack = ImageFeatureStack(
            [BandSpec("s2.tif", name="s2", bands=(1, 2, 3, 4)), BandSpec("dem.tif", name="dem")],
            scaler=FeatureScaler(method="zscore"),
        )
        stack.create_valid_mask("valid_mask.tif")
    """

    def __init__(
        self,
        specs: Sequence[Any],
        scaler: Optional[FeatureScaler] = None,
    ) -> None:
        if not specs or isinstance(specs, (str, bytes)):
            raise rsgispy.RSGISPyException(
                "ImageFeatureStack requires a non-empty sequence of BandSpec items."
            )
        self.specs: Tuple[BandSpec, ...] = tuple(_coerce_band_spec(s) for s in specs)
        self.scaler: Optional[FeatureScaler] = scaler

    @classmethod
    def from_image_band_infos(
        cls,
        img_band_info: Sequence[ImageBandInfo],
        scaler: Optional[FeatureScaler] = None,
    ) -> "ImageFeatureStack":
        """Construct an :class:`ImageFeatureStack` from ``ImageBandInfo`` objects."""
        return cls(img_band_info, scaler=scaler)

    @classmethod
    def from_env_vars(
        cls,
        env_vars: Union[Sequence[Any], Mapping[str, Any]],
        scaler: Optional[FeatureScaler] = None,
    ) -> "ImageFeatureStack":
        """Construct an :class:`ImageFeatureStack` from ``EnvVarInfo`` objects."""
        items = list(env_vars.values()) if isinstance(env_vars, Mapping) else list(env_vars)
        return cls(items, scaler=scaler)

    def to_image_band_infos(self) -> List[ImageBandInfo]:
        """Export specs as a list of :class:`ImageBandInfo` objects."""
        return [s.to_image_band_info() for s in self.specs]

    def resolve_band_specs(self) -> List[Tuple[str, int, str]]:
        """
        Validate all raster paths and return ordered ``(path, band_1based, col_name)``
        tuples with guaranteed unique feature names.
        """
        resolved: List[Tuple[str, int, str]] = []
        seen: set = set()
        seen_lower: set = set()
        for img_idx, spec in enumerate(self.specs):
            ensure_file_exists(spec.file_name)
            with rasterio.open(spec.file_name, "r") as src:
                n_bands = int(src.count)
            bands = (
                list(range(1, n_bands + 1))
                if spec.bands is None
                else list(spec.bands)
            )
            for b_idx in bands:
                validate_band_index(b_idx, n_bands, spec.file_name)

            if spec.band_names is not None:
                if len(spec.band_names) != len(bands):
                    raise rsgispy.RSGISPyException(
                        f"BandSpec for '{spec.file_name}' has {len(spec.band_names)} "
                        f"band_names for {len(bands)} bands."
                    )
                names = list(spec.band_names)
            else:
                prefix = (
                    spec.name.strip()
                    if spec.name and spec.name.strip()
                    else os.path.splitext(os.path.basename(spec.file_name))[0]
                    or f"img{img_idx + 1}"
                )
                names = [f"{prefix}_b{b_idx}" for b_idx in bands]

            for b_idx, col_name in zip(bands, names):
                if col_name in seen or col_name.lower() in seen_lower:
                    raise rsgispy.RSGISPyException(
                        f"Duplicate feature column name '{col_name}' in ImageFeatureStack."
                    )
                seen.add(col_name)
                seen_lower.add(col_name.lower())
                resolved.append((spec.file_name, int(b_idx), col_name))
        return resolved

    def resolve_band_nodata_list(self) -> List[Optional[float]]:
        """Return the effective no-data value for each expanded feature band."""
        out_nd: List[Optional[float]] = []
        for spec in self.specs:
            ensure_file_exists(spec.file_name)
            with rasterio.open(spec.file_name, "r") as src:
                bands = (
                    list(range(1, int(src.count) + 1))
                    if spec.bands is None
                    else list(spec.bands)
                )
                for b_idx in bands:
                    if spec.no_data_val is not None:
                        out_nd.append(float(spec.no_data_val))
                    else:
                        nd = src.nodatavals[b_idx - 1]
                        out_nd.append(float(nd) if nd is not None else None)
        return out_nd

    def _resolve_band_bounds_list(
        self,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Return ``(min_vld_val, max_vld_val)`` per expanded feature band."""
        out_bounds: List[Tuple[Optional[float], Optional[float]]] = []
        for spec in self.specs:
            with rasterio.open(spec.file_name, "r") as src:
                n_b = (
                    int(src.count)
                    if spec.bands is None
                    else len(spec.bands)
                )
            for _ in range(n_b):
                out_bounds.append((spec.min_vld_val, spec.max_vld_val))
        return out_bounds

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of expanded feature column names."""
        return [name for _, _, name in self.resolve_band_specs()]

    @property
    def n_features(self) -> int:
        """Total number of expanded feature bands."""
        return len(self.resolve_band_specs())

    def get_bbox(self) -> rsgispy.BBox:
        """Return the spatial intersection bounding box of all stack rasters."""
        paths = list(dict.fromkeys(s.file_name for s in self.specs))
        with ExitStack() as stack:
            datasets = {p: stack.enter_context(rasterio.open(p, "r")) for p in paths}
            first_ds = datasets[paths[0]]
            all_compat = all(grids_compatible(first_ds, ds) for ds in datasets.values())
            if all_compat:
                grid = compute_spatial_intersection(datasets)
            else:
                grid = IntersectionGrid.from_dataset(first_ds)
            left = grid.transform.c
            top = grid.transform.f
            right = left + grid.width * grid.transform.a
            bottom = top + grid.height * grid.transform.e
            epsg = grid.crs.to_epsg() if grid.crs is not None else None
            return rsgispy.BBox(
                west=min(left, right),
                east=max(left, right),
                south=min(bottom, top),
                north=max(bottom, top),
                epsg=epsg,
            )

    def create_valid_mask(
        self,
        output_img: Union[str, os.PathLike],
        block_size: Optional[int] = None,
        *,
        gdalformat: str = "GTIFF",
        datatype: Optional[int] = None,
        no_data_val: Optional[float] = None,
        calc_stats: bool = True,
        use_pbar: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Generate a binary valid-pixel mask (``1`` = valid across all stack
        bands, ``0`` = invalid/no-data/non-finite/out-of-range).

        :param output_img: Path to the output binary mask raster.
        :param block_size: Optional tile size in pixels.
        :param gdalformat: GDAL/rasterio output format (Default = ``'GTIFF'``).
        :param datatype: Optional ``rsgispy.TYPE_*`` constant
                         (Default = :data:`rsgispy.TYPE_8UINT`).
        :param no_data_val: Optional output nodata value (Default = ``0``).
        :param calc_stats: Populate thematic statistics/pyramids (Default = ``True``).
        :param use_pbar: Display a tqdm progress bar (Default = ``True``).
        :param creation_opts: Optional GDAL/rasterio creation options mapping.
        """
        # pylint: disable=too-many-locals
        out_path = os.fspath(output_img)
        ensure_parent_dir(out_path)
        out_dt = rsgispy.TYPE_8UINT if datatype is None else int(datatype)
        out_rio_dt = rsgispy.get_rasterio_datatype(out_dt)
        out_np_dt = np.dtype(rsgispy.get_numpy_datatype(out_dt))
        out_nd = 0 if no_data_val is None else no_data_val
        norm_opts = normalise_creation_options(creation_opts)

        band_specs = self.resolve_band_specs()
        band_nds = self.resolve_band_nodata_list()
        band_bounds = self._resolve_band_bounds_list()
        unique_paths = list(dict.fromkeys(p for p, _, _ in band_specs))
        ref_path = unique_paths[0]
        blk_sz = get_default_block_size() if block_size is None else int(block_size)

        with ExitStack() as stack:
            datasets = {
                p: stack.enter_context(rasterio.open(p, "r")) for p in unique_paths
            }
            ref_ds = datasets[ref_path]
            all_compat = all(grids_compatible(ref_ds, ds) for ds in datasets.values())
            grid = (
                compute_spatial_intersection(datasets)
                if all_compat
                else IntersectionGrid.from_dataset(
                    ref_ds, offsets={ref_path: (0, 0)}
                )
            )
            dst = open_output_dataset(
                stack=stack,
                output_img=out_path,
                gdalformat=gdalformat,
                out_rio_dtype=out_rio_dt,
                grid=grid,
                count=1,
                creation_opts=norm_opts,
            )
            dst.nodata = out_nd

            for win in iter_dataset_windows(
                grid.height, grid.width, block_size=blk_sz, use_pbar=use_pbar
            ):
                feat_arrays = read_feature_block(
                    band_specs=band_specs,
                    datasets=datasets,
                    ref_ds=ref_ds,
                    grid=grid,
                    out_win=win,
                    all_compatible=all_compat,
                    ref_path=ref_path,
                )
                valid_2d = np.ones((int(win.height), int(win.width)), dtype=bool)
                for (arr_2d, ds_nd), nd_val, (lo_v, hi_v) in zip(
                    feat_arrays, band_nds, band_bounds
                ):
                    eff_nd = nd_val if nd_val is not None else ds_nd
                    valid_2d &= build_valid_mask(arr_2d, eff_nd)
                    if lo_v is not None:
                        valid_2d &= arr_2d >= float(lo_v)
                    if hi_v is not None:
                        valid_2d &= arr_2d <= float(hi_v)
                dst.write(valid_2d.astype(out_np_dt), 1, window=win)

        if calc_stats:
            # pylint: disable=import-outside-toplevel
            from rsgispy.imageutils.stats import pop_thmt_img_stats

            pop_thmt_img_stats(
                out_path,
                add_clr_tab=False,
                calc_pyramids=True,
                ignore_zero=True,
                use_pbar=use_pbar,
            )

    def iter_blocks(
        self,
        in_msk_img: Optional[Union[str, os.PathLike]] = None,
        img_msk_val: int = 1,
        block_size: Optional[int] = None,
        scaler: Optional[FeatureScaler] = None,
        apply_scaler: bool = True,
        as_dataframe: bool = False,
        use_pbar: bool = True,
    ) -> Iterator[
        Tuple[
            Window,
            IntersectionGrid,
            np.ndarray,
            Union[np.ndarray, pd.DataFrame],
        ]
    ]:
        """
        Stream valid pixel feature matrices tile-by-tile across the stack.

        When ``in_msk_img`` is ``None``, valid pixels are determined on the fly
        from per-band nodata values, finiteness, and ``min_vld_val``/``max_vld_val``
        thresholds (zero intermediate mask file required).

        Yields ``(out_win, grid, valid_flat_idx, x_valid)`` per tile.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        # pylint: disable=too-many-locals
        band_specs = self.resolve_band_specs()
        feat_names = [name for _, _, name in band_specs]
        band_nds = self.resolve_band_nodata_list()
        band_bounds = self._resolve_band_bounds_list()
        eff_scaler = scaler if scaler is not None else self.scaler
        blk_sz = get_default_block_size() if block_size is None else int(block_size)

        unique_paths = list(dict.fromkeys(p for p, _, _ in band_specs))
        msk_path = os.fspath(in_msk_img) if in_msk_img is not None else None
        if msk_path is not None:
            ensure_file_exists(msk_path, "Input mask image")

        with ExitStack() as stack:
            datasets: Dict[str, DatasetReader] = {
                p: stack.enter_context(rasterio.open(p, "r")) for p in unique_paths
            }
            if msk_path is not None:
                msk_ds = (
                    datasets[msk_path]
                    if msk_path in datasets
                    else stack.enter_context(rasterio.open(msk_path, "r"))
                )
                ref_ds = msk_ds
                ref_path = msk_path
                all_ds = dict(datasets)
                all_ds[msk_path] = msk_ds
            else:
                msk_ds = None
                ref_path = unique_paths[0]
                ref_ds = datasets[ref_path]
                all_ds = datasets

            all_compat = all(grids_compatible(ref_ds, ds) for ds in all_ds.values())
            grid = (
                compute_spatial_intersection(all_ds)
                if all_compat
                else IntersectionGrid.from_dataset(
                    ref_ds, offsets={ref_path: (0, 0)}
                )
            )
            base_msk_col, base_msk_row = (
                grid.offsets[msk_path]
                if (msk_path is not None and msk_path in grid.offsets)
                else (0, 0)
            )

            for out_win in iter_dataset_windows(
                grid.height, grid.width, block_size=blk_sz, use_pbar=use_pbar
            ):
                win_h = int(out_win.height)
                win_w = int(out_win.width)
                if msk_ds is not None:
                    msk_win = Window(
                        base_msk_col + int(out_win.col_off),
                        base_msk_row + int(out_win.row_off),
                        win_w,
                        win_h,
                    )
                    msk_arr = msk_ds.read(1, window=msk_win)
                    roi_mask = msk_arr == img_msk_val
                else:
                    roi_mask = np.ones((win_h, win_w), dtype=bool)

                if not np.any(roi_mask):
                    empty_x: Union[np.ndarray, pd.DataFrame] = (
                        pd.DataFrame(columns=feat_names)
                        if as_dataframe
                        else np.empty((0, len(feat_names)), dtype=np.float64)
                    )
                    yield out_win, grid, np.empty(0, dtype=np.intp), empty_x
                    continue

                feat_arrays = read_feature_block(
                    band_specs=band_specs,
                    datasets=datasets,
                    ref_ds=ref_ds,
                    grid=grid,
                    out_win=out_win,
                    all_compatible=all_compat,
                    ref_path=ref_path,
                )
                for (arr_2d, ds_nd), nd_val, (lo_v, hi_v) in zip(
                    feat_arrays, band_nds, band_bounds
                ):
                    eff_nd = nd_val if nd_val is not None else ds_nd
                    roi_mask &= build_valid_mask(arr_2d, eff_nd)
                    if lo_v is not None:
                        roi_mask &= arr_2d >= float(lo_v)
                    if hi_v is not None:
                        roi_mask &= arr_2d <= float(hi_v)

                roi_idx = np.flatnonzero(roi_mask.ravel())
                if roi_idx.size == 0:
                    empty_x = (
                        pd.DataFrame(columns=feat_names)
                        if as_dataframe
                        else np.empty((0, len(feat_names)), dtype=np.float64)
                    )
                    yield out_win, grid, np.empty(0, dtype=np.intp), empty_x
                    continue

                valid_feats, valid_idx = extract_roi_feature_matrix(
                    feat_arrays, roi_idx
                )
                if valid_idx.size > 0 and apply_scaler and eff_scaler is not None:
                    valid_feats = np.asarray(
                        eff_scaler.transform(valid_feats), dtype=np.float64
                    )

                x_out: Union[np.ndarray, pd.DataFrame] = (
                    pd.DataFrame(valid_feats, columns=feat_names)
                    if as_dataframe
                    else valid_feats
                )
                yield out_win, grid, valid_idx, x_out

    def extract_points(
        self,
        vec_file: VectorInput,
        vec_lyr: Optional[str] = None,
        keep_cols: Optional[Sequence[str]] = None,
        drop_invalid: bool = True,
        coord_cols: Optional[Tuple[str, str]] = ("x", "y"),
    ) -> pd.DataFrame:
        """
        Extract feature values at point locations while preserving spatial
        coordinates ``coord_cols`` for spatial cross-validation.

        :param vec_file: Input point vector file path or GeoDataFrame.
        :param vec_lyr: Optional layer name when ``vec_file`` is a file path.
        :param keep_cols: Optional attribute columns from ``vec_file`` to retain
                          (retains all non-geometry columns when ``None``).
        :param drop_invalid: Drop rows with non-finite, nodata, or out-of-range
                             feature values (Default = ``True``).
        :param coord_cols: Optional ``(x_col, y_col)`` names storing projected
                           sample coordinates in the stack CRS (Default = ``('x', 'y')``).
        :return: :class:`pandas.DataFrame` with feature columns, requested
                 attribute columns, and coordinate columns.
        """
        # pylint: disable=too-many-locals
        band_specs = self.resolve_band_specs()
        band_nds = self.resolve_band_nodata_list()
        band_bounds = self._resolve_band_bounds_list()
        gdf = read_vec_gdf(vec_file, vec_lyr)
        if gdf.empty:
            cols = [name for _, _, name in band_specs]
            if coord_cols is not None:
                cols = list(coord_cols) + cols
            return pd.DataFrame(columns=cols)

        ref_path = band_specs[0][0]
        with rasterio.open(ref_path, "r") as ref_ds:
            ref_crs = ref_ds.crs
        if ref_crs is not None and gdf.crs is not None and gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)

        pts = gdf.geometry.centroid
        xs = np.asarray(pts.x, dtype=np.float64)
        ys = np.asarray(pts.y, dtype=np.float64)
        coords_list = list(zip(xs.tolist(), ys.tolist()))

        data: Dict[str, Any] = {}
        if keep_cols is None:
            attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
        else:
            attr_cols = list(keep_cols)
        for col in attr_cols:
            data[col] = gdf[col].to_numpy()
        if coord_cols is not None:
            data[coord_cols[0]] = xs
            data[coord_cols[1]] = ys

        unique_paths = list(dict.fromkeys(p for p, _, _ in band_specs))
        sampled_cache: Dict[str, np.ndarray] = {}
        with ExitStack() as stack:
            for path in unique_paths:
                ds = stack.enter_context(rasterio.open(path, "r"))
                if ds.crs is not None and ref_crs is not None and ds.crs != ref_crs:
                    pts_proj = gdf.to_crs(ds.crs).geometry.centroid
                    ds_coords = list(
                        zip(pts_proj.x.tolist(), pts_proj.y.tolist())
                    )
                else:
                    ds_coords = coords_list
                sampled = np.asarray(
                    list(ds.sample(ds_coords)), dtype=np.float64
                )
                sampled_cache[path] = sampled

        valid_rows = np.ones(len(gdf), dtype=bool)
        for (path, b_idx, col_name), nd_val, (lo_v, hi_v) in zip(
            band_specs, band_nds, band_bounds
        ):
            vals = sampled_cache[path][:, b_idx - 1]
            data[col_name] = vals
            valid_rows &= build_valid_mask(vals, nd_val)
            if lo_v is not None:
                valid_rows &= vals >= float(lo_v)
            if hi_v is not None:
                valid_rows &= vals <= float(hi_v)

        df = pd.DataFrame(data)
        if drop_invalid:
            df = df.loc[valid_rows].reset_index(drop=True)
        return df

    def extract_polygons(
        self,
        vec_file: VectorInput,
        vec_lyr: Optional[str] = None,
        keep_cols: Optional[Sequence[str]] = None,
        poly_id_col: str = "poly_fid",
        drop_invalid: bool = True,
        coord_cols: Optional[Tuple[str, str]] = ("x", "y"),
        all_touched: bool = False,
    ) -> pd.DataFrame:
        """
        Extract pixel-level features inside polygon geometries while preserving
        polygon group IDs (``poly_id_col``) and pixel centre coordinates
        (``coord_cols``) for group and spatial cross-validation.

        :param vec_file: Input polygon vector file path or GeoDataFrame.
        :param vec_lyr: Optional layer name when ``vec_file`` is a file path.
        :param keep_cols: Optional attribute columns from ``vec_file`` to attach.
        :param poly_id_col: Output column name storing integer polygon index
                            (Default = ``'poly_fid'``).
        :param drop_invalid: Drop invalid/nodata/out-of-range pixels
                             (Default = ``True``).
        :param coord_cols: Optional ``(x_col, y_col)`` pixel-centre coordinate
                           column names (Default = ``('x', 'y')``).
        :param all_touched: Include all pixels touched by polygon boundaries
                            (Default = ``False``).
        :return: :class:`pandas.DataFrame` of extracted pixels.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        # pylint: disable=too-many-locals
        band_specs = self.resolve_band_specs()
        band_nds = self.resolve_band_nodata_list()
        band_bounds = self._resolve_band_bounds_list()
        feat_names = [name for _, _, name in band_specs]
        gdf = read_vec_gdf(vec_file, vec_lyr)
        if gdf.empty:
            cols = [poly_id_col] + (list(coord_cols) if coord_cols else []) + feat_names
            return pd.DataFrame(columns=cols)

        unique_paths = list(dict.fromkeys(p for p, _, _ in band_specs))
        ref_path = unique_paths[0]
        if keep_cols is None:
            attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
        else:
            attr_cols = list(keep_cols)

        frames: List[pd.DataFrame] = []
        with ExitStack() as stack:
            datasets = {
                p: stack.enter_context(rasterio.open(p, "r")) for p in unique_paths
            }
            ref_ds = datasets[ref_path]
            if ref_ds.crs is not None and gdf.crs is not None and gdf.crs != ref_ds.crs:
                gdf = gdf.to_crs(ref_ds.crs)
            all_compat = all(grids_compatible(ref_ds, ds) for ds in datasets.values())
            grid = (
                compute_spatial_intersection(datasets)
                if all_compat
                else IntersectionGrid.from_dataset(
                    ref_ds, offsets={ref_path: (0, 0)}
                )
            )

            for poly_idx, row in enumerate(gdf.itertuples(index=False)):
                geom = getattr(row, gdf.geometry.name)
                if geom is None or geom.is_empty:
                    continue
                minx, miny, maxx, maxy = geom.bounds
                r0, c0 = rasterio.transform.rowcol(grid.transform, minx, maxy)
                r1, c1 = rasterio.transform.rowcol(grid.transform, maxx, miny)
                col_start = max(0, int(min(c0, c1)))
                col_end = min(grid.width, int(max(c0, c1)) + 1)
                row_start = max(0, int(min(r0, r1)))
                row_end = min(grid.height, int(max(r0, r1)) + 1)
                if col_end <= col_start or row_end <= row_start:
                    continue

                win = Window(
                    col_start, row_start, col_end - col_start, row_end - row_start
                )
                win_tf = rasterio.windows.transform(win, grid.transform)
                inside = geometry_mask(
                    [geom],
                    out_shape=(int(win.height), int(win.width)),
                    transform=win_tf,
                    invert=True,
                    all_touched=all_touched,
                )
                if not np.any(inside):
                    continue

                feat_arrays = read_feature_block(
                    band_specs=band_specs,
                    datasets=datasets,
                    ref_ds=ref_ds,
                    grid=grid,
                    out_win=win,
                    all_compatible=all_compat,
                    ref_path=ref_path,
                )
                if drop_invalid:
                    for (arr_2d, ds_nd), nd_val, (lo_v, hi_v) in zip(
                        feat_arrays, band_nds, band_bounds
                    ):
                        eff_nd = nd_val if nd_val is not None else ds_nd
                        inside &= build_valid_mask(arr_2d, eff_nd)
                        if lo_v is not None:
                            inside &= arr_2d >= float(lo_v)
                        if hi_v is not None:
                            inside &= arr_2d <= float(hi_v)
                rr, cc = np.where(inside)
                if rr.size == 0:
                    continue

                rec: Dict[str, Any] = {
                    poly_id_col: np.full(rr.size, poly_idx, dtype=np.int64)
                }
                for col in attr_cols:
                    rec[col] = np.repeat(gdf[col].iloc[poly_idx], rr.size)
                if coord_cols is not None:
                    px_x, px_y = rasterio.transform.xy(
                        win_tf, rr, cc, offset="center"
                    )
                    rec[coord_cols[0]] = np.asarray(px_x, dtype=np.float64)
                    rec[coord_cols[1]] = np.asarray(px_y, dtype=np.float64)
                for col_name, (arr_2d, _) in zip(feat_names, feat_arrays):
                    rec[col_name] = arr_2d[rr, cc].astype(np.float64, copy=False)
                frames.append(pd.DataFrame(rec))

        if not frames:
            cols = [poly_id_col] + (list(coord_cols) if coord_cols else []) + feat_names
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise stack specification and optional scaler to a JSON-safe dict."""
        return {
            "version": 1,
            "specs": [s.to_dict() for s in self.specs],
            "scaler": self.scaler.to_dict() if self.scaler is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageFeatureStack":
        """Reconstruct an :class:`ImageFeatureStack` from a dictionary."""
        if not isinstance(data, Mapping) or "specs" not in data:
            raise rsgispy.RSGISPyException(
                "Invalid ImageFeatureStack dictionary: missing 'specs'."
            )
        specs = [BandSpec.from_dict(s) for s in data["specs"]]
        scaler_data = data.get("scaler")
        scaler = FeatureScaler.from_dict(scaler_data) if scaler_data else None
        return cls(specs=specs, scaler=scaler)

    def save(self, json_file: Union[str, os.PathLike]) -> None:
        """Save this :class:`ImageFeatureStack` to a JSON file."""
        out_path = os.fspath(json_file)
        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, json_file: Union[str, os.PathLike]) -> "ImageFeatureStack":
        """Load an :class:`ImageFeatureStack` from a JSON file."""
        in_path = os.fspath(json_file)
        ensure_file_exists(in_path, "ImageFeatureStack JSON file")
        with open(in_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
