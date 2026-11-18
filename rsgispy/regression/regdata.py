"""
Regression data structures, configuration I/O, sample extraction from point or
polygon vector layers, and train/test/validation splitting (defaulting to
Snappy-compressed Parquet with support for other pandas-supported tabular
formats).
"""

import os
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

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageutils.masking import ImageBandInfo
from rsgispy.tools._ml_data import (
    allocate_split_counts,
    extract_vector_samples_from_gdf,
    load_samples_df,
    normalise_img_band_info_list as _normalise_img_band_info_list,
    read_ml_samples,
    resolve_img_band_info as _shared_resolve_img_band_info,
    resolve_rng,
    resolve_split_proportions,
    write_ml_samples,
)
from rsgispy.tools.files import ensure_parent_dir
from rsgispy.tools.jsonio import read_json_to_dict, write_dict_to_json
from rsgispy.vectorprocess import VectorInput, read_vec_gdf

__all__ = [
    "ImageBandInfo",
    "RegressionConfig",
    "normalise_target_cols",
    "resolve_img_band_info",
    "write_reg_config",
    "read_reg_config",
    "write_reg_samples",
    "read_reg_samples",
    "extract_reg_training_samples",
    "split_reg_samples",
    "extract_and_split_reg_samples",
]


def normalise_target_cols(
    target_cols: Union[str, Sequence[str]],
) -> List[str]:
    """
    Normalise and validate one or more continuous regression target column
    names into a list of unique non-empty strings.

    :param target_cols: Single target column name or sequence of target column
                        names (for multivariate regression).
    :return: Ordered list of validated target column names.

    .. code:: python

        import rsgispy.regression.regdata as rd

        cols = rd.normalise_target_cols(["canopy_cover", "canopy_height"])
    """
    if isinstance(target_cols, str):
        raw_list = [target_cols]
    elif isinstance(target_cols, Sequence) and not isinstance(
        target_cols, bytes
    ):
        raw_list = list(target_cols)
    else:
        raise rsgispy.RSGISPyException(
            "target_cols must be a target column name or a sequence of names."
        )

    if not raw_list:
        raise rsgispy.RSGISPyException(
            "target_cols must specify at least one target column."
        )

    out_cols: List[str] = []
    seen: set = set()
    for raw in raw_list:
        if not isinstance(raw, str) or not raw.strip():
            raise rsgispy.RSGISPyException(
                "All target_cols entries must be non-empty strings."
            )
        col = raw.strip()
        if col in seen:
            raise rsgispy.RSGISPyException(
                f"Duplicate target column name '{col}' in target_cols."
            )
        seen.add(col)
        out_cols.append(col)
    return out_cols


def resolve_img_band_info(
    img_band_info: Sequence[ImageBandInfo],
) -> List[Tuple[str, List[int], List[str]]]:
    """
    Validate input raster paths and resolve 1-based band lists and unique
    output feature column names for each :class:`ImageBandInfo`.

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects.
    :return: List of ``(file_name, bands_list, band_col_names)`` tuples.

    .. code:: python

        import rsgispy.regression.regdata as rd

        resolved = rd.resolve_img_band_info([
            rd.ImageBandInfo("landsat.tif", name="lsat", bands=[1, 2, 3]),
        ])
    """
    return _shared_resolve_img_band_info(img_band_info)


class RegressionConfig:
    """
    Container storing image/band definitions (:class:`ImageBandInfo`), target
    attribute column names (univariate or multivariate), and optional reference
    vector path with JSON serialisation.

    :param img_band_info: Sequence of :class:`ImageBandInfo` objects defining
                          input predictor rasters and bands.
    :param target_cols: Single target attribute name (e.g.,
                        ``'canopy_cover'``) or sequence of target names (e.g.,
                        ``['canopy_cover', 'canopy_height']``).
    :param vec_file: Optional vector file path (points or polygons) containing
                     reference target attribute values.
    :param vec_lyr: Optional vector layer name within ``vec_file``.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        img_band_info: Sequence[Union[ImageBandInfo, Mapping[str, Any]]],
        target_cols: Union[str, Sequence[str]],
        vec_file: Optional[Union[str, os.PathLike]] = None,
        vec_lyr: Optional[str] = None,
    ):
        self.img_band_info: List[ImageBandInfo] = _normalise_img_band_info_list(
            img_band_info
        )
        self.target_cols: List[str] = normalise_target_cols(target_cols)
        self.vec_file: Optional[str] = (
            os.fspath(vec_file) if vec_file is not None else None
        )
        self.vec_lyr: Optional[str] = vec_lyr

    @property
    def n_targets(self) -> int:
        """Number of configured regression target variables."""
        return len(self.target_cols)

    @property
    def is_multivariate(self) -> bool:
        """Return ``True`` when two or more target variables are configured."""
        return len(self.target_cols) > 1

    def get_band_names(self) -> List[str]:
        """
        Resolve and return the ordered list of predictor column names across
        all configured :class:`ImageBandInfo` items.
        """
        resolved = resolve_img_band_info(self.img_band_info)
        names: List[str] = []
        for _, _, col_names in resolved:
            names.extend(col_names)
        return names

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "version": 1,
            "target_cols": list(self.target_cols),
            "vec_file": self.vec_file,
            "vec_lyr": self.vec_lyr,
            "img_band_info": [info.to_dict() for info in self.img_band_info],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegressionConfig":
        """Construct a :class:`RegressionConfig` from a dictionary."""
        if not isinstance(data, Mapping):
            raise rsgispy.RSGISPyException(
                "Regression configuration data must be a dictionary."
            )
        if "img_band_info" not in data or "target_cols" not in data:
            raise rsgispy.RSGISPyException(
                "Regression configuration must contain 'img_band_info' and "
                "'target_cols'."
            )
        return cls(
            img_band_info=data["img_band_info"],
            target_cols=data["target_cols"],
            vec_file=data.get("vec_file"),
            vec_lyr=data.get("vec_lyr"),
        )

    def save(self, out_json_file: Union[str, os.PathLike]) -> None:
        """Save this configuration to ``out_json_file``."""
        write_reg_config(out_json_file, self)

    @classmethod
    def load(cls, in_json_file: Union[str, os.PathLike]) -> "RegressionConfig":
        """Load a :class:`RegressionConfig` from ``in_json_file``."""
        return read_reg_config(in_json_file)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegressionConfig):
            return NotImplemented
        return (
            self.img_band_info == other.img_band_info
            and self.target_cols == other.target_cols
            and self.vec_file == other.vec_file
            and self.vec_lyr == other.vec_lyr
        )

    def __repr__(self) -> str:
        return (
            f"RegressionConfig(img_band_info={self.img_band_info!r}, "
            f"target_cols={self.target_cols!r}, vec_file={self.vec_file!r}, "
            f"vec_lyr={self.vec_lyr!r})"
        )


def _coerce_config(
    img_band_info: Union[RegressionConfig, Sequence[ImageBandInfo]],
    target_cols: Optional[Union[str, Sequence[str]]] = None,
    vec_file: Optional[Union[str, os.PathLike]] = None,
    vec_lyr: Optional[str] = None,
) -> RegressionConfig:
    """Coerce a RegressionConfig or separate parameters into RegressionConfig."""
    if isinstance(img_band_info, RegressionConfig):
        if target_cols is not None:
            raise rsgispy.RSGISPyException(
                "Do not pass target_cols separately when img_band_info is a "
                "RegressionConfig instance."
            )
        return RegressionConfig(
            img_band_info=img_band_info.img_band_info,
            target_cols=img_band_info.target_cols,
            vec_file=(
                os.fspath(vec_file)
                if vec_file is not None
                else img_band_info.vec_file
            ),
            vec_lyr=vec_lyr if vec_lyr is not None else img_band_info.vec_lyr,
        )
    if target_cols is None:
        raise rsgispy.RSGISPyException(
            "target_cols must be provided when img_band_info is not a "
            "RegressionConfig instance."
        )
    return RegressionConfig(
        img_band_info=img_band_info,
        target_cols=target_cols,
        vec_file=vec_file,
        vec_lyr=vec_lyr,
    )


def write_reg_config(
    out_json_file: Union[str, os.PathLike],
    img_band_info: Union[RegressionConfig, Sequence[ImageBandInfo]],
    target_cols: Optional[Union[str, Sequence[str]]] = None,
    vec_file: Optional[Union[str, os.PathLike]] = None,
    vec_lyr: Optional[str] = None,
) -> RegressionConfig:
    """
    Save the regression image/band and target variable configuration to a JSON
    file on disk.

    :param out_json_file: Path to the output JSON configuration file.
    :param img_band_info: Either a :class:`RegressionConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param target_cols: Target attribute column name or sequence (required when
                        ``img_band_info`` is a sequence of :class:`ImageBandInfo`).
    :param vec_file: Optional reference vector file path.
    :param vec_lyr: Optional vector layer name.
    :return: The validated :class:`RegressionConfig` written to disk.

    .. code:: python

        import rsgispy.regression.regdata as rd

        img_info = [rd.ImageBandInfo("landsat.tif", name="lsat", bands=[1, 2, 3])]
        rd.write_reg_config(
            "reg_config.json",
            img_info,
            target_cols="canopy_cover",
            vec_file="canopy_plots.gpkg",
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg = _coerce_config(
        img_band_info,
        target_cols=target_cols,
        vec_file=vec_file,
        vec_lyr=vec_lyr,
    )
    out_path = ensure_parent_dir(out_json_file)
    write_dict_to_json(cfg.to_dict(), out_path)
    return cfg


def read_reg_config(
    in_json_file: Union[str, os.PathLike],
) -> RegressionConfig:
    """
    Read a regression configuration JSON file from disk and return a validated
    :class:`RegressionConfig` instance.

    :param in_json_file: Path to the input JSON configuration file.
    :return: Loaded :class:`RegressionConfig` instance.

    .. code:: python

        import rsgispy.regression.regdata as rd

        cfg = rd.read_reg_config("reg_config.json")
    """
    in_path = os.fspath(in_json_file)
    ensure_file_exists(in_path, label="Regression config")
    raw_dict = read_json_to_dict(in_path)
    return RegressionConfig.from_dict(raw_dict)


def write_reg_samples(
    samples_df: pd.DataFrame,
    out_file: Union[str, os.PathLike],
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
) -> None:
    """
    Write a regression sample :class:`pandas.DataFrame` to disk in Parquet
    (using Snappy compression by default) or any other pandas-supported format
    handled by :func:`rsgispy.tools.tables.write_dataframe_table`.

    :param samples_df: Sample :class:`pandas.DataFrame` to save.
    :param out_file: Output table file path (e.g., ``'train.parquet'``,
                     ``'train.csv'``).
    :param out_format: Optional output table format overriding extension
                       detection.
    :param out_compress: Compression setting passed to
                         :func:`rsgispy.tools.tables.write_dataframe_table`
                         (Default = ``'snappy'`` for Parquet).

    .. code:: python

        import pandas as pd
        import rsgispy.regression.regdata as rd

        df = pd.DataFrame({"canopy_cover": [45.2], "b1": [0.12], "b2": [0.34]})
        rd.write_reg_samples(df, "train.parquet")
    """
    write_ml_samples(
        samples_df=samples_df,
        out_file=out_file,
        out_format=out_format,
        out_compress=out_compress,
    )


def read_reg_samples(
    in_file: Union[str, os.PathLike],
    in_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a regression sample table file (Parquet or any other format supported
    by :func:`rsgispy.tools.tables.read_dataframe_table`) into a
    :class:`pandas.DataFrame`.

    :param in_file: Path to the input sample file.
    :param in_format: Optional input format overriding extension detection.
    :return: Loaded :class:`pandas.DataFrame`.

    .. code:: python

        import rsgispy.regression.regdata as rd

        train_df = rd.read_reg_samples("train.parquet")
    """
    return read_ml_samples(in_file, in_format=in_format)


def extract_reg_training_samples(
    img_band_info: Union[RegressionConfig, Sequence[ImageBandInfo]],
    target_cols: Optional[Union[str, Sequence[str]]] = None,
    vec_file: Optional[VectorInput] = None,
    vec_lyr: Optional[str] = None,
    out_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    drop_nonfinite_targets: bool = True,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
    use_pbar: bool = True,
) -> pd.DataFrame:
    """
    Extract raster predictor band values and continuous target attribute(s) from
    a point or polygon reference vector layer, returning a
    :class:`pandas.DataFrame` and optionally saving to Parquet or CSV.

    Supports both univariate (single target column) and multivariate (multiple
    target columns) regression workflows.

    :param img_band_info: Either a :class:`RegressionConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param target_cols: Target attribute column name or sequence of names
                        (required when ``img_band_info`` is a sequence).
    :param vec_file: Reference vector file path or
                     :class:`geopandas.GeoDataFrame` (overrides
                     ``RegressionConfig.vec_file`` when provided).
    :param vec_lyr: Optional vector layer name within ``vec_file``.
    :param out_file: Optional output table file path (e.g.,
                     ``'all_samples.parquet'``).
    :param out_format: Optional output table format overriding extension.
    :param out_compress: Compression setting passed to :func:`write_reg_samples`
                         (Default = ``'snappy'`` for Parquet).
    :param min_thres: Optional inclusive lower valid pixel threshold.
    :param max_thres: Optional inclusive upper valid pixel threshold.
    :param out_no_data_val: Optional fill/nodata sentinel value excluded when
                            ``mask_nodata=True``.
    :param mask_nodata: If ``True`` (default), drop rows containing nodata or
                        non-finite band values.
    :param all_touched: For polygon vectors, passed to
                        :func:`rsgispy.zonalstats.polygon.export_poly_pxls_to_df`.
    :param include_coords: Include ``x`` and ``y`` sample coordinates
                           (Default = ``False``).
    :param reproj_vec: Reproject vector geometries on the fly if CRSs differ.
    :param vec_def_epsg: Optional EPSG code overriding undefined vector CRS.
    :param drop_nonfinite_targets: Drop rows whose target value(s) are NaN or
                                   non-finite (Default = ``True``).
    :param target_min: Optional inclusive lower filter bound applied to all
                       target columns.
    :param target_max: Optional inclusive upper filter bound applied to all
                       target columns.
    :param use_pbar: Display progress bars during extraction (Default = ``True``).
    :return: :class:`pandas.DataFrame` containing target columns, optional
             coordinates, and extracted band feature columns.

    .. code:: python

        import rsgispy.regression.regdata as rd

        cfg = rd.read_reg_config("reg_config.json")
        samples_df = rd.extract_reg_training_samples(
            cfg, out_file="all_samples.parquet"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    cfg = _coerce_config(
        img_band_info,
        target_cols=target_cols,
        vec_file=vec_file if isinstance(vec_file, (str, os.PathLike)) else None,
        vec_lyr=vec_lyr,
    )
    eff_vec: Optional[VectorInput] = vec_file if vec_file is not None else cfg.vec_file
    if eff_vec is None:
        raise rsgispy.RSGISPyException(
            "A reference vec_file (file path or GeoDataFrame) must be supplied "
            "either in RegressionConfig or via the vec_file parameter."
        )
    if isinstance(eff_vec, (str, os.PathLike)):
        ensure_file_exists(os.fspath(eff_vec), label="Reference regression vector")
        vec_gdf = read_vec_gdf(eff_vec, vec_lyr=cfg.vec_lyr, copy=True)
    else:
        vec_gdf = read_vec_gdf(eff_vec, vec_lyr=cfg.vec_lyr, copy=True)

    resolved_imgs = resolve_img_band_info(cfg.img_band_info)
    all_feature_cols = [col for _, _, cols in resolved_imgs for col in cols]

    out_df = extract_vector_samples_from_gdf(
        vec_gdf=vec_gdf,
        resolved_imgs=resolved_imgs,
        attr_cols=cfg.target_cols,
        min_thres=min_thres,
        max_thres=max_thres,
        out_no_data_val=out_no_data_val,
        mask_nodata=mask_nodata,
        all_touched=all_touched,
        include_coords=include_coords,
        reproj_vec=reproj_vec,
        vec_def_epsg=vec_def_epsg,
        use_pbar=use_pbar,
        label="regression targets",
    )

    for t_col in cfg.target_cols:
        try:
            out_df[t_col] = pd.to_numeric(out_df[t_col], errors="raise").astype(
                np.float64
            )
        except Exception as err:
            raise rsgispy.RSGISPyException(
                f"Target column '{t_col}' must contain numeric values: {err}"
            ) from err

    if len(out_df) > 0 and (
        drop_nonfinite_targets or target_min is not None or target_max is not None
    ):
        valid = np.ones(len(out_df), dtype=bool)
        for t_col in cfg.target_cols:
            t_vals = out_df[t_col].to_numpy(dtype=np.float64)
            if drop_nonfinite_targets:
                valid &= np.isfinite(t_vals)
            if target_min is not None:
                valid &= t_vals >= float(target_min)
            if target_max is not None:
                valid &= t_vals <= float(target_max)
        out_df = out_df.loc[valid].reset_index(drop=True)

    coord_cols = ["x", "y"] if include_coords else []
    ordered_cols = list(cfg.target_cols) + coord_cols + all_feature_cols
    out_df = out_df[ordered_cols].reset_index(drop=True)

    if out_file is not None:
        write_reg_samples(
            out_df,
            out_file,
            out_format=out_format,
            out_compress=out_compress,
        )

    return out_df


def _split_index_group(
    idx_arr: np.ndarray,
    train_p: float,
    test_p: float,
    valid_p: float,
    c_train: Optional[int],
    c_test: Optional[int],
    c_valid: Optional[int],
    max_cap: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle indices and partition into train/test/valid index lists."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pos = idx_arr.copy()
    rng.shuffle(pos)
    if max_cap is not None and len(pos) > max_cap:
        pos = pos[:max_cap]
    n_tr, n_te, n_va = allocate_split_counts(
        len(pos),
        train_p,
        test_p,
        valid_p,
        c_train,
        c_test,
        c_valid,
        group_label="regression samples",
    )
    te_end = n_tr + n_te
    va_end = te_end + n_va
    return (
        pos[:n_tr].tolist(),
        pos[n_tr:te_end].tolist(),
        pos[te_end:va_end].tolist(),
    )


def split_reg_samples(
    samples_df: Union[pd.DataFrame, str, os.PathLike],
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    max_smpls: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_valid: Optional[int] = None,
    stratify_col: Optional[str] = None,
    n_bins: int = 5,
    rnd_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a regression sample table into training, testing, and validation
    DataFrames, optionally stratifying across quantile bins of a continuous
    target column and saving each split to disk.

    :param samples_df: Input sample :class:`pandas.DataFrame` or table path.
    :param train_prop: Proportion for training (defaults to
                       ``1.0 - test_prop - valid_prop``).
    :param test_prop: Proportion for testing (Default = ``0.2``).
    :param valid_prop: Proportion for validation (Default = ``0.2``).
    :param out_train_file: Optional output path for training samples.
    :param out_test_file: Optional output path for testing samples.
    :param out_valid_file: Optional output path for validation samples.
    :param out_format: Optional output table format overriding extension.
    :param out_compress: Compression setting passed to :func:`write_reg_samples`
                         (Default = ``'snappy'`` for Parquet).
    :param max_smpls: Optional upper sample limit applied prior to splitting.
    :param n_train: Optional explicit training sample count.
    :param n_test: Optional explicit testing sample count.
    :param n_valid: Optional explicit validation sample count.
    :param stratify_col: Optional target column name used to perform quantile-
                         binned stratified splitting across its distribution.
    :param n_bins: Number of quantile bins when ``stratify_col`` is specified
                   (Default = ``5``).
    :param rnd_seed: Optional random seed for reproducible splits.
    :return: Tuple ``(train_df, test_df, valid_df)`` of split DataFrames.

    .. code:: python

        import rsgispy.regression.regdata as rd

        train_df, test_df, valid_df = rd.split_reg_samples(
            samples_df,
            test_prop=0.2,
            valid_prop=0.2,
            stratify_col="canopy_cover",
            out_train_file="train.parquet",
            out_test_file="test.parquet",
            out_valid_file="valid.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    df = load_samples_df(samples_df, "samples_df")
    train_p, test_p, valid_p = resolve_split_proportions(
        train_prop, test_prop, valid_prop
    )
    rng = resolve_rng(rnd_seed)

    for label, val in (
        ("max_smpls", max_smpls),
        ("n_train", n_train),
        ("n_test", n_test),
        ("n_valid", n_valid),
    ):
        if val is not None and (isinstance(val, bool) or int(val) < 0):
            raise rsgispy.RSGISPyException(f"{label} must be >= 0, got {val!r}.")

    all_pos = df.index.to_numpy().copy()
    if max_smpls is not None and len(all_pos) > int(max_smpls):
        rng.shuffle(all_pos)
        work_df = df.loc[all_pos[: int(max_smpls)]]
    else:
        work_df = df

    use_binned = (
        stratify_col is not None
        and n_train is None
        and n_test is None
        and n_valid is None
        and int(n_bins) >= 2
        and len(work_df) >= int(n_bins) * 2
    )
    if stratify_col is not None and stratify_col not in work_df.columns:
        raise rsgispy.RSGISPyException(
            f"Stratification column '{stratify_col}' was not found in "
            f"samples_df columns {list(work_df.columns)}."
        )

    if use_binned and stratify_col is not None:
        bins = pd.qcut(
            work_df[stratify_col],
            q=int(n_bins),
            labels=False,
            duplicates="drop",
        )
        if bins.notna().all() and int(bins.nunique()) >= 2:
            train_idx: List[int] = []
            test_idx: List[int] = []
            valid_idx: List[int] = []
            for _, grp in work_df.groupby(bins, sort=False):
                tr_g, te_g, va_g = _split_index_group(
                    grp.index.to_numpy(),
                    train_p,
                    test_p,
                    valid_p,
                    None,
                    None,
                    None,
                    None,
                    rng,
                )
                train_idx.extend(tr_g)
                test_idx.extend(te_g)
                valid_idx.extend(va_g)
        else:
            train_idx, test_idx, valid_idx = _split_index_group(
                work_df.index.to_numpy(),
                train_p,
                test_p,
                valid_p,
                n_train,
                n_test,
                n_valid,
                None,
                rng,
            )
    else:
        train_idx, test_idx, valid_idx = _split_index_group(
            work_df.index.to_numpy(),
            train_p,
            test_p,
            valid_p,
            n_train,
            n_test,
            n_valid,
            None,
            rng,
        )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    valid_df = df.loc[valid_idx].reset_index(drop=True)

    for df_part, out_path in (
        (train_df, out_train_file),
        (test_df, out_test_file),
        (valid_df, out_valid_file),
    ):
        if out_path is not None:
            write_reg_samples(
                df_part,
                out_path,
                out_format=out_format,
                out_compress=out_compress,
            )

    return train_df, test_df, valid_df


def extract_and_split_reg_samples(
    img_band_info: Union[RegressionConfig, Sequence[ImageBandInfo]],
    target_cols: Optional[Union[str, Sequence[str]]] = None,
    vec_file: Optional[VectorInput] = None,
    vec_lyr: Optional[str] = None,
    train_prop: Optional[float] = None,
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    out_train_file: Optional[Union[str, os.PathLike]] = None,
    out_test_file: Optional[Union[str, os.PathLike]] = None,
    out_valid_file: Optional[Union[str, os.PathLike]] = None,
    out_all_file: Optional[Union[str, os.PathLike]] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    min_thres: Optional[float] = None,
    max_thres: Optional[float] = None,
    out_no_data_val: Optional[float] = None,
    mask_nodata: bool = True,
    all_touched: bool = False,
    include_coords: bool = False,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    drop_nonfinite_targets: bool = True,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
    max_smpls: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_valid: Optional[int] = None,
    stratify_col: Optional[str] = None,
    n_bins: int = 5,
    rnd_seed: Optional[int] = None,
    use_pbar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract regression training samples across configured raster bands and
    vector targets, split them into training, testing, and validation
    DataFrames, and optionally save each split to disk.

    :param img_band_info: Either a :class:`RegressionConfig` instance or a
                          sequence of :class:`ImageBandInfo` objects.
    :param target_cols: Target attribute column name or sequence of names
                        (required when ``img_band_info`` is a sequence).
    :param vec_file: Reference vector file path or
                     :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Optional vector layer name within ``vec_file``.
    :param train_prop: Proportion for training (defaults to
                       ``1.0 - test_prop - valid_prop``).
    :param test_prop: Proportion for testing (Default = ``0.2``).
    :param valid_prop: Proportion for validation (Default = ``0.2``).
    :param out_train_file: Optional output path for training samples.
    :param out_test_file: Optional output path for testing samples.
    :param out_valid_file: Optional output path for validation samples.
    :param out_all_file: Optional output path for all extracted samples prior
                         to splitting.
    :param out_format: Optional output table format overriding extension.
    :param out_compress: Compression setting passed to :func:`write_reg_samples`.
    :param min_thres: Optional inclusive lower valid pixel threshold.
    :param max_thres: Optional inclusive upper valid pixel threshold.
    :param out_no_data_val: Optional fill/nodata sentinel value.
    :param mask_nodata: Drop rows containing nodata/non-finite band values.
    :param all_touched: Passed to
                        :func:`rsgispy.zonalstats.polygon.export_poly_pxls_to_df`.
    :param include_coords: Include ``x`` and ``y`` columns (Default = ``False``).
    :param reproj_vec: Reproject vectors on the fly when CRSs differ.
    :param vec_def_epsg: Optional EPSG code overriding undefined vector CRS.
    :param drop_nonfinite_targets: Drop rows with non-finite target values.
    :param target_min: Optional inclusive lower target bound.
    :param target_max: Optional inclusive upper target bound.
    :param max_smpls: Optional upper sample limit prior to splitting.
    :param n_train: Optional explicit training sample count.
    :param n_test: Optional explicit testing sample count.
    :param n_valid: Optional explicit validation sample count.
    :param stratify_col: Optional target column for quantile-binned splitting.
    :param n_bins: Number of quantile bins when ``stratify_col`` is set.
    :param rnd_seed: Optional random seed for reproducible splits.
    :param use_pbar: Display progress bars during extraction (Default = ``True``).
    :return: Tuple ``(train_df, test_df, valid_df)``.

    .. code:: python

        import rsgispy.regression.regdata as rd

        cfg = rd.read_reg_config("reg_config.json")
        train_df, test_df, valid_df = rd.extract_and_split_reg_samples(
            cfg,
            out_train_file="train.parquet",
            out_test_file="test.parquet",
            out_valid_file="valid.parquet",
            rnd_seed=42,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    cfg = _coerce_config(
        img_band_info,
        target_cols=target_cols,
        vec_file=vec_file if isinstance(vec_file, (str, os.PathLike)) else None,
        vec_lyr=vec_lyr,
    )
    samples_df = extract_reg_training_samples(
        cfg,
        vec_file=vec_file,
        vec_lyr=vec_lyr,
        out_file=out_all_file,
        out_format=out_format,
        out_compress=out_compress,
        min_thres=min_thres,
        max_thres=max_thres,
        out_no_data_val=out_no_data_val,
        mask_nodata=mask_nodata,
        all_touched=all_touched,
        include_coords=include_coords,
        reproj_vec=reproj_vec,
        vec_def_epsg=vec_def_epsg,
        drop_nonfinite_targets=drop_nonfinite_targets,
        target_min=target_min,
        target_max=target_max,
        use_pbar=use_pbar,
    )
    return split_reg_samples(
        samples_df=samples_df,
        train_prop=train_prop,
        test_prop=test_prop,
        valid_prop=valid_prop,
        out_train_file=out_train_file,
        out_test_file=out_test_file,
        out_valid_file=out_valid_file,
        out_format=out_format,
        out_compress=out_compress,
        max_smpls=max_smpls,
        n_train=n_train,
        n_test=n_test,
        n_valid=n_valid,
        stratify_col=stratify_col,
        n_bins=n_bins,
        rnd_seed=rnd_seed,
    )
