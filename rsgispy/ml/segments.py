# pylint: disable=too-many-lines
"""
Segment (object) feature creation and configuration for object-based
classification and regression (:mod:`rsgispy.classification.classsegments`,
:mod:`rsgispy.regression.regsegments`).

* :class:`SegmentFeatureConfig` — the segmentation-specific settings recorded
  in a :class:`~rsgispy.classification.classdata.ClassificationConfig` or
  :class:`~rsgispy.regression.regdata.RegressionConfig` (``seg_info``): label
  raster, segment layer / feature table, ID and valid columns, statistics,
  shape metrics, category lists, the ordered model feature columns, the
  missing-value policy and the training label assignment rules.
* :func:`create_segment_features` — build the segment feature layer from a
  segment label raster: per-band statistics with
  :func:`rsgispy.zonalstats.labels.calc_label_zonal_stats`, polygon shape
  metrics with :func:`rsgispy.vector.stats.calc_shape_metrics_geoseries`, and
  polygons from :func:`rsgispy.segmentation.polygonise_segments`, written as a
  GeoParquet (default) or any vector format, plus an optional attribute-only
  feature table.
* :func:`substitute_scene` — reuse a trained model's configuration on another
  scene (date or tile) by substituting the image and label raster paths.
"""

import copy
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import rasterio

import rsgispy
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.ml.stack import BandSpec, _coerce_band_spec

__all__ = [
    "SegmentFeatureConfig",
    "DEFAULT_SEG_STATS",
    "DEFAULT_SHAPE_METRICS",
    "create_segment_features",
    "substitute_scene",
    "coerce_seg_info",
    "resolve_seg_info",
    "assign_vectors_to_segments",
    "read_segment_features",
    "apply_missing_policy",
    "attach_segment_features",
]

DEFAULT_SEG_STATS: List[str] = ["mean", "stddev", "min", "max"]
"""Default per-band statistics for segment features."""

DEFAULT_SHAPE_METRICS: List[str] = [
    "area",
    "perimeter",
    "compactness",
    "shape_index",
    "solidity",
    "rectangularity",
    "elongation",
]
"""Default polygon shape metrics for segment features."""

_POLY_ASSIGN = ("overlap", "centroid")
_CONFLICT = ("drop", "majority")
_REG_AGG = ("mean", "median")


@dataclass
class SegmentFeatureConfig:
    """
    Segmentation-specific settings for object-based classification and
    regression, stored in the ``seg_info`` block of a classification or
    regression configuration (JSON).

    :param label_img: Segment label raster path.
    :param seg_vec_file: Segment polygon layer (with the features unless they
                         are in ``feat_table_file``).
    :param seg_vec_lyr: Optional segment layer name.
    :param feat_table_file: Optional attribute-only feature table (Parquet)
                            keyed by ``id_col``.
    :param band_specs: Image bands used for the features (as
                       :class:`~rsgispy.ml.stack.BandSpec`).
    :param label_band: Label raster band (Default = ``1``).
    :param label_no_data_val: Label no data value (Default = ``0``).
    :param id_col: Segment ID column (Default = ``'seg_id'``).
    :param valid_col: Valid flag column (Default = ``'valid'``).
    :param pxl_count_col: Pixel count column (Default = ``'pxl_count'``).
    :param stats: Per-band statistics.
    :param percentiles: Per-band percentiles.
    :param quantile_method: Quantile method for the label statistics.
    :param shape_metrics: Polygon shape metrics.
    :param shape_prefix: Shape metric column prefix (Default = ``'shp_'``).
    :param shape_proj_epsg: Optional projected CRS (EPSG or ``'utm'``) for the
                            shape metrics.
    :param col_pattern: Statistic column name pattern.
    :param categories: Category lists for categorical bands.
    :param feature_cols: Ordered list of all feature columns created.
    :param selected_features: Optional ordered subset used by the model (e.g.
                              after feature selection).
    :param feat_no_data_val: Feature no data value (Default = ``-9999.0``).
    :param min_seg_pxls: Minimum segment size (pixels) for a valid segment
                         and training sample (Default = ``1``).
    :param missing: Missing feature policy (see
                    :func:`rsgispy.ml.vector.prepare_feature_matrix`):
                    ``'drop'`` (default; works with every model — segments
                    with missing features are not used for training and are
                    not classified), ``'nan'`` (recommended for XGBoost,
                    LightGBM and CatBoost, which handle NaN natively) or
                    ``'fill'`` (fill with ``fill_value``, for scikit-learn
                    models when every segment must be predicted).
    :param fill_value: Fill value for ``missing='fill'``.
    :param poly_assign: Rule assigning training polygons to segments:
                        ``'overlap'`` or ``'centroid'`` (Default = ``'overlap'``).
    :param min_overlap: Minimum fraction of a segment's area covered by a
                        training polygon for ``'overlap'`` (Default = ``0.5``).
    :param conflict: Handling of segments hit by training data of more than
                     one class: ``'drop'`` or ``'majority'``.
    :param reg_agg: Aggregation of several regression reference values in one
                    segment: ``'mean'`` or ``'median'``.
    """

    # pylint: disable=too-many-instance-attributes
    label_img: Optional[str] = None
    seg_vec_file: Optional[str] = None
    seg_vec_lyr: Optional[str] = None
    feat_table_file: Optional[str] = None
    band_specs: List[BandSpec] = field(default_factory=list)
    label_band: int = 1
    label_no_data_val: Optional[int] = 0
    id_col: str = "seg_id"
    valid_col: str = "valid"
    pxl_count_col: str = "pxl_count"
    stats: List[str] = field(default_factory=lambda: list(DEFAULT_SEG_STATS))
    percentiles: List[float] = field(default_factory=list)
    quantile_method: str = "auto"
    shape_metrics: List[str] = field(default_factory=lambda: list(DEFAULT_SHAPE_METRICS))
    shape_prefix: str = "shp_"
    shape_proj_epsg: Optional[Union[int, str]] = None
    col_pattern: str = "{feat}_{stat}"
    categories: Dict[str, List[float]] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)
    selected_features: Optional[List[str]] = None
    feat_no_data_val: float = -9999.0
    min_seg_pxls: int = 1
    missing: str = "drop"
    fill_value: float = 0.0
    poly_assign: str = "overlap"
    min_overlap: float = 0.5
    conflict: str = "drop"
    reg_agg: str = "mean"

    def __post_init__(self) -> None:
        self.band_specs = [_coerce_band_spec(s) for s in self.band_specs]
        for name, val, allowed in (
            ("poly_assign", self.poly_assign, _POLY_ASSIGN),
            ("conflict", self.conflict, _CONFLICT),
            ("reg_agg", self.reg_agg, _REG_AGG),
            ("missing", self.missing, ("nan", "drop", "fill")),
        ):
            if val not in allowed:
                raise rsgispy.RSGISPyException(
                    f"SegmentFeatureConfig.{name} must be one of {list(allowed)}, "
                    f"got {val!r}."
                )
        if not 0.0 < float(self.min_overlap) <= 1.0:
            raise rsgispy.RSGISPyException(
                f"min_overlap must be in (0, 1], got {self.min_overlap}."
            )
        for path_attr in ("label_img", "seg_vec_file", "feat_table_file"):
            val = getattr(self, path_attr)
            if val is not None:
                setattr(self, path_attr, os.fspath(val))

    @property
    def model_features(self) -> List[str]:
        """Ordered feature columns used by the model (selected or all)."""
        return list(
            self.selected_features
            if self.selected_features is not None
            else self.feature_cols
        )

    @property
    def feature_source(self) -> Any:
        """``(path, layer)`` of the file holding the features."""
        if self.feat_table_file is not None:
            return self.feat_table_file, None
        if self.seg_vec_file is None:
            raise rsgispy.RSGISPyException(
                "SegmentFeatureConfig has no seg_vec_file or feat_table_file."
            )
        return self.seg_vec_file, self.seg_vec_lyr

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        out: Dict[str, Any] = {}
        for fld in fields(self):
            val = getattr(self, fld.name)
            if fld.name == "band_specs":
                val = [spec.to_dict() for spec in val]
            elif isinstance(val, list):
                val = list(val)
            elif isinstance(val, dict):
                val = {str(k): list(v) for k, v in val.items()}
            out[fld.name] = val
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentFeatureConfig":
        """Construct from a dictionary (unknown keys are ignored)."""
        if not isinstance(data, Mapping):
            raise rsgispy.RSGISPyException("seg_info data must be a dictionary.")
        names = {fld.name for fld in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        kwargs["band_specs"] = [
            BandSpec.from_dict(s) if isinstance(s, Mapping) else s
            for s in kwargs.get("band_specs", [])
        ]
        return cls(**kwargs)

    def copy(self) -> "SegmentFeatureConfig":
        """Return a deep copy."""
        return copy.deepcopy(self)


def coerce_seg_info(seg_info: Optional[Any]) -> Optional[SegmentFeatureConfig]:
    """Coerce a ``seg_info`` mapping into a :class:`SegmentFeatureConfig` (or ``None``)."""
    if seg_info is None:
        return None
    if isinstance(seg_info, SegmentFeatureConfig):
        return seg_info
    if isinstance(seg_info, Mapping):
        return SegmentFeatureConfig.from_dict(seg_info)
    raise rsgispy.RSGISPyException(
        "seg_info must be a SegmentFeatureConfig or a dictionary."
    )


def resolve_seg_info(
    cfg: Any, seg_info: Optional[SegmentFeatureConfig] = None
) -> SegmentFeatureConfig:
    """
    Return ``seg_info`` if given, else ``cfg.seg_info`` (a classification or
    regression config or JSON path), raising if neither is set.
    """
    if seg_info is not None:
        return seg_info
    cfg_seg = getattr(cfg, "seg_info", None)
    if cfg_seg is None:
        raise rsgispy.RSGISPyException(
            "The configuration has no segmentation settings (seg_info); create "
            "the segment features with create_segment_features and store the "
            "returned SegmentFeatureConfig in the config's seg_info."
        )
    return cfg_seg


def _resolve_band_specs(
    band_specs: Any, band_opts: Optional[Mapping[str, Mapping[str, Any]]]
) -> List[BandSpec]:
    """Coerce band specifications (or a config's ``img_band_info``) to BandSpecs."""
    items = getattr(band_specs, "img_band_info", band_specs)
    if isinstance(items, (str, os.PathLike)) or not isinstance(items, Sequence):
        items = [items]
    specs = [_coerce_band_spec(item) for item in items]
    if band_opts:
        upd: List[BandSpec] = []
        for spec in specs:
            opts = dict(band_opts.get(spec.name or "", {}))
            if opts:
                base = spec.to_dict()
                base.update(opts)
                spec = BandSpec.from_dict(base)
            upd.append(spec)
        specs = upd
    return specs


def _polygonise_sorted(
    label_img: str,
    label_band: int,
    label_no_data_val: Optional[int],
    id_col: str,
) -> Any:
    """Polygonise a label raster to one (Multi)Polygon per ID, sorted by ID."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.segmentation.utils import polygonise_segments

    gdf = polygonise_segments(
        label_img,
        out_vec_file=None,
        img_band=label_band,
        no_data_val=0 if label_no_data_val is None else int(label_no_data_val),
        segment_id_field=id_col,
    )
    assert gdf is not None
    dup = gdf[id_col].duplicated(keep=False)
    if dup.any():
        multi = gdf[dup].dissolve(by=id_col, as_index=False)
        gdf = pd.concat([gdf[~dup], multi[gdf.columns]], ignore_index=True)
    return gdf.sort_values(id_col, kind="stable").reset_index(drop=True)


def _calc_shape_cols(
    geoms: Any,
    metrics: Sequence[str],
    prefix: str,
    proj_epsg: Optional[Union[int, str]],
    chunk_size: int,
) -> pd.DataFrame:
    """Calculate shape metrics in chunks, in a projected CRS."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.vector.stats import calc_shape_metrics_geoseries

    if proj_epsg is not None:
        tgt = (
            geoms.estimate_utm_crs()
            if str(proj_epsg).strip().lower() == "utm"
            else f"EPSG:{int(proj_epsg)}"
        )
        geoms = geoms.to_crs(tgt)
    elif geoms.crs is not None and geoms.crs.is_geographic:
        raise rsgispy.RSGISPyException(
            "The segments are in a geographic CRS; set shape_proj_epsg (an EPSG "
            "code or 'utm') to calculate the shape metrics in a projected CRS."
        )
    parts = []
    for start in range(0, len(geoms), int(chunk_size)):
        parts.append(
            calc_shape_metrics_geoseries(
                geoms.iloc[start : start + int(chunk_size)],  # noqa: E203
                metrics=metrics,
                col_prefix=prefix,
            )
        )
    if not parts:
        return pd.DataFrame(columns=[f"{prefix}{m}" for m in metrics])
    return pd.concat(parts)


def _check_scene_bands(specs: Sequence[BandSpec]) -> None:
    """Validate that each image has the bands referenced by its spec."""
    for spec in specs:
        ensure_file_exists(spec.file_name)
        if spec.bands is None:
            continue
        with rasterio.open(spec.file_name, "r") as src:
            n_bands = int(src.count)
        too_high = [b for b in spec.bands if b > n_bands]
        if too_high:
            raise rsgispy.RSGISPyException(
                f"Image '{spec.file_name}' ({spec.name}) has {n_bands} bands but "
                f"bands {too_high} are required."
            )


def create_segment_features(
    label_img: Union[str, os.PathLike],
    band_specs: Any,
    out_vec_file: Union[str, os.PathLike],
    out_vec_lyr: Optional[str] = None,
    out_table_file: Optional[Union[str, os.PathLike]] = None,
    vec_inc_features: bool = True,
    stats: Optional[Sequence[str]] = None,
    percentiles: Optional[Sequence[float]] = None,
    shape_metrics: Optional[Sequence[str]] = None,
    shape_proj_epsg: Optional[Union[int, str]] = None,
    band_opts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    template: Optional[SegmentFeatureConfig] = None,
    *,
    label_band: int = 1,
    label_no_data_val: Optional[int] = 0,
    id_col: str = "seg_id",
    feat_no_data_val: float = -9999.0,
    min_seg_pxls: int = 1,
    categories: Optional[Mapping[str, Sequence[float]]] = None,
    quantile_method: str = "auto",
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    overwrite: bool = False,
    chunk_size: int = 100_000,
    block_size: Optional[int] = None,
    use_pbar: bool = True,
) -> SegmentFeatureConfig:
    """
    Create the segment feature layer used for object-based classification and
    regression from a segment label raster (e.g., the output of
    :func:`rsgispy.segmentation.shepherd_segmentation`).

    For every segment (label) it computes:

    * per-band statistics (``stats``/``percentiles``) for every image band in
      ``band_specs`` with
      :func:`rsgispy.zonalstats.labels.calc_label_zonal_stats`
      (columns ``<feature name>_<stat>``, where the feature name is the band
      name if given, else ``<image name>_b<band>``, as in the pixel workflow;
      categorical bands give the mode and category proportions);
    * polygon shape metrics (``shape_metrics``, columns ``shp_<metric>``) with
      :func:`rsgispy.vector.stats.calc_shape_metrics_geoseries`;
    * the pixel count (``pxl_count``) and a valid flag (``valid`` = ``1`` when
      the segment has at least ``min_seg_pxls`` pixels and at least one band
      with valid pixels; other segments are skipped when applying a model).

    Statistics of bands with no valid pixels in a segment are set to
    ``feat_no_data_val``.

    The polygons (one per segment ID, multi-part segments as MultiPolygons,
    sorted by ID) are written with the features to ``out_vec_file``
    (GeoParquet with compression for ``.parquet``, or any vector format), and
    the features are optionally also written to an attribute-only Parquet
    table ``out_table_file`` keyed by ``id_col`` (in the same ID order). With
    ``vec_inc_features=False`` the vector layer only holds the ID, pixel count,
    valid flag and geometry (``out_table_file`` is then required).

    To create the features for another scene with exactly the same feature
    columns as a trained model, pass the model's configuration as
    ``template`` (its statistics, shape metrics and category lists are used,
    and an exception is raised if any model feature cannot be produced).

    :param label_img: Segment label raster.
    :param band_specs: Image bands: sequence of
                       :class:`~rsgispy.ml.stack.BandSpec` /
                       :class:`~rsgispy.imageutils.masking.ImageBandInfo`, or a
                       classification/regression config (its ``img_band_info``).
                       Ignored when ``template`` is given.
    :param out_vec_file: Output segment vector layer (e.g., ``'segs.parquet'``).
    :param out_vec_lyr: Optional output layer name.
    :param out_table_file: Optional attribute-only feature table (Parquet).
    :param vec_inc_features: If ``True`` (default), include the features in
                             the vector layer.
    :param stats: Per-band statistics (Default = mean, stddev, min, max).
    :param percentiles: Optional per-band percentiles.
    :param shape_metrics: Shape metrics (Default = :data:`DEFAULT_SHAPE_METRICS`;
                          ``[]`` for none).
    :param shape_proj_epsg: Optional projected CRS (EPSG or ``'utm'``) for the
                            shape metrics (required for geographic CRSs).
    :param band_opts: Optional per-image (by name) ``BandSpec`` overrides, e.g.
                      ``{'s2': {'no_data_val': 0}, 'lc': {'data_type': 2}}``.
    :param template: Optional :class:`SegmentFeatureConfig` of a trained model.
    :param label_band: Label raster band (Default = ``1``).
    :param label_no_data_val: Label no data value (Default = ``0``).
    :param id_col: Segment ID column (Default = ``'seg_id'``).
    :param feat_no_data_val: Feature no data value (Default = ``-9999.0``).
    :param min_seg_pxls: Minimum segment size for a valid segment.
    :param categories: Optional category lists for categorical bands.
    :param quantile_method: Quantile method (see
                            :func:`~rsgispy.zonalstats.labels.calc_label_zonal_stats`).
    :param out_format: Optional output vector format (inferred from extension).
    :param out_compress: GeoParquet/Parquet compression (Default = ``'snappy'``).
    :param overwrite: If ``True``, overwrite existing outputs.
    :param chunk_size: Polygons per chunk for the shape metrics.
    :param block_size: Optional raster processing block size.
    :param use_pbar: If ``True``, display progress bars.
    :return: :class:`SegmentFeatureConfig` describing the outputs (store it
             as the ``seg_info`` of a classification/regression config).

    .. code:: python

        from rsgispy.classification.classdata import ImageBandInfo
        from rsgispy.ml.segments import create_segment_features

        seg_info = create_segment_features(
            "s2_segs.kea",
            [ImageBandInfo("s2_sref.tif", name="s2", bands=[1, 2, 3, 7, 9],
                           band_names=["Blue", "Green", "Red", "NIR", "SWIR1"])],
            "s2_segs.parquet",
            band_opts={"s2": {"no_data_val": 0}},
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=import-outside-toplevel
    from rsgispy.tools.tables import write_dataframe_table
    from rsgispy.vectorprocess import write_vec_gdf
    from rsgispy.zonalstats.labels import calc_label_zonal_stats

    lab_path = os.path.abspath(os.fspath(label_img))
    ensure_file_exists(lab_path, label="Label image")
    if not vec_inc_features and out_table_file is None:
        raise rsgispy.RSGISPyException(
            "out_table_file is required when vec_inc_features=False."
        )
    if template is not None:
        specs = list(template.band_specs)
        stats = list(template.stats)
        percentiles = list(template.percentiles)
        shape_metrics = list(template.shape_metrics)
        shape_proj_epsg = template.shape_proj_epsg
        categories = dict(template.categories)
        quantile_method = template.quantile_method
        id_col = template.id_col
        label_band = template.label_band
        label_no_data_val = template.label_no_data_val
        feat_no_data_val = template.feat_no_data_val
        min_seg_pxls = template.min_seg_pxls
        _check_scene_bands(specs)
        seg_cfg = template.copy()
    else:
        specs = _resolve_band_specs(band_specs, band_opts)
        seg_cfg = SegmentFeatureConfig()
    stats_l = list(DEFAULT_SEG_STATS if stats is None else stats)
    pct_l = [float(p) for p in (percentiles or [])]
    shp_l = list(DEFAULT_SHAPE_METRICS if shape_metrics is None else shape_metrics)
    shape_prefix = template.shape_prefix if template is not None else "shp_"
    col_pattern = template.col_pattern if template is not None else "{feat}_{stat}"
    valid_col = template.valid_col if template is not None else "valid"
    pxl_col = template.pxl_count_col if template is not None else "pxl_count"

    stats_df = calc_label_zonal_stats(
        lab_path,
        specs,
        stats=stats_l,
        percentiles=pct_l or None,
        label_band=label_band,
        label_no_data_val=label_no_data_val,
        out_no_data_val=feat_no_data_val,
        id_col=id_col,
        col_pattern=col_pattern,
        quantile_method=quantile_method,
        categories=categories,
        pxl_count_col=pxl_col,
        block_size=block_size,
        use_pbar=use_pbar,
    )
    found_cats: Dict[str, List[float]] = dict(stats_df.attrs.get("categories", {}))
    stat_cols = [c for c in stats_df.columns if c not in (id_col, pxl_col)]
    feat_vals = stats_df[stat_cols].to_numpy(dtype=np.float64)
    any_valid = (
        np.any(feat_vals != float(feat_no_data_val), axis=1)
        if stat_cols
        else np.ones(len(stats_df), dtype=bool)
    )
    stats_df[valid_col] = (
        (stats_df[pxl_col].to_numpy() >= int(min_seg_pxls)) & any_valid
    ).astype(np.uint8)

    gdf = _polygonise_sorted(lab_path, label_band, label_no_data_val, id_col)
    gdf = gdf.astype({id_col: stats_df[id_col].dtype})
    if not np.array_equal(gdf[id_col].to_numpy(), stats_df[id_col].to_numpy()):
        gdf = gdf.merge(stats_df[[id_col]], on=id_col, how="inner")
        stats_df = stats_df[stats_df[id_col].isin(gdf[id_col])].reset_index(drop=True)

    shape_cols: List[str] = []
    if shp_l:
        shp_df = _calc_shape_cols(
            gdf.geometry, shp_l, shape_prefix, shape_proj_epsg, chunk_size
        )
        shape_cols = list(shp_df.columns)
        for col in shape_cols:
            stats_df[col] = shp_df[col].to_numpy()

    feature_cols = stat_cols + shape_cols
    if template is not None:
        missing = [c for c in template.model_features if c not in feature_cols]
        if missing:
            raise rsgispy.RSGISPyException(
                f"The new scene does not produce the model feature column(s) "
                f"{missing}; check the image names, bands and band names."
            )

    table_cols = [id_col, pxl_col, valid_col] + feature_cols
    out_gdf = gdf[[id_col, gdf.geometry.name]].copy()
    vec_cols = table_cols[1:] if vec_inc_features else [pxl_col, valid_col]
    for col in vec_cols:
        out_gdf[col] = stats_df[col].to_numpy()
    out_gdf = out_gdf[[id_col] + vec_cols + [gdf.geometry.name]]

    out_vec_path = os.path.abspath(os.fspath(out_vec_file))
    write_vec_gdf(
        out_gdf,
        out_vec_path,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        overwrite=overwrite,
        out_compress=out_compress,
    )
    out_tbl_path: Optional[str] = None
    if out_table_file is not None:
        out_tbl_path = os.path.abspath(os.fspath(out_table_file))
        if os.path.exists(out_tbl_path) and not overwrite:
            raise rsgispy.RSGISPyException(
                f"Output table '{out_tbl_path}' already exists and overwrite=False."
            )
        write_dataframe_table(
            stats_df[table_cols], out_tbl_path, out_compress=out_compress
        )

    seg_cfg.label_img = lab_path
    seg_cfg.seg_vec_file = out_vec_path
    seg_cfg.seg_vec_lyr = out_vec_lyr
    seg_cfg.feat_table_file = out_tbl_path
    seg_cfg.band_specs = list(specs)
    seg_cfg.label_band = int(label_band)
    seg_cfg.label_no_data_val = label_no_data_val
    seg_cfg.id_col = id_col
    seg_cfg.valid_col = valid_col
    seg_cfg.pxl_count_col = pxl_col
    seg_cfg.stats = stats_l
    seg_cfg.percentiles = pct_l
    seg_cfg.quantile_method = quantile_method
    seg_cfg.shape_metrics = shp_l
    seg_cfg.shape_prefix = shape_prefix
    seg_cfg.shape_proj_epsg = shape_proj_epsg
    seg_cfg.col_pattern = col_pattern
    seg_cfg.categories = (
        {k: list(v) for k, v in categories.items()} if categories else found_cats
    )
    seg_cfg.feat_no_data_val = float(feat_no_data_val)
    seg_cfg.min_seg_pxls = int(min_seg_pxls)
    if template is None:
        seg_cfg.feature_cols = feature_cols
    return seg_cfg


def substitute_scene(
    cfg: Any,
    img_files: Mapping[str, Union[str, os.PathLike]],
    label_img: Optional[Union[str, os.PathLike]] = None,
    seg_vec_file: Optional[Union[str, os.PathLike]] = None,
    seg_vec_lyr: Optional[str] = None,
    feat_table_file: Optional[Union[str, os.PathLike]] = None,
) -> Any:
    """
    Return a copy of a classification/regression configuration (or a
    :class:`SegmentFeatureConfig`) for another scene, replacing the image
    paths (matched by image name, i.e. ``ImageBandInfo.name`` /
    ``BandSpec.name``) and, optionally, the segment label raster and the
    segment layer / feature table. Band definitions, statistics, shape
    metrics, category lists and the model feature columns are kept, so the
    features of the new scene can be created with
    :func:`create_segment_features` (``template=new_cfg.seg_info``) and the
    trained model applied to them.

    :param cfg: Classification or regression config (with ``img_band_info``
                and optionally ``seg_info``) or a :class:`SegmentFeatureConfig`.
    :param img_files: Mapping of image name to the new image path; every
                      named image must be given.
    :param label_img: Optional new segment label raster.
    :param seg_vec_file: Optional new segment vector layer.
    :param seg_vec_lyr: Optional new segment layer name.
    :param feat_table_file: Optional new feature table.
    :return: Updated deep copy of ``cfg``.

    .. code:: python

        from rsgispy.ml.segments import substitute_scene, create_segment_features

        new_cfg = substitute_scene(
            cls_cfg, {"s2": "s2_2024_sref.tif"}, label_img="s2_2024_segs.kea"
        )
        new_cfg.seg_info = create_segment_features(
            new_cfg.seg_info.label_img, None, "s2_2024_segs.parquet",
            template=new_cfg.seg_info,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    new_cfg = copy.deepcopy(cfg)
    paths = {str(k): os.fspath(v) for k, v in img_files.items()}
    used: set = set()

    def _sub_name(name: Optional[str], cur: Optional[str]) -> Optional[str]:
        if name is None or name not in paths:
            raise rsgispy.RSGISPyException(
                f"No new image path given for image '{name}' (current: '{cur}'); "
                f"img_files keys: {sorted(paths)}."
            )
        used.add(name)
        return paths[name]

    if hasattr(new_cfg, "img_band_info"):
        for info in new_cfg.img_band_info:
            info.file_name = _sub_name(info.name, info.file_name)
    seg = new_cfg if isinstance(new_cfg, SegmentFeatureConfig) else getattr(
        new_cfg, "seg_info", None
    )
    if seg is not None:
        new_specs = []
        for spec in seg.band_specs:
            base = spec.to_dict()
            base["file_name"] = _sub_name(spec.name, spec.file_name)
            new_specs.append(BandSpec.from_dict(base))
        seg.band_specs = new_specs
        if label_img is not None:
            seg.label_img = os.fspath(label_img)
        if seg_vec_file is not None:
            seg.seg_vec_file = os.fspath(seg_vec_file)
            seg.seg_vec_lyr = seg_vec_lyr
        if feat_table_file is not None:
            seg.feat_table_file = os.fspath(feat_table_file)
        _check_scene_bands(seg.band_specs)
    unknown = sorted(set(paths) - used)
    if unknown:
        raise rsgispy.RSGISPyException(
            f"img_files contains unknown image name(s) {unknown}."
        )
    return new_cfg


def _load_training_vec(
    vec_file: Any,
    vec_lyr: Optional[str],
    tgt_crs: Any,
    reproj_vec: bool,
    vec_def_epsg: Optional[int],
) -> Any:
    """Read a training vector layer and align its CRS with the label raster."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.vectorprocess import read_vec_gdf

    gdf = read_vec_gdf(vec_file, vec_lyr)
    if gdf.crs is None and vec_def_epsg is not None:
        gdf = gdf.set_crs(epsg=int(vec_def_epsg))
    if tgt_crs is not None and gdf.crs is not None and gdf.crs != tgt_crs:
        if not reproj_vec:
            raise rsgispy.RSGISPyException(
                f"The training vector '{vec_file}' CRS ({gdf.crs.to_string()}) "
                "differs from the segment label raster CRS; set reproj_vec=True "
                "to reproject it on the fly."
            )
        gdf = gdf.to_crs(tgt_crs)
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)].reset_index(drop=True)
    return gdf


def assign_vectors_to_segments(
    seg_info: SegmentFeatureConfig,
    vec_file: Any,
    vec_lyr: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    poly_assign: Optional[str] = None,
    min_overlap: Optional[float] = None,
    reproj_vec: bool = False,
    vec_def_epsg: Optional[int] = None,
    group_prefix: str = "",
) -> pd.DataFrame:
    """
    Assign the features of a point and/or polygon vector layer to segments,
    returning one row per ``(segment, vector feature)`` hit.

    * Points are assigned to the segment under them (by sampling the segment
      label raster).
    * Polygons are assigned to every segment whose fraction of pixels (pixel
      centres) inside the polygon is at least ``min_overlap``
      (``poly_assign='overlap'``), or whose centroid is inside the polygon
      (``poly_assign='centroid'``).

    The returned :class:`pandas.DataFrame` has the segment ID column, a
    ``train_grp`` column identifying the source feature (points: one group per
    segment; polygons: one group per polygon, so segments from the same
    polygon can be kept together when splitting), ``src_type``
    (``'point'``/``'polygon'``) and the requested ``value_cols``.

    :param seg_info: Segment configuration.
    :param vec_file: Vector file or :class:`geopandas.GeoDataFrame`.
    :param vec_lyr: Optional layer name.
    :param value_cols: Attribute columns to carry over.
    :param poly_assign: ``'overlap'`` or ``'centroid'`` (Default = from ``seg_info``).
    :param min_overlap: Minimum covered fraction for ``'overlap'``
                        (Default = from ``seg_info``).
    :param reproj_vec: Reproject the vector to the label raster CRS if needed.
    :param vec_def_epsg: Optional EPSG code for a vector with no CRS.
    :param group_prefix: Prefix for the ``train_grp`` values.
    :return: :class:`pandas.DataFrame` of hits.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=import-outside-toplevel
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as win_transform

    id_col = seg_info.id_col
    rule = poly_assign if poly_assign is not None else seg_info.poly_assign
    frac_min = float(min_overlap if min_overlap is not None else seg_info.min_overlap)
    if rule not in _POLY_ASSIGN:
        raise rsgispy.RSGISPyException(f"poly_assign must be one of {list(_POLY_ASSIGN)}.")
    if seg_info.label_img is None:
        raise rsgispy.RSGISPyException("seg_info.label_img is not set.")
    v_cols = list(value_cols or [])
    out_cols = [id_col, "train_grp", "src_type"] + v_cols
    with rasterio.open(seg_info.label_img, "r") as lab_src:
        gdf = _load_training_vec(vec_file, vec_lyr, lab_src.crs, reproj_vec, vec_def_epsg)
        missing = [c for c in v_cols if c not in gdf.columns]
        if missing:
            raise rsgispy.RSGISPyException(
                f"Column(s) {missing} not found in training vector '{vec_file}'."
            )
        lab_nd = seg_info.label_no_data_val
        g_types = gdf.geometry.geom_type
        rows: List[pd.DataFrame] = []

        pt_mask = g_types.isin(["Point", "MultiPoint"]).to_numpy()
        if np.any(pt_mask):
            pts = gdf[pt_mask].explode(index_parts=False)
            coords = [(geom.x, geom.y) for geom in pts.geometry]
            in_bounds = np.array(
                [
                    lab_src.bounds.left <= x_c < lab_src.bounds.right
                    and lab_src.bounds.bottom < y_c <= lab_src.bounds.top
                    for x_c, y_c in coords
                ],
                dtype=bool,
            )
            seg_ids = np.zeros(len(coords), dtype=np.int64)
            if np.any(in_bounds):
                sampled = np.array(
                    [
                        v[0]
                        for v in lab_src.sample(
                            [c for c, ok in zip(coords, in_bounds) if ok],
                            indexes=seg_info.label_band,
                        )
                    ]
                )
                seg_ids[in_bounds] = sampled
            ok = in_bounds.copy()
            if lab_nd is not None:
                ok &= seg_ids != int(lab_nd)
            pt_df = pd.DataFrame({id_col: seg_ids[ok]})
            pt_df["train_grp"] = [f"{group_prefix}seg{sid}" for sid in seg_ids[ok]]
            pt_df["src_type"] = "point"
            for col in v_cols:
                pt_df[col] = pts[col].to_numpy()[ok]
            rows.append(pt_df)

        poly_mask = g_types.isin(["Polygon", "MultiPolygon"]).to_numpy()
        if np.any(poly_mask):
            polys = gdf[poly_mask].reset_index(drop=True)
            if rule == "overlap":
                pxl_counts = read_segment_features(
                    seg_info, None, [seg_info.pxl_count_col]
                ).set_index(id_col)[seg_info.pxl_count_col]
            else:
                # pylint: disable=import-outside-toplevel
                from rsgispy.zonalstats.labels import calc_label_zonal_stats

                cent = calc_label_zonal_stats(
                    seg_info.label_img,
                    None,
                    label_band=seg_info.label_band,
                    label_no_data_val=lab_nd,
                    id_col=id_col,
                    inc_centroid=True,
                    use_pbar=False,
                )
                import geopandas as gpd  # pylint: disable=import-outside-toplevel

                cent_gdf = gpd.GeoDataFrame(
                    cent[[id_col]],
                    geometry=gpd.points_from_xy(cent["centroid_x"], cent["centroid_y"]),
                    crs=lab_src.crs,
                )
                joined = gpd.sjoin(
                    cent_gdf, polys[[polys.geometry.name]], predicate="within"
                )
            for p_idx, geom in enumerate(polys.geometry):
                if rule == "overlap":
                    raw = from_bounds(
                        *geom.bounds, transform=lab_src.transform
                    )
                    c_0 = max(0, int(np.floor(raw.col_off)))
                    r_0 = max(0, int(np.floor(raw.row_off)))
                    c_1 = min(lab_src.width, int(np.ceil(raw.col_off + raw.width)))
                    r_1 = min(lab_src.height, int(np.ceil(raw.row_off + raw.height)))
                    if c_1 <= c_0 or r_1 <= r_0:
                        continue
                    win = Window(c_0, r_0, c_1 - c_0, r_1 - r_0)
                    lab = lab_src.read(seg_info.label_band, window=win)
                    inside = rasterize(
                        [(geom, 1)],
                        out_shape=lab.shape,
                        transform=win_transform(win, lab_src.transform),
                        fill=0,
                        dtype="uint8",
                    ).astype(bool)
                    vals = lab[inside]
                    if lab_nd is not None:
                        vals = vals[vals != int(lab_nd)]
                    if vals.size == 0:
                        continue
                    uniq, cnts = np.unique(vals, return_counts=True)
                    tot = pxl_counts.reindex(uniq).to_numpy(dtype=np.float64)
                    keep = (cnts / np.where(tot > 0, tot, np.inf)) >= frac_min - 1e-12
                    hit_ids = uniq[keep]
                else:
                    hit_ids = joined.loc[joined["index_right"] == p_idx, id_col].to_numpy()
                if hit_ids.size == 0:
                    continue
                p_df = pd.DataFrame({id_col: hit_ids})
                p_df["train_grp"] = f"{group_prefix}poly{p_idx}"
                p_df["src_type"] = "polygon"
                for col in v_cols:
                    p_df[col] = polys[col].iloc[p_idx]
                rows.append(p_df)

    if not rows:
        return pd.DataFrame(columns=out_cols)
    out = pd.concat(rows, ignore_index=True)
    return out[out_cols]


def read_segment_features(
    seg_info: SegmentFeatureConfig,
    seg_ids: Optional[Sequence[Any]],
    columns: Optional[Sequence[str]] = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Read the feature rows (``id_col`` + ``columns``, default the model
    features, pixel count and valid flag) for the given segment IDs (all
    segments when ``None``) from the segment feature layer/table in chunks,
    without reading the geometry.

    :param seg_info: Segment configuration.
    :param seg_ids: Segment IDs to read, or ``None`` for all.
    :param columns: Columns to read (Default = model features + pixel count
                    + valid flag).
    :param chunk_size: Rows per chunk.
    :return: :class:`pandas.DataFrame` (in file order).
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy.ml.vector import iter_vec_chunks

    id_col = seg_info.id_col
    cols = (
        [seg_info.pxl_count_col, seg_info.valid_col] + seg_info.model_features
        if columns is None
        else list(columns)
    )
    read_cols = list(dict.fromkeys([id_col] + cols))
    path, lyr = seg_info.feature_source
    id_set = None if seg_ids is None else np.unique(np.asarray(seg_ids))
    parts = []
    for chunk in iter_vec_chunks(path, read_cols, vec_lyr=lyr, chunk_size=chunk_size):
        if id_set is not None:
            chunk = chunk[np.isin(chunk[id_col].to_numpy(), id_set)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=read_cols)
    return pd.concat(parts, ignore_index=True)


def apply_missing_policy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    missing: str,
    fill_value: float,
    feat_no_data_val: Optional[float],
) -> pd.DataFrame:
    """
    Apply the missing feature policy to a sample table: the feature no data
    value becomes NaN, then rows with missing features are dropped
    (``'drop'``), filled with ``fill_value`` (``'fill'``) or kept as NaN
    (``'nan'``).
    """
    out = df.copy()
    feats = list(feature_cols)
    vals = out[feats].to_numpy(dtype=np.float64, copy=True)
    if feat_no_data_val is not None:
        vals[vals == float(feat_no_data_val)] = np.nan
    if missing == "fill":
        vals[~np.isfinite(vals)] = float(fill_value)
    out[feats] = vals
    if missing == "drop":
        out = out[np.all(np.isfinite(vals), axis=1)].reset_index(drop=True)
    return out


def attach_segment_features(
    seg_info: SegmentFeatureConfig,
    samples: pd.DataFrame,
    min_seg_pxls: int,
    report: Dict[str, Any],
) -> pd.DataFrame:
    """
    Join the segment features onto a per-segment sample table and remove
    invalid segments, segments smaller than ``min_seg_pxls`` pixels and (per
    the missing policy) segments with missing features, recording the number
    removed in ``report`` (``n_invalid_dropped``, ``n_small_dropped``,
    ``n_missing_dropped``).
    """
    id_col = seg_info.id_col
    feats = read_segment_features(seg_info, samples[id_col].to_numpy())
    out = samples.merge(feats, on=id_col, how="inner")
    n_before = len(out)
    out = out[out[seg_info.valid_col] != 0]
    report["n_invalid_dropped"] = int(n_before - len(out))
    n_before = len(out)
    out = out[out[seg_info.pxl_count_col] >= int(min_seg_pxls)]
    report["n_small_dropped"] = int(n_before - len(out))
    n_before = len(out)
    out = apply_missing_policy(
        out,
        seg_info.model_features,
        seg_info.missing,
        seg_info.fill_value,
        seg_info.feat_no_data_val,
    )
    report["n_missing_dropped"] = int(n_before - len(out))
    return out
