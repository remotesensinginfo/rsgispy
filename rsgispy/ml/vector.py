"""
Chunked vector / table I/O and prediction for :mod:`rsgispy.ml`.

These functions process large vector layers (e.g., millions of image segments)
in chunks of features so the whole layer is never loaded into memory:

* :func:`read_vec_layer_info` — layer columns, geometry column, CRS and
  feature count without reading the features.
* :func:`iter_vec_batches` / :func:`iter_vec_chunks` — stream a subset of
  columns as Arrow tables or :class:`pandas.DataFrame` chunks, without
  reading (or decoding) the geometry unless requested.
* :class:`ChunkedVecWriter` — write Arrow chunks to GeoParquet (with
  compression), an attribute-only Parquet table, or any OGR vector format,
  passing the geometry through as undecoded WKB, writing to a temporary file
  that is only moved into place on success, and adding provenance metadata.
* :func:`map_vector_chunks` — the generic streaming engine: read the columns a
  function needs, call it on each chunk and write the input layer (or a
  separate geometry layer joined by ID) plus the function's output columns.
* :func:`prepare_feature_matrix` — apply valid flags, feature no-data values,
  a missing-value policy and an optional
  :class:`~rsgispy.ml.scaling.FeatureScaler` consistently at training and
  prediction time.
* :func:`predict_vector` — apply any fitted estimator to a vector layer or
  attribute table in chunks.

GeoParquet / Parquet is read and written with ``pyarrow`` and every other
format with ``pyogrio``'s Arrow interface (requires ``pyarrow``, installed with
the ``classification`` / ``regression`` extras).
"""

# pylint: disable=too-many-lines

import datetime
import hashlib
import json
import math
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import uuid

import numpy as np
import pandas as pd

import rsgispy
from rsgispy.vectorprocess import check_format_name, is_geoparquet_file, is_geoparquet_format

__all__ = [
    "VecLayerInfo",
    "read_vec_layer_info",
    "iter_vec_batches",
    "iter_vec_chunks",
    "ChunkedVecWriter",
    "build_provenance",
    "map_vector_chunks",
    "prepare_feature_matrix",
    "model_input",
    "predict_vector",
    "MISSING_POLICIES",
]

MISSING_POLICIES: Tuple[str, ...] = ("nan", "drop", "fill")
"""Supported missing feature value policies (see :func:`prepare_feature_matrix`)."""

_PQ_EXTS = (".parquet", ".geoparquet", ".parq", ".pq")
_INDEX_COLS = ("__index_level_0__",)
_GEOM_NAME = "geometry"


def _require_pyarrow(feature: str) -> Tuple[Any, Any]:
    """Import ``pyarrow`` and ``pyarrow.parquet`` with an optional-dep guard."""
    try:
        import pyarrow  # pylint: disable=import-outside-toplevel,import-error
        import pyarrow.parquet  # pylint: disable=import-outside-toplevel,import-error
    except ImportError as exc:
        rsgispy.check_optional_dep(
            False, feature, "pyarrow", "classification", error_detail=str(exc)
        )
        raise
    return pyarrow, pyarrow.parquet


def _is_parquet_path(path: str, fmt: Optional[str] = None) -> bool:
    """Return ``True`` if ``path``/``fmt`` denotes a (Geo)Parquet file."""
    if fmt is not None:
        f_low = str(fmt).strip().lower()
        if f_low in ("parquet", "geoparquet"):
            return True
        try:
            return is_geoparquet_format(check_format_name(fmt))
        except rsgispy.RSGISPyException:
            return False
    low = path.lower()
    return is_geoparquet_file(path) or any(
        low.endswith(ext) or f"{ext}." in low for ext in _PQ_EXTS
    )


class VecLayerInfo:
    """
    Metadata describing a vector layer or attribute table (read without
    reading the features).

    :param path: File path.
    :param layer: Layer name (``None`` for single-layer formats / Parquet).
    :param is_parquet: ``True`` for (Geo)Parquet files.
    :param columns: Attribute column names (excluding geometry).
    :param geom_col: Geometry column name, or ``None`` for attribute-only tables.
    :param crs: CRS as a user-input string (WKT/PROJJSON/EPSG) or ``None``.
    :param geometry_type: OGR geometry type name (e.g. ``'Polygon'``) or ``None``.
    :param n_features: Number of features / rows.
    :param geo_metadata: Raw GeoParquet ``geo`` metadata (Parquet only).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(
        self,
        path: str,
        layer: Optional[str],
        is_parquet: bool,
        columns: List[str],
        geom_col: Optional[str],
        crs: Optional[str],
        geometry_type: Optional[str],
        n_features: int,
        geo_metadata: Optional[bytes] = None,
    ) -> None:
        self.path = path
        self.layer = layer
        self.is_parquet = is_parquet
        self.columns = columns
        self.geom_col = geom_col
        self.crs = crs
        self.geometry_type = geometry_type
        self.n_features = n_features
        self.geo_metadata = geo_metadata

    def __repr__(self) -> str:
        return (
            f"VecLayerInfo(path={self.path!r}, layer={self.layer!r}, "
            f"n_features={self.n_features}, geom_col={self.geom_col!r}, "
            f"columns={self.columns!r})"
        )


def read_vec_layer_info(
    vec_file: Union[str, os.PathLike], vec_lyr: Optional[str] = None
) -> VecLayerInfo:
    """
    Read the metadata of a vector layer or attribute table without reading
    its features.

    :param vec_file: Vector file (GeoParquet, GPKG, Shapefile, etc.) or
                     attribute-only Parquet/CSV table path.
    :param vec_lyr: Optional layer name.
    :return: :class:`VecLayerInfo`.

    .. code:: python

        from rsgispy.ml.vector import read_vec_layer_info

        info = read_vec_layer_info("segments.parquet")
        print(info.n_features, info.columns)
    """
    # pylint: disable=too-many-locals
    path = os.fspath(vec_file)
    if not os.path.exists(path):
        raise rsgispy.RSGISPyException(f"Vector file does not exist: '{path}'.")
    if _is_parquet_path(path):
        _, pq = _require_pyarrow("read_vec_layer_info")
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        meta = schema.metadata or {}
        geo_raw = meta.get(b"geo")
        geom_col: Optional[str] = None
        geom_cols: List[str] = []
        crs: Optional[str] = None
        geom_type: Optional[str] = None
        if geo_raw:
            geo = json.loads(geo_raw)
            geom_col = geo.get("primary_column", _GEOM_NAME)
            geom_cols = list(geo.get("columns", {}).keys()) or [geom_col]
            col_meta = geo.get("columns", {}).get(geom_col, {})
            if "crs" not in col_meta:
                crs = "OGC:CRS84"
            elif col_meta["crs"] is not None:
                crs = json.dumps(col_meta["crs"])
            g_types = col_meta.get("geometry_types") or []
            if len(g_types) == 1:
                geom_type = str(g_types[0])
        columns = [
            n for n in schema.names if n not in geom_cols and n not in _INDEX_COLS
        ]
        return VecLayerInfo(
            path, None, True, columns, geom_col, crs, geom_type,
            int(pf.metadata.num_rows), geo_raw,
        )
    import pyogrio  # pylint: disable=import-outside-toplevel

    try:
        info = pyogrio.read_info(path, layer=vec_lyr)
    except Exception as err:
        raise rsgispy.RSGISPyException(
            f"Could not read vector layer info from '{path}': {err}"
        ) from err
    n_feats = int(info.get("features", -1))
    if n_feats < 0:
        n_feats = int(
            pyogrio.read_info(path, layer=vec_lyr, force_feature_count=True)[
                "features"
            ]
        )
    g_type = info.get("geometry_type")
    has_geom = g_type is not None
    return VecLayerInfo(
        path,
        vec_lyr,
        False,
        [str(f) for f in info["fields"]],
        _GEOM_NAME if has_geom else None,
        info.get("crs"),
        g_type,
        n_feats,
    )


def _normalise_batch_geom(batch: Any, src_geom: Optional[str], pa_mod: Any) -> Any:
    """Rename the GDAL geometry column to ``geometry`` and drop extension metadata."""
    if src_geom is None or src_geom not in batch.schema.names:
        return batch
    idx = batch.schema.get_field_index(src_geom)
    col = batch.column(idx)
    if pa_mod.types.is_binary(col.type) or pa_mod.types.is_large_binary(col.type):
        plain = col
    else:  # extension type (e.g., geoarrow.wkb) -> storage
        plain = col.storage if hasattr(col, "storage") else col.cast(pa_mod.binary())
    return batch.set_column(idx, pa_mod.field(_GEOM_NAME, plain.type), plain)


def iter_vec_batches(
    vec_file: Union[str, os.PathLike, VecLayerInfo],
    columns: Optional[Sequence[str]] = None,
    vec_lyr: Optional[str] = None,
    read_geometry: bool = False,
    chunk_size: int = 100_000,
) -> Iterator[Any]:
    """
    Iterate over a vector layer or attribute table in chunks of
    ``chunk_size`` features as :class:`pyarrow.Table` objects, reading only
    ``columns`` (all attribute columns when ``None``) and, if
    ``read_geometry=True``, the geometry as undecoded WKB in a ``geometry``
    column.

    :param vec_file: File path or :class:`VecLayerInfo`.
    :param columns: Attribute columns to read (Default = all).
    :param vec_lyr: Optional layer name.
    :param read_geometry: If ``True``, include the WKB ``geometry`` column.
    :param chunk_size: Number of features per chunk (Default = ``100_000``).
    :return: Iterator of :class:`pyarrow.Table`.
    """
    # pylint: disable=too-many-locals
    pa_mod, pq = _require_pyarrow("iter_vec_batches")
    info = (
        vec_file
        if isinstance(vec_file, VecLayerInfo)
        else read_vec_layer_info(vec_file, vec_lyr)
    )
    if int(chunk_size) < 1:
        raise rsgispy.RSGISPyException(f"chunk_size must be >= 1, got {chunk_size}.")
    cols = list(info.columns) if columns is None else list(columns)
    missing = [c for c in cols if c not in info.columns]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Column(s) {missing} not found in '{info.path}'."
        )
    if read_geometry and info.geom_col is None:
        raise rsgispy.RSGISPyException(
            f"'{info.path}' has no geometry column to read."
        )
    if info.is_parquet:
        pf = pq.ParquetFile(info.path)
        read_cols = cols + ([info.geom_col] if read_geometry else [])
        for batch in pf.iter_batches(batch_size=int(chunk_size), columns=read_cols):
            tbl = pa_mod.Table.from_batches([batch])
            if read_geometry and info.geom_col != _GEOM_NAME:
                tbl = _normalise_batch_geom(tbl, info.geom_col, pa_mod)
            yield tbl
        return

    import pyogrio.raw  # pylint: disable=import-outside-toplevel

    with pyogrio.raw.open_arrow(
        info.path,
        layer=info.layer,
        columns=cols,
        read_geometry=read_geometry,
        batch_size=int(chunk_size),
        use_pyarrow=True,
    ) as (meta, reader):
        src_geom = meta.get("geometry_name") or "wkb_geometry"
        for batch in reader:
            tbl = pa_mod.Table.from_batches([batch])
            drop = [n for n in tbl.schema.names if n not in cols and n != src_geom]
            if drop:
                tbl = tbl.drop_columns(drop)
            if read_geometry:
                tbl = _normalise_batch_geom(tbl, src_geom, pa_mod)
            yield tbl.select(cols + ([_GEOM_NAME] if read_geometry else []))


def iter_vec_chunks(
    vec_file: Union[str, os.PathLike, VecLayerInfo],
    columns: Optional[Sequence[str]] = None,
    vec_lyr: Optional[str] = None,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over the attribute columns of a vector layer or table in chunks
    of ``chunk_size`` rows as :class:`pandas.DataFrame` objects, never reading
    the geometry.

    :param vec_file: File path or :class:`VecLayerInfo`.
    :param columns: Columns to read (Default = all attribute columns).
    :param vec_lyr: Optional layer name.
    :param chunk_size: Rows per chunk (Default = ``100_000``).
    :return: Iterator of :class:`pandas.DataFrame`.

    .. code:: python

        from rsgispy.ml.vector import iter_vec_chunks

        for df in iter_vec_chunks("segments.parquet", ["seg_id", "s2_Red_mean"]):
            print(len(df))
    """
    for tbl in iter_vec_batches(
        vec_file, columns=columns, vec_lyr=vec_lyr, chunk_size=chunk_size
    ):
        yield tbl.to_pandas()


def build_provenance(
    model: Any = None,
    model_file: Optional[Union[str, os.PathLike]] = None,
    config_file: Optional[Union[str, os.PathLike]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a provenance record (rsgispy version, creation time, model type,
    model file and SHA-256 hash, config file and ordered feature list) to be
    stored in output metadata.

    :param model: Optional fitted model object (its class name is recorded).
    :param model_file: Optional path of the saved model file (hashed).
    :param config_file: Optional path of the configuration JSON file.
    :param feature_cols: Optional ordered list of feature columns.
    :param extra: Optional additional JSON-serialisable entries.
    :return: Provenance dictionary.
    """
    prov: Dict[str, Any] = {
        "rsgispy_version": getattr(rsgispy, "__version__", "unknown"),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if model is not None:
        prov["model_type"] = f"{type(model).__module__}.{type(model).__name__}"
    if model_file is not None:
        m_path = os.fspath(model_file)
        prov["model_file"] = os.path.abspath(m_path)
        if os.path.isfile(m_path):
            hasher = hashlib.sha256()
            with open(m_path, "rb") as f_obj:
                for blk in iter(lambda: f_obj.read(1 << 20), b""):
                    hasher.update(blk)
            prov["model_sha256"] = hasher.hexdigest()
    if config_file is not None:
        prov["config_file"] = os.path.abspath(os.fspath(config_file))
    if feature_cols is not None:
        prov["feature_cols"] = [str(c) for c in feature_cols]
    if extra:
        prov.update({str(k): v for k, v in extra.items()})
    return prov


class ChunkedVecWriter:
    """
    Write a vector layer or attribute table chunk by chunk from Arrow tables.

    Output is written to a temporary file in the output directory and moved
    into place when the writer is closed successfully (so a failure never
    leaves a partial output). GeoParquet/Parquet outputs are written with
    ``pyarrow`` (one row group per chunk, with compression and GeoParquet
    ``geo`` metadata when a geometry column is present); other formats with
    ``pyogrio``. Geometry must be supplied as WKB in ``geom_col``.

    :param out_file: Output file path.
    :param out_vec_lyr: Optional output layer name (OGR multi-layer formats).
    :param out_format: Optional format: ``'GeoParquet'``/``'Parquet'`` or an OGR
                       driver name (Default = inferred from the extension).
    :param out_compress: Parquet compression codec (Default = ``'snappy'``).
    :param geom_col: Geometry (WKB) column name in the chunks, or ``None``
                     for an attribute-only table (Default = ``'geometry'``).
    :param crs: Output CRS (WKT, PROJJSON or ``'EPSG:XXXX'``).
    :param geometry_type: OGR geometry type name (Default = ``'Unknown'``).
    :param geo_metadata: Optional GeoParquet ``geo`` metadata to copy.
    :param metadata: Optional provenance dictionary stored as file metadata
                     (Parquet key ``rsgispy_provenance``; GPKG layer metadata).
    :param overwrite: If ``True``, replace an existing output file.

    .. code:: python

        from rsgispy.ml.vector import ChunkedVecWriter, iter_vec_batches

        with ChunkedVecWriter("copy.parquet", crs="EPSG:32630") as writer:
            for tbl in iter_vec_batches("segments.parquet", read_geometry=True):
                writer.write(tbl)
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        out_file: Union[str, os.PathLike],
        out_vec_lyr: Optional[str] = None,
        out_format: Optional[str] = None,
        out_compress: Optional[Union[str, bool]] = "snappy",
        geom_col: Optional[str] = _GEOM_NAME,
        crs: Optional[str] = None,
        geometry_type: Optional[str] = "Unknown",
        geo_metadata: Optional[bytes] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        self.pa, self.pq = _require_pyarrow("ChunkedVecWriter")
        self.out_file = os.path.abspath(os.fspath(out_file))
        self.out_vec_lyr = out_vec_lyr
        self.is_parquet = _is_parquet_path(self.out_file, out_format)
        self.driver: Optional[str] = None
        if not self.is_parquet:
            if out_format is not None:
                self.driver = check_format_name(out_format)
            else:
                import pyogrio.raw  # pylint: disable=import-outside-toplevel

                self.driver = pyogrio.raw.detect_write_driver(self.out_file)
        if out_compress is True:
            self.compression: Optional[str] = "snappy"
        elif out_compress in (None, False) or str(out_compress).lower() == "none":
            self.compression = None
        else:
            self.compression = str(out_compress)
        self.geom_col = geom_col
        self.crs = crs
        self.geometry_type = geometry_type or "Unknown"
        self.geo_metadata = geo_metadata
        self.metadata = dict(metadata) if metadata else None
        if os.path.exists(self.out_file) and not overwrite:
            raise rsgispy.RSGISPyException(
                f"Output file '{self.out_file}' already exists and overwrite=False."
            )
        out_dir = os.path.dirname(self.out_file)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(self.out_file)
        stem, ext = os.path.splitext(base)
        self.tmp_file = os.path.join(out_dir, f".{stem}.{uuid.uuid4().hex[:8]}.tmp{ext}")
        self._pq_writer: Any = None
        self._schema: Any = None
        self._n_written = 0
        self._closed = False

    def __enter__(self) -> "ChunkedVecWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(success=exc_type is None)

    def _geo_meta_bytes(self) -> bytes:
        """Build (or copy) the GeoParquet ``geo`` metadata."""
        if self.geo_metadata:
            geo = json.loads(self.geo_metadata)
            prim = geo.get("primary_column", _GEOM_NAME)
            col_meta = geo.get("columns", {}).get(prim, {})
            col_meta.pop("bbox", None)
            geo["primary_column"] = self.geom_col
            geo["columns"] = {self.geom_col: col_meta}
            return json.dumps(geo).encode("utf-8")
        col_meta: Dict[str, Any] = {"encoding": "WKB", "geometry_types": []}
        if self.crs is not None:
            import pyproj  # pylint: disable=import-outside-toplevel

            col_meta["crs"] = pyproj.CRS.from_user_input(self.crs).to_json_dict()
        else:
            col_meta["crs"] = None
        geo = {
            "version": "1.0.0",
            "primary_column": self.geom_col,
            "columns": {self.geom_col: col_meta},
        }
        return json.dumps(geo).encode("utf-8")

    def write(self, table: Any) -> None:
        """Write one chunk (a :class:`pyarrow.Table` or ``RecordBatch``)."""
        # pylint: disable=too-many-locals
        if self._closed:
            raise rsgispy.RSGISPyException("ChunkedVecWriter is already closed.")
        pa_mod = self.pa
        if isinstance(table, pa_mod.RecordBatch):
            table = pa_mod.Table.from_batches([table])
        if self.geom_col is not None and self.geom_col not in table.schema.names:
            raise rsgispy.RSGISPyException(
                f"Chunk does not contain the geometry column '{self.geom_col}'."
            )
        if self.is_parquet:
            if self._pq_writer is None:
                schema = table.schema
                meta = dict(schema.metadata or {})
                meta.pop(b"pandas", None)
                if self.geom_col is not None:
                    meta[b"geo"] = self._geo_meta_bytes()
                if self.metadata:
                    meta[b"rsgispy_provenance"] = json.dumps(self.metadata).encode()
                schema = schema.with_metadata(meta)
                self._schema = schema
                self._pq_writer = self.pq.ParquetWriter(
                    self.tmp_file, schema, compression=self.compression or "none"
                )
            table = table.select(self._schema.names).cast(self._schema)
            self._pq_writer.write_table(table)
        else:
            import pyogrio.raw  # pylint: disable=import-outside-toplevel

            if self._schema is None:
                self._schema = table.schema
            table = table.select(self._schema.names).cast(self._schema)
            kwargs: Dict[str, Any] = {
                "driver": self.driver,
                "geometry_name": self.geom_col,
                "geometry_type": self.geometry_type if self.geom_col else None,
                "crs": self.crs if self.geom_col else None,
                "append": self._n_written > 0,
            }
            kwargs["layer"] = (
                self.out_vec_lyr
                if self.out_vec_lyr is not None
                else os.path.splitext(os.path.basename(self.out_file))[0]
            )
            if self.metadata and self._n_written == 0 and self.driver == "GPKG":
                kwargs["layer_metadata"] = {
                    "rsgispy_provenance": json.dumps(self.metadata)
                }
            if table.num_rows == 0 and self._n_written > 0:
                return
            pyogrio.raw.write_arrow(table, self.tmp_file, **kwargs)
        self._n_written += int(table.num_rows)

    def close(self, success: bool = True) -> None:
        """Finish writing; move the output into place, or discard it on failure."""
        if self._closed:
            return
        self._closed = True
        if self._pq_writer is not None:
            self._pq_writer.close()
        if success and self._n_written == 0 and self._schema is None:
            success = False
            raise rsgispy.RSGISPyException("No chunks were written.")
        if success:
            if os.path.exists(self.out_file):
                from rsgispy.vectorprocess import (  # pylint: disable=import-outside-toplevel
                    delete_vec_file,
                )

                delete_vec_file(self.out_file, feedback=False)
            os.replace(self.tmp_file, self.out_file)
        elif os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)


def _resolve_copy_cols(
    copy_cols: Optional[Sequence[str]],
    available: Sequence[str],
    id_col: Optional[str],
    exclude: Sequence[str],
) -> List[str]:
    """Resolve the columns copied from the input to the output."""
    if copy_cols is None:
        cols = [c for c in available if c not in exclude]
    else:
        missing = [c for c in copy_cols if c not in available]
        if missing:
            raise rsgispy.RSGISPyException(
                f"copy_cols {missing} are not in the input layer."
            )
        cols = [c for c in copy_cols if c not in exclude]
    if id_col is not None and id_col in available and id_col not in cols:
        cols = [id_col] + cols
    return cols


def _append_outputs(tbl: Any, out_df: pd.DataFrame, pa_mod: Any) -> Any:
    """Append the output DataFrame columns to an Arrow table."""
    if len(out_df) != tbl.num_rows:
        raise rsgispy.RSGISPyException(
            f"chunk_fn returned {len(out_df)} rows for a chunk of {tbl.num_rows}."
        )
    for col in out_df.columns:
        if col in tbl.schema.names:
            raise rsgispy.RSGISPyException(
                f"Output column '{col}' already exists in the input layer."
            )
        tbl = tbl.append_column(
            str(col), pa_mod.array(out_df[col].to_numpy(), from_pandas=True)
        )
    return tbl


def map_vector_chunks(
    in_vec_file: Union[str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    read_cols: Sequence[str],
    chunk_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    in_vec_lyr: Optional[str] = None,
    geom_vec_file: Optional[Union[str, os.PathLike]] = None,
    geom_vec_lyr: Optional[str] = None,
    id_col: Optional[str] = None,
    copy_cols: Optional[Sequence[str]] = None,
    chunk_size: int = 100_000,
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Stream a vector layer (or attribute table) in chunks, call ``chunk_fn`` on
    the ``read_cols`` of each chunk (as a :class:`pandas.DataFrame`, without
    geometry) and write the copied input columns, the geometry (as undecoded
    WKB) and the columns returned by ``chunk_fn``.

    The geometry comes from ``in_vec_file`` when it has a geometry column.
    When ``in_vec_file`` is an attribute-only table, the geometry (and the
    copied columns) come from ``geom_vec_file``, joined on ``id_col``: if both
    files list the same IDs in the same order (as written by
    :func:`rsgispy.ml.segments.create_segment_features`), the two are streamed
    in lockstep; otherwise the ``read_cols`` of the table are loaded into
    memory indexed by ID (memory ≈ rows × read columns × 8 bytes) and the
    geometry layer is streamed. Without either, an attribute-only table is
    written.

    :param in_vec_file: Input vector layer or attribute table.
    :param out_vec_file: Output file (GeoParquet by default for ``.parquet``).
    :param read_cols: Columns passed to ``chunk_fn``.
    :param chunk_fn: Function ``(DataFrame) -> DataFrame`` returning one row per
                     input row (same order) with the new output columns.
    :param in_vec_lyr: Optional input layer name.
    :param geom_vec_file: Optional geometry layer for attribute-only inputs.
    :param geom_vec_lyr: Optional geometry layer name.
    :param id_col: ID column (required with ``geom_vec_file``; always copied).
    :param copy_cols: Columns copied from the input (or geometry) layer;
                      ``None`` (default) copies all its attribute columns,
                      ``[]`` copies only ``id_col``.
    :param chunk_size: Features per chunk (Default = ``100_000``).
    :param out_vec_lyr: Optional output layer name.
    :param out_format: Optional output format (see :class:`ChunkedVecWriter`).
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param metadata: Optional provenance metadata (see :func:`build_provenance`).
    :param overwrite: If ``True``, replace an existing output.
    :param use_pbar: If ``True``, display a progress bar over chunks.
    """
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    # pylint: disable=too-many-statements
    pa_mod, _ = _require_pyarrow("map_vector_chunks")
    info = read_vec_layer_info(in_vec_file, in_vec_lyr)
    read_cols = list(read_cols)
    missing = [c for c in read_cols if c not in info.columns]
    if missing:
        raise rsgispy.RSGISPyException(
            f"Required column(s) {missing} not found in '{info.path}'."
        )
    tqdm_cls = rsgispy.get_tqdm()
    n_chunks = max(1, math.ceil(info.n_features / int(chunk_size)))

    geom_info: Optional[VecLayerInfo] = None
    if info.geom_col is None and geom_vec_file is not None:
        if id_col is None:
            raise rsgispy.RSGISPyException(
                "id_col is required when geom_vec_file is given."
            )
        geom_info = read_vec_layer_info(geom_vec_file, geom_vec_lyr)
        if geom_info.geom_col is None:
            raise rsgispy.RSGISPyException(
                f"geom_vec_file '{geom_info.path}' has no geometry."
            )
        if id_col not in geom_info.columns or id_col not in info.columns:
            raise rsgispy.RSGISPyException(
                f"id_col '{id_col}' must be in both the table and geometry layer."
            )

    src_info = geom_info if geom_info is not None else info
    copy_list = _resolve_copy_cols(copy_cols, src_info.columns, id_col, exclude=[])
    out_geom = src_info.geom_col is not None
    writer = ChunkedVecWriter(
        out_vec_file,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
        geom_col=_GEOM_NAME if out_geom else None,
        crs=src_info.crs,
        geometry_type=src_info.geometry_type or "Unknown",
        geo_metadata=src_info.geo_metadata if src_info.is_parquet else None,
        metadata=metadata,
        overwrite=overwrite,
    )
    with writer:
        if geom_info is None:
            all_cols = list(dict.fromkeys(copy_list + read_cols))
            for tbl in tqdm_cls(
                iter_vec_batches(
                    info, columns=all_cols, read_geometry=out_geom, chunk_size=chunk_size
                ),
                total=n_chunks,
                disable=not use_pbar,
            ):
                out_df = chunk_fn(tbl.select(read_cols).to_pandas())
                keep = copy_list + ([_GEOM_NAME] if out_geom else [])
                writer.write(_append_outputs(tbl.select(keep), out_df, pa_mod))
            return

        # Features in an attribute-only table; geometry from another layer.
        assert id_col is not None
        tbl_ids = np.concatenate(
            [t.column(id_col).to_numpy() for t in iter_vec_batches(info, [id_col])]
            or [np.empty(0)]
        )
        geo_ids = np.concatenate(
            [
                t.column(id_col).to_numpy()
                for t in iter_vec_batches(geom_info, [id_col])
            ]
            or [np.empty(0)]
        )
        lockstep = tbl_ids.shape == geo_ids.shape and np.array_equal(tbl_ids, geo_ids)
        tbl_cols = list(dict.fromkeys([id_col] + read_cols))
        if lockstep:
            for g_tbl, f_tbl in tqdm_cls(
                zip(
                    iter_vec_batches(
                        geom_info, copy_list, read_geometry=True, chunk_size=chunk_size
                    ),
                    iter_vec_batches(info, tbl_cols, chunk_size=chunk_size),
                ),
                total=n_chunks,
                disable=not use_pbar,
            ):
                out_df = chunk_fn(f_tbl.select(read_cols).to_pandas())
                writer.write(_append_outputs(g_tbl, out_df, pa_mod))
            return
        feat_df = pd.concat(
            list(iter_vec_chunks(info, tbl_cols, chunk_size=chunk_size)),
            ignore_index=True,
        )
        if feat_df[id_col].duplicated().any():
            raise rsgispy.RSGISPyException(
                f"id_col '{id_col}' has duplicate values in '{info.path}'."
            )
        feat_df = feat_df.set_index(id_col, drop=False)
        for g_tbl in tqdm_cls(
            iter_vec_batches(
                geom_info, copy_list, read_geometry=True, chunk_size=chunk_size
            ),
            total=max(1, math.ceil(geom_info.n_features / int(chunk_size))),
            disable=not use_pbar,
        ):
            ids = g_tbl.column(id_col).to_numpy()
            sub = feat_df.reindex(ids)[read_cols].reset_index(drop=True)
            out_df = chunk_fn(sub)
            writer.write(_append_outputs(g_tbl, out_df, pa_mod))


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    valid_col: Optional[str] = None,
    missing: str = "nan",
    fill_value: float = 0.0,
    feat_no_data_val: Optional[float] = None,
    scaler: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build the model feature matrix from a chunk of a feature table, applying
    the valid flag, feature no-data value and missing-value policy, and an
    optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.

    Missing-value policies (``missing``):

    * ``'nan'`` — keep missing values as NaN (suitable for XGBoost, LightGBM
      and CatBoost, which handle NaN natively);
    * ``'drop'`` — rows with any missing feature are excluded (at prediction
      time they receive the output no-data value);
    * ``'fill'`` — missing values are replaced by ``fill_value`` (suitable for
      most scikit-learn estimators).

    :param df: Input chunk.
    :param feature_cols: Ordered feature column names.
    :param valid_col: Optional valid flag column (rows with ``0`` excluded).
    :param missing: Missing-value policy (Default = ``'nan'``).
    :param fill_value: Fill value for ``missing='fill'`` (Default = ``0.0``).
    :param feat_no_data_val: Optional feature value treated as missing (e.g.
                             ``-9999``).
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :return: ``(x_df, row_mask)``: the feature DataFrame for the selected rows
             (columns ``feature_cols``) and the boolean mask of selected rows.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    pol = str(missing).strip().lower()
    if pol not in MISSING_POLICIES:
        raise rsgispy.RSGISPyException(
            f"missing must be one of {list(MISSING_POLICIES)}, got {missing!r}."
        )
    miss_cols = [c for c in feature_cols if c not in df.columns]
    if miss_cols:
        raise rsgispy.RSGISPyException(
            f"Feature column(s) {miss_cols} are missing from the feature data."
        )
    x_arr = df[list(feature_cols)].to_numpy(dtype=np.float64, copy=True)
    if feat_no_data_val is not None and not math.isnan(float(feat_no_data_val)):
        x_arr[x_arr == float(feat_no_data_val)] = np.nan
    row_ok = np.ones(len(df), dtype=bool)
    if valid_col is not None:
        if valid_col not in df.columns:
            raise rsgispy.RSGISPyException(
                f"Valid flag column '{valid_col}' is missing from the feature data."
            )
        row_ok &= df[valid_col].fillna(0).to_numpy() != 0
    finite = np.isfinite(x_arr)
    if pol == "drop":
        row_ok &= np.all(finite, axis=1)
    elif pol == "fill":
        x_arr[~finite] = float(fill_value)
    x_sel = x_arr[row_ok]
    if scaler is not None and x_sel.shape[0] > 0:
        x_sel = np.asarray(scaler.transform(x_sel), dtype=np.float64)
    return pd.DataFrame(x_sel, columns=list(feature_cols)), row_ok


def model_input(estimator: Any, x_df: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
    """Return a DataFrame if the estimator was fitted with column names, else an array."""
    if hasattr(estimator, "feature_names_in_") and estimator.feature_names_in_ is not None:
        est_cols = [str(c) for c in estimator.feature_names_in_]
        if set(est_cols) == set(x_df.columns):
            return x_df[est_cols]
        if len(est_cols) == x_df.shape[1]:
            return x_df.set_axis(est_cols, axis=1)
        raise rsgispy.RSGISPyException(
            f"The model was fitted with {len(est_cols)} features "
            f"({est_cols}) but {x_df.shape[1]} were provided."
        )
    return x_df.to_numpy()


def predict_vector(
    estimator: Any,
    in_vec_file: Union[str, os.PathLike],
    out_vec_file: Union[str, os.PathLike],
    feature_cols: Sequence[str],
    *,
    out_col: Union[str, Sequence[str]] = "prediction",
    in_vec_lyr: Optional[str] = None,
    id_col: Optional[str] = None,
    valid_col: Optional[str] = None,
    missing: str = "nan",
    fill_value: float = 0.0,
    feat_no_data_val: Optional[float] = None,
    out_no_data_val: float = np.nan,
    scaler: Optional[Any] = None,
    geom_vec_file: Optional[Union[str, os.PathLike]] = None,
    geom_vec_lyr: Optional[str] = None,
    copy_cols: Optional[Sequence[str]] = None,
    chunk_size: int = 100_000,
    out_vec_lyr: Optional[str] = None,
    out_format: Optional[str] = None,
    out_compress: Optional[Union[str, bool]] = "snappy",
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Apply a fitted estimator (anything with ``predict()``) to a vector layer
    or attribute table in chunks, reading only the ID, valid flag and feature
    columns for prediction, and writing the layer with the prediction
    column(s) added (see :func:`map_vector_chunks`).

    :param estimator: Fitted estimator.
    :param in_vec_file: Input feature layer or table.
    :param out_vec_file: Output file.
    :param feature_cols: Ordered feature columns used by the model.
    :param out_col: Output column name, or one name per output for
                    multi-output estimators (Default = ``'prediction'``).
    :param in_vec_lyr: Optional input layer name.
    :param id_col: Optional ID column (always copied).
    :param valid_col: Optional valid flag column (``0`` = not predicted).
    :param missing: Missing feature policy (see :func:`prepare_feature_matrix`).
    :param fill_value: Fill value for ``missing='fill'``.
    :param feat_no_data_val: Optional feature no-data value.
    :param out_no_data_val: Output value for rows not predicted (Default = NaN).
    :param scaler: Optional fitted :class:`~rsgispy.ml.scaling.FeatureScaler`.
    :param geom_vec_file: Optional geometry layer for attribute-only inputs.
    :param geom_vec_lyr: Optional geometry layer name.
    :param copy_cols: Columns copied to the output (Default = all).
    :param chunk_size: Features per chunk (Default = ``100_000``).
    :param out_vec_lyr: Optional output layer name.
    :param out_format: Optional output format.
    :param out_compress: Parquet compression (Default = ``'snappy'``).
    :param metadata: Optional provenance metadata.
    :param overwrite: If ``True``, replace an existing output.
    :param use_pbar: If ``True``, display a progress bar.

    .. code:: python

        import joblib
        from rsgispy.ml.vector import predict_vector

        model = joblib.load("rf_model.joblib")
        predict_vector(
            model, "segments.parquet", "segments_pred.parquet",
            feature_cols=["s2_Red_mean", "s2_NIR_mean"], id_col="seg_id",
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    out_names = [out_col] if isinstance(out_col, str) else [str(c) for c in out_col]
    read_cols = list(
        dict.fromkeys(([valid_col] if valid_col else []) + list(feature_cols))
    )

    def _chunk(df: pd.DataFrame) -> pd.DataFrame:
        x_df, ok = prepare_feature_matrix(
            df, feature_cols, valid_col, missing, fill_value, feat_no_data_val, scaler
        )
        out = np.full((len(df), len(out_names)), out_no_data_val, dtype=np.float64)
        if np.any(ok):
            pred = np.asarray(estimator.predict(model_input(estimator, x_df)))
            pred = pred.reshape(pred.shape[0], -1)
            if pred.shape[1] != len(out_names):
                raise rsgispy.RSGISPyException(
                    f"The model returned {pred.shape[1]} outputs but "
                    f"{len(out_names)} output column name(s) were given."
                )
            out[ok] = pred
        return pd.DataFrame(out, columns=out_names)

    map_vector_chunks(
        in_vec_file,
        out_vec_file,
        read_cols,
        _chunk,
        in_vec_lyr=in_vec_lyr,
        geom_vec_file=geom_vec_file,
        geom_vec_lyr=geom_vec_lyr,
        id_col=id_col,
        copy_cols=copy_cols,
        chunk_size=chunk_size,
        out_vec_lyr=out_vec_lyr,
        out_format=out_format,
        out_compress=out_compress,
        metadata=metadata,
        overwrite=overwrite,
        use_pbar=use_pbar,
    )
