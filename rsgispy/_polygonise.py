"""
Tiled, multi-core polygonisation: implementation of
:func:`rsgispy.vector.create.polygonise_raster_tiled`.

This module (and :mod:`rsgispy._polygonise_tile`,
:mod:`rsgispy._polygonise_kernels`) only imports
``numpy``, ``rasterio``, ``shapely`` and the light :mod:`rsgispy` core
modules, so worker processes start quickly (importing
:mod:`rsgispy.vector` would add several seconds per worker).

Processing outline
------------------

1. The raster is split into tiles (:func:`plan_tile_specs`).
   Each tile is read and polygonised independently (in worker processes when
   ``n_jobs > 1``) by one of the backends in :mod:`rsgispy._polygonise_tile`,
   in global *pixel* coordinates.
2. Inside the worker, each feature is classified as

   * *interior*: it does not touch an internal tile edge (touching the image
     edge is fine). It is final: it is converted to map coordinates and WKB
     in the worker and written straight away;
   * *seam piece*: it touches an internal tile edge and may continue in a
     neighbouring tile. It is returned as WKB in pixel coordinates.

3. Results are consumed in tile order (identical output for any ``n_jobs``).
4. Seam pieces of the same value that are connected across tile edges (a
   shared edge for 4-connectivity, any contact for 8-connectivity) are
   grouped with union-find and unioned (in exact integer pixel
   coordinates); collinear vertices left on the seams are removed.
5. Optionally, rows are sorted by value and/or dissolved by value. This
   uses an out-of-core bucket sort: rows are first written to a temporary
   Arrow IPC file, then distributed to buckets of whole values, each bucket
   is sorted (and dissolved) in memory and appended to the output.

Memory use is bounded by the tile size, the number of tiles in flight, the
seam pieces (proportional to the total length of the tile seams, not the
image area) and, when sorting/dissolving, one bucket plus one ``int64``
key per output row.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import rasterio
import rasterio.windows
import shapely

import rsgispy
from rsgispy._numba import disable_numba
from rsgispy._polygonise_kernels import merge_wkb_runs_kernel
from rsgispy._polygonise_tile import (
    TilePolys,
    build_wkb,
    orient_exterior_cw,
    orient_rings,
    polygonise_tile_numba,
    polygonise_tile_rasterio,
    remove_collinear,
    to_map_coords,
)
from rsgispy._unionfind import uf_min_root_kernel

__all__ = [
    "PolygoniseParams",
    "TileSpec",
    "plan_tile_specs",
    "check_n_jobs",
    "run_tiled_polygonise",
]

#: Maximum rows per bucket in the out-of-core sort (module level so tests can
#: exercise the multi-bucket path with a small value).
SORT_BUCKET_ROWS = 2_000_000

#: Wall-clock seconds of each stage of the last run in this process
#: (``'tiles'``, ``'seams'``, ``'sort'``, ``'total'``), for benchmarking.
LAST_RUN_TIMINGS: Dict[str, float] = {}

_GEOM_COL = "geometry"
_KEY_COL = "__rsgispy_sort_key"
_FIRST_COL = "__rsgispy_first_pxl"
_MAX_BINARY_CHUNK = 2**31 - 1


class TileSpec(NamedTuple):
    """
    Tile window passed to workers (a plain tuple, so unpickling it in a
    worker does not import :mod:`rsgispy.segmentation`).
    """

    tile_id: int
    row_off: int
    col_off: int
    height: int
    width: int


def plan_tile_specs(height: int, width: int, tile_size: Any) -> List[TileSpec]:
    """
    Split a ``height`` x ``width`` raster into row-major tiles of
    ``tile_size`` pixels (``int`` or ``(rows, cols)``).
    """
    if isinstance(tile_size, (tuple, list)) and len(tile_size) == 2:
        sizes = (tile_size[0], tile_size[1])
    else:
        sizes = (tile_size, tile_size)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) <= 0
        for v in sizes
    ):
        raise rsgispy.RSGISPyException(
            f"tile_size must be a positive int or (rows, cols) tuple, got {tile_size!r}."
        )
    t_rows, t_cols = int(sizes[0]), int(sizes[1])
    specs: List[TileSpec] = []
    for row_off in range(0, height, t_rows):
        for col_off in range(0, width, t_cols):
            specs.append(
                TileSpec(
                    len(specs),
                    row_off,
                    col_off,
                    min(t_rows, height - row_off),
                    min(t_cols, width - col_off),
                )
            )
    return specs


def check_n_jobs(n_jobs: Any) -> int:
    """Return ``n_jobs`` as an ``int >= 1`` or raise an exception."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise rsgispy.RSGISPyException(f"n_jobs must be a positive integer, got {n_jobs!r}.")
    return int(n_jobs)


@dataclass(frozen=True)
class PolygoniseParams:  # pylint: disable=too-many-instance-attributes
    """Picklable settings shared by the main process and tile workers."""

    input_img: str
    img_band: int
    mask_img: Optional[str]
    mask_band: int
    no_data_val: Optional[float]
    use_8_conn: bool
    backend: str
    use_numba: bool
    img_height: int
    img_width: int
    geotransform: Tuple[float, float, float, float, float, float]
    flip: bool


@dataclass
class WkbChunk:
    """
    Features as values, pixel counts, first pixels (``row * img_width + col``
    of each feature's first pixel in raster order, used to order ties
    deterministically) and a WKB buffer with offsets.
    """

    values: np.ndarray
    counts: np.ndarray
    wkb_buf: np.ndarray
    wkb_off: np.ndarray
    first: np.ndarray

    @property
    def n_rows(self) -> int:
        """Number of features."""
        return int(self.values.shape[0])

    def wkb_list(self) -> np.ndarray:
        """Return the WKB of each feature as an object array of ``bytes``."""
        raw = self.wkb_buf.tobytes()
        offs = self.wkb_off.tolist()
        out = np.empty(self.n_rows, dtype=object)
        out[:] = [raw[offs[i] : offs[i + 1]] for i in range(self.n_rows)]  # noqa: E203
        return out


@dataclass
class TileResult:
    """Output of one tile task."""

    tile_id: int
    interior: WkbChunk
    seam: WkbChunk


def _first_pixels(polys: TilePolys, img_width: int) -> np.ndarray:
    """
    Return ``row * img_width + col`` of each feature's first pixel in raster
    order: the top-left corner of that pixel is the feature's smallest
    ``(y, x)`` vertex.
    """
    if polys.n_feats == 0:
        return np.zeros(0, dtype=np.int64)
    xy_int = polys.coords.astype(np.int64)
    key = (xy_int[:, 1] << 32) + xy_int[:, 0]
    starts = polys.ring_offsets[polys.part_offsets[polys.feat_offsets[:-1]]]
    first = np.minimum.reduceat(key, starts)
    return (first >> 32) * int(img_width) + (first & 0xFFFFFFFF)


def _polys_to_chunk(
    polys: TilePolys,
    gt: Optional[Tuple[float, ...]],
    flip: bool,
    img_width: int,
) -> WkbChunk:
    """Encode :class:`TilePolys` as WKB (map coordinates when ``gt`` is given)."""
    first = _first_pixels(polys, img_width)
    if gt is not None:
        polys = orient_rings(polys, flip)
        coords = to_map_coords(polys.coords, gt)
    else:
        coords = polys.coords
    buf, off = build_wkb(coords, polys.ring_offsets, polys.part_offsets, polys.feat_offsets)
    return WkbChunk(
        values=polys.values, counts=polys.counts, wkb_buf=buf, wkb_off=off, first=first
    )


def _read_tile(params: PolygoniseParams, tile: TileSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Read the values and validity mask of ``tile``."""
    win = rasterio.windows.Window(tile.col_off, tile.row_off, tile.width, tile.height)
    with rasterio.open(params.input_img, "r") as src:
        vals = src.read(params.img_band, window=win)
    valid = np.ones(vals.shape, dtype=bool)
    if params.mask_img is not None:
        with rasterio.open(params.mask_img, "r") as msk:
            valid &= msk.read(params.mask_band, window=win) != 0
    if params.no_data_val is not None:
        if np.isnan(params.no_data_val):
            if np.issubdtype(vals.dtype, np.floating):
                valid &= ~np.isnan(vals)
        else:
            valid &= vals != params.no_data_val
    return vals, valid


def polygonise_tile_task(params: PolygoniseParams, tile: TileSpec) -> TileResult:
    """
    Polygonise one tile and split its features into interior features (WKB in
    map coordinates) and seam pieces (WKB in global pixel coordinates).
    """
    ctx = nullcontext() if params.use_numba else disable_numba()
    with ctx:
        vals, valid = _read_tile(params, tile)
        poly_fn = (
            polygonise_tile_numba if params.backend == "numba" else polygonise_tile_rasterio
        )
        polys = poly_fn(vals, valid, params.use_8_conn, tile.row_off, tile.col_off)
        del vals, valid
        bbox = polys.bbox
        r_1 = tile.row_off + tile.height - 1
        c_1 = tile.col_off + tile.width - 1
        seam = np.zeros(polys.n_feats, dtype=bool)
        if tile.row_off > 0:
            seam |= bbox[:, 0] == tile.row_off
        if r_1 < params.img_height - 1:
            seam |= bbox[:, 1] == r_1
        if tile.col_off > 0:
            seam |= bbox[:, 2] == tile.col_off
        if c_1 < params.img_width - 1:
            seam |= bbox[:, 3] == c_1
        if seam.any():
            seam_chunk = _polys_to_chunk(
                polys.subset(np.nonzero(seam)[0]), None, False, params.img_width
            )
            polys = polys.subset(np.nonzero(~seam)[0])
        else:
            seam_chunk = _polys_to_chunk(
                polys.subset(np.zeros(0, dtype=np.int64)), None, False, params.img_width
            )
        return TileResult(
            tile_id=tile.tile_id,
            interior=_polys_to_chunk(polys, params.geotransform, params.flip, params.img_width),
            seam=seam_chunk,
        )


def _worker_init() -> None:
    """Pin BLAS/OpenMP/NumExpr thread pools to one thread in worker processes."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _iter_tile_results(
    params: PolygoniseParams,
    tiles: Sequence[Any],
    n_jobs: int,
) -> Iterator[TileResult]:
    """Yield tile results in tile order (serially or from worker processes)."""
    if n_jobs <= 1 or len(tiles) <= 1:
        for tile in tiles:
            yield polygonise_tile_task(params, tile)
        return
    max_ahead = 2 * n_jobs
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_worker_init) as executor:
        running: Dict[Future, int] = {}
        done: Dict[int, TileResult] = {}
        next_submit = 0
        next_yield = 0
        while next_yield < len(tiles):
            while next_submit < len(tiles) and next_submit < next_yield + max_ahead:
                fut = executor.submit(polygonise_tile_task, params, tiles[next_submit])
                running[fut] = next_submit
                next_submit += 1
            if next_yield in done:
                yield done.pop(next_yield)
                next_yield += 1
                continue
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                t_idx = running.pop(fut)
                done[t_idx] = fut.result()


def _union_groups(geoms: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Union the geometries of each group (vectorised per group size)."""
    out = np.empty(n_groups, dtype=object)
    order = np.argsort(group, kind="stable")
    sizes = np.bincount(group, minlength=n_groups)
    starts = np.zeros(n_groups, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    single = sizes == 1
    out[single] = geoms[order[starts[single]]]
    for size in np.unique(sizes[~single]):
        grp_ids = np.nonzero(sizes == size)[0]
        idx = starts[grp_ids][:, None] + np.arange(size)[None, :]
        mat = geoms[order[idx]]
        out[grp_ids] = shapely.union_all(mat, axis=1)
    return out


def _empty_chunk() -> WkbChunk:
    """Return a :class:`WkbChunk` with no features."""
    return WkbChunk(
        values=np.zeros(0, dtype=np.int64),
        counts=np.zeros(0, dtype=np.int64),
        wkb_buf=np.zeros(0, dtype=np.uint8),
        wkb_off=np.zeros(1, dtype=np.int64),
        first=np.zeros(0, dtype=np.int64),
    )


class SeamMerger:  # pylint: disable=too-many-instance-attributes
    """
    Merge the seam pieces of polygons cut by tile edges.

    Pieces (in pixel coordinates) are added as tiles finish. When a complete
    row of tiles is finished, :meth:`flush` groups the pending pieces of the
    same value that are connected across tile edges (a shared edge with
    4-connectivity, any contact with 8-connectivity; union-find over the
    candidate pairs from a spatial index) and unions each group. Groups that
    cannot continue into the next row of tiles (no piece on the last pixel
    row of the finished row) are final and returned as WKB in map
    coordinates; the others are kept as one partial piece each. Only the
    seam pieces of about one row of tiles are therefore held in memory, and
    with ``n_jobs > 1`` the merging overlaps with the workers' processing of
    the following rows.
    """

    def __init__(self, use_8_conn: bool, gt: Tuple[float, ...], flip: bool) -> None:
        self.use_8_conn = use_8_conn
        self.gt = gt
        self.flip = flip
        self.geoms: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.counts: List[np.ndarray] = []
        self.first: List[np.ndarray] = []
        self.dirty: List[np.ndarray] = []

    def add(self, chunk: WkbChunk) -> None:
        """Add the seam pieces of a tile (WKB in pixel coordinates)."""
        if chunk.n_rows == 0:
            return
        self.geoms.append(shapely.from_wkb(chunk.wkb_list()))
        self.values.append(np.asarray(chunk.values))
        self.counts.append(np.asarray(chunk.counts, dtype=np.int64))
        self.first.append(np.asarray(chunk.first, dtype=np.int64))
        self.dirty.append(np.zeros(chunk.n_rows, dtype=bool))

    def _groups(self, geoms: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group connected pieces. Returns ``(group, first_piece)``: the group of
        each piece (groups numbered by their smallest piece index) and the
        smallest piece index of each group.
        """
        tree = shapely.STRtree(geoms)
        left, right = tree.query(geoms, predicate="intersects")
        sel = (left < right) & (values[left] == values[right])
        left = left[sel]
        right = right[sel]
        if (not self.use_8_conn) and left.shape[0] > 0:
            # 4-connectivity: the pieces must share an edge (boundary/boundary
            # intersection of dimension 1), not just a corner.
            touch = shapely.relate_pattern(geoms[left], geoms[right], "****1****")
            left = left[touch]
            right = right[touch]
        roots = uf_min_root_kernel(
            np.ascontiguousarray(left, dtype=np.int64),
            np.ascontiguousarray(right, dtype=np.int64),
            geoms.shape[0],
            0,
        )
        first_piece, group = np.unique(roots, return_inverse=True)
        return group.reshape(-1), first_piece

    def flush(self, bottom_row: Optional[int] = None) -> WkbChunk:
        """
        Merge the pending pieces and return the final features (in order of
        each feature's first piece). ``bottom_row`` is the last pixel row of
        the finished row of tiles (``None`` when all tiles are finished).
        """
        # pylint: disable=too-many-locals
        if not self.geoms:
            return _empty_chunk()
        geoms = np.concatenate(self.geoms)
        values = np.concatenate(self.values)
        counts = np.concatenate(self.counts)
        first = np.concatenate(self.first)
        dirty = np.concatenate(self.dirty)
        group, rep = self._groups(geoms, values)
        n_groups = rep.shape[0]
        sizes = np.bincount(group, minlength=n_groups)
        grp_first = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(grp_first, group, first)
        grp_counts = np.bincount(group, weights=counts, minlength=n_groups).astype(np.int64)
        grp_dirty = (sizes > 1) | (np.bincount(group, weights=dirty, minlength=n_groups) > 0)
        merged = _union_groups(geoms, group, n_groups)
        if bottom_row is None:
            keep_open = np.zeros(n_groups, dtype=bool)
        else:
            on_bottom = (np.rint(shapely.bounds(geoms)[:, 3]).astype(np.int64) - 1) == bottom_row
            keep_open = np.bincount(group, weights=on_bottom, minlength=n_groups) > 0
        open_idx = np.nonzero(keep_open)[0]
        self.geoms = [merged[open_idx]] if open_idx.shape[0] else []
        self.values = [values[rep[open_idx]]] if open_idx.shape[0] else []
        self.counts = [grp_counts[open_idx]] if open_idx.shape[0] else []
        self.first = [grp_first[open_idx]] if open_idx.shape[0] else []
        self.dirty = [grp_dirty[open_idx]] if open_idx.shape[0] else []

        done = np.nonzero(~keep_open)[0]
        if done.shape[0] == 0:
            return _empty_chunk()
        final = merged[done]
        clean = grp_dirty[done]
        if clean.any():
            final[clean] = orient_exterior_cw(remove_collinear(final[clean]))
        if self.flip:
            final = shapely.reverse(final)
        final = shapely.transform(final, lambda xy: to_map_coords(xy, self.gt))
        wkbs = shapely.to_wkb(final)
        lens = np.fromiter((len(w) for w in wkbs), dtype=np.int64, count=wkbs.shape[0])
        off = np.zeros(lens.shape[0] + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        return WkbChunk(
            values=values[rep[done]],
            counts=grp_counts[done],
            wkb_buf=np.frombuffer(b"".join(wkbs.tolist()), dtype=np.uint8),
            wkb_off=off,
            first=grp_first[done],
        )

    @property
    def n_pending(self) -> int:
        """Number of pending seam pieces."""
        return int(sum(g.shape[0] for g in self.geoms))


class _TableBuilder:
    """Build Arrow tables from :class:`WkbChunk` objects."""

    def __init__(
        self,
        val_col: str,
        val_dtype: np.dtype,
        count_col: Optional[str],
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from rsgispy.ml.vector import _require_pyarrow

        self.pa, _ = _require_pyarrow("polygonise_raster_tiled")
        self.val_col = val_col
        self.val_dtype = val_dtype
        self.count_col = count_col
        fields = [self.pa.field(val_col, self.pa.from_numpy_dtype(val_dtype))]
        if count_col is not None:
            fields.append(self.pa.field(count_col, self.pa.int64()))
        fields.append(self.pa.field(_GEOM_COL, self.pa.binary()))
        self.schema = self.pa.schema(fields)

    def tables(self, chunk: WkbChunk) -> List[Any]:
        """Return the chunk as one or more tables (WKB < 2 GiB per table)."""
        pa_mod = self.pa
        n_rows = chunk.n_rows
        if n_rows == 0:
            return []
        out = []
        start = 0
        while start < n_rows:
            stop = int(
                np.searchsorted(
                    chunk.wkb_off, chunk.wkb_off[start] + _MAX_BINARY_CHUNK, side="right"
                )
                - 1
            )
            stop = min(max(stop, start + 1), n_rows)
            b_0 = int(chunk.wkb_off[start])
            b_1 = int(chunk.wkb_off[stop])
            offs = (chunk.wkb_off[start : stop + 1] - b_0).astype(np.int32)  # noqa: E203
            geom = pa_mod.Array.from_buffers(
                pa_mod.binary(),
                stop - start,
                [None, pa_mod.py_buffer(offs), pa_mod.py_buffer(chunk.wkb_buf[b_0:b_1])],
            )
            cols = [pa_mod.array(chunk.values[start:stop].astype(self.val_dtype))]
            if self.count_col is not None:
                cols.append(pa_mod.array(chunk.counts[start:stop].astype(np.int64)))
            cols.append(geom)
            out.append(pa_mod.Table.from_arrays(cols, schema=self.schema))
            start = stop
        return out

    def empty(self) -> Any:
        """Return an empty table with the output schema."""
        return self.schema.empty_table()


class _ChunkedSink:
    """Buffer tables and write them in chunks of ``chunk_size`` rows."""

    def __init__(self, write_fn: Callable[[Any], None], chunk_size: int, pa_mod: Any) -> None:
        self.write_fn = write_fn
        self.chunk_size = int(chunk_size)
        self.pa = pa_mod
        self.buf: List[Any] = []
        self.n_buf = 0
        self.n_written = 0

    def add(self, table: Any) -> None:
        """Add a table (written once ``chunk_size`` rows are buffered)."""
        if table.num_rows == 0:
            return
        self.buf.append(table)
        self.n_buf += table.num_rows
        while self.n_buf >= self.chunk_size:
            tbl = self.pa.concat_tables(self.buf)
            head = tbl.slice(0, self.chunk_size)
            rest = tbl.slice(self.chunk_size)
            self.write_fn(head)
            self.n_written += head.num_rows
            self.buf = [rest] if rest.num_rows > 0 else []
            self.n_buf = rest.num_rows

    def flush(self) -> None:
        """Write any buffered rows."""
        if self.n_buf > 0:
            tbl = self.pa.concat_tables(self.buf)
            self.write_fn(tbl)
            self.n_written += tbl.num_rows
        self.buf = []
        self.n_buf = 0


def _dissolve_sorted(table: Any, keys: np.ndarray, builder: _TableBuilder) -> List[Any]:
    """
    Combine consecutive rows with equal ``keys`` (one value per key) into one
    feature (a MultiPolygon when there are several parts). The WKB is merged
    at byte level, so the geometries are not parsed.
    """
    # pylint: disable=too-many-locals
    n_rows = keys.shape[0]
    if n_rows == 0:
        return [table]
    run_start = np.concatenate([[0], np.nonzero(keys[1:] != keys[:-1])[0] + 1])
    run_len = np.diff(np.append(run_start, n_rows)).astype(np.int64)
    if np.all(run_len == 1):
        return [table]
    geom = table.column(_GEOM_COL).combine_chunks()
    _, off_buf, data_buf = geom.buffers()
    offs = np.frombuffer(off_buf, dtype=np.int32)[
        geom.offset : geom.offset + len(geom) + 1  # noqa: E203
    ].astype(np.int64)
    buf = np.frombuffer(data_buf, dtype=np.uint8) if data_buf is not None else np.zeros(0, np.uint8)
    out_buf, out_off = merge_wkb_runs_kernel(buf, offs, run_start.astype(np.int64), run_len)
    values = table.column(builder.val_col).to_numpy()[run_start]
    if builder.count_col is not None:
        cnt = table.column(builder.count_col).to_numpy().astype(np.int64)
        counts = np.add.reduceat(cnt, run_start)
    else:
        counts = np.zeros(run_start.shape[0], dtype=np.int64)
    return builder.tables(
        WkbChunk(
            values=values,
            counts=counts,
            wkb_buf=out_buf,
            wkb_off=out_off,
            first=np.zeros(0, dtype=np.int64),
        )
    )


def _process_bucket(
    table: Any,
    keys: np.ndarray,
    first: np.ndarray,
    dissolve: bool,
    builder: _TableBuilder,
    sink: _ChunkedSink,
) -> None:
    """
    Sort one bucket by key, then first pixel (raster order), optionally
    dissolve, and write it.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if keys.shape[0] == 0:
        order = np.zeros(0, dtype=np.int64)
    else:
        k_min = int(keys.min())
        f_span = int(first.max()) + 1
        if (int(keys.max()) - k_min + 1) * f_span < 2**62:
            order = np.argsort((keys - k_min) * f_span + first, kind="stable")
        else:
            order = np.lexsort((first, keys))
    if not np.all(order == np.arange(order.shape[0])):
        table = table.take(builder.pa.array(order))
        keys = keys[order]
    if dissolve:
        for tbl in _dissolve_sorted(table, keys, builder):
            sink.add(tbl)
    else:
        sink.add(table)


def sort_dissolve(
    stage_path: Optional[str],
    stage_tables: Optional[List[Any]],
    values: np.ndarray,
    first: np.ndarray,
    sort_by_value: bool,
    dissolve: bool,
    builder: _TableBuilder,
    sink: _ChunkedSink,
    tmp_dir: str,
) -> None:
    """
    Write the staged rows (an Arrow IPC file or in-memory tables) to ``sink``
    sorted by value and/or dissolved by value, using an out-of-core bucket
    sort when there are more than :data:`SORT_BUCKET_ROWS` rows.

    Rows are ordered by value (``sort_by_value``), or by the first pixel of
    each value in raster order (dissolve only), with ties broken by each
    row's first pixel (``first``), so the output does not depend on the tile
    size, number of workers or backend.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    pa_mod = builder.pa
    import pyarrow.ipc  # pylint: disable=import-outside-toplevel,import-error

    n_rows = values.shape[0]
    uniq, inv = np.unique(values, return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    if sort_by_value:
        keys = inv
    else:
        # dissolve only: values in raster order of their first pixel
        val_first = np.full(uniq.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(val_first, inv, first)
        rank = np.empty(uniq.shape[0], dtype=np.int64)
        rank[np.argsort(val_first, kind="stable")] = np.arange(uniq.shape[0])
        keys = rank[inv]

    def iter_batches() -> Iterator[Any]:
        if stage_tables is not None:
            for tbl in stage_tables:
                yield from tbl.to_batches()
        else:
            with pa_mod.OSFile(stage_path, "rb") as src:
                reader = pyarrow.ipc.open_file(src)
                for b_i in range(reader.num_record_batches):
                    yield reader.get_batch(b_i)

    if n_rows <= SORT_BUCKET_ROWS:
        batches = [b for b in iter_batches() if b.num_rows > 0]
        table = (
            pa_mod.Table.from_batches(batches, schema=builder.schema)
            if batches
            else builder.empty()
        )
        _process_bucket(table, keys, first, dissolve, builder, sink)
        return

    key_counts = np.bincount(keys)
    before = np.concatenate([[0], np.cumsum(key_counts)[:-1]])
    _, bucket_of_key = np.unique(before // SORT_BUCKET_ROWS, return_inverse=True)
    bucket_of_key = bucket_of_key.reshape(-1)
    n_buckets = int(bucket_of_key.max()) + 1
    bucket_schema = builder.schema.append(pa_mod.field(_KEY_COL, pa_mod.int64())).append(
        pa_mod.field(_FIRST_COL, pa_mod.int64())
    )
    bucket_dir = tempfile.mkdtemp(prefix="rsgispy_sort_", dir=tmp_dir)
    writers: Dict[int, Tuple[Any, Any]] = {}

    def _close_writers() -> None:
        for sink_file, wrt in writers.values():
            try:
                wrt.close()
            finally:
                sink_file.close()
        writers.clear()

    try:
        row0 = 0
        for batch in iter_batches():
            n_b = batch.num_rows
            if n_b == 0:
                continue
            b_keys = keys[row0 : row0 + n_b]  # noqa: E203
            b_bucket = bucket_of_key[b_keys]
            row0 += n_b
            tbl = (
                pa_mod.Table.from_batches([batch])
                .append_column(_KEY_COL, pa_mod.array(b_keys))
                .append_column(_FIRST_COL, pa_mod.array(first[row0 - n_b : row0]))
            )
            order = np.argsort(b_bucket, kind="stable")
            s_bucket = b_bucket[order]
            cuts = np.nonzero(s_bucket[1:] != s_bucket[:-1])[0] + 1
            starts = np.concatenate([[0], cuts])
            stops = np.append(cuts, n_b)
            for s_0, s_1 in zip(starts, stops):
                b_id = int(s_bucket[s_0])
                part = tbl.take(pa_mod.array(order[s_0:s_1]))
                if b_id not in writers:
                    sink_file = pa_mod.OSFile(
                        os.path.join(bucket_dir, f"bucket_{b_id}.arrow"), "wb"
                    )
                    try:
                        wrt = pyarrow.ipc.new_file(sink_file, bucket_schema)
                    except Exception:
                        sink_file.close()
                        raise
                    writers[b_id] = (sink_file, wrt)
                writers[b_id][1].write_table(part)
        _close_writers()
        for b_id in range(n_buckets):
            path = os.path.join(bucket_dir, f"bucket_{b_id}.arrow")
            if not os.path.exists(path):
                continue
            with pa_mod.OSFile(path, "rb") as src:
                table = pyarrow.ipc.open_file(src).read_all()
            os.remove(path)
            b_keys = table.column(_KEY_COL).to_numpy()
            b_first = table.column(_FIRST_COL).to_numpy()
            table = table.drop_columns([_KEY_COL, _FIRST_COL])
            _process_bucket(table, b_keys, b_first, dissolve, builder, sink)
            del table
    finally:
        _close_writers()
        shutil.rmtree(bucket_dir, ignore_errors=True)


def run_tiled_polygonise(
    params: PolygoniseParams,
    tiles: Sequence[Any],
    n_jobs: int,
    builder: _TableBuilder,
    final_write: Callable[[Any], None],
    chunk_size: int,
    sort_by_value: bool,
    dissolve_by_value: bool,
    tmp_dir: Optional[str],
    use_pbar: bool,
) -> int:
    """
    Run the tiled polygonisation and pass the output tables to
    ``final_write`` (in chunks of ``chunk_size`` rows). Returns the number of
    rows written.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    pa_mod = builder.pa
    import pyarrow.ipc  # pylint: disable=import-outside-toplevel,import-error

    final_sink = _ChunkedSink(final_write, chunk_size, pa_mod)
    post = sort_by_value or dissolve_by_value
    in_memory = tmp_dir is None
    stage_tables: Optional[List[Any]] = [] if (post and in_memory) else None
    stage_path: Optional[str] = None
    stage_file: Any = None
    stage_writer: Any = None
    work_dir: Optional[str] = None
    if post and not in_memory:
        work_dir = tempfile.mkdtemp(prefix="rsgispy_poly_", dir=tmp_dir)
        stage_path = os.path.join(work_dir, "stage.arrow")
        try:
            stage_file = pa_mod.OSFile(stage_path, "wb")
            stage_writer = pyarrow.ipc.new_file(
                stage_file,
                builder.schema,
                options=pyarrow.ipc.IpcWriteOptions(compression="lz4"),
            )
        except Exception:
            if stage_file is not None:
                stage_file.close()
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

    vals_parts: List[np.ndarray] = []
    first_parts: List[np.ndarray] = []

    def stage_write(tbl: Any) -> None:
        if stage_tables is not None:
            stage_tables.append(tbl)
        else:
            stage_writer.write_table(tbl)

    stage_sink = _ChunkedSink(stage_write, chunk_size, pa_mod) if post else final_sink

    def add_chunk(chunk: WkbChunk) -> None:
        if chunk.n_rows == 0:
            return
        if post:
            vals_parts.append(np.asarray(chunk.values))
            first_parts.append(np.asarray(chunk.first, dtype=np.int64))
        for tbl in builder.tables(chunk):
            stage_sink.add(tbl)

    LAST_RUN_TIMINGS.clear()
    t_start = time.perf_counter()
    try:
        merger = SeamMerger(params.use_8_conn, params.geotransform, params.flip)
        tiles_iter = _iter_tile_results(params, tiles, n_jobs)
        if use_pbar:
            tqdm = rsgispy.get_tqdm()
            tiles_iter = tqdm(tiles_iter, total=len(tiles), desc="Polygonising tiles")
        for res in tiles_iter:
            add_chunk(res.interior)
            merger.add(res.seam)
            tile = tiles[res.tile_id]
            t_next = res.tile_id + 1
            if t_next < len(tiles) and tiles[t_next].row_off != tile.row_off:
                # A row of tiles is complete: finalise the seam polygons that
                # cannot continue into the next row.
                add_chunk(merger.flush(tile.row_off + tile.height - 1))
        t_tiles = time.perf_counter()
        LAST_RUN_TIMINGS["tiles"] = t_tiles - t_start
        add_chunk(merger.flush(None))
        del merger
        stage_sink.flush()
        t_seams = time.perf_counter()
        LAST_RUN_TIMINGS["seams"] = t_seams - t_tiles
        if post:
            if stage_writer is not None:
                stage_writer.close()
                stage_writer = None
            if stage_file is not None:
                stage_file.close()
                stage_file = None
            values = (
                np.concatenate(vals_parts) if vals_parts else np.zeros(0, dtype=builder.val_dtype)
            )
            sort_dissolve(
                stage_path,
                stage_tables,
                values,
                np.concatenate(first_parts) if first_parts else np.zeros(0, np.int64),
                sort_by_value,
                dissolve_by_value,
                builder,
                final_sink,
                work_dir if work_dir is not None else tempfile.gettempdir(),
            )
            final_sink.flush()
            LAST_RUN_TIMINGS["sort"] = time.perf_counter() - t_seams
        LAST_RUN_TIMINGS["total"] = time.perf_counter() - t_start
    finally:
        if stage_writer is not None:
            try:
                stage_writer.close()
            finally:
                stage_writer = None
        if stage_file is not None:
            try:
                stage_file.close()
            finally:
                stage_file = None
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
    return final_sink.n_written
