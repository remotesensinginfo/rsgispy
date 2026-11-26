"""
Tile-graph DAG planning, validation, halo propagation, and streaming
execution for :mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
import math
import os
import queue
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rasterio.transform import Affine

import rsgispy
from rsgispy._tilegraph._nodes import (
    HeaderCache,
    NodeMeta,
    RunResources,
    SourceGrid,
    TileContext,
    TileNode,
    TileSink,
    TileTensor,
    _crop_tensor,
)

#: Relative tolerance (of the pixel size) for the scale and rotation terms
#: of source transforms.
GRID_SCALE_RTOL = 1e-6
#: Tolerance, in pixels, for the origins of source transforms.
GRID_ORIGIN_TOL_PX = 1e-3


def _transforms_match(t_a: Affine, t_b: Affine) -> bool:
    """``True`` if two geotransforms describe the same grid within tolerance."""
    pix = max(math.hypot(t_a.a, t_a.d), math.hypot(t_a.b, t_a.e), 1e-300)
    for v_a, v_b in ((t_a.a, t_b.a), (t_a.b, t_b.b), (t_a.d, t_b.d), (t_a.e, t_b.e)):
        if abs(v_a - v_b) > GRID_SCALE_RTOL * pix:
            return False
    return (
        abs(t_a.c - t_b.c) <= GRID_ORIGIN_TOL_PX * pix
        and abs(t_a.f - t_b.f) <= GRID_ORIGIN_TOL_PX * pix
    )


def _crs_match(crs_a: Any, crs_b: Any) -> bool:
    """``True`` if two CRS definitions are equal."""
    if crs_a == crs_b:
        return True
    try:
        import pyproj  # pylint: disable=import-outside-toplevel

        return pyproj.CRS.from_user_input(crs_a) == pyproj.CRS.from_user_input(crs_b)
    except Exception:  # pylint: disable=broad-exception-caught
        return str(crs_a) == str(crs_b)


def check_source_grids(grids: Sequence[SourceGrid]) -> SourceGrid:
    """
    Check that all source grids have the same dimensions, transform and CRS
    (within :data:`GRID_SCALE_RTOL` / :data:`GRID_ORIGIN_TOL_PX`); a missing
    (``None``) transform or CRS is not compared. Returns the first grid.

    :param grids: Grids of the graph's sources.
    :return: Reference (first) grid.
    """
    ref = grids[0]
    for grid in grids[1:]:
        if (grid.height, grid.width) != (ref.height, ref.width):
            raise rsgispy.RSGISPyException(
                f"TileGraph sources must have matching dimensions; "
                f"got {(grid.height, grid.width)} for {grid.label} vs "
                f"{(ref.height, ref.width)} for {ref.label}."
            )
        if (
            grid.transform is not None
            and ref.transform is not None
            and not _transforms_match(ref.transform, grid.transform)
        ):
            raise rsgispy.RSGISPyException(
                f"TileGraph sources must be on the same pixel grid; the "
                f"transform of {grid.label} ({tuple(grid.transform)[:6]}) differs "
                f"from that of {ref.label} ({tuple(ref.transform)[:6]})."
            )
        if (
            grid.crs is not None
            and ref.crs is not None
            and not _crs_match(ref.crs, grid.crs)
        ):
            raise rsgispy.RSGISPyException(
                f"TileGraph sources must have the same CRS; {grid.label} has "
                f"{grid.crs} but {ref.label} has {ref.crs}."
            )
    return ref


def _topo_sort(roots: Sequence[TileNode]) -> Tuple[TileNode, ...]:
    """Topologically sort the nodes reachable from ``roots``; reject cycles."""
    visited: Dict[TileNode, int] = {}
    order: List[TileNode] = []

    def _dfs(node: TileNode) -> None:
        state = visited.get(node, 0)
        if state == 1:
            raise rsgispy.RSGISPyException(
                "Cycle detected in TileGraph definition."
            )
        if state == 2:
            return
        visited[node] = 1
        for parent in node.inputs:
            if not isinstance(parent, TileNode):
                raise rsgispy.RSGISPyException(
                    f"{node.label} has an input that is not a TileNode "
                    f"({type(parent).__name__})."
                )
            _dfs(parent)
        visited[node] = 2
        order.append(node)

    for root in roots:
        _dfs(root)
    return tuple(order)


def _derive_metadata(
    order: Sequence[TileNode],
    node_meta: Dict[TileNode, NodeMeta],
    source_grids: Dict[TileNode, SourceGrid],
) -> None:
    """
    Derive the metadata of the nodes in topological ``order`` (reading the
    source headers once), filling ``node_meta`` and ``source_grids``.
    """
    headers = HeaderCache()
    for node in order:
        if node.is_source:
            grid, meta = node.source_info(headers)
            source_grids[node] = grid
        else:
            meta = node.infer_meta([node_meta[p] for p in node.inputs])
        if not isinstance(meta, NodeMeta):
            raise rsgispy.RSGISPyException(
                f"{node.label} returned {type(meta).__name__} as metadata, "
                "expected NodeMeta."
            )
        node_meta[node] = meta


def infer_node_meta(node: TileNode) -> NodeMeta:
    """
    Return the metadata (band count, band names, dtype, no-data value) of a
    node's output, derived from its sources and parameters exactly as
    :class:`TileGraph` does when it is built (the source headers are read,
    no pixels).

    :param node: Any :class:`TileNode`.
    :return: :class:`NodeMeta` of ``node``.
    :raises rsgispy.RSGISPyException: If the upstream graph is invalid.

    .. code:: python

        from rsgispy.imageprocess import RasterSource, infer_node_meta

        print(infer_node_meta(RasterSource("dem.tif")).nodata)
    """
    node_meta: Dict[TileNode, NodeMeta] = {}
    _derive_metadata(_topo_sort([node]), node_meta, {})
    return node_meta[node]


def resolve_input_nodata(
    source: TileNode, no_data_val: Optional[float] = None
) -> Optional[float]:
    """
    Resolve the input no-data value of a node builder: ``no_data_val`` when
    given, otherwise the no-data value of ``source``'s metadata
    (:func:`infer_node_meta`; ``None`` when it has none or it cannot be
    derived, in which case building the :class:`TileGraph` reports the
    problem). This is the no-data convention of every ``*_node`` builder.

    :param source: Input node of the builder.
    :param no_data_val: Explicit no-data value, or ``None``.
    :return: The no-data value to use, or ``None`` (no no-data handling).
    """
    if no_data_val is not None:
        return no_data_val
    try:
        return infer_node_meta(source).nodata
    except Exception:  # pylint: disable=broad-exception-caught
        return None


class TileGraph:
    """
    Directed acyclic graph of raster tile operations executed tile-by-tile
    in memory with automatic cumulative halo propagation.

    When the graph is built it is checked: the sources must be on the same
    pixel grid (dimensions, transform and CRS), and every node's metadata
    (band count, band names, dtype, no data) is derived from its inputs, so
    errors such as incompatible :class:`BandMathOp` band counts or a band
    index out of range are reported before any pixel is read. Use
    :meth:`describe` to print the graph.

    :param sinks: Single sink (e.g. :class:`DiskSink` or :class:`MemorySink`)
                  or sequence of sinks defining graph outputs.
    :param block_size: Core processing tile size in pixels (Default = ``None``,
                       the package default block size
                       :func:`rsgispy.imageprocess.get_default_block_size`,
                       which is 256 unless changed; up to rsgispy 0.3 the
                       default was a fixed 256, independent of the package
                       default).

    .. code:: python

        from rsgispy.imageprocess import BandMathOp, DiskSink, RasterSource, TileGraph

        red = RasterSource("s2.tif", bands=3)
        nir = RasterSource("s2.tif", bands=8)
        ndvi = BandMathOp({"red": red, "nir": nir}, "(nir - red) / (nir + red)")
        graph = TileGraph(DiskSink(ndvi, "ndvi.tif", band_names=["NDVI"]))
        print(graph.describe())
        graph.run()
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        sinks: Union[TileSink, Sequence[TileSink]],
        block_size: Optional[int] = None,
    ) -> None:
        if isinstance(sinks, TileNode):
            sink_list: List[TileSink] = [sinks]  # type: ignore[list-item]
        else:
            sink_list = list(sinks)
        if not sink_list:
            raise rsgispy.RSGISPyException(
                "TileGraph requires at least one DiskSink or MemorySink."
            )
        for sink in sink_list:
            if not isinstance(sink, TileSink):
                raise rsgispy.RSGISPyException(
                    "TileGraph sinks must be TileSink objects (e.g. DiskSink or "
                    f"MemorySink), got {type(sink).__name__}."
                )
        if block_size is None:
            # pylint: disable=import-outside-toplevel
            from rsgispy.imageprocess import get_default_block_size

            block_size = get_default_block_size()
        if block_size <= 0:
            raise rsgispy.RSGISPyException(
                f"block_size must be > 0, got {block_size}."
            )
        self.sinks: Tuple[TileSink, ...] = tuple(sink_list)
        self.block_size: int = int(block_size)
        self.topo_order: Tuple[TileNode, ...] = self._build_topo_order()
        self.required_halos: Dict[TileNode, Tuple[int, int]] = (
            self._compute_required_halos()
        )
        self.consumer_counts: Dict[TileNode, int] = self._compute_consumer_counts()
        self.sources: Tuple[TileNode, ...] = tuple(
            n for n in self.topo_order if n.is_source
        )
        if not self.sources:
            raise rsgispy.RSGISPyException(
                "TileGraph must contain at least one RasterSource or ArraySource."
            )
        self.source_grids: Dict[TileNode, SourceGrid] = {}
        self.node_meta: Dict[TileNode, NodeMeta] = {}
        self._infer_metadata()
        self.grid: SourceGrid = check_source_grids(
            [self.source_grids[s] for s in self.sources]
        )

    def _build_topo_order(self) -> Tuple[TileNode, ...]:
        """Topologically sort all reachable nodes and reject cycles."""
        return _topo_sort(self.sinks)

    def _compute_required_halos(self) -> Dict[TileNode, Tuple[int, int]]:
        """Propagate cumulative halo radii backward from sinks to sources."""
        req: Dict[TileNode, Tuple[int, int]] = {
            node: (0, 0) for node in self.topo_order
        }
        for node in reversed(self.topo_order):
            hy, hx = req[node]
            in_halos = node.input_halos(hy, hx)
            if len(in_halos) != len(node.inputs):
                raise rsgispy.RSGISPyException(
                    f"{node.label}.input_halos() returned {len(in_halos)} "
                    f"halo(s) for {len(node.inputs)} input(s)."
                )
            for parent, (phy_need, phx_need) in zip(node.inputs, in_halos):
                phy, phx = req[parent]
                req[parent] = (max(phy, int(phy_need)), max(phx, int(phx_need)))
        return req

    def _compute_consumer_counts(self) -> Dict[TileNode, int]:
        """Count downstream consumers per node for eager per-tile tensor freeing."""
        counts: Dict[TileNode, int] = {node: 0 for node in self.topo_order}
        for node in self.topo_order:
            for parent in node.inputs:
                counts[parent] += 1
        return counts

    def _infer_metadata(self) -> None:
        """Read source grids and derive (and validate) every node's metadata."""
        _derive_metadata(self.topo_order, self.node_meta, self.source_grids)

    def describe(self) -> str:
        """
        Return a text description of the graph: the grid and block size, and
        for each node (in evaluation order) its inputs, parameters, the halo
        its output is computed with, its own halo radius and its metadata.

        :return: Multi-line string (print it).

        .. code:: python

            print(graph.describe())
        """
        index = {node: idx for idx, node in enumerate(self.topo_order)}
        grid = self.grid
        lines = [
            f"TileGraph: {len(self.topo_order)} node(s), {len(self.sinks)} sink(s), "
            f"block_size={self.block_size}",
            f"  grid: {grid.height} rows x {grid.width} columns, crs={grid.crs}, "
            f"transform={None if grid.transform is None else tuple(grid.transform)[:6]}",
        ]
        for node in self.topo_order:
            inputs = (
                " <- " + ", ".join(f"[{index[p]}]" for p in node.inputs)
                if node.inputs
                else ""
            )
            params = node.describe_params()
            lines.append(
                f"  [{index[node]}] {node.label}{inputs}"
                + (f" ({params})" if params else "")
            )
            hy, hx = self.required_halos[node]
            ry, rx = node.halo_radius
            lines.append(
                f"      halo=({hy}, {hx}) radius=({ry}, {rx}) "
                f"{self.node_meta[node].describe()}"
            )
        return "\n".join(lines)

    def _start_states(self, resources: RunResources) -> Dict[TileNode, Any]:
        """Per-run state of every (non-sink) node, using ``resources``."""
        states: Dict[TileNode, Any] = {}
        for node in self.topo_order:
            if not isinstance(node, TileSink):
                hy, hx = self.required_halos[node]
                states[node] = node.start_run(resources, hy, hx)
        return states

    def _run_grid(self, states: Dict[TileNode, Any]) -> SourceGrid:
        """Re-check the grids of the files actually opened; return the reference."""
        run_grids = []
        for src in self.sources:
            grid = self.source_grids[src]
            dset = getattr(states.get(src), "dataset", None)
            if dset is not None:
                grid = SourceGrid(
                    height=int(dset.height),
                    width=int(dset.width),
                    transform=dset.transform,
                    crs=dset.crs,
                    nodata=grid.nodata,
                    label=grid.label,
                )
            run_grids.append(grid)
        return check_source_grids(run_grids)

    def _eval_tile(
        self, win: Any, states: Dict[TileNode, Any], ref: SourceGrid
    ) -> List[Tuple[TileSink, np.ndarray]]:
        """
        Evaluate every node for the tile ``win`` and return the core block of
        each sink (in sink order), without writing them.
        """
        img_h, img_w = int(ref.height), int(ref.width)
        base_ctx = TileContext(
            row_off=int(win.row_off),
            col_off=int(win.col_off),
            height=int(win.height),
            width=int(win.width),
            img_height=img_h,
            img_width=img_w,
            halo_y=0,
            halo_x=0,
            oob_top=0,
            oob_bottom=0,
            oob_left=0,
            oob_right=0,
            transform=ref.transform,
            crs=ref.crs,
            nodata=ref.nodata,
        )

        tensors: Dict[TileNode, TileTensor] = {}
        remaining_uses = dict(self.consumer_counts)
        blocks: List[Tuple[TileSink, np.ndarray]] = []
        for node in self.topo_order:
            if isinstance(node, TileSink):
                blocks.append((node, _crop_tensor(tensors[node.inputs[0]], 0, 0).data))
            else:
                hy, hx = self.required_halos[node]
                tensors[node] = node.evaluate(
                    [tensors[p] for p in node.inputs],
                    hy,
                    hx,
                    base_ctx,
                    states[node],
                )

            for parent in node.inputs:
                remaining_uses[parent] -= 1
                if remaining_uses[parent] <= 0:
                    tensors.pop(parent, None)
        return blocks

    @staticmethod
    def _write_blocks(blocks: List[Tuple[TileSink, np.ndarray]], win: Any) -> None:
        for sink, block in blocks:
            sink.write(block, win)

    def _run_parallel(
        self,
        windows: List[Any],
        bundles: List[Dict[TileNode, Any]],
        ref: SourceGrid,
        use_pbar: bool,
    ) -> None:
        """
        Evaluate tiles on ``len(bundles)`` threads (each thread takes a free
        bundle of per-run states, so no two threads share a dataset handle
        or parser) and write the blocks in tile order from this thread.
        """
        # pylint: disable=too-many-locals
        free: "queue.SimpleQueue[Dict[TileNode, Any]]" = queue.SimpleQueue()
        for bundle in bundles:
            free.put(bundle)

        def _task(win: Any) -> List[Tuple[TileSink, np.ndarray]]:
            states = free.get()
            try:
                return self._eval_tile(win, states, ref)
            finally:
                free.put(states)

        n_workers = len(bundles)
        max_pending = 2 * n_workers
        pbar = rsgispy.get_tqdm()(total=len(windows)) if use_pbar else None
        pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="tilegraph")
        pending: Deque[Tuple[Any, Future]] = deque()
        try:
            for win in windows:
                pending.append((win, pool.submit(_task, win)))
                if len(pending) >= max_pending:
                    done_win, fut = pending.popleft()
                    self._write_blocks(fut.result(), done_win)
                    if pbar is not None:
                        pbar.update(1)
            while pending:
                done_win, fut = pending.popleft()
                self._write_blocks(fut.result(), done_win)
                if pbar is not None:
                    pbar.update(1)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)
            if pbar is not None:
                pbar.close()

    def run(self, use_pbar: bool = True, n_jobs: Optional[int] = 1) -> Dict[str, Any]:
        """
        Execute the tile graph across all tiles, writing every :class:`DiskSink`
        and populating every :class:`MemorySink`.

        With ``n_jobs > 1`` the tiles are evaluated on a pool of threads (NumPy,
        GDAL reads and the numba kernels release the GIL). Every thread has its
        own per-run node states (its own open datasets, shared readers and
        parsers) and the blocks are written to the sinks from the calling
        thread in tile order, so the results are identical to ``n_jobs=1``.
        Custom nodes must keep per-run state in the object returned by
        :meth:`TileNode.start_run`, not on the node, to be run in parallel.

        :param use_pbar: If ``True`` (default), display a ``tqdm`` progress bar
                         over output tiles.
        :param n_jobs: Number of threads evaluating tiles (Default = ``1``, in
                       the calling thread; ``None`` or ``-1``: the number of
                       CPUs).
        :return: Dictionary mapping each :class:`MemorySink`'s ``name`` (or
                 ``'memory_<i>'`` when unnamed) to its assembled 3D
                 ``(C, H, W)`` NumPy array (and the results of any other sink
                 with ``returns_result = True``, e.g. :class:`StatsSink`).
        """
        # pylint: disable=import-outside-toplevel,too-many-locals
        from rsgispy.imageprocess import IntersectionGrid, iter_dataset_windows

        n_workers = _resolve_n_jobs(n_jobs)
        with ExitStack() as stack:
            bundles = [
                self._start_states(RunResources(stack)) for _ in range(n_workers)
            ]
            ref = self._run_grid(bundles[0])
            img_h, img_w = int(ref.height), int(ref.width)
            out_grid = IntersectionGrid(
                crs=ref.crs,
                transform=ref.transform if ref.transform is not None else Affine.identity(),
                width=img_w,
                height=img_h,
                offsets={},
            )
            for sink in self.sinks:
                sink.open(
                    out_grid,
                    self.node_meta[sink.inputs[0]],
                    stack=stack,
                    use_pbar=use_pbar,
                )

            if n_workers == 1:
                for win in iter_dataset_windows(
                    img_h, img_w, block_size=self.block_size, use_pbar=use_pbar
                ):
                    self._write_blocks(self._eval_tile(win, bundles[0], ref), win)
            else:
                windows = list(
                    iter_dataset_windows(
                        img_h, img_w, block_size=self.block_size, use_pbar=False
                    )
                )
                self._run_parallel(windows, bundles, ref, use_pbar)

            for sink in self.sinks:
                sink.close()

        results: Dict[str, Any] = {}
        prefix_counts: Dict[str, int] = {}
        for sink in self.sinks:
            if sink.returns_result:
                idx = prefix_counts.get(sink.result_prefix, 0)
                key = sink.name if sink.name is not None else f"{sink.result_prefix}_{idx}"
                results[key] = sink.result
                prefix_counts[sink.result_prefix] = idx + 1
        return results


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Number of tile threads for ``n_jobs`` (``None`` / ``-1``: CPU count)."""
    if n_jobs is None or (isinstance(n_jobs, (int, np.integer)) and int(n_jobs) == -1):
        return max(1, os.cpu_count() or 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise rsgispy.RSGISPyException(
            f"n_jobs must be a positive integer, -1 or None, got {n_jobs!r}."
        )
    return int(n_jobs)


def run_tile_graph(
    sinks: Union[TileSink, Sequence[TileSink]],
    block_size: Optional[int] = None,
    use_pbar: bool = True,
    n_jobs: Optional[int] = 1,
) -> Dict[str, Any]:
    """
    Construct and immediately execute a :class:`TileGraph`.

    :param sinks: Single :class:`DiskSink` / :class:`MemorySink` or sequence of
                  sinks.
    :param block_size: Core tile size in pixels (Default = ``None``, the
                       package default block size).
    :param use_pbar: Display a ``tqdm`` progress bar (Default = ``True``).
    :param n_jobs: Number of threads evaluating tiles (Default = ``1``; see
                   :meth:`TileGraph.run`).
    :return: Dictionary of :class:`MemorySink` outputs (empty when only
             :class:`DiskSink` instances are used).
    """
    return TileGraph(sinks=sinks, block_size=block_size).run(
        use_pbar=use_pbar, n_jobs=n_jobs
    )
