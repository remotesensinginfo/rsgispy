"""
Tile-graph DAG planning, halo propagation, and streaming execution for
:mod:`rsgispy._tilegraph`.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymu_eval
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._exec import (
    eval_band_math_op,
    eval_pointwise_op,
    eval_window_op,
    read_array_source,
    read_raster_source,
)
from rsgispy._tilegraph._nodes import (
    ArraySource,
    BandMathOp,
    DiskSink,
    MemorySink,
    PointwiseOp,
    RasterSource,
    TileContext,
    TileNode,
    WindowOp,
    _TileTensor,
    _crop_tensor,
)


class TileGraph:
    """
    Directed acyclic graph of raster tile operations executed tile-by-tile
    in memory with automatic cumulative halo propagation.

    :param sinks: Single sink (:class:`DiskSink` or :class:`MemorySink`) or
                  sequence of sinks defining graph outputs.
    :param block_size: Core processing tile size in pixels (Default = ``256``).
    """

    def __init__(
        self,
        sinks: Union[DiskSink, MemorySink, Sequence[Union[DiskSink, MemorySink]]],
        block_size: int = 256,
    ) -> None:
        if isinstance(sinks, (DiskSink, MemorySink)):
            sink_list: List[Union[DiskSink, MemorySink]] = [sinks]
        else:
            sink_list = list(sinks)
        if not sink_list:
            raise rsgispy.RSGISPyException(
                "TileGraph requires at least one DiskSink or MemorySink."
            )
        for sink in sink_list:
            if not isinstance(sink, (DiskSink, MemorySink)):
                raise rsgispy.RSGISPyException(
                    f"TileGraph sinks must be DiskSink or MemorySink, got {type(sink).__name__}."
                )
        if block_size <= 0:
            raise rsgispy.RSGISPyException(
                f"block_size must be > 0, got {block_size}."
            )
        self.sinks: Tuple[Union[DiskSink, MemorySink], ...] = tuple(sink_list)
        self.block_size: int = int(block_size)
        self.topo_order: Tuple[TileNode, ...] = self._build_topo_order()
        self.required_halos: Dict[TileNode, Tuple[int, int]] = (
            self._compute_required_halos()
        )
        self.consumer_counts: Dict[TileNode, int] = self._compute_consumer_counts()

    def _build_topo_order(self) -> Tuple[TileNode, ...]:
        """Topologically sort all reachable nodes and reject cycles."""
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
                _dfs(parent)
            visited[node] = 2
            order.append(node)

        for sink in self.sinks:
            _dfs(sink)
        return tuple(order)

    def _compute_required_halos(self) -> Dict[TileNode, Tuple[int, int]]:
        """Propagate cumulative halo radii backward from sinks to sources."""
        req: Dict[TileNode, Tuple[int, int]] = {
            node: (0, 0) for node in self.topo_order
        }
        for node in reversed(self.topo_order):
            hy, hx = req[node]
            if isinstance(node, WindowOp):
                parent = node.inputs[0]
                phy, phx = req[parent]
                req[parent] = (
                    max(phy, hy + node.radius_y),
                    max(phx, hx + node.radius_x),
                )
            elif isinstance(node, (PointwiseOp, BandMathOp, DiskSink, MemorySink)):
                for parent in node.inputs:
                    phy, phx = req[parent]
                    req[parent] = (max(phy, hy), max(phx, hx))
        return req

    def _compute_consumer_counts(self) -> Dict[TileNode, int]:
        """Count downstream consumers per node for eager per-tile tensor freeing."""
        counts: Dict[TileNode, int] = {node: 0 for node in self.topo_order}
        for node in self.topo_order:
            for parent in node.inputs:
                counts[parent] += 1
        return counts

    @staticmethod
    def _read_raster_source(
        node: RasterSource,
        ds: rasterio.io.DatasetReader,
        win: Window,
        hy: int,
        hx: int,
    ) -> _TileTensor:
        return read_raster_source(node, ds, win, hy, hx)

    @staticmethod
    def _read_array_source(
        node: ArraySource,
        win: Window,
        hy: int,
        hx: int,
    ) -> _TileTensor:
        return read_array_source(node, win, hy, hx)

    @staticmethod
    def _eval_window_op(
        node: WindowOp,
        in_tensor: _TileTensor,
        out_hy: int,
        out_hx: int,
        base_ctx: TileContext,
    ) -> _TileTensor:
        return eval_window_op(node, in_tensor, out_hy, out_hx, base_ctx)

    @staticmethod
    def _eval_pointwise_op(
        node: PointwiseOp,
        in_tensors: Sequence[_TileTensor],
        out_hy: int,
        out_hx: int,
        base_ctx: TileContext,
    ) -> _TileTensor:
        return eval_pointwise_op(node, in_tensors, out_hy, out_hx, base_ctx)

    @staticmethod
    def _eval_band_math_op(
        node: BandMathOp,
        in_tensors: Sequence[_TileTensor],
        out_hy: int,
        out_hx: int,
        parsers: Dict[BandMathOp, pymu_eval.Parser],
    ) -> _TileTensor:
        return eval_band_math_op(node, in_tensors, out_hy, out_hx, parsers)

    def run(self, use_pbar: bool = True) -> Dict[str, np.ndarray]:
        """
        Execute the tile graph across all tiles, writing every :class:`DiskSink`
        and populating every :class:`MemorySink`.

        :param use_pbar: If ``True`` (default), display a ``tqdm`` progress bar
                         over output tiles.
        :return: Dictionary mapping each :class:`MemorySink`'s ``name`` (or
                 ``'memory_<i>'`` when unnamed) to its assembled 3D
                 ``(C, H, W)`` NumPy array.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import (
            IntersectionGrid,
            cast_result_block,
            ensure_file_exists,
            iter_dataset_windows,
            open_output_dataset,
        )
        from rsgispy.imageprocess import normalise_creation_options

        sources = [
            n for n in self.topo_order if isinstance(n, (RasterSource, ArraySource))
        ]
        if not sources:
            raise rsgispy.RSGISPyException(
                "TileGraph must contain at least one RasterSource or ArraySource."
            )

        with ExitStack() as stack:
            open_rasters: Dict[RasterSource, rasterio.io.DatasetReader] = {}
            img_h: Optional[int] = None
            img_w: Optional[int] = None
            ref_transform: Optional[Affine] = None
            ref_crs: Any = None
            ref_nodata: Optional[float] = None

            for src_node in sources:
                if isinstance(src_node, RasterSource):
                    ensure_file_exists(src_node.path)
                    ds = stack.enter_context(rasterio.open(src_node.path, "r"))
                    open_rasters[src_node] = ds
                    cur_h, cur_w = int(ds.height), int(ds.width)
                    cur_transform, cur_crs = ds.transform, ds.crs
                    cur_nd = (
                        src_node.nodata if src_node.nodata is not None else ds.nodata
                    )
                else:
                    cur_h = int(src_node.array.shape[1])
                    cur_w = int(src_node.array.shape[2])
                    cur_transform = src_node.transform
                    cur_crs = src_node.crs
                    cur_nd = src_node.nodata

                if img_h is None:
                    img_h, img_w = cur_h, cur_w
                    ref_transform = cur_transform
                    ref_crs = cur_crs
                    ref_nodata = cur_nd
                elif (cur_h, cur_w) != (img_h, img_w):
                    raise rsgispy.RSGISPyException(
                        f"TileGraph sources must have matching dimensions; "
                        f"got {(cur_h, cur_w)} vs {(img_h, img_w)}."
                    )

            if img_h is None or img_w is None:
                raise rsgispy.RSGISPyException(
                    "TileGraph could not determine image dimensions from sources."
                )
            grid = IntersectionGrid(
                crs=ref_crs,
                transform=(
                    ref_transform if ref_transform is not None else Affine.identity()
                ),
                width=img_w,
                height=img_h,
                offsets={},
            )

            open_sinks: Dict[DiskSink, rasterio.io.DatasetWriter] = {}
            mem_arrays: Dict[MemorySink, np.ndarray] = {}
            band_math_parsers: Dict[BandMathOp, pymu_eval.Parser] = {}

            for win in iter_dataset_windows(
                img_h, img_w, block_size=self.block_size, use_pbar=use_pbar
            ):
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
                    transform=ref_transform,
                    crs=ref_crs,
                    nodata=ref_nodata,
                )

                tensors: Dict[TileNode, _TileTensor] = {}
                remaining_uses = dict(self.consumer_counts)

                for node in self.topo_order:
                    hy, hx = self.required_halos[node]
                    if isinstance(node, RasterSource):
                        tensors[node] = self._read_raster_source(
                            node, open_rasters[node], win, hy, hx
                        )
                    elif isinstance(node, ArraySource):
                        tensors[node] = self._read_array_source(node, win, hy, hx)
                    elif isinstance(node, WindowOp):
                        parent = node.inputs[0]
                        tensors[node] = self._eval_window_op(
                            node, tensors[parent], hy, hx, base_ctx
                        )
                    elif isinstance(node, PointwiseOp):
                        in_t = [tensors[p] for p in node.inputs]
                        tensors[node] = self._eval_pointwise_op(
                            node, in_t, hy, hx, base_ctx
                        )
                    elif isinstance(node, BandMathOp):
                        in_t = [tensors[p] for p in node.inputs]
                        tensors[node] = self._eval_band_math_op(
                            node, in_t, hy, hx, band_math_parsers
                        )
                    elif isinstance(node, DiskSink):
                        core = _crop_tensor(tensors[node.inputs[0]], 0, 0).data
                        if node not in open_sinks:
                            out_rio_dt = rsgispy.get_rasterio_datatype(node.datatype)
                            dst = open_output_dataset(
                                stack=stack,
                                output_img=node.output_img,
                                gdalformat=node.gdalformat,
                                out_rio_dtype=out_rio_dt,
                                grid=grid,
                                count=int(core.shape[0]),
                                output_exists=False,
                                creation_opts=normalise_creation_options(
                                    node.creation_opts
                                ),
                            )
                            if node.nodata is not None:
                                dst.nodata = node.nodata
                            open_sinks[node] = dst
                        out_np_dt = np.dtype(
                            rsgispy.get_numpy_datatype(node.datatype)
                        )
                        cast_blk = cast_result_block(
                            core,
                            int(win.height),
                            int(win.width),
                            out_np_dt,
                            clamp_int=node.clamp_int,
                        )
                        open_sinks[node].write(cast_blk, window=win)
                    elif isinstance(node, MemorySink):
                        core = _crop_tensor(tensors[node.inputs[0]], 0, 0).data
                        out_np_dt = (
                            node.dtype if node.dtype is not None else core.dtype
                        )
                        if node not in mem_arrays:
                            mem_arrays[node] = np.empty(
                                (int(core.shape[0]), img_h, img_w),
                                dtype=out_np_dt,
                            )
                        cast_blk = cast_result_block(
                            core,
                            int(win.height),
                            int(win.width),
                            out_np_dt,
                            clamp_int=node.clamp_int,
                        )
                        r0 = int(win.row_off)
                        c0 = int(win.col_off)
                        r1 = r0 + int(win.height)
                        c1 = c0 + int(win.width)
                        mem_arrays[node][:, r0:r1, c0:c1] = cast_blk

                    for parent in node.inputs:
                        remaining_uses[parent] -= 1
                        if remaining_uses[parent] <= 0:
                            tensors.pop(parent, None)

        # Post-close metadata / statistics for DiskSinks
        for sink in self.sinks:
            if isinstance(sink, DiskSink):
                if sink.band_names is not None:
                    # pylint: disable=import-outside-toplevel
                    from rsgispy.imageutils.metadata import set_band_names

                    set_band_names(
                        sink.output_img, sink.band_names, feedback=False
                    )
                if sink.calc_stats:
                    # pylint: disable=import-outside-toplevel
                    from rsgispy.imageutils.stats import pop_img_stats

                    pop_img_stats(
                        sink.output_img,
                        no_data_val=sink.nodata,
                        calc_pyramids=True,
                        use_pbar=use_pbar,
                    )

        mem_results: Dict[str, np.ndarray] = {}
        mem_idx = 0
        for sink in self.sinks:
            if isinstance(sink, MemorySink):
                arr_out = mem_arrays[sink]
                sink.result = arr_out
                key = sink.name if sink.name is not None else f"memory_{mem_idx}"
                mem_results[key] = arr_out
                mem_idx += 1
        return mem_results


def run_tile_graph(
    sinks: Union[DiskSink, MemorySink, Sequence[Union[DiskSink, MemorySink]]],
    block_size: int = 256,
    use_pbar: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Construct and immediately execute a :class:`TileGraph`.

    :param sinks: Single :class:`DiskSink` / :class:`MemorySink` or sequence of
                  sinks.
    :param block_size: Core tile size in pixels (Default = ``256``).
    :param use_pbar: Display a ``tqdm`` progress bar (Default = ``True``).
    :return: Dictionary of :class:`MemorySink` outputs (empty when only
             :class:`DiskSink`s are used).
    """
    return TileGraph(sinks=sinks, block_size=block_size).run(use_pbar=use_pbar)
