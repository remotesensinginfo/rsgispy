"""
Composable in-memory, halo-fused tile-graph raster execution engine for
:mod:`rsgispy`.

This module allows multi-stage raster workflows — including chains of
moving-window spatial filters, morphological operators, skip connections, and
point-wise / ``pymu-eval`` band-math expressions — to execute per tile in
memory before writing final output tile(s) once to disk.

Key guarantees
--------------
* **Cumulative halo propagation**: reverse topological traversal computes the
  exact cumulative halo radius ``(sum_ry, sum_rx)`` required at every upstream
  node so interior pixels of every stage are unaffected by tile boundaries.
* **Bit-exact image-border semantics**: each intermediate tensor tracks its
  exact out-of-image border overhang ``(oob_top, oob_bottom, oob_left,
  oob_right)`` and re-pads out-of-bounds margins using each :class:`WindowOp`'s
  configured :class:`HaloBorder` before evaluation, matching multi-pass disk
  execution even at image edges.
* **Skip connections & multi-sink fan-out**: nodes with multiple consumers
  compute the maximum required halo across consumers; downstream
  :class:`PointwiseOp`, :class:`BandMathOp`, and sinks crop mismatched halos
  symmetrically to align pixel-for-pixel.
* **Low peak memory**: intermediate tile tensors are released via per-tile
  reference counting as soon as their last consumer finishes.
"""

from rsgispy._tilegraph import (
    ArraySource,
    BandMathOp,
    Cast,
    DiskSink,
    HaloBorder,
    MemorySink,
    NodeMeta,
    PointwiseOp,
    RasterSource,
    RasterStats,
    SelectBands,
    SetNoData,
    SourceGrid,
    StackBands,
    StatsSink,
    TileContext,
    TileGraph,
    TileNode,
    TileSink,
    TileTensor,
    WindowOp,
    ZonalStatsSink,
    infer_node_meta,
    resolve_input_nodata,
    run_tile_graph,
    same_bands_meta,
)

__all__ = [
    "HaloBorder",
    "TileContext",
    "TileNode",
    "RasterSource",
    "ArraySource",
    "WindowOp",
    "PointwiseOp",
    "BandMathOp",
    "DiskSink",
    "MemorySink",
    "StatsSink",
    "ZonalStatsSink",
    "RasterStats",
    "TileGraph",
    "run_tile_graph",
    "TileSink",
    "TileTensor",
    "NodeMeta",
    "SourceGrid",
    "SelectBands",
    "StackBands",
    "Cast",
    "SetNoData",
    "infer_node_meta",
    "resolve_input_nodata",
    "same_bands_meta",
]
