"""
Internal composable tile-graph raster execution package for :mod:`rsgispy`.
"""

from rsgispy._tilegraph._graph import (
    TileGraph,
    infer_node_meta,
    resolve_input_nodata,
    run_tile_graph,
)
from rsgispy._tilegraph._nodes import (
    ArraySource,
    BandMathOp,
    Cast,
    DiskSink,
    HaloBorder,
    MemorySink,
    NodeMeta,
    PointwiseOp,
    RasterSource,
    SelectBands,
    SetBandNames,
    SetNoData,
    SourceGrid,
    StackBands,
    TileContext,
    TileNode,
    TileSink,
    TileTensor,
    WindowOp,
    datatype_from_meta,
    same_bands_meta,
)
from rsgispy._tilegraph._sample import SampleSink
from rsgispy._tilegraph._stats import RasterStats, StatsSink, ZonalStatsSink
from rsgispy._tilegraph._tensor import halo_border_from_mode

__all__ = [
    "HaloBorder",
    "halo_border_from_mode",
    "TileContext",
    "TileNode",
    "TileSink",
    "TileTensor",
    "NodeMeta",
    "SourceGrid",
    "same_bands_meta",
    "RasterSource",
    "ArraySource",
    "WindowOp",
    "PointwiseOp",
    "BandMathOp",
    "SelectBands",
    "StackBands",
    "Cast",
    "SetNoData",
    "SetBandNames",
    "datatype_from_meta",
    "infer_node_meta",
    "resolve_input_nodata",
    "DiskSink",
    "MemorySink",
    "StatsSink",
    "ZonalStatsSink",
    "SampleSink",
    "RasterStats",
    "TileGraph",
    "run_tile_graph",
]
