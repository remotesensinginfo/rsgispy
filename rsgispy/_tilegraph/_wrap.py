"""
Helpers for file-based functions implemented as thin tile-graph wrappers
(``source(s) -> operation node -> sink(s) -> TileGraph.run``):

* :class:`GridRasterSource` reads an input on the spatial intersection grid
  of several images (:func:`rsgispy.imageprocess.compute_spatial_intersection`),
  i.e. a fixed pixel offset into the file, as the file-based functions do;
* :class:`WrapperDiskSink` is a :class:`DiskSink` with the output options
  the file-based functions need: thematic statistics, per-band no-data
  values and actions run on the closed output (before a COG is created);
* :func:`run_wrapper_graph` runs the graph with the package default block
  size, so the tiles match those of the former block loops.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._exec import _finish_read, halo_read_bounds
from rsgispy._tilegraph._nodes import (
    DiskSink,
    HeaderCache,
    RunResources,
    TileNode,
    TileSink,
    _ctx_window,
)
from rsgispy._tilegraph._tensor import (
    HaloBorder,
    NodeMeta,
    SourceGrid,
    TileContext,
    TileTensor,
)


class _GridReaderState:
    """Open dataset of a :class:`GridRasterSource` during a run."""

    # pylint: disable=too-few-public-methods

    def __init__(self, ds: rasterio.io.DatasetReader) -> None:
        # Not called ``dataset``: the graph must not re-check this source
        # against the full file extent.
        self.reader = ds


class GridRasterSource(TileNode):
    """
    Source node reading a raster on a given output grid: the image is the
    ``grid.height`` x ``grid.width`` window of the file starting at
    ``offset`` (column, row), and the node reports the grid's transform and
    CRS. Used by wrappers of functions that process the spatial intersection
    of several images; the grids were checked by
    :func:`~rsgispy.imageprocess.compute_spatial_intersection`. Out-of-window
    halo pixels are padded with ``border``, as at an image edge.

    :param path: Input raster file path.
    :param grid: Output grid (``height``, ``width``, ``transform``, ``crs``).
    :param offset: ``(col_off, row_off)`` of the grid in the file.
    :param bands: Optional 1-based band index or indices (all when ``None``).
    :param dtype: NumPy dtype of the blocks (Default = ``float64``).
    :param border: Padding outside the window (Default = ``HaloBorder('edge')``).
    :param nodata: Optional no-data override (the file's when ``None``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        path: Union[str, os.PathLike],
        grid: Any,
        offset: Tuple[int, int] = (0, 0),
        bands: Optional[Union[int, Sequence[int]]] = None,
        dtype: Any = np.float64,
        border: Optional[HaloBorder] = None,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(), name=name)
        self.path: str = os.fspath(path)
        self.height = int(grid.height)
        self.width = int(grid.width)
        self.transform = grid.transform
        self.crs = grid.crs
        self.col_off = int(offset[0])
        self.row_off = int(offset[1])
        if bands is None:
            self.bands: Optional[List[int]] = None
        elif isinstance(bands, (int, np.integer)):
            self.bands = [int(bands)]
        else:
            self.bands = [int(b) for b in bands]
        self.dtype = np.dtype(dtype)
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.nodata = nodata

    def source_info(self, headers: HeaderCache) -> Tuple[SourceGrid, NodeMeta]:
        hdr = headers.get(self.path)
        bands = self.bands if self.bands is not None else list(range(1, hdr.count + 1))
        for band in bands:
            if band < 1 or band > hdr.count:
                raise rsgispy.RSGISPyException(
                    f"Band {band} requested from '{self.path}', which has "
                    f"{hdr.count} band(s) (bands are 1-based)."
                )
        descs = [hdr.descriptions[b - 1] for b in bands]
        names = (
            tuple(d if d else f"Band {b}" for d, b in zip(descs, bands))
            if any(descs)
            else None
        )
        nodata = self.nodata if self.nodata is not None else hdr.nodata
        grid = SourceGrid(
            height=self.height,
            width=self.width,
            transform=self.transform,
            crs=self.crs,
            nodata=nodata,
            label=f"'{self.path}'",
        )
        meta = NodeMeta(
            n_bands=len(bands), band_names=names, dtype=self.dtype, nodata=nodata
        )
        return grid, meta

    def start_run(
        self, resources: RunResources, halo_y: int, halo_x: int
    ) -> _GridReaderState:
        del halo_y, halo_x
        ds = resources.stack.enter_context(rasterio.open(self.path, "r"))
        return _GridReaderState(ds)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        bounds = halo_read_bounds(
            _ctx_window(ctx), self.height, self.width, halo_y, halo_x
        )
        read_win = Window(
            self.col_off + bounds[1],
            self.row_off + bounds[0],
            bounds[3] - bounds[1],
            bounds[2] - bounds[0],
        )
        ds = state.reader
        # GDAL converts to the block dtype while reading (no extra copy).
        if self.bands:
            raw = ds.read(self.bands, window=read_win, out_dtype=self.dtype)
        else:
            raw = ds.read(window=read_win, out_dtype=self.dtype)
        return _finish_read(raw, self.dtype, self.border, halo_y, halo_x, bounds[4:])

    def describe_params(self) -> str:
        bands = "all" if self.bands is None else str(self.bands)
        return f"path='{self.path}' bands={bands} offset=({self.col_off}, {self.row_off})"


class WrapperDiskSink(DiskSink):
    """
    :class:`DiskSink` for the wrappers of file-based functions, adding:

    :param thematic: Calculate thematic statistics and a colour table (and
                     use ``NEAREST`` COG overviews) instead of continuous
                     statistics (Default = ``False``).
    :param per_band_nodata: Optional per-band no-data values (for formats
                            such as KEA that store one per band); overrides
                            ``nodata``.
    :param close_actions: Callables ``action(path)`` run on the closed output
                          (before the statistics and before a COG is
                          created), e.g. :func:`set_img_thematic`.

    The other parameters are those of :class:`DiskSink`; ``inherit_meta`` is
    ``False`` (the wrappers give the no-data value and band names).
    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        source: TileNode,
        output_img: Union[str, os.PathLike],
        *,
        thematic: bool = False,
        per_band_nodata: Optional[Sequence[float]] = None,
        close_actions: Sequence[Callable[[str], None]] = (),
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("inherit_meta", False)
        super().__init__(source, output_img, **kwargs)
        self.thematic = bool(thematic)
        self.per_band_nodata = (
            [float(v) for v in per_band_nodata] if per_band_nodata is not None else None
        )
        self.close_actions = tuple(close_actions)

    def _open_dataset(self, count: int) -> rasterio.io.DatasetWriter:
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import (
            normalise_creation_options,
            on_output_close,
            open_output_dataset,
            pop_stats_on_close,
        )

        run = self._run
        if run is None:
            raise rsgispy.RSGISPyException("DiskSink.write() called before open().")
        dst = open_output_dataset(
            stack=run["stack"],
            output_img=self.output_img,
            gdalformat=self.gdalformat,
            out_rio_dtype=rsgispy.get_rasterio_datatype(self.out_datatype),
            grid=run["grid"],
            count=int(count),
            output_exists=False,
            creation_opts=normalise_creation_options(self.creation_opts),
        )
        if self.per_band_nodata is not None:
            dst._set_nodatavals(self.per_band_nodata)  # pylint: disable=protected-access
        elif run["nodata"] is not None:
            dst.nodata = run["nodata"]
        names = run["band_names"]
        if names is not None:
            for b_idx, b_name in enumerate(names, start=1):
                if b_name is not None:  # None: leave the band unnamed
                    dst.set_band_description(b_idx, str(b_name))
        for action in self.close_actions:
            on_output_close(dst, action)
        pop_stats_on_close(
            dst,
            self.calc_stats,
            no_data_val=run["nodata"],
            thematic=self.thematic,
            use_pbar=run["use_pbar"],
        )
        run["dst"] = dst
        return dst


def run_wrapper_graph(
    sinks: Union[TileSink, Sequence[TileSink]], use_pbar: bool = True
) -> None:
    """
    Run the graph of a file-based wrapper with the package default block size
    (:func:`rsgispy.imageprocess.get_default_block_size`).

    :param sinks: Sink or sinks of the graph.
    :param use_pbar: Show a progress bar over the tiles.
    """
    # pylint: disable=import-outside-toplevel
    from rsgispy._tilegraph._graph import TileGraph

    TileGraph(sinks, block_size=None).run(use_pbar=use_pbar)
