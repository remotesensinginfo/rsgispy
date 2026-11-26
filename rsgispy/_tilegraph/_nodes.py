# pylint: disable=too-many-lines
"""
Tile-graph node definitions for :mod:`rsgispy._tilegraph`.

Every node implements the same small protocol (see :class:`TileNode`), so
:class:`~rsgispy.imageprocess.TileGraph` evaluates any node type, including
user-defined ones, without knowing about it:

* :attr:`TileNode.halo_radius` — the halo the node needs around its inputs;
* :meth:`TileNode.input_halos` — the halo each input must provide;
* :meth:`TileNode.infer_meta` — output metadata (bands, band names, dtype,
  no data) from the inputs' metadata, raising on invalid combinations;
* :meth:`TileNode.start_run` — per-run state (open files, parsers, ...);
* :meth:`TileNode.evaluate` — the output block of one tile.

Source nodes (no inputs) also implement :meth:`TileNode.source_info`.
Sinks derive from :class:`TileSink` and implement ``open``, ``write`` and
``close``.
"""

from __future__ import annotations

import math
import os
from contextlib import ExitStack
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pymu_eval
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

import rsgispy
from rsgispy._tilegraph._exec import (
    SharedRasterReader,
    eval_band_math_op,
    eval_pointwise_op,
    eval_window_op,
    read_array_source,
)
# pylint: disable-next=unused-import
from rsgispy._tilegraph._tensor import (  # noqa: F401 (re-exported)
    HaloBorder,
    NodeMeta,
    SourceGrid,
    TileContext,
    TileTensor,
    _TileTensor,
    _crop_tensor,
    _ensure_3d,
    _normalise_radius,
    _pad_3d,
    _repad_oob,
    same_bands_meta,
)

MetaFunc = Callable[[Sequence[NodeMeta]], NodeMeta]


def _path_key(path: str) -> str:
    """Key identifying a raster file, so nodes on the same file share a reader."""
    if path.startswith("/vsi") or "://" in path:
        return path
    return os.path.realpath(path)


class RasterHeader:
    """Header of a raster file read once while a :class:`TileGraph` is built."""

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(self, path: str, ds: rasterio.io.DatasetReader) -> None:
        self.path = path
        self.count = int(ds.count)
        self.height = int(ds.height)
        self.width = int(ds.width)
        self.transform: Affine = ds.transform
        self.crs = ds.crs
        self.nodata: Optional[float] = ds.nodata
        self.dtypes: Tuple[str, ...] = tuple(ds.dtypes)
        self.descriptions: Tuple[Optional[str], ...] = tuple(ds.descriptions)


class HeaderCache:
    """Opens each raster file once to read its header while a graph is built."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        self._headers: Dict[str, RasterHeader] = {}

    def get(self, path: str) -> RasterHeader:
        """Return the (cached) header of ``path``."""
        key = _path_key(path)
        if key not in self._headers:
            # pylint: disable=import-outside-toplevel
            from rsgispy.imageprocess import ensure_file_exists

            ensure_file_exists(path)
            with rasterio.open(path, "r") as ds:
                self._headers[key] = RasterHeader(path, ds)
        return self._headers[key]


class RunResources:
    """
    Resources shared by the nodes of one :meth:`TileGraph.run`: the
    ``ExitStack`` that closes everything at the end of the run, and one
    :class:`SharedRasterReader` per raster file.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, stack: ExitStack) -> None:
        self.stack = stack
        self._readers: Dict[str, SharedRasterReader] = {}

    def raster_reader(self, path: str) -> SharedRasterReader:
        """Return the reader of ``path``, opening the file on first use."""
        key = _path_key(path)
        if key not in self._readers:
            # pylint: disable=import-outside-toplevel
            from rsgispy.imageprocess import ensure_file_exists

            ensure_file_exists(path)
            ds = self.stack.enter_context(rasterio.open(path, "r"))
            self._readers[key] = SharedRasterReader(ds)
        return self._readers[key]


def _ctx_window(ctx: TileContext) -> Window:
    """Core tile window of ``ctx``."""
    return Window(ctx.col_off, ctx.row_off, ctx.width, ctx.height)


class TileNode:
    """
    Base class for nodes in a :class:`~rsgispy.imageprocess.TileGraph`.

    A new node type subclasses :class:`TileNode` and overrides the methods
    below; the graph needs no changes. An operation node usually overrides
    :meth:`evaluate`, :meth:`infer_meta` and, for a moving window,
    :attr:`halo_radius`:

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import (
            ArraySource, MemorySink, TileGraph, TileNode,
        )

        class Scale(TileNode):
            def __init__(self, source, factor, name=None):
                super().__init__(inputs=(source,), name=name)
                self.factor = factor

            def infer_meta(self, inputs):
                return inputs[0].replace(dtype=np.float64)

            def evaluate(self, inputs, halo_y, halo_x, ctx, state=None):
                aligned = inputs[0].crop(halo_y, halo_x)
                return aligned.with_data(aligned.data * self.factor)

        src = ArraySource(np.ones((1, 100, 100)))
        out = TileGraph(MemorySink(Scale(src, 2.0), name="x")).run()["x"]

    :param inputs: Upstream nodes.
    :param name: Optional node identifier.
    """

    def __init__(
        self,
        inputs: Sequence["TileNode"] = (),
        name: Optional[str] = None,
    ) -> None:
        self.inputs: Tuple[TileNode, ...] = tuple(inputs)
        self.name: Optional[str] = name

    @property
    def is_source(self) -> bool:
        """``True`` for a node without inputs (a source)."""
        return not self.inputs

    @property
    def halo_radius(self) -> Tuple[int, int]:
        """
        Halo ``(rows, columns)`` this node needs around each input block
        (``(0, 0)`` for a pixel-wise node).
        """
        return (0, 0)

    def input_halos(self, halo_y: int, halo_x: int) -> List[Tuple[int, int]]:
        """
        Halo each input must provide when this node's output is needed with a
        ``(halo_y, halo_x)`` halo. The default adds :attr:`halo_radius`.

        :param halo_y: Output halo in rows.
        :param halo_x: Output halo in columns.
        :return: One ``(rows, columns)`` halo per input.
        """
        ry, rx = self.halo_radius
        return [(halo_y + ry, halo_x + rx)] * len(self.inputs)

    def source_info(self, headers: HeaderCache) -> Tuple[SourceGrid, NodeMeta]:
        """
        Return the pixel grid and metadata of a source node (one without
        inputs). Called once when the graph is built.

        :param headers: Cache that opens each raster file once.
        :return: ``(grid, meta)``.
        """
        raise rsgispy.RSGISPyException(
            f"{type(self).__name__} has no inputs but does not implement "
            "source_info()."
        )

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        """
        Return the output metadata from the inputs' metadata, raising
        :class:`rsgispy.RSGISPyException` if the inputs are not compatible.
        The default returns unknown metadata.

        :param inputs: Metadata of each input, in order.
        :return: Output :class:`NodeMeta`.
        """
        del inputs
        return NodeMeta()

    def start_run(
        self, resources: RunResources, halo_y: int, halo_x: int
    ) -> Any:
        """
        Prepare per-run state (e.g. an open file) before the first tile. The
        returned object is passed to every :meth:`evaluate` call of the run.

        :param resources: Resources shared by the nodes of this run.
        :param halo_y: Output halo of this node in rows.
        :param halo_x: Output halo of this node in columns.
        :return: State object (``None`` by default).
        """
        del resources, halo_y, halo_x

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        """
        Compute this node's block for one tile.

        :param inputs: Input blocks; each has at least the halo requested by
                       :meth:`input_halos` (crop with
                       :meth:`TileTensor.crop` where needed).
        :param halo_y: Halo in rows the output must have.
        :param halo_x: Halo in columns the output must have.
        :param ctx: Position of the core tile in the image (halo 0).
        :param state: Object returned by :meth:`start_run`.
        :return: Output :class:`TileTensor`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement evaluate()."
        )

    def describe_params(self) -> str:
        """Short text describing the node's parameters (used by ``describe``)."""
        return ""

    @property
    def label(self) -> str:
        """Type and name of the node, for messages."""
        if self.name:
            return f"{type(self).__name__} '{self.name}'"
        return type(self).__name__


class TileSink(TileNode):
    """
    Base class for sinks: terminal nodes that receive the core block (halo
    removed) of their input for every tile.

    A sink implements :meth:`open` (once, before the first tile),
    :meth:`write` (once per tile) and :meth:`close` (after the last tile).
    Sinks with ``returns_result = True`` put :attr:`result` in the dictionary
    returned by :meth:`~rsgispy.imageprocess.TileGraph.run`.

    :param source: Upstream node.
    :param name: Optional sink identifier.
    """

    returns_result: bool = False
    result_prefix: str = "sink"

    def __init__(self, source: TileNode, name: Optional[str] = None) -> None:
        super().__init__(inputs=(source,), name=name)
        self.result: Any = None

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return inputs[0]

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        """
        Prepare for writing.

        :param grid: Output :class:`~rsgispy.imageprocess.IntersectionGrid`
                     (size, transform and CRS).
        :param meta: Metadata of the sink's input.
        :param stack: ``ExitStack`` of the run, for resources that must be
                      closed (also on errors).
        :param use_pbar: Whether progress bars may be shown.
        """
        raise NotImplementedError

    def write(self, block: np.ndarray, window: Window) -> None:
        """
        Receive the core block ``(bands, height, width)`` of one tile.

        :param block: Block without halo.
        :param window: Position of the block in the image.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Finish after the last tile (not called if the run fails)."""

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        raise rsgispy.RSGISPyException("A TileSink cannot be evaluated.")


class RasterSource(TileNode):
    """
    Source node reading tiled blocks with halo padding from an on-disk raster.

    All :class:`RasterSource` nodes of a graph on the same file share one
    open dataset and read their bands together for each tile.

    :param path: Input raster file path.
    :param bands: Optional 1-based band index or sequence of band indices to
                  read (reads all bands when ``None``).
    :param dtype: NumPy dtype to cast read blocks into (Default = ``np.float64``).
    :param border: Border padding policy outside image bounds
                   (Default = ``HaloBorder('edge')``).
    :param nodata: Optional override for the source nodata value (uses raster
                   metadata nodata when ``None``).
    :param name: Optional node identifier.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        path: Union[str, os.PathLike],
        bands: Optional[Union[int, Sequence[int]]] = None,
        dtype: Any = np.float64,
        border: Optional[HaloBorder] = None,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(), name=name)
        self.path: str = os.fspath(path)
        if bands is None:
            self.bands: Optional[List[int]] = None
        elif isinstance(bands, (int, np.integer)):
            self.bands = [int(bands)]
        else:
            self.bands = [int(b) for b in bands]
        self.dtype = np.dtype(dtype)
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.nodata: Optional[float] = nodata
        self._count: Optional[int] = None

    def source_info(self, headers: HeaderCache) -> Tuple[SourceGrid, NodeMeta]:
        hdr = headers.get(self.path)
        self._count = hdr.count
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
            height=hdr.height,
            width=hdr.width,
            transform=hdr.transform,
            crs=hdr.crs,
            nodata=nodata,
            label=f"'{self.path}'",
        )
        meta = NodeMeta(
            n_bands=len(bands), band_names=names, dtype=self.dtype, nodata=nodata
        )
        return grid, meta

    def start_run(
        self, resources: RunResources, halo_y: int, halo_x: int
    ) -> SharedRasterReader:
        reader = resources.raster_reader(self.path)
        count = int(reader.dataset.count)
        bands = self.bands if self.bands is not None else list(range(1, count + 1))
        reader.register(self, bands, halo_y, halo_x, self.dtype)
        return reader

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        return state.read(self, _ctx_window(ctx), self.dtype, self.border)

    def describe_params(self) -> str:
        bands = "all" if self.bands is None else str(self.bands)
        return f"path='{self.path}' bands={bands}"


class ArraySource(TileNode):
    """
    Source node reading tiled blocks with halo padding from an in-memory NumPy
    array of shape ``(H, W)`` or ``(C, H, W)``.

    :param array: Input 2D ``(H, W)`` or 3D ``(C, H, W)`` array.
    :param transform: Optional affine geotransform for downstream sinks.
    :param crs: Optional CRS for downstream sinks.
    :param border: Border padding policy outside array bounds
                   (Default = ``HaloBorder('edge')``).
    :param nodata: Optional nodata value associated with ``array``.
    :param dtype: Optional NumPy dtype to cast tiles into (preserves array
                  dtype when ``None``).
    :param name: Optional node identifier.
    :param band_names: Optional band names (one per band).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        array: np.ndarray,
        transform: Optional[Affine] = None,
        crs: Any = None,
        border: Optional[HaloBorder] = None,
        nodata: Optional[float] = None,
        dtype: Any = None,
        name: Optional[str] = None,
        *,
        band_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(inputs=(), name=name)
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise rsgispy.RSGISPyException(
                f"ArraySource expects a 2D or 3D array, got shape {arr.shape}."
            )
        self.array: np.ndarray = arr
        self.transform: Optional[Affine] = transform
        self.crs: Any = crs
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.nodata: Optional[float] = nodata
        self.dtype: np.dtype = np.dtype(dtype) if dtype is not None else arr.dtype
        if band_names is not None and len(band_names) != arr.shape[0]:
            raise rsgispy.RSGISPyException(
                f"ArraySource has {arr.shape[0]} band(s) but {len(band_names)} "
                "band names were given."
            )
        self.band_names: Optional[Tuple[str, ...]] = (
            tuple(str(b) for b in band_names) if band_names is not None else None
        )

    def source_info(self, headers: HeaderCache) -> Tuple[SourceGrid, NodeMeta]:
        del headers
        grid = SourceGrid(
            height=int(self.array.shape[1]),
            width=int(self.array.shape[2]),
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            label=self.label,
        )
        meta = NodeMeta(
            n_bands=int(self.array.shape[0]),
            band_names=self.band_names,
            dtype=self.dtype,
            nodata=self.nodata,
        )
        return grid, meta

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        return read_array_source(self, _ctx_window(ctx), halo_y, halo_x)

    def describe_params(self) -> str:
        return f"shape={tuple(self.array.shape)}"


class WindowOp(TileNode):
    """
    Moving-window neighbourhood operation consuming a single upstream node
    with halo radius ``radius``.

    ``func`` receives a 3D array of shape ``(C, H_out + 2*ry, W_out + 2*rx)``
    (and optionally a :class:`TileContext` when ``pass_ctx=True``) whose
    out-of-image margins have already been padded using ``border``.  ``func``
    may return either:

    * a valid-cropped array of shape ``(C_out, H_out, W_out)``, or
    * a same-size array of shape ``(C_out, H_out + 2*ry, W_out + 2*rx)``,
      which is automatically cropped by ``(ry, rx)``.

    :param source: Upstream :class:`TileNode`.
    :param radius: Non-negative integer or ``(ry, rx)`` halo radius.
    :param func: Callable transforming the padded tile array.
    :param border: Border padding policy applied to out-of-image edges before
                   calling ``func`` (Default = ``HaloBorder('edge')``).
    :param pass_ctx: If ``True``, pass ``(arr, ctx)`` to ``func``; otherwise
                     pass ``(arr,)`` (Default = ``False``).
    :param name: Optional node identifier.
    :param meta_fn: Optional callable ``(input_metas) -> NodeMeta`` giving
                    the output metadata (e.g. :func:`same_bands_meta`);
                    unknown when ``None``.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments

    def __init__(
        self,
        source: TileNode,
        radius: Union[int, Tuple[int, int]],
        func: Callable[..., np.ndarray],
        border: Optional[HaloBorder] = None,
        pass_ctx: bool = False,
        name: Optional[str] = None,
        *,
        meta_fn: Optional[MetaFunc] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.radius_y, self.radius_x = _normalise_radius(radius)
        self.func = func
        self.border: HaloBorder = border if border is not None else HaloBorder("edge")
        self.pass_ctx: bool = bool(pass_ctx)
        self.meta_fn: Optional[MetaFunc] = meta_fn

    @property
    def halo_radius(self) -> Tuple[int, int]:
        return (self.radius_y, self.radius_x)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return self.meta_fn(inputs) if self.meta_fn is not None else NodeMeta()

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        return eval_window_op(self, inputs[0], halo_y, halo_x, ctx)

    def describe_params(self) -> str:
        return f"radius=({self.radius_y}, {self.radius_x}) border={self.border.mode}"


class PointwiseOp(TileNode):
    """
    Pixel-wise operation combining one or more upstream :class:`TileNode`
    objects at identical spatial alignment (automatically cropping mismatched
    halos for skip connections).

    :param inputs: Single :class:`TileNode` or sequence of :class:`TileNode` objects.
    :param func: Callable receiving unpacked 3D tile arrays ``(*arrays)`` (or
                 ``(*arrays, ctx)`` when ``pass_ctx=True``).
    :param pass_ctx: If ``True``, append :class:`TileContext` as the final
                     argument to ``func`` (Default = ``False``).
    :param name: Optional node identifier.
    :param meta_fn: Optional callable ``(input_metas) -> NodeMeta`` giving
                    the output metadata (e.g. :func:`same_bands_meta`);
                    unknown when ``None``.
    """

    def __init__(
        self,
        inputs: Union[TileNode, Sequence[TileNode]],
        func: Callable[..., np.ndarray],
        pass_ctx: bool = False,
        name: Optional[str] = None,
        *,
        meta_fn: Optional[MetaFunc] = None,
    ) -> None:
        node_list = (inputs,) if isinstance(inputs, TileNode) else tuple(inputs)
        if not node_list:
            raise rsgispy.RSGISPyException(
                "PointwiseOp requires at least one upstream input node."
            )
        super().__init__(inputs=node_list, name=name)
        self.func = func
        self.pass_ctx: bool = bool(pass_ctx)
        self.meta_fn: Optional[MetaFunc] = meta_fn

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return self.meta_fn(inputs) if self.meta_fn is not None else NodeMeta()

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        return eval_pointwise_op(self, inputs, halo_y, halo_x, ctx)


class BandMathOp(TileNode):
    """
    Point-wise band-math expression evaluated via ``pymu-eval`` over named
    upstream :class:`TileNode` inputs.

    When all upstream nodes have the same number of bands ``C``, the expression
    is evaluated band-by-band producing ``C`` output bands.  Single-band nodes
    broadcast automatically against multi-band nodes. Incompatible band
    counts are reported when the :class:`TileGraph` is built (when the
    inputs' band counts are known).

    :param inputs: Mapping of ``{var_name: TileNode}`` referenced in ``expr``.
    :param expr: ``pymu-eval`` expression string (e.g. ``'(nir - red) / (nir + red)'``
                 or ``'raw - smooth'``).
    :param nodata: Optional nodata value; when set, pixels matching ``nodata``
                   in any input are preserved as ``nodata`` in the output.
    :param name: Optional node identifier.
    """

    def __init__(
        self,
        inputs: Mapping[str, TileNode],
        expr: str,
        nodata: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        if not inputs:
            raise rsgispy.RSGISPyException(
                "BandMathOp requires a non-empty inputs mapping."
            )
        if not expr or not expr.strip():
            raise rsgispy.RSGISPyException("BandMathOp expr must be non-empty.")
        self.var_names: Tuple[str, ...] = tuple(inputs.keys())
        super().__init__(inputs=tuple(inputs.values()), name=name)
        self.expr: str = str(expr)
        self.nodata: Optional[float] = nodata

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        counts = [m.n_bands for m in inputs]
        multi = {c for c in counts if c is not None and c != 1}
        if len(multi) > 1:
            raise rsgispy.RSGISPyException(
                f"BandMathOp inputs have incompatible band counts: {counts} "
                f"(variables {list(self.var_names)})."
            )
        n_bands: Optional[int] = None
        names: Optional[Tuple[str, ...]] = None
        if all(c is not None for c in counts):
            n_bands = max(c for c in counts if c is not None)
            if n_bands > 1:
                for meta in inputs:
                    if meta.n_bands == n_bands and meta.band_names is not None:
                        names = meta.band_names
                        break
        return NodeMeta(
            n_bands=n_bands, band_names=names, dtype=np.float64, nodata=self.nodata
        )

    def start_run(
        self, resources: RunResources, halo_y: int, halo_x: int
    ) -> Dict[Any, pymu_eval.Parser]:
        del resources, halo_y, halo_x
        return {}

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        parsers = state if state is not None else {}
        return eval_band_math_op(self, inputs, halo_y, halo_x, parsers)

    def describe_params(self) -> str:
        return f"expr='{self.expr}' vars={list(self.var_names)}"


class SelectBands(TileNode):
    """
    Select (and reorder or repeat) bands of a node's output.

    :param source: Upstream :class:`TileNode`.
    :param bands: 1-based band index or sequence of band indices.
    :param name: Optional node identifier.

    .. code:: python

        from rsgispy.imageprocess import (
            BandMathOp, DiskSink, RasterSource, SelectBands, TileGraph,
        )

        s2 = RasterSource("s2.tif")
        red = SelectBands(s2, 3)
        nir = SelectBands(s2, 8)
        ndvi = BandMathOp({"red": red, "nir": nir}, "(nir - red) / (nir + red)")
        TileGraph(DiskSink(ndvi, "ndvi.tif", band_names=["NDVI"])).run()
    """

    def __init__(
        self,
        source: TileNode,
        bands: Union[int, Sequence[int]],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        if isinstance(bands, (int, np.integer)):
            band_list = [int(bands)]
        else:
            band_list = [int(b) for b in bands]
        if not band_list:
            raise rsgispy.RSGISPyException("SelectBands requires at least one band.")
        for band in band_list:
            if band < 1:
                raise rsgispy.RSGISPyException(
                    f"SelectBands band indices are 1-based, got {band}."
                )
        self.bands: List[int] = band_list
        self._index = [b - 1 for b in band_list]

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        if meta.n_bands is not None:
            for band in self.bands:
                if band > meta.n_bands:
                    raise rsgispy.RSGISPyException(
                        f"{self.label}: band {band} requested but the input "
                        f"has {meta.n_bands} band(s)."
                    )
        names = (
            tuple(meta.band_names[i] for i in self._index)
            if meta.band_names is not None
            else None
        )
        return meta.replace(n_bands=len(self.bands), band_names=names)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = inputs[0].crop(halo_y, halo_x)
        if max(self._index) >= aligned.data.shape[0]:
            raise rsgispy.RSGISPyException(
                f"{self.label}: bands {self.bands} requested but the input "
                f"has {aligned.data.shape[0]} band(s)."
            )
        return aligned.with_data(aligned.data[self._index])

    def describe_params(self) -> str:
        return f"bands={self.bands}"


class StackBands(TileNode):
    """
    Stack the bands of several nodes into one multi-band output (in the
    order given), carrying the band names through.

    :param sources: Sequence of upstream :class:`TileNode` objects.
    :param name: Optional node identifier.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, StackBands, TileGraph
        from rsgispy.imagefilter.stats import mean_filter_node, stddev_filter_node

        src = RasterSource("in.tif", bands=1)
        stack = StackBands(
            [src, mean_filter_node(src, 5), stddev_filter_node(src, 5)]
        )
        TileGraph(DiskSink(stack, "features.tif")).run()
    """

    def __init__(
        self,
        sources: Sequence[TileNode],
        name: Optional[str] = None,
    ) -> None:
        src_list = tuple(sources)
        if not src_list:
            raise rsgispy.RSGISPyException("StackBands requires at least one input.")
        super().__init__(inputs=src_list, name=name)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        counts = [m.n_bands for m in inputs]
        n_bands = (
            sum(c for c in counts if c is not None)
            if all(c is not None for c in counts)
            else None
        )
        names: Optional[Tuple[str, ...]] = None
        if n_bands is not None and any(m.band_names is not None for m in inputs):
            name_list: List[str] = []
            for idx, meta in enumerate(inputs):
                if meta.band_names is not None:
                    name_list.extend(meta.band_names)
                else:
                    name_list.extend(
                        f"input{idx + 1}_b{b + 1}" for b in range(int(meta.n_bands or 0))
                    )
            names = tuple(name_list)
        dtypes = [m.dtype for m in inputs]
        dtype = (
            np.result_type(*dtypes) if all(d is not None for d in dtypes) else None
        )
        nodatas = [m.nodata for m in inputs]
        nodata = nodatas[0] if all(_same_nodata(nodatas[0], n) for n in nodatas) else None
        return NodeMeta(n_bands=n_bands, band_names=names, dtype=dtype, nodata=nodata)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        return aligned[0].with_data(np.concatenate([t.data for t in aligned], axis=0))


class Cast(TileNode):
    """
    Cast a node's blocks to another NumPy dtype. Integer casts replace
    non-finite values with ``0`` and, with ``clamp_int=True``, clamp values
    to the dtype's range first (as :class:`DiskSink` does).

    :param source: Upstream :class:`TileNode`.
    :param dtype: Target NumPy dtype.
    :param clamp_int: Clamp to the integer range before casting
                      (Default = ``True``).
    :param name: Optional node identifier.

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import Cast, MemorySink, RasterSource, TileGraph

        cls = Cast(RasterSource("classes.tif"), np.uint8)
        arr = TileGraph(MemorySink(cls, name="cls")).run()["cls"]
    """

    def __init__(
        self,
        source: TileNode,
        dtype: Any,
        clamp_int: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.dtype = np.dtype(dtype)
        self.clamp_int = bool(clamp_int)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return inputs[0].replace(dtype=self.dtype)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import cast_result_block

        aligned = inputs[0].crop(halo_y, halo_x)
        arr = aligned.data
        out = cast_result_block(
            arr, int(arr.shape[1]), int(arr.shape[2]), self.dtype, clamp_int=self.clamp_int
        )
        return aligned.with_data(out)

    def describe_params(self) -> str:
        return f"dtype={self.dtype}"


class SetNoData(TileNode):
    """
    Set (or clear) the no-data value in a node's metadata without changing
    its pixels. Node builders called with ``no_data_val=None`` use the
    no-data value of their input's metadata, so this controls the no-data
    handling of everything downstream:

    * ``SetNoData(node, None)`` — no no-data handling (only non-finite values
      are treated as invalid), e.g. when ``0`` is a valid value although the
      file declares it as no data;
    * ``SetNoData(node, value)`` — treat ``value`` as no data, e.g. for a
      file without a no-data value (as ``RasterSource(path, nodata=value)``).

    :param source: Upstream :class:`TileNode`.
    :param nodata: The no-data value, or ``None`` for none.
    :param name: Optional node identifier.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, SetNoData, TileGraph
        from rsgispy.imagefilter.stats import median_filter_node

        # The file declares 0 as no data, but 0 is a valid value here.
        src = SetNoData(RasterSource("temperature.tif"), None)
        TileGraph(DiskSink(median_filter_node(src, 5), "temp_med5.tif")).run()
    """

    def __init__(
        self,
        source: TileNode,
        nodata: Optional[float],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.nodata: Optional[float] = None if nodata is None else float(nodata)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        return inputs[0].replace(nodata=self.nodata)

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        return inputs[0].crop(halo_y, halo_x)

    def describe_params(self) -> str:
        return f"nodata={self.nodata!r}"


def _same_nodata(val_a: Optional[float], val_b: Optional[float]) -> bool:
    """Compare two no-data values (``None`` and NaN aware)."""
    if val_a is None or val_b is None:
        return val_a is None and val_b is None
    if math.isnan(float(val_a)) or math.isnan(float(val_b)):
        return math.isnan(float(val_a)) and math.isnan(float(val_b))
    return float(val_a) == float(val_b)


def _nodata_fits(nodata: float, np_dtype: np.dtype) -> bool:
    """``True`` if ``nodata`` can be stored in rasters of ``np_dtype``."""
    val = float(nodata)
    if np.issubdtype(np_dtype, np.integer):
        if not math.isfinite(val) or not val.is_integer():
            return False
        info = np.iinfo(np_dtype)
        return info.min <= val <= info.max
    if math.isnan(val) or math.isinf(val):
        return True
    return abs(val) <= float(np.finfo(np_dtype).max)  # pylint: disable=no-member


class DiskSink(TileSink):
    """
    Terminal sink writing a node's tiled output to an on-disk raster dataset.

    When ``nodata`` or ``band_names`` are not given (and ``inherit_meta`` is
    ``True``), the no-data value and band names of the input node's metadata
    are used (the no-data value only when the output data type can store it).

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param output_img: Output raster file path.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant (Default = ``TYPE_32FLOAT``).
    :param nodata: Optional output nodata value.
    :param clamp_int: Clamp values to integer type bounds before casting
                      (Default = ``True``).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    :param band_names: Optional sequence of output band description strings.
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       after writing (Default = ``False``).
    :param name: Optional sink identifier.
    :param inherit_meta: Use the input's no-data value and band names when
                         ``nodata`` / ``band_names`` are ``None``
                         (Default = ``True``).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes

    result_prefix = "disk"

    def __init__(
        self,
        source: TileNode,
        output_img: Union[str, os.PathLike],
        gdalformat: str = "GTIFF",
        datatype: int = rsgispy.TYPE_32FLOAT,
        nodata: Optional[float] = None,
        clamp_int: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
        band_names: Optional[Sequence[str]] = None,
        calc_stats: bool = False,
        name: Optional[str] = None,
        *,
        inherit_meta: bool = True,
    ) -> None:
        super().__init__(source, name=name)
        self.output_img: str = os.fspath(output_img)
        self.gdalformat: str = str(gdalformat)
        self.datatype: int = int(datatype)
        self.nodata: Optional[float] = nodata
        self.clamp_int: bool = bool(clamp_int)
        self.creation_opts: Optional[Mapping[str, Any]] = creation_opts
        self.band_names: Optional[List[str]] = (
            list(band_names) if band_names is not None else None
        )
        self.calc_stats: bool = bool(calc_stats)
        self.inherit_meta: bool = bool(inherit_meta)
        self._run: Optional[Dict[str, Any]] = None

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        if (
            self.band_names is not None
            and meta.n_bands is not None
            and len(self.band_names) > meta.n_bands
        ):
            raise rsgispy.RSGISPyException(
                f"Cannot set {len(self.band_names)} band names on image "
                f"'{self.output_img}' with {meta.n_bands} band(s)."
            )
        return meta

    def resolve_output_meta(
        self, meta: NodeMeta
    ) -> Tuple[Optional[float], Optional[List[str]]]:
        """
        Return the ``(nodata, band_names)`` written for input metadata
        ``meta`` (explicit values first, then the inherited ones).
        """
        out_np_dt = np.dtype(rsgispy.get_numpy_datatype(self.datatype))
        nodata = self.nodata
        if (
            nodata is None
            and self.inherit_meta
            and meta.nodata is not None
            and _nodata_fits(meta.nodata, out_np_dt)
        ):
            nodata = float(meta.nodata)
        names = self.band_names
        if names is None and self.inherit_meta and meta.band_names is not None:
            names = list(meta.band_names)
        return nodata, names

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        nodata, names = self.resolve_output_meta(meta)
        self._run = {
            "grid": grid,
            "stack": stack,
            "use_pbar": use_pbar,
            "nodata": nodata,
            "band_names": names,
            "dst": None,
            "np_dtype": np.dtype(rsgispy.get_numpy_datatype(self.datatype)),
        }

    def _open_dataset(self, count: int) -> rasterio.io.DatasetWriter:
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import (
            normalise_creation_options,
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
            out_rio_dtype=rsgispy.get_rasterio_datatype(self.datatype),
            grid=run["grid"],
            count=int(count),
            output_exists=False,
            creation_opts=normalise_creation_options(self.creation_opts),
        )
        if run["nodata"] is not None:
            dst.nodata = run["nodata"]
        names = run["band_names"]
        if names is not None:
            if len(names) > dst.count:
                raise rsgispy.RSGISPyException(
                    f"Cannot set {len(names)} band names on image "
                    f"'{self.output_img}' with {dst.count} band(s)."
                )
            for b_idx, b_name in enumerate(names, start=1):
                dst.set_band_description(b_idx, str(b_name))
        # Statistics run once the writer has closed (for COG, before the
        # temporary GeoTIFF is converted).
        pop_stats_on_close(
            dst,
            self.calc_stats,
            no_data_val=run["nodata"],
            use_pbar=run["use_pbar"],
        )
        run["dst"] = dst
        return dst

    def write(self, block: np.ndarray, window: Window) -> None:
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import cast_result_block

        run = self._run
        if run is None:
            raise rsgispy.RSGISPyException("DiskSink.write() called before open().")
        dst = run["dst"]
        if dst is None:
            dst = self._open_dataset(int(block.shape[0]))
        cast_blk = cast_result_block(
            block,
            int(window.height),
            int(window.width),
            run["np_dtype"],
            clamp_int=self.clamp_int,
        )
        dst.write(cast_blk, window=window)

    def close(self) -> None:
        self._run = None

    def describe_params(self) -> str:
        return (
            f"output='{self.output_img}' format={self.gdalformat} "
            f"datatype={rsgispy.get_rasterio_datatype(self.datatype)}"
        )


class MemorySink(TileSink):
    """
    Terminal sink assembling a node's tiled output into an in-memory NumPy
    array accessible on :attr:`result` after :meth:`TileGraph.run`.

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param dtype: Optional output NumPy dtype (inferred from first tile when
                  ``None``).
    :param clamp_int: Clamp values to integer bounds when casting
                      (Default = ``True``).
    :param name: Optional sink identifier (used as key in
                 :meth:`TileGraph.run` return dictionary).
    """

    returns_result = True
    result_prefix = "memory"

    def __init__(
        self,
        source: TileNode,
        dtype: Any = None,
        clamp_int: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, name=name)
        self.dtype: Optional[np.dtype] = np.dtype(dtype) if dtype is not None else None
        self.clamp_int: bool = bool(clamp_int)
        self.result: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None
        self._array: Optional[np.ndarray] = None

    def open(
        self,
        grid: Any,
        meta: NodeMeta,
        *,
        stack: ExitStack,
        use_pbar: bool = True,
    ) -> None:
        del meta, stack, use_pbar
        self._shape = (int(grid.height), int(grid.width))
        self._array = None

    def write(self, block: np.ndarray, window: Window) -> None:
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import cast_result_block

        if self._shape is None:
            raise rsgispy.RSGISPyException("MemorySink.write() called before open().")
        out_np_dt = self.dtype if self.dtype is not None else block.dtype
        if self._array is None:
            self._array = np.empty(
                (int(block.shape[0]), self._shape[0], self._shape[1]),
                dtype=out_np_dt,
            )
        cast_blk = cast_result_block(
            block,
            int(window.height),
            int(window.width),
            out_np_dt,
            clamp_int=self.clamp_int,
        )
        r0 = int(window.row_off)
        c0 = int(window.col_off)
        self._array[:, r0:r0 + int(window.height), c0:c0 + int(window.width)] = cast_blk

    def close(self) -> None:
        self.result = self._array
        self._array = None
        self._shape = None

    def describe_params(self) -> str:
        return f"dtype={'auto' if self.dtype is None else self.dtype}"
