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
import warnings
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
                       :meth:`TileNode.input_halos` (crop with
                       :meth:`TileTensor.crop` where needed).
        :param halo_y: Halo in rows the output must have.
        :param halo_x: Halo in columns the output must have.
        :param ctx: Position of the core tile in the image (halo 0).
        :param state: Object returned by :meth:`TileNode.start_run`.
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


def _band_names_arg(band_names: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """Normalise a ``band_names`` argument (``None`` or a sequence of names)."""
    if band_names is None:
        return None
    if isinstance(band_names, str):
        band_names = [band_names]
    names = tuple(str(n) for n in band_names)
    if not names:
        raise rsgispy.RSGISPyException("band_names must not be empty.")
    return names


def _check_band_names(
    label: str, names: Tuple[str, ...], n_bands: Optional[int]
) -> None:
    """Raise when ``names`` does not give one name per band."""
    if n_bands is not None and len(names) != int(n_bands):
        raise rsgispy.RSGISPyException(
            f"{label}: {len(names)} band name(s) given for {n_bands} band(s)."
        )


class SelectBands(TileNode):
    """
    Select (and reorder or repeat) bands of a node's output.

    :param source: Upstream :class:`TileNode`.
    :param bands: 1-based band index or sequence of band indices.
    :param name: Optional node identifier.
    :param band_names: Optional names of the selected bands (one per band;
                       Default = ``None``: the input's names).

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
        *,
        band_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.band_names = _band_names_arg(band_names)
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
        if self.band_names is not None:
            _check_band_names(f"SelectBands {band_list}", self.band_names, len(band_list))

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        if meta.n_bands is not None:
            for band in self.bands:
                if band > meta.n_bands:
                    raise rsgispy.RSGISPyException(
                        f"{self.label}: band {band} requested but the input "
                        f"has {meta.n_bands} band(s)."
                    )
        if self.band_names is not None:
            names: Optional[Tuple[str, ...]] = self.band_names
        elif meta.band_names is not None:
            names = tuple(meta.band_names[i] for i in self._index)
        else:
            names = None
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
        if self.band_names is not None:
            return f"bands={self.bands} band_names={list(self.band_names)}"
        return f"bands={self.bands}"


class StackBands(TileNode):
    """
    Stack the bands of several nodes into one multi-band output (in the
    order given), carrying the band names through.

    **No data.** The stack has one no-data value for all its bands:

    * with ``no_data_val=None`` (the default), the bands are stacked
      unchanged. When every input has the same no-data value (or none has
      one) the stack has that value. When they differ the stack has none
      (``None``) and a :class:`UserWarning` says so: the sinks would then not
      exclude the inputs' no-data pixels;
    * with ``no_data_val`` given, every band of a pixel where any input is
      invalid (equal to that input's own no-data value, or not finite) is set
      to ``no_data_val``, which becomes the stack's no-data value. This keeps
      the sinks' rule (a pixel invalid in any band is invalid) working for
      every input, e.g. Sentinel-2 bands with no data ``0`` stacked with an
      index with no data ``-9999``;
    * with ``no_data_val='auto'``, the same, with the first input's no-data
      value when it has one, else NaN for a floating-point stack; for any
      other stack :class:`rsgispy.RSGISPyException` asks for a value.

    ``TileGraph.describe()`` shows the stack's no-data value and whether
    invalid pixels are masked across the inputs. A valid value equal to the
    stack's no-data value becomes no data downstream, so choose
    ``no_data_val`` outside the data range of every input. Masking makes,
    e.g., a valid slope of ``0`` no data where an aspect band stacked with it
    has its no-data value (flat pixels), so leave ``no_data_val=None`` when
    the bands' values are needed as they are.

    :param sources: Sequence of upstream :class:`TileNode` objects.
    :param name: Optional node identifier.
    :param band_names: Optional names of the stacked bands, one per band
                       (Default = ``None``: the inputs' names). Use it when
                       inputs have the same band names, e.g. two filters of
                       the same bands.
    :param no_data_val: No-data value of the stack, written into every band
                        of a pixel where any input is invalid, or ``'auto'``
                        (Default = ``None``: the inputs' shared no-data value,
                        nothing masked; see above). It must fit the stack's
                        data type (the NumPy result type of the inputs).

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, StackBands, TileGraph
        from rsgispy.imagefilter.stats import mean_filter_node, stddev_filter_node

        src = RasterSource("in.tif", bands=1)
        stack = StackBands(
            [src, mean_filter_node(src, 5), stddev_filter_node(src, 5)],
            band_names=["value", "mean5", "stddev5"],
        )
        TileGraph(DiskSink(stack, "features.tif")).run()

        # Sentinel-2 (no data 0) with an index (no data -999): the stack
        # writes -999 into every band where either input is no data, so a
        # SampleSink or predict_raster_node on it skips those pixels.
        s2 = RasterSource("s2.tif", bands=[4, 8])
        ndvi = RasterSource("ndvi.tif")
        feats = StackBands([s2, ndvi], no_data_val=-999)
    """

    def __init__(
        self,
        sources: Sequence[TileNode],
        name: Optional[str] = None,
        *,
        band_names: Optional[Sequence[str]] = None,
        no_data_val: Optional[Union[float, str]] = None,
    ) -> None:
        src_list = tuple(sources)
        if not src_list:
            raise rsgispy.RSGISPyException("StackBands requires at least one input.")
        super().__init__(inputs=src_list, name=name)
        self.band_names = _band_names_arg(band_names)
        if isinstance(no_data_val, str):
            if no_data_val != "auto":
                raise rsgispy.RSGISPyException(
                    f"StackBands no_data_val must be a number, 'auto' or None, "
                    f"got {no_data_val!r}."
                )
            self.no_data_val: Optional[Union[float, str]] = no_data_val
        else:
            self.no_data_val = None if no_data_val is None else float(no_data_val)
        # (stack no-data value, each input's no-data value) when invalid
        # pixels are masked across the inputs; set by infer_meta.
        self._nodata_plan: Optional[Tuple[float, Tuple[Optional[float], ...]]] = None
        self._nodata_differ = False
        self._input_nodatas: Tuple[Optional[float], ...] = ()

    def _stacked_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        """Bands, names and dtype of the stack; the shared no-data value or ``None``."""
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
        if self.band_names is not None:
            _check_band_names(self.label, self.band_names, n_bands)
            names = self.band_names
            if n_bands is None:
                n_bands = len(names)
        return NodeMeta(n_bands=n_bands, band_names=names, dtype=dtype, nodata=nodata)

    def _resolve_nodata(self, inputs: Sequence[NodeMeta], dtype: Any) -> Optional[float]:
        """The stack's no-data value; sets the masking plan (see the class docs)."""
        nodatas = tuple(m.nodata for m in inputs)
        self._nodata_plan = None
        self._nodata_differ = not all(_same_nodata(nodatas[0], n) for n in nodatas)
        self._input_nodatas = nodatas
        if self.no_data_val is None:
            if self._nodata_differ:
                warnings.warn(
                    f"{self.label}: the inputs have different no-data values "
                    f"{list(nodatas)}, so the stack has no no-data value and sinks "
                    "will not exclude the inputs' no-data pixels. Give "
                    "StackBands(..., no_data_val=<value>) or no_data_val='auto' to "
                    "write one no-data value into every band of a pixel where any "
                    "input is invalid.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            return nodatas[0]
        if self.no_data_val != "auto":
            target = float(self.no_data_val)
        elif nodatas[0] is not None:
            target = float(nodatas[0])
        elif dtype is not None and np.issubdtype(np.dtype(dtype), np.floating):
            target = float("nan")
        else:
            raise rsgispy.RSGISPyException(
                f"{self.label}: no_data_val='auto' but the first input has no "
                f"no-data value and the stack is not floating point (dtype="
                f"{None if dtype is None else np.dtype(dtype)}); give "
                "StackBands(..., no_data_val=<value>)."
            )
        if dtype is not None and not _nodata_fits(target, np.dtype(dtype)):
            raise rsgispy.RSGISPyException(
                f"{self.label}: the no-data value {target!r} does not fit the "
                f"stack's data type {np.dtype(dtype)}; give a no_data_val that "
                "fits it (or Cast an input)."
            )
        if any(n is not None for n in nodatas) or not math.isnan(target):
            self._nodata_plan = (target, nodatas)
        return target

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = self._stacked_meta(inputs)
        return meta.replace(nodata=self._resolve_nodata(inputs, meta.dtype))

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        data = np.concatenate([t.data for t in aligned], axis=0)
        plan = self._nodata_plan
        if plan is not None:
            # pylint: disable-next=import-outside-toplevel
            from rsgispy.imageprocess import build_valid_mask

            target, nodatas = plan
            invalid = np.zeros(data.shape[1:], dtype=bool)
            for tensor, nodata in zip(aligned, nodatas):
                invalid |= ~np.all(build_valid_mask(tensor.data, nodata), axis=0)
            if np.any(invalid):
                data[:, invalid] = target
        return aligned[0].with_data(data)

    def describe_params(self) -> str:
        parts = []
        if self.band_names is not None:
            parts.append(f"band_names={list(self.band_names)}")
        if self.no_data_val is not None:
            parts.append(f"no_data_val={self.no_data_val!r}")
        if self._nodata_plan is not None:
            parts.append(
                f"nodata={self._nodata_plan[0]!r} where invalid in any input "
                f"(inputs' nodata={list(self._nodata_plan[1])})"
            )
        elif self._nodata_differ and self.no_data_val is None:
            parts.append(
                f"inputs' nodata={list(self._input_nodatas)} differ: stack nodata=None, "
                "not masked (give no_data_val)"
            )
        return " ".join(parts)


def _warn_clamp_int(clamp_int: Any, where: str) -> None:
    """Deprecation warning for ``clamp_int=False`` (integer casts always clamp)."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess import warn_if_clamp_int_disabled

    warn_if_clamp_int_disabled(clamp_int, where)


class Cast(TileNode):
    """
    Cast a node's blocks to another NumPy dtype. Integer casts replace
    non-finite values with ``0``, round to the nearest integer and clamp to
    the dtype's range (as :class:`DiskSink` does).

    When the input has a no-data value, no-data pixels are written as the
    output no-data value, and valid pixels never take it. If the target
    dtype cannot hold the input no-data value (e.g. ``-9999`` cast to
    ``uint8``), the output no-data value becomes the type's maximum for
    unsigned and its minimum for signed integers, and the node metadata
    records it.

    :param source: Upstream :class:`TileNode`.
    :param dtype: Target NumPy dtype.
    :param clamp_int: Deprecated and ignored: integer casts always round and
                      clamp (``False`` issues a :class:`DeprecationWarning`).
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
        _warn_clamp_int(clamp_int, "Cast")
        self.clamp_int = True
        self._in_nodata: Optional[float] = None

    def _resolve_nodata(self, in_nd: Optional[float]) -> Optional[float]:
        """
        Output no-data value: ``in_nd`` when ``self.dtype`` can hold it,
        otherwise the type's maximum (unsigned) or minimum (signed integer).
        """
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageprocess import fits_dtype_range

        if in_nd is None:
            return None
        if not np.issubdtype(self.dtype, np.integer):
            return float(in_nd)
        if math.isfinite(float(in_nd)) and fits_dtype_range(in_nd, self.dtype):
            return float(in_nd)
        info = np.iinfo(self.dtype)
        return float(info.max if info.min == 0 else info.min)

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        self._in_nodata = inputs[0].nodata
        return inputs[0].replace(
            dtype=self.dtype, nodata=self._resolve_nodata(inputs[0].nodata)
        )

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
        in_nd = self._in_nodata
        if in_nd is None:
            out = cast_result_block(
                arr, int(arr.shape[1]), int(arr.shape[2]), self.dtype
            )
        else:
            arr_f = np.asarray(arr, dtype=np.float64)
            nd_mask = np.isnan(arr_f) if math.isnan(in_nd) else (arr_f == in_nd)
            out = cast_result_block(
                arr_f,
                int(arr.shape[1]),
                int(arr.shape[2]),
                self.dtype,
                nodata=self._resolve_nodata(in_nd),
                valid_mask=~nd_mask,
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


class SetBandNames(TileNode):
    """
    Set the band names in a node's metadata without changing its pixels,
    e.g. to name the result of a filter (whose bands keep the input's
    names) before stacking it with other nodes or writing it. A sink given
    ``band_names`` still overrides them.

    :param source: Upstream :class:`TileNode`.
    :param band_names: One name per band of ``source``.
    :param name: Optional node identifier.

    .. code:: python

        from rsgispy.imageprocess import (
            DiskSink, RasterSource, SetBandNames, StackBands, TileGraph,
        )
        from rsgispy.imagefilter.stats import median_filter_node

        ndvi = RasterSource("ndvi.tif")  # one band, named "NDVI"
        med = SetBandNames(median_filter_node(ndvi, 3), ["NDVI_med3"])
        TileGraph(DiskSink(StackBands([ndvi, med]), "ndvi_feats.tif")).run()
    """

    def __init__(
        self,
        source: TileNode,
        band_names: Sequence[str],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        names = _band_names_arg(band_names)
        if names is None:
            raise rsgispy.RSGISPyException("SetBandNames requires band names.")
        self.band_names: Tuple[str, ...] = names

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        meta = inputs[0]
        _check_band_names(self.label, self.band_names, meta.n_bands)
        return meta.replace(n_bands=len(self.band_names), band_names=self.band_names)

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
        return f"band_names={list(self.band_names)}"


def _same_nodata(val_a: Optional[float], val_b: Optional[float]) -> bool:
    """Compare two no-data values (``None`` and NaN aware):
    :func:`rsgispy.imageprocess._dtype_nodata._same_nodata`."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess._dtype_nodata import _same_nodata as _shared

    return _shared(val_a, val_b)


def datatype_from_meta(meta: NodeMeta) -> int:
    """
    The ``rsgispy.TYPE_*`` a :class:`DiskSink` with ``datatype=None`` writes
    for a node with metadata ``meta``: integer and ``float32`` data types are
    kept (``bool`` is written as ``uint8``); ``float64``, other types and an
    unknown data type give ``TYPE_32FLOAT``.

    :param meta: Metadata of the sink's input node.
    :return: ``rsgispy.TYPE_*`` constant.
    """
    if meta.dtype is None:
        return rsgispy.TYPE_32FLOAT
    np_dt = np.dtype(meta.dtype)
    if np_dt == np.bool_:
        return rsgispy.TYPE_8UINT
    if np.issubdtype(np_dt, np.integer) or np_dt == np.float32:
        return int(rsgispy.rsgis_dtype_from_np_dtype(np_dt, fallback=rsgispy.TYPE_32FLOAT))
    return rsgispy.TYPE_32FLOAT


def _nodata_fits(nodata: float, np_dtype: np.dtype) -> bool:
    """``True`` if ``nodata`` can be stored in rasters of ``np_dtype`` (NaN and
    infinities fit float types): :func:`rsgispy.imageprocess.fits_dtype_range`."""
    # pylint: disable=import-outside-toplevel
    from rsgispy.imageprocess import fits_dtype_range

    return fits_dtype_range(nodata, np_dtype, allow_nonfinite_float=True)


def _parse_sink_classes(
    colour_table: Optional[Mapping[int, Any]],
    classes: Optional[Mapping[int, Any]],
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int, int, int]]]:
    """Validated class names and RGBA colours of a thematic :class:`DiskSink`."""
    # Lazy import: rsgispy.imageutils.colourtable imports rsgispy.imageprocess.
    # pylint: disable=import-outside-toplevel,protected-access
    from rsgispy.imageutils import colourtable as _ct

    if colour_table is not None and classes is not None:
        raise rsgispy.RSGISPyException(
            "DiskSink: give colour_table or classes, not both."
        )
    if classes is not None:
        return _ct._parse_thematic_classes(classes)
    if colour_table is None:
        return {}, {}
    if not colour_table:
        raise rsgispy.RSGISPyException(
            "DiskSink: colour_table must be a non-empty mapping."
        )
    colours = {int(k): _ct._parse_colour_entry(v) for k, v in colour_table.items()}
    if min(colours) < 0:
        raise rsgispy.RSGISPyException(
            "DiskSink: colour table pixel values must be non-negative."
        )
    return {}, colours


class DiskSink(TileSink):
    """
    Terminal sink writing a node's tiled output to an on-disk raster dataset.

    When ``nodata`` or ``band_names`` are not given (and ``inherit_meta`` is
    ``True``), the no-data value and band names of the input node's metadata
    are used (the no-data value only when the output data type can store it).

    When ``datatype`` is ``None`` (the default) the data type comes from the
    input node's metadata (see :func:`datatype_from_meta`): integer and
    ``float32`` nodes are written in their own type (e.g. ``uint8`` for
    :func:`~rsgispy.elevation.illumination.hillshade_node` or a
    :func:`~rsgispy.classification.classpostprocess.majority_filter_node`
    with ``dtype=np.uint8``), ``bool`` as ``uint8``, and ``float64`` (the
    type of most nodes' blocks) or unknown types as ``float32``. Pass
    ``rsgispy.TYPE_64FLOAT`` for a ``float64`` file. Before rsgispy 0.4 the
    default was always ``TYPE_32FLOAT``.

    **Class maps.** With ``thematic=True`` (implied by ``colour_table`` or
    ``classes``) the output is a thematic image: when the writer closes
    (for a COG, before it is converted) every band is marked thematic
    (``LAYER_TYPE=thematic``), the colour table and class names are written
    to band 1 as by
    :func:`~rsgispy.imageutils.colourtable.set_thematic_classes`, and, unless
    ``calc_stats=False``, thematic statistics (pixel value ``0`` excluded
    when it is the no-data value) and pyramids are built (``NEAREST`` COG
    overviews), without replacing the colour table. Without a colour table,
    the statistics add a random one, as
    :func:`~rsgispy.imageutils.stats.pop_thmt_img_stats` does. The output
    must be an integer type, and ``uint8`` or ``uint16`` for a colour table.
    Read the classes back with
    :func:`~rsgispy.imageutils.colourtable.get_thematic_classes`.

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param output_img: Output raster file path.
    :param gdalformat: GDAL/rasterio driver format (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant (Default = ``None``: from the
                     input node's metadata).
    :param nodata: Optional output nodata value.
    :param clamp_int: Deprecated and ignored: integer outputs are always
                      rounded and clamped to the type's range (``False``
                      issues a :class:`DeprecationWarning`).
    :param creation_opts: Optional GDAL/rasterio creation options mapping.
    :param band_names: Optional sequence of output band description strings.
    :param calc_stats: If ``True``, populate raster statistics and pyramids
                       after writing (thematic statistics for a thematic
                       output). Default = ``None``: ``True`` for a thematic
                       output, else ``False``.
    :param name: Optional sink identifier.
    :param inherit_meta: Use the input's no-data value and band names when
                         ``nodata`` / ``band_names`` are ``None``
                         (Default = ``True``).
    :param thematic: Write a thematic (class) image (Default = ``False``;
                     ``True`` when ``colour_table`` or ``classes`` is given).
    :param colour_table: Optional colour table of band 1: a mapping of pixel
                         value to a hex string or ``(r, g, b[, a])``
                         sequence, as for
                         :func:`~rsgispy.imageutils.colourtable.define_colour_table`.
    :param classes: Optional class names and colours of band 1, a mapping of
                    pixel value to ``(name, colour)`` as for
                    :func:`~rsgispy.imageutils.colourtable.set_thematic_classes`
                    (not with ``colour_table``).

    .. code:: python

        import numpy as np
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.classification.classpostprocess import majority_filter_node

        cls = majority_filter_node(RasterSource("cls.tif"), 3, dtype=np.uint8)
        TileGraph(
            DiskSink(
                cls,
                "cls_smooth.tif",
                classes={1: ("Water", "#1f78b4"), 2: ("Forest", "#33a02c")},
            )
        ).run()
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes

    result_prefix = "disk"

    def __init__(
        self,
        source: TileNode,
        output_img: Union[str, os.PathLike],
        gdalformat: str = "GTIFF",
        datatype: Optional[int] = None,
        nodata: Optional[float] = None,
        clamp_int: bool = True,
        creation_opts: Optional[Mapping[str, Any]] = None,
        band_names: Optional[Sequence[str]] = None,
        calc_stats: Optional[bool] = None,
        name: Optional[str] = None,
        *,
        inherit_meta: bool = True,
        thematic: bool = False,
        colour_table: Optional[Mapping[int, Any]] = None,
        classes: Optional[Mapping[int, Any]] = None,
    ) -> None:
        super().__init__(source, name=name)
        self._cls_names, self._cls_colours = _parse_sink_classes(colour_table, classes)
        self.colour_table = dict(colour_table) if colour_table is not None else None
        self.classes = dict(classes) if classes is not None else None
        # Kept apart from ``thematic``, which subclasses may redefine.
        self._thematic_out = bool(thematic or self._cls_colours)
        self.thematic: bool = self._thematic_out
        self.output_img: str = os.fspath(output_img)
        self.gdalformat: str = str(gdalformat)
        #: The ``datatype`` argument (``None``: from the input's metadata).
        self.datatype: Optional[int] = None if datatype is None else int(datatype)
        self._resolved_datatype: Optional[int] = self.datatype
        self.nodata: Optional[float] = nodata
        _warn_clamp_int(clamp_int, type(self).__name__)
        self.clamp_int: bool = True
        self.creation_opts: Optional[Mapping[str, Any]] = creation_opts
        self.band_names: Optional[List[str]] = (
            list(band_names) if band_names is not None else None
        )
        self.calc_stats: bool = bool(
            self._thematic_out if calc_stats is None else calc_stats
        )
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
        self._resolved_datatype = (
            self.datatype if self.datatype is not None else datatype_from_meta(meta)
        )
        if self._thematic_out:
            self._check_thematic_dtype()
        return meta

    def _check_thematic_dtype(self) -> None:
        """Raise unless the output data type can hold a class map (and its colours)."""
        out_dt = np.dtype(rsgispy.get_numpy_datatype(self.out_datatype))
        if not np.issubdtype(out_dt, np.integer):
            raise rsgispy.RSGISPyException(
                f"DiskSink '{self.output_img}': a thematic output needs an integer "
                f"data type, got {out_dt} (pass datatype=rsgispy.TYPE_8UINT or "
                "Cast the input)."
            )
        if self._cls_colours:
            if out_dt not in (np.dtype(np.uint8), np.dtype(np.uint16)):
                raise rsgispy.RSGISPyException(
                    f"DiskSink '{self.output_img}': a colour table needs a uint8 or "
                    f"uint16 output, got {out_dt}."
                )
            if max(self._cls_colours) > np.iinfo(out_dt).max:
                raise rsgispy.RSGISPyException(
                    f"DiskSink '{self.output_img}': class {max(self._cls_colours)} "
                    f"does not fit a {out_dt} output."
                )

    def _thematic_on_close(
        self, path: str, nodata: Optional[float], is_cog: bool, use_pbar: bool
    ) -> None:
        """Mark ``path`` thematic, write its classes and build its statistics."""
        # pylint: disable=import-outside-toplevel
        from rsgispy.imageutils import colourtable as _ct
        from rsgispy.imageutils.stats import pop_thmt_img_stats

        ignore_zero = nodata is not None and float(nodata) == 0.0
        if self._cls_colours:
            # pylint: disable-next=protected-access
            _ct._apply_thematic_classes(
                path,
                self._cls_names,
                self._cls_colours,
                img_band=1,
                ignore_zero=ignore_zero,
                calc_stats=self.calc_stats,
                calc_pyramids=self.calc_stats and not is_cog,
                ignore_cog=False,
                use_pbar=use_pbar,
            )
        elif self.calc_stats:
            pop_thmt_img_stats(
                path,
                add_clr_tab=True,
                calc_pyramids=not is_cog,
                ignore_zero=ignore_zero,
                use_pbar=use_pbar,
            )
        else:
            _ct.set_img_thematic(path)

    @property
    def out_datatype(self) -> int:
        """
        ``rsgispy.TYPE_*`` written: ``datatype``, or the type resolved from the
        input's metadata when the graph was built.
        """
        if self._resolved_datatype is not None:
            return self._resolved_datatype
        return self.datatype if self.datatype is not None else rsgispy.TYPE_32FLOAT

    def resolve_output_meta(
        self, meta: NodeMeta
    ) -> Tuple[Optional[float], Optional[List[str]]]:
        """
        Return the ``(nodata, band_names)`` written for input metadata
        ``meta`` (explicit values first, then the inherited ones).
        """
        if self.datatype is None:
            self._resolved_datatype = datatype_from_meta(meta)
        out_np_dt = np.dtype(rsgispy.get_numpy_datatype(self.out_datatype))
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
            "np_dtype": np.dtype(rsgispy.get_numpy_datatype(self.out_datatype)),
        }

    def _open_dataset(self, count: int) -> rasterio.io.DatasetWriter:
        # pylint: disable=import-outside-toplevel,too-many-locals
        from rsgispy.imageprocess import (
            get_output_stage,
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
        if self._thematic_out:
            # NEAREST COG overviews; the statistics are built on close below.
            pop_stats_on_close(dst, False, thematic=True, use_pbar=run["use_pbar"])
            stage = get_output_stage(dst)
            is_cog = stage is not None and stage.is_cog
            nodata = run["nodata"]
            use_pbar = run["use_pbar"]
            on_output_close(
                dst,
                lambda path: self._thematic_on_close(path, nodata, is_cog, use_pbar),
            )
        else:
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
        )
        dst.write(cast_blk, window=window)

    def close(self) -> None:
        self._run = None

    def describe_params(self) -> str:
        text = (
            f"output='{self.output_img}' format={self.gdalformat} "
            f"datatype={rsgispy.get_rasterio_datatype(self.out_datatype)}"
            + (" (from input)" if self.datatype is None else "")
        )
        if self._thematic_out:
            text += " thematic"
            if self._cls_names:
                text += f" classes={len(self._cls_names)}"
            elif self._cls_colours:
                text += f" colour_table={len(self._cls_colours)}"
        return text


class MemorySink(TileSink):
    """
    Terminal sink assembling a node's tiled output into an in-memory NumPy
    array accessible on :attr:`result` after :meth:`TileGraph.run`.

    :param source: Upstream :class:`TileNode` producing the output tiles.
    :param dtype: Optional output NumPy dtype (inferred from first tile when
                  ``None``).
    :param clamp_int: Deprecated and ignored: integer outputs are always
                      rounded and clamped to the type's range (``False``
                      issues a :class:`DeprecationWarning`).
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
        _warn_clamp_int(clamp_int, type(self).__name__)
        self.clamp_int: bool = True
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
