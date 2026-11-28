"""
In-memory halo-fused multi-step filter pipelines for :mod:`rsgispy.imagefilter`.

:func:`apply_filter_chain` fuses an arbitrary sequence of spatial moving-window
filters, convolution kernels, edge detectors, SAR speckle filters, and
point-wise / ``pymu-eval`` band-math expressions into a single tiled
:class:`~rsgispy._io.TileGraph` pass with cumulative halo propagation and zero
intermediate disk writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import rsgispy
from rsgispy._io import (
    BandMathOp,
    DiskSink,
    HaloBorder,
    PointwiseOp,
    RasterSource,
    TileGraph,
    TileNode,
    resolve_input_nodata,
)
from rsgispy.imagefilter._helpers import _resolve_out_nodata
from rsgispy.imagefilter import edge as _edge
from rsgispy.imagefilter import sar as _sar
from rsgispy.imagefilter import smooth as _smooth
from rsgispy.imagefilter import stats as _stats
from rsgispy.imagefilter import texture as _texture
from rsgispy.imageprocess import ensure_file_exists, get_default_block_size
from rsgispy.imageutils.metadata import get_band_names, get_img_no_data_value

__all__ = [
    "FilterStep",
    "build_filter_chain_node",
    "apply_filter_chain",
]

#: Window-statistic ``FilterStep`` kinds and their node builders (all take
#: ``(source, filter_size, *, no_data_val, name)``).
_STAT_DISPATCH: Dict[str, Callable[..., TileNode]] = {
    "mean": _stats.mean_filter_node,
    "median": _stats.median_filter_node,
    "min": _stats.min_filter_node,
    "max": _stats.max_filter_node,
    "mode": _stats.mode_filter_node,
    "range": _stats.range_filter_node,
    "total": _stats.total_filter_node,
    "stddev": _stats.stddev_filter_node,
    "variance": _stats.texture_var_filter_node,
    "coeff_of_var": _stats.coeff_of_var_filter_node,
    "norm_var": _stats.norm_var_filter_node,
    "norm_var_sqrt": _stats.norm_var_sqrt_filter_node,
    "norm_var_ln": _stats.norm_var_ln_filter_node,
    "mean_diff": _stats.mean_diff_filter_node,
    "mean_diff_abs": _stats.mean_diff_abs_filter_node,
    "total_diff": _stats.total_diff_filter_node,
    "total_diff_abs": _stats.total_diff_abs_filter_node,
    "kuwahara": _smooth.kuwahara_filter_node,
}

#: SAR speckle-filter ``FilterStep`` kinds: node builder and the float
#: ``params`` it takes (each defaulting to ``1.0``), passed after
#: ``filter_size``.
_SAR_DISPATCH: Dict[str, Tuple[Callable[..., TileNode], Tuple[str, ...]]] = {
    "lee": (_sar.lee_filter_node, ("n_looks",)),
    "enhanced_lee": (_sar.enhanced_lee_filter_node, ("n_looks", "damping")),
    "frost": (_sar.frost_filter_node, ("damping",)),
    "enhanced_frost": (_sar.enhanced_frost_filter_node, ("n_looks", "damping")),
    "kuan": (_sar.kuan_filter_node, ("n_looks",)),
    "gamma_map": (_sar.gamma_map_filter_node, ("n_looks",)),
    "refined_lee": (_sar.refined_lee_filter_node, ("n_looks",)),
}

#: Fixed 3x3 edge ``FilterStep`` kinds (no parameters).
_EDGE_DISPATCH: Dict[str, Callable[..., TileNode]] = {
    "sobel": _edge.sobel_filter_node,
    "sobel_x": _edge.sobel_x_filter_node,
    "sobel_y": _edge.sobel_y_filter_node,
    "prewitt": _edge.prewitt_filter_node,
    "prewitt_x": _edge.prewitt_x_filter_node,
    "prewitt_y": _edge.prewitt_y_filter_node,
}

#: Oriented Gaussian ``FilterStep`` kinds (``filter_size`` and
#: ``params={'stddev': ...}`` or ``'stddev_x'``/``'stddev_y'``,
#: ``'filter_angle'``).
_GAUSS_DISPATCH: Dict[str, Callable[..., TileNode]] = {
    "gaussian": _smooth.gaussian_smooth_filter_node,
    "gaussian_1st_deriv": _edge.gaussian_1st_deriv_filter_node,
    "gaussian_2nd_deriv": _edge.gaussian_2nd_deriv_filter_node,
}

#: ``params`` keys passed to :func:`rsgispy.imagefilter.texture.glcm_filter_node`.
_GLCM_PARAMS = (
    "properties",
    "distances",
    "angles",
    "n_levels",
    "min_val",
    "max_val",
    "symmetric",
    "average_angles",
    "band_bounds",
)


@dataclass(frozen=True)
class FilterStep:
    """
    Specification of one stage in an in-memory :func:`apply_filter_chain`
    pipeline.

    Supported ``kind`` values:

    * Window statistics (use ``filter_size``): ``'mean'``, ``'median'``,
      ``'min'``, ``'max'``, ``'mode'``, ``'range'``, ``'total'``, ``'stddev'``,
      ``'variance'``, ``'coeff_of_var'``, ``'norm_var'``, ``'norm_var_sqrt'``,
      ``'norm_var_ln'``, ``'mean_diff'``, ``'mean_diff_abs'``, ``'total_diff'``,
      ``'total_diff_abs'``, ``'kuwahara'``.
    * SAR speckle filters (use ``filter_size``): ``'lee'``, ``'kuan'``,
      ``'gamma_map'``, ``'refined_lee'`` (``params={'n_looks': ...}``),
      ``'enhanced_lee'``, ``'enhanced_frost'`` (``params={'n_looks': ...,
      'damping': ...}``) and ``'frost'`` (``params={'damping': ...}``); each
      parameter defaults to ``1.0``.
    * Convolution / smoothing: ``'gaussian'`` (``filter_size``,
      ``params={'stddev': 1.0}`` or ``'stddev_x'``/``'stddev_y'``, and
      ``'filter_angle'``), ``'kernel'`` (true convolution,
      ``params={'kernel': arr, 'zero_dc': False}``) and
      ``'kernel_correlation'`` (the same parameters, kernel applied
      without flipping).
    * Edges and derivatives: ``'sobel'``, ``'prewitt'`` (magnitudes),
      ``'sobel_x'``, ``'sobel_y'``, ``'prewitt_x'``, ``'prewitt_y'``
      (no parameters), ``'gaussian_1st_deriv'``, ``'gaussian_2nd_deriv'``
      (``filter_size`` and the ``'gaussian'`` parameters), ``'laplacian'``
      (``filter_size``, ``params={'stddev': 1.0}``).
    * Texture: ``'glcm'`` (``filter_size`` and the
      :func:`~rsgispy.imagefilter.texture.glcm_filter_node` parameters in
      ``params``: ``'properties'``, ``'distances'``, ``'angles'``,
      ``'n_levels'``, ``'symmetric'``, ``'average_angles'``, and the
      grey-level range ``'min_val'`` / ``'max_val'`` or ``'band_bounds'``,
      one of which is required).
    * Point-wise / band-math: ``'band_math'`` (``params={'expr': 'input - x'}``,
      where ``x`` is the current stage and ``input`` is the original raster),
      ``'pointwise'`` (``params={'func': callable}``).

    Each filter kind builds its node with the public node builder of the
    filter, named after the file-based function (e.g. ``'median'`` calls
    :func:`rsgispy.imagefilter.stats.median_filter_node`, ``'variance'``
    :func:`~rsgispy.imagefilter.stats.texture_var_filter_node`, ``'kuwahara'``
    :func:`~rsgispy.imagefilter.smooth.kuwahara_filter_node`, ``'gaussian'``
    :func:`~rsgispy.imagefilter.smooth.gaussian_smooth_filter_node`,
    ``'kernel'`` :func:`~rsgispy.imagefilter.smooth.kernel_filter_node`,
    ``'kernel_correlation'``
    :func:`~rsgispy.imagefilter.smooth.kernel_correlation_filter_node`,
    ``'sobel_x'`` :func:`~rsgispy.imagefilter.edge.sobel_x_filter_node`,
    ``'refined_lee'`` :func:`~rsgispy.imagefilter.sar.refined_lee_filter_node`
    and ``'glcm'`` :func:`~rsgispy.imagefilter.texture.glcm_filter_node`), so
    a chain gives the same result as the builders used directly.

    :param kind: Filter or operation name.
    :param filter_size: Odd window size ``>= 3`` for windowed filters
                        (Default = ``3``).
    :param params: Optional dictionary of algorithm-specific parameters.
    :param name: Optional stage name (also exposed as a variable to downstream
                 ``'band_math'`` steps).
    """

    kind: str
    filter_size: int = 3
    params: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None


def _coerce_step(step: Union[FilterStep, Mapping[str, Any]]) -> FilterStep:
    """Normalise a :class:`FilterStep` or mapping dict into a :class:`FilterStep`."""
    if isinstance(step, FilterStep):
        return step
    if isinstance(step, Mapping):
        if "kind" not in step:
            raise rsgispy.RSGISPyException(
                "Filter step mapping must contain a 'kind' key."
            )
        extra = {
            k: v
            for k, v in step.items()
            if k not in ("kind", "filter_size", "params", "name")
        }
        merged_params = dict(step.get("params") or {})
        merged_params.update(extra)
        return FilterStep(
            kind=str(step["kind"]),
            filter_size=int(step.get("filter_size", 3)),
            params=merged_params,
            name=step.get("name"),
        )
    raise rsgispy.RSGISPyException(
        f"Filter step must be a FilterStep or mapping, got {type(step).__name__}."
    )


def build_filter_chain_node(
    source: TileNode,
    steps: Sequence[Union[FilterStep, Mapping[str, Any]]],
    *,
    no_data_val: Optional[float] = None,
) -> TileNode:
    """
    Construct a fused :class:`~rsgispy._io.TileNode` DAG for ``steps`` starting
    from ``source``.

    :param source: Upstream :class:`~rsgispy._io.TileNode` (e.g.
                   :class:`~rsgispy._io.RasterSource` or
                   :class:`~rsgispy._io.ArraySource`).
    :param steps: Non-empty sequence of :class:`FilterStep` objects or dicts.
    :param no_data_val: Optional no-data value honoured by the windowed and
                        band-math steps (Default = ``None``: the no-data value
                        of ``source``'s metadata, see :ref:`tilegraph-nodata`;
                        :func:`apply_filter_chain` passes the input image's).
    :return: Terminal :class:`~rsgispy._io.TileNode` of the fused chain.
    """
    # pylint: disable=too-many-branches,too-many-locals
    if not steps:
        raise rsgispy.RSGISPyException(
            "apply_filter_chain requires at least one FilterStep."
        )
    no_data_val = resolve_input_nodata(source, no_data_val)

    current: TileNode = source
    named_nodes: Dict[str, TileNode] = {"input": source}

    for idx, raw_step in enumerate(steps):
        step = _coerce_step(raw_step)
        kind = step.kind.strip().lower()
        step_name = step.name or f"step_{idx}_{kind}"

        if kind in _STAT_DISPATCH:
            current = _STAT_DISPATCH[kind](
                current,
                step.filter_size,
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind in _SAR_DISPATCH:
            sar_builder, sar_keys = _SAR_DISPATCH[kind]
            current = sar_builder(
                current,
                step.filter_size,
                **{key: float(step.params.get(key, 1.0)) for key in sar_keys},
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind in _GAUSS_DISPATCH:
            sd = float(step.params.get("stddev", 1.0))
            current = _GAUSS_DISPATCH[kind](
                current,
                step.filter_size,
                stddev_x=float(step.params.get("stddev_x", sd)),
                stddev_y=float(step.params.get("stddev_y", sd)),
                filter_angle=float(step.params.get("filter_angle", 0.0)),
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind == "laplacian":
            current = _edge.laplacian_filter_node(
                current,
                step.filter_size,
                stddev=float(step.params.get("stddev", 1.0)),
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind in _EDGE_DISPATCH:
            current = _EDGE_DISPATCH[kind](
                current, no_data_val=no_data_val, name=step_name
            )
        elif kind == "glcm":
            current = _texture.glcm_filter_node(
                current,
                filter_size=step.filter_size,
                **{k: v for k, v in step.params.items() if k in _GLCM_PARAMS},
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind in ("kernel", "kernel_correlation"):
            if "kernel" not in step.params:
                raise rsgispy.RSGISPyException(
                    f"FilterStep(kind='{kind}') requires params['kernel']."
                )
            kernel_builder = (
                _smooth.kernel_filter_node
                if kind == "kernel"
                else _smooth.kernel_correlation_filter_node
            )
            current = kernel_builder(
                current,
                np.asarray(step.params["kernel"], dtype=np.float64),
                zero_dc=bool(step.params.get("zero_dc", False)),
                no_data_val=no_data_val,
                name=step_name,
            )
        elif kind == "band_math":
            expr = step.params.get("expr")
            if not expr:
                raise rsgispy.RSGISPyException(
                    "FilterStep(kind='band_math') requires params['expr']."
                )
            env_nodes: Dict[str, TileNode] = dict(named_nodes)
            env_nodes["x"] = current
            current = BandMathOp(
                inputs=env_nodes,
                expr=str(expr),
                nodata=no_data_val,
                name=step_name,
            )
        elif kind == "pointwise":
            func: Optional[Callable[..., np.ndarray]] = step.params.get("func")
            if func is None or not callable(func):
                raise rsgispy.RSGISPyException(
                    "FilterStep(kind='pointwise') requires callable params['func']."
                )
            current = PointwiseOp(inputs=current, func=func, name=step_name)
        else:
            raise rsgispy.RSGISPyException(
                f"Unsupported FilterStep kind '{step.kind}'."
            )

        if step.name:
            named_nodes[step.name] = current

    return current


def apply_filter_chain(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    steps: Sequence[Union[FilterStep, Mapping[str, Any]]],
    block_size: Optional[int] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Execute a multi-step sequence of spatial filters and point-wise band-math
    operations per tile in memory without writing intermediate files to disk.

    :param input_img: Path to the input raster image.
    :param output_img: Path to the output raster image.
    :param steps: Ordered sequence of :class:`FilterStep` instances or dicts
                  describing each stage (e.g. Gaussian smoothing followed by
                  Sobel edge detection, or unsharp masking via
                  ``FilterStep('band_math', params={'expr': 'input - x'})``).
    :param block_size: Core processing tile size in pixels (Default = ``None``,
                       the package default block size
                       :func:`rsgispy.imageprocess.get_default_block_size`;
                       this was a fixed ``256`` before).
    :param gdalformat: GDAL/rasterio output driver format (Default = ``'GTIFF'``).
    :param datatype: ``rsgispy.TYPE_*`` constant (Default = ``None``, resolves
                     to :data:`rsgispy.TYPE_32FLOAT`).
    :param no_data_val: Optional no-data value (read from ``input_img`` header
                        when ``None``).
    :param calc_stats: If ``True`` (default), populate image statistics and
                       pyramids on ``output_img``.
    :param use_pbar: If ``True`` (default), display a ``tqdm`` progress bar.
    :param creation_opts: Optional GDAL/rasterio creation options mapping.

    .. code:: python

        import rsgispy.imagefilter.chained as ifc

        ifc.apply_filter_chain(
            "image.tif",
            "unsharp_edges.tif",
            steps=[
                ifc.FilterStep("gaussian", filter_size=5, params={"stddev": 1.2}),
                ifc.FilterStep("sobel"),
            ],
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    ensure_file_exists(in_path)

    eff_dtype = rsgispy.TYPE_32FLOAT if datatype is None else int(datatype)
    out_np_dtype = np.dtype(rsgispy.get_numpy_datatype(eff_dtype))
    eff_in_nd = (
        float(no_data_val)
        if no_data_val is not None
        else get_img_no_data_value(in_path)
    )
    out_nd = _resolve_out_nodata(eff_in_nd, out_np_dtype)
    band_names: Optional[Sequence[str]] = get_band_names(in_path)
    if any(_coerce_step(step).kind.strip().lower() == "glcm" for step in steps):
        # GLCM changes the bands: use the names derived by the nodes.
        band_names = None

    src_node = RasterSource(
        path=in_path,
        dtype=np.float64,
        border=HaloBorder("edge"),
        nodata=eff_in_nd,
        name="input",
    )
    tail = build_filter_chain_node(
        source=src_node,
        steps=steps,
        no_data_val=eff_in_nd,
    )
    if out_nd is not None and eff_in_nd is not None and out_nd != eff_in_nd:
        from rsgispy.imageprocess import build_valid_mask  # pylint: disable=import-outside-toplevel

        def _remap_nd(arr: np.ndarray) -> np.ndarray:
            return np.where(
                ~build_valid_mask(arr, no_data_val=float(eff_in_nd)),
                float(out_nd),
                arr,
            )

        tail = PointwiseOp(tail, _remap_nd, name="remap_nodata")

    sink = DiskSink(
        source=tail,
        output_img=out_path,
        gdalformat=gdalformat,
        datatype=eff_dtype,
        nodata=out_nd,
        creation_opts=creation_opts,
        band_names=band_names,
        calc_stats=calc_stats,
    )
    blk_sz = get_default_block_size() if block_size is None else int(block_size)
    TileGraph(sinks=sink, block_size=blk_sz).run(use_pbar=use_pbar)
