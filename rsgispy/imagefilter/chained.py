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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

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
)
from rsgispy.imagefilter._helper import (
    _resolve_out_nodata,
    gaussian_kernel,
    gradient_magnitude_node,
    kernel_conv_node,
    laplacian_kernel,
    make_lee_stat_fn,
    prewitt_kernels,
    sobel_kernels,
    stat_coeff_of_var,
    stat_kuwahara,
    stat_max,
    stat_mean,
    stat_mean_diff,
    stat_mean_diff_abs,
    stat_median,
    stat_min,
    stat_mode,
    stat_norm_var,
    stat_norm_var_ln,
    stat_norm_var_sqrt,
    stat_range,
    stat_stddev,
    stat_texture_var,
    stat_total,
    stat_total_diff,
    stat_total_diff_abs,
    window_stat_node,
)
from rsgispy.imageprocess import ensure_file_exists
from rsgispy.imageutils.metadata import get_band_names, get_img_no_data_value

__all__ = [
    "FilterStep",
    "build_filter_chain_node",
    "apply_filter_chain",
]

_STAT_DISPATCH: Dict[str, Any] = {
    "mean": stat_mean,
    "median": stat_median,
    "min": stat_min,
    "max": stat_max,
    "mode": stat_mode,
    "range": stat_range,
    "total": stat_total,
    "stddev": stat_stddev,
    "variance": stat_texture_var,
    "coeff_of_var": stat_coeff_of_var,
    "norm_var": stat_norm_var,
    "norm_var_sqrt": stat_norm_var_sqrt,
    "norm_var_ln": stat_norm_var_ln,
    "mean_diff": stat_mean_diff,
    "mean_diff_abs": stat_mean_diff_abs,
    "total_diff": stat_total_diff,
    "total_diff_abs": stat_total_diff_abs,
    "kuwahara": stat_kuwahara,
}


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
    * SAR speckle filter: ``'lee'`` (uses ``filter_size`` and ``params={'n_looks': ...}``).
    * Convolution / smoothing / edge: ``'gaussian'`` (``filter_size``,
      ``params={'stddev': 1.0}`` or ``'stddev_x'``/``'stddev_y'``),
      ``'laplacian'`` (``filter_size``, ``params={'stddev': 1.0}``),
      ``'sobel'``, ``'prewitt'``, ``'kernel'`` (``params={'kernel': arr, 'zero_dc': False}``).
    * Point-wise / band-math: ``'band_math'`` (``params={'expr': 'input - x'}``,
      where ``x`` is the current stage and ``input`` is the original raster),
      ``'pointwise'`` (``params={'func': callable}``).

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
    :param no_data_val: Optional nodata value honoured by windowed and
                        band-math operations.
    :return: Terminal :class:`~rsgispy._io.TileNode` of the fused chain.
    """
    # pylint: disable=too-many-branches
    if not steps:
        raise rsgispy.RSGISPyException(
            "apply_filter_chain requires at least one FilterStep."
        )

    current: TileNode = source
    named_nodes: Dict[str, TileNode] = {"input": source}
    border = HaloBorder("edge")

    for idx, raw_step in enumerate(steps):
        step = _coerce_step(raw_step)
        kind = step.kind.strip().lower()
        step_name = step.name or f"step_{idx}_{kind}"

        if kind in _STAT_DISPATCH:
            current = window_stat_node(
                source=current,
                filter_size=step.filter_size,
                stat_fn=_STAT_DISPATCH[kind],
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "lee":
            n_looks = float(step.params.get("n_looks", 1.0))
            current = window_stat_node(
                source=current,
                filter_size=step.filter_size,
                stat_fn=make_lee_stat_fn(n_looks),
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "gaussian":
            sd = float(step.params.get("stddev", 1.0))
            sd_x = float(step.params.get("stddev_x", sd))
            sd_y = float(step.params.get("stddev_y", sd))
            angle = float(step.params.get("filter_angle", 0.0))
            kern = gaussian_kernel(step.filter_size, sd_x, sd_y, angle)
            current = kernel_conv_node(
                source=current,
                kernel=kern,
                zero_dc=False,
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "laplacian":
            sd = float(step.params.get("stddev", 1.0))
            kern = laplacian_kernel(step.filter_size, sd)
            current = kernel_conv_node(
                source=current,
                kernel=kern,
                zero_dc=True,
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "sobel":
            kx, ky = sobel_kernels()
            current = gradient_magnitude_node(
                source=current,
                kernel_x=kx,
                kernel_y=ky,
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "prewitt":
            kx, ky = prewitt_kernels()
            current = gradient_magnitude_node(
                source=current,
                kernel_x=kx,
                kernel_y=ky,
                no_data_val=no_data_val,
                border=border,
                name=step_name,
            )
        elif kind == "kernel":
            if "kernel" not in step.params:
                raise rsgispy.RSGISPyException(
                    "FilterStep(kind='kernel') requires params['kernel']."
                )
            current = kernel_conv_node(
                source=current,
                kernel=np.asarray(step.params["kernel"], dtype=np.float64),
                zero_dc=bool(step.params.get("zero_dc", False)),
                no_data_val=no_data_val,
                border=border,
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
    block_size: int = 256,
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
    :param block_size: Core processing tile size in pixels (Default = ``256``).
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
    band_names = get_band_names(in_path)

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
        def _remap_nd(arr: np.ndarray) -> np.ndarray:
            return np.where(
                (~np.isfinite(arr)) | (arr == eff_in_nd),
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
        clamp_int=True,
        creation_opts=creation_opts,
        band_names=band_names,
        calc_stats=calc_stats,
    )
    TileGraph(sinks=sink, block_size=block_size).run(use_pbar=use_pbar)
