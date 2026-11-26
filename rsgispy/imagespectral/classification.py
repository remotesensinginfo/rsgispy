"""
The imagespectral.classification module provides spectral similarity mapping
and classification functions using reference endmember spectra, including
Spectral Angle Mapper (SAM) and Spectral Correlation Mapper (SCM).
"""

from contextlib import ExitStack
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import rsgispy
from rsgispy._tilegraph._wrap import (
    GridRasterSource,
    WrapperDiskSink,
    run_wrapper_graph,
)
from rsgispy.imageprocess import (
    NodeMeta,
    PointwiseOp,
    TileContext,
    TileNode,
    TileTensor,
    ensure_file_exists,
    rsgis_dtype_from_np_dtype,
)
from rsgispy.imagespectral._helpers import open_spectral_inputs, valid_cube_mask
from rsgispy.imagespectral.endmembers import (
    EndmemberInput,
    prepare_endmembers_array,
)

__all__: List[str] = [
    "spectral_angle_mapper",
    "spectral_angle_mapper_node",
    "spectral_correlation_mapper",
    "spectral_correlation_mapper_node",
]


class _SimilaritySetup:
    """Endmembers and settings of a SAM / SCM run."""

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(
        self,
        endmembers: EndmemberInput,
        mode: str,
        endmembers_gain: float,
        file_format: Optional[str],
        score_threshold: Optional[float],
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        em_names, n_em, n_em_bands, em_arr = prepare_endmembers_array(
            endmembers, gain=endmembers_gain, file_format=file_format
        )
        if n_em == 0:
            raise rsgispy.RSGISPyException(
                "Endmembers table must contain at least one class column."
            )
        self.is_scm = mode == "scm"
        if self.is_scm:
            if n_em_bands < 2:
                raise rsgispy.RSGISPyException(
                    "Spectral Correlation Mapper requires at least 2 spectral bands."
                )
            if score_threshold is not None and (
                float(score_threshold) < -1.0 or float(score_threshold) > 1.0
            ):
                raise rsgispy.RSGISPyException(
                    f"min_corr must be within [-1.0, 1.0], got {score_threshold}."
                )
            em_work = em_arr - np.mean(em_arr, axis=1, keepdims=True)
            em_norms = np.linalg.norm(em_work, axis=1, keepdims=True)
            if np.any(em_norms == 0.0) or not np.all(np.isfinite(em_norms)):
                raise rsgispy.RSGISPyException(
                    "All endmember spectra must be finite and have non-zero "
                    "variance across bands."
                )
        else:
            em_work = em_arr
            em_norms = np.linalg.norm(em_work, axis=1, keepdims=True)
            if np.any(em_norms == 0.0) or not np.all(np.isfinite(em_norms)):
                raise rsgispy.RSGISPyException(
                    "All endmember spectra must be finite and have non-zero norm."
                )
        self.em_names = em_names
        self.n_em = n_em
        self.n_em_bands = n_em_bands
        self.em_unit = em_work / em_norms  # (n_em, n_bands)
        self.score_threshold = score_threshold
        self.cls_np_dtype = np.dtype(np.uint8 if n_em <= 255 else np.uint16)


_OUT_FLOAT_ND = -9999.0


class _SimilarityNode(TileNode):
    """
    SAM / SCM node: ``n_em`` score bands, the best score and the class
    (1-based index of the best endmember, ``0`` when not classified), in a
    ``float64`` block of ``n_em + 2`` bands.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        inputs: Sequence[TileNode],
        setup: _SimilaritySetup,
        settings: Tuple[float, bool, int, Optional[float]],
        name: Optional[str] = None,
        summary_name: str = "score",
    ) -> None:
        super().__init__(inputs=tuple(inputs), name=name)
        self.setup = setup
        self.summary_name = summary_name
        self.img_gain, self.degrees, self.valid_msk_val, self.no_data_val = settings
        self.eff_nd: Optional[float] = self.no_data_val

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        n_in = inputs[0].n_bands
        if n_in is not None and int(n_in) != self.setup.n_em_bands:
            raise rsgispy.RSGISPyException(
                f"Input image band count ({n_in}) does not match "
                f"endmember band count ({self.setup.n_em_bands})."
            )
        if self.no_data_val is None:
            self.eff_nd = inputs[0].nodata
        return NodeMeta(
            n_bands=self.setup.n_em + 2,
            band_names=list(self.setup.em_names) + [self.summary_name, "class"],
            dtype=np.float64,
        )

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        # pylint: disable=too-many-locals
        aligned = [t.crop(halo_y, halo_x) for t in inputs]
        setup = self.setup
        n_em = setup.n_em
        blk = np.asarray(aligned[0].data, dtype=np.float64)
        win_h, win_w = int(blk.shape[1]), int(blk.shape[2])
        n_pxls = win_h * win_w
        msk_blk = aligned[1].data[0] if len(aligned) > 1 else None
        vld_flat = valid_cube_mask(blk, self.eff_nd, msk_blk, self.valid_msk_val).ravel()
        scores_out = np.full((n_pxls, n_em), _OUT_FLOAT_ND, dtype=np.float32)
        summary_out = np.full(n_pxls, _OUT_FLOAT_ND, dtype=np.float32)
        cls_out = np.zeros(n_pxls, dtype=setup.cls_np_dtype)
        if np.any(vld_flat):
            pxls = np.moveaxis(blk, 0, -1).reshape(-1, setup.n_em_bands)[vld_flat]
            if self.img_gain > 1:
                pxls = pxls / float(self.img_gain)
            if setup.is_scm:
                pxls_work = pxls - np.mean(pxls, axis=1, keepdims=True)
            else:
                pxls_work = pxls
            norms = np.linalg.norm(pxls_work, axis=1)
            non_zero = norms > 0.0
            if np.any(non_zero):
                nz_unit = pxls_work[non_zero] / norms[non_zero, None]
                cos_sim = np.clip(nz_unit @ setup.em_unit.T, -1.0, 1.0)
                if setup.is_scm:
                    metric_vals = cos_sim
                    best_idx = np.argmax(metric_vals, axis=1)
                else:
                    metric_vals = np.arccos(cos_sim)
                    if self.degrees:
                        metric_vals = np.degrees(metric_vals)
                    best_idx = np.argmin(metric_vals, axis=1)
                nz_indices = np.flatnonzero(vld_flat)[non_zero]
                scores_out[nz_indices] = metric_vals.astype(np.float32)
                best_score = metric_vals[np.arange(metric_vals.shape[0]), best_idx]
                summary_out[nz_indices] = best_score.astype(np.float32)
                cls_vals = (best_idx + 1).astype(setup.cls_np_dtype)
                thresh = setup.score_threshold
                if thresh is not None:
                    keep = (
                        best_score >= float(thresh)
                        if setup.is_scm
                        else best_score <= float(thresh)
                    )
                    cls_vals = np.where(keep, cls_vals, 0).astype(setup.cls_np_dtype)
                cls_out[nz_indices] = cls_vals
        out = np.empty((n_em + 2, win_h, win_w), dtype=np.float64)
        out[:n_em] = np.moveaxis(scores_out.reshape((win_h, win_w, n_em)), -1, 0)
        out[n_em] = summary_out.reshape((win_h, win_w))
        out[n_em + 1] = cls_out.reshape((win_h, win_w))
        return aligned[0].with_data(out)


def _similarity_output(
    node: _SimilarityNode,
    output: str,
    summary_name: str,
    name: Optional[str],
) -> TileNode:
    """Select one output of a :class:`_SimilarityNode` with its metadata."""
    setup = node.setup
    n_em = setup.n_em
    if output == "scores":
        idx = list(range(n_em))
        meta = NodeMeta(n_bands=n_em, band_names=setup.em_names, dtype=np.float32,
                        nodata=_OUT_FLOAT_ND)
    elif output == "best":
        idx = [n_em]
        meta = NodeMeta(n_bands=1, band_names=(summary_name,), dtype=np.float32,
                        nodata=_OUT_FLOAT_ND)
    elif output == "all":
        return node
    elif output == "class":
        idx = [n_em + 1]
        meta = NodeMeta(n_bands=1, dtype=setup.cls_np_dtype, nodata=0.0)
    else:
        raise rsgispy.RSGISPyException(
            f"output must be 'scores', 'best', 'class' or 'all', got {output!r}."
        )
    dtype = meta.dtype

    return PointwiseOp(
        node,
        lambda arr: arr[idx].astype(dtype),
        name=name or f"{node.name}_{output}",
        meta_fn=lambda inputs: meta,
    )


def _similarity_node(
    source: TileNode,
    vld_msk: Optional[TileNode],
    setup: _SimilaritySetup,
    settings: Tuple[float, bool, int, Optional[float]],
    names: Tuple[str, str],
) -> _SimilarityNode:
    """Build a :class:`_SimilarityNode`; ``names`` is ``(name, summary_name)``."""
    inputs = [source] + ([vld_msk] if vld_msk is not None else [])
    return _SimilarityNode(inputs, setup, settings, name=names[0], summary_name=names[1])


def spectral_angle_mapper_node(
    source: TileNode,
    endmembers: EndmemberInput,
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    degrees: bool = False,
    max_angle: Optional[float] = None,
    vld_msk: Optional[TileNode] = None,
    valid_msk_val: int = 1,
    *,
    output: str = "scores",
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`spectral_angle_mapper`, for chaining in a
    :class:`~rsgispy.imageprocess.TileGraph`. ``output`` selects what the
    node gives (the outputs of :func:`spectral_angle_mapper`):

    * ``'scores'``: one ``float32`` band per endmember with the spectral
      angles, named after the endmembers (no data ``-9999``);
    * ``'best'``: the minimum angle (``float32``, band ``'min_angle'``);
    * ``'class'``: the 1-based index of the closest endmember (``0`` for no
      data, zero-norm spectra or angles above ``max_angle``; ``uint8``, or
      ``uint16`` for more than 255 endmembers).

    * ``'all'``: the scores, the best score and the class in one
      ``float64`` block of ``n_endmembers + 2`` bands (select them with
      :class:`~rsgispy.imageprocess.SelectBands` to write several outputs
      from one calculation).

    The other parameters are those of :func:`spectral_angle_mapper`, with
    the valid mask as an optional node ``vld_msk``; ``source`` must have the
    endmembers' bands.

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param endmembers: Endmembers table or file (bands as rows).
    :param img_gain: Gain divisor of ``source`` when ``> 1`` (Default = ``1``).
    :param endmembers_gain: Gain divisor of the endmembers (Default = ``1``).
    :param file_format: Optional table format of an endmembers file.
    :param degrees: Angles in degrees (Default = ``False``: radians).
    :param max_angle: Optional maximum angle for ``'class'``.
    :param vld_msk: Optional node whose first band is a valid-pixel mask.
    :param valid_msk_val: Valid value in ``vld_msk`` (Default = ``1``).
    :param output: ``'scores'`` (Default), ``'best'``, ``'class'`` or ``'all'``.
    :param no_data_val: Optional input no-data value (Default = ``None``: that
                        of ``source``).
    :param name: Optional node name.
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        import rsgispy
        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagespectral.classification import spectral_angle_mapper_node

        refl = RasterSource("refl.tif")
        cls = spectral_angle_mapper_node(
            refl, "endmembers.csv", img_gain=10000, degrees=True, max_angle=15.0,
            output="class",
        )
        TileGraph(DiskSink(cls, "sam_class.tif", datatype=rsgispy.TYPE_8UINT)).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    setup = _SimilaritySetup(endmembers, "sam", endmembers_gain, file_format, max_angle)
    node = _similarity_node(
        source, vld_msk, setup, (img_gain, degrees, valid_msk_val, no_data_val),
        ("spectral_angle_mapper", "min_angle"),
    )
    return _similarity_output(node, output, "min_angle", name)


def spectral_correlation_mapper_node(
    source: TileNode,
    endmembers: EndmemberInput,
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    min_corr: Optional[float] = None,
    vld_msk: Optional[TileNode] = None,
    valid_msk_val: int = 1,
    *,
    output: str = "scores",
    no_data_val: Optional[float] = None,
    name: Optional[str] = None,
) -> TileNode:
    """
    Tile-graph node of :func:`spectral_correlation_mapper`: as
    :func:`spectral_angle_mapper_node`, with the SCM correlations as
    ``'scores'``, the maximum correlation (band ``'max_corr'``) as
    ``'best'``, and for ``'class'`` the most correlated endmember (``0``
    below ``min_corr``).

    :param source: Upstream :class:`~rsgispy.imageprocess.TileNode`.
    :param endmembers: Endmembers table or file (bands as rows).
    :param img_gain: Gain divisor of ``source`` when ``> 1`` (Default = ``1``).
    :param endmembers_gain: Gain divisor of the endmembers (Default = ``1``).
    :param file_format: Optional table format of an endmembers file.
    :param min_corr: Optional minimum correlation in ``[-1, 1]`` for ``'class'``.
    :param vld_msk: Optional node whose first band is a valid-pixel mask.
    :param valid_msk_val: Valid value in ``vld_msk`` (Default = ``1``).
    :param output: ``'scores'`` (Default), ``'best'``, ``'class'`` or ``'all'``.
    :param no_data_val: Optional input no-data value (Default = ``None``: that
                        of ``source``).
    :param name: Optional node name.
    :return: :class:`~rsgispy.imageprocess.TileNode`.

    .. code:: python

        from rsgispy.imageprocess import DiskSink, RasterSource, TileGraph
        from rsgispy.imagespectral.classification import (
            spectral_correlation_mapper_node,
        )

        corr = spectral_correlation_mapper_node(
            RasterSource("refl.tif"), "endmembers.csv", output="best"
        )
        TileGraph(DiskSink(corr, "scm_max_corr.tif")).run()
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    setup = _SimilaritySetup(endmembers, "scm", endmembers_gain, file_format, min_corr)
    node = _similarity_node(
        source, vld_msk, setup, (img_gain, False, valid_msk_val, no_data_val),
        ("spectral_correlation_mapper", "max_corr"),
    )
    return _similarity_output(node, output, "max_corr", name)


def _run_spectral_similarity_mapper(
    input_img: str,
    endmembers: EndmemberInput,
    output_img: str,
    mode: str,
    out_class_img: Optional[str] = None,
    out_summary_img: Optional[str] = None,
    summary_band_name: str = "score",
    gdalformat: str = "GTIFF",
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    degrees: bool = False,
    score_threshold: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """Shared tile-graph runner for SAM and SCM spectral similarity mapping."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    ensure_file_exists(input_img, "Input image")
    setup = _SimilaritySetup(
        endmembers, mode, endmembers_gain, file_format, score_threshold
    )
    with ExitStack() as stack:
        src, msk_ds, grid, img_off, msk_off = open_spectral_inputs(
            stack, input_img, valid_msk_img
        )
        if int(src.count) != setup.n_em_bands:
            raise rsgispy.RSGISPyException(
                f"Input image band count ({src.count}) does not match "
                f"endmember band count ({setup.n_em_bands})."
            )
        eff_nd = no_data_val if no_data_val is not None else src.nodata
        msk_dtype = msk_ds.dtypes[0] if msk_ds is not None else None

    vld_msk = None
    if valid_msk_img is not None and msk_off is not None:
        vld_msk = GridRasterSource(valid_msk_img, grid, msk_off, bands=1, dtype=msk_dtype)
    node = _similarity_node(
        GridRasterSource(input_img, grid, img_off),
        vld_msk,
        setup,
        (img_gain, degrees, valid_msk_val, eff_nd),
        (
            "spectral_angle_mapper" if mode == "sam" else "spectral_correlation_mapper",
            summary_band_name,
        ),
    )
    sinks = [
        WrapperDiskSink(
            _similarity_output(node, "scores", summary_band_name, None),
            output_img,
            gdalformat=gdalformat,
            datatype=rsgispy.TYPE_32FLOAT,
            nodata=_OUT_FLOAT_ND,
            band_names=setup.em_names,
            calc_stats=calc_stats,
        )
    ]
    if out_summary_img is not None:
        sinks.append(
            WrapperDiskSink(
                _similarity_output(node, "best", summary_band_name, None),
                out_summary_img,
                gdalformat=gdalformat,
                datatype=rsgispy.TYPE_32FLOAT,
                nodata=_OUT_FLOAT_ND,
                band_names=[summary_band_name],
                calc_stats=calc_stats,
            )
        )
    if out_class_img is not None:
        sinks.append(
            WrapperDiskSink(
                _similarity_output(node, "class", summary_band_name, None),
                out_class_img,
                gdalformat=gdalformat,
                datatype=rsgis_dtype_from_np_dtype(setup.cls_np_dtype),
                nodata=0,
                calc_stats=calc_stats,
                thematic=True,
            )
        )
    run_wrapper_graph(sinks, use_pbar=use_pbar)


def spectral_angle_mapper(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    out_class_img: Optional[str] = None,
    out_min_angle_img: Optional[str] = None,
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    degrees: bool = False,
    max_angle: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute Spectral Angle Mapper (SAM; Kruse et al., 1993) spectral angles
    between every pixel spectrum in ``input_img`` and reference endmembers,
    processed window-by-window without loading full images into memory.

    For a pixel spectrum :math:`x` and endmember spectrum :math:`e`, the
    spectral angle is
    :math:`\\theta = \\arccos(\\langle x, e \\rangle / (\\|x\\|\\|e\\|))`,
    providing illumination- and gain-invariant similarity scoring in
    ``[0, pi]`` radians (or ``[0, 180]`` degrees when ``degrees=True``).

    :param input_img: Path to the input multi-band reflectance/radiance image.
    :param output_img: Path to the output ``float32`` multi-band spectral angle
                       raster (one band per endmember class).
    :param endmembers: :class:`pandas.DataFrame` (bands as rows, endmember class
                       names as columns) or path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param out_class_img: Optional output thematic classification raster path
                          assigning each pixel the 1-based index of the
                          smallest spectral angle (``0`` for nodata/zero-norm
                          or when exceeding ``max_angle``).
    :param out_min_angle_img: Optional single-band ``float32`` output raster
                              recording the minimum spectral angle per pixel.
    :param img_gain: Gain divisor applied to ``input_img`` when ``> 1``
                     (Default = ``1``).
    :param endmembers_gain: Gain divisor applied to ``endmembers`` when ``> 1``
                            (Default = ``1``).
    :param file_format: Optional table format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param degrees: If ``True``, output angles (and interpret ``max_angle``) in
                    degrees rather than radians (Default = ``False``).
    :param max_angle: Optional maximum spectral angle threshold for
                      ``out_class_img``; pixels whose minimum angle exceeds
                      ``max_angle`` are assigned class ``0`` (unclassified).
    :param valid_msk_img: Optional single-band valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute statistics and pyramids on output
                       rasters (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.classification as sc

        sc.spectral_angle_mapper(
            "refl.tif",
            "sam_angles.tif",
            "endmembers.csv",
            out_class_img="sam_class.tif",
            out_min_angle_img="sam_min_angle.tif",
            degrees=True,
            max_angle=15.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_spectral_similarity_mapper(
        input_img=input_img,
        endmembers=endmembers,
        output_img=output_img,
        mode="sam",
        out_class_img=out_class_img,
        out_summary_img=out_min_angle_img,
        summary_band_name="min_angle",
        gdalformat=gdalformat,
        img_gain=img_gain,
        endmembers_gain=endmembers_gain,
        file_format=file_format,
        degrees=degrees,
        score_threshold=max_angle,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )


def spectral_correlation_mapper(
    input_img: str,
    output_img: str,
    endmembers: EndmemberInput,
    out_class_img: Optional[str] = None,
    out_max_corr_img: Optional[str] = None,
    img_gain: float = 1,
    endmembers_gain: float = 1,
    file_format: Optional[str] = None,
    min_corr: Optional[float] = None,
    valid_msk_img: Optional[str] = None,
    valid_msk_val: int = 1,
    *,
    gdalformat: str = "GTIFF",
    no_data_val: Optional[float] = None,
    calc_stats: bool = True,
    use_pbar: bool = True,
) -> None:
    """
    Compute Spectral Correlation Mapper (SCM; de Carvalho & Meneses, 2000)
    Pearson correlation scores between every pixel spectrum in ``input_img``
    and reference endmembers, processed window-by-window.

    Unlike :func:`spectral_angle_mapper`, SCM centres each spectrum by its
    band mean before computing the cosine similarity:
    :math:`r = \\langle x - \\bar{x}, e - \\bar{e} \\rangle / `
    :math:`(\\|x - \\bar{x}\\|\\|e - \\bar{e}\\|) \\in [-1, 1]`.
    This makes SCM invariant to both positive multiplicative scaling and
    additive illumination/offset shifts across bands (higher correlation values
    indicate closer spectral match).

    :param input_img: Path to the input multi-band reflectance/radiance image
                      (requires at least 2 spectral bands).
    :param output_img: Path to the output ``float32`` multi-band SCM correlation
                       raster (values in ``[-1, 1]``, one band per endmember
                       class).
    :param endmembers: :class:`pandas.DataFrame` (bands as rows, endmember class
                       names as columns) or path readable by
                       :func:`rsgispy.imagespectral.endmembers.read_endmembers_df`.
    :param out_class_img: Optional output thematic classification raster path
                          assigning each pixel the 1-based index of the
                          highest SCM correlation (``0`` for nodata/zero-variance
                          spectra or when below ``min_corr``).
    :param out_max_corr_img: Optional single-band ``float32`` output raster
                             recording the maximum SCM correlation per pixel.
    :param img_gain: Gain divisor applied to ``input_img`` when ``> 1``
                     (Default = ``1``).
    :param endmembers_gain: Gain divisor applied to ``endmembers`` when ``> 1``
                            (Default = ``1``).
    :param file_format: Optional table format override when ``endmembers`` is a
                        file path (Default = ``None``).
    :param min_corr: Optional minimum correlation threshold in ``[-1.0, 1.0]``
                     for ``out_class_img``; pixels whose maximum correlation is
                     strictly below ``min_corr`` are assigned class ``0``
                     (unclassified).
    :param valid_msk_img: Optional single-band valid-region mask raster file.
    :param valid_msk_val: Pixel value identifying valid pixels in
                          ``valid_msk_img`` (Default = ``1``).
    :param gdalformat: Output GDAL format name (Default = ``'GTIFF'``).
    :param no_data_val: Explicit input nodata value overriding metadata
                        (Default = ``None``).
    :param calc_stats: If ``True``, compute statistics and pyramids on output
                       rasters (Default = ``True``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).

    .. code:: python

        import rsgispy.imagespectral.classification as sc

        sc.spectral_correlation_mapper(
            "refl.tif",
            "scm_corr.tif",
            "endmembers.csv",
            out_class_img="scm_class.tif",
            out_max_corr_img="scm_max_corr.tif",
            min_corr=0.85,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    _run_spectral_similarity_mapper(
        input_img=input_img,
        endmembers=endmembers,
        output_img=output_img,
        mode="scm",
        out_class_img=out_class_img,
        out_summary_img=out_max_corr_img,
        summary_band_name="max_corr",
        gdalformat=gdalformat,
        img_gain=img_gain,
        endmembers_gain=endmembers_gain,
        file_format=file_format,
        degrees=False,
        score_threshold=min_corr,
        valid_msk_img=valid_msk_img,
        valid_msk_val=valid_msk_val,
        no_data_val=no_data_val,
        calc_stats=calc_stats,
        use_pbar=use_pbar,
    )
