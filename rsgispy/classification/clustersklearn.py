"""
Scikit-learn unsupervised pixel classification (clustering) for raster imagery.

Provides two approaches matching :mod:`rsgislib.classification.clustersklearn`
while using the same :class:`~rsgispy.classification.classdata.ImageBandInfo`
and :class:`~rsgispy.classification.classdata.ClassificationConfig` image input
interface as :mod:`rsgispy.classification.classsklearn`:

1. :func:`img_pixel_cluster` — Loads all valid pixels from the input image(s)
   into memory and fits the ``scikit-learn`` clusterer across the entire pixel
   dataset at once. This supports all ``scikit-learn`` clustering algorithms
   (including transductive estimators without ``predict()``, such as ``DBSCAN``,
   ``OPTICS``, ``AgglomerativeClustering``, and ``SpectralClustering``) with
   no tile boundary artefacts, but can be slow and memory-intensive on large
   images.
2. :func:`img_pixel_sample_cluster` — Extracts a sample of valid pixel values
   from the input image(s), fits the ``scikit-learn`` clusterer on that subset,
   and then applies the fitted model block-by-block across the whole image
   using ``predict()``. This is substantially faster and much more
   memory-efficient for large datasets while remaining free of tile boundary
   artefacts.
3. :func:`apply_img_pixel_clusterer` — Applies a clusterer fitted by
   :func:`img_pixel_sample_cluster` (or saved to a ``.joblib`` file) to
   another image, block by block, with the same prediction code.
"""

from contextlib import ExitStack
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

import rsgispy
from rsgispy.classification._helpers import (
    check_sklearn_available,
    finalise_thematic_output_img,
    resolve_apply_band_specs,
    resolve_out_nodata,
    select_out_int_dtype,
)
from rsgispy.tools._ml_data import (
    read_feature_block,
    resolve_rng,
)
from rsgispy.classification.classdata import (
    ClassificationConfig,
    ImageBandInfo,
    read_cls_config,
)
from rsgispy.classification.classsklearn import (
    load_sklearn_classifier,
    save_sklearn_classifier,
)
from rsgispy.imageprocess import (
    IntersectionGrid,
    compute_spatial_intersection,
    ensure_file_exists,
    extract_roi_feature_matrix,
    grids_compatible,
    iter_dataset_windows,
    open_output_dataset,
    validate_band_index,
)
from rsgispy.imageprocess import normalise_creation_options
from rsgispy.imageutils.sampling import extract_img_pxl_sample
from rsgispy.tools._sklearn import (
    check_sklearn_estimator_fitted,
    clone_sklearn_estimator,
    fit_clean_sklearn_estimator,
    sklearn,
    sync_sklearn_estimator_state,
)
from rsgispy.tools.files import ensure_parent_dir

__all__ = [
    "ClusterImageInput",
    "img_pixel_cluster",
    "img_pixel_sample_cluster",
    "apply_img_pixel_clusterer",
]

ClusterImageInput = Union[
    ClassificationConfig,
    ImageBandInfo,
    Sequence[ImageBandInfo],
    str,
    os.PathLike,
]


def _resolve_cluster_band_specs(
    img_band_info: ClusterImageInput,
) -> List[Tuple[str, int, str]]:
    """
    Resolve ordered ``(img_path, band_1based, col_name)`` feature specs.

    A path ending in ``.json`` is read as a classification config; any other
    path is an image whose bands are all used.
    """
    if isinstance(img_band_info, ClassificationConfig):
        return resolve_apply_band_specs(cfg=img_band_info, img_band_info=None)
    if isinstance(img_band_info, ImageBandInfo):
        return resolve_apply_band_specs(cfg=None, img_band_info=[img_band_info])
    if isinstance(img_band_info, (str, os.PathLike)):
        in_path = os.fspath(img_band_info)
        if os.path.splitext(in_path)[1].lower() == ".json":
            cfg = read_cls_config(in_path)
            return resolve_apply_band_specs(cfg=cfg, img_band_info=None)
        ensure_file_exists(in_path, label="Input image")
        return resolve_apply_band_specs(
            cfg=None, img_band_info=[ImageBandInfo(in_path, bands=None)]
        )
    return resolve_apply_band_specs(cfg=None, img_band_info=img_band_info)


def _validate_cluster_options(
    clusterer: Optional[Any],
    use_mean_shift_est_band_width: bool,
    out_id_offset: int,
    require_predict: bool,
    func_name: str,
) -> None:
    """Validate clusterer object and output ID offset arguments."""
    if isinstance(out_id_offset, bool) or not isinstance(
        out_id_offset, (int, np.integer)
    ):
        raise rsgispy.RSGISPyException(
            f"out_id_offset must be an integer, got {out_id_offset!r}."
        )
    if int(out_id_offset) < 0:
        raise rsgispy.RSGISPyException(
            "out_id_offset must be non-negative so output cluster IDs remain "
            "unsigned integers."
        )

    if use_mean_shift_est_band_width:
        if clusterer is not None:
            raise rsgispy.RSGISPyException(
                "clusterer must be None when use_mean_shift_est_band_width=True."
            )
        return

    if clusterer is None:
        raise rsgispy.RSGISPyException(
            f"{func_name} requires a scikit-learn clusterer instance (or set "
            "use_mean_shift_est_band_width=True with clusterer=None)."
        )

    if not hasattr(clusterer, "fit"):
        raise rsgispy.RSGISPyException(
            "clusterer must implement a fit() method."
        )
    if require_predict and not hasattr(clusterer, "predict"):
        raise rsgispy.RSGISPyException(
            f"{func_name} requires a clusterer implementing predict(). Use "
            "img_pixel_cluster() for clustering algorithms without predict()."
        )


def _open_cluster_datasets(
    stack: ExitStack,
    band_specs: Sequence[Tuple[str, int, str]],
    in_msk_img: Optional[Union[str, os.PathLike]],
) -> Tuple[
    Dict[str, DatasetReader],
    DatasetReader,
    str,
    Optional[DatasetReader],
    Optional[str],
    IntersectionGrid,
    bool,
]:
    """Open feature (and optional mask) datasets and compute spatial grid."""
    datasets: Dict[str, DatasetReader] = {}
    first_img_path = band_specs[0][0]
    for img_path, band_idx, _ in band_specs:
        if img_path not in datasets:
            datasets[img_path] = stack.enter_context(
                rasterio.open(img_path, "r")
            )
        validate_band_index(band_idx, int(datasets[img_path].count), img_path)

    msk_ds: Optional[DatasetReader] = None
    msk_path: Optional[str] = None
    if in_msk_img is not None:
        msk_path = os.fspath(in_msk_img)
        ensure_file_exists(msk_path, label="Input mask image")
        msk_ds = stack.enter_context(rasterio.open(msk_path, "r"))
        datasets[msk_path] = msk_ds
        ref_ds = msk_ds
        ref_path = msk_path
    else:
        ref_ds = datasets[first_img_path]
        ref_path = first_img_path

    all_compatible = all(grids_compatible(ref_ds, ds) for ds in datasets.values())
    if all_compatible:
        grid = compute_spatial_intersection(datasets)
    else:
        grid = IntersectionGrid.from_dataset(ref_ds, offsets={ref_path: (0, 0)})

    return datasets, ref_ds, ref_path, msk_ds, msk_path, grid, all_compatible


def _build_mean_shift_from_data(
    pxl_vals: np.ndarray,
    quantile: float,
    n_samples: Optional[int],
    rnd_seed: Optional[int],
) -> Any:
    """Estimate MeanShift bandwidth from pixel features and return MeanShift."""
    q_val = float(quantile)
    if q_val <= 0.0 or q_val > 1.0:
        raise rsgispy.RSGISPyException(
            f"mean_shift_quantile must be in (0, 1], got {quantile}."
        )
    bw = float(
        sklearn.cluster.estimate_bandwidth(
            pxl_vals,
            quantile=q_val,
            n_samples=n_samples,
            random_state=rnd_seed,
        )
    )
    if not np.isfinite(bw) or bw <= 0.0:
        raise rsgispy.RSGISPyException(
            "Estimated MeanShift bandwidth was non-positive; feature values "
            "may have insufficient variance."
        )
    return sklearn.cluster.MeanShift(bandwidth=bw, bin_seeding=True)


def _resolve_cluster_out_dtype(
    datatype: Optional[int],
    max_out_id: int,
    eff_nodata: int,
) -> Tuple[str, np.dtype]:
    """Resolve rasterio and NumPy output datatypes for cluster IDs."""
    if datatype is None:
        out_rsgis_dtype, out_np_dtype = select_out_int_dtype(
            [0, max_out_id, eff_nodata]
        )
        return rsgispy.get_rasterio_datatype(out_rsgis_dtype), out_np_dtype
    return (
        rsgispy.get_rasterio_datatype(datatype),
        np.dtype(rsgispy.get_numpy_datatype(datatype)),
    )


def _infer_max_raw_cluster_id(estimator: Any) -> int:
    """Inspect fitted clustering estimator to determine maximum raw label ID."""
    candidates: List[int] = [0]
    labels_attr = getattr(estimator, "labels_", None)
    if labels_attr is not None:
        arr = np.asarray(labels_attr)
        if arr.size > 0:
            pos = arr[arr >= 0]
            if pos.size > 0:
                candidates.append(int(np.max(pos)))

    for attr_name in ("cluster_centers_", "means_"):
        centers = getattr(estimator, attr_name, None)
        if centers is not None and len(centers) > 0:
            candidates.append(int(len(centers)) - 1)

    for attr_name in ("n_clusters", "n_components"):
        val = getattr(estimator, attr_name, None)
        if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
            candidates.append(max(0, int(val) - 1))

    return max(candidates)


def _map_labels_to_output(
    raw_labels: np.ndarray,
    out_id_offset: int,
    eff_nodata: int,
    out_np_dtype: np.dtype,
) -> np.ndarray:
    """Shift non-negative cluster IDs by offset and map outliers to nodata."""
    labels_i64 = np.asarray(raw_labels, dtype=np.int64)
    mapped = np.where(
        labels_i64 >= 0,
        labels_i64 + int(out_id_offset),
        int(eff_nodata),
    )
    return mapped.astype(out_np_dtype, copy=False)


def _extract_block_roi_idx(
    msk_ds: Optional[DatasetReader],
    msk_path: Optional[str],
    grid: IntersectionGrid,
    out_win: Window,
    img_msk_val: int,
) -> np.ndarray:
    """Return flat pixel indices inside ``out_win`` selected by mask (or all)."""
    win_h = int(out_win.height)
    win_w = int(out_win.width)
    if msk_ds is not None and msk_path is not None:
        base_msk_col, base_msk_row = grid.offsets[msk_path]
        msk_win = Window(
            base_msk_col + int(out_win.col_off),
            base_msk_row + int(out_win.row_off),
            win_w,
            win_h,
        )
        msk_arr = msk_ds.read(1, window=msk_win)
        return np.flatnonzero(msk_arr.ravel() == int(img_msk_val))
    return np.arange(win_h * win_w, dtype=np.intp)


def _sample_single_image_can_reuse(
    in_msk_img: Optional[Union[str, os.PathLike]],
    band_specs: Sequence[Tuple[str, int, str]],
    no_data_val: Optional[float],
) -> Tuple[bool, Optional[float]]:
    """
    Determine whether :func:`extract_img_pxl_sample` can extract samples cleanly.
    """
    if in_msk_img is not None or not band_specs:
        return False, None
    unique_imgs = {spec[0] for spec in band_specs}
    if len(unique_imgs) != 1:
        return False, None

    img_path = band_specs[0][0]
    with rasterio.open(img_path, "r") as src:
        if [spec[1] for spec in band_specs] != list(range(1, int(src.count) + 1)):
            return False, None
        if no_data_val is not None:
            return True, float(no_data_val)
        nd_vals = list(src.nodatavals)
        unique_nd = {
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            for v in nd_vals
        }
        has_nan_nd = any(isinstance(v, float) and np.isnan(v) for v in nd_vals)
        if has_nan_nd:
            if len(unique_nd) == 1 and all(
                isinstance(v, float) and np.isnan(v) for v in nd_vals
            ):
                return True, float("nan")
            return False, None
        if len(unique_nd) == 1:
            return True, next(iter(unique_nd))
        return False, None


def img_pixel_cluster(
    img_band_info: ClusterImageInput,
    output_img: Union[str, os.PathLike],
    clusterer: Optional[Any] = None,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    out_id_offset: int = 1,
    use_mean_shift_est_band_width: bool = False,
    mean_shift_quantile: float = 0.3,
    mean_shift_n_samples: Optional[int] = None,
    calc_stats: bool = True,
    add_clr_tab: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    rnd_seed: Optional[int] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = False,
    use_pbar: bool = True,
) -> Optional[Any]:
    """
    Perform unsupervised pixel classification (clustering) across the **whole
    image dataset** in a single ``scikit-learn`` fit operation.

    All valid pixels from the input image(s) specified via ``img_band_info``
    are loaded into memory and passed directly to ``clusterer``. Because all
    pixels are clustered together without sampling approximation or tiling,
    this supports **any** ``scikit-learn`` clustering algorithm—including
    transductive methods that do not implement ``predict()`` (such as
    ``DBSCAN``, ``OPTICS``, ``AgglomerativeClustering``, and
    ``SpectralClustering``) — with no tile boundary artefacts.

    .. warning::

       Fitting directly on the entire image dataset can be **slow and
       memory-intensive** for large scenes. For larger imagery with inductive
       clusterers that expose ``predict()`` (e.g., ``KMeans``,
       ``MiniBatchKMeans``, ``Birch``, ``MeanShift``), consider using
       :func:`img_pixel_sample_cluster`.

    By default, non-negative cluster IDs (``0, 1, ..., K-1``) are shifted by
    ``out_id_offset=1`` to ``1, 2, ..., K`` so unsigned integer ``0`` is cleanly
    reserved as output nodata (and any negative outlier/noise labels such as
    ``-1`` from ``DBSCAN`` map to nodata).

    :param img_band_info: Input image/band definition: an image path (all its
                          bands), a
                          :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                          a JSON classification configuration file path
                          (``.json``), or a sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`
                          objects.
    :param output_img: Path to output classified cluster raster image.
    :param clusterer: Unfitted ``scikit-learn`` clusterer instance (or ``None``
                      when ``use_mean_shift_est_band_width=True``).
    :param in_msk_img: Optional mask image path restricting clustered pixels to
                       where band 1 equals ``img_msk_val``.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying ROI
                        pixels (Default = ``1``).
    :param no_data_val: Optional explicit input nodata value overriding raster
                        band metadata.
    :param gdalformat: Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for
                     ``output_img``. When ``None`` (default), automatically
                     selects the smallest unsigned integer datatype fitting all
                     cluster IDs and nodata.
    :param out_no_data_val: Optional unsigned integer nodata value for
                            ``output_img``. Defaults to ``0`` when unused by
                            cluster IDs, otherwise next unused unsigned int.
    :param out_id_offset: Non-negative offset added to cluster labels ``>= 0``
                          (Default = ``1`` so clusters are ``1..K`` with ``0``
                          as nodata).
    :param use_mean_shift_est_band_width: If ``True`` and ``clusterer=None``,
                                          estimate bandwidth automatically and
                                          run ``sklearn.cluster.MeanShift``.
    :param mean_shift_quantile: Quantile in ``(0, 1]`` passed to
                                ``sklearn.cluster.estimate_bandwidth``
                                (Default = ``0.3``).
    :param mean_shift_n_samples: Optional subsample count passed to
                                 ``estimate_bandwidth``.
    :param calc_stats: If ``True`` (default), compute thematic image
                       statistics, histogram, and pyramids.
    :param add_clr_tab: If ``True`` (default) and ``calc_stats=True``,
                        populate an RGB colour table for integer outputs.
    :param block_size: Optional raster read/write block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param rnd_seed: Optional random seed applied to cloned estimator /
                     bandwidth estimation when supported.
    :param out_model_file: Optional ``.joblib`` file path to save the fitted
                           clusterer.
    :param return_model: If ``True``, return the fitted clusterer object
                         (Default = ``False``).
    :param use_pbar: Display tqdm progress bars (Default = ``True``).
    :return: Fitted clusterer if ``return_model=True``, else ``None``.

    .. code:: python

        from sklearn.cluster import MiniBatchKMeans
        import rsgispy.classification.classdata as cd
        import rsgispy.classification.clustersklearn as clsk

        bands = [cd.ImageBandInfo("input_img.tif", name="s2", bands=[1, 2, 3])]
        clsk.img_pixel_cluster(
            bands,
            "clusters_full.tif",
            clusterer=MiniBatchKMeans(n_clusters=10, random_state=42),
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches
    check_sklearn_available("img_pixel_cluster")
    _validate_cluster_options(
        clusterer=clusterer,
        use_mean_shift_est_band_width=use_mean_shift_est_band_width,
        out_id_offset=out_id_offset,
        require_predict=False,
        func_name="img_pixel_cluster",
    )
    band_specs = _resolve_cluster_band_specs(img_band_info)
    out_cls_path = ensure_parent_dir(output_img)
    c_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, ref_ds, ref_path, msk_ds, msk_path, grid, all_compat = (
            _open_cluster_datasets(stack, band_specs, in_msk_img)
        )
        n_total_pxls = int(grid.height) * int(grid.width)
        flat_idx_chunks: List[np.ndarray] = []
        pxl_chunks: List[np.ndarray] = []

        for out_win in iter_dataset_windows(
            grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
        ):
            win_w = int(out_win.width)
            col_off = int(out_win.col_off)
            row_off = int(out_win.row_off)
            roi_idx = _extract_block_roi_idx(
                msk_ds, msk_path, grid, out_win, img_msk_val
            )
            if roi_idx.size == 0:
                continue

            feat_arrays = read_feature_block(
                band_specs=band_specs,
                datasets=datasets,
                ref_ds=ref_ds,
                grid=grid,
                out_win=out_win,
                all_compatible=all_compat,
                ref_path=ref_path,
                explicit_no_data=no_data_val,
            )
            valid_feats, valid_idx = extract_roi_feature_matrix(
                feat_arrays, roi_idx
            )
            if valid_idx.size > 0:
                rows = (valid_idx // win_w) + row_off
                cols = (valid_idx % win_w) + col_off
                flat_idx_chunks.append(rows * int(grid.width) + cols)
                pxl_chunks.append(valid_feats)

        if not pxl_chunks:
            raise rsgispy.RSGISPyException(
                "No valid pixels found in input image(s) for img_pixel_cluster."
            )
        all_pxls = np.concatenate(pxl_chunks, axis=0)

        if use_mean_shift_est_band_width:
            ms_est = _build_mean_shift_from_data(
                all_pxls, mean_shift_quantile, mean_shift_n_samples, rnd_seed
            )
            fitted_est = ms_est.fit(all_pxls)
            raw_labels = np.asarray(fitted_est.labels_, dtype=np.int64)
        else:
            fitted_est = clone_sklearn_estimator(
                clusterer, rnd_seed=rnd_seed, extra="classification"
            )
            if hasattr(fitted_est, "fit_predict"):
                raw_labels = np.asarray(
                    fitted_est.fit_predict(all_pxls), dtype=np.int64
                )
            else:
                fitted_est.fit(all_pxls)
                if hasattr(fitted_est, "labels_"):
                    raw_labels = np.asarray(fitted_est.labels_, dtype=np.int64)
                elif hasattr(fitted_est, "predict"):
                    raw_labels = np.asarray(
                        fitted_est.predict(all_pxls), dtype=np.int64
                    )
                else:
                    raise rsgispy.RSGISPyException(
                        "Fitted clusterer exposes neither fit_predict(), "
                        "labels_, nor predict()."
                    )
            sync_sklearn_estimator_state(clusterer, fitted_est)

        non_neg = raw_labels[raw_labels >= 0]
        if non_neg.size > 0:
            shifted_ids = [
                int(np.min(non_neg)) + int(out_id_offset),
                int(np.max(non_neg)) + int(out_id_offset),
            ]
        else:
            shifted_ids = [int(out_id_offset)]
        eff_nodata = resolve_out_nodata(out_no_data_val, shifted_ids)
        out_rio_dtype, out_np_dtype = _resolve_cluster_out_dtype(
            datatype, max(shifted_ids), eff_nodata
        )

        out_full_flat = np.full(n_total_pxls, eff_nodata, dtype=out_np_dtype)
        out_full_flat[np.concatenate(flat_idx_chunks)] = _map_labels_to_output(
            raw_labels, out_id_offset, eff_nodata, out_np_dtype
        )
        out_full_2d = out_full_flat.reshape((int(grid.height), int(grid.width)))

        out_dst = open_output_dataset(
            stack=stack,
            output_img=out_cls_path,
            gdalformat=gdalformat,
            out_rio_dtype=out_rio_dtype,
            grid=grid,
            count=1,
            creation_opts=c_opts,
        )
        out_dst.nodata = eff_nodata
        for out_win in iter_dataset_windows(
            grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
        ):
            r0 = int(out_win.row_off)
            r1 = r0 + int(out_win.height)
            c0 = int(out_win.col_off)
            c1 = c0 + int(out_win.width)
            out_dst.write(out_full_2d[r0:r1, c0:c1], 1, window=out_win)

    finalise_thematic_output_img(
        out_cls_path=out_cls_path,
        eff_nodata=eff_nodata,
        out_rio_dtype=out_rio_dtype,
        out_np_dtype=out_np_dtype,
        calc_stats=calc_stats,
        add_clr_tab=add_clr_tab,
        clr_lut=None,
        use_pbar=use_pbar,
    )
    if out_model_file is not None:
        save_sklearn_classifier(fitted_est, out_model_file)
    if return_model:
        return fitted_est
    return None


def _extract_cluster_sample_matrix(
    in_msk_img: Optional[Union[str, os.PathLike]],
    img_msk_val: int,
    band_specs: Sequence[Tuple[str, int, str]],
    datasets: Mapping[str, DatasetReader],
    ref_ds: DatasetReader,
    ref_path: str,
    msk_ds: Optional[DatasetReader],
    msk_path: Optional[str],
    grid: IntersectionGrid,
    all_compatible: bool,
    n_img_smpl: int,
    max_samples: Optional[int],
    no_data_val: Optional[float],
    block_size: Optional[int],
    rnd_seed: Optional[int],
    use_pbar: bool,
) -> np.ndarray:
    """Extract a representative sample matrix of valid image pixel vectors."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    can_reuse, reuse_nd = _sample_single_image_can_reuse(
        in_msk_img=in_msk_img,
        band_specs=band_specs,
        no_data_val=no_data_val,
    )
    if can_reuse:
        smpl = extract_img_pxl_sample(
            band_specs[0][0],
            pxl_n_sample=int(n_img_smpl),
            no_data_val=reuse_nd,
            use_pbar=use_pbar,
        ).astype(np.float64, copy=False)
        if smpl.shape[0] > 0:
            smpl = smpl[np.all(np.isfinite(smpl), axis=1)]
    else:
        chunks: List[np.ndarray] = []
        offset = 0
        stride = int(n_img_smpl)
        for out_win in iter_dataset_windows(
            grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
        ):
            roi_idx = _extract_block_roi_idx(
                msk_ds, msk_path, grid, out_win, img_msk_val
            )
            if roi_idx.size == 0:
                continue
            feat_arrays = read_feature_block(
                band_specs=band_specs,
                datasets=datasets,
                ref_ds=ref_ds,
                grid=grid,
                out_win=out_win,
                all_compatible=all_compatible,
                ref_path=ref_path,
                explicit_no_data=no_data_val,
            )
            valid_feats, _ = extract_roi_feature_matrix(feat_arrays, roi_idx)
            n_valid = valid_feats.shape[0]
            if n_valid == 0:
                continue
            start_idx = (-offset) % stride
            if start_idx < n_valid:
                chunks.append(valid_feats[start_idx::stride])
            offset = (offset + n_valid) % stride
        smpl = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, len(band_specs)), dtype=np.float64)
        )

    if smpl.shape[0] == 0:
        raise rsgispy.RSGISPyException(
            "No valid pixels were sampled from input image(s) for "
            "img_pixel_sample_cluster."
        )
    if max_samples is not None and smpl.shape[0] > int(max_samples):
        rng = resolve_rng(rnd_seed)
        sel_idx = np.sort(
            rng.choice(smpl.shape[0], size=int(max_samples), replace=False)
        )
        smpl = smpl[sel_idx]
    return smpl


def _write_cluster_predictions(
    stack: ExitStack,
    fitted_est: Any,
    opened: Tuple[
        Dict[str, DatasetReader],
        DatasetReader,
        str,
        Optional[DatasetReader],
        Optional[str],
        IntersectionGrid,
        bool,
    ],
    band_specs: Sequence[Tuple[str, int, str]],
    out_cls_path: str,
    *,
    img_msk_val: int,
    no_data_val: Optional[float],
    out_id_offset: int,
    gdalformat: str,
    datatype: Optional[int],
    out_no_data_val: Optional[int],
    block_size: Optional[int],
    creation_opts: Optional[Mapping[str, Any]],
    use_pbar: bool,
) -> Tuple[int, str, np.dtype]:
    """
    Apply a fitted clusterer's ``predict()`` block by block to the opened
    datasets (from :func:`_open_cluster_datasets`) and write the cluster IDs.

    :return: ``(eff_nodata, out_rio_dtype, out_np_dtype)`` of the output.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    datasets, ref_ds, ref_path, msk_ds, msk_path, grid, all_compat = opened
    max_raw_id = _infer_max_raw_cluster_id(fitted_est)
    max_out_id = max_raw_id + int(out_id_offset)
    expected_ids = list(range(int(out_id_offset), max_out_id + 1))
    eff_nodata = resolve_out_nodata(out_no_data_val, expected_ids)
    out_rio_dtype, out_np_dtype = _resolve_cluster_out_dtype(
        datatype, max_out_id, eff_nodata
    )

    out_dst = open_output_dataset(
        stack=stack,
        output_img=out_cls_path,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        grid=grid,
        count=1,
        creation_opts=creation_opts,
    )
    out_dst.nodata = eff_nodata

    for out_win in iter_dataset_windows(
        grid.height, grid.width, block_size=block_size, use_pbar=use_pbar
    ):
        win_h = int(out_win.height)
        win_w = int(out_win.width)
        roi_idx = _extract_block_roi_idx(
            msk_ds, msk_path, grid, out_win, img_msk_val
        )
        cls_flat = np.full(win_h * win_w, eff_nodata, dtype=out_np_dtype)
        if roi_idx.size > 0:
            feat_arrays = read_feature_block(
                band_specs=band_specs,
                datasets=datasets,
                ref_ds=ref_ds,
                grid=grid,
                out_win=out_win,
                all_compatible=all_compat,
                ref_path=ref_path,
                explicit_no_data=no_data_val,
            )
            valid_feats, valid_idx = extract_roi_feature_matrix(
                feat_arrays, roi_idx
            )
            if valid_idx.size > 0:
                raw_preds = np.asarray(
                    fitted_est.predict(valid_feats), dtype=np.int64
                )
                cls_flat[valid_idx] = _map_labels_to_output(
                    raw_preds, out_id_offset, eff_nodata, out_np_dtype
                )
        out_dst.write(cls_flat.reshape((win_h, win_w)), 1, window=out_win)
    return eff_nodata, out_rio_dtype, out_np_dtype


def img_pixel_sample_cluster(
    img_band_info: ClusterImageInput,
    output_img: Union[str, os.PathLike],
    clusterer: Optional[Any] = None,
    n_img_smpl: int = 100,
    max_samples: Optional[int] = None,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    *,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    out_id_offset: int = 1,
    use_mean_shift_est_band_width: bool = False,
    mean_shift_quantile: float = 0.3,
    mean_shift_n_samples: Optional[int] = None,
    calc_stats: bool = True,
    add_clr_tab: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    rnd_seed: Optional[int] = None,
    out_model_file: Optional[Union[str, os.PathLike]] = None,
    return_model: bool = False,
    use_pbar: bool = True,
) -> Optional[Any]:
    """
    Perform unsupervised pixel classification by fitting a ``scikit-learn``
    clusterer on a **sample of image pixels** and applying it block-by-block to
    the whole image via ``predict()``.

    Every ``n_img_smpl``-th valid pixel (e.g., ``100`` for an ~1% systematic
    sample, optionally capped at ``max_samples``) is extracted across the
    image(s) specified via ``img_band_info`` to train the clustering model,
    which is then evaluated across all valid pixels block by block. Because
    model fitting runs only on the subset and inference streams in windows,
    this is far faster and uses minimal RAM compared to
    :func:`img_pixel_cluster` without introducing tile boundary artefacts.

    Only compatible with clustering estimators implementing ``predict()``
    (e.g., ``KMeans``, ``MiniBatchKMeans``, ``BisectingKMeans``, ``Birch``,
    ``MeanShift``, ``AffinityPropagation``, or ``GaussianMixture``).

    :param img_band_info: Input image/band definition: an image path (all its
                          bands), a
                          :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                          a JSON classification configuration file path
                          (``.json``), or a sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`
                          objects.
    :param output_img: Path to output classified cluster raster image.
    :param clusterer: Unfitted ``scikit-learn`` clusterer instance with a
                      ``predict()`` method (or ``None`` when
                      ``use_mean_shift_est_band_width=True``).
    :param n_img_smpl: Pixel sampling interval across valid pixels
                       (e.g., ``100`` samples every 100th valid pixel,
                       Default = ``100``).
    :param max_samples: Optional maximum cap on sampled training pixels.
    :param in_msk_img: Optional mask image path restricting both sampling and
                       prediction to where band 1 equals ``img_msk_val``.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying ROI
                        pixels (Default = ``1``).
    :param no_data_val: Optional explicit input nodata value overriding raster
                        band metadata.
    :param gdalformat: Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for
                     ``output_img``. When ``None`` (default), automatically
                     selects the smallest unsigned integer datatype fitting all
                     cluster IDs and nodata.
    :param out_no_data_val: Optional unsigned integer nodata value for
                            ``output_img``. Defaults to ``0`` when unused by
                            cluster IDs, otherwise next unused unsigned int.
    :param out_id_offset: Non-negative offset added to cluster labels ``>= 0``
                          (Default = ``1`` so clusters are ``1..K`` with ``0``
                          as nodata).
    :param use_mean_shift_est_band_width: If ``True`` and ``clusterer=None``,
                                          estimate bandwidth automatically from
                                          sampled pixels and run ``MeanShift``.
    :param mean_shift_quantile: Quantile in ``(0, 1]`` passed to
                                ``sklearn.cluster.estimate_bandwidth``
                                (Default = ``0.3``).
    :param mean_shift_n_samples: Optional subsample count passed to
                                 ``estimate_bandwidth``.
    :param calc_stats: If ``True`` (default), compute thematic image
                       statistics, histogram, and pyramids.
    :param add_clr_tab: If ``True`` (default) and ``calc_stats=True``,
                        populate an RGB colour table for integer outputs.
    :param block_size: Optional raster processing block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param rnd_seed: Optional random seed applied to cloned estimator /
                     ``max_samples`` subsampling / bandwidth estimation.
    :param out_model_file: Optional ``.joblib`` file path to save the fitted
                           clusterer.
    :param return_model: If ``True``, return the fitted clusterer object
                         (Default = ``False``).
    :param use_pbar: Display tqdm progress bars (Default = ``True``).
    :return: Fitted clusterer if ``return_model=True``, else ``None``.

    .. code:: python

        from sklearn.cluster import MiniBatchKMeans
        import rsgispy.classification.classdata as cd
        import rsgispy.classification.clustersklearn as clsk

        bands = [cd.ImageBandInfo("input_img.tif", name="s2", bands=[1, 2, 3])]
        clsk.img_pixel_sample_cluster(
            bands,
            "clusters_sample.tif",
            clusterer=MiniBatchKMeans(n_clusters=10, random_state=42),
            n_img_smpl=100,
            no_data_val=0.0,
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    check_sklearn_available("img_pixel_sample_cluster")
    if isinstance(n_img_smpl, bool) or int(n_img_smpl) < 1:
        raise rsgispy.RSGISPyException(
            f"n_img_smpl must be a positive integer, got {n_img_smpl!r}."
        )
    if max_samples is not None and (
        isinstance(max_samples, bool) or int(max_samples) < 1
    ):
        raise rsgispy.RSGISPyException(
            f"max_samples must be a positive integer, got {max_samples!r}."
        )
    _validate_cluster_options(
        clusterer=clusterer,
        use_mean_shift_est_band_width=use_mean_shift_est_band_width,
        out_id_offset=out_id_offset,
        require_predict=True,
        func_name="img_pixel_sample_cluster",
    )
    band_specs = _resolve_cluster_band_specs(img_band_info)
    out_cls_path = ensure_parent_dir(output_img)
    c_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        datasets, ref_ds, ref_path, msk_ds, msk_path, grid, all_compat = (
            _open_cluster_datasets(stack, band_specs, in_msk_img)
        )
        smpl = _extract_cluster_sample_matrix(
            in_msk_img=in_msk_img,
            img_msk_val=img_msk_val,
            band_specs=band_specs,
            datasets=datasets,
            ref_ds=ref_ds,
            ref_path=ref_path,
            msk_ds=msk_ds,
            msk_path=msk_path,
            grid=grid,
            all_compatible=all_compat,
            n_img_smpl=n_img_smpl,
            max_samples=max_samples,
            no_data_val=no_data_val,
            block_size=block_size,
            rnd_seed=rnd_seed,
            use_pbar=use_pbar,
        )

        if use_mean_shift_est_band_width:
            ms_est = _build_mean_shift_from_data(
                smpl, mean_shift_quantile, mean_shift_n_samples, rnd_seed
            )
            fitted_est = ms_est.fit(smpl)
        else:
            fitted_est = fit_clean_sklearn_estimator(
                clusterer,
                smpl,
                train_y=None,
                sync_input=True,
                rnd_seed=rnd_seed,
                extra="classification",
            )
        check_sklearn_estimator_fitted(fitted_est, label="clusterer")

        out_info = _write_cluster_predictions(
            stack,
            fitted_est,
            (datasets, ref_ds, ref_path, msk_ds, msk_path, grid, all_compat),
            band_specs,
            out_cls_path,
            img_msk_val=img_msk_val,
            no_data_val=no_data_val,
            out_id_offset=out_id_offset,
            gdalformat=gdalformat,
            datatype=datatype,
            out_no_data_val=out_no_data_val,
            block_size=block_size,
            creation_opts=c_opts,
            use_pbar=use_pbar,
        )

    finalise_thematic_output_img(
        out_cls_path=out_cls_path,
        eff_nodata=out_info[0],
        out_rio_dtype=out_info[1],
        out_np_dtype=out_info[2],
        calc_stats=calc_stats,
        add_clr_tab=add_clr_tab,
        clr_lut=None,
        use_pbar=use_pbar,
    )
    if out_model_file is not None:
        save_sklearn_classifier(fitted_est, out_model_file)
    if return_model:
        return fitted_est
    return None


_PREDICT_CLUSTERERS = (
    "KMeans, MiniBatchKMeans, BisectingKMeans, Birch, MeanShift, "
    "AffinityPropagation, GaussianMixture or BayesianGaussianMixture"
)


def _check_apply_clusterer(clusterer: Any, n_feats: int) -> None:
    """Raise if ``clusterer`` cannot be applied to ``n_feats`` image bands."""
    if not hasattr(clusterer, "predict"):
        raise rsgispy.RSGISPyException(
            f"The clusterer ({type(clusterer).__name__}) has no predict() method, "
            "so it cannot be applied to other pixels (e.g. DBSCAN, OPTICS, "
            "AgglomerativeClustering and SpectralClustering only label the "
            "pixels they were fitted on). Use a clusterer with predict(): "
            f"{_PREDICT_CLUSTERERS}."
        )
    check_sklearn_estimator_fitted(clusterer, label="clusterer")
    n_fit = getattr(clusterer, "n_features_in_", None)
    if isinstance(n_fit, (int, np.integer)) and int(n_fit) != int(n_feats):
        raise rsgispy.RSGISPyException(
            f"The clusterer was fitted on {int(n_fit)} bands but img_band_info "
            f"gives {int(n_feats)} bands. Use the same bands, in the same order, "
            "as the image the clusterer was fitted on."
        )


def apply_img_pixel_clusterer(
    img_band_info: ClusterImageInput,
    clusterer: Union[Any, str, os.PathLike],
    output_img: Union[str, os.PathLike],
    *,
    in_msk_img: Optional[Union[str, os.PathLike]] = None,
    img_msk_val: int = 1,
    no_data_val: Optional[float] = None,
    out_id_offset: int = 1,
    gdalformat: str = "GTIFF",
    datatype: Optional[int] = None,
    out_no_data_val: Optional[int] = None,
    calc_stats: bool = True,
    add_clr_tab: bool = True,
    block_size: Optional[int] = None,
    creation_opts: Optional[Mapping[str, Any]] = None,
    use_pbar: bool = True,
) -> None:
    """
    Apply a **fitted** ``scikit-learn`` clusterer to an image, block by block,
    with ``predict()``: for example a clusterer fitted on one image (or date)
    by :func:`img_pixel_sample_cluster` (``return_model=True`` or
    ``out_model_file``) applied to another, so the cluster IDs mean the same
    thing in both outputs.

    The prediction code is shared with :func:`img_pixel_sample_cluster`:
    applied to the image the clusterer was fitted on, with the same
    ``out_id_offset``, mask and no-data options, the output is identical to
    that function's output. Cluster ``k`` (``0``-based) is written as
    ``k + out_id_offset``. The bands must be the same, in the same order, as
    those the clusterer was fitted on.

    Only clusterers with ``predict()`` can be applied: ``KMeans``,
    ``MiniBatchKMeans``, ``BisectingKMeans``, ``Birch``, ``MeanShift``,
    ``AffinityPropagation``, ``GaussianMixture`` and
    ``BayesianGaussianMixture``. ``DBSCAN``, ``OPTICS``,
    ``AgglomerativeClustering`` and ``SpectralClustering`` only label the
    pixels they were fitted on (use :func:`img_pixel_cluster`).

    In a :class:`~rsgispy.imageprocess.TileGraph`, apply a fitted clusterer
    with :func:`~rsgispy.ml.predict.predict_raster_node` (``thematic=True``)
    and a ``lut`` adding the offset, e.g. ``lut={k: k + 1 for k in
    range(n_clusters)}``, which gives the same cluster IDs as this function.

    :param img_band_info: Input image/band definition: an image path (all its
                          bands), a
                          :class:`~rsgispy.classification.classdata.ClassificationConfig`,
                          a JSON classification configuration file path
                          (``.json``), or a sequence of
                          :class:`~rsgispy.classification.classdata.ImageBandInfo`
                          objects.
    :param clusterer: Fitted clusterer with ``predict()``, or the path to the
                      ``.joblib`` file saved with ``out_model_file`` by
                      :func:`img_pixel_sample_cluster` or
                      :func:`img_pixel_cluster`.
    :param output_img: Path to the output cluster image.
    :param in_msk_img: Optional mask image path restricting prediction to
                       where band 1 equals ``img_msk_val``.
    :param img_msk_val: Pixel value within ``in_msk_img`` identifying ROI
                        pixels (Default = ``1``).
    :param no_data_val: Optional explicit input no-data value overriding the
                        raster band metadata.
    :param out_id_offset: Non-negative offset added to cluster labels ``>= 0``
                          (Default = ``1`` so clusters are ``1..K`` with ``0``
                          as no data).
    :param gdalformat: Output GDAL/rasterio driver name (Default = ``'GTIFF'``).
    :param datatype: Optional ``rsgispy.TYPE_*`` integer datatype for
                     ``output_img``. When ``None`` (default), the smallest
                     unsigned integer datatype fitting all cluster IDs and no
                     data.
    :param out_no_data_val: Optional unsigned integer no-data value for
                            ``output_img``. Defaults to ``0`` when unused by
                            cluster IDs, otherwise the next unused value.
    :param calc_stats: If ``True`` (default), compute thematic image
                       statistics, histogram, and pyramids.
    :param add_clr_tab: If ``True`` (default) and ``calc_stats=True``,
                        populate an RGB colour table for integer outputs.
    :param block_size: Optional raster processing block size in pixels.
    :param creation_opts: Optional GDAL creation options mapping.
    :param use_pbar: Display tqdm progress bars (Default = ``True``).

    .. code:: python

        from sklearn.cluster import KMeans
        import rsgispy.classification.clustersklearn as clsk

        # Fit on one image and save the clusterer...
        clsk.img_pixel_sample_cluster(
            "sen2_2018.tif",
            "clusters_2018.tif",
            clusterer=KMeans(n_clusters=10, n_init=10, random_state=42),
            no_data_val=0,
            out_model_file="kmeans.joblib",
        )
        # ...and apply it to a second image with the same bands.
        clsk.apply_img_pixel_clusterer(
            "sen2_2019.tif", "kmeans.joblib", "clusters_2019.tif", no_data_val=0
        )
    """
    # pylint: disable=too-many-arguments,too-many-locals
    check_sklearn_available("apply_img_pixel_clusterer")
    if isinstance(clusterer, (str, os.PathLike)):
        clusterer = load_sklearn_classifier(clusterer)
    band_specs = _resolve_cluster_band_specs(img_band_info)
    _check_apply_clusterer(clusterer, len(band_specs))
    _validate_cluster_options(
        clusterer=clusterer,
        use_mean_shift_est_band_width=False,
        out_id_offset=out_id_offset,
        require_predict=True,
        func_name="apply_img_pixel_clusterer",
    )
    out_cls_path = ensure_parent_dir(output_img)
    c_opts = normalise_creation_options(creation_opts)

    with ExitStack() as stack:
        out_info = _write_cluster_predictions(
            stack,
            clusterer,
            _open_cluster_datasets(stack, band_specs, in_msk_img),
            band_specs,
            out_cls_path,
            img_msk_val=img_msk_val,
            no_data_val=no_data_val,
            out_id_offset=out_id_offset,
            gdalformat=gdalformat,
            datatype=datatype,
            out_no_data_val=out_no_data_val,
            block_size=block_size,
            creation_opts=c_opts,
            use_pbar=use_pbar,
        )

    finalise_thematic_output_img(
        out_cls_path=out_cls_path,
        eff_nodata=out_info[0],
        out_rio_dtype=out_info[1],
        out_np_dtype=out_info[2],
        calc_stats=calc_stats,
        add_clr_tab=add_clr_tab,
        clr_lut=None,
        use_pbar=use_pbar,
    )
