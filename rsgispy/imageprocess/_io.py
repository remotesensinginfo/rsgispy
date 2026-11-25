"""
Raster dataset opening and output-creation context helpers for
:mod:`rsgispy.imageprocess`.

Cloud Optimised GeoTIFF (COG) outputs: GDAL's COG driver can only create a
file by copying a complete dataset (``CreateCopy``), so opening it in write
mode makes rasterio build the whole raster in memory. Instead, rsgispy writes
COG outputs block by block to a temporary tiled GeoTIFF next to the output
(:class:`StagedOutput`) and converts it with ``rasterio.shutil.copy(...,
driver="COG")`` once the writer has closed. Existing COGs are never edited in
place: :func:`update_cog` rebuilds them through a temporary GeoTIFF.
"""

from contextlib import ExitStack
import os
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import uuid

import rasterio
import rasterio.shutil
from rasterio.io import DatasetReader, DatasetWriter

import rsgispy
from rsgispy.imageprocess._arrays import BandDefn, IntersectionGrid
from rsgispy.imageprocess._dtype_nodata import get_gdal_img_creation_opts

_STAGING_SUFFIX = ".rsgispy_tmp.tif"
_SIDECAR_EXTS = (".aux.xml", ".ovr", ".msk")
# Creation options of the GeoTIFF driver that the COG driver does not accept.
_GTIFF_ONLY_OPTS = frozenset({"TILED", "BLOCKXSIZE", "BLOCKYSIZE"})
_DEFAULT_COG_BLOCKSIZE = 512
# Items of the "ghost" header that GDAL's COG driver writes after the TIFF
# header; GDAL refuses in-place updates of files that have all of them.
_COG_GHOST_MARKERS = (
    b"GDAL_STRUCTURAL_METADATA_SIZE=",
    b"LAYOUT=IFDS_BEFORE_DATA",
    b"KNOWN_INCOMPATIBLE_EDITION=NO",
)
# id(writer) -> StagedOutput for writers opened by open_output_dataset.
_WRITER_STAGES: Dict[int, "StagedOutput"] = {}


def _staging_path(output_img: str) -> str:
    """Return a unique temporary file path in the directory of ``output_img``."""
    out_abs = os.path.abspath(output_img)
    out_dir, base = os.path.split(out_abs)
    return os.path.join(out_dir, f".{base}.{uuid.uuid4().hex[:12]}{_STAGING_SUFFIX}")


def _remove_raster_files(img_path: str) -> None:
    """Delete ``img_path`` and any GDAL sidecar files, ignoring missing files."""
    for path in (img_path, *(img_path + ext for ext in _SIDECAR_EXTS)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def is_cog_layout(input_img: Union[str, os.PathLike]) -> bool:
    """
    Return ``True`` if ``input_img`` is a Cloud Optimised GeoTIFF written by
    GDAL's COG driver whose layout is still intact, i.e. a file that GDAL
    refuses to update in place (unless ``IGNORE_COG_LAYOUT_BREAK=YES``) and
    that rsgispy therefore rebuilds instead of editing.

    The test reads the COG "ghost" header GDAL writes after the TIFF header
    (``LAYOUT=IFDS_BEFORE_DATA`` and ``KNOWN_INCOMPATIBLE_EDITION=NO``). The
    ``LAYOUT=COG`` item of the ``IMAGE_STRUCTURE`` metadata is not used,
    because recent GDAL versions also report it for small tiled GeoTIFFs
    that happen to have a COG-compatible layout and can be edited normally.

    :param input_img: Path to a local image file.
    :return: ``True`` for an intact COG, ``False`` otherwise (including
             missing files, other formats and COGs already edited in place).

    .. code:: python

        import rsgispy.imageprocess

        if rsgispy.imageprocess.is_cog_layout("img.tif"):
            print("img.tif is a COG")
    """
    img_path = os.fspath(input_img)
    if not os.path.isfile(img_path):
        return False
    try:
        with open(img_path, "rb") as img_fobj:
            header = img_fobj.read(1024)
    except OSError:
        return False
    if header[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return False
    return all(marker in header for marker in _COG_GHOST_MARKERS)


def resolve_cog_creation_opts(
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve the creation options passed to GDAL's COG driver: those of
    :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` for ``'COG'``
    (environment variables only; there is no built-in default) with
    ``creation_opts`` merged on top.

    GeoTIFF-only options are translated or dropped, because the COG driver
    does not accept them: ``BLOCKXSIZE`` becomes ``BLOCKSIZE`` (unless
    ``BLOCKSIZE`` is given) and ``TILED``/``BLOCKYSIZE`` are removed.

    :param creation_opts: Optional explicit creation options.
    :return: New dictionary of COG creation options.

    .. code:: python

        import rsgispy.imageprocess

        opts = rsgispy.imageprocess.resolve_cog_creation_opts(
            {"COMPRESS": "DEFLATE", "TILED": "YES", "BLOCKXSIZE": 256}
        )
        # {'COMPRESS': 'DEFLATE', 'BLOCKSIZE': 256}
    """
    merged: Dict[str, Any] = dict(get_gdal_img_creation_opts("COG"))
    if creation_opts:
        merged.update(creation_opts)
    cog_opts: Dict[str, Any] = {}
    blk_x: Optional[Any] = None
    for key, val in merged.items():
        key_up = str(key).upper().strip()
        if key_up == "BLOCKXSIZE":
            blk_x = val
        if key_up not in _GTIFF_ONLY_OPTS:
            cog_opts[key_up] = val
    if blk_x is not None and "BLOCKSIZE" not in cog_opts:
        cog_opts["BLOCKSIZE"] = blk_x
    return cog_opts


def _staging_gtiff_opts(cog_opts: Mapping[str, Any]) -> Dict[str, Any]:
    """Creation options for the temporary GeoTIFF a COG is written to first."""
    try:
        blk = int(cog_opts.get("BLOCKSIZE", _DEFAULT_COG_BLOCKSIZE))
    except (TypeError, ValueError):
        blk = _DEFAULT_COG_BLOCKSIZE
    return {
        "TILED": "YES",
        "BLOCKXSIZE": blk,
        "BLOCKYSIZE": blk,
        "INTERLEAVE": "BAND",
        # Fast, lossless compression keeps the temporary file small.
        "COMPRESS": "DEFLATE",
        "ZLEVEL": 1,
        "BIGTIFF": "IF_SAFER",
    }


def cog_creation_opts_from_file(input_img: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Return COG creation options that reproduce the layout of an existing COG:
    its compression (``COMPRESS``, ``NONE`` if uncompressed), predictor,
    block size, overview resampling and interleave. Other options (e.g.
    ``LEVEL`` or ``QUALITY``, which are not recorded in the file) come from
    :func:`resolve_cog_creation_opts`.

    :param input_img: Path to an existing COG.
    :return: Dictionary of COG creation options.

    .. code:: python

        import rsgispy.imageprocess

        opts = rsgispy.imageprocess.cog_creation_opts_from_file("img_cog.tif")
    """
    cog_opts = resolve_cog_creation_opts()
    with rasterio.open(os.fspath(input_img), "r") as src:
        struct = {
            str(k).upper(): str(v) for k, v in src.tags(ns="IMAGE_STRUCTURE").items()
        }
        blk_h, blk_w = src.block_shapes[0]
    compression = struct.get("COMPRESSION", "NONE").upper()
    # e.g. 'YCbCr JPEG' for JPEG-compressed RGB images.
    cog_opts["COMPRESS"] = "JPEG" if "JPEG" in compression else compression
    predictor = {"2": "STANDARD", "3": "FLOATING_POINT"}.get(
        struct.get("PREDICTOR", "")
    )
    if predictor is not None:
        cog_opts["PREDICTOR"] = predictor
    else:
        cog_opts.pop("PREDICTOR", None)
    if "OVERVIEW_RESAMPLING" in struct:
        cog_opts["OVERVIEW_RESAMPLING"] = struct["OVERVIEW_RESAMPLING"]
    if blk_w == blk_h:
        cog_opts["BLOCKSIZE"] = int(blk_w)
    if struct.get("INTERLEAVE", "PIXEL").upper() in ("BAND", "TILE"):
        cog_opts["INTERLEAVE"] = struct["INTERLEAVE"].upper()
    cog_opts.setdefault("BIGTIFF", "IF_SAFER")
    return cog_opts


def write_cog(
    input_img: Union[str, os.PathLike],
    output_img: Union[str, os.PathLike],
    cog_opts: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Convert ``input_img`` (e.g. a tiled GeoTIFF) to a Cloud Optimised GeoTIFF
    with GDAL's COG driver, which builds the overviews and copies the band
    metadata (statistics, no-data value, descriptions, colour tables).

    The COG is written to a temporary file next to ``output_img`` and then
    moved into place, so an existing ``output_img`` (which may be
    ``input_img``'s source) is only replaced once the conversion has
    succeeded. Stale ``.aux.xml``/``.ovr``/``.msk`` sidecars of
    ``output_img`` are removed.

    :param input_img: Path to the image to convert (read from disk).
    :param output_img: Path to the output COG.
    :param cog_opts: COG creation options (Default: those of
                     :func:`resolve_cog_creation_opts`).

    .. code:: python

        import rsgispy.imageprocess

        rsgispy.imageprocess.write_cog(
            "mosaic_tiled.tif", "mosaic_cog.tif", {"COMPRESS": "DEFLATE"}
        )
    """
    out_path = os.fspath(output_img)
    eff_opts = resolve_cog_creation_opts() if cog_opts is None else dict(cog_opts)
    tmp_cog = _staging_path(out_path)
    try:
        rasterio.shutil.copy(
            os.fspath(input_img), tmp_cog, driver="COG", **eff_opts
        )
        os.replace(tmp_cog, out_path)
        for ext in _SIDECAR_EXTS:
            if os.path.exists(out_path + ext):
                os.remove(out_path + ext)
    finally:
        _remove_raster_files(tmp_cog)


def update_cog(
    input_img: Union[str, os.PathLike],
    update_func: Callable[[str], None],
    *,
    overview_resampling: Optional[str] = None,
) -> None:
    """
    Change an existing Cloud Optimised GeoTIFF without breaking its layout.

    A COG cannot be edited in place (GDAL refuses, or with
    ``IGNORE_COG_LAYOUT_BREAK=YES`` moves the image file directory to the end
    of the file and invalidates the layout). Instead, ``input_img`` is copied to
    a temporary tiled GeoTIFF next to it, ``update_func(tmp_path)`` edits that
    copy (e.g. adds statistics, a colour table or band names), and the copy is
    converted back to a COG with the original compression, predictor, block
    size and overview resampling (:func:`cog_creation_opts_from_file`), which
    replaces ``input_img``. If anything fails, ``input_img`` is left unchanged
    and the temporary files are removed.

    This decodes and re-encodes the whole image and rebuilds the overviews,
    so it needs temporary disk space for one (fast-compressed) copy of the
    image and takes roughly as long as creating the COG did. Lossy
    compression (JPEG, WEBP) is applied again.

    :param input_img: Path to an existing COG.
    :param update_func: Callable receiving the path of the temporary GeoTIFF.
    :param overview_resampling: Optional overview resampling for the rebuilt
                                COG (e.g. ``'NEAREST'``); by default the one
                                recorded in the file is used.

    .. code:: python

        import rsgispy.imageprocess
        from rsgispy.imageutils.metadata import set_band_names

        rsgispy.imageprocess.update_cog(
            "ndvi_cog.tif", lambda path: set_band_names(path, ["NDVI"])
        )
    """
    img_path = os.fspath(input_img)
    if not is_cog_layout(img_path):
        raise rsgispy.RSGISPyException(f"'{img_path}' is not a COG.")
    cog_opts = cog_creation_opts_from_file(img_path)
    if overview_resampling is not None:
        cog_opts["OVERVIEW_RESAMPLING"] = str(overview_resampling).upper()
    staging = _staging_path(img_path)
    try:
        rasterio.shutil.copy(
            img_path, staging, driver="GTiff", **_staging_gtiff_opts(cog_opts)
        )
        update_func(staging)
        write_cog(staging, img_path, cog_opts)
    finally:
        _remove_raster_files(staging)


class StagedOutput:
    """
    Context manager giving the path and format to write an output raster to,
    so outputs in formats that GDAL can only create by copying (COG) are
    written block by block on disk instead of in memory.

    * For ``gdalformat='COG'`` the data is written to a temporary tiled
      GeoTIFF next to ``output_img`` (:attr:`path`, :attr:`write_format`
      ``'GTIFF'`` and :attr:`write_opts`). When the context exits without an
      exception, the functions registered with :meth:`add_close_action` are
      called with :attr:`path` (e.g. to add statistics, which are then carried
      into the COG), the file is converted with :func:`write_cog` and the
      temporary file is deleted. On an exception the temporary file is
      deleted and ``output_img`` is not created or changed. If
      ``output_exists=True`` and ``output_img`` exists, it is first copied to
      the temporary file so it can be updated, and rebuilt with its own COG
      layout (:func:`cog_creation_opts_from_file`).
    * For other formats :attr:`path` is ``output_img``, :attr:`write_opts`
      are the resolved creation options (:func:`get_gdal_img_creation_opts`
      plus ``creation_opts``) and the close actions run on the output itself.

    An existing output that has the COG layout is also staged when
    ``output_exists=True``, whatever ``gdalformat`` is, so updating it keeps
    it a valid COG.

    :param output_img: Path to the final output image.
    :param gdalformat: Output GDAL format name (e.g. ``'GTIFF'``, ``'COG'``).
    :param creation_opts: Optional explicit creation options.
    :param output_exists: If ``True`` and ``output_img`` exists, it will be
                          updated rather than created.

    .. code:: python

        import rasterio
        import rsgispy.imageprocess
        import rsgispy.imageutils.blank

        with rsgispy.imageprocess.StagedOutput("out_cog.tif", "COG") as stage:
            rsgispy.imageutils.blank.create_copy_img(
                "ref.tif",
                stage.path,
                n_bands=1,
                pxl_val=0,
                gdalformat=stage.write_format,
                creation_opts=stage.write_opts,
            )
            with rasterio.open(stage.path, "r+") as dst:
                ...  # edit the file freely
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        output_img: Union[str, os.PathLike],
        gdalformat: str,
        creation_opts: Optional[Mapping[str, Any]] = None,
        output_exists: bool = False,
    ) -> None:
        self.output_img: str = os.fspath(output_img)
        self.gdalformat: str = str(gdalformat)
        self.update_existing: bool = bool(output_exists) and os.path.exists(
            self.output_img
        )
        existing_cog = self.update_existing and is_cog_layout(self.output_img)
        self.is_cog: bool = existing_cog or (
            self.gdalformat.upper().strip() == "COG"
        )
        self._close_actions: List[Callable[[str], None]] = []
        self._writer_ids: List[int] = []
        if self.is_cog:
            self.cog_opts: Optional[Dict[str, Any]] = (
                cog_creation_opts_from_file(self.output_img)
                if existing_cog
                else resolve_cog_creation_opts(creation_opts)
            )
            self.path: str = _staging_path(self.output_img)
            self.write_format: str = "GTIFF"
            self.write_opts: Dict[str, Any] = _staging_gtiff_opts(self.cog_opts)
        else:
            self.cog_opts = None
            self.path = self.output_img
            self.write_format = self.gdalformat
            self.write_opts = dict(get_gdal_img_creation_opts(self.gdalformat))
            if creation_opts:
                self.write_opts.update(creation_opts)

    def add_close_action(self, action: Callable[[str], None]) -> None:
        """
        Register ``action(path)`` to run when the context exits without an
        exception, in registration order, after any writer opened on
        :attr:`path` within the same ``ExitStack`` has closed and before a COG
        is converted.

        :param action: Callable receiving the path of the written file.
        """
        self._close_actions.append(action)

    def track_writer(self, dst: DatasetWriter) -> None:
        """
        Associate an open writer on :attr:`path` with this stage, so
        :func:`on_output_close` and :func:`get_output_stage` can find it while
        the stage is open.

        :param dst: Open rasterio writer on :attr:`path`.
        """
        _WRITER_STAGES[id(dst)] = self
        self._writer_ids.append(id(dst))

    def __enter__(self) -> "StagedOutput":
        if self.is_cog and self.update_existing:
            try:
                rasterio.shutil.copy(
                    self.output_img, self.path, driver="GTiff", **self.write_opts
                )
            except BaseException:
                _remove_raster_files(self.path)
                raise
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        for writer_id in self._writer_ids:
            _WRITER_STAGES.pop(writer_id, None)
        try:
            if exc_type is None:
                for action in self._close_actions:
                    action(self.path)
                if self.is_cog:
                    write_cog(self.path, self.output_img, self.cog_opts)
        finally:
            if self.is_cog:
                _remove_raster_files(self.path)
        return False


def on_output_close(dst: DatasetWriter, action: Callable[[str], None]) -> None:
    """
    Register ``action(path)`` to run once the writer ``dst`` returned by
    :func:`open_output_dataset` (or :func:`create_output_from_dataset`) has
    closed, i.e. when its ``ExitStack`` exits without an exception.

    ``path`` is the file that was written: the output itself, or for a COG
    output the temporary tiled GeoTIFF, which is converted to the COG after
    the actions have run. Use this for post-processing that must be in the
    final COG (statistics, colour tables, metadata) instead of opening the
    output again after the ``with`` block, which would break the COG layout.

    :param dst: Writer returned by :func:`open_output_dataset`.
    :param action: Callable receiving the path of the written file.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess
        from rsgispy.imageutils.metadata import set_band_names

        with ExitStack() as stack:
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out.tif", "COG", "float32", grid
            )
            rsgispy.imageprocess.on_output_close(
                dst, lambda path: set_band_names(path, ["ndvi"])
            )
            ...  # write blocks
    """
    stage = _WRITER_STAGES.get(id(dst))
    if stage is None:
        raise rsgispy.RSGISPyException(
            "on_output_close requires an open writer returned by "
            "open_output_dataset or create_output_from_dataset."
        )
    stage.add_close_action(action)


def get_output_stage(dst: DatasetWriter) -> Optional[StagedOutput]:
    """
    Return the :class:`StagedOutput` of an open writer returned by
    :func:`open_output_dataset`, or ``None`` for any other dataset.

    :param dst: Open rasterio writer.
    :return: The writer's :class:`StagedOutput` or ``None``.
    """
    return _WRITER_STAGES.get(id(dst))


def open_input_datasets(
    stack: ExitStack, norm_defs: Sequence[BandDefn]
) -> Dict[str, DatasetReader]:
    """
    Open and validate all unique input raster datasets within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param norm_defs: Validated sequence of BandDefn objects.
    :return: Mapping of file path to opened rasterio DatasetReader.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        band_defs = [rsgispy.imageprocess.BandDefn("b1", "img.tif", 1)]
        with ExitStack() as stack:
            datasets = rsgispy.imageprocess.open_input_datasets(stack, band_defs)
    """
    datasets: Dict[str, DatasetReader] = {}
    for b_def in norm_defs:
        if b_def.input_img is None or b_def.img_band is None:
            raise rsgispy.RSGISPyException(
                "BandDefn must define both input_img and img_band."
            )
        if b_def.input_img not in datasets:
            try:
                datasets[b_def.input_img] = stack.enter_context(
                    rasterio.open(b_def.input_img, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{b_def.input_img}': {err}"
                ) from err

        ds = datasets[b_def.input_img]
        if b_def.img_band > ds.count:
            raise rsgispy.RSGISPyException(
                f"Requested band {b_def.img_band} exceeds band count ({ds.count}) "
                f"for image '{b_def.input_img}'."
            )
    return datasets


def open_multi_image_datasets(
    stack: ExitStack,
    input_imgs: Sequence[Union[str, os.PathLike]],
) -> Tuple[Dict[str, DatasetReader], List[str]]:
    """
    Open and validate all unique input raster datasets from a sequence of
    image paths within an ExitStack.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param input_imgs: Sequence of input raster file paths.
    :return: Tuple of ``(datasets, paths)`` where ``datasets`` maps each unique
             file path to an opened rasterio DatasetReader and ``paths`` is the
             resolved list of file paths preserving the input order.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, paths = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["a.tif", "b.tif"]
            )
    """
    if not input_imgs:
        raise rsgispy.RSGISPyException("input_imgs must not be empty.")
    paths: List[str] = []
    datasets: Dict[str, DatasetReader] = {}
    for img in input_imgs:
        img_path = os.fspath(img)
        if not os.path.exists(img_path):
            raise rsgispy.RSGISPyException(
                f"Input image file does not exist: '{img_path}'."
            )
        paths.append(img_path)
        if img_path not in datasets:
            try:
                datasets[img_path] = stack.enter_context(
                    rasterio.open(img_path, "r")
                )
            except Exception as err:
                raise rsgispy.RSGISPyException(
                    f"Could not open input image '{img_path}': {err}"
                ) from err
    return datasets, paths


def open_output_dataset(
    stack: ExitStack,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    grid: IntersectionGrid,
    count: int = 1,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an existing output raster for update or create a new output raster.

    New files get the creation options of
    :func:`rsgispy.imageprocess.get_gdal_img_creation_opts` for ``gdalformat``
    (tiled, DEFLATE-compressed GeoTIFF unless configured otherwise with the
    ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` or
    ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` environment variables), with any
    explicit ``creation_opts`` merged on top.

    For ``gdalformat='COG'`` the returned writer is a temporary tiled GeoTIFF
    next to ``output_img``, so the raster is written block by block on disk
    rather than in memory. When ``stack`` exits without an exception, the
    temporary file is converted to the COG (``creation_opts`` and the
    ``*_IMG_CRT_OPTS_COG`` variables are passed to the COG driver, which also
    builds the overviews) and deleted; on an exception it is deleted and the
    output is not created. Metadata set on the writer (no-data value, band
    descriptions, tags, colour tables) is carried into the COG. Use
    :func:`on_output_close` (or :func:`rsgispy.imageprocess.pop_stats_on_close`)
    for anything that has to be done on the closed file before the
    conversion, such as statistics. Existing COGs opened with
    ``output_exists=True`` are updated through a temporary copy in the same
    way, so they stay valid COGs.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., 'GTIFF').
    :param out_rio_dtype: Rasterio dtype string (e.g., 'float32').
    :param grid: IntersectionGrid defining output dimensions, transform, and CRS.
    :param count: Number of output bands when creating a new file (Default = 1).
    :param output_exists: If True and the file exists, open in 'r+' mode.
    :param creation_opts: Optional mapping of GDAL creation options (e.g.,
                          ``{'COMPRESS': 'DEFLATE', 'TILED': 'YES'}``).
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rsgispy.imageprocess

        with ExitStack() as stack:
            datasets, _ = rsgispy.imageprocess.open_multi_image_datasets(
                stack, ["in.tif"]
            )
            grid = rsgispy.imageprocess.compute_spatial_intersection(datasets)
            dst = rsgispy.imageprocess.open_output_dataset(
                stack, "out.tif", "GTIFF", "float32", grid, count=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # The stage is entered before the writer, so it exits after the writer
    # has closed (ExitStack callbacks run in reverse order).
    stage = stack.enter_context(
        StagedOutput(
            output_img,
            gdalformat,
            creation_opts=creation_opts,
            output_exists=output_exists,
        )
    )
    if stage.update_existing:
        dst = stack.enter_context(rasterio.open(stage.path, "r+"))
        if dst.width != grid.width or dst.height != grid.height:
            raise rsgispy.RSGISPyException(
                f"Existing output image '{output_img}' dimensions "
                f"({dst.width}x{dst.height}) do not match calculated "
                f"output dimensions ({grid.width}x{grid.height})."
            )
    else:
        profile: Dict[str, Any] = {
            "driver": stage.write_format,
            "height": grid.height,
            "width": grid.width,
            "count": count,
            "dtype": out_rio_dtype,
            "crs": grid.crs,
            "transform": grid.transform,
            **stage.write_opts,
        }
        dst = stack.enter_context(rasterio.open(stage.path, "w", **profile))
    stage.track_writer(dst)
    return dst


def create_output_from_dataset(
    stack: ExitStack,
    src: DatasetReader,
    output_img: str,
    gdalformat: str,
    out_rio_dtype: str,
    count: Optional[int] = None,
    output_exists: bool = False,
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> DatasetWriter:
    """
    Open an output raster matching the spatial extent, transform, and CRS of an
    opened reference dataset ``src``. Creation options and COG outputs are
    handled as in :func:`open_output_dataset`.

    :param stack: Contextlib ExitStack managing open dataset handles.
    :param src: Reference opened rasterio DatasetReader.
    :param output_img: Path to the output raster file.
    :param gdalformat: GDAL/rasterio driver name (e.g., ``'GTIFF'``).
    :param out_rio_dtype: Rasterio dtype string (e.g., ``'float32'``).
    :param count: Number of output bands when creating a new file
                  (defaults to ``src.count`` if ``None``).
    :param output_exists: If ``True`` and the file exists, open in ``'r+'`` mode.
    :param creation_opts: Optional mapping of GDAL creation options.
    :return: Opened rasterio DatasetWriter.

    .. code:: python

        from contextlib import ExitStack
        import rasterio
        import rsgispy.imageprocess

        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open("in.tif", "r"))
            dst = rsgispy.imageprocess.create_output_from_dataset(
                stack, src, "out.tif", "GTIFF", "float32", count=1
            )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid = IntersectionGrid.from_dataset(src)
    return open_output_dataset(
        stack=stack,
        output_img=output_img,
        gdalformat=gdalformat,
        out_rio_dtype=out_rio_dtype,
        grid=grid,
        count=int(src.count) if count is None else int(count),
        output_exists=output_exists,
        creation_opts=creation_opts,
    )
