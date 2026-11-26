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
import posixpath
import tempfile
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
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
# Options of GDAL's COG driver that GDAL records under the same name in the
# IMAGE_STRUCTURE metadata of a COG and that are carried over when one is
# rebuilt (COMPRESS, PREDICTOR, INTERLEAVE and the quality settings are
# translated separately in cog_creation_opts_from_file).
_COG_RECORDED_OPTS = (
    "OVERVIEW_RESAMPLING",
    "MAX_Z_ERROR",
    "MAX_Z_ERROR_OVERVIEW",
    "JXL_EFFORT",
    "JXL_DISTANCE",
    "JXL_ALPHA_DISTANCE",
)
_RESAMPLING_OPTS = frozenset({"RESAMPLING", "OVERVIEW_RESAMPLING"})
# id(writer) -> StagedOutput for writers opened by open_output_dataset.
_WRITER_STAGES: Dict[int, "StagedOutput"] = {}
# Absolute paths of the temporary GeoTIFFs of open StagedOutput contexts.
_ACTIVE_STAGING_PATHS: Set[str] = set()


def _is_vsi_path(img_path: str) -> bool:
    """Return ``True`` for GDAL virtual file system paths (``/vsimem/`` etc.)."""
    return str(img_path).startswith("/vsi")


def _staging_path(output_img: str) -> str:
    """
    Return a unique temporary file path in the directory of ``output_img``,
    or in the local temporary directory for GDAL virtual file system paths
    (``/vsimem/``, ``/vsis3/``, ...), which cannot be written block by block or
    renamed with ``os`` functions.
    """
    if _is_vsi_path(output_img):
        out_dir = tempfile.gettempdir()
        base = posixpath.basename(str(output_img).rstrip("/")) or "img"
    else:
        out_dir, base = os.path.split(os.path.abspath(output_img))
    return os.path.join(out_dir, f".{base}.{uuid.uuid4().hex[:12]}{_STAGING_SUFFIX}")


def _is_active_staging_path(img_path: str) -> bool:
    """Return ``True`` if ``img_path`` is the temporary file of an open stage."""
    return (not _is_vsi_path(img_path)) and (
        os.path.abspath(img_path) in _ACTIVE_STAGING_PATHS
    )


def _merge_img_creation_opts(
    default_opts: Mapping[str, Any],
    creation_opts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge explicit ``creation_opts`` on top of ``default_opts`` (the resolved
    built-in or environment variable options), comparing keys
    case-insensitively so an explicit option always replaces the default one.

    If the explicit options set ``BLOCKYSIZE`` without ``TILED`` or
    ``BLOCKXSIZE`` (i.e. a striped GeoTIFF with that many rows per strip),
    the default ``TILED`` and ``BLOCKXSIZE`` are dropped so the output is
    striped rather than failing or being tiled.
    """
    explicit = dict(creation_opts) if creation_opts else {}
    explicit_keys = {str(key).upper().strip() for key in explicit}
    drop_keys = set(explicit_keys)
    if "BLOCKYSIZE" in explicit_keys and not explicit_keys & {"TILED", "BLOCKXSIZE"}:
        drop_keys.update({"TILED", "BLOCKXSIZE"})
    merged: Dict[str, Any] = {
        key: val
        for key, val in default_opts.items()
        if str(key).upper().strip() not in drop_keys
    }
    merged.update(explicit)
    return merged


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
    Return COG creation options that reproduce the layout of an existing COG
    from the settings GDAL records in its ``IMAGE_STRUCTURE`` metadata: the
    compression (``COMPRESS``, ``NONE`` if uncompressed), predictor, block
    size, overview resampling, interleave, ``NBITS`` and the lossy
    compression settings (``QUALITY`` from ``JPEG_QUALITY`` or
    ``WEBP_LEVEL``, ``QUALITY=100`` for lossless WEBP, ``MAX_Z_ERROR`` for
    LERC and the ``JXL_*`` options). Options that are not recorded in the
    file (e.g. ``LEVEL``, or ``QUALITY`` if GDAL could not read it) come
    from :func:`resolve_cog_creation_opts`.

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
        band_struct = {
            str(k).upper(): str(v)
            for k, v in src.tags(1, ns="IMAGE_STRUCTURE").items()
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
    for opt_name in _COG_RECORDED_OPTS:
        if opt_name in struct:
            cog_opts[opt_name] = struct[opt_name]
    reversibility = struct.get("COMPRESSION_REVERSIBILITY", "").upper()
    if cog_opts["COMPRESS"] == "JPEG" and "JPEG_QUALITY" in struct:
        cog_opts["QUALITY"] = struct["JPEG_QUALITY"]
    elif cog_opts["COMPRESS"] == "WEBP":
        if "WEBP_LEVEL" in struct:
            cog_opts["QUALITY"] = struct["WEBP_LEVEL"]
        elif reversibility == "LOSSLESS":
            # The COG driver writes lossless WEBP for QUALITY=100.
            cog_opts["QUALITY"] = "100"
    elif cog_opts["COMPRESS"] == "JXL" and reversibility in ("LOSSLESS", "LOSSY"):
        cog_opts["JXL_LOSSLESS"] = "YES" if reversibility == "LOSSLESS" else "NO"
    if "NBITS" in band_struct:
        cog_opts["NBITS"] = band_struct["NBITS"]
    if blk_w == blk_h:
        cog_opts["BLOCKSIZE"] = int(blk_w)
    if struct.get("INTERLEAVE", "PIXEL").upper() in ("BAND", "TILE"):
        cog_opts["INTERLEAVE"] = struct["INTERLEAVE"].upper()
    cog_opts.setdefault("BIGTIFF", "IF_SAFER")
    return cog_opts


def _is_thematic_img(img_path: str) -> bool:
    """Return ``True`` if every band of ``img_path`` has ``LAYER_TYPE=thematic``."""
    with rasterio.open(img_path, "r") as src:
        return src.count > 0 and all(
            str(src.tags(bidx).get("LAYER_TYPE", "")).lower() == "thematic"
            for bidx in src.indexes
        )


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
    succeeded. Sidecars written by GDAL with the COG (e.g. an ``.aux.xml``
    holding a raster attribute table or category names) are moved with it,
    and stale ``.aux.xml``/``.ovr``/``.msk`` sidecars of ``output_img`` that
    have no replacement are removed. A GDAL virtual file system path
    (``/vsimem/``, ``/vsis3/``, ...) is written directly by GDAL.

    If every band of ``input_img`` is thematic (``LAYER_TYPE=thematic``) and
    ``cog_opts`` set neither ``OVERVIEW_RESAMPLING`` nor ``RESAMPLING``, the
    overviews are built with ``NEAREST`` resampling.

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
    in_path = os.fspath(input_img)
    out_path = os.fspath(output_img)
    eff_opts = resolve_cog_creation_opts() if cog_opts is None else dict(cog_opts)
    if not {str(key).upper() for key in eff_opts} & _RESAMPLING_OPTS:
        if _is_thematic_img(in_path):
            eff_opts["OVERVIEW_RESAMPLING"] = "NEAREST"
    if _is_vsi_path(out_path):
        rasterio.shutil.copy(in_path, out_path, driver="COG", **eff_opts)
        return
    tmp_cog = _staging_path(out_path)
    try:
        rasterio.shutil.copy(in_path, tmp_cog, driver="COG", **eff_opts)
        os.replace(tmp_cog, out_path)
        for ext in _SIDECAR_EXTS:
            if os.path.exists(tmp_cog + ext):
                os.replace(tmp_cog + ext, out_path + ext)
            elif os.path.exists(out_path + ext):
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
    size, overview resampling and lossy compression settings (e.g. JPEG or
    WEBP quality, LERC maximum error; see :func:`cog_creation_opts_from_file`),
    which replaces ``input_img``. Metadata GDAL keeps in an ``.aux.xml``
    sidecar (e.g. a raster attribute table or category names) is kept. If
    anything fails, ``input_img`` is left unchanged and the temporary files
    are removed.

    This decodes and re-encodes the whole image and rebuilds the overviews,
    so it needs temporary disk space for one (fast-compressed) copy of the
    image and takes roughly as long as creating the COG did. Lossy
    compression (JPEG, WEBP) is applied again, with the same quality.

    :param input_img: Path to an existing COG.
    :param update_func: Callable receiving the path of the temporary GeoTIFF.
    :param overview_resampling: Optional overview resampling for the rebuilt
                                COG (e.g. ``'NEAREST'``); by default the one
                                recorded in the file is used. Some GDAL
                                versions do not record it, in which case the
                                COG driver's default is used unless it is
                                given here (thematic images always get
                                ``'NEAREST'`` or keep ``'MODE'``).

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
    _ACTIVE_STAGING_PATHS.add(staging)
    try:
        rasterio.shutil.copy(
            img_path, staging, driver="GTiff", **_staging_gtiff_opts(cog_opts)
        )
        update_func(staging)
        write_cog(staging, img_path, cog_opts)
    finally:
        _ACTIVE_STAGING_PATHS.discard(staging)
        _remove_raster_files(staging)


class StagedOutput:
    """
    Context manager giving the path and format to write an output raster to,
    so outputs in formats that GDAL can only create by copying (COG) are
    written block by block on disk instead of in memory.

    * For ``gdalformat='COG'`` the data is written to a temporary tiled
      GeoTIFF (:attr:`path`, :attr:`write_format` ``'GTIFF'``) next to
      ``output_img``, or in the local temporary directory for GDAL virtual
      file system paths such as ``/vsimem/`` or ``/vsis3/``. Its
      :attr:`write_opts` are fixed staging options; the ``GTIFF`` defaults
      and environment variables do not apply to it, also when :attr:`path`
      is passed on to a helper that creates it with
      :func:`open_output_dataset`. When the context exits without an
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
      with ``creation_opts`` merged on top, see :func:`open_output_dataset`)
      and the close actions run on the output itself. If ``output_img`` is
      the temporary file of another open stage (e.g. passed on to a helper
      that writes it), :attr:`write_opts` are ``creation_opts`` only.

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
        # True if cog_opts were read from an existing COG being updated.
        self.existing_cog: bool = existing_cog
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
            if _is_active_staging_path(self.output_img):
                # The temporary GeoTIFF of a COG output only gets the staging
                # options, never the user's GTIFF defaults or variables.
                self.write_opts = dict(creation_opts) if creation_opts else {}
            else:
                self.write_opts = _merge_img_creation_opts(
                    get_gdal_img_creation_opts(self.gdalformat), creation_opts
                )

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
        if self.is_cog:
            _ACTIVE_STAGING_PATHS.add(self.path)
        if self.is_cog and self.update_existing:
            try:
                rasterio.shutil.copy(
                    self.output_img, self.path, driver="GTiff", **self.write_opts
                )
            except BaseException:
                _ACTIVE_STAGING_PATHS.discard(self.path)
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
                _ACTIVE_STAGING_PATHS.discard(self.path)
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
    explicit ``creation_opts`` merged on top (keys compared
    case-insensitively, so an explicit option replaces the default one). If
    ``creation_opts`` set ``BLOCKYSIZE`` but neither ``TILED`` nor
    ``BLOCKXSIZE`` (e.g. ``{'BLOCKYSIZE': 1}`` for one row per strip), the
    default ``TILED`` and ``BLOCKXSIZE`` are dropped and a striped GeoTIFF is
    written.

    For ``gdalformat='COG'`` the returned writer is a temporary tiled GeoTIFF
    next to ``output_img`` (in the local temporary directory for GDAL virtual
    file system paths such as ``/vsimem/`` or ``/vsis3/``), so the raster is
    written block by block on disk rather than in memory. When ``stack``
    exits without an exception, the temporary file is converted to the COG
    (``creation_opts`` and the ``*_IMG_CRT_OPTS_COG`` variables are passed to
    the COG driver, which also builds the overviews) and deleted; on an
    exception it is deleted and the output is not created. Metadata set on
    the writer (no-data value, band descriptions, tags, colour tables) is
    carried into the COG. Use
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
