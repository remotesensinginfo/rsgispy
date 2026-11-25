"""
The imageutils.envopts module provides functions for configuring the GDAL
creation options that rsgispy uses for new output images.

**Defaults.** With nothing configured, GeoTIFF (``gdalformat="GTIFF"``)
outputs are tiled (GDAL's default 256 x 256 tiles) and DEFLATE-compressed:
``TILED=YES``, ``COMPRESS=DEFLATE``, ``INTERLEAVE=BAND`` and
``BIGTIFF=IF_SAFER``, as listed in ``DEFAULT_IMG_CREATION_OPTS`` (the same
object as ``rsgispy.imageprocess.DEFAULT_IMG_CREATION_OPTS``). Other formats
(``'COG'``, ``'KEA'``, ...) have no built-in options, so their GDAL driver
defaults apply. These defaults are built into
:func:`get_gdal_img_creation_opts`; rsgispy does not set any environment
variables when it is imported, so other software in the same process (such as
RSGISLib) is not affected.

**Order of precedence** for a format ``<FORMAT>``:

1. the ``RSGISPY_IMG_CRT_OPTS_<FORMAT>`` environment variable, if set;
2. otherwise ``RSGISLIB_IMG_CRT_OPTS_<FORMAT>``, if set;
3. otherwise the built-in default.

A variable that is set replaces the default (it is not merged with it), and a
variable set to an empty string means "no creation options". Explicit
``creation_opts`` passed to an rsgispy function are merged on top.

**Changing the defaults.** Call :func:`set_img_creation_opts` (or one of the
``set_env_vars_*`` helpers), e.g. ``set_img_creation_opts("GTIFF", {})`` for
plain striped, uncompressed GeoTIFFs, and :func:`reset_img_creation_opts` to
go back to the built-in defaults. These helpers set both the ``RSGISPY_*``
and the ``RSGISLIB_*`` variables, so RSGISLib functions called later in the
same process see the same options.
"""

import os
from typing import Any, Mapping, Optional

from rsgispy.imageprocess import (
    DEFAULT_IMG_CREATION_OPTS,
    ENV_VAR_IMG_CRT_OPTS_PREFIX,
    ENV_VAR_RSGISLIB_IMG_CRT_OPTS_PREFIX,
    get_gdal_img_creation_opts,
)

__all__ = [
    "DEFAULT_IMG_CREATION_OPTS",
    "get_gdal_img_creation_opts",
    "set_img_creation_opts",
    "reset_img_creation_opts",
    "set_env_vars_lzw_gtiff_outs",
    "set_env_vars_deflate_gtiff_outs",
    "set_env_vars_zstd_gtiff_outs",
    "set_env_vars_lzw_cog_outs",
    "set_env_vars_deflate_cog_outs",
    "set_env_vars_zstd_cog_outs",
]


def set_img_creation_opts(
    gdalformat: str, creation_opts: Mapping[str, Any]
) -> None:
    """
    Set the environment variables ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` and
    ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` from a mapping of creation options.

    The options replace the built-in defaults for that format (they are not
    merged with them). An empty mapping opts out of all creation options, so
    ``set_img_creation_opts("GTIFF", {})`` gives plain striped, uncompressed
    GeoTIFFs. Use :func:`reset_img_creation_opts` to restore the defaults.

    :param gdalformat: GDAL format string (e.g., 'GTIFF', 'COG').
    :param creation_opts: Mapping of GDAL creation option keys and values.

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_img_creation_opts(
            "GTIFF", {"TILED": "YES", "COMPRESS": "LZW"}
        )
        # Plain (striped, uncompressed) GeoTIFFs:
        rsgispy.imageutils.envopts.set_img_creation_opts("GTIFF", {})
    """
    fmt_upper = gdalformat.upper().strip()
    opts_str = ":".join(f"{k}={v}" for k, v in creation_opts.items())
    os.environ[f"{ENV_VAR_RSGISLIB_IMG_CRT_OPTS_PREFIX}{fmt_upper}"] = opts_str
    os.environ[f"{ENV_VAR_IMG_CRT_OPTS_PREFIX}{fmt_upper}"] = opts_str


def reset_img_creation_opts(gdalformat: Optional[str] = None) -> None:
    """
    Clear the ``RSGISPY_IMG_CRT_OPTS_<GDAL_FORMAT>`` and
    ``RSGISLIB_IMG_CRT_OPTS_<GDAL_FORMAT>`` environment variables so the
    built-in defaults (``DEFAULT_IMG_CREATION_OPTS``) are used again.

    :param gdalformat: GDAL format string (e.g., 'GTIFF'). If ``None``
                       (default), the variables of every format are cleared.

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_img_creation_opts("GTIFF", {})
        # ... write plain GeoTIFFs ...
        rsgispy.imageutils.envopts.reset_img_creation_opts("GTIFF")
    """
    prefixes = (ENV_VAR_IMG_CRT_OPTS_PREFIX, ENV_VAR_RSGISLIB_IMG_CRT_OPTS_PREFIX)
    if gdalformat is None:
        for env_name in list(os.environ):
            if env_name.startswith(prefixes):
                os.environ.pop(env_name, None)
        return
    fmt_upper = gdalformat.upper().strip()
    for prefix in prefixes:
        os.environ.pop(f"{prefix}{fmt_upper}", None)


def set_env_vars_lzw_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the LZW algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts
        rsgispy.imageutils.envopts.set_env_vars_lzw_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"TILED=YES:COMPRESS=LZW:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_deflate_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the Deflate (zlib) algorithm.

    With ``bigtiff=False`` these are the same options as rsgispy's built-in
    GeoTIFF default (``DEFAULT_IMG_CREATION_OPTS``), so this is only needed to
    pass the options on to RSGISLib in the same process, to request BigTIFF,
    or to replace options set earlier with :func:`set_img_creation_opts`
    (which :func:`reset_img_creation_opts` also does).

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts
        rsgispy.imageutils.envopts.set_env_vars_deflate_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = (
        f"TILED=YES:COMPRESS=DEFLATE:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    )
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_zstd_gtiff_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted GeoTIFF files are created
    tiled and compressed using the ZSTD algorithm.

    :param bigtiff: If True, GTIFF files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts
        rsgispy.imageutils.envopts.set_env_vars_zstd_gtiff_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = (
        f"TILED=YES:COMPRESS=ZSTD:INTERLEAVE=BAND:BIGTIFF={bigtiff_val}"
    )
    os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_GTIFF"] = opts_str


def set_env_vars_lzw_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the LZW algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_lzw_cog_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=LZW:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str


def set_env_vars_deflate_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the Deflate algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_deflate_cog_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=DEFLATE:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str


def set_env_vars_zstd_cog_outs(bigtiff: bool = False) -> None:
    """
    Set environment variables such that outputted Cloud Optimized GeoTIFF (COG)
    files are compressed using the ZSTD algorithm.

    :param bigtiff: If True, COG files will be created as BigTIFF (``BIGTIFF=YES``),
                    otherwise ``BIGTIFF=IF_SAFER`` (Default = False).

    .. code:: python

        import rsgispy.imageutils.envopts

        rsgispy.imageutils.envopts.set_env_vars_zstd_cog_outs()
    """
    bigtiff_val = "YES" if bigtiff else "IF_SAFER"
    opts_str = f"COMPRESS=ZSTD:BIGTIFF={bigtiff_val}"
    os.environ["RSGISLIB_IMG_CRT_OPTS_COG"] = opts_str
    os.environ["RSGISPY_IMG_CRT_OPTS_COG"] = opts_str
