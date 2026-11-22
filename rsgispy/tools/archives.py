"""
The tools.archives module provides utilities for creating and extracting
zip and tar archives.
"""

import os
import stat
import sys
import tarfile
from typing import Optional, Sequence, Tuple
import zipfile

import rsgispy
import rsgispy.tools.files

__all__ = [
    "create_directory_archive",
    "untar_file",
    "untar_gz_file",
    "unzip_file",
    "untar_bz_file",
    "create_targz_arch",
]


def _validate_extract_member_path(real_dest: str, member_name: str) -> str:
    """Validate that an archive member resolves inside ``real_dest``."""
    if not member_name or os.path.isabs(member_name):
        raise rsgispy.RSGISPyException(
            f"Unsafe archive member path detected: '{member_name}'"
        )
    drive, _ = os.path.splitdrive(member_name)
    if drive:
        raise rsgispy.RSGISPyException(
            f"Unsafe archive member path detected: '{member_name}'"
        )
    target_path = os.path.realpath(os.path.join(real_dest, member_name))
    try:
        common = os.path.commonpath([real_dest, target_path])
    except ValueError as err:
        raise rsgispy.RSGISPyException(
            f"Unsafe archive member path detected: '{member_name}'"
        ) from err
    if common != real_dest:
        raise rsgispy.RSGISPyException(
            f"Unsafe archive member path escapes destination: '{member_name}'"
        )
    return target_path


def _prepare_extract_dir(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool,
    verbose: bool,
) -> Tuple[str, str]:
    """Validate archive inputs and create the target extraction directory."""
    if not os.path.exists(out_dir):
        raise rsgispy.RSGISPyException("Output directory does not exist.")
    if not os.path.exists(in_file):
        raise rsgispy.RSGISPyException("Input file does not exist.")

    in_file_abs = os.path.abspath(in_file)
    out_dir_abs = os.path.abspath(out_dir)
    process_dir = out_dir_abs
    if gen_arch_dir:
        basename = rsgispy.tools.files.get_file_basename(in_file_abs)
        process_dir = os.path.join(out_dir_abs, basename)
    os.makedirs(process_dir, exist_ok=True)

    if verbose:
        print(f"Extracting: {in_file_abs}")
        print(f"Output to: {process_dir}")
    return in_file_abs, process_dir


def _validate_tar_member(real_dest: str, member: tarfile.TarInfo) -> None:
    """Validate a tar archive member against path traversal and unsafe links."""
    target_path = _validate_extract_member_path(real_dest, member.name)
    if member.isdev():
        raise rsgispy.RSGISPyException(
            f"Unsafe special/device archive member rejected: '{member.name}'"
        )
    if member.issym():
        if os.path.isabs(member.linkname):
            raise rsgispy.RSGISPyException(
                f"Unsafe absolute symlink target rejected: '{member.name}'"
            )
        link_target = os.path.realpath(
            os.path.join(os.path.dirname(target_path), member.linkname)
        )
        try:
            common = os.path.commonpath([real_dest, link_target])
        except ValueError as err:
            raise rsgispy.RSGISPyException(
                f"Unsafe symlink target escapes destination: '{member.name}'"
            ) from err
        if common != real_dest:
            raise rsgispy.RSGISPyException(
                f"Unsafe symlink target escapes destination: '{member.name}'"
            )
    elif member.islnk():
        _validate_extract_member_path(real_dest, member.linkname)


def _extract_tar_archive(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool,
    verbose: bool,
    tar_mode: str,
) -> str:
    """Safely extract a tar archive using Python's ``tarfile`` module."""
    in_file_abs, process_dir = _prepare_extract_dir(
        in_file, out_dir, gen_arch_dir, verbose
    )
    real_dest = os.path.realpath(process_dir)

    try:
        with tarfile.open(in_file_abs, tar_mode) as tar_obj:
            members = tar_obj.getmembers()
            for member in members:
                _validate_tar_member(real_dest, member)
                if verbose:
                    print(member.name)
            if sys.version_info >= (3, 12):
                tar_obj.extractall(
                    path=process_dir, members=members, filter="data"
                )
            else:
                tar_obj.extractall(path=process_dir, members=members)
    except rsgispy.RSGISPyException:
        raise
    except (tarfile.TarError, OSError) as err:
        raise rsgispy.RSGISPyException(
            f"Could not extract tar archive '{in_file_abs}': {err}"
        ) from err

    return process_dir


def create_directory_archive(
    in_dir: str, out_arch: str, arch_format: str
) -> str:
    """
    Create an archive file from an input directory using Python's standard
    ``tarfile`` and ``zipfile`` libraries.

    :param in_dir: Input directory path to archive.
    :param out_arch: Output archive path without file extension.
    :param arch_format: One of ``'zip'``, ``'tar'``, ``'gztar'``, ``'bztar'``,
                        or ``'xztar'``.
    :return: Full output archive file path including extension.

    .. code:: python

        import rsgispy.tools.archives
        arch_path = rsgispy.tools.archives.create_directory_archive(
            "data_dir", "data_backup", "gztar"
        )
    """
    # pylint: disable=too-many-locals
    if not rsgispy.tools.files.is_path_valid(in_dir):
        raise rsgispy.RSGISPyException(
            f"The input directory is not valid: {in_dir}"
        )
    if not os.path.isdir(in_dir):
        raise rsgispy.RSGISPyException(
            f"The input directory does not exist: {in_dir}"
        )

    out_arch_abs = os.path.abspath(out_arch.strip())
    in_dir_abs = os.path.abspath(in_dir)
    dir_name = os.path.split(in_dir_abs)[1]

    format_specs = {
        "zip": (".zip", "zip"),
        "tar": (".tar", "w"),
        "gztar": (".tar.gz", "w:gz"),
        "bztar": (".tar.bz2", "w:bz2"),
        "xztar": (".tar.xz", "w:xz"),
    }
    if arch_format not in format_specs:
        raise rsgispy.RSGISPyException(
            "Do not recognise the archive format specifed."
        )

    ext, mode = format_specs[arch_format]
    out_arch_file = f"{out_arch_abs}{ext}"
    if not rsgispy.tools.files.does_path_exists_or_creatable(out_arch_file):
        raise rsgispy.RSGISPyException(
            f"Output file path is not creatable: {out_arch_file}"
        )

    try:
        if mode == "zip":
            with zipfile.ZipFile(
                out_arch_file, "w", compression=zipfile.ZIP_DEFLATED
            ) as zip_out:
                for root, dirs, files in os.walk(in_dir_abs):
                    rel_root = os.path.relpath(root, in_dir_abs)
                    arc_root = (
                        dir_name
                        if rel_root == "."
                        else os.path.join(dir_name, rel_root)
                    )
                    if not files and not dirs:
                        zip_out.writestr(f"{arc_root}/", "")
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        zip_out.write(fpath, arcname=os.path.join(arc_root, fname))
        else:
            with tarfile.open(out_arch_file, mode) as tar_out:
                tar_out.add(in_dir_abs, arcname=dir_name)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as err:
        raise rsgispy.RSGISPyException(
            f"Could not create archive '{out_arch_file}': {err}"
        ) from err

    return out_arch_file


def untar_file(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool = True,
    verbose: bool = False,
) -> str:
    """
    Extract a ``.tar`` archive into ``out_dir``.

    :param in_file: Input archive file path.
    :param out_dir: Existing destination directory.
    :param gen_arch_dir: Create a subdirectory named after ``in_file`` (Default = True).
    :param verbose: Print progress messages and extracted member names (Default = False).
    :return: Output directory where files were extracted.

    .. code:: python

        import rsgispy.tools.archives
        out_path = rsgispy.tools.archives.untar_file("archive.tar", "extracted/")
    """
    return _extract_tar_archive(in_file, out_dir, gen_arch_dir, verbose, "r:")


def untar_gz_file(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool = True,
    verbose: bool = False,
) -> str:
    """
    Extract a ``.tar.gz`` archive into ``out_dir``.

    :param in_file: Input archive file path.
    :param out_dir: Existing destination directory.
    :param gen_arch_dir: Create a subdirectory named after ``in_file`` (Default = True).
    :param verbose: Print progress messages and extracted member names (Default = False).
    :return: Output directory where files were extracted.

    .. code:: python

        import rsgispy.tools.archives
        out_path = rsgispy.tools.archives.untar_gz_file("archive.tar.gz", "extracted/")
    """
    return _extract_tar_archive(in_file, out_dir, gen_arch_dir, verbose, "r:gz")


def unzip_file(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool = True,
    verbose: bool = False,
) -> str:
    """
    Extract a ``.zip`` archive into ``out_dir``.

    :param in_file: Input archive file path.
    :param out_dir: Existing destination directory.
    :param gen_arch_dir: Create a subdirectory named after ``in_file`` (Default = True).
    :param verbose: Print progress messages and extracted member names (Default = False).
    :return: Output directory where files were extracted.

    .. code:: python

        import rsgispy.tools.archives
        out_path = rsgispy.tools.archives.unzip_file("archive.zip", "extracted/")
    """
    in_file_abs, process_dir = _prepare_extract_dir(
        in_file, out_dir, gen_arch_dir, verbose
    )
    real_dest = os.path.realpath(process_dir)

    try:
        with zipfile.ZipFile(in_file_abs, "r") as zip_obj:
            infos = zip_obj.infolist()
            for info in infos:
                _validate_extract_member_path(real_dest, info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise rsgispy.RSGISPyException(
                        f"Unsafe symlink member rejected in zip archive: '{info.filename}'"
                    )
                if verbose:
                    print(info.filename)
            zip_obj.extractall(path=process_dir)
    except rsgispy.RSGISPyException:
        raise
    except (zipfile.BadZipFile, OSError) as err:
        raise rsgispy.RSGISPyException(
            f"Could not extract zip archive '{in_file_abs}': {err}"
        ) from err

    return process_dir


def untar_bz_file(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool = True,
    verbose: bool = False,
) -> str:
    """
    Extract a ``.tar.bz2`` archive into ``out_dir``.

    :param in_file: Input archive file path.
    :param out_dir: Existing destination directory.
    :param gen_arch_dir: Create a subdirectory named after ``in_file`` (Default = True).
    :param verbose: Print progress messages and extracted member names (Default = False).
    :return: Output directory where files were extracted.

    .. code:: python

        import rsgispy.tools.archives
        out_path = rsgispy.tools.archives.untar_bz_file("archive.tar.bz2", "extracted/")
    """
    return _extract_tar_archive(in_file, out_dir, gen_arch_dir, verbose, "r:bz2")


def create_targz_arch(
    out_arch_file: str,
    file_list: Sequence[str],
    base_path: Optional[str] = None,
) -> None:
    """
    Create a ``.tar.gz`` archive containing ``file_list``.

    :param out_arch_file: Output ``.tar.gz`` file path.
    :param file_list: Sequence of file paths to include in the archive.
    :param base_path: Optional common base directory stripped from archive
                      member paths.

    .. code:: python

        import rsgispy.tools.archives
        rsgispy.tools.archives.create_targz_arch(
            "out.tar.gz", ["data/a.tif", "data/b.tif"], base_path="data"
        )
    """
    out_arch_abs = os.path.abspath(out_arch_file)
    with tarfile.open(out_arch_abs, "w:gz") as tar_out_file:
        for c_file in file_list:
            if base_path is not None:
                arcname = os.path.relpath(c_file, base_path)
                tar_out_file.add(c_file, arcname=arcname)
            else:
                tar_out_file.add(c_file)
