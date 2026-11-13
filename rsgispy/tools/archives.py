"""
The tools.archives module provides utilities for creating and extracting
zip and tar archives.
"""

import os
import subprocess
import tarfile
from typing import Callable, List, Optional, Sequence

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


def create_directory_archive(
    in_dir: str, out_arch: str, arch_format: str
) -> str:
    """
    Create an archive file from an input directory using external archive tools.

    :param in_dir: Input directory path to archive.
    :param out_arch: Output archive path without file extension.
    :param arch_format: One of ``'zip'``, ``'tar'``, ``'gztar'``, ``'bztar'``,
                        or ``'xztar'``.
    :return: Full output archive file path including extension.
    """
    if not rsgispy.tools.files.is_path_valid(in_dir):
        raise rsgispy.RSGISPyException(
            f"The input directory is not valid: {in_dir}"
        )

    out_arch_abs = os.path.abspath(out_arch.strip())
    in_dir_abs = os.path.abspath(in_dir)
    base_dir = os.path.dirname(in_dir_abs)
    dir_name = os.path.split(in_dir_abs)[1]

    format_specs = {
        "zip": (".zip", ["zip", "-r"]),
        "tar": (".tar", ["tar", "-cvf"]),
        "gztar": (".tar.gz", ["tar", "-cvzf"]),
        "bztar": (".tar.bz2", ["tar", "-cvjSf"]),
        "xztar": (".tar.xz", ["tar", "-cvJf"]),
    }
    if arch_format not in format_specs:
        raise rsgispy.RSGISPyException(
            "Do not recognise the archive format specifed."
        )

    ext, cmd_prefix = format_specs[arch_format]
    out_arch_file = f"{out_arch_abs}{ext}"
    if not rsgispy.tools.files.does_path_exists_or_creatable(out_arch_file):
        raise rsgispy.RSGISPyException(
            f"Output file path is not creatable: {out_arch_file}"
        )

    cmd = [*cmd_prefix, out_arch_file, dir_name]
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        raise rsgispy.RSGISPyException(
            f"Could not create archive: {cmd}"
        ) from err

    return out_arch_file


def _extract_archive_cmd(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool,
    verbose: bool,
    cmd_builder: Callable[[str, bool], List[str]],
) -> str:
    """Common implementation for extracting tar/tar.gz/tar.bz2/zip archives."""
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

    cmd = cmd_builder(in_file_abs, verbose)
    if verbose:
        print(f"Extracting: {in_file_abs}")
        print(f"Output to: {process_dir}")
        print(cmd)

    try:
        subprocess.run(cmd, cwd=process_dir, check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        raise rsgispy.RSGISPyException(f"Could not extract data: {cmd}") from err

    return process_dir


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
    :param verbose: Print progress messages and command details (Default = False).
    :return: Output directory where files were extracted.
    """
    return _extract_archive_cmd(
        in_file,
        out_dir,
        gen_arch_dir,
        verbose,
        lambda fpath, verb: ["tar", "-xvf" if verb else "-xf", fpath],
    )


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
    :param verbose: Print progress messages and command details (Default = False).
    :return: Output directory where files were extracted.
    """
    return _extract_archive_cmd(
        in_file,
        out_dir,
        gen_arch_dir,
        verbose,
        lambda fpath, verb: ["tar", "-xvzf" if verb else "-xzf", fpath],
    )


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
    :param verbose: Print progress messages and command details (Default = False).
    :return: Output directory where files were extracted.
    """
    return _extract_archive_cmd(
        in_file,
        out_dir,
        gen_arch_dir,
        verbose,
        lambda fpath, _verb: ["unzip", fpath],
    )


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
    :param verbose: Print progress messages and command details (Default = False).
    :return: Output directory where files were extracted.
    """
    return _extract_archive_cmd(
        in_file,
        out_dir,
        gen_arch_dir,
        verbose,
        lambda fpath, verb: ["tar", "-xvjf" if verb else "-xjf", fpath],
    )


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
    """
    out_arch_abs = os.path.abspath(out_arch_file)
    with tarfile.open(out_arch_abs, "w:gz") as tar_out_file:
        for c_file in file_list:
            if base_path is not None:
                arcname = os.path.relpath(c_file, base_path)
                tar_out_file.add(c_file, arcname=arcname)
            else:
                tar_out_file.add(c_file)
