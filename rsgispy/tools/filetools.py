"""
The tools.filetools module contains functions for manipulating, querying,
hashing, locking, and archiving files and directories.
"""

# pylint: disable=too-many-lines
from contextlib import contextmanager
import datetime
import errno
import glob
import hashlib
import os
import pathlib
import re
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pyproj
import rasterio

import rsgispy

__all__ = [
    "get_file_basename",
    "is_path_valid",
    "is_path_sibling_creatable",
    "does_path_exists_or_creatable",
    "get_dir_name",
    "split_path_all",
    "delete_file_silent",
    "delete_file_with_basename",
    "rm_files_size_gt",
    "find_file",
    "find_file_none",
    "find_files_ext",
    "find_files_mpaths_ext",
    "find_first_file",
    "get_files_mod_time",
    "find_files_size_limits",
    "is_cmd_tool_avail",
    "file_is_hidden",
    "get_dir_list",
    "get_dir_list_srch",
    "convert_file_size_units",
    "get_file_size",
    "get_file_mod_time",
    "get_file_lock",
    "release_file_lock",
    "clean_file_locks",
    "sort_imgs_to_dirs_utm",
    "natural_sort_file_names",
    "sort_file_by_datetime",
    "create_directory_archive",
    "create_sha1_hash",
    "create_sha224_hash",
    "create_sha256_hash",
    "create_sha384_hash",
    "create_sha512_hash",
    "create_md5_hash",
    "create_blake2b_hash",
    "create_blake2s_hash",
    "create_sha3_224_hash",
    "create_sha3_256_hash",
    "create_sha3_384_hash",
    "create_sha3_512_hash",
    "untar_file",
    "untar_gz_file",
    "unzip_file",
    "untar_bz_file",
    "create_targz_arch",
    "work_dir_context",
]

_WIN_ERROR_INVALID_NAME = 123
_SIZE_UNITS = {
    "bytes": 1.0,
    "kb": 1024.0,
    "mb": 1024.0**2,
    "gb": 1024.0**3,
    "tb": 1024.0**4,
}


def get_file_basename(
    input_file: str,
    check_valid: bool = False,
    n_comps: int = 0,
    rm_n_exts: int = 0,
) -> str:
    """
    Return the basename of ``input_file`` (directory path and extension removed).

    :param input_file: Input file path.
    :param check_valid: If True, replace spaces with underscores and strip
                        punctuation other than ``_`` and ``-`` (Default = False).
    :param n_comps: If > 0, split the basename on ``_`` and keep the first
                    ``n_comps`` components (Default = 0).
    :param rm_n_exts: Number of extensions to strip. ``0`` (default) strips all
                      dot-separated extensions.
    :return: Extracted basename string.
    """
    file_base_name = os.path.basename(input_file)
    n_exts = file_base_name.count(".")
    if (rm_n_exts == 0) or (rm_n_exts > n_exts):
        rm_n_exts = n_exts

    basename = file_base_name
    if n_exts > 0:
        for _ in range(rm_n_exts):
            basename = os.path.splitext(basename)[0]

    if check_valid:
        basename = basename.replace(" ", "_")
        for punct in string.punctuation:
            if punct not in ("_", "-"):
                basename = basename.replace(punct, "")

    if n_comps > 0:
        basename_split = basename.split("_")
        if len(basename_split) < n_comps:
            raise rsgispy.RSGISPyException(
                "The number of components specified is more than the number "
                "of components in the basename."
            )
        basename = "_".join(basename_split[:n_comps])

    return basename


def is_path_valid(file_path_name: str) -> bool:
    """
    Test whether a file path is syntactically valid for the current OS.

    :param file_path_name: File path to test.
    :return: True if valid, False otherwise.
    """
    valid = isinstance(file_path_name, str) and bool(file_path_name)
    if valid:
        try:
            _, stripped_path = os.path.splitdrive(file_path_name)
            root_dirname = (
                os.environ.get("HOMEDRIVE", "C:")
                if sys.platform == "win32"
                else os.path.sep
            )
            if not os.path.isdir(root_dirname):
                valid = False
            else:
                root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep
                for pathname_part in stripped_path.split(os.path.sep):
                    try:
                        os.lstat(root_dirname + pathname_part)
                    except ValueError:
                        valid = False
                        break
                    except OSError as exc:
                        if (
                            getattr(exc, "winerror", None)
                            == _WIN_ERROR_INVALID_NAME
                            or exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}
                        ):
                            valid = False
                            break
        except TypeError:
            valid = False
    return valid


def is_path_sibling_creatable(file_path_name: str) -> bool:
    """
    Check whether a sibling file can be created in the parent directory of
    ``file_path_name``.

    :param file_path_name: File path to check.
    :return: True if a file can be created in the parent directory, False otherwise.
    """
    dirname = os.path.dirname(file_path_name) or os.getcwd()
    try:
        with tempfile.TemporaryFile(dir=dirname):
            pass
        return True
    except OSError:
        return False


def does_path_exists_or_creatable(file_path_name: str) -> bool:
    """
    Check whether ``file_path_name`` is valid and either already exists or
    can be created on the current filesystem.

    :param file_path_name: File path to check.
    :return: True if path exists or is creatable, False otherwise.
    """
    try:
        return is_path_valid(file_path_name) and (
            os.path.exists(file_path_name)
            or is_path_sibling_creatable(file_path_name)
        )
    except OSError:
        return False


def get_dir_name(input_file: str) -> str:
    """
    Return the immediate directory name containing ``input_file`` (if a file)
    or the directory name itself (if a directory).

    :param input_file: Path to an existing file or directory.
    :return: Directory name (without parent path).
    """
    abs_path = os.path.abspath(input_file)
    if os.path.isfile(abs_path):
        dir_path = os.path.dirname(abs_path)
    elif os.path.isdir(abs_path):
        dir_path = abs_path
    else:
        raise rsgispy.RSGISPyException(
            "Input path must be either a file or directory"
        )
    return os.path.basename(dir_path)


def split_path_all(input_path: str) -> List[str]:
    """
    Split all components of ``input_path`` into a list of path parts.

    :param input_path: Input file or directory path.
    :return: List of path components.
    """
    all_parts: List[str] = []
    cur_path = input_path
    while True:
        parts = os.path.split(cur_path)
        if parts[0] == cur_path:
            all_parts.insert(0, parts[0])
            break
        if parts[1] == cur_path:
            all_parts.insert(0, parts[1])
            break
        cur_path = parts[0]
        all_parts.insert(0, parts[1])
    return all_parts


def delete_file_silent(input_file: str) -> bool:
    """
    Remove ``input_file`` if it exists and is a regular file, catching any
    exceptions and returning success status.

    :param input_file: Path to the file to remove.
    :return: True if removed or did not exist; False if removal failed.
    """
    try:
        if os.path.exists(input_file) and os.path.isfile(input_file):
            os.remove(input_file)
        return True
    except OSError:
        return False


def delete_file_with_basename(input_file: str, print_rms: bool = True) -> None:
    """
    Delete all files sharing the same path and base name as ``input_file``
    (e.g., sidecar files with different extensions).

    :param input_file: Input file path whose basename defines files to delete.
    :param print_rms: If True, print each deleted file path (Default = True).
    """
    base_name = os.path.splitext(input_file)[0]
    for file_path in glob.glob(f"{base_name}.*"):
        if print_rms:
            print(f"Deleting file: {file_path}")
        delete_file_silent(file_path)


def rm_files_size_gt(
    file_path: str,
    file_srch: str,
    min_size: int,
    rm_file: bool = False,
    rm_use_basename: bool = False,
) -> None:
    """
    Find files matching ``os.path.join(file_path, file_srch)`` larger than
    ``min_size`` bytes and either print ``rm`` commands or delete them.

    :param file_path: Directory to search.
    :param file_srch: Glob pattern (e.g., ``'*.tif'``).
    :param min_size: Minimum file size in bytes (exclusive).
    :param rm_file: If True, delete matching files; otherwise print commands.
    :param rm_use_basename: If True, delete all files sharing each basename.
    """
    file_lst = find_files_size_limits(file_path, file_srch, min_size=min_size)
    for in_file in file_lst:
        if rm_file:
            if rm_use_basename:
                delete_file_with_basename(in_file)
            else:
                delete_file_silent(in_file)
        else:
            print(f"rm {in_file}")


def find_file(dir_path: str, file_search: str) -> str:
    """
    Search for a single file matching ``file_search`` in ``dir_path``.

    :param dir_path: Directory path to search.
    :param file_search: Glob pattern containing wildcard(s).
    :return: Matching file path.
    """
    files = glob.glob(os.path.join(dir_path, file_search))
    if len(files) != 1:
        raise rsgispy.RSGISPyException(
            f"Could not find a single file ({file_search}); found {len(files)} files."
        )
    return files[0]


def find_file_none(dir_path: str, file_search: str) -> Optional[str]:
    """
    Search for a single file matching ``file_search`` in ``dir_path``,
    returning ``None`` unless exactly one match is found.

    :param dir_path: Directory path to search.
    :param file_search: Glob pattern containing wildcard(s).
    :return: Matching file path or ``None``.
    """
    files = glob.glob(os.path.join(dir_path, file_search))
    if len(files) != 1:
        return None
    return files[0]


def find_files_ext(dir_path: str, ending: str) -> Dict[str, str]:
    """
    Recursively find all files under ``dir_path`` ending with ``ending``.

    :param dir_path: Base directory path to search.
    :param ending: Filename suffix/extension (e.g., ``'.tif'``).
    :return: Dictionary mapping filename to full file path.
    """
    out_file_dict: Dict[str, str] = {}
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith(ending):
                file_found = os.path.join(root, file_name)
                if os.path.isfile(file_found):
                    out_file_dict[file_name] = file_found
    return out_file_dict


def find_files_mpaths_ext(
    dir_paths: Sequence[str], ending: str
) -> Dict[str, str]:
    """
    Recursively find all files ending with ``ending`` across multiple base
    directories.

    :param dir_paths: Sequence of base directory paths to search.
    :param ending: Filename suffix/extension (e.g., ``'.tif'``).
    :return: Dictionary mapping filename to full file path.
    """
    out_file_dict: Dict[str, str] = {}
    for dir_path in dir_paths:
        out_file_dict.update(find_files_ext(dir_path, ending))
    return out_file_dict


def find_first_file(
    dir_path: str, file_search: str, rtn_except: bool = True
) -> Optional[str]:
    """
    Walk ``dir_path`` recursively and return the first single glob match for
    ``file_search``.

    :param dir_path: Base directory path to search.
    :param file_search: Filename glob pattern (e.g., ``'*N15W093*.tif'``).
    :param rtn_except: If True, raise :class:`rsgispy.RSGISPyException` when
                       zero or multiple files match; otherwise return ``None``.
    :return: Matched file path, or ``None`` if ``rtn_except=False``.
    """
    files: List[str] = []
    for root, _, _ in os.walk(dir_path):
        files = glob.glob(os.path.join(root, file_search))
        if files:
            break

    if len(files) == 1:
        return files[0]
    if rtn_except:
        raise rsgispy.RSGISPyException(
            f"Could not find a single file ({file_search}) in {dir_path}; "
            f"found {len(files)} files."
        )
    return None


def get_files_mod_time(
    file_lst: Sequence[str],
    dt_before: Optional[datetime.datetime] = None,
    dt_after: Optional[datetime.datetime] = None,
) -> List[str]:
    """
    Subset a list of file paths by last modification timestamp. Non-existent
    files are ignored.

    :param file_lst: Sequence of file paths.
    :param dt_before: Keep files modified strictly before this datetime.
    :param dt_after: Keep files modified strictly after this datetime.
    :return: Filtered list of file paths.
    """
    if (dt_before is None) and (dt_after is None):
        raise rsgispy.RSGISPyException(
            "You must define at least one of dt_before or dt_after"
        )
    out_file_lst: List[str] = []
    for cfile in file_lst:
        if os.path.exists(cfile):
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cfile))
            before_ok = dt_before is None or mod_time < dt_before
            after_ok = dt_after is None or mod_time > dt_after
            if before_ok and after_ok:
                out_file_lst.append(cfile)
    return out_file_lst


def find_files_size_limits(
    dir_path: str,
    file_search: str,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> List[str]:
    """
    Search for files matching ``file_search`` in ``dir_path`` whose size in
    bytes satisfies ``min_size < size`` (and ``size < max_size`` if given).

    :param dir_path: Directory path to search.
    :param file_search: Glob pattern.
    :param min_size: Exclusive lower file size threshold in bytes (Default = 0).
    :param max_size: Optional exclusive upper file size threshold in bytes.
    :return: List of matching file paths.
    """
    files = glob.glob(os.path.join(dir_path, file_search))
    out_files: List[str] = []
    for c_file in files:
        if not os.path.isfile(c_file):
            continue
        file_size = get_file_size(c_file)
        if file_size > min_size and (max_size is None or file_size < max_size):
            out_files.append(c_file)
    return out_files


def is_cmd_tool_avail(
    cmd_name: str, test_call_cmd: Optional[List[str]] = None
) -> bool:
    """
    Check whether an executable command is available on ``PATH``, optionally
    executing a test command to verify it runs without error.

    :param cmd_name: Name of the executable (e.g., ``'tar'``, ``'gzip'``).
    :param test_call_cmd: Optional command argument list to execute as a check.
    :return: True if available (and test command succeeds if provided).
    """
    command_exists = shutil.which(cmd_name) is not None
    if (test_call_cmd is not None) and command_exists:
        if not isinstance(test_call_cmd, list):
            raise rsgispy.RSGISPyException(
                "Input test command was not provided as a list."
            )
        try:
            rtn_info = subprocess.run(
                test_call_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return rtn_info.returncode == 0
        except (subprocess.CalledProcessError, OSError):
            return False
    return command_exists


def file_is_hidden(dir_path: str) -> bool:
    """
    Test whether a file or directory is hidden on the current OS.

    :param dir_path: Input file or directory path.
    :return: True if hidden, False otherwise.
    """
    abs_path = os.path.abspath(dir_path)
    if os.name == "nt":
        # pylint: disable=import-outside-toplevel,import-error
        import win32api
        import win32con

        attribute = win32api.GetFileAttributes(abs_path)
        return bool(
            attribute
            & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
        )
    return os.path.basename(abs_path).startswith(".")


def get_dir_list(dir_path: str, inc_hidden: bool = False) -> List[str]:
    """
    List immediate subdirectories within ``dir_path``.

    :param dir_path: Directory path to inspect.
    :param inc_hidden: Whether to include hidden directories (Default = False).
    :return: List of subdirectory paths.
    """
    out_dir_lst: List[str] = []
    for item in os.listdir(dir_path):
        c_path = os.path.join(dir_path, item)
        if os.path.isdir(c_path) and (inc_hidden or not file_is_hidden(c_path)):
            out_dir_lst.append(c_path)
    return out_dir_lst


def get_dir_list_srch(
    dir_path: str, name_inc: str, inc_hidden: bool = False
) -> List[str]:
    """
    List immediate subdirectories within ``dir_path`` whose name contains
    ``name_inc``.

    :param dir_path: Directory path to inspect.
    :param name_inc: Substring required in directory name.
    :param inc_hidden: Whether to include hidden directories (Default = False).
    :return: List of matching subdirectory paths.
    """
    out_dir_lst: List[str] = []
    for item in os.listdir(dir_path):
        c_path = os.path.join(dir_path, item)
        if (
            os.path.isdir(c_path)
            and (name_inc in item)
            and (inc_hidden or not file_is_hidden(c_path))
        ):
            out_dir_lst.append(c_path)
    return out_dir_lst


def convert_file_size_units(
    in_size: Union[int, float], in_unit: str, out_unit: str
) -> float:
    """
    Convert a file size between ``bytes``, ``kb``, ``mb``, ``gb``, and ``tb``.

    :param in_size: Input file size value.
    :param in_unit: Input unit (``'bytes'``, ``'kb'``, ``'mb'``, ``'gb'``, ``'tb'``).
    :param out_unit: Output unit (``'bytes'``, ``'kb'``, ``'mb'``, ``'gb'``, ``'tb'``).
    :return: Converted file size as float.
    """
    in_unit_l = in_unit.lower()
    if in_unit_l not in _SIZE_UNITS:
        raise rsgispy.RSGISPyException(
            "Input unit must be one of: bytes, kb, mb, gb, tb"
        )
    out_unit_l = out_unit.lower()
    if out_unit_l not in _SIZE_UNITS:
        raise rsgispy.RSGISPyException(
            "Output unit must be one of: bytes, kb, mb, gb, tb"
        )

    file_size_bytes = float(in_size) * _SIZE_UNITS[in_unit_l]
    return file_size_bytes / _SIZE_UNITS[out_unit_l]


def get_file_size(file_path: str, unit: str = "bytes") -> float:
    """
    Return the size of ``file_path`` in the specified unit.

    :param file_path: Path to an existing regular file.
    :param unit: Size unit (``'bytes'``, ``'kb'``, ``'mb'``, ``'gb'``, ``'tb'``).
    :return: File size in requested unit.
    """
    unit_l = unit.lower()
    if unit_l not in _SIZE_UNITS:
        raise rsgispy.RSGISPyException(
            "Unit must be one of: bytes, kb, mb, gb, tb"
        )

    path_obj = pathlib.Path(file_path)
    if path_obj.exists() and path_obj.is_file():
        return convert_file_size_units(
            path_obj.stat().st_size, "bytes", unit_l
        )
    raise rsgispy.RSGISPyException("Input file path does not exist")


def get_file_mod_time(file_path: str) -> datetime.datetime:
    """
    Return the last modification timestamp of ``file_path`` as a
    :class:`datetime.datetime` object.

    :param file_path: Path to an existing file or directory.
    :return: Modification timestamp as ``datetime.datetime``.
    """
    if not os.path.exists(file_path):
        raise rsgispy.RSGISPyException(f"File path does not exist: {file_path}")
    return datetime.datetime.fromtimestamp(os.path.getmtime(file_path))


def get_file_lock(
    input_file: str,
    sleep_period: float = 1,
    wait_iters: int = 120,
    use_except: bool = False,
) -> bool:
    """
    Acquire a hidden ``.<filename>.lok`` lock file alongside ``input_file``.

    :param input_file: Target file path for which the lock is created.
    :param sleep_period: Seconds to wait between retries (Default = 1).
    :param wait_iters: Maximum retry iterations (Default = 120).
    :param use_except: If True, raise :class:`rsgispy.RSGISPyException` when
                       lock acquisition fails; otherwise return False.
    :return: True if lock was acquired, False otherwise.
    """
    file_path, file_name = os.path.split(input_file)
    lock_file_path = os.path.join(file_path, f".{file_name}.lok")

    got_lock = False
    for _ in range(wait_iters + 1):
        if not os.path.exists(lock_file_path):
            got_lock = True
            break
        time.sleep(sleep_period)

    if got_lock:
        c_datetime = datetime.datetime.now()
        with open(lock_file_path, "w", encoding="utf-8") as fobj:
            fobj.write(f"{c_datetime.isoformat()}\n")
            fobj.flush()
    elif use_except:
        raise rsgispy.RSGISPyException(
            f"Lock could not be gained for file: {input_file}"
        )

    return got_lock


def release_file_lock(input_file: str) -> None:
    """
    Release the ``.<filename>.lok`` lock file for ``input_file``.

    :param input_file: Target file path whose lock file should be removed.
    """
    file_path, file_name = os.path.split(input_file)
    lock_file_path = os.path.join(file_path, f".{file_name}.lok")
    if os.path.exists(lock_file_path):
        if not delete_file_silent(lock_file_path):
            raise rsgispy.RSGISPyException(
                "Could not delete the lock file... something has gone wrong!"
            )


def clean_file_locks(dir_path: str, timeout: int = 3600) -> None:
    """
    Remove stale ``.*.lok`` lock files in ``dir_path`` older than ``timeout``
    seconds.

    :param dir_path: Directory containing lock files.
    :param timeout: Lock expiry threshold in seconds (Default = 3600).
    """
    c_datetime = datetime.datetime.now()
    for lock_file_path in glob.glob(os.path.join(dir_path, ".*.lok")):
        with open(lock_file_path, "r", encoding="utf-8") as fobj:
            create_date_str = fobj.read().replace("\r", "").replace("\n", "").strip()
        create_date = datetime.datetime.fromisoformat(create_date_str)
        time_since_create = (c_datetime - create_date).total_seconds()
        if time_since_create > timeout:
            if not delete_file_silent(lock_file_path):
                raise rsgispy.RSGISPyException(
                    "Could not delete the lock file... something has gone wrong!"
                )


def _get_img_utm_zone_and_files(img_file: str) -> Tuple[Optional[str], List[str]]:
    """Return ``(utm_zone_str, companion_file_list)`` for a raster via rasterio."""
    with rasterio.open(img_file, "r") as src:
        img_crs = src.crs
        img_files = list(dict.fromkeys(src.files))

    if img_crs is None:
        return None, img_files

    pyproj_crs = pyproj.CRS.from_user_input(img_crs)
    utm_zone = pyproj_crs.utm_zone
    if utm_zone is not None and len(utm_zone) == 2:
        utm_zone = f"0{utm_zone}"
    return utm_zone, img_files


def sort_imgs_to_dirs_utm(
    input_imgs_dir: str, file_search_str: str, out_base_dir: str
) -> None:
    """
    Sort UTM-projected raster images (and their companion files) matching
    ``file_search_str`` inside ``input_imgs_dir`` into ``utm<zone>``
    subdirectories under ``out_base_dir``.

    Note: input files are moved on disk.

    :param input_imgs_dir: Input directory containing images.
    :param file_search_str: Glob search pattern relative to ``input_imgs_dir``.
    :param out_base_dir: Destination base directory for ``utm<zone>`` folders.
    """
    in_files = glob.glob(os.path.join(input_imgs_dir, file_search_str))
    for img_file in in_files:
        utm_zone, img_file_list = _get_img_utm_zone_and_files(img_file)
        if utm_zone is not None:
            out_dir = os.path.join(out_base_dir, f"utm{utm_zone}")
            os.makedirs(out_dir, exist_ok=True)
            for tmp_file in img_file_list:
                print(f"Moving: {tmp_file}")
                out_file = os.path.join(out_dir, os.path.basename(tmp_file))
                shutil.move(tmp_file, out_file)


def natural_sort_file_names(in_file_lst: List[str]) -> List[str]:
    """
    Perform an in-place natural sort on a list of file paths based on their
    basenames and return the sorted list.

    :param in_file_lst: List of file paths to sort.
    :return: Naturally sorted list of file paths.
    """
    nsre = re.compile("([0-9]+)")

    def _natural_sort_key(file_path: str) -> List[Union[int, str]]:
        bname = get_file_basename(file_path)
        return [
            int(text) if text.isdigit() else text.lower()
            for text in nsre.split(bname)
        ]

    in_file_lst.sort(key=_natural_sort_key)
    return in_file_lst


def sort_file_by_datetime(in_file_lst: List[str]) -> List[str]:
    """
    Sort a list of file paths in-place by modification time (oldest first).

    :param in_file_lst: List of existing file paths.
    :return: Sorted list of file paths.
    """
    in_file_lst.sort(key=os.path.getmtime)
    return in_file_lst


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
    if not is_path_valid(in_dir):
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
    if not does_path_exists_or_creatable(out_arch_file):
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


def _calc_file_hash(
    hash_name: str, input_file: str, block_size: int = 4096
) -> str:
    """Compute a hexadecimal digest of ``input_file`` using ``hashlib``."""
    hash_obj = hashlib.new(hash_name)
    with open(input_file, "rb") as fobj:
        for byte_block in iter(lambda: fobj.read(block_size), b""):
            hash_obj.update(byte_block)
    return hash_obj.hexdigest()


def create_sha1_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-1 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-1 hex digest string.
    """
    return _calc_file_hash("sha1", input_file, block_size)


def create_sha224_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-224 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-224 hex digest string.
    """
    return _calc_file_hash("sha224", input_file, block_size)


def create_sha256_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-256 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-256 hex digest string.
    """
    return _calc_file_hash("sha256", input_file, block_size)


def create_sha384_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-384 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-384 hex digest string.
    """
    return _calc_file_hash("sha384", input_file, block_size)


def create_sha512_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA-512 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA-512 hex digest string.
    """
    return _calc_file_hash("sha512", input_file, block_size)


def create_md5_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the MD5 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: MD5 hex digest string.
    """
    return _calc_file_hash("md5", input_file, block_size)


def create_blake2b_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the BLAKE2b hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: BLAKE2b hex digest string.
    """
    return _calc_file_hash("blake2b", input_file, block_size)


def create_blake2s_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the BLAKE2s hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: BLAKE2s hex digest string.
    """
    return _calc_file_hash("blake2s", input_file, block_size)


def create_sha3_224_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-224 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-224 hex digest string.
    """
    return _calc_file_hash("sha3_224", input_file, block_size)


def create_sha3_256_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-256 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-256 hex digest string.
    """
    return _calc_file_hash("sha3_256", input_file, block_size)


def create_sha3_384_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-384 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-384 hex digest string.
    """
    return _calc_file_hash("sha3_384", input_file, block_size)


def create_sha3_512_hash(input_file: str, block_size: int = 4096) -> str:
    """
    Calculate the SHA3-512 hex digest of ``input_file``.

    :param input_file: Path to the input file.
    :param block_size: Read buffer size in bytes (Default = 4096).
    :return: SHA3-512 hex digest string.
    """
    return _calc_file_hash("sha3_512", input_file, block_size)


def _extract_archive_cmd(
    in_file: str,
    out_dir: str,
    gen_arch_dir: bool,
    verbose: bool,
    cmd_builder,
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
        basename = get_file_basename(in_file_abs)
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


@contextmanager
def work_dir_context(tmp_dir: Optional[str] = None) -> Iterator[str]:
    """
    Yield a writable working directory path.

    If ``tmp_dir`` is ``None``, a temporary directory is created and removed
    when the context exits. Otherwise ``tmp_dir`` is created if needed and
    kept after exit.

    :param tmp_dir: Optional user-provided directory path (Default = ``None``).
    :return: Path to the working directory.
    """
    if tmp_dir is None:
        with tempfile.TemporaryDirectory() as tmp_path:
            yield tmp_path
    else:
        os.makedirs(tmp_dir, exist_ok=True)
        yield tmp_dir
