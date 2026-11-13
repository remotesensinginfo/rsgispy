"""
The tools.ftp module provides FTP directory traversal, download/upload,
and tracking database utilities.

Functions requiring ``pysondb`` are only defined and exported when
``pysondb`` is installed.
"""

import ftplib
import os
import posixpath
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import rsgispy
import rsgispy.tools.files

try:
    import pysondb  # pylint: disable=import-error

    HAVE_PYSONDB = True
except ImportError:
    HAVE_PYSONDB = False

__all__ = [
    "get_ftp_file_listings",
    "download_ftp_file",
    "download_curl_ftp_file",
    "upload_ftp_file",
]

if HAVE_PYSONDB:
    __all__.extend(
        [
            "create_file_listings_db",
            "filter_file_listings_db",
            "download_files_use_lst_db",
        ]
    )


def _ftp_list_with_retries(
    list_fn: Callable[[], Any],
    try_n_times: int,
) -> Any:
    """Execute an FTP directory listing callback with retries on failure."""
    for _ in range(try_n_times):
        try:
            return list_fn()
        except Exception as err:  # pylint: disable=broad-exception-caught
            print(f"FTP connection failed but trying again: {err}")
            time.sleep(5)
    raise rsgispy.RSGISPyException(
        "Tried multiple times which failed to get directory "
        "listing on FTP server so failing."
    )


def _traverse_mlsd_ftp_dirs(
    ftp_conn: ftplib.FTP,
    ftp_path: str,
    ftp_files: Dict[str, List[str]],
    try_n_times: int = 5,
) -> List[str]:
    """Recursively list files on an FTP server using ``MLSD``."""
    dirs: List[str] = []
    non_dirs: List[str] = []
    if ftp_path not in ftp_files:
        ftp_files[ftp_path] = []

    dir_lst = list(
        _ftp_list_with_retries(
            lambda: list(ftp_conn.mlsd(ftp_path, ["type"])), try_n_times
        )
    )
    for name, facts in dir_lst:
        if facts.get("type") == "dir":
            c_dir = posixpath.join(ftp_path, name)
            dirs.append(c_dir)
            if c_dir not in ftp_files:
                ftp_files[c_dir] = []
        elif name not in (".", ".."):
            c_file = posixpath.join(ftp_path, name)
            non_dirs.append(c_file)
            ftp_files[ftp_path].append(c_file)

    for subdir in sorted(dirs):
        non_dirs.extend(
            _traverse_mlsd_ftp_dirs(ftp_conn, subdir, ftp_files, try_n_times)
        )
    return non_dirs


def _traverse_nlst_ftp_dirs(
    ftp_conn: ftplib.FTP,
    ftp_path: str,
    ftp_files: Dict[str, List[str]],
    try_n_times: int = 5,
) -> List[str]:
    """Recursively list files on an FTP server using ``LIST``."""
    dirs: List[str] = []
    non_dirs: List[str] = []
    if ftp_path not in ftp_files:
        ftp_files[ftp_path] = []

    ftp_conn.cwd(ftp_path)

    def _list_lines() -> List[str]:
        lines: List[str] = []
        ftp_conn.retrlines("LIST", lines.append)
        return lines

    dir_lst = _ftp_list_with_retries(_list_lines, try_n_times)
    for parts in map(str.split, dir_lst):
        if not parts:
            continue
        entry_name = parts[-1]
        if parts[0][0] == "d":
            c_dir = posixpath.join(ftp_path, entry_name)
            dirs.append(c_dir)
            if c_dir not in ftp_files:
                ftp_files[c_dir] = []
        elif entry_name not in (".", ".."):
            c_file = posixpath.join(ftp_path, entry_name)
            non_dirs.append(c_file)
            ftp_files[ftp_path].append(c_file)

    for subdir in sorted(dirs):
        non_dirs.extend(
            _traverse_nlst_ftp_dirs(ftp_conn, subdir, ftp_files, try_n_times)
        )
    return non_dirs


def get_ftp_file_listings(
    ftp_url: str,
    ftp_path: str,
    ftp_user: str = "",
    ftp_pass: str = "",
    ftp_timeout: int = 300,
    try_n_times: int = 5,
    mlsd_not_avail: bool = False,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Traverse an FTP server directory tree and return all file paths.

    :param ftp_url: FTP server hostname.
    :param ftp_path: Root path on the FTP server.
    :param ftp_user: FTP username.
    :param ftp_pass: FTP password.
    :param ftp_timeout: Connection timeout in seconds (Default = ``300``).
    :param try_n_times: Number of retries per directory listing (Default = ``5``).
    :param mlsd_not_avail: Fallback to ``LIST`` when ``MLSD`` is unsupported.
    :return: Tuple ``(dir_to_files_dict, flat_file_list)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    ftp_files: Dict[str, List[str]] = {}
    ftp_conn = ftplib.FTP(ftp_url, timeout=ftp_timeout)
    ftp_conn.login(user=ftp_user, passwd=ftp_pass)
    try:
        if mlsd_not_avail:
            non_dirs_lst = _traverse_nlst_ftp_dirs(
                ftp_conn, ftp_path, ftp_files, try_n_times
            )
        else:
            non_dirs_lst = _traverse_mlsd_ftp_dirs(
                ftp_conn, ftp_path, ftp_files, try_n_times
            )
    finally:
        ftp_conn.quit()
    return ftp_files, non_dirs_lst


if HAVE_PYSONDB:

    def create_file_listings_db(
        db_json: str,
        ftp_url: str,
        ftp_path: str,
        ftp_user: str = "",
        ftp_pass: str = "",
        ftp_timeout: int = 300,
        try_n_times: int = 5,
        mlsd_not_avail: bool = False,
    ) -> None:
        """
        Traverse an FTP server directory tree and save file listings to a ``pysondb``
        JSON database.

        :param db_json: Path to the output JSON database file.
        :param ftp_url: FTP server hostname.
        :param ftp_path: Root path on the FTP server.
        :param ftp_user: FTP username.
        :param ftp_pass: FTP password.
        :param ftp_timeout: Connection timeout in seconds (Default = ``300``).
        :param try_n_times: Number of retries per directory listing (Default = ``5``).
        :param mlsd_not_avail: Fallback to ``LIST`` when ``MLSD`` is unsupported.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        tqdm_cls = rsgispy.get_tqdm()
        _, non_dirs_lst = get_ftp_file_listings(
            ftp_url,
            ftp_path,
            ftp_user,
            ftp_pass,
            ftp_timeout,
            try_n_times,
            mlsd_not_avail,
        )
        lst_db = pysondb.getDb(db_json)
        db_data = [
            {
                "ftp_url": ftp_url,
                "rmt_path": c_file,
                "lcl_path": "",
                "downloaded": False,
            }
            for c_file in tqdm_cls(non_dirs_lst)
        ]
        if db_data:
            lst_db.addMany(db_data)

    def filter_file_listings_db(
        db_json: str,
        filter_vals: List[str],
        out_db_file: str,
        filter_path: bool = False,
    ) -> None:
        """
        Filter a ``pysondb`` FTP listings database for paths/filenames containing any
        string in ``filter_vals`` and write matching records to ``out_db_file``.

        :param db_json: Input JSON database file path.
        :param filter_vals: Substrings to match.
        :param out_db_file: Output filtered JSON database file path.
        :param filter_path: Match against full remote path when ``True`` (Default = ``False``).
        """
        tqdm_cls = rsgispy.get_tqdm()
        lst_db = pysondb.getDb(db_json)
        db_data = lst_db.getAll()

        out_db_data = []
        for file_rcd in tqdm_cls(db_data):
            target = (
                file_rcd["rmt_path"]
                if filter_path
                else os.path.basename(file_rcd["rmt_path"])
            )
            if any(val in target for val in filter_vals):
                out_db_data.append(
                    {
                        "ftp_url": file_rcd["ftp_url"],
                        "rmt_path": file_rcd["rmt_path"],
                        "lcl_path": file_rcd["lcl_path"],
                        "downloaded": file_rcd["downloaded"],
                    }
                )

        if out_db_data:
            lst_out_db = pysondb.getDb(out_db_file)
            lst_out_db.addMany(out_db_data)


def download_ftp_file(
    ftp_url: str,
    remote_file: str,
    local_file: str,
    ftp_timeout: int = 300,
    ftp_user: Optional[str] = None,
    ftp_pass: Optional[str] = None,
    print_info: bool = True,
) -> bool:
    """
    Download ``remote_file`` from ``ftp_url`` to ``local_file`` via ``ftplib``.

    :param ftp_url: FTP server hostname.
    :param remote_file: Remote file path on the FTP server.
    :param local_file: Destination local file path.
    :param ftp_timeout: Timeout in seconds (Default = ``300``).
    :param ftp_user: Optional FTP username.
    :param ftp_pass: Optional FTP password.
    :param print_info: Print progress messages when ``True``.
    :return: ``True`` on success, ``False`` on failure.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    try:
        ftp_conn = ftplib.FTP(ftp_url, timeout=ftp_timeout)
        ftp_conn.login(user=ftp_user or "", passwd=ftp_pass or "")
        try:
            if print_info:
                print(f"Downloading: {remote_file}")
            with open(local_file, "wb") as out_fp:
                ftp_conn.retrbinary(f"RETR {remote_file}", out_fp.write)
        finally:
            ftp_conn.quit()
        return True
    except Exception as err:  # pylint: disable=broad-exception-caught
        if print_info:
            print(f"An error occurred when downloading {remote_file}: {err}")
        return False


def download_curl_ftp_file(
    ftp_url: str,
    remote_file: str,
    local_file: str,
    ftp_timeout: int = 300,
    ftp_user: Optional[str] = None,
    ftp_pass: Optional[str] = None,
    print_info: bool = True,
) -> bool:
    """
    Download ``remote_file`` from ``ftp_url`` using ``pycurl``.

    :param ftp_url: Base FTP URL (e.g., ``'ftp://host'``).
    :param remote_file: Remote path on the FTP server.
    :param local_file: Destination local file path.
    :param ftp_timeout: Timeout in seconds (Default = ``300``).
    :param ftp_user: Optional FTP username.
    :param ftp_pass: Optional FTP password.
    :param print_info: Print progress info when ``True``.
    :return: ``True`` on success, ``False`` on failure.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=import-outside-toplevel,c-extension-no-member,no-member
    import pycurl

    full_path_url = ftp_url + remote_file
    try:
        with open(local_file, "wb") as out_fp:
            curl = pycurl.Curl()
            curl.setopt(pycurl.URL, full_path_url)
            curl.setopt(pycurl.FOLLOWLOCATION, 1)
            curl.setopt(pycurl.NOPROGRESS, 0 if print_info else 1)
            curl.setopt(pycurl.MAXREDIRS, 5)
            curl.setopt(pycurl.CONNECTTIMEOUT, 50)
            curl.setopt(pycurl.TIMEOUT, ftp_timeout)
            curl.setopt(pycurl.FTP_RESPONSE_TIMEOUT, 600)
            curl.setopt(pycurl.NOSIGNAL, 1)
            if (ftp_user is not None) and (ftp_pass is not None):
                curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_ANY)
                curl.setopt(pycurl.USERPWD, f"{ftp_user}:{ftp_pass}")
            curl.setopt(pycurl.WRITEDATA, out_fp)
            if print_info:
                print(f"Starting download of {full_path_url}")
            curl.perform()
            if print_info:
                print(
                    f"Finished download in {curl.getinfo(curl.TOTAL_TIME)} of "
                    f"{curl.getinfo(curl.SIZE_DOWNLOAD)} bytes for {full_path_url}"
                )
        return True
    except pycurl.error as err:
        print(
            f"An error occurred when downloading {posixpath.join(ftp_url, remote_file)}."
        )
        print(err)
        return False


if HAVE_PYSONDB:

    def download_files_use_lst_db(
        db_json: str,
        out_dir_path: str,
        ftp_timeout: int = 300,
        ftp_user: Optional[str] = None,
        ftp_pass: Optional[str] = None,
        create_dir_struct: bool = False,
        use_curl: bool = False,
    ) -> None:
        """
        Download pending files listed in a ``pysondb`` FTP listings database.

        :param db_json: Path to the JSON database file.
        :param out_dir_path: Local output root directory.
        :param ftp_timeout: Timeout in seconds (Default = ``300``).
        :param ftp_user: Optional FTP username.
        :param ftp_pass: Optional FTP password.
        :param create_dir_struct: Preserve remote directory hierarchy when ``True``.
        :param use_curl: Use :func:`download_curl_ftp_file` instead of :func:`download_ftp_file`.
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        tqdm_cls = rsgispy.get_tqdm()
        lst_db = pysondb.getDb(db_json)
        dwld_files = lst_db.getByQuery({"downloaded": False})

        for dwn_file in tqdm_cls(dwld_files):
            basename = os.path.basename(dwn_file["rmt_path"])
            if create_dir_struct:
                dir_path = os.path.dirname(dwn_file["rmt_path"]).lstrip("/")
                local_dir_path = os.path.join(out_dir_path, dir_path)
                os.makedirs(local_dir_path, exist_ok=True)
                local_path = os.path.join(local_dir_path, basename)
            else:
                local_path = os.path.join(out_dir_path, basename)

            dl_fn = download_curl_ftp_file if use_curl else download_ftp_file
            dwnlded = dl_fn(
                ftp_url=dwn_file["ftp_url"],
                remote_file=dwn_file["rmt_path"],
                local_file=local_path,
                ftp_timeout=ftp_timeout,
                ftp_user=ftp_user,
                ftp_pass=ftp_pass,
                print_info=False,
            )
            if dwnlded:
                lst_db.updateById(
                    dwn_file["id"], {"lcl_path": local_path, "downloaded": True}
                )


def upload_ftp_file(
    local_file: str,
    ftp_url: str,
    remote_path: str,
    ftp_timeout: int = 300,
    ftp_user: Optional[str] = None,
    ftp_pass: Optional[str] = None,
    print_info: bool = True,
) -> bool:
    """
    Upload ``local_file`` to ``remote_path`` on ``ftp_url``, creating missing
    remote directories as needed.

    :param local_file: Path to the local file to upload.
    :param ftp_url: FTP server hostname.
    :param remote_path: Remote destination directory path.
    :param ftp_timeout: Timeout in seconds (Default = ``300``).
    :param ftp_user: Optional FTP username.
    :param ftp_pass: Optional FTP password.
    :param print_info: Print progress messages when ``True``.
    :return: ``True`` on success, ``False`` on failure.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    if not os.path.isfile(local_file):
        raise rsgispy.RSGISPyException("Input path must be a file.")

    lcl_file_name = os.path.basename(local_file)
    if print_info:
        print(f"Opening FTP Connection to {ftp_url}")

    ftp_conn = ftplib.FTP(ftp_url, timeout=ftp_timeout)
    ftp_conn.login(user=ftp_user or "", passwd=ftp_pass or "")
    try:
        remote_path_parts = rsgispy.tools.files.split_path_all(remote_path)
        if remote_path_parts and remote_path_parts[0] == "/":
            remote_path_parts.pop(0)

        for rmt_part in remote_path_parts:
            dir_lst: List[str] = []
            ftp_conn.retrlines("LIST", dir_lst.append)
            found_part = False
            part_dir = False
            for parts in map(str.split, dir_lst):
                if parts and parts[-1] == rmt_part:
                    found_part = True
                    part_dir = parts[0][0] == "d"
                    break

            if found_part and not part_dir:
                raise rsgispy.RSGISPyException(
                    "Part of the path was not a directory."
                )
            if not found_part:
                ftp_conn.mkd(rmt_part)
            ftp_conn.cwd(rmt_part)

        try:
            with open(local_file, "rb") as lcl_file_obj:
                ftp_conn.storbinary(f"STOR {lcl_file_name}", lcl_file_obj)
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False
    finally:
        try:
            ftp_conn.quit()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
