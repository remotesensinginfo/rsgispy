"""
Internal shared HTTP streaming download and ``pysondb`` download-queue helpers
for :mod:`rsgispy.tools` and :mod:`rsgispy.dataaccess`.
"""

import contextlib
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import rsgispy
import rsgispy.tools.hashing

__all__ = [
    "stream_http_download",
    "write_download_queue_db",
    "process_download_queue_db",
]


def _enter_resource(stack: contextlib.ExitStack, obj: Any) -> Any:
    """Enter ``obj`` as a context manager if supported, else close on exit."""
    if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
        return stack.enter_context(obj)
    if hasattr(obj, "close"):
        stack.callback(obj.close)
    return obj


# pylint: disable=too-many-arguments,too-many-locals
def stream_http_download(
    session: Any,
    url: str,
    out_file_path: str,
    *,
    tmp_file_path: Optional[str] = None,
    chunk_size: int = 2**20,
    total_bytes: Optional[int] = None,
    timeout: float = 60.0,
    get_kwargs: Optional[Dict[str, Any]] = None,
    expected_md5: Optional[str] = None,
    use_pbar: bool = True,
    complete_label: Optional[str] = None,
    check_response_fn: Optional[Callable[[Any, str], bool]] = None,
) -> None:
    """
    Stream a remote HTTP resource to a ``.incomplete`` staging file, optionally
    verify its MD5 checksum, and atomically move it to ``out_file_path`` via
    :func:`os.replace`.

    Any partial staging file is removed if an error occurs during download or
    checksum verification.

    :param session: ``requests.Session``-compatible object providing ``.get()``.
    :param url: Remote URL to download.
    :param out_file_path: Final destination file path.
    :param tmp_file_path: Optional custom staging file path (defaults to
                          ``f"{out_file_path}.incomplete"``).
    :param chunk_size: Streaming chunk size in bytes (Default = ``1 MiB``).
    :param total_bytes: Optional expected size in bytes for progress estimation;
                        falls back to the response ``content-length`` header.
    :param timeout: Request timeout in seconds (Default = ``60.0``).
    :param get_kwargs: Optional extra keyword arguments passed to
                       ``session.get()``.
    :param expected_md5: Optional expected MD5 hex digest to verify before
                         finalising the download.
    :param use_pbar: Display a tqdm progress bar when ``True`` (Default = ``True``).
    :param complete_label: Optional label printed on completion (defaults to
                           ``out_file_path``).
    :param check_response_fn: Response validator callable ``(response, url) -> bool``.
                              Defaults to
                              :func:`rsgispy.tools.http.check_http_response`.
    """
    if check_response_fn is None:
        # pylint: disable=import-outside-toplevel
        from rsgispy.tools.http import check_http_response

        check_response_fn = check_http_response

    staging_path = (
        tmp_file_path
        if tmp_file_path is not None
        else f"{out_file_path}.incomplete"
    )
    tqdm_cls = rsgispy.get_tqdm()
    req_kwargs = dict(get_kwargs) if get_kwargs is not None else {}

    try:
        with contextlib.ExitStack() as stack:
            ses = _enter_resource(stack, session)
            resp = _enter_resource(
                stack,
                ses.get(url, stream=True, timeout=timeout, **req_kwargs),
            )
            if check_response_fn(resp, url):
                total = (
                    int(total_bytes)
                    if total_bytes is not None
                    else int(resp.headers.get("content-length", 0))
                )
                n_chunks = int(total / chunk_size) + 1
                with open(staging_path, "wb") as out_fp:
                    for chunk in tqdm_cls(
                        resp.iter_content(chunk_size=chunk_size),
                        total=n_chunks,
                        disable=not use_pbar,
                    ):
                        if chunk:
                            out_fp.write(chunk)

        if not os.path.exists(staging_path):
            raise rsgispy.RSGISPyException(
                "Failed to download file - unknown error"
            )

        if expected_md5 is not None:
            md5_checksum = rsgispy.tools.hashing.create_md5_hash(staging_path)
            if md5_checksum.lower() != str(expected_md5).strip().lower():
                raise rsgispy.RSGISPyException("Error - checksum mismatch.")

        os.replace(staging_path, out_file_path)
        label = complete_label if complete_label is not None else out_file_path
        print(f"Download Complete: {label}")
    except Exception:
        if os.path.exists(staging_path):
            with contextlib.suppress(OSError):
                os.remove(staging_path)
        raise


def write_download_queue_db(
    pysondb_mod: Any,
    db_json: str,
    records: Sequence[Dict[str, Any]],
) -> None:
    """
    Open or initialise a ``pysondb`` JSON queue database and insert ``records``
    when non-empty.

    :param pysondb_mod: Imported ``pysondb`` module object.
    :param db_json: Path to the JSON database file.
    :param records: Sequence of queue record dictionaries to append.
    """
    lst_db = pysondb_mod.getDb(db_json)
    if records:
        lst_db.addMany(list(records))


def process_download_queue_db(
    pysondb_mod: Any,
    db_json: str,
    download_record_fn: Callable[[Dict[str, Any]], Tuple[bool, str]],
    *,
    use_pbar: bool = True,
) -> None:
    """
    Process pending (``downloaded=False``) records in a ``pysondb`` download
    queue database, updating each completed entry with ``lcl_path`` and
    ``downloaded=True``.

    :param pysondb_mod: Imported ``pysondb`` module object.
    :param db_json: Path to the JSON database file.
    :param download_record_fn: Callback accepting a queue record dict and
                               returning ``(success, local_path)``.
    :param use_pbar: Display a tqdm progress bar when ``True`` (Default = ``True``).
    """
    tqdm_cls = rsgispy.get_tqdm()
    lst_db = pysondb_mod.getDb(db_json)
    dwld_files = lst_db.getByQuery({"downloaded": False})

    for record in tqdm_cls(dwld_files, disable=not use_pbar):
        success, lcl_path = download_record_fn(record)
        if success:
            lst_db.updateById(
                record["id"], {"lcl_path": lcl_path, "downloaded": True}
            )
