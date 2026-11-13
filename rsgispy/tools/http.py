"""
The tools.http module provides HTTP request, file download, and download
tracking database utilities.

Functions requiring ``requests`` or ``pysondb`` are only defined and exported
when those packages are installed.
"""

import json
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import rsgispy
import rsgispy.tools.hashing

try:
    import requests  # pylint: disable=import-error

    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

try:
    import pysondb  # pylint: disable=import-error

    HAVE_PYSONDB = True
except ImportError:
    HAVE_PYSONDB = False

__all__ = [
    "RSGISPyResponseException",
    "wget_download_file",
]

if HAVE_REQUESTS:
    __all__.extend(
        [
            "check_url_exists",
            "check_http_response",
            "send_http_json_request",
            "download_file_http",
        ]
    )

if HAVE_PYSONDB:
    __all__.extend(
        [
            "create_file_listings_db",
            "download_http_files_use_lst_db",
        ]
    )


class RSGISPyResponseException(rsgispy.RSGISPyException):
    """Exception raised when an HTTP response returns an error status."""

    def __init__(
        self,
        value: str,
        response: Optional[Any] = None,
    ):
        """Initialise with an error message and optional ``requests.Response``."""
        super().__init__(value)
        self.value = value
        self.response = response

    def __str__(self) -> str:
        """Return a formatted HTTP status and error message."""
        if self.response is None:
            return repr(self.value)
        return (
            f"HTTP status {self.response.status_code} "
            f"{self.response.reason}: {repr(self.value)}"
        )


if HAVE_REQUESTS:

    def check_url_exists(url: str, timeout: float = 30.0) -> bool:
        """
        Check whether ``url`` exists on a remote server using an HTTP HEAD request.

        :param url: Remote URL to check.
        :param timeout: Request timeout in seconds (Default = ``30.0``).
        :return: ``True`` if HTTP status is OK (200), ``False`` otherwise.

        .. code:: python

            import rsgispy.tools.http
            exists = rsgispy.tools.http.check_url_exists("https://example.com/data.zip")
        """
        resp = requests.head(url, timeout=timeout)
        return resp.status_code == requests.codes.ok  # pylint: disable=no-member

    def check_http_response(response: Any, url: str) -> bool:
        """
        Validate an HTTP response and raise :class:`RSGISPyResponseException` on error.

        :param response: :class:`requests.Response` instance.
        :param url: Request URL used in fallback error messages.
        :return: ``True`` when response succeeded.

        .. code:: python

            import requests
            import rsgispy.tools.http
            resp = requests.get("https://example.com/data.json", timeout=30.0)
            ok = rsgispy.tools.http.check_http_response(resp, "https://example.com/data.json")
        """
        try:
            response.raise_for_status()
            return True
        except (requests.HTTPError, ValueError):
            excpt_msg = (
                f"Unknown error ('{response.reason}'), "
                f"check url in a web browser: '{url}'"
            )
            if "cause-message" in response.headers:
                excpt_msg = response.headers["cause-message"]
            else:
                try:
                    err_json = response.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_obj = err_json["error"]
                        if isinstance(err_obj, dict) and "message" in err_obj:
                            msg_obj = err_obj["message"]
                            excpt_msg = (
                                msg_obj["value"]
                                if isinstance(msg_obj, dict) and "value" in msg_obj
                                else str(msg_obj)
                            )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            api_error = RSGISPyResponseException(excpt_msg, response)
            api_error.__cause__ = None
            raise api_error from None

    def send_http_json_request(
        url: str,
        data: Any = None,
        api_key: Optional[str] = None,
        convert_to_json: bool = True,
        header_data: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> Any:
        """
        Send an HTTP POST request and parse the JSON response.

        :param url: Target endpoint URL.
        :param data: Payload dictionary or raw data to send.
        :param api_key: Optional ``X-Auth-Token`` header value.
        :param convert_to_json: Serialise ``data`` with ``json.dumps`` when ``True``.
        :param header_data: Optional dictionary of HTTP request headers.
        :param timeout: Request timeout in seconds (Default = ``60.0``).
        :return: Parsed JSON response object.

        .. code:: python

            import rsgispy.tools.http
            resp = rsgispy.tools.http.send_http_json_request(
                "https://example.com/api/query", data={"limit": 10}
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        params_data = (
            None if data is None else json.dumps(data)
        ) if convert_to_json else data

        headers = dict(header_data) if header_data is not None else {}
        if api_key is not None:
            headers["X-Auth-Token"] = api_key

        response = requests.post(
            url,
            params_data,
            headers=headers or None,
            timeout=timeout,
        )
        try:
            http_status_code = response.status_code
            if http_status_code == 404:
                raise rsgispy.RSGISPyException("404 Not Found")
            if http_status_code == 401:
                raise rsgispy.RSGISPyException("401 Unauthorized")
            if http_status_code == 400:
                raise rsgispy.RSGISPyException(f"Error Code: {http_status_code}")

            output = json.loads(response.text)
        except rsgispy.RSGISPyException:
            response.close()
            raise
        except Exception as err:
            response.close()
            raise rsgispy.RSGISPyException(f"{err}") from err
        response.close()

        return output

    def download_file_http(
        input_url: str,
        out_file_path: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        no_except: bool = True,
        timeout: float = 60.0,
    ) -> bool:
        """
        Download a file from ``input_url`` to ``out_file_path``, staging to
        ``out_file_path + '.incomplete'`` until finished.

        :param input_url: Remote file URL.
        :param out_file_path: Local destination file path.
        :param username: Optional basic-auth username.
        :param password: Optional basic-auth password.
        :param no_except: Return ``False`` on failure instead of raising when ``True``.
        :param timeout: Request timeout in seconds (Default = ``60.0``).
        :return: ``True`` if download succeeded, ``False`` otherwise.

        .. code:: python

            import rsgispy.tools.http
            ok = rsgispy.tools.http.download_file_http(
                "https://example.com/data.zip", "data.zip"
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        tqdm_cls = rsgispy.get_tqdm()
        session_http = requests.Session()
        if (username is not None) and (password is not None):
            session_http.auth = (username, password)
        session_http.headers["User-Agent"] = f"rsgispy/{rsgispy.get_rsgispy_version()}"

        tmp_dwnld_path = f"{out_file_path}.incomplete"

        try:
            with session_http.get(
                input_url,
                stream=True,
                auth=session_http.auth,
                timeout=timeout,
            ) as resp:
                if check_http_response(resp, input_url):
                    total = int(resp.headers.get("content-length", 0))
                    chunk_size = 2**20
                    n_chunks = int(total / chunk_size) + 1

                    with open(tmp_dwnld_path, "wb") as out_fp:
                        for chunk in tqdm_cls(
                            resp.iter_content(chunk_size=chunk_size), total=n_chunks
                        ):
                            if chunk:
                                out_fp.write(chunk)
            if os.path.exists(tmp_dwnld_path):
                os.rename(tmp_dwnld_path, out_file_path)
                print(f"Download Complete: {out_file_path}")
        except Exception as err:  # pylint: disable=broad-exception-caught
            if no_except:
                print(err)
                return False
            raise rsgispy.RSGISPyException(f"{err}") from err
        return True


def wget_download_file(
    input_url: str,
    out_file_path: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    try_number: int = 10,
    time_out: int = 60,
    input_url_md5: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download a file using the ``wget`` CLI tool with optional MD5 verification.

    :param input_url: Remote file URL.
    :param out_file_path: Local destination file path.
    :param username: Optional username for ``wget``.
    :param password: Optional password for ``wget``.
    :param try_number: Retry count (Default = ``10``).
    :param time_out: Timeout in seconds (Default = ``60``).
    :param input_url_md5: Optional expected MD5 hex digest.
    :return: Tuple ``(success, message)``.

    .. code:: python

        import rsgispy.tools.http
        ok, msg = rsgispy.tools.http.wget_download_file(
            "https://example.com/data.zip", "data.zip"
        )
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    success = False
    out_message = ""
    command = [
        "wget",
        "-c",
        "-O",
        out_file_path,
        "-t",
        str(try_number),
        "-T",
        str(time_out),
        "--no-check-certificate",
    ]
    if (username is not None) and (password is not None):
        command.extend(["--user", username, "--password", password])
    command.append(input_url)

    download_state = -1
    try:
        download_state = subprocess.call(command)
    except Exception as err:  # pylint: disable=broad-exception-caught
        out_message = f"Download of file ({out_file_path}) failed.: Exception:\n{err}"

    if download_state == 0:
        if input_url_md5 is not None:
            dwnld_file_md5 = rsgispy.tools.hashing.create_md5_hash(out_file_path)
            if dwnld_file_md5 == input_url_md5:
                success = True
                out_message = "File Downloaded and MD5 to checked."
            else:
                success = False
                out_message = "File Downloaded but MD5 did not match."
        else:
            success = True
            out_message = "File Downloaded but no MD5 to check against."

    if out_message == "":
        out_message = "File did not successfully download but no exception thrown."

    return success, out_message


if HAVE_PYSONDB:

    def create_file_listings_db(
        db_json: str,
        file_urls: Dict[str, str],
    ) -> None:
        """
        Build a ``pysondb`` JSON tracking database for HTTP file URLs.

        :param db_json: Path to the JSON database file.
        :param file_urls: Mapping of ``filename -> URL``.

        .. code:: python

            import rsgispy.tools.http
            rsgispy.tools.http.create_file_listings_db(
                "http_files.json", {"data.zip": "https://example.com/data.zip"}
            )
        """
        tqdm_cls = rsgispy.get_tqdm()
        lst_db = pysondb.getDb(db_json)
        db_data = [
            {
                "http_url": file_urls[c_file],
                "file_name": c_file,
                "lcl_path": "",
                "downloaded": False,
            }
            for c_file in tqdm_cls(file_urls)
        ]
        if db_data:
            lst_db.addMany(db_data)

    def download_http_files_use_lst_db(
        db_json: str,
        out_dir_path: str,
        http_user: Optional[str] = None,
        http_pass: Optional[str] = None,
        use_wget: bool = False,
        wget_time_out: int = 60,
        check_file_exists: bool = False,
    ) -> None:
        """
        Download pending files listed in a ``pysondb`` JSON database via HTTP or ``wget``.

        :param db_json: Path to the JSON database file.
        :param out_dir_path: Output directory where files will be saved.
        :param http_user: Optional HTTP username.
        :param http_pass: Optional HTTP password.
        :param use_wget: Use ``wget`` instead of ``requests`` when ``True``.
        :param wget_time_out: ``wget`` timeout in seconds (Default = ``60``).
        :param check_file_exists: Skip downloading if target file already exists locally.

        .. code:: python

            import rsgispy.tools.http
            rsgispy.tools.http.download_http_files_use_lst_db(
                "http_files.json", "./downloads", check_file_exists=True
            )
        """
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        if not use_wget and not HAVE_REQUESTS:
            raise rsgispy.RSGISPyException(
                "The requests module is required when use_wget=False."
            )

        lst_db = pysondb.getDb(db_json)
        dwld_files = lst_db.getByQuery({"downloaded": False})
        os.makedirs(out_dir_path, exist_ok=True)

        for dwn_file in dwld_files:
            basename = dwn_file["file_name"]
            lcl_path = os.path.join(out_dir_path, basename)
            file_exists = os.path.exists(lcl_path) if check_file_exists else False
            if not file_exists:
                if use_wget:
                    dwnlded, _ = wget_download_file(
                        input_url=dwn_file["http_url"],
                        out_file_path=lcl_path,
                        username=http_user,
                        password=http_pass,
                        try_number=10,
                        time_out=wget_time_out,
                        input_url_md5=None,
                    )
                else:
                    # pylint: disable-next=possibly-used-before-assignment
                    dwnlded = download_file_http(
                        input_url=dwn_file["http_url"],
                        out_file_path=lcl_path,
                        username=http_user,
                        password=http_pass,
                        no_except=True,
                    )
            else:
                dwnlded = True

            if dwnlded:
                lst_db.updateById(
                    dwn_file["id"], {"lcl_path": lcl_path, "downloaded": True}
                )
