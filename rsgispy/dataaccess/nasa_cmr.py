"""
Tools for accessing (searching and downloading) datasets from the NASA
EOSDIS Common Metadata Repository API (https://cmr.earthdata.nasa.gov/search/).

Classes and functions requiring ``requests`` or ``pysondb`` raise
:class:`rsgispy.RSGISPyException` if those optional dependencies are not
installed.
"""

import datetime
import re
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import rsgispy
from rsgispy.tools._download_common import (
    process_download_queue_db,
    stream_http_download,
    write_download_queue_db,
)
import rsgispy.tools.credentials
import rsgispy.tools.dicts
import rsgispy.tools.files
import rsgispy.tools.http

try:
    import requests  # pylint: disable=import-error

    HAVE_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAVE_REQUESTS = False

try:
    import pysondb  # pylint: disable=import-error

    HAVE_PYSONDB = True
except ImportError:
    pysondb = None  # type: ignore[assignment]
    HAVE_PYSONDB = False

_SessionBase = requests.Session if HAVE_REQUESTS else object

CMR_COLLECTS_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

__all__ = [
    "CMR_COLLECTS_URL",
    "CMR_GRANULES_URL",
    "get_total_file_size",
    "SessionWithHeaderNASARedirection",
    "get_prods_info",
    "check_prod_version_avail",
    "get_max_prod_version",
    "find_granules",
    "find_all_granules",
    "cmr_download_file_http",
    "create_cmr_dwnld_db",
    "download_granules_use_dwnld_db",
]


class SessionWithHeaderNASARedirection(_SessionBase):
    """``requests.Session`` that preserves auth headers across Earthdata redirects."""

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        rsgispy.check_optional_dep(
            HAVE_REQUESTS,
            "SessionWithHeaderNASARedirection",
            "requests",
            "dataaccess",
        )
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """
        Keep credentials only when redirected over HTTPS to ``urs.earthdata.nasa.gov``.
        """
        _ = response
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            redirect_parsed = urlparse(url)
            if not (
                redirect_parsed.hostname == self.AUTH_HOST
                and redirect_parsed.scheme == "https"
            ):
                del headers["Authorization"]


def _check_cmr_response(data: Dict) -> Union[str, List, Dict]:
    """Validate a CMR response dictionary and extract ``feed.entry``."""
    if not rsgispy.tools.dicts.dict_struct_does_path_exist(data, ["feed", "entry"]):
        print(data)
        raise rsgispy.RSGISPyException("Data structure is not as expected - check.")
    return data["feed"]["entry"]


def _version_sort_key(
    version_str: str,
) -> Tuple[Tuple[int, Union[int, str]], ...]:
    """Build a natural-order comparison key for CMR product version strings."""
    cleaned = str(version_str).strip()
    tokens = re.findall(r"\d+|[^\d.\-_]+", cleaned)
    if not tokens:
        return ((0, cleaned.lower()),)
    key_parts: List[Tuple[int, Union[int, str]]] = []
    for tok in tokens:
        if tok.isdigit():
            key_parts.append((1, int(tok)))
        else:
            key_parts.append((0, tok.lower()))
    return tuple(key_parts)


def get_prods_info(prod_short_name: str) -> List[Dict]:
    """
    Return collection metadata from CMR for ``prod_short_name``.

    :param prod_short_name: Product short name.
    :return: List of collection entries (often across versions).

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr
        prods = nasa_cmr.get_prods_info("ATL08")
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "get_prods_info", "requests", "dataaccess"
    )
    prod_srch_params = {"short_name": prod_short_name}
    header_info = {"Accept": "application/json"}

    rtnd_data = rsgispy.tools.http.send_http_json_request(
        CMR_COLLECTS_URL,
        data=prod_srch_params,
        convert_to_json=False,
        header_data=header_info,
    )
    return _check_cmr_response(rtnd_data)


def check_prod_version_avail(prod_short_name: str, version: str) -> bool:
    """
    Check whether a given ``version`` is available for ``prod_short_name``.

    :param prod_short_name: Product short name.
    :param version: Version string to check.
    :return: ``True`` if that version exists in CMR.

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr
        avail = nasa_cmr.check_prod_version_avail("ATL08", "006")
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "check_prod_version_avail", "requests", "dataaccess"
    )
    prod_lst = get_prods_info(prod_short_name)
    return any(prod["version_id"] == version for prod in prod_lst)


def get_max_prod_version(prod_short_name: str) -> str:
    """
    Return the highest (latest) version string for a product.

    Handles both integer version IDs (e.g., ``"006"``) and non-integer
    version strings (e.g., ``"6.1"``, ``"V001"``).

    :param prod_short_name: Product short name.
    :return: String representation of the highest version.

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr
        latest = nasa_cmr.get_max_prod_version("ATL08")
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "get_max_prod_version", "requests", "dataaccess"
    )
    prod_lst = get_prods_info(prod_short_name)
    best_key: Optional[Tuple[Tuple[int, Union[int, str]], ...]] = None
    max_version = ""
    for prod in prod_lst:
        version_id = str(prod["version_id"])
        version_key = _version_sort_key(version_id)
        if best_key is None or version_key > best_key:
            best_key = version_key
            max_version = version_id
    return max_version


# pylint: disable=too-many-arguments,too-many-positional-arguments
def find_granules(
    prod_short_name: str,
    version: str,
    only_dnwld: bool = True,
    bbox: Optional[List[float]] = None,
    pt: Optional[List[float]] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_min: int = 0,
    cloud_max: Optional[int] = None,
    sort_date: bool = True,
    sort_desc: bool = True,
    page_size: int = 100,
    page_num: int = 1,
    other_params: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Query a single page of granules from the NASA CMR API.

    :param prod_short_name: Product short name.
    :param version: Product version string.
    :param only_dnwld: If ``True``, restrict to downloadable granules.
    :param bbox: Bounding box ``[x_min, x_max, y_min, y_max]``.
    :param pt: Point ``[x, y]`` (takes precedence over ``bbox`` if given).
    :param start_date: Start acquisition datetime.
    :param end_date: End acquisition datetime.
    :param cloud_min: Minimum cloud cover (default: 0).
    :param cloud_max: Maximum cloud cover.
    :param sort_date: Whether to sort by acquisition start date.
    :param sort_desc: Sort descending when ``True``, ascending otherwise.
    :param page_size: Number of granules per page.
    :param page_num: 1-based page number.
    :param other_params: Optional dictionary of additional CMR parameters.
    :return: List of granule metadata dictionaries.

    .. code:: python

        import datetime
        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        granules = nasa_cmr.find_granules(
            prod_short_name="ATL08",
            version="006",
            bbox=[-5.0, -3.0, 52.0, 53.0],
            start_date=datetime.datetime(2023, 1, 1),
            end_date=datetime.datetime(2023, 6, 30),
        )
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "find_granules", "requests", "dataaccess"
    )
    prod_srch_params: Dict[str, Union[str, int, bool]] = {
        "short_name": prod_short_name,
        "version": version,
        "page_size": page_size,
        "page_num": page_num,
    }
    header_info = {"Accept": "application/json"}

    if only_dnwld:
        prod_srch_params["downloadable"] = True

    if start_date is not None or end_date is not None:
        start_date_str = (
            start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if start_date is not None
            else ""
        )
        end_date_str = (
            end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if end_date is not None
            else ""
        )
        prod_srch_params["temporal"] = f"{start_date_str},{end_date_str}"

    if cloud_min > 0 or cloud_max is not None:
        cloud_min_str = f"{cloud_min}" if cloud_min > 0 else ""
        cloud_max_str = f"{cloud_max}" if cloud_max is not None else ""
        prod_srch_params["cloud_cover"] = f"{cloud_min_str},{cloud_max_str}"

    if pt is not None:
        prod_srch_params["point"] = f"{pt[0]},{pt[1]}"
    elif bbox is not None:
        prod_srch_params["bounding_box"] = (
            f"{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}"
        )

    if sort_date:
        prod_srch_params["sort_key"] = "-start_date" if sort_desc else "+start_date"

    if other_params is not None:
        prod_srch_params = {**prod_srch_params, **other_params}

    rtnd_data = rsgispy.tools.http.send_http_json_request(
        CMR_GRANULES_URL,
        data=prod_srch_params,
        convert_to_json=False,
        header_data=header_info,
    )
    return _check_cmr_response(rtnd_data)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def find_all_granules(
    prod_short_name: str,
    version: str,
    only_dnwld: bool = True,
    bbox: Optional[List[float]] = None,
    pt: Optional[List[float]] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_min: int = 0,
    cloud_max: Optional[int] = None,
    sort_date: bool = True,
    sort_desc: bool = True,
    page_size: int = 100,
    max_n_pages: int = 100,
    other_params: Optional[Dict[str, str]] = None,
    use_pbar: bool = True,
) -> List[Dict]:
    """
    Paginate through :func:`find_granules` up to ``max_n_pages`` pages.

    :param use_pbar: If ``True``, display a tqdm progress bar during pagination
                     (Default = ``True``).
    :return: Combined list of granule metadata dictionaries.

    .. code:: python

        import datetime
        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        granules = nasa_cmr.find_all_granules(
            prod_short_name="ATL08",
            version="006",
            bbox=[-5.0, -3.0, 52.0, 53.0],
            start_date=datetime.datetime(2023, 1, 1),
            end_date=datetime.datetime(2023, 6, 30),
            max_n_pages=10,
        )
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "find_all_granules", "requests", "dataaccess"
    )
    tqdm_cls = rsgispy.get_tqdm()
    rtn_granules: List[Dict] = []
    print("Note. Progress bar is for the maximum number of pages...")
    print("Hopefully you do not need the maximum so it will be quicker...")
    for i in tqdm_cls(range(max_n_pages), disable=not use_pbar):
        page_granules = find_granules(
            prod_short_name=prod_short_name,
            version=version,
            only_dnwld=only_dnwld,
            bbox=bbox,
            pt=pt,
            start_date=start_date,
            end_date=end_date,
            cloud_min=cloud_min,
            cloud_max=cloud_max,
            sort_date=sort_date,
            sort_desc=sort_desc,
            page_size=page_size,
            page_num=i + 1,
            other_params=other_params,
        )
        if page_granules:
            rtn_granules += page_granules
        else:
            break

    return rtn_granules


# pylint: disable=too-many-arguments,too-many-positional-arguments
def cmr_download_file_http(
    input_url: str,
    out_file_path: str,
    username: str,
    password: str,
    no_except: bool = True,
    use_pbar: bool = True,
) -> bool:
    """
    Download a CMR granule file over HTTP using Earthdata redirect authentication.

    :param input_url: Remote URL to download.
    :param out_file_path: Local destination path.
    :param username: Earthdata username.
    :param password: Earthdata password.
    :param no_except: If ``True``, print errors and return ``False`` instead of raising.
    :param use_pbar: If ``True``, display a tqdm progress bar during download
                     (Default = ``True``).
    :return: ``True`` if download succeeded.

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        ok = nasa_cmr.cmr_download_file_http(
            input_url="https://data.nsidc.earthdatacloud.nasa.gov/example.h5",
            out_file_path="example.h5",
            username="earthdata_user",
            password="earthdata_password",
        )
    """
    rsgispy.check_optional_dep(
        HAVE_REQUESTS, "cmr_download_file_http", "requests", "dataaccess"
    )
    session_http = SessionWithHeaderNASARedirection(username, password)
    session_http.headers["User-Agent"] = f"rsgispy/{rsgispy.get_rsgispy_version()}"

    try:
        stream_http_download(
            session=session_http,
            url=input_url,
            out_file_path=out_file_path,
            timeout=120,
            use_pbar=use_pbar,
            check_response_fn=rsgispy.tools.http.check_http_response,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if no_except:
            print(exc)
            return False
        raise rsgispy.RSGISPyException(f"{exc}") from exc
    return True


def get_total_file_size(granule_lst: List[Dict]) -> float:
    """
    Sum ``granule_size`` across a list of granules returned by CMR.

    :param granule_lst: List of granule dictionaries.
    :return: Total size (typically MB).

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        granules = nasa_cmr.find_granules("ATL08", "006", bbox=[-5.0, -3.0, 52.0, 53.0])
        total_mb = nasa_cmr.get_total_file_size(granules)
    """
    return sum(float(granule["granule_size"]) for granule in granule_lst)


def create_cmr_dwnld_db(
    db_json: str,
    granule_lst: List[Dict],
    dwnld_file_mime_type: str,
    use_pbar: bool = True,
) -> List[str]:
    """
    Build a pysondb JSON download-tracking database from a CMR granule list.

    :param db_json: Path to the JSON database file.
    :param granule_lst: List of granules from :func:`find_granules` or
                        :func:`find_all_granules`.
    :param dwnld_file_mime_type: Link MIME type (e.g., ``"application/x-hdfeos"``).
    :param use_pbar: If ``True``, display a tqdm progress bar during processing
                     (Default = ``True``).
    :return: List of ``producer_granule_id`` values with no matching URL.

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        granules = nasa_cmr.find_granules("ATL08", "006", bbox=[-5.0, -3.0, 52.0, 53.0])
        missing = nasa_cmr.create_cmr_dwnld_db(
            db_json="cmr_downloads.json",
            granule_lst=granules,
            dwnld_file_mime_type="application/x-hdfeos",
        )
    """
    rsgispy.check_optional_dep(
        HAVE_PYSONDB, "create_cmr_dwnld_db", "pysondb", "dataaccess"
    )
    tqdm_cls = rsgispy.get_tqdm()
    db_data = []
    granules_no_url = []
    for granule in tqdm_cls(granule_lst, disable=not use_pbar):
        id_val = granule["id"]
        producer_granule_id = granule["producer_granule_id"]
        granule_size = granule["granule_size"]
        granule_url = None
        for link in granule["links"]:
            if link.get("type") == dwnld_file_mime_type:
                granule_url = link["href"]
                break
        if granule_url is not None:
            db_data.append(
                {
                    "id": id_val,
                    "producer_granule_id": producer_granule_id,
                    "granule_size": granule_size,
                    "granule_url": granule_url,
                    "lcl_path": "",
                    "downloaded": False,
                }
            )
        else:
            granules_no_url.append(producer_granule_id)

    write_download_queue_db(pysondb, db_json, db_data)

    return granules_no_url


def download_granules_use_dwnld_db(
    db_json: str,
    out_path: str,
    user_pass_file: str,
    use_wget: bool = False,
    use_pbar: bool = True,
) -> None:
    """
    Download pending granules recorded in a JSON database created by
    :func:`create_cmr_dwnld_db`.

    :param db_json: Path to the JSON database file.
    :param out_path: Local directory for downloaded files.
    :param user_pass_file: Path to the base64-encoded username/password file.
    :param use_wget: If ``True``, download using ``wget`` instead of ``requests``.
    :param use_pbar: If ``True``, display tqdm progress bars during downloads
                     (Default = ``True``).

    .. code:: python

        import rsgispy.dataaccess.nasa_cmr as nasa_cmr

        nasa_cmr.download_granules_use_dwnld_db(
            db_json="cmr_downloads.json",
            out_path="./downloads",
            user_pass_file="earthdata_creds.txt",
        )
    """
    rsgispy.check_optional_dep(
        HAVE_PYSONDB, "download_granules_use_dwnld_db", "pysondb", "dataaccess"
    )
    if not use_wget:
        rsgispy.check_optional_dep(
            HAVE_REQUESTS,
            "download_granules_use_dwnld_db",
            "requests",
            "dataaccess",
        )

    username, password = rsgispy.tools.credentials.get_username_password(user_pass_file)

    def _download_record(dwn_file: Dict) -> Tuple[bool, str]:
        # The granule ID comes from the CMR response: keep it in out_path.
        out_file_path = rsgispy.tools.files.safe_join(
            out_path, dwn_file["producer_granule_id"]
        )
        print(f"Downloading: {out_file_path}")
        if use_wget:
            dwn_success, _ = rsgispy.tools.http.wget_download_file(
                input_url=dwn_file["granule_url"],
                out_file_path=out_file_path,
                username=username,
                password=password,
            )
        else:
            dwn_success = cmr_download_file_http(
                input_url=dwn_file["granule_url"],
                out_file_path=out_file_path,
                username=username,
                password=password,
                use_pbar=use_pbar,
            )
        return dwn_success, out_file_path

    process_download_queue_db(
        pysondb, db_json, _download_record, use_pbar=use_pbar
    )
