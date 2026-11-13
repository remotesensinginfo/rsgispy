"""
Tools for accessing (searching and downloading) datasets from the NASA
EOSDIS Common Metadata Repository API (https://cmr.earthdata.nasa.gov/search/).

Classes and functions requiring ``requests`` or ``pysondb`` are only defined
and exported when those packages are installed.
"""

import datetime
import os
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import rsgispy
import rsgispy.tools.credentials
import rsgispy.tools.dicts
import rsgispy.tools.http

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

CMR_COLLECTS_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

__all__ = [
    "CMR_COLLECTS_URL",
    "CMR_GRANULES_URL",
    "get_total_file_size",
]

if HAVE_REQUESTS:
    __all__.extend(
        [
            "SessionWithHeaderNASARedirection",
            "get_prods_info",
            "check_prod_version_avail",
            "get_max_prod_version",
            "find_granules",
            "find_all_granules",
            "cmr_download_file_http",
        ]
    )

if HAVE_PYSONDB:
    __all__.extend(
        [
            "create_cmr_dwnld_db",
            "download_granules_use_dwnld_db",
        ]
    )


if HAVE_REQUESTS:

    class SessionWithHeaderNASARedirection(requests.Session):
        """``requests.Session`` that preserves auth headers across Earthdata redirects."""

        AUTH_HOST = "urs.earthdata.nasa.gov"

        def __init__(self, username: str, password: str):
            super().__init__()
            self.auth = (username, password)

        def rebuild_auth(self, prepared_request, response):
            """
            Keep credentials when redirected to or from ``urs.earthdata.nasa.gov``.
            """
            headers = prepared_request.headers
            url = prepared_request.url

            if "Authorization" in headers:
                original_parsed = urlparse(response.request.url)
                redirect_parsed = urlparse(url)
                if (
                    (original_parsed.hostname != redirect_parsed.hostname)
                    and redirect_parsed.hostname != self.AUTH_HOST
                    and original_parsed.hostname != self.AUTH_HOST
                ):
                    del headers["Authorization"]

    def _check_cmr_response(data: Dict) -> Union[str, List, Dict]:
        """Validate a CMR response dictionary and extract ``feed.entry``."""
        if not rsgispy.tools.dicts.dict_struct_does_path_exist(data, ["feed", "entry"]):
            print(data)
            raise rsgispy.RSGISPyException("Data structure is not as expected - check.")
        return data["feed"]["entry"]

    def get_prods_info(prod_short_name: str) -> List[Dict]:
        """
        Return collection metadata from CMR for ``prod_short_name``.

        :param prod_short_name: Product short name.
        :return: List of collection entries (often across versions).

        .. code:: python

            import rsgispy.dataaccess.nasa_cmr as nasa_cmr
            prods = nasa_cmr.get_prods_info("ATL08")
        """
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
        prod_lst = get_prods_info(prod_short_name)
        return any(prod["version_id"] == version for prod in prod_lst)

    def get_max_prod_version(prod_short_name: str) -> str:
        """
        Return the highest (latest) integer-comparable version string for a product.

        :param prod_short_name: Product short name.
        :return: String representation of the highest version.

        .. code:: python

            import rsgispy.dataaccess.nasa_cmr as nasa_cmr
            latest = nasa_cmr.get_max_prod_version("ATL08")
        """
        prod_lst = get_prods_info(prod_short_name)
        first = True
        max_version_int = 0
        max_version = ""
        for prod in prod_lst:
            version_int = int(prod["version_id"])
            if first or version_int > max_version_int:
                max_version_int = version_int
                max_version = prod["version_id"]
                first = False
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
                start_date.strftime("%Y-%m-%dT00:00:00Z")
                if start_date is not None
                else ""
            )
            end_date_str = (
                end_date.strftime("%Y-%m-%dT00:00:00Z") if end_date is not None else ""
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
            prod_srch_params["sort_key"] = "-start_date" if sort_desc else "%2Bstart_date"

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
    ) -> List[Dict]:
        """
        Paginate through :func:`find_granules` up to ``max_n_pages`` pages.

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
        # pylint: disable=import-outside-toplevel
        import tqdm

        rtn_granules: List[Dict] = []
        print("Note. Progress bar is for the maximum number of pages...")
        print("Hopefully you do not need the maximum so it will be quicker...")
        for i in tqdm.tqdm(range(max_n_pages)):
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

    def cmr_download_file_http(
        input_url: str,
        out_file_path: str,
        username: str,
        password: str,
        no_except: bool = True,
    ) -> bool:
        """
        Download a CMR granule file over HTTP using Earthdata redirect authentication.

        :param input_url: Remote URL to download.
        :param out_file_path: Local destination path.
        :param username: Earthdata username.
        :param password: Earthdata password.
        :param no_except: If ``True``, print errors and return ``False`` instead of raising.
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
        # pylint: disable=import-outside-toplevel
        import tqdm

        session_http = SessionWithHeaderNASARedirection(username, password)
        session_http.headers["User-Agent"] = f"rsgispy/{rsgispy.get_rsgispy_version()}"

        tmp_dwnld_path = f"{out_file_path}.incomplete"

        try:
            with session_http.get(input_url, stream=True, timeout=120) as resp:
                if rsgispy.tools.http.check_http_response(resp, input_url):
                    total = int(resp.headers.get("content-length", 0))
                    chunk_size = 2**20
                    n_chunks = int(total / chunk_size) + 1

                    with open(tmp_dwnld_path, "wb") as out_fh:
                        for chunk in tqdm.tqdm(
                            resp.iter_content(chunk_size=chunk_size), total=n_chunks
                        ):
                            if chunk:
                                out_fh.write(chunk)
            if os.path.exists(tmp_dwnld_path):
                os.rename(tmp_dwnld_path, out_file_path)
                print(f"Download Complete: {out_file_path}")
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


if HAVE_PYSONDB:

    def create_cmr_dwnld_db(
        db_json: str,
        granule_lst: List[Dict],
        dwnld_file_mime_type: str,
    ) -> List[str]:
        """
        Build a pysondb JSON download-tracking database from a CMR granule list.

        :param db_json: Path to the JSON database file.
        :param granule_lst: List of granules from :func:`find_granules` or
                            :func:`find_all_granules`.
        :param dwnld_file_mime_type: Link MIME type (e.g., ``"application/x-hdfeos"``).
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
        # pylint: disable=import-outside-toplevel
        import tqdm

        db_data = []
        granules_no_url = []
        for granule in tqdm.tqdm(granule_lst):
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

        lst_db = pysondb.getDb(db_json)
        if db_data:
            lst_db.addMany(db_data)

        return granules_no_url

    def download_granules_use_dwnld_db(
        db_json: str,
        out_path: str,
        user_pass_file: str,
        use_wget: bool = False,
    ) -> None:
        """
        Download pending granules recorded in a JSON database created by
        :func:`create_cmr_dwnld_db`.

        :param db_json: Path to the JSON database file.
        :param out_path: Local directory for downloaded files.
        :param user_pass_file: Path to the base64-encoded username/password file.
        :param use_wget: If ``True``, download using ``wget`` instead of ``requests``.

        .. code:: python

            import rsgispy.dataaccess.nasa_cmr as nasa_cmr

            nasa_cmr.download_granules_use_dwnld_db(
                db_json="cmr_downloads.json",
                out_path="./downloads",
                user_pass_file="earthdata_creds.txt",
            )
        """
        if not use_wget and not HAVE_REQUESTS:
            raise rsgispy.RSGISPyException(
                "The requests module is required when use_wget=False."
            )

        username, password = rsgispy.tools.credentials.get_username_password(user_pass_file)

        lst_db = pysondb.getDb(db_json)
        dwld_files = lst_db.getByQuery({"downloaded": False})

        for dwn_file in dwld_files:
            out_file_path = os.path.join(out_path, dwn_file["producer_granule_id"])
            print(f"Downloading: {out_file_path}")
            if use_wget:
                dwn_success, _ = rsgispy.tools.http.wget_download_file(
                    input_url=dwn_file["granule_url"],
                    out_file_path=out_file_path,
                    username=username,
                    password=password,
                )
            else:
                # pylint: disable=possibly-used-before-assignment
                dwn_success = cmr_download_file_http(
                    input_url=dwn_file["granule_url"],
                    out_file_path=out_file_path,
                    username=username,
                    password=password,
                )

            if dwn_success:
                lst_db.updateById(
                    dwn_file["id"], {"lcl_path": out_file_path, "downloaded": True}
                )
