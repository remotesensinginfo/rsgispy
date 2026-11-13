"""
Tools for accessing (searching and downloading) datasets from the USGS M2M API.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

import rsgispy
import rsgispy.tools.httptools
import rsgispy.tools.utils

DATA_PRODUCTS: Dict[str, str] = {
    "landsat_tm_c1": "5e83d08fd9932768",
    "landsat_etm_c1": "5e83a507d6aaa3db",
    "landsat_8_c1": "5e83d0b84df8d8c2",
    "landsat_tm_c2_l1": "5e83d0a0f94d7d8d",
    "landsat_etm_c2_l1": "5e83d0d0d2aaa488",
    "landsat_ot_c2_l1": "5e81f14ff4f9941c",
    "landsat_mss_c2_l1": "5e83d0e09a752cff",
    "landsat_tm_c2_l2": "5e83d11933473426",
    "landsat_etm_c2_l2": "5e83d12aada2e3c5",
    "landsat_ot_c2_l2": "5e83d14f30ea90a9",
    "sentinel_2a": "5e83a42c6eba8084",
}

USGS_WRS1 = 1
USGS_WRS2 = 2

USGS_M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"

__all__ = [
    "DATA_PRODUCTS",
    "USGS_WRS1",
    "USGS_WRS2",
    "USGS_M2M_URL",
    "usgs_login",
    "usgs_logout",
    "can_user_dwnld",
    "can_user_order",
    "get_wrs_pt",
    "get_wrs_bbox",
    "get_dataset_info",
    "get_earth_explorer_datasets",
    "usgs_search",
    "get_all_usgs_search",
    "get_download_ids",
    "create_scene_list",
    "remove_scene_list",
    "check_dwnld_opts",
    "request_downloads",
]


def _check_usgs_response(data: Dict) -> Any:
    """Validate a USGS M2M JSON response and return its ``data`` payload."""
    if (
        rsgispy.tools.utils.dict_struct_does_path_exist(data, ["errorCode"])
        and data["errorCode"] is not None
    ):
        raise rsgispy.RSGISPyException(
            f"{data['errorCode']} - {data['errorMessage']}"
        )

    if not rsgispy.tools.utils.dict_struct_does_path_exist(data, ["data"]):
        print(data)
        raise rsgispy.RSGISPyException("Data structure is not as expected - check.")

    return data["data"]


def _validate_dataset(dataset: str, api_key: str) -> None:
    """Ensure ``dataset`` is a known USGS EarthExplorer dataset alias."""
    if dataset in DATA_PRODUCTS:
        return
    all_datasets = get_earth_explorer_datasets(api_key)
    for cat_datasets in all_datasets.values():
        if dataset in cat_datasets:
            return
    raise rsgispy.RSGISPyException(f"Do not recognise dataset: {dataset}")


def usgs_login(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """
    Log in to the USGS M2M service and return a session API key.

    :param username: USGS EarthExplorer username. Reads ``RSGIS_USGS_USER`` if ``None``.
    :param password: USGS EarthExplorer password. Reads ``RSGIS_USGS_PASS`` if ``None``.
    :return: API key string for the USGS session.
    """
    if username is None:
        username = rsgispy.tools.utils.get_environment_variable("RSGIS_USGS_USER")

    if password is None:
        password = rsgispy.tools.utils.get_environment_variable("RSGIS_USGS_PASS")

    login_data = {"username": username, "password": password}
    login_url = USGS_M2M_URL + "login"
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(login_url, login_data)
    return _check_usgs_response(rtnd_data)


def usgs_logout(api_key: str) -> None:
    """
    Log out of the USGS M2M system using the session ``api_key``.

    :param api_key: API key created at login.
    """
    logout_url = USGS_M2M_URL + "logout"
    rsgispy.tools.httptools.send_http_json_request(logout_url, None, api_key)


def _has_user_permission(api_key: str, permission: str) -> bool:
    permissions_url = USGS_M2M_URL + "permissions"
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        permissions_url, api_key=api_key
    )
    permissions_info = _check_usgs_response(rtnd_data)
    return permissions_info is not None and permission in permissions_info


def can_user_dwnld(api_key: str) -> bool:
    """
    Check whether the authenticated user has permission to download data.

    :param api_key: API key created at login.
    :return: ``True`` if download permission is granted.
    """
    return _has_user_permission(api_key, "download")


def can_user_order(api_key: str) -> bool:
    """
    Check whether the authenticated user has permission to order data.

    :param api_key: API key created at login.
    :return: ``True`` if order permission is granted.
    """
    return _has_user_permission(api_key, "order")


def _query_grid2ll(
    api_key: str,
    row: int,
    path: int,
    grid_version: int,
    response_shape: str,
) -> Dict:
    grid2ll_url = USGS_M2M_URL + "grid2ll"
    grid2ll_data = {
        "gridType": "WRS1" if grid_version == USGS_WRS1 else "WRS2",
        "path": path,
        "row": row,
        "responseShape": response_shape,
    }
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        grid2ll_url, grid2ll_data, api_key=api_key
    )
    return _check_usgs_response(rtnd_data)


def get_wrs_pt(
    api_key: str,
    row: int,
    path: int,
    grid_version: int = USGS_WRS2,
) -> Tuple[float, float]:
    """
    Get a centre point ``(longitude, latitude)`` for a WRS row/path.

    :param api_key: API key created at login.
    :param row: WRS row number.
    :param path: WRS path number.
    :param grid_version: ``USGS_WRS1`` or ``USGS_WRS2`` (default: ``USGS_WRS2``).
    :return: ``(longitude, latitude)`` tuple.
    """
    grid2ll_info = _query_grid2ll(api_key, row, path, grid_version, "point")
    if grid2ll_info["shape"] != "point":
        raise rsgispy.RSGISPyException("The returned shape was not a point.")
    coord = grid2ll_info["coordinates"][0]
    return coord["longitude"], coord["latitude"]


def get_wrs_bbox(
    api_key: str,
    row: int,
    path: int,
    grid_version: int = USGS_WRS2,
) -> Tuple[float, float, float, float]:
    """
    Get a bounding box ``(x_min, x_max, y_min, y_max)`` for a WRS row/path.

    :param api_key: API key created at login.
    :param row: WRS row number.
    :param path: WRS path number.
    :param grid_version: ``USGS_WRS1`` or ``USGS_WRS2`` (default: ``USGS_WRS2``).
    :return: ``(min_lon, max_lon, min_lat, max_lat)`` tuple.
    """
    grid2ll_info = _query_grid2ll(api_key, row, path, grid_version, "polygon")
    if grid2ll_info["shape"] != "polygon":
        raise rsgispy.RSGISPyException("The returned shape was not a polygon.")

    lat_vals = [coord["latitude"] for coord in grid2ll_info["coordinates"]]
    lon_vals = [coord["longitude"] for coord in grid2ll_info["coordinates"]]
    return min(lon_vals), max(lon_vals), min(lat_vals), max(lat_vals)


def get_dataset_info(dataset: str, api_key: str) -> Dict:
    """
    Get metadata information for a specific USGS dataset.

    :param dataset: Name of the dataset.
    :param api_key: API key created at login.
    :return: Dictionary of dataset metadata.
    """
    dataset_url = USGS_M2M_URL + "dataset"
    dataset_data = {"datasetName": dataset}
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        dataset_url, dataset_data, api_key=api_key
    )
    return _check_usgs_response(rtnd_data)


def get_earth_explorer_datasets(api_key: str) -> Dict[str, List[str]]:
    """
    Get datasets available in EarthExplorer grouped by top-level category.

    :param api_key: API key created at login.
    :return: Dictionary mapping category names to lists of dataset aliases.
    """
    dataset_url = USGS_M2M_URL + "dataset-categories"
    dataset_data = {"catalog": "EE"}
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        dataset_url, dataset_data, api_key=api_key
    )
    dataset_info = _check_usgs_response(rtnd_data)

    dataset_names: Dict[str, List[str]] = {}
    for dataset_id in dataset_info:
        cat_name = dataset_info[dataset_id]["categoryName"]
        dataset_names[cat_name] = []
        if "subCategories" in dataset_info[dataset_id]:
            for sub_cat_id in dataset_info[dataset_id]["subCategories"]:
                if not isinstance(sub_cat_id, dict):
                    sub_datasets = dataset_info[dataset_id]["subCategories"][sub_cat_id][
                        "datasets"
                    ]
                    for dataset_item in sub_datasets:
                        dataset_names[cat_name].append(dataset_item["datasetAlias"])
        for dataset_item in dataset_info[dataset_id]["datasets"]:
            dataset_names[cat_name].append(dataset_item["datasetAlias"])
    return dataset_names


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def usgs_search(
    dataset: str,
    api_key: str,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_min: int = 0,
    cloud_max: Optional[int] = None,
    pt: Optional[List[float]] = None,
    bbox: Optional[List[float]] = None,
    poly_geom: Optional[str] = None,
    months: Optional[List[int]] = None,
    full_meta: bool = False,
    max_n_rslts: Optional[int] = None,
    start_n: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Search for imagery scenes using the USGS M2M API.

    :param dataset: Name of the dataset to query.
    :param api_key: API key created at login.
    :param start_date: Optional acquisition start datetime.
    :param end_date: Optional acquisition end datetime (defaults to ``now`` if
                     ``start_date`` is given).
    :param cloud_min: Minimum cloud cover percentage (default: 0).
    :param cloud_max: Optional maximum cloud cover percentage.
    :param pt: Optional point ``[lat, lon]`` (note USGS coordinate order in GeoJSON).
    :param bbox: Optional bounding box ``[min_x, max_x, min_y, max_y]``.
    :param poly_geom: Reserved for polygon filtering (not implemented).
    :param months: Optional list of months ``(1-12)`` for seasonal filtering.
    :param full_meta: Request full metadata when ``True``.
    :param max_n_rslts: Maximum number of scenes to return (must be <= 100).
    :param start_n: Optional starting record index for pagination.
    :return: Tuple of ``(scenes_list, search_metadata_dict)``.
    """
    _validate_dataset(dataset, api_key)

    if max_n_rslts is not None and max_n_rslts > 100:
        raise rsgispy.RSGISPyException(
            "The maximum number of results requested should not be higher than 100."
        )

    search_data: Dict[str, Any] = {"datasetName": dataset}
    if end_date is None:
        end_date = datetime.datetime.now()

    scn_filter: Dict[str, Any] = {}
    if start_date is not None:
        scn_filter["acquisitionFilter"] = {
            "start": f"{start_date}",
            "end": f"{end_date}",
        }

    if cloud_max is not None:
        scn_filter["cloudCoverFilter"] = {
            "min": cloud_min,
            "max": cloud_max,
        }

    if pt is not None:
        scn_filter["spatialFilter"] = {
            "filterType": "geojson",
            "geoJson": {
                "type": "Point",
                "coordinates": [pt[1], pt[0]],
            },
        }
    elif bbox is not None:
        print("Use bbox to filter.")
        scn_filter["spatialFilter"] = {
            "filterType": "mbr",
            "lowerLeft": {"longitude": bbox[0], "latitude": bbox[2]},
            "upperRight": {"longitude": bbox[1], "latitude": bbox[3]},
        }
    elif poly_geom is not None:
        raise rsgispy.RSGISPyException(
            "Trying to use a polygon to filter but not implemented yet."
        )

    if months is not None:
        scn_filter["seasonalFilter"] = months

    if scn_filter:
        search_data["sceneFilter"] = scn_filter
    if full_meta:
        search_data["metadataType"] = "full"
    if max_n_rslts is not None:
        search_data["maxResults"] = max_n_rslts
    if start_n is not None:
        search_data["startingNumber"] = start_n

    search_url = USGS_M2M_URL + "scene-search"
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        search_url, search_data, api_key=api_key
    )
    srch_data = _check_usgs_response(rtnd_data)

    scns = srch_data["results"] if srch_data["totalHits"] > 0 else []
    srch_meta = {
        "totalHits": srch_data["totalHits"],
        "startingNumber": srch_data["startingNumber"],
        "nextRecord": srch_data["nextRecord"],
        "recordsReturned": srch_data["recordsReturned"],
    }
    return scns, srch_meta


# pylint: disable=too-many-arguments,too-many-positional-arguments
def get_all_usgs_search(
    dataset: str,
    api_key: str,
    max_n_rslts: int = 1000,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_min: int = 0,
    cloud_max: Optional[int] = None,
    pt: Optional[List[float]] = None,
    bbox: Optional[List[float]] = None,
    poly_geom: Optional[str] = None,
    months: Optional[List[int]] = None,
    full_meta: bool = False,
) -> List[Dict]:
    """
    Paginate through :func:`usgs_search` to collect up to ``max_n_rslts`` scenes.

    :return: Combined list of scene dictionaries.
    """
    if max_n_rslts <= 100:
        scns, _ = usgs_search(
            dataset,
            api_key,
            start_date,
            end_date,
            cloud_min,
            cloud_max,
            pt,
            bbox,
            poly_geom,
            months,
            full_meta,
            max_n_rslts=max_n_rslts,
        )
    else:
        scns, srch_meta = usgs_search(
            dataset,
            api_key,
            start_date,
            end_date,
            cloud_min,
            cloud_max,
            pt,
            bbox,
            poly_geom,
            months,
            full_meta,
        )
        tot_n_scns = srch_meta["recordsReturned"]
        total_hits = srch_meta["totalHits"]
        if total_hits > 0:
            target_n = min(max_n_rslts, total_hits)
            while tot_n_scns < target_n and srch_meta.get("nextRecord") is not None:
                xtr_scns, srch_meta = usgs_search(
                    dataset,
                    api_key,
                    start_date,
                    end_date,
                    cloud_min,
                    cloud_max,
                    pt,
                    bbox,
                    poly_geom,
                    months,
                    full_meta,
                    start_n=srch_meta["nextRecord"],
                )
                if xtr_scns:
                    scns += xtr_scns
                    tot_n_scns += len(xtr_scns)
                else:
                    break

    if len(scns) > max_n_rslts:
        scns = scns[0:max_n_rslts]

    return scns


def get_download_ids(
    scns: List[Dict],
    bulk: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Extract display IDs and entity IDs from a list of searched USGS scenes.

    :param scns: List of scene dictionaries from :func:`usgs_search`.
    :param bulk: If ``True``, include only scenes available for bulk download.
    :return: Tuple ``(display_ids, entity_ids)``.
    """
    scn_dsp_ids: List[str] = []
    scn_ent_ids: List[str] = []
    for scn in scns:
        opts = scn.get("options", {})
        if opts.get("download") is True:
            if not bulk or opts.get("bulk") is True:
                scn_dsp_ids.append(scn["displayId"])
                scn_ent_ids.append(scn["entityId"])
    return scn_dsp_ids, scn_ent_ids


def create_scene_list(
    api_key: str,
    dataset: str,
    scn_ent_ids: List[str],
    lst_name: str,
    lst_period: str = "P1W",
) -> int:
    """
    Create a temporary scene list on the USGS M2M system for download checking.

    :param api_key: API key created at login.
    :param dataset: Name of the dataset.
    :param scn_ent_ids: List of scene entity IDs.
    :param lst_name: Name for the temporary scene list.
    :param lst_period: Retention duration in ISO 8601 format (default: ``"P1W"``).
    :return: Number of scenes added.
    """
    _validate_dataset(dataset, api_key)

    add_scn_info = {
        "listId": lst_name,
        "datasetName": dataset,
        "idField": "entityId",
        "entityIds": scn_ent_ids,
        "timeToLive": lst_period,
    }
    scn_lst_add_url = USGS_M2M_URL + "scene-list-add"
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        scn_lst_add_url, add_scn_info, api_key=api_key
    )
    return _check_usgs_response(rtnd_data)


def remove_scene_list(api_key: str, lst_name: str) -> None:
    """
    Remove a temporary scene list from the USGS M2M system.

    :param api_key: API key created at login.
    :param lst_name: Scene list name created with :func:`create_scene_list`.
    """
    rm_scn_info = {"listId": lst_name}
    scn_lst_rm_url = USGS_M2M_URL + "scene-list-remove"
    rsgispy.tools.httptools.send_http_json_request(
        scn_lst_rm_url, rm_scn_info, api_key=api_key
    )


def check_dwnld_opts(
    api_key: str,
    lst_name: str,
    dataset: str,
    dwnld_filetype: str = "bundle",
    rm_lst: bool = True,
) -> List[Dict[str, str]]:
    """
    Check available bulk download options for a scene list.

    :param api_key: API key created at login.
    :param lst_name: Scene list name created with :func:`create_scene_list`.
    :param dataset: Name of the dataset.
    :param dwnld_filetype: ``"bundle"``, ``"band"``, or ``"all"`` (default: ``"bundle"``).
    :param rm_lst: Remove the scene list after checking when ``True``.
    :return: List of ``{"entityId": ..., "productId": ...}`` dictionaries.
    """
    if not can_user_dwnld(api_key):
        raise rsgispy.RSGISPyException(
            "Your user account does not have permission to download data."
        )

    if dwnld_filetype not in ("bundle", "band", "all"):
        raise rsgispy.RSGISPyException(
            "dwnld_filetype is not recognised - must be either bundle, band or all."
        )

    dwnld_opts_params = {"listId": lst_name, "datasetName": dataset}
    dwnld_opts_url = USGS_M2M_URL + "download-options"
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        dwnld_opts_url, dwnld_opts_params, api_key
    )
    dnwld_opt_out_info = _check_usgs_response(rtnd_data)

    dwlds_lst: List[Dict[str, str]] = []
    for prod in dnwld_opt_out_info:
        if dwnld_filetype in ("bundle", "all") and prod["bulkAvailable"]:
            dwlds_lst.append({"entityId": prod["entityId"], "productId": prod["id"]})
        if dwnld_filetype == "band" or (
            dwnld_filetype == "all" and prod["bulkAvailable"]
        ):
            for sec_dwnld in prod.get("secondaryDownloads") or []:
                if sec_dwnld["bulkAvailable"]:
                    dwlds_lst.append(
                        {
                            "entityId": sec_dwnld["entityId"],
                            "productId": sec_dwnld["id"],
                        }
                    )

    if rm_lst:
        remove_scene_list(api_key, lst_name)

    return dwlds_lst


def request_downloads(
    api_key: str,
    dwlds_lst: List[Dict[str, str]],
    dwnld_label: str,
) -> Tuple[Dict[Any, str], List[Any]]:
    """
    Request download URLs for products returned by :func:`check_dwnld_opts`.

    :param api_key: API key created at login.
    :param dwlds_lst: Download list returned by :func:`check_dwnld_opts`.
    :param dwnld_label: Label for the download request.
    :return: Tuple ``(available_urls_by_id, preparing_download_ids)``.
    """
    dwnld_reqst_url = USGS_M2M_URL + "download-request"
    dwnld_reqst_data = {
        "downloads": dwlds_lst,
        "label": dwnld_label,
        "returnAvailable": True,
    }
    rtnd_data = rsgispy.tools.httptools.send_http_json_request(
        dwnld_reqst_url, dwnld_reqst_data, api_key=api_key
    )
    dwnld_reqst_info = _check_usgs_response(rtnd_data)

    avail_dwn_urls = {
        info["downloadId"]: info["url"]
        for info in dwnld_reqst_info["availableDownloads"]
    }
    prep_dwnld_ids = [
        info["downloadId"] for info in dwnld_reqst_info["preparingDownloads"]
    ]
    return avail_dwn_urls, prep_dwnld_ids
