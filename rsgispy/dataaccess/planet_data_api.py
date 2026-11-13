"""
Tools for accessing (searching and downloading) datasets from Planet.

Note to use this module you need to have the planet API installed:

.. code-block:: bash

   pip install planet

Item Types
-----------
    * RSGIS_PLANET_ITEM_PSScene = 1
    * RSGIS_PLANET_ITEM_REOrthoTile = 2
    * RSGIS_PLANET_ITEM_REScene = 3
    * RSGIS_PLANET_ITEM_SkySatScene = 4
    * RSGIS_PLANET_ITEM_SkySatCollect = 5
    * RSGIS_PLANET_ITEM_SkySatVideo = 6

Asset Types
------------
    * RSGIS_PLANET_ASSET_basic_analytic_4b = 1
    * RSGIS_PLANET_ASSET_basic_analytic_4b_rpc = 2
    * RSGIS_PLANET_ASSET_basic_analytic_4b_xml = 3
    * RSGIS_PLANET_ASSET_basic_analytic_8b = 4
    * RSGIS_PLANET_ASSET_basic_analytic_8b_xml = 5
    * RSGIS_PLANET_ASSET_basic_udm2 = 6
    * RSGIS_PLANET_ASSET_ortho_analytic_4b = 7
    * RSGIS_PLANET_ASSET_ortho_analytic_4b_sr = 8
    * RSGIS_PLANET_ASSET_ortho_analytic_4b_xml = 9
    * RSGIS_PLANET_ASSET_ortho_analytic_8b = 10
    * RSGIS_PLANET_ASSET_ortho_analytic_8b_sr = 11
    * RSGIS_PLANET_ASSET_ortho_analytic_8b_xml = 12
    * RSGIS_PLANET_ASSET_ortho_udm2 = 13
    * RSGIS_PLANET_ASSET_ortho_visual = 14

Bundle Types
---------------
    * RSGIS_PLANET_BUNDLE_analytic_udm2 = 1
    * RSGIS_PLANET_BUNDLE_analytic_3b_udm2 = 2
    * RSGIS_PLANET_BUNDLE_analytic_8b_udm2 = 3
    * RSGIS_PLANET_BUNDLE_visual = 4
    * RSGIS_PLANET_BUNDLE_basic_analytic_udm2 = 5
    * RSGIS_PLANET_BUNDLE_basic_analytic_8b_udm2 = 6
    * RSGIS_PLANET_BUNDLE_analytic_sr_udm2 = 7
    * RSGIS_PLANET_BUNDLE_analytic_8b_sr_udm2 = 8
"""

import asyncio
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import rsgispy
import rsgispy.tools.geometry
import rsgispy.tools.system

# pylint: disable=invalid-name
PLANET_AVAIL = True
try:
    import planet  # type: ignore[import-not-found]
except ImportError:
    PLANET_AVAIL = False

# Sensor Options
RSGIS_PLANET_ITEM_PSScene = 1
RSGIS_PLANET_ITEM_REOrthoTile = 2
RSGIS_PLANET_ITEM_REScene = 3
RSGIS_PLANET_ITEM_SkySatScene = 4
RSGIS_PLANET_ITEM_SkySatCollect = 5
RSGIS_PLANET_ITEM_SkySatVideo = 6

# Asset Options
RSGIS_PLANET_ASSET_basic_analytic_4b = 1
RSGIS_PLANET_ASSET_basic_analytic_4b_rpc = 2
RSGIS_PLANET_ASSET_basic_analytic_4b_xml = 3
RSGIS_PLANET_ASSET_basic_analytic_8b = 4
RSGIS_PLANET_ASSET_basic_analytic_8b_xml = 5
RSGIS_PLANET_ASSET_basic_udm2 = 6
RSGIS_PLANET_ASSET_ortho_analytic_4b = 7
RSGIS_PLANET_ASSET_ortho_analytic_4b_sr = 8
RSGIS_PLANET_ASSET_ortho_analytic_4b_xml = 9
RSGIS_PLANET_ASSET_ortho_analytic_8b = 10
RSGIS_PLANET_ASSET_ortho_analytic_8b_sr = 11
RSGIS_PLANET_ASSET_ortho_analytic_8b_xml = 12
RSGIS_PLANET_ASSET_ortho_udm2 = 13
RSGIS_PLANET_ASSET_ortho_visual = 14

# Bundle Options
RSGIS_PLANET_BUNDLE_analytic_udm2 = 1
RSGIS_PLANET_BUNDLE_analytic_3b_udm2 = 2
RSGIS_PLANET_BUNDLE_analytic_8b_udm2 = 3
RSGIS_PLANET_BUNDLE_visual = 4
RSGIS_PLANET_BUNDLE_basic_analytic_udm2 = 5
RSGIS_PLANET_BUNDLE_basic_analytic_8b_udm2 = 6
RSGIS_PLANET_BUNDLE_analytic_sr_udm2 = 7
RSGIS_PLANET_BUNDLE_analytic_8b_sr_udm2 = 8
# pylint: enable=invalid-name

_ITEM_TYPE_MAP: Dict[int, str] = {
    RSGIS_PLANET_ITEM_PSScene: "PSScene",
    RSGIS_PLANET_ITEM_REOrthoTile: "REOrthoTile",
    RSGIS_PLANET_ITEM_REScene: "REScene",
    RSGIS_PLANET_ITEM_SkySatScene: "SkySatScene",
    RSGIS_PLANET_ITEM_SkySatCollect: "SkySatCollect",
    RSGIS_PLANET_ITEM_SkySatVideo: "SkySatVideo",
}

_ASSET_TYPE_MAP: Dict[int, str] = {
    RSGIS_PLANET_ASSET_basic_analytic_4b: "basic_analytic_4b",
    RSGIS_PLANET_ASSET_basic_analytic_4b_rpc: "basic_analytic_4b_rpc",
    RSGIS_PLANET_ASSET_basic_analytic_4b_xml: "basic_analytic_4b_xml",
    RSGIS_PLANET_ASSET_basic_analytic_8b: "basic_analytic_8b",
    RSGIS_PLANET_ASSET_basic_analytic_8b_xml: "basic_analytic_8b_xml",
    RSGIS_PLANET_ASSET_basic_udm2: "basic_udm2",
    RSGIS_PLANET_ASSET_ortho_analytic_4b: "ortho_analytic_4b",
    RSGIS_PLANET_ASSET_ortho_analytic_4b_sr: "ortho_analytic_4b_sr",
    RSGIS_PLANET_ASSET_ortho_analytic_4b_xml: "ortho_analytic_4b_xml",
    RSGIS_PLANET_ASSET_ortho_analytic_8b: "ortho_analytic_8b",
    RSGIS_PLANET_ASSET_ortho_analytic_8b_sr: "ortho_analytic_8b_sr",
    RSGIS_PLANET_ASSET_ortho_analytic_8b_xml: "ortho_analytic_8b_xml",
    RSGIS_PLANET_ASSET_ortho_udm2: "ortho_udm2",
    RSGIS_PLANET_ASSET_ortho_visual: "ortho_visual",
}

_BUNDLE_TYPE_MAP: Dict[int, str] = {
    RSGIS_PLANET_BUNDLE_analytic_udm2: "analytic_udm2",
    RSGIS_PLANET_BUNDLE_analytic_3b_udm2: "analytic_3b_udm2",
    RSGIS_PLANET_BUNDLE_analytic_8b_udm2: "analytic_8b_udm2",
    RSGIS_PLANET_BUNDLE_visual: "visual",
    RSGIS_PLANET_BUNDLE_basic_analytic_udm2: "basic_analytic_udm2",
    RSGIS_PLANET_BUNDLE_basic_analytic_8b_udm2: "basic_analytic_8b_udm2",
    RSGIS_PLANET_BUNDLE_analytic_sr_udm2: "analytic_sr_udm2",
    RSGIS_PLANET_BUNDLE_analytic_8b_sr_udm2: "analytic_8b_sr_udm2",
}

__all__ = [
    "PLANET_AVAIL",
    "RSGIS_PLANET_ITEM_PSScene",
    "RSGIS_PLANET_ITEM_REOrthoTile",
    "RSGIS_PLANET_ITEM_REScene",
    "RSGIS_PLANET_ITEM_SkySatScene",
    "RSGIS_PLANET_ITEM_SkySatCollect",
    "RSGIS_PLANET_ITEM_SkySatVideo",
    "RSGIS_PLANET_ASSET_basic_analytic_4b",
    "RSGIS_PLANET_ASSET_basic_analytic_4b_rpc",
    "RSGIS_PLANET_ASSET_basic_analytic_4b_xml",
    "RSGIS_PLANET_ASSET_basic_analytic_8b",
    "RSGIS_PLANET_ASSET_basic_analytic_8b_xml",
    "RSGIS_PLANET_ASSET_basic_udm2",
    "RSGIS_PLANET_ASSET_ortho_analytic_4b",
    "RSGIS_PLANET_ASSET_ortho_analytic_4b_sr",
    "RSGIS_PLANET_ASSET_ortho_analytic_4b_xml",
    "RSGIS_PLANET_ASSET_ortho_analytic_8b",
    "RSGIS_PLANET_ASSET_ortho_analytic_8b_sr",
    "RSGIS_PLANET_ASSET_ortho_analytic_8b_xml",
    "RSGIS_PLANET_ASSET_ortho_udm2",
    "RSGIS_PLANET_ASSET_ortho_visual",
    "RSGIS_PLANET_BUNDLE_analytic_udm2",
    "RSGIS_PLANET_BUNDLE_analytic_3b_udm2",
    "RSGIS_PLANET_BUNDLE_analytic_8b_udm2",
    "RSGIS_PLANET_BUNDLE_visual",
    "RSGIS_PLANET_BUNDLE_basic_analytic_udm2",
    "RSGIS_PLANET_BUNDLE_basic_analytic_8b_udm2",
    "RSGIS_PLANET_BUNDLE_analytic_sr_udm2",
    "RSGIS_PLANET_BUNDLE_analytic_8b_sr_udm2",
    "get_item_type_str",
    "get_asset_type_str",
    "get_bundle_type_str",
    "planet_auth",
    "run_search_planet_items",
    "run_download_and_validate_item",
    "run_create_planet_order",
    "run_download_planet_order",
    "run_get_planet_orders",
    "run_cancel_planet_orders",
]


def _require_planet() -> None:
    if not PLANET_AVAIL:
        raise rsgispy.RSGISPyException(
            "The Planet API (pip install planet) needs to be available."
        )


def get_item_type_str(item_type: int) -> str:
    """
    Get the string representation of a given Planet item type (sensor).

    :param item_type: ``RSGIS_PLANET_ITEM_*`` value.
    :return: String representation of the item type.
    """
    if item_type not in _ITEM_TYPE_MAP:
        raise rsgispy.RSGISPyException("item_type was not recognised")
    return _ITEM_TYPE_MAP[item_type]


def get_asset_type_str(asset_type: int) -> str:
    """
    Get the string representation of a given Planet asset type.

    :param asset_type: ``RSGIS_PLANET_ASSET_*`` value.
    :return: String representation of the asset type.
    """
    if asset_type not in _ASSET_TYPE_MAP:
        raise rsgispy.RSGISPyException("asset_type was not recognised")
    return _ASSET_TYPE_MAP[asset_type]


def get_bundle_type_str(bundle_type: int) -> str:
    """
    Get the string representation of a given Planet order bundle type.

    :param bundle_type: ``RSGIS_PLANET_BUNDLE_*`` value.
    :return: String representation of the bundle type.
    """
    if bundle_type not in _BUNDLE_TYPE_MAP:
        raise rsgispy.RSGISPyException("bundle_type was not recognised")
    return _BUNDLE_TYPE_MAP[bundle_type]


def planet_auth(
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Authenticate with the Planet API.

    :param username: Planet username. Reads ``RSGIS_PLANET_USER`` if ``None``.
    :param password: Planet password. Reads ``RSGIS_PLANET_PASS`` if ``None``.
    :param api_key: Planet API key. Reads ``RSGIS_PLANET_API_KEY`` or
                    ``PL_API_KEY`` if ``None``.
    :return: A ``planet.Auth`` object.
    """
    _require_planet()

    if username is None:
        username = rsgispy.tools.system.get_environment_variable("RSGIS_PLANET_USER")

    if password is None:
        password = rsgispy.tools.system.get_environment_variable("RSGIS_PLANET_PASS")

    if api_key is None:
        api_key = rsgispy.tools.system.get_environment_variable("RSGIS_PLANET_API_KEY")
    if api_key is None:
        api_key = rsgispy.tools.system.get_environment_variable("PL_API_KEY")

    if api_key is not None:
        return planet.Auth.from_key(api_key)
    return planet.Auth.from_login(username, password)


# pylint: disable=too-many-arguments,too-many-positional-arguments
async def _search_planet_items(
    auth_obj: Any,
    item_type: int,
    bbox: Union[Tuple[float, float, float, float], List[float]],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_cover: Optional[float] = None,
    sun_elevation_min: Optional[float] = None,
    sun_elevation_max: Optional[float] = None,
    view_angle_min: Optional[float] = None,
    view_angle_max: Optional[float] = None,
    max_n_rslts: int = 25,
) -> List[Dict]:
    _require_planet()

    item_type_str = get_item_type_str(item_type)

    data_filters = [planet.data_filter.permission_filter()]
    bbox_geo_dict = rsgispy.tools.geometry.get_bbox_geojson_poly(bbox)
    data_filters.append(planet.data_filter.geometry_filter(bbox_geo_dict))

    if start_date is not None or end_date is not None:
        data_filters.append(
            planet.data_filter.date_range_filter(
                "acquired", gte=start_date, lte=end_date
            )
        )

    if cloud_cover is not None:
        data_filters.append(
            planet.data_filter.range_filter("cloud_percent", lte=cloud_cover)
        )

    if sun_elevation_min is not None or sun_elevation_max is not None:
        data_filters.append(
            planet.data_filter.range_filter(
                "sun_elevation", gte=sun_elevation_min, lte=sun_elevation_max
            )
        )

    if view_angle_min is not None or view_angle_max is not None:
        data_filters.append(
            planet.data_filter.range_filter(
                "view_angle", gte=view_angle_min, lte=view_angle_max
            )
        )

    planet_data_filter = planet.data_filter.and_filter(data_filters)

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_data_cl = planet_sess.client("data")
        items = [
            item
            async for item in planet_data_cl.search(
                item_types=[item_type_str],
                search_filter=planet_data_filter,
                sort="acquired asc",
                limit=max_n_rslts,
            )
        ]

    return items


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_search_planet_items(
    auth_obj: Any,
    item_type: int,
    bbox: Union[Tuple[float, float, float, float], List[float]],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_cover: Optional[float] = None,
    sun_elevation_min: Optional[float] = None,
    sun_elevation_max: Optional[float] = None,
    view_angle_min: Optional[float] = None,
    view_angle_max: Optional[float] = None,
    max_n_rslts: int = 25,
) -> List[Dict]:
    """
    Search the Planet Data API for matching scenes/items.

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :param item_type: Item type constant (``RSGIS_PLANET_ITEM_*``).
    :param bbox: Bounding box ``(x_min, x_max, y_min, y_max)`` in EPSG:4326.
    :param start_date: Optional acquisition start datetime.
    :param end_date: Optional acquisition end datetime.
    :param cloud_cover: Optional maximum cloud percentage (0-100).
    :param sun_elevation_min: Optional minimum solar elevation (degrees).
    :param sun_elevation_max: Optional maximum solar elevation (degrees).
    :param view_angle_min: Optional minimum view angle (degrees).
    :param view_angle_max: Optional maximum view angle (degrees).
    :param max_n_rslts: Maximum number of results to return (default: 25).
    :return: List of scene item dictionaries.
    """
    return asyncio.run(
        _search_planet_items(
            auth_obj,
            item_type,
            bbox,
            start_date,
            end_date,
            cloud_cover,
            sun_elevation_min,
            sun_elevation_max,
            view_angle_min,
            view_angle_max,
            max_n_rslts,
        )
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
async def _download_and_validate_item(
    auth_obj: Any,
    item_type: int,
    item_id: str,
    asset_type: int,
    out_file_path: str,
    overwrite: bool = False,
) -> None:
    _require_planet()

    item_type_str = get_item_type_str(item_type)
    asset_type_str = get_asset_type_str(asset_type)

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_cl = planet_sess.client("data")

        with planet.reporting.StateBar(state="Get Asset Info") as status_bar:
            planet_asset = await planet_cl.get_asset(
                item_type_str, item_id, asset_type_str
            )
            status_bar.update(state="Activate")

            await planet_cl.activate_asset(planet_asset)
            status_bar.update(state="Wait for Activation")

            asset = await planet_cl.wait_asset(
                planet_asset, callback=status_bar.update_state
            )

        path = await planet_cl.download_asset(
            asset, directory=out_file_path, overwrite=overwrite, progress_bar=True
        )
        planet_cl.validate_checksum(asset, path)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_download_and_validate_item(
    auth_obj: Any,
    item_type: int,
    item_id: str,
    asset_type: int,
    out_file_path: str,
    overwrite: bool = False,
) -> None:
    """
    Activate, download, and checksum-validate a single Planet item asset.

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :param item_type: Item type constant (``RSGIS_PLANET_ITEM_*``).
    :param item_id: Unique item identifier.
    :param asset_type: Asset type constant (``RSGIS_PLANET_ASSET_*``).
    :param out_file_path: Output directory where the asset is downloaded.
    :param overwrite: Overwrite existing files if ``True``.
    """
    asyncio.run(
        _download_and_validate_item(
            auth_obj,
            item_type,
            item_id,
            asset_type,
            out_file_path,
            overwrite,
        )
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
async def _create_planet_order(
    auth_obj: Any,
    order_name: str,
    items: List[Dict],
    item_type: int,
    bundle_type: int,
    email_notification: bool = True,
) -> Dict:
    _require_planet()

    item_type_str = get_item_type_str(item_type)
    image_ids = [item["id"] for item in items]
    bundle_type_str = get_bundle_type_str(bundle_type)

    products_lst = [
        planet.order_request.product(
            item_ids=image_ids,
            product_bundle=bundle_type_str,
            item_type=item_type_str,
        )
    ]

    planet_request = planet.order_request.build_request(
        name=order_name,
        products=products_lst,
        notifications=planet.order_request.notifications(email=email_notification),
    )

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_cl = planet_sess.client("orders")
        planet_order = await planet_cl.create_order(planet_request)

    return planet_order


# pylint: disable=too-many-arguments,too-many-positional-arguments
def run_create_planet_order(
    auth_obj: Any,
    order_name: str,
    items: List[Dict],
    item_type: int,
    bundle_type: int,
    email_notification: bool = True,
) -> Dict:
    """
    Create a Planet Orders API order for a list of searched items.

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :param order_name: Name for the new order.
    :param items: List of item dictionaries returned from search.
    :param item_type: Item type constant (``RSGIS_PLANET_ITEM_*``).
    :param bundle_type: Bundle type constant (``RSGIS_PLANET_BUNDLE_*``).
    :param email_notification: Send email when ready for download (default: ``True``).
    :return: Dictionary describing the created order.
    """
    return asyncio.run(
        _create_planet_order(
            auth_obj, order_name, items, item_type, bundle_type, email_notification
        )
    )


async def _download_planet_order(
    auth_obj: Any,
    order_id: str,
    out_file_path: str,
    overwrite: bool = False,
) -> List:
    _require_planet()

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_cl = planet_sess.client("orders")
        planet_order = await planet_cl.get_order(order_id)

        if planet_order["state"] == "success":
            return await planet_cl.download_order(
                order_id,
                directory=out_file_path,
                overwrite=overwrite,
                progress_bar=True,
            )
        raise rsgispy.RSGISPyException(
            "There is an error with the order - likely that is not yet "
            "available for downloading."
        )


def run_download_planet_order(
    auth_obj: Any,
    order_id: str,
    out_file_path: str,
    overwrite: bool = False,
) -> List:
    """
    Download a completed Planet order to ``out_file_path``.

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :param order_id: Order ID (not name) to download.
    :param out_file_path: Local directory for downloaded files.
    :param overwrite: Overwrite existing files if ``True``.
    :return: List of downloaded file paths.
    """
    return asyncio.run(
        _download_planet_order(auth_obj, order_id, out_file_path, overwrite)
    )


async def _get_planet_orders(auth_obj: Any) -> List[Dict]:
    _require_planet()

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_cl = planet_sess.client("orders")
        orders_lst = planet_cl.list_orders()

    return orders_lst


def run_get_planet_orders(auth_obj: Any) -> List[Dict]:
    """
    List all current Planet orders for the authenticated account.

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :return: List of order dictionaries.
    """
    return asyncio.run(_get_planet_orders(auth_obj))


async def _cancel_planet_orders(
    auth_obj: Any,
    order_ids: Optional[List[str]] = None,
) -> Dict:
    _require_planet()

    async with planet.Session(auth=auth_obj) as planet_sess:
        planet_cl = planet_sess.client("orders")
        orders_info = planet_cl.cancel_orders(order_ids=order_ids)

    return orders_info


def run_cancel_planet_orders(
    auth_obj: Any,
    order_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Cancel Planet orders by ID (or all active orders when ``order_ids`` is ``None``).

    :param auth_obj: ``planet.Auth`` object from :func:`planet_auth`.
    :param order_ids: Optional list of order IDs to cancel.
    :return: Dictionary describing cancelled orders.
    """
    return asyncio.run(_cancel_planet_orders(auth_obj, order_ids))
