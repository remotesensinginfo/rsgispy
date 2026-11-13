"""
Tools for accessing (searching and downloading) datasets from the
Copernicus Data Space Ecosystem OData API.

Sensors:
--------
    * RSGIS_ODATA_SEN1 = 1
    * RSGIS_ODATA_SEN1_RTC = 11
    * RSGIS_ODATA_SEN2 = 2
    * RSGIS_ODATA_SEN3 = 3
    * RSGIS_ODATA_SEN5P = 5
    * RSGIS_ODATA_SEN6 = 6
    * RSGIS_ODATA_COP_DEM = 100
    * RSGIS_ODATA_SMOS = 101

Orbit Direction:
-----------------
    * RSGIS_ODATA_ORBIT_DIR_ASC = 1
    * RSGIS_ODATA_ORBIT_DIR_DESC = 2

Order By Direction:
---------------------
    * RSGIS_ODATA_ORDERBY_ASC = 1
    * RSGIS_ODATA_ORDERBY_DESC = 2

Product Types:
--------------
    * RSGIS_ODATA_PROD_TYPE_S1_SLC = 101
    * RSGIS_ODATA_PROD_TYPE_S1_GRD = 102
    * RSGIS_ODATA_PROD_TYPE_S1_OCN = 103
    * RSGIS_ODATA_PROD_TYPE_S2_MSI_1C = 201
    * RSGIS_ODATA_PROD_TYPE_S2_MSI_2A = 202
    * RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA = 301
    * RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_A = 302
    * RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_BS = 303
    * RSGIS_ODATA_PROD_TYPE_S3_SR_2_LAN = 304
    * RSGIS_ODATA_PROD_TYPE_S3_OL_1_EFR = 305
    * RSGIS_ODATA_PROD_TYPE_S3_OL_1_ERR = 306
    * RSGIS_ODATA_PROD_TYPE_S3_OL_2_LFR = 307
    * RSGIS_ODATA_PROD_TYPE_S3_OL_2_LRR = 308
    * RSGIS_ODATA_PROD_TYPE_S3_SL_1_RBT = 309
    * RSGIS_ODATA_PROD_TYPE_S3_SL_2_LST = 310
    * RSGIS_ODATA_PROD_TYPE_S3_SY_2_SYN = 311
    * RSGIS_ODATA_PROD_TYPE_S3_SY_2_V10 = 312
    * RSGIS_ODATA_PROD_TYPE_S3_SY_2_VG1 = 313
    * RSGIS_ODATA_PROD_TYPE_S3_SY_2_VGP = 314
    * RSGIS_ODATA_PROD_TYPE_S3_SY_2_AOD = 315
    * RSGIS_ODATA_PROD_TYPE_S3_SL_2_FRP = 316
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_SIR = 501
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_UVN = 502
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD1 = 503
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD2 = 504
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD3 = 505
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD4 = 506
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD5 = 507
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD6 = 508
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD7 = 509
    * RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD8 = 510
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_AI = 511
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_LH = 512
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_CH4 = 513
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_CLOUD = 514
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_CO = 515
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_HCHO = 516
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_NO2 = 517
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD3 = 518
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD6 = 519
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD7 = 520
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_O3_TCL = 521
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_O3 = 522
    * RSGIS_ODATA_PROD_TYPE_S5P_L2_SO2 = 523
    * RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMFCT = 524
    * RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMANA = 525
"""

import datetime
import os
from typing import Dict, List, Optional, Tuple, Union

import requests
import tqdm

import rsgispy
import rsgispy.tools.filetools
import rsgispy.tools.geometrytools
import rsgispy.tools.httptools
import rsgispy.tools.utils

RSGIS_ODATA_SEN1 = 1
RSGIS_ODATA_SEN1_RTC = 11
RSGIS_ODATA_SEN2 = 2
RSGIS_ODATA_SEN3 = 3
RSGIS_ODATA_SEN5P = 5
RSGIS_ODATA_SEN6 = 6
RSGIS_ODATA_COP_DEM = 100
RSGIS_ODATA_SMOS = 101

RSGIS_ODATA_ORBIT_DIR_ASC = 1
RSGIS_ODATA_ORBIT_DIR_DESC = 2

RSGIS_ODATA_ORDERBY_ASC = 1
RSGIS_ODATA_ORDERBY_DESC = 2

RSGIS_ODATA_PROD_TYPE_S1_SLC = 101
RSGIS_ODATA_PROD_TYPE_S1_GRD = 102
RSGIS_ODATA_PROD_TYPE_S1_OCN = 103

RSGIS_ODATA_PROD_TYPE_S2_MSI_1C = 201
RSGIS_ODATA_PROD_TYPE_S2_MSI_2A = 202

RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA = 301
RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_A = 302
RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_BS = 303
RSGIS_ODATA_PROD_TYPE_S3_SR_2_LAN = 304
RSGIS_ODATA_PROD_TYPE_S3_OL_1_EFR = 305
RSGIS_ODATA_PROD_TYPE_S3_OL_1_ERR = 306
RSGIS_ODATA_PROD_TYPE_S3_OL_2_LFR = 307
RSGIS_ODATA_PROD_TYPE_S3_OL_2_LRR = 308
RSGIS_ODATA_PROD_TYPE_S3_SL_1_RBT = 309
RSGIS_ODATA_PROD_TYPE_S3_SL_2_LST = 310
RSGIS_ODATA_PROD_TYPE_S3_SY_2_SYN = 311
RSGIS_ODATA_PROD_TYPE_S3_SY_2_V10 = 312
RSGIS_ODATA_PROD_TYPE_S3_SY_2_VG1 = 313
RSGIS_ODATA_PROD_TYPE_S3_SY_2_VGP = 314
RSGIS_ODATA_PROD_TYPE_S3_SY_2_AOD = 315
RSGIS_ODATA_PROD_TYPE_S3_SL_2_FRP = 316

RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_SIR = 501
RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_UVN = 502
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD1 = 503
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD2 = 504
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD3 = 505
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD4 = 506
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD5 = 507
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD6 = 508
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD7 = 509
RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD8 = 510
RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_AI = 511
RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_LH = 512
RSGIS_ODATA_PROD_TYPE_S5P_L2_CH4 = 513
RSGIS_ODATA_PROD_TYPE_S5P_L2_CLOUD = 514
RSGIS_ODATA_PROD_TYPE_S5P_L2_CO = 515
RSGIS_ODATA_PROD_TYPE_S5P_L2_HCHO = 516
RSGIS_ODATA_PROD_TYPE_S5P_L2_NO2 = 517
RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD3 = 518
RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD6 = 519
RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD7 = 520
RSGIS_ODATA_PROD_TYPE_S5P_L2_O3_TCL = 521
RSGIS_ODATA_PROD_TYPE_S5P_L2_O3 = 522
RSGIS_ODATA_PROD_TYPE_S5P_L2_SO2 = 523
RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMFCT = 524
RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMANA = 525

RSGIS_ODATA_PRODUCTS_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)
RSGIS_ODATA_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

_ODATA_SENSOR_MAP: Dict[int, str] = {
    RSGIS_ODATA_SEN1: "SENTINEL-1",
    RSGIS_ODATA_SEN1_RTC: "SENTINEL-1-RTC",
    RSGIS_ODATA_SEN2: "SENTINEL-2",
    RSGIS_ODATA_SEN3: "SENTINEL-3",
    RSGIS_ODATA_SEN5P: "SENTINEL-5P",
    RSGIS_ODATA_SEN6: "SENTINEL-6",
    RSGIS_ODATA_COP_DEM: "COP-DEM",
    RSGIS_ODATA_SMOS: "SMOS",
}

_ODATA_ORBIT_DIR_MAP: Dict[int, str] = {
    RSGIS_ODATA_ORBIT_DIR_ASC: "ASCENDING",
    RSGIS_ODATA_ORBIT_DIR_DESC: "DESCENDING",
}

_ODATA_ORDERBY_MAP: Dict[int, str] = {
    RSGIS_ODATA_ORDERBY_ASC: "asc",
    RSGIS_ODATA_ORDERBY_DESC: "desc",
}

_ODATA_PROD_TYPE_MAP: Dict[int, str] = {
    RSGIS_ODATA_PROD_TYPE_S1_SLC: "SLC",
    RSGIS_ODATA_PROD_TYPE_S1_GRD: "GRD",
    RSGIS_ODATA_PROD_TYPE_S1_OCN: "OCN",
    RSGIS_ODATA_PROD_TYPE_S2_MSI_1C: "S2MSI1C",
    RSGIS_ODATA_PROD_TYPE_S2_MSI_2A: "S2MSI2A",
    RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA: "SR_1_SRA___",
    RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_A: "SR_1_SRA_A",
    RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_BS: "SR_1_SRA_BS",
    RSGIS_ODATA_PROD_TYPE_S3_SR_2_LAN: "SR_2_LAN___",
    RSGIS_ODATA_PROD_TYPE_S3_OL_1_EFR: "OL_1_EFR___",
    RSGIS_ODATA_PROD_TYPE_S3_OL_1_ERR: "OL_1_ERR___",
    RSGIS_ODATA_PROD_TYPE_S3_OL_2_LFR: "OL_2_LFR___",
    RSGIS_ODATA_PROD_TYPE_S3_OL_2_LRR: "OL_2_LRR___",
    RSGIS_ODATA_PROD_TYPE_S3_SL_1_RBT: "SL_1_RBT___",
    RSGIS_ODATA_PROD_TYPE_S3_SL_2_LST: "SL_2_LST___",
    RSGIS_ODATA_PROD_TYPE_S3_SY_2_SYN: "SY_2_SYN___",
    RSGIS_ODATA_PROD_TYPE_S3_SY_2_V10: "SY_2_V10___",
    RSGIS_ODATA_PROD_TYPE_S3_SY_2_VG1: "SY_2_VG1___",
    RSGIS_ODATA_PROD_TYPE_S3_SY_2_VGP: "SY_2_VGP___",
    RSGIS_ODATA_PROD_TYPE_S3_SY_2_AOD: "SY_2_AOD__",
    RSGIS_ODATA_PROD_TYPE_S3_SL_2_FRP: "SL_2_FRP__",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_SIR: "L1B_IR_SIR",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_UVN: "L1B_IR_UVN",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD1: "L1B_RA_BD1",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD2: "L1B_RA_BD2",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD3: "L1B_RA_BD3",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD4: "L1B_RA_BD4",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD5: "L1B_RA_BD5",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD6: "L1B_RA_BD6",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD7: "L1B_RA_BD7",
    RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD8: "L1B_RA_BD8",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_AI: "L2__AER_AI",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_LH: "L2__AER_LH",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_CH4: "L2__CH4",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_CLOUD: "L2__CLOUD_",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_CO: "L2__CO____",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_HCHO: "L2__HCHO__",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_NO2: "L2__NO2___",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD3: "L2__NP_BD3",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD6: "L2__NP_BD6",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD7: "L2__NP_BD7",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_O3_TCL: "L2__O3_TCL",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_O3: "L2__O3____",
    RSGIS_ODATA_PROD_TYPE_S5P_L2_SO2: "L2__SO2___",
    RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMFCT: "AUX_CTMFCT",
    RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMANA: "AUX_CTMANA",
}

__all__ = [
    "RSGIS_ODATA_SEN1",
    "RSGIS_ODATA_SEN1_RTC",
    "RSGIS_ODATA_SEN2",
    "RSGIS_ODATA_SEN3",
    "RSGIS_ODATA_SEN5P",
    "RSGIS_ODATA_SEN6",
    "RSGIS_ODATA_COP_DEM",
    "RSGIS_ODATA_SMOS",
    "RSGIS_ODATA_ORBIT_DIR_ASC",
    "RSGIS_ODATA_ORBIT_DIR_DESC",
    "RSGIS_ODATA_ORDERBY_ASC",
    "RSGIS_ODATA_ORDERBY_DESC",
    "RSGIS_ODATA_PROD_TYPE_S1_SLC",
    "RSGIS_ODATA_PROD_TYPE_S1_GRD",
    "RSGIS_ODATA_PROD_TYPE_S1_OCN",
    "RSGIS_ODATA_PROD_TYPE_S2_MSI_1C",
    "RSGIS_ODATA_PROD_TYPE_S2_MSI_2A",
    "RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA",
    "RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_A",
    "RSGIS_ODATA_PROD_TYPE_S3_SR_1_SRA_BS",
    "RSGIS_ODATA_PROD_TYPE_S3_SR_2_LAN",
    "RSGIS_ODATA_PROD_TYPE_S3_OL_1_EFR",
    "RSGIS_ODATA_PROD_TYPE_S3_OL_1_ERR",
    "RSGIS_ODATA_PROD_TYPE_S3_OL_2_LFR",
    "RSGIS_ODATA_PROD_TYPE_S3_OL_2_LRR",
    "RSGIS_ODATA_PROD_TYPE_S3_SL_1_RBT",
    "RSGIS_ODATA_PROD_TYPE_S3_SL_2_LST",
    "RSGIS_ODATA_PROD_TYPE_S3_SY_2_SYN",
    "RSGIS_ODATA_PROD_TYPE_S3_SY_2_V10",
    "RSGIS_ODATA_PROD_TYPE_S3_SY_2_VG1",
    "RSGIS_ODATA_PROD_TYPE_S3_SY_2_VGP",
    "RSGIS_ODATA_PROD_TYPE_S3_SY_2_AOD",
    "RSGIS_ODATA_PROD_TYPE_S3_SL_2_FRP",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_SIR",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_IR_UVN",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD1",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD2",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD3",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD4",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD5",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD6",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD7",
    "RSGIS_ODATA_PROD_TYPE_S5P_L1B_RA_BD8",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_AI",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_AER_LH",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_CH4",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_CLOUD",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_CO",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_HCHO",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_NO2",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD3",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD6",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_NP_BD7",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_O3_TCL",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_O3",
    "RSGIS_ODATA_PROD_TYPE_S5P_L2_SO2",
    "RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMFCT",
    "RSGIS_ODATA_PROD_TYPE_S5P_AUX_CTMANA",
    "RSGIS_ODATA_PRODUCTS_URL",
    "RSGIS_ODATA_TOKEN_URL",
    "get_sensor_collection_name",
    "get_access_token",
    "query_scn_lst",
    "query_scn",
    "download_scn",
    "download_scns",
]


def get_sensor_collection_name(sensor: int) -> str:
    """
    Return the OData collection name for the ``RSGIS_ODATA_*`` sensor constant.

    :param sensor: ``RSGIS_ODATA_*`` sensor identifier.
    :return: OData string name for the sensor collection.
    """
    if sensor not in _ODATA_SENSOR_MAP:
        raise rsgispy.RSGISPyException("Did not recognise sensor type")
    return _ODATA_SENSOR_MAP[sensor]


def get_access_token(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[str]:
    """
    Obtain an OAuth2 access token from the Copernicus Data Space Ecosystem.

    :param username: CDSE username. Reads ``RSGIS_COP_USER`` if ``None``.
    :param password: CDSE password. Reads ``RSGIS_COP_PASS`` if ``None``.
    :return: Access token string, or ``None`` if validation failed.
    """
    if username is None:
        username = rsgispy.tools.utils.get_environment_variable("RSGIS_COP_USER")

    if password is None:
        password = rsgispy.tools.utils.get_environment_variable("RSGIS_COP_PASS")

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    resp_obj = requests.post(RSGIS_ODATA_TOKEN_URL, data=data, timeout=60)
    try:
        if rsgispy.tools.httptools.check_http_response(resp_obj, RSGIS_ODATA_TOKEN_URL):
            resp_dict = resp_obj.json()
            if "access_token" not in resp_dict:
                raise rsgispy.RSGISPyException(
                    f"Failed to get access token. Returned Data: {resp_dict}"
                )
            return resp_dict["access_token"]
        return None
    except rsgispy.RSGISPyException:
        raise
    except Exception as exc:
        raise rsgispy.RSGISPyException(
            "Failed to get access token - check username and password"
        ) from exc


# pylint: disable=too-many-arguments,too-many-positional-arguments
def query_scn_lst(
    sensor: int,
    bbox: Union[Tuple[float, float, float, float], List[float]],
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    cloud_cover: Optional[float] = None,
    orbit_dir: Optional[int] = None,
    product_type: Optional[int] = None,
    order_by: int = RSGIS_ODATA_ORDERBY_ASC,
    max_n_rslts: int = 25,
    start_n: Optional[int] = None,
) -> List[Dict]:
    """
    Query the Copernicus Data Space Ecosystem OData API for matching scenes.

    :param sensor: Sensor constant (e.g., ``RSGIS_ODATA_SEN2``).
    :param bbox: Bounding box ``(x_min, x_max, y_min, y_max)`` in EPSG:4326.
    :param start_date: Optional acquisition start datetime.
    :param end_date: Optional acquisition end datetime.
    :param cloud_cover: Optional maximum cloud cover percentage (0-100).
    :param orbit_dir: Optional orbit direction (``RSGIS_ODATA_ORBIT_DIR_*``).
    :param product_type: Optional product type (``RSGIS_ODATA_PROD_TYPE_*``).
    :param order_by: Date ordering (``RSGIS_ODATA_ORDERBY_*``).
    :param max_n_rslts: Maximum number of scenes to return (default: 25).
    :param start_n: Optional offset skipping the first ``n`` scenes.
    :return: List of scene metadata dictionaries.
    """
    sensor_name = get_sensor_collection_name(sensor)
    query_url = f"{RSGIS_ODATA_PRODUCTS_URL}?$filter=Collection/Name eq '{sensor_name}'"

    bbox_wkt_str = rsgispy.tools.geometrytools.get_bbox_wkt_poly(bbox)
    bbox_url_str = f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt_str}')"
    query_url = f"{query_url} and {bbox_url_str}"

    if (start_date is None) != (end_date is None):
        raise rsgispy.RSGISPyException(
            "Both start_date and end_date must be provided together"
        )
    if start_date is not None and end_date is not None:
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        dates_url_str = (
            f"ContentDate/Start gt {start_date_str} "
            f"and ContentDate/End lt {end_date_str}"
        )
        query_url = f"{query_url} and {dates_url_str}"

    if cloud_cover is not None:
        if cloud_cover < 0 or cloud_cover > 100:
            raise rsgispy.RSGISPyException("Cloud cover must be between 0 and 100")
        cloud_cover_str = (
            "Attributes/OData.CSC.DoubleAttribute/any("
            "att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {cloud_cover})"
        )
        query_url = f"{query_url} and {cloud_cover_str}"

    if orbit_dir is not None:
        if orbit_dir not in _ODATA_ORBIT_DIR_MAP:
            raise rsgispy.RSGISPyException(
                "orbit_dir must be either Ascending or Descending"
            )
        orbit_dir_str = _ODATA_ORBIT_DIR_MAP[orbit_dir]
        orbit_dir_query_str = (
            "Attributes/OData.CSC.StringAttribute/any("
            "att:att/Name eq 'orbitDirection' and "
            f"att/OData.CSC.StringAttribute/Value eq '{orbit_dir_str}')"
        )
        query_url = f"{query_url} and {orbit_dir_query_str}"

    if product_type is not None:
        if product_type not in _ODATA_PROD_TYPE_MAP:
            raise rsgispy.RSGISPyException("product_type was not recognised")
        product_type_str = _ODATA_PROD_TYPE_MAP[product_type]
        product_type_query_str = (
            "Attributes/OData.CSC.StringAttribute/any("
            "att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{product_type_str}')"
        )
        query_url = f"{query_url} and {product_type_query_str}"

    if order_by not in _ODATA_ORDERBY_MAP:
        raise rsgispy.RSGISPyException("order_by must be either ascending or descending")
    query_url = (
        f"{query_url}&$orderby=ContentDate/Start {_ODATA_ORDERBY_MAP[order_by]}"
    )

    query_url = f"{query_url}&$top={max_n_rslts}"
    if start_n is not None and start_n > 0:
        query_url = f"{query_url}&skip={start_n}"

    query_url = f"{query_url}&$expand=Locations"

    resp_obj = requests.get(query_url, timeout=60)
    rsgispy.tools.httptools.check_http_response(resp_obj, query_url)
    resp_dict = resp_obj.json()
    if "value" in resp_dict:
        return resp_dict["value"]
    raise rsgispy.RSGISPyException("Error - response didn't have a 'value'.")


def query_scn(scn_name: str) -> Optional[Dict]:
    """
    Query for a single scene by product name.

    For example:
    ``S2B_MSIL2A_20240602T112119_N0510_R037_T30UVD_20240602T125034.SAFE``

    :param scn_name: Name of the scene to query.
    :return: Dictionary containing metadata for the scene, or ``None``.
    """
    query_url = (
        f"{RSGIS_ODATA_PRODUCTS_URL}?$filter=Name eq '{scn_name}'&$expand=Locations"
    )
    resp_obj = requests.get(query_url, timeout=60)

    if rsgispy.tools.httptools.check_http_response(resp_obj, query_url):
        resp_dict = resp_obj.json()
        if "value" not in resp_dict:
            raise rsgispy.RSGISPyException("Error - response didn't have a 'value'.")
        n_scns = len(resp_dict["value"])
        if n_scns == 1:
            return resp_dict["value"][0]
        raise rsgispy.RSGISPyException(
            f"There were {n_scns} scenes in the list. Was expecting one."
        )
    return None


def download_scn(access_token: str, scn_info: Dict, out_path: str) -> None:
    """
    Download a single Copernicus OData scene to ``out_path``.

    During download the file uses an ``.incomplete`` suffix and is verified against
    its MD5 checksum (when provided) before being renamed to ``<Name>.zip``.

    :param access_token: OAuth2 bearer token from :func:`get_access_token`.
    :param scn_info: Scene dictionary from :func:`query_scn` or :func:`query_scn_lst`.
    :param out_path: Output directory where the downloaded zip will be saved.
    """
    scn_name = scn_info["Name"]
    print(f"Downloading {scn_name}")

    out_file_path = os.path.join(out_path, f"{scn_name}.zip")
    if os.path.exists(out_file_path):
        print(f"File already exists: {out_file_path}")
        return

    scn_dwnld_info = None
    for scn_loc in scn_info["Locations"]:
        if scn_loc["FormatType"] == "Extracted":
            scn_dwnld_info = scn_loc
            break

    if scn_dwnld_info is None:
        raise rsgispy.RSGISPyException("Error - no download location found.")

    scn_url = scn_dwnld_info["DownloadLink"]
    scn_size = scn_dwnld_info["ContentLength"]
    scn_checksum = None
    for checksum_info in scn_dwnld_info.get("Checksum", []):
        if checksum_info["Algorithm"] == "MD5":
            scn_checksum = checksum_info["Value"]

    ses = requests.Session()
    ses.headers.update(
        {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": f"rsgispy/{rsgispy.get_rsgispy_version()}",
        }
    )

    if "//catalogue." in scn_url:
        scn_url = scn_url.replace("//catalogue.", "//download.")

    resp_obj = ses.get(
        scn_url, stream=True, verify=True, allow_redirects=True, timeout=120
    )

    tmp_file_path = os.path.join(out_path, f"{scn_name}.incomplete")
    chunk_size = 8192
    n_chunks = int(scn_size / chunk_size) + 1

    if rsgispy.tools.httptools.check_http_response(resp_obj, scn_url):
        with open(tmp_file_path, "wb") as file:
            for chunk in tqdm.tqdm(
                resp_obj.iter_content(chunk_size=chunk_size), total=n_chunks
            ):
                if chunk:
                    file.write(chunk)

    if os.path.exists(tmp_file_path):
        if scn_checksum is not None:
            md5_checksum = rsgispy.tools.filetools.create_md5_hash(tmp_file_path)
            if md5_checksum != scn_checksum:
                raise rsgispy.RSGISPyException("Error - checksum mismatch.")
        os.rename(tmp_file_path, out_file_path)
        print(f"Download Complete: {scn_name}")
    else:
        raise rsgispy.RSGISPyException("Failed to download file - unknown error")


def download_scns(
    access_token: str,
    scns_info: List[Dict],
    out_path: str,
    no_except: bool = True,
) -> None:
    """
    Download a list of scenes using :func:`download_scn`.

    :param access_token: OAuth2 bearer token from :func:`get_access_token`.
    :param scns_info: List of scene dictionaries from :func:`query_scn_lst`.
    :param out_path: Output directory where scenes will be saved.
    :param no_except: If ``True`` (default), print exceptions and continue.
    """
    for scn_info in scns_info:
        try:
            download_scn(access_token, scn_info, out_path)
        except rsgispy.RSGISPyException as exc:
            if no_except:
                print(exc)
            else:
                raise
