"""
The tools.dates module provides calendar, season, and datetime sequence
utilities.
"""

import calendar
import datetime
from typing import List, Tuple

import rsgispy

__all__ = [
    "get_days_since",
    "get_days_since_date",
    "find_month_end_date",
    "create_year_month_start_end_lst",
    "create_year_month_n_months_lst",
    "is_summer_winter",
    "get_datetime_increment_lst",
]


def get_days_since(year: int, day_of_year: int, base_date: datetime.date) -> int:
    """
    Calculate the number of days from ``base_date`` to a given ``year`` and
    1-based ``day_of_year``.

    :param year: Target year (e.g., 2020).
    :param day_of_year: Day within the year (1..366).
    :param base_date: Reference ``datetime.date``.
    :return: Number of days elapsed since ``base_date``.
    """
    if year < base_date.year:
        raise rsgispy.RSGISPyException(
            "The year specified is before the base date."
        )
    date_val = datetime.date(year=int(year), month=1, day=1) + datetime.timedelta(
        days=int(day_of_year - 1)
    )
    return (date_val - base_date).days


def get_days_since_date(
    year: int,
    month: int,
    day: int,
    base_date: datetime.date,
) -> int:
    """
    Calculate the number of days from ``base_date`` to ``year-month-day``.

    :param year: Target year (e.g., 2020).
    :param month: Target month (1..12).
    :param day: Target day of month (1..31).
    :param base_date: Reference ``datetime.date``.
    :return: Number of days elapsed since ``base_date``.
    """
    if year < base_date.year:
        raise rsgispy.RSGISPyException(
            "The year specified is before the base date."
        )
    date_val = datetime.date(year=int(year), month=int(month), day=int(day))
    return (date_val - base_date).days


def find_month_end_date(year: int, month: int) -> int:
    """
    Return the last day number of ``month`` in ``year``.

    :param year: Year integer (e.g., 2019).
    :param month: Month integer (1..12).
    :return: Number of the last day of the month.
    """
    _, n_days = calendar.monthrange(year, month)
    return int(n_days)


def create_year_month_start_end_lst(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, int]]:
    """
    Create an inclusive list of ``(year, month)`` tuples between start and end.

    :param start_year: Start year.
    :param start_month: Start month (1..12).
    :param end_year: End year.
    :param end_month: End month (1..12).
    :return: List of ``(year, month)`` tuples.
    """
    out_year_mnt_lst: List[Tuple[int, int]] = []
    for year in range(int(start_year), int(end_year) + 1):
        m_start = int(start_month) if year == start_year else 1
        m_end = int(end_month) if year == end_year else 12
        for month in range(m_start, m_end + 1):
            out_year_mnt_lst.append((year, month))
    return out_year_mnt_lst


def create_year_month_n_months_lst(
    start_year: int,
    start_month: int,
    n_months_fwd: int,
    n_months_bwd: int = 0,
) -> List[Tuple[int, int]]:
    """
    Create a list of ``(year, month)`` tuples spanning ``n_months_bwd`` months
    before and ``n_months_fwd`` months after ``(start_year, start_month)``.

    :param start_year: Reference start year (>= 0).
    :param start_month: Reference start month (1..12).
    :param n_months_fwd: Number of months forward from the start month.
    :param n_months_bwd: Number of months backward from the start month.
    :return: List of ``(year, month)`` tuples.
    """
    if start_year < 0:
        raise rsgispy.RSGISPyException("Year must be positive")
    if (start_month < 1) or (start_month > 12):
        raise rsgispy.RSGISPyException("Month must be between 1-12")

    base_idx = int(start_year) * 12 + (int(start_month) - 1)
    out_year_mnt_lst: List[Tuple[int, int]] = []
    for offset in range(-int(n_months_bwd), int(n_months_fwd) + 1):
        year_val, month_zero = divmod(base_idx + offset, 12)
        out_year_mnt_lst.append((year_val, month_zero + 1))
    return out_year_mnt_lst


def is_summer_winter(lat: float, date: datetime.date) -> int:
    """
    Determine whether ``date`` falls in summer (1) or winter (2) at latitude ``lat``.

    Southern Hemisphere Winter (``lat < 0``): May - October (months 5..10).
    Northern Hemisphere Summer (``lat >= 0``): April - September (months 4..9).

    :param lat: Latitude in decimal degrees.
    :param date: ``datetime.date`` (or ``datetime.datetime``) object.
    :return: ``1`` for summer, ``2`` for winter.
    """
    if lat < 0:
        if 4 < date.month < 11:
            return 2
        return 1
    if 3 < date.month < 10:
        return 1
    return 2


def get_datetime_increment_lst(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: datetime.timedelta,
) -> List[datetime.datetime]:
    """
    Return a list of datetimes starting after ``start_date`` in steps of
    ``interval`` until reaching or exceeding ``end_date``.

    :param start_date: Start datetime of the sequence.
    :param end_date: End datetime of the sequence.
    :param interval: Step ``datetime.timedelta``.
    :return: List of ``datetime.datetime`` objects.
    """
    date_lst: List[datetime.datetime] = []
    c_date = start_date
    while c_date < end_date:
        c_date = c_date + interval
        date_lst.append(c_date)
    return date_lst
