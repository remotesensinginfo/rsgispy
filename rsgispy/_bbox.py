"""
Typed bounding box representation for :mod:`rsgispy`.

:class:`BBox` stores coordinates in RSGIS order
``(west, east, south, north)`` / ``(xmin, xmax, ymin, ymax)`` while
subclassing :class:`tuple` so 4-tuple unpacking, indexing, slicing,
JSON serialisation, and ``pytest.approx`` comparisons remain seamlessly
compatible.  Use :meth:`BBox.as_geo` and :meth:`BBox.from_geo` to convert
explicitly to and from Shapely / Rasterio / GeoPandas
``(minx, miny, maxx, maxy)`` order.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rasterio.warp import transform_bounds
import shapely.geometry

__all__ = ["BBox"]


def _as_rsgis_coords(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Extract ``(west, east, south, north)`` floats from a ``BBox`` or 4-sequence."""
    if isinstance(bbox, BBox):
        return bbox.as_rsgis()
    if len(bbox) < 4:
        raise ValueError(
            f"Bounding box must have 4 elements (west, east, south, north), got {len(bbox)}."
        )
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


class BBox(tuple):
    """
    Immutable typed bounding box storing ``(west, east, south, north)`` with
    optional ``epsg`` metadata.

    Because :class:`BBox` subclasses :class:`tuple`, it supports standard
    4-tuple unpacking (``xmin, xmax, ymin, ymax = bbox``), indexing
    (``bbox[0..3]``), slicing (``bbox[:4]``), equality with 4-tuples, and
    native JSON serialisation, while providing explicit :meth:`as_rsgis` and
    :meth:`as_geo` conversions to prevent coordinate-order bugs.

    :param west: Minimum X coordinate (``xmin``).
    :param east: Maximum X coordinate (``xmax``).
    :param south: Minimum Y coordinate (``ymin``).
    :param north: Maximum Y coordinate (``ymax``).
    :param epsg: Optional EPSG integer CRS code (Default = ``None``).

    .. code:: python

        import rsgispy

        bbox = rsgispy.BBox(500000.0, 510000.0, 5200000.0, 5210000.0, epsg=32630)
        xmin, xmax, ymin, ymax = bbox            # RSGIS 4-tuple unpacking
        minx, miny, maxx, maxy = bbox.as_geo()   # Shapely / Rasterio order
    """

    def __new__(
        cls,
        west: float,
        east: float,
        south: float,
        north: float,
        epsg: Optional[int] = None,
    ) -> "BBox":
        obj = super().__new__(
            cls, (float(west), float(east), float(south), float(north))
        )
        object.__setattr__(obj, "_epsg", int(epsg) if epsg is not None else None)
        return obj

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BBox is immutable.")

    def __reduce__(self) -> Tuple[Any, Tuple[float, float, float, float, Optional[int]]]:
        """Preserve ``epsg`` across pickle and copy operations."""
        return (self.__class__, (self[0], self[1], self[2], self[3], self._epsg))

    def __repr__(self) -> str:
        if self._epsg is not None:
            return (
                f"BBox(west={self[0]!r}, east={self[1]!r}, "
                f"south={self[2]!r}, north={self[3]!r}, epsg={self._epsg!r})"
            )
        return (
            f"BBox(west={self[0]!r}, east={self[1]!r}, "
            f"south={self[2]!r}, north={self[3]!r})"
        )

    @property
    def west(self) -> float:
        """Minimum X coordinate (``xmin``)."""
        return self[0]

    @property
    def east(self) -> float:
        """Maximum X coordinate (``xmax``)."""
        return self[1]

    @property
    def south(self) -> float:
        """Minimum Y coordinate (``ymin``)."""
        return self[2]

    @property
    def north(self) -> float:
        """Maximum Y coordinate (``ymax``)."""
        return self[3]

    @property
    def xmin(self) -> float:
        """Alias for :attr:`west`."""
        return self[0]

    @property
    def xmax(self) -> float:
        """Alias for :attr:`east`."""
        return self[1]

    @property
    def ymin(self) -> float:
        """Alias for :attr:`south`."""
        return self[2]

    @property
    def ymax(self) -> float:
        """Alias for :attr:`north`."""
        return self[3]

    @property
    def epsg(self) -> Optional[int]:
        """Optional EPSG CRS integer code."""
        return self._epsg

    @property
    def width(self) -> float:
        """Horizontal span (``east - west``)."""
        return self[1] - self[0]

    @property
    def height(self) -> float:
        """Vertical span (``north - south``)."""
        return self[3] - self[2]

    @property
    def area(self) -> float:
        """Bounding box area ``abs((east - west) * (north - south))``."""
        return abs(self.width * self.height)

    @property
    def centre(self) -> Tuple[float, float]:
        """Centre point ``(cx, cy)`` of the bounding box."""
        return ((self[0] + self[1]) / 2.0, (self[2] + self[3]) / 2.0)

    def as_rsgis(self) -> Tuple[float, float, float, float]:
        """Return plain 4-tuple in RSGIS order ``(xmin, xmax, ymin, ymax)``."""
        return (self[0], self[1], self[2], self[3])

    def as_geo(self) -> Tuple[float, float, float, float]:
        """Return plain 4-tuple in Shapely/Rasterio order ``(minx, miny, maxx, maxy)``."""
        return (self[0], self[2], self[1], self[3])

    def to_polygon(self) -> shapely.geometry.Polygon:
        """Convert to a :class:`shapely.geometry.Polygon` rectangle."""
        return shapely.geometry.box(self[0], self[2], self[1], self[3])

    def with_epsg(self, epsg: Optional[int]) -> "BBox":
        """Return a copy of this bounding box with a new ``epsg`` code."""
        return BBox(self[0], self[1], self[2], self[3], epsg=epsg)

    def buffer(self, dist: float) -> "BBox":
        """Return a new :class:`BBox` expanded outward by ``dist`` on all sides."""
        d = float(dist)
        return BBox(
            self[0] - d,
            self[1] + d,
            self[2] - d,
            self[3] + d,
            epsg=self._epsg,
        )

    def reproject(
        self,
        out_epsg: int,
        in_epsg: Optional[int] = None,
    ) -> "BBox":
        """
        Reproject this bounding box to ``out_epsg``.

        :param out_epsg: Target EPSG integer code.
        :param in_epsg: Source EPSG integer code (defaults to :attr:`epsg`).
        :return: Reprojected :class:`BBox` with ``epsg=out_epsg``.
        """
        src_epsg = in_epsg if in_epsg is not None else self._epsg
        if src_epsg is None:
            raise ValueError(
                "Source EPSG code must be set on BBox or passed via in_epsg."
            )
        if int(src_epsg) == int(out_epsg):
            return self.with_epsg(int(out_epsg))
        min_x, min_y, max_x, max_y = transform_bounds(
            f"EPSG:{int(src_epsg)}",
            f"EPSG:{int(out_epsg)}",
            self[0],
            self[2],
            self[1],
            self[3],
        )
        return BBox(min_x, max_x, min_y, max_y, epsg=int(out_epsg))

    def intersects(self, other: Sequence[float]) -> bool:
        """Return ``True`` if this bounding box intersects ``other`` (RSGIS order)."""
        ow, oe, os_, on = _as_rsgis_coords(other)
        return not (
            self[1] < ow or self[0] > oe or self[3] < os_ or self[2] > on
        )

    def intersection(self, other: Sequence[float]) -> Optional["BBox"]:
        """Return the intersecting :class:`BBox`, or ``None`` if disjoint."""
        ow, oe, os_, on = _as_rsgis_coords(other)
        w = max(self[0], ow)
        e = min(self[1], oe)
        s = max(self[2], os_)
        n = min(self[3], on)
        if w > e or s > n:
            return None
        out_epsg = self._epsg
        if out_epsg is None and isinstance(other, BBox):
            out_epsg = other.epsg
        return BBox(w, e, s, n, epsg=out_epsg)

    def union(self, other: Sequence[float]) -> "BBox":
        """Return the minimal enclosing :class:`BBox` covering ``self`` and ``other``."""
        ow, oe, os_, on = _as_rsgis_coords(other)
        out_epsg = self._epsg
        if out_epsg is None and isinstance(other, BBox):
            out_epsg = other.epsg
        return BBox(
            min(self[0], ow),
            max(self[1], oe),
            min(self[2], os_),
            max(self[3], on),
            epsg=out_epsg,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Return ``True`` if point ``(x, y)`` lies within or on the box edges."""
        return (self[0] <= x <= self[1]) and (self[2] <= y <= self[3])

    def equals(
        self,
        other: Sequence[float],
        tol: float = 0.0,
        check_epsg: bool = False,
    ) -> bool:
        """
        Compare bounding boxes with optional tolerance and CRS check.

        :param other: Another :class:`BBox` or RSGIS 4-sequence.
        :param tol: Absolute coordinate tolerance (Default = ``0.0``).
        :param check_epsg: If ``True``, require matching ``epsg`` metadata.
        :return: ``True`` if equal within ``tol``.
        """
        ow, oe, os_, on = _as_rsgis_coords(other)
        coords_ok = all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=tol)
            for a, b in zip(self.as_rsgis(), (ow, oe, os_, on))
        )
        if not coords_ok:
            return False
        if check_epsg:
            other_epsg = other.epsg if isinstance(other, BBox) else None
            return self._epsg == other_epsg
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "west": self[0],
            "east": self[1],
            "south": self[2],
            "north": self[3],
            "epsg": self._epsg,
        }

    @classmethod
    def from_rsgis(
        cls,
        bbox: Sequence[float],
        epsg: Optional[int] = None,
    ) -> "BBox":
        """
        Construct a :class:`BBox` from an RSGIS-order sequence
        ``(xmin, xmax, ymin, ymax)`` (or another :class:`BBox`).
        """
        if isinstance(bbox, BBox):
            eff_epsg = epsg if epsg is not None else bbox.epsg
            return cls(bbox[0], bbox[1], bbox[2], bbox[3], epsg=eff_epsg)
        if len(bbox) < 4:
            raise ValueError(
                f"Expected at least 4 coordinates (xmin, xmax, ymin, ymax), got {len(bbox)}."
            )
        return cls(bbox[0], bbox[1], bbox[2], bbox[3], epsg=epsg)

    @classmethod
    def from_geo(
        cls,
        bounds: Sequence[float],
        epsg: Optional[int] = None,
    ) -> "BBox":
        """
        Construct a :class:`BBox` from a Shapely / Rasterio / GeoPandas
        ``(minx, miny, maxx, maxy)`` sequence (or an existing :class:`BBox`).
        """
        if isinstance(bounds, BBox):
            eff_epsg = epsg if epsg is not None else bounds.epsg
            return cls(bounds[0], bounds[1], bounds[2], bounds[3], epsg=eff_epsg)
        if len(bounds) < 4:
            raise ValueError(
                f"Expected at least 4 coordinates (minx, miny, maxx, maxy), got {len(bounds)}."
            )
        return cls(bounds[0], bounds[2], bounds[1], bounds[3], epsg=epsg)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BBox":
        """Construct a :class:`BBox` from a dictionary created by :meth:`to_dict`."""
        return cls(
            west=data["west"],
            east=data["east"],
            south=data["south"],
            north=data["north"],
            epsg=data.get("epsg"),
        )
