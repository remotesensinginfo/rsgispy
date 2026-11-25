"""
Per-tile polygonisation for
:func:`rsgispy.vector.create.polygonise_raster_tiled`.

Two backends turn a tile of pixel values (plus a validity mask) into the same
intermediate structure, :class:`TilePolys`: flat coordinate arrays in
*global pixel coordinates* (integers stored as ``float64``) with GeoArrow /
:func:`shapely.from_ragged_array` style offsets, the pixel value, pixel count
and pixel bounding box of every output feature.

* ``"numba"``: the compiled kernels in :mod:`rsgispy._polygonise_kernels`
  (labelling, boundary tracing, 8-connectivity grouping), with the rings
  ordered into polygons here.
* ``"rasterio"``: :func:`rasterio.features.shapes` (GDAL) per tile, with the
  GeoJSON coordinates gathered straight into flat arrays (no per-polygon
  shapely objects). With 8-connectivity, GDAL's invalid self-touching rings
  are repaired with :func:`shapely.make_valid`.

Features are converted to map coordinates with the full image geotransform
(``x = c + col * a + row * b``, ``y = f + col * d + row * e``), so pieces
from different tiles share exactly the same vertex coordinates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import shapely

from rsgispy._numba import numba_enabled
from rsgispy._polygonise_kernels import (
    _diag_pairs_kernel,
    _label4_kernel,
    _offsets,
    _trace_rings_kernel,
    _wkb_kernel,
    _DX,
    _DY,
    _LP_DC,
    _LP_DR,
)
from rsgispy._unionfind import uf_min_root_kernel

__all__ = [
    "TilePolys",
    "empty_tile_polys",
    "tile_codes",
    "polygonise_tile_numba",
    "polygonise_tile_rasterio",
    "make_valid_polygonal",
    "orient_exterior_cw",
    "to_map_coords",
    "orient_rings",
    "build_wkb",
    "features_to_geoms",
    "remove_collinear",
]

# rasterio.features.shapes dtypes that need no remapping (others, e.g.
# uint32 / int64 / uint64, are remapped to int32 codes per tile).
_SHAPES_DTYPES = frozenset(
    np.dtype(name)
    for name in ("int8", "int16", "int32", "uint8", "uint16", "float32", "float64")
)


@dataclass
class TilePolys:
    """
    Polygons of one tile in global pixel coordinates.

    * ``coords``: ``(n_pts, 2)`` float64 ``(x=col, y=row)`` ring vertices
      (rings are closed: last vertex == first vertex).
    * ``ring_offsets``: ``(n_rings + 1,)`` offsets into ``coords``.
    * ``part_offsets``: ``(n_parts + 1,)`` offsets into the rings (a part is
      one Polygon: exterior ring followed by its holes).
    * ``feat_offsets``: ``(n_feats + 1,)`` offsets into the parts (a feature
      with several parts is a MultiPolygon).
    * ``values``: ``(n_feats,)`` pixel value of each feature (input dtype).
    * ``counts``: ``(n_feats,)`` int64 pixel count of each feature.
    * ``bbox``: ``(n_feats, 4)`` int64 global pixel bounds
      ``[min_row, max_row, min_col, max_col]`` (inclusive).
    """

    coords: np.ndarray
    ring_offsets: np.ndarray
    part_offsets: np.ndarray
    feat_offsets: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    bbox: np.ndarray

    @property
    def n_feats(self) -> int:
        """Number of features."""
        return int(self.values.shape[0])

    def subset(self, idx: np.ndarray) -> "TilePolys":
        """Return the features ``idx`` (in that order) as a new object."""
        idx = np.asarray(idx, dtype=np.int64)
        feat_parts = np.diff(self.feat_offsets)[idx]
        part_idx = _ranges(self.feat_offsets[idx], feat_parts)
        part_rings = np.diff(self.part_offsets)[part_idx]
        ring_idx = _ranges(self.part_offsets[part_idx], part_rings)
        ring_pts = np.diff(self.ring_offsets)[ring_idx]
        pt_idx = _ranges(self.ring_offsets[ring_idx], ring_pts)
        return TilePolys(
            coords=self.coords[pt_idx],
            ring_offsets=_offsets(ring_pts),
            part_offsets=_offsets(part_rings),
            feat_offsets=_offsets(feat_parts),
            values=self.values[idx],
            counts=self.counts[idx],
            bbox=self.bbox[idx],
        )


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(s, s + n)`` for each ``(s, n)`` (vectorised)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = _offsets(lengths)
    return np.repeat(np.asarray(starts, dtype=np.int64) - offs[:-1], lengths) + np.arange(
        total, dtype=np.int64
    )


def empty_tile_polys(value_dtype: np.dtype) -> TilePolys:
    """Return a :class:`TilePolys` with no features."""
    zero = np.zeros(1, dtype=np.int64)
    return TilePolys(
        coords=np.zeros((0, 2), dtype=np.float64),
        ring_offsets=zero.copy(),
        part_offsets=zero.copy(),
        feat_offsets=zero.copy(),
        values=np.zeros(0, dtype=value_dtype),
        counts=np.zeros(0, dtype=np.int64),
        bbox=np.zeros((0, 4), dtype=np.int64),
    )


def tile_codes(values: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(codes, valid)``: ``int64`` codes that are equal exactly when the
    pixel values are equal (integers cast or reinterpreted, floats compared by
    their ``float64`` bit pattern after normalising ``-0.0``), and the validity
    mask with ``NaN`` pixels removed for float rasters.
    """
    if np.issubdtype(values.dtype, np.floating):
        f64 = values.astype(np.float64)
        valid = valid & ~np.isnan(f64)
        f64 = np.where(f64 == 0.0, 0.0, f64)
        return np.ascontiguousarray(f64).view(np.int64), valid
    if values.dtype == np.uint64:
        return np.ascontiguousarray(values).view(np.int64), valid
    return values.astype(np.int64), valid


def polygonise_tile_numba(
    values: np.ndarray,
    valid: np.ndarray,
    use_8_conn: bool,
    row_off: int = 0,
    col_off: int = 0,
) -> TilePolys:
    """
    Polygonise a tile with the custom kernel.

    :param values: 2D pixel values of the tile.
    :param valid: 2D boolean mask of pixels to polygonise.
    :param use_8_conn: Group diagonally touching regions of the same value.
    :param row_off: Global row offset of the tile.
    :param col_off: Global column offset of the tile.
    :return: :class:`TilePolys` in global pixel coordinates.
    """
    # pylint: disable=too-many-locals
    codes, valid = tile_codes(values, valid)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    labels, n_comp, first_pxl, counts, bbox = _label4_kernel(codes, valid)
    n_comp = int(n_comp)
    if n_comp == 0:
        return empty_tile_polys(values.dtype)
    xs, ys, ring_off, ring_lab, ring_area2 = _trace_rings_kernel(
        labels, _DX, _DY, _LP_DR, _LP_DC
    )

    if use_8_conn:
        pair_u, pair_v = _diag_pairs_kernel(labels, codes)
        if pair_u.shape[0] > 0:
            root = uf_min_root_kernel(pair_u, pair_v, n_comp, 0)
            # roots are the smallest component ID of each group, so groups are
            # numbered in raster order of their first pixel.
            _, feat_of_comp = np.unique(root, return_inverse=True)
        else:
            feat_of_comp = np.arange(n_comp, dtype=np.int64)
    else:
        feat_of_comp = np.arange(n_comp, dtype=np.int64)
    del labels, codes, valid
    feat_of_comp = feat_of_comp.astype(np.int64, copy=False)
    n_feat = int(feat_of_comp.max()) + 1

    # Order components by feature, rings by (component rank, exterior first,
    # trace order).
    comp_order = np.lexsort((np.arange(n_comp), feat_of_comp))
    comp_rank = np.empty(n_comp, dtype=np.int64)
    comp_rank[comp_order] = np.arange(n_comp, dtype=np.int64)
    is_hole = ring_area2 > 0
    ring_rank = comp_rank[ring_lab]
    ring_order = np.lexsort((np.arange(ring_lab.shape[0]), is_hole, ring_rank))
    ring_len = np.diff(ring_off)[ring_order]
    pt_idx = _ranges(ring_off[:-1][ring_order], ring_len)
    coords = np.empty((pt_idx.shape[0], 2), dtype=np.float64)
    coords[:, 0] = xs[pt_idx].astype(np.float64) + float(col_off)
    coords[:, 1] = ys[pt_idx].astype(np.float64) + float(row_off)
    del xs, ys, pt_idx

    rings_per_part = np.bincount(ring_rank, minlength=n_comp)
    parts_per_feat = np.bincount(feat_of_comp, minlength=n_feat)
    sorted_comps = comp_order
    feat_offsets = _offsets(parts_per_feat)
    feat_first = feat_offsets[:-1]
    first_comp = sorted_comps[feat_first]
    feat_counts = np.add.reduceat(counts[sorted_comps], feat_first)
    s_bbox = bbox[sorted_comps]
    feat_bbox = np.empty((n_feat, 4), dtype=np.int64)
    feat_bbox[:, 0] = np.minimum.reduceat(s_bbox[:, 0], feat_first) + row_off
    feat_bbox[:, 1] = np.maximum.reduceat(s_bbox[:, 1], feat_first) + row_off
    feat_bbox[:, 2] = np.minimum.reduceat(s_bbox[:, 2], feat_first) + col_off
    feat_bbox[:, 3] = np.maximum.reduceat(s_bbox[:, 3], feat_first) + col_off
    return TilePolys(
        coords=coords,
        ring_offsets=_offsets(ring_len),
        part_offsets=_offsets(rings_per_part),
        feat_offsets=feat_offsets,
        values=values.ravel()[first_pxl[first_comp]],
        counts=feat_counts.astype(np.int64),
        bbox=feat_bbox,
    )


def _geoms_to_tile_polys(
    geoms: np.ndarray, values: np.ndarray, counts: Optional[np.ndarray] = None
) -> TilePolys:
    """Convert (Multi)Polygon geometries (pixel coordinates) to :class:`TilePolys`."""
    if geoms.shape[0] == 0:
        return empty_tile_polys(values.dtype)
    parts, part_feat = shapely.get_parts(geoms, return_index=True)
    _, coords, (ring_off, part_off) = shapely.to_ragged_array(parts)
    parts_per_feat = np.bincount(part_feat, minlength=geoms.shape[0])
    if counts is None:
        counts = np.rint(shapely.area(geoms)).astype(np.int64)
    bnds = shapely.bounds(geoms)
    bbox = np.empty((geoms.shape[0], 4), dtype=np.int64)
    bbox[:, 0] = np.rint(bnds[:, 1]).astype(np.int64)
    bbox[:, 1] = np.rint(bnds[:, 3]).astype(np.int64) - 1
    bbox[:, 2] = np.rint(bnds[:, 0]).astype(np.int64)
    bbox[:, 3] = np.rint(bnds[:, 2]).astype(np.int64) - 1
    return TilePolys(
        coords=np.ascontiguousarray(coords, dtype=np.float64),
        ring_offsets=np.asarray(ring_off, dtype=np.int64),
        part_offsets=np.asarray(part_off, dtype=np.int64),
        feat_offsets=_offsets(parts_per_feat),
        values=values,
        counts=np.asarray(counts, dtype=np.int64),
        bbox=bbox,
    )


def _self_touching_polys(
    coords: np.ndarray, ring_off: np.ndarray, poly_of_ring: np.ndarray
) -> np.ndarray:
    """
    Return the indices of polygons with a vertex repeated within the polygon
    (excluding ring closing vertices): candidates for GDAL's invalid
    self-touching 8-connected rings.
    """
    n_rings = ring_off.shape[0] - 1
    if n_rings == 0:
        return np.zeros(0, dtype=np.int64)
    ring_len = np.diff(ring_off)
    keep = np.ones(coords.shape[0], dtype=bool)
    keep[ring_off[1:] - 1] = False  # closing vertices
    pt_poly = np.repeat(poly_of_ring, ring_len)[keep]
    pts = coords[keep]
    order = np.lexsort((pts[:, 1], pts[:, 0], pt_poly))
    s_poly = pt_poly[order]
    s_pts = pts[order]
    dup = (
        (s_poly[1:] == s_poly[:-1])
        & (s_pts[1:, 0] == s_pts[:-1, 0])
        & (s_pts[1:, 1] == s_pts[:-1, 1])
    )
    return np.unique(s_poly[1:][dup])


def polygonise_tile_rasterio(
    values: np.ndarray,
    valid: np.ndarray,
    use_8_conn: bool,
    row_off: int = 0,
    col_off: int = 0,
) -> TilePolys:
    """
    Polygonise a tile with :func:`rasterio.features.shapes` (GDAL), building
    the coordinate arrays directly from the GeoJSON-like output.

    Parameters as for :func:`polygonise_tile_numba`.
    """
    # pylint: disable=too-many-locals,import-outside-toplevel
    # pylint: disable=too-many-branches,too-many-statements
    import rasterio.features
    from affine import Affine

    valid = np.asarray(valid, dtype=bool)
    if np.issubdtype(values.dtype, np.floating):
        valid = valid & ~np.isnan(values)
    if not valid.any():
        return empty_tile_polys(values.dtype)
    remap: Optional[np.ndarray] = None
    src = values
    if values.dtype not in _SHAPES_DTYPES:
        if np.issubdtype(values.dtype, np.floating):
            src = values.astype(np.float64)
        else:
            remap, inv = np.unique(values[valid], return_inverse=True)
            src = np.zeros(values.shape, dtype=np.int32)
            src[valid] = (inv + 1).astype(np.int32)
    transform = Affine(1.0, 0.0, float(col_off), 0.0, 1.0, float(row_off))
    ring_len = []
    rings_per_poly = []
    flat = []
    pxl_vals = []
    for geom, pxl_val in rasterio.features.shapes(
        src, mask=valid, connectivity=8 if use_8_conn else 4, transform=transform
    ):
        rings = geom["coordinates"]
        rings_per_poly.append(len(rings))
        pxl_vals.append(pxl_val)
        for ring in rings:
            ring_len.append(len(ring))
            flat.extend(ring)
    coords = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    ring_off = _offsets(np.asarray(ring_len, dtype=np.int64))
    part_off = _offsets(np.asarray(rings_per_poly, dtype=np.int64))
    raw_vals = np.asarray(pxl_vals)
    if remap is not None:
        out_vals = remap[np.rint(raw_vals).astype(np.int64) - 1]
    else:
        out_vals = raw_vals.astype(values.dtype)
    n_poly = part_off.shape[0] - 1
    if use_8_conn:
        poly_of_ring = np.repeat(np.arange(n_poly, dtype=np.int64), np.diff(part_off))
        cand = _self_touching_polys(coords, ring_off, poly_of_ring)
        if cand.shape[0] > 0:
            geoms = shapely.from_ragged_array(
                shapely.GeometryType.POLYGON, coords, (ring_off, part_off)
            )
            bad = cand[~shapely.is_valid(geoms[cand])]
            if bad.shape[0] > 0:
                geoms = geoms.astype(object)
                geoms[bad] = orient_exterior_cw(make_valid_polygonal(geoms[bad]))
                return _geoms_to_tile_polys(geoms, out_vals)
    # Polygon area (pixel units == pixel count) = exterior - hole areas.
    abs_area = np.abs(_ring_areas(coords, ring_off))
    poly_of_ring = np.repeat(np.arange(n_poly, dtype=np.int64), np.diff(part_off))
    is_ext = np.zeros(ring_off.shape[0] - 1, dtype=bool)
    is_ext[part_off[:-1]] = True
    counts = np.bincount(
        poly_of_ring, weights=np.where(is_ext, abs_area, -abs_area), minlength=n_poly
    )
    bbox = np.empty((n_poly, 4), dtype=np.int64)
    if n_poly:
        # The exterior ring (first ring of each polygon) holds the bounds.
        x_min, x_max, y_min, y_max = _ring_bounds(
            coords, ring_off[part_off[:-1]], np.diff(ring_off)[part_off[:-1]]
        )
        bbox[:, 0] = y_min
        bbox[:, 1] = y_max - 1
        bbox[:, 2] = x_min
        bbox[:, 3] = x_max - 1
    return TilePolys(
        coords=coords,
        ring_offsets=ring_off,
        part_offsets=part_off,
        feat_offsets=np.arange(n_poly + 1, dtype=np.int64),
        values=out_vals,
        counts=np.rint(counts).astype(np.int64),
        bbox=bbox,
    )


def _ring_areas(coords: np.ndarray, ring_off: np.ndarray) -> np.ndarray:
    """Signed shoelace area of each closed ring."""
    if coords.shape[0] == 0:
        return np.zeros(ring_off.shape[0] - 1, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    term = np.zeros(coords.shape[0], dtype=np.float64)
    term[:-1] = x[:-1] * y[1:] - x[1:] * y[:-1]
    # zero the cross-ring terms (last vertex of a ring to first of the next)
    term[ring_off[1:] - 1] = 0.0
    sums = np.add.reduceat(term, ring_off[:-1]) if ring_off.shape[0] > 1 else np.zeros(0)
    return 0.5 * sums


def _ring_bounds(
    coords: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bounds ``(x_min, x_max, y_min, y_max)`` of the rings ``starts``/``lengths``."""
    idx = _ranges(starts, lengths)
    sub = coords[idx]
    offs = _offsets(lengths)[:-1]
    return (
        np.minimum.reduceat(sub[:, 0], offs).astype(np.int64),
        np.maximum.reduceat(sub[:, 0], offs).astype(np.int64),
        np.minimum.reduceat(sub[:, 1], offs).astype(np.int64),
        np.maximum.reduceat(sub[:, 1], offs).astype(np.int64),
    )


def make_valid_polygonal(geoms: np.ndarray) -> np.ndarray:
    """
    Repair invalid polygons, keeping polygonal output only (point-set
    preserving for GDAL's self-touching 8-connected rings).
    """
    try:
        return shapely.make_valid(geoms, method="structure", keep_collapsed=False)
    except TypeError:  # shapely < 2.1
        fixed = shapely.make_valid(geoms)
        out = np.empty(fixed.shape[0], dtype=object)
        for i, geom in enumerate(fixed):
            parts = [
                p for p in shapely.get_parts(geom)
                if shapely.get_type_id(p) in (3, 6)
            ]
            out[i] = shapely.union_all(parts) if len(parts) != 1 else parts[0]
        return out


def orient_exterior_cw(geoms: np.ndarray) -> np.ndarray:
    """
    Orient polygons with clockwise exteriors and counter-clockwise holes in
    *pixel* coordinate numbers (rows increase downwards, so this is the
    conventional counter-clockwise exterior once mapped to north-up map
    coordinates).
    """
    if hasattr(shapely, "orient_polygons"):
        return shapely.orient_polygons(geoms, exterior_cw=True)
    from shapely.geometry.polygon import orient  # pylint: disable=import-outside-toplevel

    out = np.empty(geoms.shape[0], dtype=object)
    for i, geom in enumerate(geoms):
        if shapely.get_type_id(geom) == 6:
            out[i] = shapely.multipolygons([orient(p, sign=-1.0) for p in geom.geoms])
        else:
            out[i] = orient(geom, sign=-1.0)
    return out


def to_map_coords(coords: np.ndarray, gt: Tuple[float, ...]) -> np.ndarray:
    """
    Convert global pixel coordinates to map coordinates with the GDAL
    geotransform ``gt = (c, a, b, f, d, e)``:
    ``x = c + col * a + row * b``, ``y = f + col * d + row * e``.
    """
    out = np.empty_like(coords)
    out[:, 0] = gt[0] + coords[:, 0] * gt[1] + coords[:, 1] * gt[2]
    out[:, 1] = gt[3] + coords[:, 0] * gt[4] + coords[:, 1] * gt[5]
    return out


def orient_rings(polys: TilePolys, flip: bool) -> TilePolys:
    """
    Reverse every ring when ``flip`` is ``True`` (used when the geotransform
    mirrors the pixel grid so exteriors stay counter-clockwise in map
    coordinates).
    """
    if not flip or polys.coords.shape[0] == 0:
        return polys
    ring_len = np.diff(polys.ring_offsets)
    starts = polys.ring_offsets[:-1]
    # reversed index within each ring: start + (len - 1 - k)
    local = np.arange(polys.coords.shape[0], dtype=np.int64) - np.repeat(starts, ring_len)
    rev_idx = np.repeat(starts + ring_len - 1, ring_len) - local
    return TilePolys(
        coords=polys.coords[rev_idx],
        ring_offsets=polys.ring_offsets,
        part_offsets=polys.part_offsets,
        feat_offsets=polys.feat_offsets,
        values=polys.values,
        counts=polys.counts,
        bbox=polys.bbox,
    )


def build_wkb(
    coords: np.ndarray,
    ring_off: np.ndarray,
    part_off: np.ndarray,
    feat_off: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode features as WKB: returns ``(buffer uint8, offsets int64)`` (one
    feature per offset pair). Uses the compiled kernel when ``numba`` is
    enabled, otherwise :func:`shapely.to_wkb` (identical bytes).
    """
    n_feat = feat_off.shape[0] - 1
    if n_feat == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(1, dtype=np.int64)
    if numba_enabled() and _wkb_kernel.jit_func is not None:
        if np.little_endian:
            xy_bytes = np.ascontiguousarray(coords, dtype=np.float64).view(np.uint8).ravel()
            return _wkb_kernel(xy_bytes, ring_off, part_off, feat_off)
    geoms = features_to_geoms(coords, ring_off, part_off, feat_off)
    wkbs = shapely.to_wkb(geoms)
    lens = np.fromiter((len(w) for w in wkbs), dtype=np.int64, count=wkbs.shape[0])
    buf = np.frombuffer(b"".join(wkbs.tolist()), dtype=np.uint8)
    return buf, _offsets(lens)


def features_to_geoms(
    coords: np.ndarray,
    ring_off: np.ndarray,
    part_off: np.ndarray,
    feat_off: np.ndarray,
) -> np.ndarray:
    """Build shapely geometries (Polygon / MultiPolygon) from the flat arrays."""
    polys = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON, coords, (ring_off, part_off)
    )
    parts_per_feat = np.diff(feat_off)
    if np.all(parts_per_feat == 1):
        return polys
    out = np.empty(parts_per_feat.shape[0], dtype=object)
    single = parts_per_feat == 1
    out[single] = polys[feat_off[:-1][single]]
    multi = np.nonzero(~single)[0]
    part_idx = _ranges(feat_off[multi], parts_per_feat[multi])
    grp = np.repeat(np.arange(multi.shape[0]), parts_per_feat[multi])
    out[multi] = shapely.multipolygons(polys[part_idx], indices=grp)
    return out


def remove_collinear(geoms: np.ndarray) -> np.ndarray:
    """
    Remove collinear (and repeated) vertices from (Multi)Polygons whose
    coordinates are on the pixel grid, keeping only corner vertices (used
    after merging pieces across tile seams).
    """
    # pylint: disable=too-many-locals
    if geoms.shape[0] == 0:
        return geoms
    tp = _geoms_to_tile_polys(geoms, np.zeros(geoms.shape[0]), np.zeros(geoms.shape[0]))
    coords = tp.coords
    ring_off = tp.ring_offsets
    ring_len = np.diff(ring_off)
    open_len = ring_len - 1
    starts = ring_off[:-1]
    # open ring vertices (excluding the closing vertex)
    idx = _ranges(starts, open_len)
    ring_of = np.repeat(np.arange(open_len.shape[0]), open_len)
    local = idx - np.repeat(starts, open_len)
    n_open = open_len[ring_of]
    prev_idx = starts[ring_of] + (local - 1) % n_open
    next_idx = starts[ring_of] + (local + 1) % n_open
    cur = coords[idx]
    prv = coords[prev_idx]
    nxt = coords[next_idx]
    cross = (cur[:, 0] - prv[:, 0]) * (nxt[:, 1] - cur[:, 1]) - (
        cur[:, 1] - prv[:, 1]
    ) * (nxt[:, 0] - cur[:, 0])
    keep = cross != 0.0
    kept_idx = idx[keep]
    kept_ring = ring_of[keep]
    new_open = np.bincount(kept_ring, minlength=open_len.shape[0])
    new_off = _offsets(new_open + 1)
    out = np.empty((int(new_off[-1]), 2), dtype=np.float64)
    # positions of kept vertices in the new array
    pos = np.arange(kept_idx.shape[0]) + np.repeat(np.arange(open_len.shape[0]), new_open)
    out[pos] = coords[kept_idx]
    first_new = new_off[:-1]
    out[new_off[1:] - 1] = out[first_new]
    return features_to_geoms(out, new_off, tp.part_offsets, tp.feat_offsets)
