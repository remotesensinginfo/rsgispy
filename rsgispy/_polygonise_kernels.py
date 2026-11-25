"""
Compiled kernels for the custom (``"numba"``) polygonisation backend of
:func:`rsgispy.vector.create.polygonise_raster_tiled`.

Every kernel is wrapped with :func:`rsgispy._numba.optional_njit`, so it is
``numba``-compiled when available and otherwise runs the same algorithm in
plain Python (correct, but slow):

1. :func:`_label4_kernel`: 4-connected component labelling (union-find on
   pixel indices; component IDs in raster order of their first pixel).
2. :func:`_trace_rings_kernel`: boundary tracing along pixel edges with the
   component on the left of every directed edge. Only corner vertices are
   emitted, so collinear vertices are removed while tracing. Where a
   component touches itself diagonally (a "saddle" vertex), the walk turns
   so that the component's outer boundary and the boundary of the region it
   encloses are separate rings touching at that vertex. Rings never touch
   themselves, so every polygon is valid (OGC simple feature rules). The
   exterior ring has a negative signed area in pixel (row-down) coordinates,
   holes a positive one.
3. :func:`_diag_pairs_kernel`: pairs of components of equal value touching
   diagonally, grouped into one feature with 8-connectivity (written as a
   MultiPolygon whose parts touch at corners; GDAL writes these regions as a
   single, invalid, self-touching ring).
4. :func:`_wkb_kernel` / :func:`merge_wkb_runs_kernel`: encode features as
   WKB directly from the coordinate arrays, and merge the WKB of consecutive
   rows (dissolve by value) without parsing the geometries.
"""

import numpy as np

from rsgispy._numba import optional_njit

__all__ = ["merge_wkb_runs_kernel"]

# Directions in pixel (row-down) space: 0 = E (+x), 1 = N (-y), 2 = W (-x),
# 3 = S (+y). ``_LP_*`` give the row / column offset (relative to the vertex)
# of the pixel on the *left* of an edge leaving the vertex in that direction.
_DX = np.array([1, 0, -1, 0], dtype=np.int64)
_DY = np.array([0, -1, 0, 1], dtype=np.int64)
_LP_DR = np.array([-1, -1, 0, 0], dtype=np.int64)
_LP_DC = np.array([0, -1, -1, 0], dtype=np.int64)


def _offsets(lengths: np.ndarray) -> np.ndarray:
    """Return ``[0, cumsum(lengths)]`` as int64."""
    out = np.zeros(lengths.shape[0] + 1, dtype=np.int64)
    np.cumsum(lengths, out=out[1:])
    return out


@optional_njit(cache=True)
def _label4_kernel(codes, valid):  # pragma: no cover - exercised compiled & fallback
    """
    4-connected component labelling of equal ``codes`` over ``valid`` pixels.

    Returns ``(labels, n_comp, first_pxl, counts, bbox)``: ``labels`` int32
    (``-1`` for invalid pixels) with component IDs in raster order of each
    component's first pixel, the flat index of that first pixel, the pixel
    count and the inclusive ``[min_row, max_row, min_col, max_col]`` bounds.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    n_rows, n_cols = codes.shape
    n_pxl = n_rows * n_cols
    parent = np.empty(n_pxl, dtype=np.int64)
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if not valid[row, col]:
                parent[idx] = -1
                continue
            parent[idx] = idx
            val = codes[row, col]
            if col > 0 and valid[row, col - 1] and codes[row, col - 1] == val:
                root_a = idx - 1
                while parent[root_a] != root_a:
                    parent[root_a] = parent[parent[root_a]]
                    root_a = parent[root_a]
                parent[idx] = root_a
            if row > 0 and valid[row - 1, col] and codes[row - 1, col] == val:
                root_a = idx - n_cols
                while parent[root_a] != root_a:
                    parent[root_a] = parent[parent[root_a]]
                    root_a = parent[root_a]
                root_b = idx
                while parent[root_b] != root_b:
                    parent[root_b] = parent[parent[root_b]]
                    root_b = parent[root_b]
                if root_a < root_b:
                    parent[root_b] = root_a
                elif root_b < root_a:
                    parent[root_a] = root_b

    labels = np.empty((n_rows, n_cols), dtype=np.int32)
    cap = 1024
    first_pxl = np.empty(cap, dtype=np.int64)
    counts = np.empty(cap, dtype=np.int64)
    bbox = np.empty((cap, 4), dtype=np.int64)
    n_comp = 0
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if parent[idx] < 0:
                labels[row, col] = -1
                continue
            root = idx
            while parent[root] != root:
                root = parent[root]
            parent[idx] = root
            if root == idx:
                if n_comp == cap:
                    cap *= 2
                    tmp_f = np.empty(cap, dtype=np.int64)
                    tmp_f[:n_comp] = first_pxl[:n_comp]
                    first_pxl = tmp_f
                    tmp_c = np.empty(cap, dtype=np.int64)
                    tmp_c[:n_comp] = counts[:n_comp]
                    counts = tmp_c
                    tmp_b = np.empty((cap, 4), dtype=np.int64)
                    tmp_b[:n_comp] = bbox[:n_comp]
                    bbox = tmp_b
                comp = n_comp
                n_comp += 1
                first_pxl[comp] = idx
                counts[comp] = 0
                bbox[comp, 0] = row
                bbox[comp, 1] = row
                bbox[comp, 2] = col
                bbox[comp, 3] = col
            else:
                comp = labels[root // n_cols, root % n_cols]
            labels[row, col] = comp
            counts[comp] += 1
            if row > bbox[comp, 1]:
                bbox[comp, 1] = row
            if col < bbox[comp, 2]:
                bbox[comp, 2] = col
            if col > bbox[comp, 3]:
                bbox[comp, 3] = col
    return (
        labels,
        n_comp,
        first_pxl[:n_comp].copy(),
        counts[:n_comp].copy(),
        bbox[:n_comp].copy(),
    )


@optional_njit(cache=True)
def _trace_rings_kernel(labels, d_x, d_y, lp_dr, lp_dc):  # pragma: no cover
    """
    Trace the boundary rings of every component in ``labels``.

    Each ring is traced with its component on the left (row-down pixel
    space) and only corner vertices are emitted; rings are closed (the first
    vertex is repeated). At a vertex the walk prefers a right turn, then
    straight on, then a left turn, which separates a component's exterior
    from the regions it encloses at diagonal "saddle" vertices (so no ring
    touches itself).

    Returns ``(xs, ys, ring_off, ring_lab, ring_area2)``: tile-local int32
    vertex coordinates, ring offsets, the component of each ring and twice
    its signed area (negative for exteriors, positive for holes).
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    n_rows, n_cols = labels.shape
    n_vcols = n_cols + 1
    visited = np.zeros((n_rows + 1) * n_vcols, dtype=np.uint8)
    cap = 4 * (n_rows + n_cols) + 1024
    xs = np.empty(cap, dtype=np.int32)
    ys = np.empty(cap, dtype=np.int32)
    rcap = 1024
    ring_off = np.empty(rcap + 1, dtype=np.int64)
    ring_lab = np.empty(rcap, dtype=np.int32)
    ring_area2 = np.empty(rcap, dtype=np.int64)
    n_pts = 0
    n_ring = 0
    ring_off[0] = 0
    for row in range(n_rows):
        for col in range(n_cols):
            lab = labels[row, col]
            if lab < 0:
                continue
            if row > 0 and labels[row - 1, col] == lab:
                continue
            # The top edge of this pixel, walked west from (col + 1, row).
            start_vid = row * n_vcols + col + 1
            if visited[start_vid] & 4:
                continue
            visited[start_vid] |= 4
            if n_ring == rcap:
                rcap *= 2
                tmp_o = np.empty(rcap + 1, dtype=np.int64)
                tmp_o[: n_ring + 1] = ring_off[: n_ring + 1]
                ring_off = tmp_o
                tmp_l = np.empty(rcap, dtype=np.int32)
                tmp_l[:n_ring] = ring_lab[:n_ring]
                ring_lab = tmp_l
                tmp_a = np.empty(rcap, dtype=np.int64)
                tmp_a[:n_ring] = ring_area2[:n_ring]
                ring_area2 = tmp_a
            x0 = col
            y0 = row
            pos_x = x0
            pos_y = y0
            cur_d = 2
            area2 = 0
            first = True
            prev_x = 0
            prev_y = 0
            while True:
                if (not first) and pos_x == x0 and pos_y == y0 and cur_d == 2:
                    break
                # Choose the outgoing direction (right, straight, left).
                nxt_d = (cur_d + 3) & 3
                l_r = pos_y + lp_dr[nxt_d]
                l_c = pos_x + lp_dc[nxt_d]
                ok = (
                    0 <= l_r < n_rows and 0 <= l_c < n_cols and labels[l_r, l_c] == lab
                )
                if not ok:
                    nxt_d = cur_d
                    l_r = pos_y + lp_dr[nxt_d]
                    l_c = pos_x + lp_dc[nxt_d]
                    ok = (
                        0 <= l_r < n_rows
                        and 0 <= l_c < n_cols
                        and labels[l_r, l_c] == lab
                    )
                    if not ok:
                        nxt_d = (cur_d + 1) & 3
                if nxt_d != cur_d or first:
                    if n_pts + 2 > cap:
                        cap *= 2
                        tmp_x = np.empty(cap, dtype=np.int32)
                        tmp_x[:n_pts] = xs[:n_pts]
                        xs = tmp_x
                        tmp_y = np.empty(cap, dtype=np.int32)
                        tmp_y[:n_pts] = ys[:n_pts]
                        ys = tmp_y
                    xs[n_pts] = pos_x
                    ys[n_pts] = pos_y
                    if not first:
                        area2 += prev_x * pos_y - pos_x * prev_y
                    prev_x = pos_x
                    prev_y = pos_y
                    n_pts += 1
                    first = False
                visited[pos_y * n_vcols + pos_x] |= np.uint8(1 << nxt_d)
                pos_x += d_x[nxt_d]
                pos_y += d_y[nxt_d]
                cur_d = nxt_d
            # close the ring
            xs[n_pts] = x0
            ys[n_pts] = y0
            area2 += prev_x * y0 - x0 * prev_y
            n_pts += 1
            ring_lab[n_ring] = lab
            ring_area2[n_ring] = area2
            n_ring += 1
            ring_off[n_ring] = n_pts
    return (
        xs[:n_pts].copy(),
        ys[:n_pts].copy(),
        ring_off[: n_ring + 1].copy(),
        ring_lab[:n_ring].copy(),
        ring_area2[:n_ring].copy(),
    )


@optional_njit(cache=True)
def _diag_pairs_kernel(labels, codes):  # pragma: no cover
    """
    Return ``(u, v)`` component pairs of equal value touching diagonally
    (pixels ``(r, c)`` and ``(r + 1, c +/- 1)`` in different components).
    """
    # pylint: disable=too-many-locals
    n_rows, n_cols = labels.shape
    cap = 1024
    u_arr = np.empty(cap, dtype=np.int64)
    v_arr = np.empty(cap, dtype=np.int64)
    n_pairs = 0
    for row in range(n_rows - 1):
        for col in range(n_cols):
            lab = labels[row, col]
            if lab < 0:
                continue
            for d_c in (-1, 1):
                n_c = col + d_c
                if n_c < 0 or n_c >= n_cols:
                    continue
                other = labels[row + 1, n_c]
                if other < 0 or other == lab or codes[row + 1, n_c] != codes[row, col]:
                    continue
                if n_pairs == cap:
                    cap *= 2
                    tmp_u = np.empty(cap, dtype=np.int64)
                    tmp_u[:n_pairs] = u_arr[:n_pairs]
                    u_arr = tmp_u
                    tmp_v = np.empty(cap, dtype=np.int64)
                    tmp_v[:n_pairs] = v_arr[:n_pairs]
                    v_arr = tmp_v
                u_arr[n_pairs] = lab
                v_arr[n_pairs] = other
                n_pairs += 1
    return u_arr[:n_pairs].copy(), v_arr[:n_pairs].copy()


@optional_njit(cache=True)
def _wkb_kernel(xy_bytes, ring_off, part_off, feat_off):  # pragma: no cover
    """
    Encode features as little-endian ISO WKB (Polygon for single-part
    features, MultiPolygon otherwise). ``xy_bytes`` is the C-ordered
    ``(n_pts, 2)`` ``float64`` coordinate array viewed as ``uint8`` (each
    ring's points are copied as one block). Returns ``(buffer, offsets)``.
    """
    # pylint: disable=too-many-locals
    n_feat = feat_off.shape[0] - 1
    offs = np.empty(n_feat + 1, dtype=np.int64)
    offs[0] = 0
    for f_i in range(n_feat):
        p_0 = feat_off[f_i]
        p_1 = feat_off[f_i + 1]
        size = 9 if (p_1 - p_0) > 1 else 0
        for p_i in range(p_0, p_1):
            size += 9
            for r_i in range(part_off[p_i], part_off[p_i + 1]):
                size += 4 + 16 * (ring_off[r_i + 1] - ring_off[r_i])
        offs[f_i + 1] = offs[f_i] + size
    buf = np.empty(offs[n_feat], dtype=np.uint8)
    pos = 0
    for f_i in range(n_feat):
        p_0 = feat_off[f_i]
        p_1 = feat_off[f_i + 1]
        if (p_1 - p_0) > 1:
            n_val = p_1 - p_0
            buf[pos] = 1
            buf[pos + 1] = 6
            buf[pos + 2] = 0
            buf[pos + 3] = 0
            buf[pos + 4] = 0
            buf[pos + 5] = n_val & 255
            buf[pos + 6] = (n_val >> 8) & 255
            buf[pos + 7] = (n_val >> 16) & 255
            buf[pos + 8] = (n_val >> 24) & 255
            pos += 9
        for p_i in range(p_0, p_1):
            n_val = part_off[p_i + 1] - part_off[p_i]
            buf[pos] = 1
            buf[pos + 1] = 3
            buf[pos + 2] = 0
            buf[pos + 3] = 0
            buf[pos + 4] = 0
            buf[pos + 5] = n_val & 255
            buf[pos + 6] = (n_val >> 8) & 255
            buf[pos + 7] = (n_val >> 16) & 255
            buf[pos + 8] = (n_val >> 24) & 255
            pos += 9
            for r_i in range(part_off[p_i], part_off[p_i + 1]):
                n_val = ring_off[r_i + 1] - ring_off[r_i]
                buf[pos] = n_val & 255
                buf[pos + 1] = (n_val >> 8) & 255
                buf[pos + 2] = (n_val >> 16) & 255
                buf[pos + 3] = (n_val >> 24) & 255
                pos += 4
                n_bytes = 16 * n_val
                src = 16 * ring_off[r_i]
                buf[pos : pos + n_bytes] = xy_bytes[src : src + n_bytes]
                pos += n_bytes
    return buf, offs


def _merge_wkb_runs_py(buf, offs, run_start, run_len):
    """Pure-Python implementation of :func:`merge_wkb_runs_kernel`."""
    raw = buf.tobytes()
    pieces = []
    out_len = np.empty(run_start.shape[0], dtype=np.int64)
    for r_i in range(run_start.shape[0]):
        start = int(run_start[r_i])
        n_rows = int(run_len[r_i])
        if n_rows == 1:
            item = raw[int(offs[start]) : int(offs[start + 1])]  # noqa: E203
        else:
            parts = []
            n_parts = 0
            for row in range(start, start + n_rows):
                wkb = raw[int(offs[row]) : int(offs[row + 1])]  # noqa: E203
                if wkb[1] == 6:
                    n_parts += int.from_bytes(wkb[5:9], "little")
                    parts.append(wkb[9:])
                else:
                    n_parts += 1
                    parts.append(wkb)
            item = b"\x01\x06\x00\x00\x00" + n_parts.to_bytes(4, "little") + b"".join(parts)
        pieces.append(item)
        out_len[r_i] = len(item)
    return np.frombuffer(b"".join(pieces), dtype=np.uint8), _offsets(out_len)


@optional_njit(cache=True, fallback=_merge_wkb_runs_py)
def merge_wkb_runs_kernel(buf, offs, run_start, run_len):  # pragma: no cover
    """
    Combine runs of consecutive little-endian (Multi)Polygon WKB rows into
    one feature per run (a MultiPolygon holding every part of the run; runs
    of one row are copied unchanged). Used to dissolve features by value
    without parsing the geometries. Returns ``(buffer, offsets)``.
    """
    # pylint: disable=too-many-locals,too-many-branches
    n_runs = run_start.shape[0]
    out_off = np.empty(n_runs + 1, dtype=np.int64)
    out_off[0] = 0
    for r_i in range(n_runs):
        start = run_start[r_i]
        if run_len[r_i] == 1:
            size = offs[start + 1] - offs[start]
        else:
            size = 9
            for row in range(start, start + run_len[r_i]):
                if buf[offs[row] + 1] == 6:
                    size += offs[row + 1] - offs[row] - 9
                else:
                    size += offs[row + 1] - offs[row]
        out_off[r_i + 1] = out_off[r_i] + size
    out = np.empty(out_off[n_runs], dtype=np.uint8)
    for r_i in range(n_runs):
        start = run_start[r_i]
        pos = out_off[r_i]
        if run_len[r_i] == 1:
            n_bytes = offs[start + 1] - offs[start]
            out[pos : pos + n_bytes] = buf[offs[start] : offs[start + 1]]
            continue
        n_parts = 0
        for row in range(start, start + run_len[r_i]):
            b_0 = offs[row]
            if buf[b_0 + 1] == 6:
                n_parts += (
                    np.int64(buf[b_0 + 5])
                    + (np.int64(buf[b_0 + 6]) << 8)
                    + (np.int64(buf[b_0 + 7]) << 16)
                    + (np.int64(buf[b_0 + 8]) << 24)
                )
            else:
                n_parts += 1
        out[pos] = 1
        out[pos + 1] = 6
        out[pos + 2] = 0
        out[pos + 3] = 0
        out[pos + 4] = 0
        out[pos + 5] = n_parts & 255
        out[pos + 6] = (n_parts >> 8) & 255
        out[pos + 7] = (n_parts >> 16) & 255
        out[pos + 8] = (n_parts >> 24) & 255
        pos += 9
        for row in range(start, start + run_len[r_i]):
            b_0 = offs[row]
            b_1 = offs[row + 1]
            if buf[b_0 + 1] == 6:
                b_0 += 9
            out[pos : pos + (b_1 - b_0)] = buf[b_0:b_1]
            pos += b_1 - b_0
    return out, out_off
