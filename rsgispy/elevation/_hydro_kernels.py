# pylint: disable=too-many-lines
"""
Numba-accelerated numerical kernels and vectorised NumPy fallbacks for
:mod:`rsgispy.elevation.hydrology`.
"""

import math
from typing import Tuple

import numpy as np

from rsgispy._numba import HAS_NUMBA, njit, optional_njit, prange  # noqa: F401


# Clockwise D8 encoding starting East:
# 0:E(1), 1:SE(2), 2:S(4), 3:SW(8), 4:W(16), 5:NW(32), 6:N(64), 7:NE(128)
D8_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
D8_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
D8_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
D8_REV_CODES = np.array([16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)
D8_CONTOUR_L = np.array([0.5, 0.354, 0.5, 0.354, 0.5, 0.354, 0.5, 0.354], dtype=np.float64)


@njit
def _d8_to_idx(code: int) -> int:
    """Return neighbour index 0..7 for a D8 bitmask, or -1 if unresolved/invalid."""
    if code == 1:
        return 0
    if code == 2:
        return 1
    if code == 4:
        return 2
    if code == 8:
        return 3
    if code == 16:
        return 4
    if code == 32:
        return 5
    if code == 64:
        return 6
    if code == 128:
        return 7
    return -1


@njit
def _d8_to_offset(code: int) -> Tuple[int, int]:
    """Return ``(dr, dc)`` offset for a D8 bitmask, or ``(0, 0)`` if invalid."""
    idx = _d8_to_idx(code)
    if idx < 0:
        return 0, 0
    return int(D8_DR[idx]), int(D8_DC[idx])


@njit
def _step_dist(k: int, dx: float, dy: float, diag_d: float) -> float:
    """Return metric centre-to-centre distance for D8 neighbour index ``k``."""
    if k == 0 or k == 4:
        return dx
    if k == 2 or k == 6:
        return dy
    return diag_d


@njit
def _heap_push(
    keys: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    heap_size: int,
    key: float,
    row: int,
    col: int,
) -> int:
    """Push ``(key, row, col)`` onto an array-backed binary min-heap."""
    idx = heap_size
    keys[idx] = key
    rows[idx] = row
    cols[idx] = col
    while idx > 0:
        parent = (idx - 1) >> 1
        if keys[parent] <= keys[idx]:
            break
        tmp_k = keys[parent]
        tmp_r = rows[parent]
        tmp_c = cols[parent]
        keys[parent] = keys[idx]
        rows[parent] = rows[idx]
        cols[parent] = cols[idx]
        keys[idx] = tmp_k
        rows[idx] = tmp_r
        cols[idx] = tmp_c
        idx = parent
    return heap_size + 1


@njit
def _heap_pop(
    keys: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    heap_size: int,
) -> Tuple[float, int, int, int]:
    """Pop minimum ``(key, row, col, new_heap_size)`` from min-heap."""
    min_k = keys[0]
    min_r = int(rows[0])
    min_c = int(cols[0])
    new_size = heap_size - 1
    if new_size > 0:
        keys[0] = keys[new_size]
        rows[0] = rows[new_size]
        cols[0] = cols[new_size]
        idx = 0
        while True:
            left = (idx << 1) + 1
            if left >= new_size:
                break
            right = left + 1
            smallest = left
            if right < new_size and keys[right] < keys[left]:
                smallest = right
            if keys[idx] <= keys[smallest]:
                break
            tmp_k = keys[idx]
            tmp_r = rows[idx]
            tmp_c = cols[idx]
            keys[idx] = keys[smallest]
            rows[idx] = rows[smallest]
            cols[idx] = cols[smallest]
            keys[smallest] = tmp_k
            rows[smallest] = tmp_r
            cols[smallest] = tmp_c
            idx = smallest
    return min_k, min_r, min_c, new_size


@njit
def _queue_push(
    q_rows: np.ndarray,
    q_cols: np.ndarray,
    tail: int,
    count: int,
    cap: int,
    row: int,
    col: int,
) -> Tuple[int, int]:
    """Push ``(row, col)`` into circular FIFO queue, returning ``(new_tail, new_count)``."""
    q_rows[tail] = row
    q_cols[tail] = col
    new_tail = tail + 1
    if new_tail == cap:
        new_tail = 0
    return new_tail, count + 1


@njit
def _queue_pop(
    q_rows: np.ndarray,
    q_cols: np.ndarray,
    head: int,
    count: int,
    cap: int,
) -> Tuple[int, int, int, int]:
    """Pop ``(row, col, new_head, new_count)`` from circular FIFO queue."""
    r = int(q_rows[head])
    c = int(q_cols[head])
    new_head = head + 1
    if new_head == cap:
        new_head = 0
    return r, c, new_head, count - 1


@njit
def priority_flood_kernel(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Barnes et al. (2014) Priority-Flood + Pit FIFO Queue depression filling
    in ``float64`` without epsilon increments.
    """
    # pylint: disable=too-many-locals,too-many-branches
    height, width = dem.shape
    z_out = dem.astype(np.float64).copy()
    visited = ~valid

    cap = height * width + 1
    h_keys = np.empty(cap, dtype=np.float64)
    h_rows = np.empty(cap, dtype=np.int32)
    h_cols = np.empty(cap, dtype=np.int32)
    h_size = 0

    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            is_boundary = r == 0 or r == height - 1 or c == 0 or c == width - 1
            if not is_boundary:
                for k in range(8):
                    nr = r + int(D8_DR[k])
                    nc = c + int(D8_DC[k])
                    if not valid[nr, nc]:
                        is_boundary = True
                        break
            if is_boundary:
                visited[r, c] = True
                h_size = _heap_push(
                    h_keys, h_rows, h_cols, h_size, z_out[r, c], r, c
                )

    while q_count > 0 or h_size > 0:
        if q_count > 0:
            r, c, q_head, q_count = _queue_pop(
                q_rows, q_cols, q_head, q_count, cap
            )
        else:
            _, r, c, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)

        z_curr = z_out[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True
            z_nbr = dem[nr, nc]
            if z_nbr <= z_curr:
                z_out[nr, nc] = z_curr
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )
            else:
                z_out[nr, nc] = z_nbr
                h_size = _heap_push(
                    h_keys, h_rows, h_cols, h_size, z_nbr, nr, nc
                )
    return z_out


@njit
def single_cell_pits_kernel(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return boolean mask of valid cells whose valid 8-neighbours are all strictly higher."""
    height, width = dem.shape
    pits = np.zeros((height, width), dtype=np.bool_)
    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                continue
            z0 = dem[r, c]
            n_valid = 0
            all_higher = True
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if valid[nr, nc]:
                    n_valid += 1
                    if dem[nr, nc] <= z0:
                        all_higher = False
                        break
            if all_higher and n_valid == 8:
                pits[r, c] = True
    return pits


@njit
def _grow_heap_arrays(
    keys: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Double min-heap buffer capacity inside Numba."""
    new_cap = cap * 2
    n_keys = np.empty(new_cap, dtype=np.float64)
    n_rows = np.empty(new_cap, dtype=np.int32)
    n_cols = np.empty(new_cap, dtype=np.int32)
    n_keys[:cap] = keys
    n_rows[:cap] = rows
    n_cols[:cap] = cols
    return n_keys, n_rows, n_cols, new_cap


@njit
def breach_depressions_kernel(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    max_depth: float,
    has_max_depth: bool,
    max_length: int,
    has_max_length: bool,
) -> np.ndarray:
    """
    Lindsay (2016) constrained least-cost depression breaching kernel.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    height, width = dem.shape
    z_out = dem.astype(np.float64).copy()
    diag_d = math.hypot(dx, dy)

    # Locate candidate pit cells (interior valid cells with no strictly lower neighbour)
    max_pits = height * width
    pit_z = np.empty(max_pits, dtype=np.float64)
    pit_r = np.empty(max_pits, dtype=np.int32)
    pit_c = np.empty(max_pits, dtype=np.int32)
    n_pits = 0

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if not valid[r, c]:
                continue
            z0 = z_out[r, c]
            has_lower = False
            touches_nd = False
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if not valid[nr, nc]:
                    touches_nd = True
                    break
                if z_out[nr, nc] < z0:
                    has_lower = True
                    break
            if (not has_lower) and (not touches_nd):
                pit_z[n_pits] = z0
                pit_r[n_pits] = r
                pit_c[n_pits] = c
                n_pits += 1

    if n_pits == 0:
        return z_out

    order = np.argsort(pit_z[:n_pits])

    # Persistent local-Dijkstra buffers (sparse reset via touched list)
    best_cost = np.full((height, width), 1e300, dtype=np.float64)
    closed = np.zeros((height, width), dtype=np.bool_)
    parent_r = np.full((height, width), -1, dtype=np.int32)
    parent_c = np.full((height, width), -1, dtype=np.int32)
    steps_arr = np.zeros((height, width), dtype=np.int32)

    touched_r = np.empty(height * width, dtype=np.int32)
    touched_c = np.empty(height * width, dtype=np.int32)

    h_cap = max(1024, min(height * width * 2, 65536))
    h_keys = np.empty(h_cap, dtype=np.float64)
    h_rows = np.empty(h_cap, dtype=np.int32)
    h_cols = np.empty(h_cap, dtype=np.int32)

    for idx_i in range(n_pits):
        p_idx = int(order[idx_i])
        r0 = int(pit_r[p_idx])
        c0 = int(pit_c[p_idx])
        z0 = z_out[r0, c0]

        # Skip if earlier breach already gave this cell a strictly lower neighbour
        already_drained = False
        for k in range(8):
            nr = r0 + int(D8_DR[k])
            nc = c0 + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                if z_out[nr, nc] < z0:
                    already_drained = True
                    break
        if already_drained:
            continue

        h_size = 0
        n_touched = 0

        best_cost[r0, c0] = 0.0
        steps_arr[r0, c0] = 0
        parent_r[r0, c0] = -1
        parent_c[r0, c0] = -1
        touched_r[n_touched] = r0
        touched_c[n_touched] = c0
        n_touched += 1
        h_size = _heap_push(h_keys, h_rows, h_cols, h_size, 0.0, r0, c0)

        found = False
        rt = -1
        ct = -1

        while h_size > 0:
            cost_u, r, c, h_size = _heap_pop(h_keys, h_rows, h_cols, h_size)
            if closed[r, c]:
                continue
            if cost_u > best_cost[r, c]:
                continue
            closed[r, c] = True

            if not (r == r0 and c == c0):
                z_curr = z_out[r, c]
                is_edge_outlet = r == 0 or r == height - 1 or c == 0 or c == width - 1
                if not is_edge_outlet:
                    for k in range(8):
                        if not valid[r + int(D8_DR[k]), c + int(D8_DC[k])]:
                            is_edge_outlet = True
                            break
                if z_curr < z0 or (is_edge_outlet and z_curr <= z0):
                    found = True
                    rt = r
                    ct = c
                    break

            next_step = int(steps_arr[r, c]) + 1
            if has_max_length and next_step > max_length:
                continue

            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not valid[nr, nc] or closed[nr, nc]:
                    continue

                dz = z_out[nr, nc] - z0
                if has_max_depth and dz > max_depth:
                    continue

                s_dist = _step_dist(k, dx, dy, diag_d)
                excess = dz if dz > 0.0 else 0.0
                # Tiny step-distance tie-breaker prefers shorter breach paths
                edge_cost = (excess + 1e-12) * s_dist
                new_cost = cost_u + edge_cost

                if new_cost < best_cost[nr, nc]:
                    if best_cost[nr, nc] > 1e299:
                        touched_r[n_touched] = nr
                        touched_c[n_touched] = nc
                        n_touched += 1
                    best_cost[nr, nc] = new_cost
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    steps_arr[nr, nc] = next_step
                    if h_size >= h_cap:
                        h_keys, h_rows, h_cols, h_cap = _grow_heap_arrays(
                            h_keys, h_rows, h_cols, h_cap
                        )
                    h_size = _heap_push(
                        h_keys, h_rows, h_cols, h_size, new_cost, nr, nc
                    )

        if found:
            path_len = int(steps_arr[rt, ct])
            if path_len > 0:
                zt = z_out[rt, ct]
                curr_r = rt
                curr_c = ct
                step_k = path_len
                while curr_r != -1:
                    z_carved = z0 - (float(step_k) / float(path_len)) * (z0 - zt)
                    if z_carved < z_out[curr_r, curr_c]:
                        z_out[curr_r, curr_c] = z_carved
                    pr_val = int(parent_r[curr_r, curr_c])
                    pc_val = int(parent_c[curr_r, curr_c])
                    curr_r = pr_val
                    curr_c = pc_val
                    step_k -= 1

        # Sparse cleanup
        for t_i in range(n_touched):
            tr = int(touched_r[t_i])
            tc = int(touched_c[t_i])
            best_cost[tr, tc] = 1e300
            closed[tr, tc] = False
            parent_r[tr, tc] = -1
            parent_c[tr, tc] = -1
            steps_arr[tr, tc] = 0

    return z_out


@njit
def resolve_flats_kernel(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Barnes et al. (2014) double-gradient flat resolution surface ``G``.

    Returns ``int32`` grid with positive routing values on drainable flat cells
    (decreasing strictly toward outlets), ``0`` on non-flat valid cells (and
    undrainable closed flats), and ``-1`` on nodata cells.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    height, width = dem.shape
    has_lower = np.zeros((height, width), dtype=np.bool_)
    is_flat = np.zeros((height, width), dtype=np.bool_)
    g_high = np.zeros((height, width), dtype=np.int32)
    g_low = np.zeros((height, width), dtype=np.int32)
    out_g = np.zeros((height, width), dtype=np.int32)

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                out_g[r, c] = -1
                continue
            z0 = dem[r, c]
            low = False
            eq = False
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not valid[nr, nc]:
                    continue
                zn = dem[nr, nc]
                if zn < z0:
                    low = True
                    break
                if zn == z0:
                    eq = True
            has_lower[r, c] = low
            if (not low) and eq:
                is_flat[r, c] = True

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)

    # Pass 1: BFS away from strictly higher terrain
    q_head = 0
    q_tail = 0
    q_count = 0
    for r in range(height):
        for c in range(width):
            if not is_flat[r, c]:
                continue
            z0 = dem[r, c]
            near_high = False
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                    if dem[nr, nc] > z0:
                        near_high = True
                        break
            if near_high:
                g_high[r, c] = 1
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    h_max = 0
    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        gh = int(g_high[r, c])
        if gh > h_max:
            h_max = gh
        z0 = dem[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width:
                if is_flat[nr, nc] and dem[nr, nc] == z0 and g_high[nr, nc] == 0:
                    g_high[nr, nc] = gh + 1
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr, nc
                    )

    # Pass 2: BFS inward from equal-elevation outlet cells (has_lower == True)
    q_head = 0
    q_tail = 0
    q_count = 0
    for r in range(height):
        for c in range(width):
            if not is_flat[r, c]:
                continue
            z0 = dem[r, c]
            near_outlet = False
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                    if dem[nr, nc] == z0 and has_lower[nr, nc]:
                        near_outlet = True
                        break
            if near_outlet:
                g_low[r, c] = 1
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        gl = int(g_low[r, c])
        z0 = dem[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width:
                if is_flat[nr, nc] and dem[nr, nc] == z0 and g_low[nr, nc] == 0:
                    g_low[nr, nc] = gl + 1
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr, nc
                    )

    # Combine gradients so values decrease strictly toward outlets (0 at outlet)
    for r in range(height):
        for c in range(width):
            if is_flat[r, c] and g_low[r, c] > 0:
                out_g[r, c] = 1 + (h_max - int(g_high[r, c])) + 2 * int(g_low[r, c])

    return out_g


@njit(parallel=True)
def flow_dir_d8_njit(
    dem: np.ndarray, valid: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Compute D8 bitmask flow directions in parallel using Numba ``prange``."""
    # pylint: disable=too-many-locals
    height, width = dem.shape
    fdir = np.full((height, width), 255, dtype=np.uint8)
    diag_d = math.hypot(dx, dy)

    for r in prange(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            z0 = dem[r, c]
            best_slope = 0.0
            best_code = np.uint8(0)
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not valid[nr, nc]:
                    continue
                dz = z0 - dem[nr, nc]
                if dz > 0.0:
                    s = dz / _step_dist(k, dx, dy, diag_d)
                    if s > best_slope:
                        best_slope = s
                        best_code = D8_CODES[k]
            fdir[r, c] = best_code
    return fdir


def flow_dir_d8_numpy(
    dem: np.ndarray, valid: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Vectorised NumPy D8 bitmask flow direction calculation."""
    height, width = dem.shape
    diag_d = math.hypot(dx, dy)
    slopes = np.zeros((8, height, width), dtype=np.float64)

    dem_pad = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    val_pad = np.pad(valid, 1, mode="constant", constant_values=False)

    for k in range(8):
        dr = int(D8_DR[k])
        dc = int(D8_DC[k])
        r0 = 1 + dr
        c0 = 1 + dc
        nbr_z = dem_pad[r0:r0 + height, c0:c0 + width]
        nbr_v = val_pad[r0:r0 + height, c0:c0 + width]
        step = dx if k in (0, 4) else (dy if k in (2, 6) else diag_d)
        ok = valid & nbr_v
        dz = np.where(ok, dem - nbr_z, 0.0)
        slopes[k] = np.where(dz > 0.0, dz / step, 0.0)

    best_idx = np.argmax(slopes, axis=0)
    best_max = np.take_along_axis(slopes, best_idx[None, ...], axis=0)[0]
    fdir = np.where(best_max > 0.0, D8_CODES[best_idx], np.uint8(0))
    fdir = np.where(valid, fdir, np.uint8(255))
    return fdir.astype(np.uint8, copy=False)


@njit
def apply_flat_routing_d8_kernel(
    dem: np.ndarray,
    valid: np.ndarray,
    fdir: np.ndarray,
    g_grid: np.ndarray,
    dx: float,
    dy: float,
) -> None:
    """Route flat cells (``fdir == 0`` & ``g_grid > 0``) down the Barnes gradient."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = dem.shape
    diag_d = math.hypot(dx, dy)
    for r in range(height):
        for c in range(width):
            if not valid[r, c] or fdir[r, c] != 0 or g_grid[r, c] <= 0:
                continue
            z0 = dem[r, c]
            g0 = float(g_grid[r, c])
            best_drop = 0.0
            best_code = np.uint8(0)
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not valid[nr, nc] or dem[nr, nc] != z0:
                    continue
                gn = int(g_grid[nr, nc])
                if gn >= 0 and gn < g_grid[r, c]:
                    drop = (g0 - float(gn)) / _step_dist(k, dx, dy, diag_d)
                    if drop > best_drop:
                        best_drop = drop
                        best_code = D8_CODES[k]
            fdir[r, c] = best_code


@njit(parallel=True)
def flow_dir_dinf_njit(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    nd_val: float,
) -> np.ndarray:
    """Tarboton (1997) D-Infinity continuous flow direction in parallel."""
    # pylint: disable=too-many-locals,too-many-branches
    height, width = dem.shape
    ang = np.full((height, width), nd_val, dtype=np.float32)
    diag_d = math.hypot(dx, dy)
    two_pi = 2.0 * math.pi
    half_pi = 0.5 * math.pi

    # Facet lookup: (e1_dr, e1_dc, e2_dr, e2_dc, d1, d2, a_c, a_f)
    e1_dr = np.array([0, -1, -1, 0, 0, 1, 1, 0], dtype=np.int32)
    e1_dc = np.array([1, 0, 0, -1, -1, 0, 0, 1], dtype=np.int32)
    e2_dr = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.int32)
    e2_dc = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=np.int32)
    a_c = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0], dtype=np.float64)
    a_f = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)

    for r in prange(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            e0 = dem[r, c]
            best_s = 0.0
            best_ang = -1.0

            for k in range(8):
                r1 = r + int(e1_dr[k])
                c1 = c + int(e1_dc[k])
                r2 = r + int(e2_dr[k])
                c2 = c + int(e2_dc[k])

                v1 = 0 <= r1 < height and 0 <= c1 < width and valid[r1, c1]
                v2 = 0 <= r2 < height and 0 <= c2 < width and valid[r2, c2]

                d1 = dx if e1_dr[k] == 0 else dy
                d2 = dy if e1_dr[k] == 0 else dx

                if v1 and v2:
                    e1 = dem[r1, c1]
                    e2 = dem[r2, c2]
                    s1 = (e0 - e1) / d1
                    s2 = (e1 - e2) / d2
                    r_max = math.atan2(d2, d1)
                    r_ang = math.atan2(s2, s1)
                    if r_ang < 0.0:
                        r_ang = 0.0
                        s_val = s1
                    elif r_ang > r_max:
                        r_ang = r_max
                        s_val = (e0 - e2) / diag_d
                    else:
                        s_val = math.hypot(s1, s2)
                    if s_val > best_s:
                        best_s = s_val
                        theta = (a_c[k] * half_pi + a_f[k] * r_ang) % two_pi
                        if theta >= two_pi - 1e-12:
                            theta = 0.0
                        best_ang = theta
                elif v1:
                    s_val = (e0 - dem[r1, c1]) / d1
                    if s_val > best_s:
                        best_s = s_val
                        theta = (a_c[k] * half_pi) % two_pi
                        if theta >= two_pi - 1e-12:
                            theta = 0.0
                        best_ang = theta
                elif v2:
                    s_val = (e0 - dem[r2, c2]) / diag_d
                    if s_val > best_s:
                        best_s = s_val
                        r_max = math.atan2(d2, d1)
                        theta = (a_c[k] * half_pi + a_f[k] * r_max) % two_pi
                        if theta >= two_pi - 1e-12:
                            theta = 0.0
                        best_ang = theta

            ang[r, c] = np.float32(best_ang)
    return ang


def flow_dir_dinf_numpy(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    nd_val: float,
) -> np.ndarray:
    """Vectorised NumPy Tarboton (1997) D-Infinity flow direction."""
    # pylint: disable=too-many-locals
    height, width = dem.shape
    diag_d = math.hypot(dx, dy)
    two_pi = 2.0 * math.pi
    half_pi = 0.5 * math.pi

    e1_dr = [0, -1, -1, 0, 0, 1, 1, 0]
    e1_dc = [1, 0, 0, -1, -1, 0, 0, 1]
    e2_dr = [-1, -1, -1, -1, 1, 1, 1, 1]
    e2_dc = [1, 1, -1, -1, -1, -1, 1, 1]
    a_c = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    a_f = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

    dem_pad = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    val_pad = np.pad(valid, 1, mode="constant", constant_values=False)

    best_s = np.zeros((height, width), dtype=np.float64)
    best_ang = np.full((height, width), -1.0, dtype=np.float64)

    for k in range(8):
        d1 = dx if e1_dr[k] == 0 else dy
        d2 = dy if e1_dr[k] == 0 else dx
        r_max = math.atan2(d2, d1)

        r1_off = 1 + e1_dr[k]
        c1_off = 1 + e1_dc[k]
        r2_off = 1 + e2_dr[k]
        c2_off = 1 + e2_dc[k]
        e1 = dem_pad[r1_off:r1_off + height, c1_off:c1_off + width]
        v1 = val_pad[r1_off:r1_off + height, c1_off:c1_off + width]
        e2 = dem_pad[r2_off:r2_off + height, c2_off:c2_off + width]
        v2 = val_pad[r2_off:r2_off + height, c2_off:c2_off + width]

        both = valid & v1 & v2
        only1 = valid & v1 & (~v2)
        only2 = valid & (~v1) & v2

        s1 = np.where(valid & v1, (dem - e1) / d1, 0.0)
        s2 = np.where(both, (e1 - e2) / d2, 0.0)
        r_ang = np.arctan2(s2, s1)

        # Clamp within [0, r_max]
        neg_r = both & (r_ang < 0.0)
        hi_r = both & (r_ang > r_max)
        mid_r = both & (~neg_r) & (~hi_r)

        s_val = np.zeros((height, width), dtype=np.float64)
        r_eff = np.zeros((height, width), dtype=np.float64)

        s_val[neg_r] = s1[neg_r]
        r_eff[neg_r] = 0.0

        s_val[hi_r] = ((dem - e2) / diag_d)[hi_r]
        r_eff[hi_r] = r_max

        s_val[mid_r] = np.hypot(s1[mid_r], s2[mid_r])
        r_eff[mid_r] = r_ang[mid_r]

        s_val[only1] = s1[only1]
        r_eff[only1] = 0.0

        s_val[only2] = ((dem - e2) / diag_d)[only2]
        r_eff[only2] = r_max

        theta = np.mod(a_c[k] * half_pi + a_f[k] * r_eff, two_pi)
        theta[theta >= two_pi - 1e-12] = 0.0

        upd = valid & (v1 | v2) & (s_val > best_s)
        best_s[upd] = s_val[upd]
        best_ang[upd] = theta[upd]

    out = np.where(valid, best_ang, nd_val)
    return out.astype(np.float32)


@njit(parallel=True)
def flow_dir_mfd_njit(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    exponent: float,
    nd_val: float,
) -> np.ndarray:
    """Quinn et al. (1991) 8-band Multiple Flow Direction in parallel."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = dem.shape
    mfd = np.full((8, height, width), nd_val, dtype=np.float32)
    diag_d = math.hypot(dx, dy)

    for r in prange(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            z0 = dem[r, c]
            w_sum = 0.0
            w_arr = np.zeros(8, dtype=np.float64)
            for k in range(8):
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if not valid[nr, nc]:
                    continue
                dz = z0 - dem[nr, nc]
                if dz > 0.0:
                    tan_b = dz / _step_dist(k, dx, dy, diag_d)
                    w = (tan_b ** exponent) * float(D8_CONTOUR_L[k])
                    w_arr[k] = w
                    w_sum += w
            if w_sum > 0.0:
                for k in range(8):
                    mfd[k, r, c] = np.float32(w_arr[k] / w_sum)
            else:
                for k in range(8):
                    mfd[k, r, c] = np.float32(0.0)
    return mfd


def flow_dir_mfd_numpy(
    dem: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    exponent: float,
    nd_val: float,
) -> np.ndarray:
    """Vectorised NumPy Quinn et al. (1991) 8-band Multiple Flow Direction."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    height, width = dem.shape
    diag_d = math.hypot(dx, dy)
    weights = np.zeros((8, height, width), dtype=np.float64)

    dem_pad = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    val_pad = np.pad(valid, 1, mode="constant", constant_values=False)

    for k in range(8):
        dr = int(D8_DR[k])
        dc = int(D8_DC[k])
        r_off = 1 + dr
        c_off = 1 + dc
        nbr_z = dem_pad[r_off:r_off + height, c_off:c_off + width]
        nbr_v = val_pad[r_off:r_off + height, c_off:c_off + width]
        step = dx if k in (0, 4) else (dy if k in (2, 6) else diag_d)
        ok = valid & nbr_v
        dz = np.where(ok, dem - nbr_z, 0.0)
        tan_b = np.where(dz > 0.0, dz / step, 0.0)
        weights[k] = np.where(dz > 0.0, (tan_b ** exponent) * float(D8_CONTOUR_L[k]), 0.0)

    w_sum = np.sum(weights, axis=0)
    fracs = np.divide(
        weights,
        w_sum[None, ...],
        out=np.zeros_like(weights),
        where=w_sum[None, ...] > 0.0,
    )
    out = np.where(valid[None, ...], fracs, nd_val)
    return out.astype(np.float32)


@njit
def _count_unreleased(valid: np.ndarray, in_deg: np.ndarray) -> int:
    """
    Count the valid cells that a Kahn's topological sort never released
    (in-degree still above zero): they sit on or downstream of a cycle in
    the flow graph, so their accumulation is incomplete.
    """
    height, width = in_deg.shape
    n_left = 0
    for r in range(height):
        for c in range(width):
            if valid[r, c] and in_deg[r, c] > 0:
                n_left += 1
    return n_left


@njit
def flow_acc_d8_kernel(
    flow_dir: np.ndarray, valid: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    O(N) Kahn's topological sort D8 flow accumulation.

    Returns the accumulation and the number of valid cells never released by
    the sort (non-zero only if the flow graph has a cycle).
    """
    # pylint: disable=too-many-locals,too-many-branches
    height, width = flow_dir.shape
    in_deg = np.zeros((height, width), dtype=np.int32)
    acc = np.zeros((height, width), dtype=np.float64)

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            acc[r, c] = weights[r, c]
            dr, dc = _d8_to_offset(int(flow_dir[r, c]))
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                in_deg[nr, nc] += 1

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and in_deg[r, c] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        dr, dc = _d8_to_offset(int(flow_dir[r, c]))
        if dr == 0 and dc == 0:
            continue
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
            acc[nr, nc] += acc[r, c]
            in_deg[nr, nc] -= 1
            if in_deg[nr, nc] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )

    return acc, _count_unreleased(valid, in_deg)


# Angular tolerance (radians) for snapping a D-Infinity angle onto a facet
# boundary (a cardinal or diagonal direction). The angles are stored as
# float32, which can't represent the boundary angles exactly (e.g. pi/2 and
# pi round up by ~4e-8 and ~9e-8 rad), so without snapping a tiny share of
# the flow (alpha ~ 1e-8) is sent to the other neighbour of the facet, which
# may be higher than the cell; those uphill edges form cycles in the flow
# graph. 1e-5 rad is far above float32 rounding (~2.4e-7 rad at 2*pi) and far
# below any real facet split (a 1e-5 rad angle sends a share of ~1.3e-5).
DINF_SNAP_TOL = 1e-5


@njit
def _dinf_targets(
    ang: float, dx: float, dy: float
) -> Tuple[int, int, int, int, float, float]:
    """
    Resolve D-Infinity CCW angle ``ang in [0, 2*pi)`` into two neighbour
    offsets ``(dr1, dc1, dr2, dc2, w1, w2)``.

    Angles within :data:`DINF_SNAP_TOL` radians of a facet boundary (a
    cardinal or diagonal direction for the pixel geometry) are snapped onto
    it, so all the flow goes to that one neighbour (``w2 = 0``).
    """
    # pylint: disable=too-many-branches,too-many-statements
    phi = math.atan2(dy, dx)
    half_pi = 0.5 * math.pi
    pi = math.pi
    three_half_pi = 1.5 * math.pi
    two_pi = 2.0 * math.pi

    if ang < 0.0:
        return 0, 0, 0, 0, 0.0, 0.0

    # Normalise near 2*pi
    if ang >= two_pi - DINF_SNAP_TOL:
        ang = 0.0

    # ``width`` is the angular width of the facet, so ``alpha * width`` is
    # the angle from the first neighbour's direction.
    if ang <= phi:
        width = phi
        alpha = ang / width
        dr1, dc1, dr2, dc2 = 0, 1, -1, 1
    elif ang <= half_pi:
        width = half_pi - phi
        alpha = (half_pi - ang) / width
        dr1, dc1, dr2, dc2 = -1, 0, -1, 1
    elif ang <= pi - phi:
        width = half_pi - phi
        alpha = (ang - half_pi) / width
        dr1, dc1, dr2, dc2 = -1, 0, -1, -1
    elif ang <= pi:
        width = phi
        alpha = (pi - ang) / width
        dr1, dc1, dr2, dc2 = 0, -1, -1, -1
    elif ang <= pi + phi:
        width = phi
        alpha = (ang - pi) / width
        dr1, dc1, dr2, dc2 = 0, -1, 1, -1
    elif ang <= three_half_pi:
        width = half_pi - phi
        alpha = (three_half_pi - ang) / width
        dr1, dc1, dr2, dc2 = 1, 0, 1, -1
    elif ang <= two_pi - phi:
        width = half_pi - phi
        alpha = (ang - three_half_pi) / width
        dr1, dc1, dr2, dc2 = 1, 0, 1, 1
    else:
        width = phi
        alpha = (two_pi - ang) / width
        dr1, dc1, dr2, dc2 = 0, 1, 1, 1

    if alpha * width <= DINF_SNAP_TOL:
        return dr1, dc1, 0, 0, 1.0, 0.0
    if (1.0 - alpha) * width <= DINF_SNAP_TOL:
        return dr2, dc2, 0, 0, 1.0, 0.0
    return dr1, dc1, dr2, dc2, 1.0 - alpha, alpha


@njit
def flow_acc_dinf_kernel(
    flow_dir: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, int]:
    """
    O(N) Kahn's topological sort D-Infinity flow accumulation.

    Returns the accumulation and the number of valid cells never released by
    the sort (non-zero only if the flow graph has a cycle).
    """
    # pylint: disable=too-many-locals,too-many-branches
    height, width = flow_dir.shape
    in_deg = np.zeros((height, width), dtype=np.int32)
    acc = np.zeros((height, width), dtype=np.float64)

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            acc[r, c] = weights[r, c]
            ang = float(flow_dir[r, c])
            if ang < 0.0:
                continue
            dr1, dc1, dr2, dc2, w1, w2 = _dinf_targets(ang, dx, dy)
            if w1 > 0.0:
                nr1, nc1 = r + dr1, c + dc1
                if 0 <= nr1 < height and 0 <= nc1 < width and valid[nr1, nc1]:
                    in_deg[nr1, nc1] += 1
            if w2 > 0.0:
                nr2, nc2 = r + dr2, c + dc2
                if 0 <= nr2 < height and 0 <= nc2 < width and valid[nr2, nc2]:
                    in_deg[nr2, nc2] += 1

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and in_deg[r, c] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        ang = float(flow_dir[r, c])
        if ang < 0.0:
            continue
        dr1, dc1, dr2, dc2, w1, w2 = _dinf_targets(ang, dx, dy)
        val_acc = acc[r, c]
        if w1 > 0.0:
            nr1, nc1 = r + dr1, c + dc1
            if 0 <= nr1 < height and 0 <= nc1 < width and valid[nr1, nc1]:
                acc[nr1, nc1] += val_acc * w1
                in_deg[nr1, nc1] -= 1
                if in_deg[nr1, nc1] == 0:
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr1, nc1
                    )
        if w2 > 0.0:
            nr2, nc2 = r + dr2, c + dc2
            if 0 <= nr2 < height and 0 <= nc2 < width and valid[nr2, nc2]:
                acc[nr2, nc2] += val_acc * w2
                in_deg[nr2, nc2] -= 1
                if in_deg[nr2, nc2] == 0:
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr2, nc2
                    )

    return acc, _count_unreleased(valid, in_deg)


@njit
def flow_acc_mfd_kernel(
    flow_dir_mfd: np.ndarray, valid: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    O(N) Kahn's topological sort 8-band MFD flow accumulation.

    Returns the accumulation and the number of valid cells never released by
    the sort (non-zero only if the flow graph has a cycle).
    """
    # pylint: disable=too-many-locals,too-many-branches
    _, height, width = flow_dir_mfd.shape
    in_deg = np.zeros((height, width), dtype=np.int32)
    acc = np.zeros((height, width), dtype=np.float64)

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            acc[r, c] = weights[r, c]
            for k in range(8):
                frac = float(flow_dir_mfd[k, r, c])
                if frac > 0.0:
                    nr = r + int(D8_DR[k])
                    nc = c + int(D8_DC[k])
                    if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                        in_deg[nr, nc] += 1

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and in_deg[r, c] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        val_acc = acc[r, c]
        for k in range(8):
            frac = float(flow_dir_mfd[k, r, c])
            if frac > 0.0:
                nr = r + int(D8_DR[k])
                nc = c + int(D8_DC[k])
                if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                    acc[nr, nc] += val_acc * frac
                    in_deg[nr, nc] -= 1
                    if in_deg[nr, nc] == 0:
                        q_tail, q_count = _queue_push(
                            q_rows, q_cols, q_tail, q_count, cap, nr, nc
                        )

    return acc, _count_unreleased(valid, in_deg)


@njit
def flow_length_downstream_kernel(
    flow_dir: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Downstream flow length from each cell to its basin outlet via reverse BFS."""
    # pylint: disable=too-many-locals,too-many-branches
    height, width = flow_dir.shape
    diag_d = math.hypot(dx, dy)
    out_len = np.full((height, width), np.nan, dtype=np.float64)

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            dr, dc = _d8_to_offset(int(flow_dir[r, c]))
            is_outlet = False
            if dr == 0 and dc == 0:
                is_outlet = True
            else:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width or not valid[nr, nc]:
                    is_outlet = True
            if is_outlet:
                out_len[r, c] = 0.0
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        curr_len = out_len[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if not valid[nr, nc]:
                continue
            if flow_dir[nr, nc] == D8_REV_CODES[k] and math.isnan(out_len[nr, nc]):
                step_c = _step_dist(k, dx, dy, diag_d) * weights[nr, nc]
                out_len[nr, nc] = curr_len + step_c
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )

    return out_len


@njit
def flow_length_upstream_kernel(
    flow_dir: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Longest upstream flow path length from divide to each cell via Kahn's sort."""
    # pylint: disable=too-many-locals,too-many-branches
    height, width = flow_dir.shape
    diag_d = math.hypot(dx, dy)
    in_deg = np.zeros((height, width), dtype=np.int32)
    out_len = np.full((height, width), np.nan, dtype=np.float64)

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                continue
            out_len[r, c] = 0.0
            dr, dc = _d8_to_offset(int(flow_dir[r, c]))
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                in_deg[nr, nc] += 1

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and in_deg[r, c] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        idx = _d8_to_idx(int(flow_dir[r, c]))
        if idx < 0:
            continue
        nr = r + int(D8_DR[idx])
        nc = c + int(D8_DC[idx])
        if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
            cand = out_len[r, c] + _step_dist(idx, dx, dy, diag_d) * weights[r, c]
            if cand > out_len[nr, nc]:
                out_len[nr, nc] = cand
            in_deg[nr, nc] -= 1
            if in_deg[nr, nc] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )

    return out_len


@njit
def flow_distance_horiz_kernel(
    flow_dir: np.ndarray,
    valid: np.ndarray,
    streams: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Horizontal overland flow distance to nearest downstream stream cell."""
    # pylint: disable=too-many-locals
    height, width = flow_dir.shape
    diag_d = math.hypot(dx, dy)
    dist = np.full((height, width), np.nan, dtype=np.float64)

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and streams[r, c] == 1:
                dist[r, c] = 0.0
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        curr_d = dist[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                if flow_dir[nr, nc] == D8_REV_CODES[k] and math.isnan(dist[nr, nc]):
                    dist[nr, nc] = curr_d + _step_dist(k, dx, dy, diag_d)
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr, nc
                    )

    return dist


@njit
def flow_distance_vert_kernel(
    flow_dir: np.ndarray,
    valid: np.ndarray,
    streams: np.ndarray,
    dem: np.ndarray,
) -> np.ndarray:
    """Vertical height above nearest drainage (HAND / vertical flow distance)."""
    # pylint: disable=too-many-locals
    height, width = flow_dir.shape
    drain_elev = np.full((height, width), np.nan, dtype=np.float64)

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid[r, c] and streams[r, c] == 1:
                drain_elev[r, c] = dem[r, c]
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        z_drain = drain_elev[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width and valid[nr, nc]:
                if flow_dir[nr, nc] == D8_REV_CODES[k] and math.isnan(drain_elev[nr, nc]):
                    drain_elev[nr, nc] = z_drain
                    q_tail, q_count = _queue_push(
                        q_rows, q_cols, q_tail, q_count, cap, nr, nc
                    )

    out_v = np.full((height, width), np.nan, dtype=np.float64)
    for r in range(height):
        for c in range(width):
            if valid[r, c] and not math.isnan(drain_elev[r, c]):
                diff = dem[r, c] - drain_elev[r, c]
                out_v[r, c] = diff if diff > 0.0 else 0.0
    return out_v


@njit(parallel=True)
def extract_streams_njit(
    acc: np.ndarray, valid: np.ndarray, threshold: float, nd_val: int
) -> np.ndarray:
    """Threshold flow accumulation in parallel into binary stream mask."""
    height, width = acc.shape
    out = np.full((height, width), nd_val, dtype=np.uint8)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                out[r, c] = np.uint8(1) if acc[r, c] >= threshold else np.uint8(0)
    return out


def extract_streams_numpy(
    acc: np.ndarray, valid: np.ndarray, threshold: float, nd_val: int
) -> np.ndarray:
    """Vectorised NumPy binary stream extraction."""
    out = np.full(acc.shape, nd_val, dtype=np.uint8)
    out[valid] = np.where(acc[valid] >= threshold, np.uint8(1), np.uint8(0))
    return out


@njit
def stream_in_degree_kernel(
    flow_dir: np.ndarray, streams: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Compute upstream stream in-degree for every valid stream pixel."""
    height, width = streams.shape
    in_deg = np.zeros((height, width), dtype=np.int32)
    for r in range(height):
        for c in range(width):
            if not valid[r, c] or streams[r, c] != 1:
                continue
            dr, dc = _d8_to_offset(int(flow_dir[r, c]))
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if valid[nr, nc] and streams[nr, nc] == 1:
                    in_deg[nr, nc] += 1
    return in_deg


@njit
def stream_link_kernel(
    flow_dir: np.ndarray, streams: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment stream network into unique topological reach IDs (``1..K``).
    Starts a new reach at every headwater (``in_deg == 0``) and every
    confluence junction (``in_deg >= 2``).
    """
    # pylint: disable=too-many-locals,too-many-branches
    height, width = streams.shape
    in_deg = stream_in_degree_kernel(flow_dir, streams, valid)
    link_grid = np.zeros((height, width), dtype=np.int32)
    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                link_grid[r, c] = -1

    next_id = 1
    for r in range(height):
        for c in range(width):
            if not valid[r, c] or streams[r, c] != 1:
                continue
            if in_deg[r, c] == 0 or in_deg[r, c] >= 2:
                curr_r = r
                curr_c = c
                link_id = next_id
                next_id += 1
                while True:
                    if link_grid[curr_r, curr_c] > 0:
                        break
                    link_grid[curr_r, curr_c] = link_id
                    dr, dc = _d8_to_offset(int(flow_dir[curr_r, curr_c]))
                    if dr == 0 and dc == 0:
                        break
                    nr = curr_r + dr
                    nc = curr_c + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        break
                    if not valid[nr, nc] or streams[nr, nc] != 1:
                        break
                    if in_deg[nr, nc] >= 2:
                        break
                    curr_r = nr
                    curr_c = nc

    return link_grid, in_deg


@njit
def stream_order_kernel(
    flow_dir: np.ndarray,
    streams: np.ndarray,
    valid: np.ndarray,
    is_strahler: bool,
) -> np.ndarray:
    """Topological Strahler or Shreve stream ordering via Kahn's BFS."""
    # pylint: disable=too-many-locals,too-many-branches
    height, width = streams.shape
    in_deg = stream_in_degree_kernel(flow_dir, streams, valid)
    rem_deg = in_deg.copy()
    order_grid = np.zeros((height, width), dtype=np.int32)
    max_inc = np.zeros((height, width), dtype=np.int32)
    cnt_max = np.zeros((height, width), dtype=np.int32)

    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if not valid[r, c]:
                order_grid[r, c] = -1
            elif streams[r, c] == 1 and in_deg[r, c] == 0:
                order_grid[r, c] = 1
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        ord_u = int(order_grid[r, c])
        dr, dc = _d8_to_offset(int(flow_dir[r, c]))
        if dr == 0 and dc == 0:
            continue
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= height or nc < 0 or nc >= width:
            continue
        if not valid[nr, nc] or streams[nr, nc] != 1:
            continue

        if is_strahler:
            if ord_u > max_inc[nr, nc]:
                max_inc[nr, nc] = ord_u
                cnt_max[nr, nc] = 1
            elif ord_u == max_inc[nr, nc]:
                cnt_max[nr, nc] += 1
            rem_deg[nr, nc] -= 1
            if rem_deg[nr, nc] == 0:
                if cnt_max[nr, nc] >= 2:
                    order_grid[nr, nc] = max_inc[nr, nc] + 1
                else:
                    order_grid[nr, nc] = max_inc[nr, nc]
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )
        else:
            order_grid[nr, nc] += ord_u
            rem_deg[nr, nc] -= 1
            if rem_deg[nr, nc] == 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, nr, nc
                )

    return order_grid


@njit
def reverse_bfs_label_kernel(
    flow_dir: np.ndarray, valid_fdir: np.ndarray, seed_grid: np.ndarray
) -> np.ndarray:
    """
    Propagate positive labels in ``seed_grid`` upstream along reverse D8 edges.
    Shared by watershed, basin, and subcatchment delineation.
    """
    # pylint: disable=too-many-locals
    height, width = flow_dir.shape
    out_grid = seed_grid.copy()
    cap = height * width + 1
    q_rows = np.empty(cap, dtype=np.int32)
    q_cols = np.empty(cap, dtype=np.int32)
    q_head = 0
    q_tail = 0
    q_count = 0

    for r in range(height):
        for c in range(width):
            if valid_fdir[r, c] and out_grid[r, c] > 0:
                q_tail, q_count = _queue_push(
                    q_rows, q_cols, q_tail, q_count, cap, r, c
                )

    while q_count > 0:
        r, c, q_head, q_count = _queue_pop(q_rows, q_cols, q_head, q_count, cap)
        lbl = out_grid[r, c]
        for k in range(8):
            nr = r + int(D8_DR[k])
            nc = c + int(D8_DC[k])
            if 0 <= nr < height and 0 <= nc < width:
                if valid_fdir[nr, nc] and out_grid[nr, nc] == 0:
                    if flow_dir[nr, nc] == D8_REV_CODES[k]:
                        out_grid[nr, nc] = lbl
                        q_tail, q_count = _queue_push(
                            q_rows, q_cols, q_tail, q_count, cap, nr, nc
                        )

    return out_grid


@njit
def find_terminal_outlets_kernel(
    flow_dir: np.ndarray, valid_fdir: np.ndarray
) -> np.ndarray:
    """Label all terminal outlet cells ``1..B`` (0 elsewhere, -1 on nodata)."""
    height, width = flow_dir.shape
    seed_grid = np.zeros((height, width), dtype=np.int32)
    next_id = 1
    for r in range(height):
        for c in range(width):
            if not valid_fdir[r, c]:
                seed_grid[r, c] = -1
                continue
            dr, dc = _d8_to_offset(int(flow_dir[r, c]))
            is_out = False
            if dr == 0 and dc == 0:
                is_out = True
            else:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width or not valid_fdir[nr, nc]:
                    is_out = True
            if is_out:
                seed_grid[r, c] = next_id
                next_id += 1
    return seed_grid


@njit(parallel=True)
def flood_inundation_scalar_njit(
    hand: np.ndarray, valid: np.ndarray, stage: float, nd_val: float
) -> np.ndarray:
    """Compute flood inundation depth ``max(0, stage - HAND)`` for scalar stage."""
    height, width = hand.shape
    out = np.full((height, width), nd_val, dtype=np.float32)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                d = stage - hand[r, c]
                out[r, c] = np.float32(d if d > 0.0 else 0.0)
    return out


def flood_inundation_scalar_numpy(
    hand: np.ndarray, valid: np.ndarray, stage: float, nd_val: float
) -> np.ndarray:
    """Vectorised NumPy scalar-stage flood inundation depth."""
    out = np.full(hand.shape, nd_val, dtype=np.float32)
    out[valid] = np.maximum(0.0, stage - hand[valid]).astype(np.float32)
    return out


@njit(parallel=True)
def flood_inundation_raster_njit(
    hand: np.ndarray, stage_arr: np.ndarray, valid: np.ndarray, nd_val: float
) -> np.ndarray:
    """Compute flood inundation depth for spatially varying stage raster."""
    height, width = hand.shape
    out = np.full((height, width), nd_val, dtype=np.float32)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                d = stage_arr[r, c] - hand[r, c]
                out[r, c] = np.float32(d if d > 0.0 else 0.0)
    return out


def flood_inundation_raster_numpy(
    hand: np.ndarray, stage_arr: np.ndarray, valid: np.ndarray, nd_val: float
) -> np.ndarray:
    """Vectorised NumPy raster-stage flood inundation depth."""
    out = np.full(hand.shape, nd_val, dtype=np.float32)
    out[valid] = np.maximum(0.0, stage_arr[valid] - hand[valid]).astype(np.float32)
    return out


@njit
def _slope_to_tan(s_val: float, unit_code: int) -> float:
    """Convert slope value (0=deg, 1=rad, 2=percent) to tan(beta)."""
    if unit_code == 0:
        return math.tan(math.radians(s_val))
    if unit_code == 1:
        return math.tan(s_val)
    return s_val / 100.0


@njit
def _slope_to_rad(s_val: float, unit_code: int) -> float:
    """Convert slope value (0=deg, 1=rad, 2=percent) to radians beta."""
    if unit_code == 0:
        return math.radians(s_val)
    if unit_code == 1:
        return s_val
    return math.atan(s_val / 100.0)


@njit(parallel=True)
def calc_twi_njit(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    min_slope: float,
    dx: float,
    nd_val: float,
) -> np.ndarray:
    """Parallel Numba Topographic Wetness Index ``ln(a_eff / tan_eff)``."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    height, width = sca.shape
    out = np.full((height, width), nd_val, dtype=np.float32)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                tan_b = _slope_to_tan(slope_arr[r, c], unit_code)
                tan_eff = tan_b if tan_b > min_slope else min_slope
                a_val = sca[r, c]
                a_eff = a_val if a_val > dx else dx
                out[r, c] = np.float32(math.log(a_eff / tan_eff))
    return out


def calc_twi_numpy(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    min_slope: float,
    dx: float,
    nd_val: float,
) -> np.ndarray:
    """Vectorised NumPy Topographic Wetness Index."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out = np.full(sca.shape, nd_val, dtype=np.float32)
    s_v = slope_arr[valid]
    if unit_code == 0:
        tan_b = np.tan(np.deg2rad(s_v))
    elif unit_code == 1:
        tan_b = np.tan(s_v)
    else:
        tan_b = s_v / 100.0
    tan_eff = np.maximum(tan_b, min_slope)
    a_eff = np.maximum(sca[valid], dx)
    out[valid] = np.log(a_eff / tan_eff).astype(np.float32)
    return out


@njit(parallel=True)
def calc_spi_njit(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    nd_val: float,
) -> np.ndarray:
    """Parallel Numba Stream Power Index ``a * tan(beta)``."""
    height, width = sca.shape
    out = np.full((height, width), nd_val, dtype=np.float32)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                tan_b = _slope_to_tan(slope_arr[r, c], unit_code)
                out[r, c] = np.float32(sca[r, c] * tan_b)
    return out


def calc_spi_numpy(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    nd_val: float,
) -> np.ndarray:
    """Vectorised NumPy Stream Power Index."""
    out = np.full(sca.shape, nd_val, dtype=np.float32)
    s_v = slope_arr[valid]
    if unit_code == 0:
        tan_b = np.tan(np.deg2rad(s_v))
    elif unit_code == 1:
        tan_b = np.tan(s_v)
    else:
        tan_b = s_v / 100.0
    out[valid] = (sca[valid] * tan_b).astype(np.float32)
    return out


@njit(parallel=True)
def calc_ls_njit(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    m_exp: float,
    n_exp: float,
    nd_val: float,
) -> np.ndarray:
    """Parallel Numba Moore & Burch (1986) RUSLE LS Factor."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    height, width = sca.shape
    out = np.full((height, width), nd_val, dtype=np.float32)
    for r in prange(height):
        for c in range(width):
            if valid[r, c]:
                beta = _slope_to_rad(slope_arr[r, c], unit_code)
                sin_b = math.sin(beta)
                if sin_b < 0.0:
                    sin_b = 0.0
                a_val = sca[r, c]
                if a_val < 0.0:
                    a_val = 0.0
                ls = (
                    (m_exp + 1.0)
                    * ((a_val / 22.13) ** m_exp)
                    * ((sin_b / 0.0896) ** n_exp)
                )
                out[r, c] = np.float32(ls)
    return out


def calc_ls_numpy(
    sca: np.ndarray,
    slope_arr: np.ndarray,
    valid: np.ndarray,
    unit_code: int,
    m_exp: float,
    n_exp: float,
    nd_val: float,
) -> np.ndarray:
    """Vectorised NumPy Moore & Burch (1986) RUSLE LS Factor."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    out = np.full(sca.shape, nd_val, dtype=np.float32)
    s_v = slope_arr[valid]
    if unit_code == 0:
        beta = np.deg2rad(s_v)
    elif unit_code == 1:
        beta = s_v
    else:
        beta = np.arctan(s_v / 100.0)
    sin_b = np.maximum(np.sin(beta), 0.0)
    a_val = np.maximum(sca[valid], 0.0)
    ls = (m_exp + 1.0) * ((a_val / 22.13) ** m_exp) * ((sin_b / 0.0896) ** n_exp)
    out[valid] = ls.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# D8 path tracing kernels (trace_flow_path / vectorise_streams).
#
# Decorated with ``optional_njit``: the undecorated ``py_func`` is the
# pure-Python implementation used when numba is unavailable, so compiled and
# fallback paths execute identical logic.
# ---------------------------------------------------------------------------

# 256-entry lookup tables indexed by uint8 D8 code
D8_LUT_OK = np.zeros(256, dtype=np.bool_)
D8_LUT_DR = np.zeros(256, dtype=np.int64)
D8_LUT_DC = np.zeros(256, dtype=np.int64)
D8_LUT_IDX = np.full(256, -1, dtype=np.int64)
D8_LUT_ANGLE = np.full(256, np.nan, dtype=np.float64)
for _k in range(8):
    _code = int(D8_CODES[_k])
    D8_LUT_OK[_code] = True
    D8_LUT_DR[_code] = int(D8_DR[_k])
    D8_LUT_DC[_code] = int(D8_DC[_k])
    D8_LUT_IDX[_code] = _k
for _code, _ang in (
    (1, 0.0),
    (128, 0.25 * np.pi),
    (64, 0.5 * np.pi),
    (32, 0.75 * np.pi),
    (16, np.pi),
    (8, 1.25 * np.pi),
    (4, 1.5 * np.pi),
    (2, 1.75 * np.pi),
):
    D8_LUT_ANGLE[_code] = _ang
del _k, _code, _ang


@optional_njit(cache=True, warn=False)
def trace_d8_path_kernel(
    fdir: np.ndarray,
    valid: np.ndarray,
    acc_arr: np.ndarray,
    acc_valid: np.ndarray,
    r0: int,
    c0: int,
    upstream: bool,
    visited: np.ndarray,
    lut_ok: np.ndarray,
    lut_dr: np.ndarray,
    lut_dc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trace one D8 flow path from ``(r0, c0)`` and return ``(rows, cols)`` of
    visited cells in order. Downstream follows ``fdir`` codes; upstream
    follows the unvisited donor neighbour with maximum ``acc_arr`` (first in
    D8 order on ties). ``visited`` (``uint8``, all zero on entry) is used as
    scratch space and is reset to zero before returning.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-nested-blocks
    height, width = fdir.shape
    rows = [np.int64(r0)]
    cols = [np.int64(c0)]
    rows.pop()
    cols.pop()
    curr_r = np.int64(r0)
    curr_c = np.int64(c0)
    while True:
        if visited[curr_r, curr_c]:
            break
        visited[curr_r, curr_c] = 1
        rows.append(curr_r)
        cols.append(curr_c)
        if not upstream:
            code = fdir[curr_r, curr_c]
            if not lut_ok[code]:
                break
            nr = curr_r + lut_dr[code]
            nc = curr_c + lut_dc[code]
            if not (0 <= nr < height and 0 <= nc < width and valid[nr, nc]):
                break
            curr_r = nr
            curr_c = nc
        else:
            best_acc = -1.0
            best_r = np.int64(-1)
            best_c = np.int64(-1)
            for k in range(8):
                nr = curr_r + np.int64(D8_DR[k])
                nc = curr_c + np.int64(D8_DC[k])
                if 0 <= nr < height and 0 <= nc < width:
                    if valid[nr, nc] and acc_valid[nr, nc]:
                        if not visited[nr, nc] and fdir[nr, nc] == D8_REV_CODES[k]:
                            if acc_arr[nr, nc] > best_acc:
                                best_acc = acc_arr[nr, nc]
                                best_r = nr
                                best_c = nc
            if best_r < 0:
                break
            curr_r = best_r
            curr_c = best_c
    n = len(rows)
    out_r = np.empty(n, dtype=np.int64)
    out_c = np.empty(n, dtype=np.int64)
    for i in range(n):
        out_r[i] = rows[i]
        out_c[i] = cols[i]
        visited[rows[i], cols[i]] = 0
    return out_r, out_c


@optional_njit(cache=True, warn=False)
def trace_stream_reach_kernel(
    fdir: np.ndarray,
    valid: np.ndarray,
    streams_bin: np.ndarray,
    link_grid: np.ndarray,
    r0: int,
    c0: int,
    lut_ok: np.ndarray,
    lut_dr: np.ndarray,
    lut_dc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Follow D8 directions from reach start ``(r0, c0)`` while cells remain on
    the same stream link. Returns ``(rows, cols, down_link_id, conf_r,
    conf_c)`` where ``(conf_r, conf_c)`` is the downstream confluence cell on
    another link (``-1`` if the reach ends without one). Stops after
    ``height * width`` steps to guard against cyclic direction grids.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    height, width = fdir.shape
    lid = link_grid[r0, c0]
    rows = [np.int64(r0)]
    cols = [np.int64(c0)]
    rows.pop()
    cols.pop()
    r = np.int64(r0)
    c = np.int64(c0)
    down_link_id = np.int64(-1)
    conf_r = np.int64(-1)
    conf_c = np.int64(-1)
    max_steps = height * width
    for _ in range(max_steps):
        rows.append(r)
        cols.append(c)
        code = fdir[r, c]
        if not lut_ok[code]:
            break
        nr = r + lut_dr[code]
        nc = c + lut_dc[code]
        if not (0 <= nr < height and 0 <= nc < width and valid[nr, nc]):
            break
        if streams_bin[nr, nc] != 1:
            break
        if link_grid[nr, nc] != lid:
            down_link_id = np.int64(link_grid[nr, nc])
            conf_r = nr
            conf_c = nc
            break
        r = nr
        c = nc
    n = len(rows)
    out_r = np.empty(n, dtype=np.int64)
    out_c = np.empty(n, dtype=np.int64)
    for i in range(n):
        out_r[i] = rows[i]
        out_c[i] = cols[i]
    return out_r, out_c, down_link_id, conf_r, conf_c
