"""
Optional ``numba``-accelerated union-find (disjoint-set) kernels shared by
:mod:`rsgispy.segmentation` (tiled seam reconciliation and run-length-encoded
connected-component labelling).

All kernels resolve each node to the *minimum* node index in its connected
component, which is independent of union order. This makes the compiled
union-by-rank / path-halving implementation produce results identical to the
pure-Python "attach larger root to smaller root" fallbacks.
"""

from typing import Tuple

import numpy as np

from rsgispy._numba import optional_njit

__all__ = [
    "uf_min_root_kernel",
    "uf_first_appearance_labels_kernel",
]


@optional_njit(cache=True)
def uf_min_root_kernel(
    u_arr: np.ndarray,
    v_arr: np.ndarray,
    n_nodes: int,
    min_node: int,
) -> np.ndarray:
    """
    Union every pair ``(u_arr[i], v_arr[i])`` over nodes ``0 .. n_nodes - 1``
    and return ``lut`` (``int64``, shape ``(n_nodes,)``) mapping each node to
    the minimum node index of its component.

    Pairs with either endpoint ``< min_node`` or ``>= n_nodes`` are ignored.
    Uses union-by-rank with path halving, tracking the component minimum at
    each root so the output is canonical regardless of union order.
    """
    # pylint: disable=too-many-branches
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.uint8)
    comp_min = np.arange(n_nodes, dtype=np.int64)
    for i in range(u_arr.shape[0]):
        a = np.int64(u_arr[i])
        b = np.int64(v_arr[i])
        if a < min_node or b < min_node or a >= n_nodes or b >= n_nodes:
            continue
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        # ``a`` becomes the root of the merged set
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        if comp_min[b] < comp_min[a]:
            comp_min[a] = comp_min[b]

    lut = np.empty(n_nodes, dtype=np.int64)
    for idx in range(n_nodes):
        root = idx
        while parent[root] != root:
            root = parent[root]
        lut[idx] = comp_min[root]
    return lut


@optional_njit(cache=True)
def uf_first_appearance_labels_kernel(min_root_lut: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Convert a component-minimum LUT (from :func:`uf_min_root_kernel`) into
    component IDs ``0 .. n_comp - 1`` in first-appearance (scanline) order.
    """
    n_nodes = min_root_lut.shape[0]
    labels = np.empty(n_nodes, dtype=np.int64)
    next_id = 0
    for idx in range(n_nodes):
        root = min_root_lut[idx]
        if root == idx:
            labels[idx] = next_id
            next_id += 1
        else:
            labels[idx] = labels[root]
    return next_id, labels
