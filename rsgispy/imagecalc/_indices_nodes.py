"""
Tile-graph node shared by :func:`rsgispy.imagecalc.calcindices.calc_indices`
and :func:`rsgispy.imagecalc.calcindices.calc_sar_indices` (and their node
builders): one output band per requested index.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import rsgispy
from rsgispy.imageprocess import (
    NodeMeta,
    TileContext,
    TileNode,
    TileTensor,
    build_valid_mask,
)


@dataclass
class IndicesSetup:
    """Parameters of an index stack resolved against the input's bands."""

    # pylint: disable=too-many-instance-attributes

    canonical_to_idx: Dict[str, int]
    resolved_indices: List[Tuple[str, Any]]
    req_band_set: List[str]
    in_gain_map: Dict[str, float]
    in_offset_map: Dict[str, float]
    out_gain_list: List[float]
    out_offset_list: List[float]
    in_nodata_map: Dict[str, Optional[float]]

    @property
    def out_names(self) -> List[str]:
        """Output band names (one per requested index)."""
        return [name for name, _ in self.resolved_indices]


def scale_bands(
    data: np.ndarray,
    setup: IndicesSetup,
    positions: Mapping[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Mask and unscale the required bands of a ``float64`` block, as
    ``(dn - in_offset) / in_gain``; ``positions`` maps 1-based input band
    numbers to positions in ``data``.
    """
    scaled_bands: Dict[str, np.ndarray] = {}
    invalid_masks: Dict[str, np.ndarray] = {}
    for b_name in setup.req_band_set:
        raw_arr = data[positions[setup.canonical_to_idx[b_name]]]
        invalid_masks[b_name] = ~build_valid_mask(raw_arr, setup.in_nodata_map[b_name])
        scaled_bands[b_name] = (
            raw_arr - setup.in_offset_map[b_name]
        ) / setup.in_gain_map[b_name]
    return scaled_bands, invalid_masks


class IndicesNode(TileNode):
    """
    Node computing a stack of spectral or SAR indices from one input node.

    The index parameters are resolved against the input's band count (and
    no-data value) when the graph is built, with ``setup_fn(n_bands,
    band_nodata, label)``; ``block_fn(data, setup, positions, shape)`` computes
    the output block.

    :param source: Upstream node with the image bands.
    :param setup_fn: Callable resolving an :class:`IndicesSetup`.
    :param block_fn: Callable computing the ``(n_indices, H, W)`` block.
    :param out_np_dtype: Output dtype.
    :param out_nd: Output no-data value.
    :param no_data_val: Optional input no-data value (all bands).
    :param check_fn: Optional callable validating the setup.
    :param src_bands: 1-based numbers of the file bands held by ``source``
                      (``None``: band ``i`` of ``source`` is band ``i``).
    :param n_bands: Number of bands to validate ``band_names`` against
                    (``None``: those of ``source``).
    :param band_nodata: Per-band input no-data values (index ``i`` for band
                        ``i + 1``) used when ``no_data_val`` is ``None``
                        (``None``: the no-data value of ``source``).
    :param img_label: Image name used in error messages.
    :param name: Optional node name.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    def __init__(
        self,
        source: TileNode,
        setup_fn: Callable[[int, Sequence[Optional[float]], str], IndicesSetup],
        block_fn: Callable[..., np.ndarray],
        out_np_dtype: np.dtype,
        out_nd: float,
        *,
        no_data_val: Optional[float] = None,
        check_fn: Optional[Callable[[IndicesSetup], None]] = None,
        src_bands: Optional[Sequence[int]] = None,
        n_bands: Optional[int] = None,
        band_nodata: Optional[Sequence[Optional[float]]] = None,
        img_label: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs=(source,), name=name)
        self.setup_fn = setup_fn
        self.block_fn = block_fn
        self.out_np_dtype = np.dtype(out_np_dtype)
        self.out_nd = float(out_nd)
        self.no_data_val = no_data_val
        self.check_fn = check_fn
        self.src_bands = list(src_bands) if src_bands is not None else None
        self.n_bands = n_bands
        self.band_nodata = list(band_nodata) if band_nodata is not None else None
        self.img_label = img_label
        self.setup: Optional[IndicesSetup] = None

    def resolve(self, in_meta: NodeMeta) -> IndicesSetup:
        """Resolve (and validate) the index parameters for input ``in_meta``."""
        n_bands = self.n_bands if self.n_bands is not None else in_meta.n_bands
        if n_bands is None:
            raise rsgispy.RSGISPyException(
                f"{self.label} needs an input with a known number of bands."
            )
        if self.no_data_val is not None:
            nodata: List[Optional[float]] = [float(self.no_data_val)] * int(n_bands)
        elif self.band_nodata is not None:
            nodata = list(self.band_nodata)
        else:
            nodata = [in_meta.nodata] * int(n_bands)
        label = self.img_label if self.img_label is not None else f"input of {self.label}"
        setup = self.setup_fn(int(n_bands), nodata, label)
        if self.check_fn is not None:
            self.check_fn(setup)
        return setup

    def infer_meta(self, inputs: Sequence[NodeMeta]) -> NodeMeta:
        self.setup = self.resolve(inputs[0])
        return NodeMeta(
            n_bands=len(self.setup.resolved_indices),
            band_names=tuple(self.setup.out_names),
            dtype=self.out_np_dtype,
            nodata=self.out_nd,
        )

    def evaluate(
        self,
        inputs: Sequence[TileTensor],
        halo_y: int,
        halo_x: int,
        ctx: TileContext,
        state: Any = None,
    ) -> TileTensor:
        aligned = inputs[0].crop(halo_y, halo_x)
        setup = self.setup
        if setup is None:  # pragma: no cover - set when the graph is built
            raise rsgispy.RSGISPyException(f"{self.label} was not resolved.")
        data = np.asarray(aligned.data, dtype=np.float64)
        if self.src_bands is not None:
            positions = {band: pos for pos, band in enumerate(self.src_bands)}
        else:
            positions = {band: band - 1 for band in range(1, data.shape[0] + 1)}
        out = self.block_fn(data, setup, positions, data.shape[1:])
        return aligned.with_data(out)

    def describe_params(self) -> str:
        names = self.setup.out_names if self.setup is not None else "?"
        return f"indices={names} dtype={self.out_np_dtype}"
