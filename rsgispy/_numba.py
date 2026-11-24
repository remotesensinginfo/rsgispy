"""
Shared optional ``numba`` detection, no-op decorators, and fallback warning
helpers used across :mod:`rsgispy.elevation` and :mod:`rsgispy.segmentation`.
"""

import warnings
from typing import Any, Callable, Optional

try:
    import numba  # pylint: disable=import-error
    from numba import njit, prange  # pylint: disable=import-error

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        """No-op fallback decorator when ``numba`` is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    prange = range  # type: ignore[assignment]

HAVE_NUMBA = HAS_NUMBA

__all__ = [
    "HAS_NUMBA",
    "HAVE_NUMBA",
    "numba",
    "njit",
    "prange",
    "is_numba_available",
    "make_njit",
    "warn_numba_fallback",
]


def is_numba_available() -> bool:
    """Return ``True`` if optional ``numba`` JIT acceleration is installed."""
    return HAS_NUMBA


def make_njit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate ``func`` with ``numba.njit(cache=False)`` when available."""
    if HAS_NUMBA and numba is not None:
        return numba.njit(cache=False)(func)
    return func


def warn_numba_fallback(
    func_name: str,
    fallback_desc: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """
    Emit a standardised :class:`RuntimeWarning` when ``numba`` is unavailable
    and a function falls back to pure Python or vectorised NumPy.
    """
    if fallback_desc:
        msg = (
            f"numba is not installed; running '{func_name}' using "
            f"{fallback_desc}."
        )
    else:
        msg = (
            f"numba is not installed; running '{func_name}' in pure Python, "
            "which will be significantly slower on large rasters."
        )
    warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)
