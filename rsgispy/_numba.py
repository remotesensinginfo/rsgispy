"""
Shared optional ``numba`` acceleration infrastructure for :mod:`rsgispy`.

``numba`` is an *optional* dependency (``pip install rsgispy[numba]``). Every
module in :mod:`rsgispy` must import cleanly and produce the same results
without it, so compiled kernels are always paired with a pure-Python / NumPy
fallback. This module centralises that pattern:

* :data:`HAS_NUMBA` - ``True`` when ``numba`` imported successfully and has
  not been disabled through the ``RSGISPY_DISABLE_NUMBA`` environment variable
  (:data:`NUMBA_INSTALLED` reports importability alone).
* :func:`optional_njit` - decorator returning an :class:`OptionalJitFunction`
  that runs a ``numba.njit`` compiled kernel when available and otherwise
  (``numba`` missing, compilation/typing failure, or acceleration disabled)
  calls a fallback implementation (the undecorated function by default, or an
  explicit ``fallback=`` callable with the same signature).
* :func:`numba_enabled` / :func:`disable_numba` - runtime switch, primarily
  for equivalence testing of compiled and fallback paths.
* :func:`warn_numba_fallback` - consistent, optionally de-duplicated
  :class:`RuntimeWarning` emission.

``njit`` / ``prange`` / :func:`make_njit` are retained for kernels that must
call other compiled helpers from inside ``nopython`` code (an
:class:`OptionalJitFunction` cannot be called from within a compiled kernel).
"""

from contextlib import contextmanager
import functools
import os
import threading
from typing import Any, Callable, Dict, Iterator, Optional, Set
import warnings

_ENV_DISABLE = "RSGISPY_DISABLE_NUMBA"


def _env_disabled() -> bool:
    """Return ``True`` if ``RSGISPY_DISABLE_NUMBA`` is set to a truthy value."""
    val = os.environ.get(_ENV_DISABLE, "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


try:
    import numba  # pylint: disable=import-error
    from numba import njit as _numba_njit  # pylint: disable=import-error
    from numba import prange  # pylint: disable=import-error

    _NUMBA_IMPORTED = True
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
    # ``ImportError`` in the common case, but a broken numba / llvmlite
    # install can raise other errors at import time; never let that break
    # importing rsgispy.
    # pylint: disable=invalid-name
    numba = None  # type: ignore[assignment]
    _numba_njit = None  # type: ignore[assignment]
    _NUMBA_IMPORTED = False
    prange = range  # type: ignore[assignment,misc]
    # pylint: enable=invalid-name

NUMBA_INSTALLED: bool = _NUMBA_IMPORTED
HAS_NUMBA: bool = _NUMBA_IMPORTED and not _env_disabled()
HAVE_NUMBA: bool = HAS_NUMBA

if _NUMBA_IMPORTED:
    from numba.core.errors import NumbaError  # pylint: disable=import-error

    _COMPILE_ERRORS: tuple = (NumbaError,)
else:  # pragma: no cover
    _COMPILE_ERRORS = ()

__all__ = [
    "HAS_NUMBA",
    "HAVE_NUMBA",
    "NUMBA_INSTALLED",
    "OptionalJitFunction",
    "disable_numba",
    "is_numba_available",
    "make_njit",
    "njit",
    "numba",
    "numba_enabled",
    "optional_njit",
    "prange",
    "reset_numba_warnings",
    "use_compiled",
    "warn_numba_fallback",
]

_STATE = threading.local()
_WARNED: Set[str] = set()
_WARNED_LOCK = threading.Lock()

# numba's ``workqueue`` threading layer (used when neither TBB nor OpenMP is
# available, e.g. with the macOS wheels) aborts the process when parallel
# kernels are launched from more than one thread at once, as the tile graph
# does with ``n_jobs > 1``. Parallel kernel launches are therefore serialised
# unless the active layer is known to be thread-safe.
_PARALLEL_LOCK = threading.RLock()
_THREADSAFE_LAYERS = ("tbb", "omp")
_PARALLEL_THREADSAFE = False


def _parallel_launch_threadsafe() -> bool:
    """
    ``True`` once numba's threading layer is known to be thread-safe (TBB or
    OpenMP). Before the first parallel kernel has run the layer is not chosen
    yet, so this returns ``False``.
    """
    global _PARALLEL_THREADSAFE  # pylint: disable=global-statement
    if not _PARALLEL_THREADSAFE and _NUMBA_IMPORTED:
        try:
            layer = numba.threading_layer()
        except ValueError:  # no parallel kernel has run yet
            return False
        _PARALLEL_THREADSAFE = layer in _THREADSAFE_LAYERS
    return _PARALLEL_THREADSAFE


def _noop_decorator(*args: Any, **kwargs: Any) -> Any:
    """No-op stand-in for ``numba.njit`` supporting both decorator forms."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


njit: Callable[..., Any] = _numba_njit if HAS_NUMBA else _noop_decorator


def is_numba_available() -> bool:
    """Return ``True`` if optional ``numba`` JIT acceleration is usable."""
    return HAS_NUMBA


def numba_enabled() -> bool:
    """
    Return ``True`` if compiled kernels should be used right now: ``numba`` is
    installed, not disabled via ``RSGISPY_DISABLE_NUMBA``, and not disabled for
    the current thread by :func:`disable_numba`.
    """
    return HAS_NUMBA and not getattr(_STATE, "disabled", False)


@contextmanager
def disable_numba() -> Iterator[None]:
    """
    Context manager forcing :class:`OptionalJitFunction` objects (and code
    branching on :func:`numba_enabled`) onto their fallback paths within the
    current thread. Used to compare compiled and fallback results.

    .. code:: python

        from rsgispy._numba import disable_numba

        with disable_numba():
            ...  # pure-Python / NumPy fallbacks only
    """
    prev = getattr(_STATE, "disabled", False)
    _STATE.disabled = True
    try:
        yield
    finally:
        _STATE.disabled = prev


def reset_numba_warnings() -> None:
    """Clear the de-duplication registry used by :func:`warn_numba_fallback`."""
    with _WARNED_LOCK:
        _WARNED.clear()


def warn_numba_fallback(
    func_name: str,
    fallback_desc: Optional[str] = None,
    stacklevel: int = 3,
    once: bool = False,
    reason: Optional[str] = None,
) -> None:
    """
    Emit a standardised :class:`RuntimeWarning` when ``numba`` acceleration is
    unavailable or bypassed and ``func_name`` falls back to pure Python or
    vectorised NumPy.

    :param func_name: Name of the function/kernel falling back.
    :param fallback_desc: Description of the fallback (e.g.
                          ``'the vectorised NumPy fallback'``).
    :param stacklevel: ``warnings.warn`` stack level (Default = ``3``).
    :param once: If ``True``, emit at most one warning per ``func_name`` per
                 process (Default = ``False``).
    :param reason: Optional reason replacing the default
                   ``'numba is not installed'`` prefix.
    """
    if once:
        with _WARNED_LOCK:
            if func_name in _WARNED:
                return
            _WARNED.add(func_name)
    prefix = reason if reason else "numba is not installed"
    if fallback_desc:
        msg = f"{prefix}; running '{func_name}' using {fallback_desc}."
    else:
        msg = (
            f"{prefix}; running '{func_name}' in pure Python, "
            "which will be significantly slower on large rasters. "
            "Install the optional extra with `pip install rsgispy[numba]`."
        )
    warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)


class OptionalJitFunction:
    """
    Callable wrapping a kernel that is ``numba.njit`` compiled when possible
    and otherwise dispatches to a pure-Python / NumPy fallback.

    Attributes
    ----------
    py_func : callable
        The undecorated kernel source function.
    fallback : callable
        Function called when compiled execution is unavailable (``py_func``
        unless an explicit ``fallback`` was supplied).
    jit_options : dict
        Options passed to ``numba.njit``.
    """

    def __init__(
        self,
        py_func: Callable[..., Any],
        jit_options: Dict[str, Any],
        fallback: Optional[Callable[..., Any]] = None,
        fallback_desc: Optional[str] = None,
        warn: bool = True,
    ) -> None:
        self.py_func = py_func
        self.fallback = fallback if fallback is not None else py_func
        self.jit_options = dict(jit_options)
        self.is_parallel = bool(self.jit_options.get("parallel", False))
        self.fallback_desc = fallback_desc
        self.warn = warn
        self._dispatcher: Any = None
        self._failed_reason: Optional[str] = None
        functools.update_wrapper(self, py_func)
        if HAS_NUMBA:
            self._dispatcher = self._build_dispatcher()

    def _build_dispatcher(self) -> Any:
        """Create the lazy ``numba`` dispatcher, retrying without caching."""
        opts = dict(self.jit_options)
        try:
            return _numba_njit(**opts)(self.py_func)
        except Exception as err:  # pylint: disable=broad-except
            if opts.pop("cache", False):
                # e.g. no writable cache locator for read-only installs
                try:
                    return _numba_njit(**opts)(self.py_func)
                except Exception as err2:  # pylint: disable=broad-except
                    err = err2
            self._failed_reason = f"numba could not compile ({err})"
            return None

    @property
    def jit_func(self) -> Any:
        """Compiled ``numba`` dispatcher, or ``None`` if unavailable."""
        return self._dispatcher if self._failed_reason is None else None

    @property
    def is_compiled_available(self) -> bool:
        """``True`` if the compiled path is usable in the current context."""
        return self.jit_func is not None and numba_enabled()

    def _warn_fallback(self) -> None:
        if not self.warn:
            return
        warn_numba_fallback(
            self.py_func.__name__,
            fallback_desc=self.fallback_desc,
            stacklevel=4,
            once=True,
            reason=self._failed_reason,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if numba_enabled():
            disp = self.jit_func
            if disp is not None:
                try:
                    if self.is_parallel and not _parallel_launch_threadsafe():
                        with _PARALLEL_LOCK:
                            return disp(*args, **kwargs)
                    return disp(*args, **kwargs)
                except _COMPILE_ERRORS as err:
                    # Typing/lowering failure: stop retrying compilation and
                    # use the fallback from now on.
                    text = str(err).strip()
                    first_line = text.splitlines()[0] if text else type(err).__name__
                    self._failed_reason = (
                        f"numba could not compile for these inputs ({first_line})"
                    )
            self._warn_fallback()
        elif not _NUMBA_IMPORTED:
            self._warn_fallback()
        return self.fallback(*args, **kwargs)

    def __repr__(self) -> str:
        state = "compiled" if self.jit_func is not None else "fallback"
        return f"<OptionalJitFunction {self.py_func.__qualname__} ({state})>"


def optional_njit(  # pylint: disable=too-many-arguments
    func: Optional[Callable[..., Any]] = None,
    *,
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
    nogil: bool = False,
    fallback: Optional[Callable[..., Any]] = None,
    fallback_desc: Optional[str] = None,
    warn: bool = True,
) -> Any:
    """
    Decorate a kernel so it is compiled with ``numba.njit`` when available and
    otherwise runs a pure-Python / NumPy fallback.

    Compilation is lazy (on first call). If ``numba`` is missing, disabled
    (``RSGISPY_DISABLE_NUMBA=1`` or :func:`disable_numba`), or fails to compile
    for the supplied argument types, ``fallback`` (default: the undecorated
    function) is called instead and a de-duplicated :class:`RuntimeWarning`
    is emitted (except when acceleration was explicitly disabled).

    :param func: Kernel function (when used as ``@optional_njit``).
    :param parallel: Enable ``numba`` automatic parallelisation / ``prange``.
    :param cache: Cache compiled machine code on disk (Default = ``True``).
    :param fastmath: Allow ``fastmath`` floating-point reassociation
                     (Default = ``False`` to keep results reproducible).
    :param nogil: Release the GIL inside the compiled kernel.
    :param fallback: Optional alternative implementation with the same
                     signature to call when compiled execution is unavailable.
    :param fallback_desc: Description used in fallback warnings.
    :param warn: Emit a fallback :class:`RuntimeWarning` (Default = ``True``).

    .. code:: python

        from rsgispy._numba import optional_njit, prange

        @optional_njit(parallel=True)
        def _double(arr):
            out = arr.copy()
            for i in prange(arr.shape[0]):
                out[i] = arr[i] * 2.0
            return out
    """
    options: Dict[str, Any] = {
        "cache": bool(cache),
        "fastmath": bool(fastmath),
        "nogil": bool(nogil),
    }
    if parallel:
        options["parallel"] = True

    def decorator(inner: Callable[..., Any]) -> OptionalJitFunction:
        return OptionalJitFunction(
            inner,
            options,
            fallback=fallback,
            fallback_desc=fallback_desc,
            warn=warn,
        )

    if func is not None and callable(func):
        return decorator(func)
    return decorator


def use_compiled(
    kernel: Any,
    func_name: Optional[str] = None,
    fallback_desc: Optional[str] = None,
) -> bool:
    """
    Return ``True`` if ``kernel`` (an :class:`OptionalJitFunction`) should be
    run compiled; otherwise return ``False`` so the caller can branch to its
    own pure-Python / NumPy implementation.

    A de-duplicated :class:`RuntimeWarning` is emitted when the fallback is
    taken because ``numba`` is missing or failed to compile, but not when
    acceleration was deliberately disabled (``RSGISPY_DISABLE_NUMBA`` or
    :func:`disable_numba`).
    """
    jit_func = getattr(kernel, "jit_func", None)
    if numba_enabled() and jit_func is not None:
        return True
    if numba_enabled() or not _NUMBA_IMPORTED:
        name = func_name or getattr(kernel, "__name__", "rsgispy kernel")
        warn_numba_fallback(
            name,
            fallback_desc=fallback_desc,
            stacklevel=3,
            once=True,
            reason=getattr(kernel, "_failed_reason", None),
        )
    return False


def make_njit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorate ``func`` with ``numba.njit(cache=False)`` when available.

    Retained for kernels dispatched through
    :func:`rsgispy.segmentation.utils.call_kernel`, which selects between the
    compiled dispatcher and ``func.py_func`` itself.
    """
    if HAS_NUMBA and _numba_njit is not None:
        return _numba_njit(cache=False)(func)
    return func
