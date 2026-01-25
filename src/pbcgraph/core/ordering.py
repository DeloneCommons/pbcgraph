"""Deterministic ordering helpers.

NetworkX iteration order is insertion-based and can vary with construction
sequence. pbcgraph enforces deterministic ordering at the container level.

Strategy
--------
For a collection of objects we first try native sorting (``sorted(items)``).
If objects are not mutually comparable (``TypeError``), we fall back to a
stable composite key based on type identity and ``repr``.

The fallback is deterministic provided that:

- the type's module and qualname are stable (normally true), and
- ``repr(obj)`` is stable across processes and does not embed memory
  addresses.

For best cross-process determinism, prefer primitive node ids (``int``,
``str``, tuples of primitives).
"""

from __future__ import annotations

from typing import (
    Any, Iterable, List, Sequence, Tuple, TypeVar,
)


T = TypeVar('T')


def fallback_key(x: Any) -> Tuple[str, str, str]:
    """Return a stable fallback key for objects that are not comparable."""
    tp = type(x)
    return (tp.__module__, tp.__qualname__, repr(x))


def stable_sorted(items: Iterable[T]) -> List[T]:
    """Return a deterministically sorted list.

    This tries native ordering first and falls back to a composite key.
    """
    seq = list(items)
    try:
        return sorted(seq)
    except TypeError:
        return sorted(seq, key=fallback_key)  # type: ignore[arg-type]


def stable_unique_sorted(items: Iterable[T]) -> List[T]:
    """Return unique items in deterministic order."""
    return stable_sorted(set(items))


def stable_tvec(tvec: Sequence[Any]) -> Tuple[int, ...]:
    """Canonicalize a translation vector to a tuple of Python ints."""
    return tuple(int(x) for x in tvec)


def try_sort_edges(
    records: List[Tuple[Any, Any, Tuple[int, ...], int, Any]],
) -> None:
    """In-place deterministic sort for edge records.

    Records are ``(u, v, tvec, key, payload)``.
    """
    try:
        records.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    except TypeError:
        records.sort(
            key=lambda r: (fallback_key(r[0]), fallback_key(r[1]), r[2], r[3])
        )


def try_sort_neighbor_edges(
    records: List[Tuple[Any, Tuple[int, ...], int, Any]],
) -> None:
    """In-place deterministic sort for neighbor-edge records.

    Records are ``(v, tvec, key, payload)``.
    """
    try:
        records.sort(key=lambda r: (r[0], r[1], r[2]))
    except TypeError:
        records.sort(key=lambda r: (fallback_key(r[0]), r[1], r[2]))
