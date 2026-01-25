"""Core types and translation-vector helpers.

The public API uses a small number of lightweight aliases:

Attributes:
    TVec: Integer translation vector (length `dim`).
    NodeId: Hashable quotient-node identifier.
    EdgeKey: Integer key identifying a directed edge in a multigraph.
    NodeInst: Node instance in the infinite lift: ``(node_id, shift)``.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence, Tuple

import numbers

import numpy as np

TVec = Tuple[int, ...]
NodeId = Hashable
EdgeKey = int
NodeInst = Tuple[NodeId, TVec]


def zero_tvec(dim: int) -> TVec:
    """Return the zero translation vector of length `dim`."""
    if dim <= 0:
        raise ValueError('dim must be positive')
    return (0,) * dim


def validate_tvec(tvec: Any, dim: int) -> None:
    """Validate that `tvec` is an integer translation vector
    of dimension `dim`.

    Accepts:
        - tuples/lists of integer-likes,
        - 1D NumPy arrays,
        - sequences containing NumPy integer scalars.

    Args:
        tvec: Translation vector.
        dim: Required dimension.

    Raises:
        ValueError: If `tvec` has wrong length, wrong shape, or non-integer
            entries.

    Notes:
        - Entries must be instances of :class:`numbers.Integral`.
        - ``bool`` is rejected (even though it is a subclass of ``int``).
        - This function validates only; containers store translation vectors
          internally as tuples of Python ``int``.
    """

    if dim <= 0:
        raise ValueError('dim must be positive')

    if isinstance(tvec, np.ndarray):
        if tvec.ndim != 1:
            raise ValueError('tvec must be a 1D sequence')
        if tvec.shape[0] != dim:
            raise ValueError(
                f'tvec must have length {dim}, got {int(tvec.shape[0])}'
            )
        seq: Sequence[Any] = tvec.tolist()
    else:
        try:
            n = len(tvec)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError('tvec must be a sized sequence') from e
        if n != dim:
            raise ValueError(f'tvec must have length {dim}, got {n}')
        seq = tvec  # type: ignore[assignment]

    for x in seq:
        if isinstance(x, bool) or not isinstance(x, numbers.Integral):
            raise ValueError('tvec entries must be integer-like (Integral)')


def add_tvec(a: TVec, b: TVec) -> TVec:
    """Add two translation vectors."""
    return tuple(x + y for x, y in zip(a, b))


def sub_tvec(a: TVec, b: TVec) -> TVec:
    """Subtract two translation vectors."""
    return tuple(x - y for x, y in zip(a, b))


def neg_tvec(a: TVec) -> TVec:
    """Negate a translation vector."""
    return tuple(-x for x in a)
