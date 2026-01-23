"""Core types and translation-vector helpers.

The public API uses a small number of lightweight aliases:

Attributes:
    TVec: Integer translation vector (length `dim`).
    NodeId: Hashable quotient-node identifier.
    EdgeKey: Integer key identifying a directed edge in a multigraph.
    NodeInst: Node instance in the infinite lift: ``(node_id, shift)``.
"""

from __future__ import annotations

from typing import Hashable, Tuple

TVec = Tuple[int, ...]
NodeId = Hashable
EdgeKey = int
NodeInst = Tuple[NodeId, TVec]


def zero_tvec(dim: int) -> TVec:
    """Return the zero translation vector of length `dim`."""
    if dim <= 0:
        raise ValueError('dim must be positive')
    return (0,) * dim


def validate_tvec(tvec: TVec, dim: int) -> None:
    """Validate that `tvec` is an integer tuple of the correct dimension.

    Args:
        tvec: Translation vector.
        dim: Required dimension.

    Raises:
        ValueError: If `tvec` has wrong length or non-int entries.
    """
    if len(tvec) != dim:
        raise ValueError(f'tvec must have length {dim}, got {len(tvec)}')
    for x in tvec:
        if not isinstance(x, int):
            raise ValueError('tvec entries must be int')


def add_tvec(a: TVec, b: TVec) -> TVec:
    """Add two translation vectors."""
    return tuple(x + y for x, y in zip(a, b))


def sub_tvec(a: TVec, b: TVec) -> TVec:
    """Subtract two translation vectors."""
    return tuple(x - y for x, y in zip(a, b))


def neg_tvec(a: TVec) -> TVec:
    """Negate a translation vector."""
    return tuple(-x for x in a)
