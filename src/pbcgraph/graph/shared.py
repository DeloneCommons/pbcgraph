"""Shared helpers for periodic graph containers.

This module contains internal implementation details that are shared across
container classes in :mod:`pbcgraph.graph`.

Notes:
    - Translation vectors are stored under the internal edge-data key
      :data:`_TVEC_ATTR`.
    - User attributes are stored in a dedicated mapping under
      :data:`_USER_ATTRS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict

from pbcgraph.core.constants import PBC_META_KEY
from pbcgraph.core.types import EdgeKey


_TVEC_ATTR = '_tvec'
_USER_ATTRS = '_attrs'


def ro(mapping: Dict[str, Any]) -> MappingProxyType:
    """Return a read-only live view of a mapping."""
    return MappingProxyType(mapping)


def validate_edge_key(key: EdgeKey) -> None:
    """Validate an edge key.

    Edge keys must be ints, but ``bool`` is rejected (even though it is a
    subclass of ``int``).
    """
    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError('edge key must be an int (bool is not allowed)')


def check_reserved_edge_attrs(attrs: Dict[str, Any]) -> None:
    """Reject reserved edge-attribute keys.

    The key :data:`pbcgraph.PBC_META_KEY` is reserved for pbcgraph export
    metadata (for example, in NetworkX edge attribute dicts produced by
    :meth:`pbcgraph.alg.lift.LiftPatch.to_networkx`).
    """
    if PBC_META_KEY in attrs:
        raise ValueError(
            f'edge attribute key {PBC_META_KEY!r} is reserved for pbcgraph '
            'metadata'
        )


@dataclass(frozen=True)
class UKey:
    """Private directed-edge key for undirected containers.

    `PeriodicGraph` and `PeriodicMultiGraph` represent each undirected edge as
    two directed realizations. When `u == v` (a quotient self-loop), these two
    realizations would collide in NetworkX if they shared the same
    ``(u, v, key)`` triple.

    `UKey` splits the user-visible base key into two internal keys
    distinguished by `dir` in {+1, -1}.

    The public API always exposes the base key (an int).
    """

    base: int
    dir: int

    def __post_init__(self) -> None:
        if self.dir not in (-1, 1):
            raise ValueError('dir must be +1 or -1')


def base_key(k: object) -> int:
    """Return the public base edge key for an internal key."""
    if isinstance(k, UKey):
        return int(k.base)
    return int(k)
