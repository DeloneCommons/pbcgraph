"""Structural protocols for algorithm modules.

Algorithms in :mod:`pbcgraph.alg` and
:class:`~pbcgraph.component.PeriodicComponent`
operate on a narrow "graph-like" surface. This keeps algorithms independent
from the underlying NetworkX backend.

These protocols are intentionally minimal and may grow between versions.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from pbcgraph.core.types import EdgeKey, NodeId, TVec


class PeriodicDiGraphLike(Protocol):
    """Protocol for periodic graph containers used by algorithms."""

    dim: int
    structural_version: int
    data_version: int
    is_undirected: bool
    is_multigraph: bool

    # Nodes
    def has_node(self, u: NodeId) -> bool:
        ...

    def nodes(self, data: bool = False) -> Iterable:
        ...

    def successors(self, u: NodeId) -> Iterable[NodeId]:
        ...

    def predecessors(self, u: NodeId) -> Iterable[NodeId]:
        ...

    # Directed edge access
    def neighbors(
        self, u: NodeId, keys: bool = False, data: bool = False
    ) -> Iterable:
        ...

    def in_neighbors(
        self, u: NodeId, keys: bool = False, data: bool = False
    ) -> Iterable:
        ...

    def edges(
        self, keys: bool = False, data: bool = False, tvec: bool = False
    ) -> Iterable:
        ...

    # Lifted neighborhoods
    def neighbors_inst(
        self, node_inst: tuple[NodeId, TVec], keys: bool = False,
        data: bool = False
    ) -> Iterable:
        ...

    def in_neighbors_inst(
        self, node_inst: tuple[NodeId, TVec], keys: bool = False,
        data: bool = False
    ) -> Iterable:
        ...

    def edge_tvec(self, u: NodeId, v: NodeId, key: EdgeKey) -> TVec:
        ...
