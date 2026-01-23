"""Structural protocols for algorithm modules."""

from __future__ import annotations

from typing import Iterable, Protocol

from pbcgraph.core.types import NodeId


class PeriodicDiGraphLike(Protocol):
    """Protocol for pbcgraph containers used by algorithms."""

    dim: int
    structural_version: int
    data_version: int
    is_undirected: bool

    def has_node(self, u: NodeId) -> bool:
        ...

    def nodes(self, data: bool = False) -> Iterable:
        ...

    def successors(self, u: NodeId) -> Iterable[NodeId]:
        ...

    def predecessors(self, u: NodeId) -> Iterable[NodeId]:
        ...
