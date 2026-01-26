"""Directed periodic multigraph container."""

from __future__ import annotations

from typing import Any, Optional

from pbcgraph.core.types import EdgeKey, NodeId, TVec
from pbcgraph.graph.directed import PeriodicDiGraph


class PeriodicMultiDiGraph(PeriodicDiGraph):
    """Directed periodic multigraph on ``Z^d``.

    Unlike `PeriodicDiGraph`, this container allows multiple edges for the same
    directed triple ``(u, v, tvec)``. Such parallel edges are distinguished by
    their edge keys.
    """

    @property
    def is_multigraph(self) -> bool:
        """Whether this container allows multiple edges per `(u, v, tvec)`."""
        return True

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        tvec: TVec,
        key: Optional[EdgeKey] = None,
        **attrs: Any,
    ) -> EdgeKey:
        """Add a directed periodic edge (parallel edges allowed)."""
        return self._add_edge_impl(u, v, tvec, key=key, attrs=dict(attrs))


