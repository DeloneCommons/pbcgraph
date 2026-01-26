"""Undirected periodic multigraph container."""

from __future__ import annotations

from typing import Any, Optional

from pbcgraph.core.types import EdgeKey, NodeId, TVec
from pbcgraph.graph.undirected import PeriodicGraph


class PeriodicMultiGraph(PeriodicGraph):
    """Undirected periodic multigraph.

    Unlike `PeriodicGraph`, this container allows multiple undirected edges for
    the same motif pair and translation (i.e. multiple edges for the same
    undirected ``{u, v, tvec}`` up to reversal). Parallel edges are
    distinguished by their edge keys.
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
        """Add an undirected periodic edge (parallel edges allowed)."""
        return self._add_undirected_impl(
            u, v, tvec, key=key, attrs=dict(attrs)
        )
