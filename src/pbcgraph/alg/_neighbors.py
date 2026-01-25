"""Traversal helpers for algorithms.

This module contains small internal helpers used by multiple algorithms.

The intent is similar to NetworkX's internal traversal utilities: keep
deterministic, well-tested graph-walk primitives in one place so that
algorithm modules do not duplicate them.

Note:
    This is not part of the public API and may change between versions.
"""

from __future__ import annotations

from typing import Iterable, List, Protocol

from pbcgraph.core.types import NodeId


class _SupportsPredSucc(Protocol):
    """Protocol for graphs that expose successor and predecessor iteration."""

    def successors(self, u: NodeId) -> Iterable[NodeId]:
        ...

    def predecessors(self, u: NodeId) -> Iterable[NodeId]:
        ...


def weak_neighbors(G: _SupportsPredSucc, u: NodeId) -> List[NodeId]:
    """Return deterministic weak neighbors of `u`.

    The weak neighborhood treats the directed quotient graph as undirected.
    Order is deterministic:

    1) successors in deterministic order as provided by `G.successors(u)`,
    2) then predecessors in deterministic order as provided by
       `G.predecessors(u)` (excluding nodes already yielded).

    Args:
        G: A graph providing `successors(u)` and `predecessors(u)`.
        u: Quotient node id.

    Returns:
        A list of quotient node ids.
    """
    succ = list(G.successors(u))
    seen = set(succ)
    out = list(succ)
    for p in G.predecessors(u):
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out
