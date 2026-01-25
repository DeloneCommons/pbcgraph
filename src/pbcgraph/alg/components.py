"""Connected component extraction (quotient)."""

from __future__ import annotations

from collections import deque
from typing import List, Set

from pbcgraph.alg._neighbors import weak_neighbors as _weak_neighbors
from pbcgraph.component import PeriodicComponent
from pbcgraph.core.protocols import PeriodicDiGraphLike
from pbcgraph.core.types import NodeId


def components(G: PeriodicDiGraphLike) -> List[PeriodicComponent]:
    """Return PeriodicComponent objects for each quotient component.

    Notes:
        - For undirected containers (`is_undirected=True`): components are the
          usual undirected components (the container stores both directions per
          undirected edge).
        - For directed containers: v0.1 uses weak connectivity (direction
          ignored).

    Args:
        G: A periodic graph container (structural protocol).

    Returns:
        List of PeriodicComponent objects in deterministic order.

        The order follows the graph's deterministic node iteration
        (`G.nodes(...)`), so the first component is the one containing the
        smallest node under the container's stable ordering.
    """
    visited: Set[NodeId] = set()
    out: List[PeriodicComponent] = []

    for u in G.nodes(data=False):
        if u in visited:
            continue

        q = deque([u])
        visited.add(u)
        comp_nodes: List[NodeId] = []

        while q:
            x = q.popleft()
            comp_nodes.append(x)
            for y in _weak_neighbors(G, x):
                if y in visited:
                    continue
                visited.add(y)
                q.append(y)

        nodes_frozen = frozenset(comp_nodes)
        out.append(
            PeriodicComponent(
                graph=G,  # type: ignore[arg-type]
                nodes=nodes_frozen,
                root=u,
                created_structural_version=G.structural_version,
            )
        )

    return out


connected_components = components
