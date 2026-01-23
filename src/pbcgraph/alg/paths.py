"""Path algorithms (v0.1)."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Literal, Optional

from pbcgraph.alg._neighbors import weak_neighbors as _weak_neighbors
from pbcgraph.core.protocols import PeriodicDiGraphLike
from pbcgraph.core.types import NodeId

Connectivity = Literal['directed', 'weak']


def shortest_path_quotient(
    G: PeriodicDiGraphLike,
    source: NodeId,
    target: NodeId,
    *,
    connectivity: Optional[Connectivity] = None,
) -> List[NodeId]:
    """Shortest path in the quotient graph (BFS).

    Args:
        G: A periodic graph container (structural protocol).
        source: Source quotient node id.
        target: Target quotient node id.
        connectivity: Connectivity mode:
            - None: container-dependent default
                * is_undirected=False -> 'directed'
                * is_undirected=True -> 'weak'
            - 'directed': respect edge directions (successors only).
            - 'weak': ignore directions (successors plus predecessors).

    Returns:
        A list of quotient node ids (including source and target).

    Raises:
        ValueError: If no path exists, or if an invalid connectivity
            is requested.
        KeyError: If source or target is not in the graph.
    """
    if not G.has_node(source):
        raise KeyError(source)
    if not G.has_node(target):
        raise KeyError(target)
    if source == target:
        return [source]

    if connectivity is None:
        connectivity = (
            'weak' if getattr(G, 'is_undirected', False) else 'directed'
        )

    if connectivity == 'directed' and getattr(G, 'is_undirected', False):
        raise ValueError(
            "PeriodicGraph-like containers do not support "
            "connectivity='directed'"
        )

    if connectivity not in ('directed', 'weak'):
        raise ValueError(f'unknown connectivity: {connectivity}')

    parent: Dict[NodeId, NodeId] = {}
    q: Deque[NodeId] = deque()
    q.append(source)
    seen = {source}

    while q:
        u = q.popleft()
        if connectivity == 'directed':
            nbrs = list(G.successors(u))
        else:
            nbrs = _weak_neighbors(G, u)

        for v in nbrs:
            if v in seen:
                continue
            seen.add(v)
            parent[v] = u
            if v == target:
                q.clear()
                break
            q.append(v)

    if target not in seen:
        raise ValueError('no path from source to target')

    path: List[NodeId] = [target]
    cur = target
    while cur != source:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path
