"""Finite lift patches of periodic graphs.

This module provides `lift_patch(...)` and the `LiftPatch` container,
a finite, non-periodic view extracted from the infinite lift of a
periodic quotient graph.
"""


from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx

from pbcgraph.core.constants import PBC_META_KEY
from pbcgraph.core.exceptions import LiftPatchError
from pbcgraph.core.ordering import fallback_key, stable_sorted
from pbcgraph.core.protocols import PeriodicDiGraphLike
from pbcgraph.core.types import (
    NodeInst,
    TVec,
    validate_tvec,
)


PatchEdgeRec = Tuple[NodeInst, NodeInst, Dict[str, Any]]
PatchMultiEdgeRec = Tuple[NodeInst, NodeInst, int, Dict[str, Any]]


def _validate_box(
    box: Sequence[Sequence[int]],
    dim: int,
) -> Tuple[Tuple[int, int], ...]:
    if len(box) != dim:
        raise LiftPatchError('box dimension mismatch')
    out: List[Tuple[int, int]] = []
    for rng in box:
        if len(rng) != 2:
            raise LiftPatchError('box must be a sequence of (lo, hi) pairs')
        lo = int(rng[0])
        hi = int(rng[1])
        if hi < lo:
            raise LiftPatchError('box has invalid range (hi < lo)')
        out.append((lo, hi))
    return tuple(out)


def _intersect_boxes(
    a: Optional[Tuple[Tuple[int, int], ...]],
    b: Optional[Tuple[Tuple[int, int], ...]],
    dim: int,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    if a is None:
        return b
    if b is None:
        return a
    if len(a) != dim or len(b) != dim:
        raise LiftPatchError('box dimension mismatch')
    out: List[Tuple[int, int]] = []
    for (lo1, hi1), (lo2, hi2) in zip(a, b):
        lo = max(lo1, lo2)
        hi = min(hi1, hi2)
        if hi < lo:
            # Empty intersection: still return a valid box.
            out.append((lo, lo))
        else:
            out.append((lo, hi))
    return tuple(out)


def _in_box(shift: TVec, box: Optional[Tuple[Tuple[int, int], ...]]) -> bool:
    if box is None:
        return True
    for x, (lo, hi) in zip(shift, box):
        if x < lo or x >= hi:
            return False
    return True


def _try_sort_patch_edges(
    records: List[Tuple[Any, Any, int, Any]],
) -> None:
    """Sort patch edge candidates deterministically.

    Records are (u_inst, v_inst, key, payload).
    """
    try:
        records.sort(key=lambda r: (r[0], r[1], r[2]))
    except TypeError:
        records.sort(
            key=lambda r: (fallback_key(r[0]), fallback_key(r[1]), r[2])
        )


@dataclass(frozen=True)
class LiftPatch:
    """A finite patch extracted from the infinite lift.

    Attributes:
        nodes: Node instances `(u, shift)` in canonical order.
        edges: Edges between included node instances.

            - For simple containers: `(u_inst, v_inst, attrs)`.
            - For multigraph containers: `(u_inst, v_inst, key, attrs)`.

            For directed patches, `(u_inst, v_inst)` is ordered.
            For undirected patches, endpoints are in canonical order.
        seed: Seed node instance.
        radius: BFS radius in the lifted graph (weak connectivity), if used.
        box: Effective absolute box constraint after intersection, if used.
    """

    nodes: Tuple[NodeInst, ...]
    edges: Tuple[Union[PatchEdgeRec, PatchMultiEdgeRec], ...]
    seed: NodeInst
    radius: Optional[int]
    box: Optional[Tuple[Tuple[int, int], ...]]
    _is_multigraph: bool = False
    _is_directed: bool = False

    @property
    def is_multigraph(self) -> bool:
        """Whether the patch edges include keys."""
        return bool(self._is_multigraph)

    @property
    def is_directed(self) -> bool:
        """Whether the patch edges are directed."""
        return bool(self._is_directed)

    def to_networkx(
        self,
        *,
        as_undirected: Optional[bool] = None,
        undirected_mode: Literal['multigraph', 'orig_edges'] = 'multigraph',
    ) -> Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
        """Export the patch as a NetworkX graph.

        Notes:
            - By default, directed patches export as directed NetworkX graphs,
              and undirected patches export as undirected.
            - For directed patches, `as_undirected=True` provides an undirected
              view:
                - `undirected_mode='multigraph'` returns a MultiGraph where
                  each directed edge becomes a distinct undirected multiedge,
                  with direction metadata stored under the `__pbcgraph__`
                  edge attribute.
                - `undirected_mode='orig_edges'` returns a simple Graph where
                  each undirected adjacency stores `orig_edges=[...]`
                  snapshots under the `__pbcgraph__` edge attribute.
        """
        if as_undirected is None:
            as_undirected = not self.is_directed

        if not self.is_directed and as_undirected is False:
            raise ValueError('cannot export an undirected patch as directed')

        if not as_undirected:
            return _lift_patch_to_networkx_directed(self)

        if not self.is_directed:
            return _lift_patch_to_networkx_undirected(self)

        if undirected_mode == 'multigraph':
            return _lift_patch_to_networkx_directed_multigraph(self)

        if undirected_mode == 'orig_edges':
            return _lift_patch_to_networkx_directed_orig_edges(self)

        raise ValueError('invalid undirected_mode')


def _lift_patch_to_networkx_directed(
    patch: LiftPatch,
) -> Union[nx.DiGraph, nx.MultiDiGraph]:
    if patch.is_multigraph:
        Gd: Union[nx.DiGraph, nx.MultiDiGraph] = nx.MultiDiGraph()
    else:
        Gd = nx.DiGraph()

    for node in patch.nodes:
        Gd.add_node(node)

    if patch.is_multigraph:
        for u, v, key, attrs in patch.edges:  # type: ignore[misc]
            Gd.add_edge(u, v, key=int(key), **dict(attrs))
    else:
        for u, v, attrs in patch.edges:  # type: ignore[misc]
            Gd.add_edge(u, v, **dict(attrs))
    return Gd


def _lift_patch_to_networkx_undirected(
    patch: LiftPatch,
) -> Union[nx.Graph, nx.MultiGraph]:
    if patch.is_multigraph:
        Gu: Union[nx.Graph, nx.MultiGraph] = nx.MultiGraph()
    else:
        Gu = nx.Graph()

    for node in patch.nodes:
        Gu.add_node(node)

    if patch.is_multigraph:
        for u, v, key, attrs in patch.edges:  # type: ignore[misc]
            Gu.add_edge(u, v, key=int(key), **dict(attrs))
    else:
        for u, v, attrs in patch.edges:  # type: ignore[misc]
            Gu.add_edge(u, v, **dict(attrs))
    return Gu


def _lift_patch_to_networkx_directed_multigraph(
    patch: LiftPatch,
) -> nx.MultiGraph:
    Gu = nx.MultiGraph()
    for node in patch.nodes:
        Gu.add_node(node)

    if patch.is_multigraph:
        for u, v, key, attrs in patch.edges:  # type: ignore[misc]
            data = dict(attrs)
            data[PBC_META_KEY] = {
                'tail': u,
                'head': v,
                'key': int(key),
            }
            Gu.add_edge(u, v, **data)
    else:
        for u, v, attrs in patch.edges:  # type: ignore[misc]
            data = dict(attrs)
            data[PBC_META_KEY] = {
                'tail': u,
                'head': v,
                'key': None,
            }
            Gu.add_edge(u, v, **data)
    return Gu


def _lift_patch_to_networkx_directed_orig_edges(
    patch: LiftPatch,
) -> nx.Graph:
    Gu = nx.Graph()
    for node in patch.nodes:
        Gu.add_node(node)

    def _canon_pair(a: NodeInst, b: NodeInst) -> Tuple[NodeInst, NodeInst]:
        uu, vv = stable_sorted([a, b])
        return uu, vv

    buckets: Dict[Tuple[NodeInst, NodeInst], List[Dict[str, Any]]] = {}
    if patch.is_multigraph:
        for u, v, key, attrs in patch.edges:  # type: ignore[misc]
            a, b = _canon_pair(u, v)
            rec = {
                'tail': u,
                'head': v,
                'key': int(key),
                'attrs': dict(attrs),
            }
            buckets.setdefault((a, b), []).append(rec)
    else:
        for u, v, attrs in patch.edges:  # type: ignore[misc]
            a, b = _canon_pair(u, v)
            rec = {
                'tail': u,
                'head': v,
                'key': None,
                'attrs': dict(attrs),
            }
            buckets.setdefault((a, b), []).append(rec)

    for (a, b), recs in buckets.items():
        try:
            recs.sort(key=lambda r: (r['tail'], r['head'], r['key']))
        except TypeError:
            recs.sort(
                key=lambda r: (
                    fallback_key(r['tail']),
                    fallback_key(r['head']),
                    -1 if r['key'] is None else int(r['key']),
                )
            )
        Gu.add_edge(a, b, **{PBC_META_KEY: {'orig_edges': recs}})
    return Gu


def lift_patch(
    G: PeriodicDiGraphLike,
    seed: NodeInst,
    *,
    radius: Optional[int] = None,
    box: Optional[Tuple[Tuple[int, int], ...]] = None,
    box_rel: Optional[Tuple[Tuple[int, int], ...]] = None,
    include_edges: bool = True,
    max_nodes: Optional[int] = None,
    node_order: Optional[Callable[[NodeInst], Any]] = None,
    edge_order: Optional[Callable[[Tuple[Any, ...]], Any]] = None,
) -> LiftPatch:
    """Extract a finite patch of the lifted graph around a seed.

    The traversal uses weak connectivity in the infinite lift: from an instance
    it considers both outgoing and incoming quotient edges.

    Notes:
        The returned patch is directed if `G.is_undirected == False`, and
        undirected otherwise. Use `LiftPatch.to_networkx(as_undirected=True,
        ...)` to obtain undirected views of directed patches.


    Args:
        G: A periodic graph container.
        seed: Seed instance `(u, shift)`.
        radius: Optional BFS radius in the lifted graph.
        box: Optional absolute half-open bounds per coordinate.
        box_rel: Optional bounds relative to `seed.shift`.
        include_edges: Whether to include edges between included nodes.
        max_nodes: If provided, raise if the patch would include more than
            `max_nodes` nodes.
        node_order: Optional key function for ordering node instances.
        edge_order: Optional key function for ordering edge records.

    Returns:
        A :class:`~pbcgraph.alg.lift.LiftPatch`.

    Raises:
        LiftPatchError: On invalid inputs or if `max_nodes` is exceeded.
    """
    dim = int(G.dim)
    u0, s0 = seed
    validate_tvec(s0, dim)
    if radius is None and box is None and box_rel is None:
        raise LiftPatchError(
            'at least one of radius, box, or box_rel is required'
        )
    if radius is not None:
        radius = int(radius)
        if radius < 0:
            raise LiftPatchError('radius must be non-negative')

    abs_box: Optional[Tuple[Tuple[int, int], ...]] = None
    if box is not None:
        abs_box = _validate_box(box, dim)

    abs_box_rel: Optional[Tuple[Tuple[int, int], ...]] = None
    if box_rel is not None:
        rel = _validate_box(box_rel, dim)
        out: List[Tuple[int, int]] = []
        for (lo, hi), x0 in zip(rel, s0):
            out.append((int(x0) + lo, int(x0) + hi))
        abs_box_rel = tuple(out)

    eff_box = _intersect_boxes(abs_box, abs_box_rel, dim)
    if not _in_box(s0, eff_box):
        raise LiftPatchError('seed instance is outside the effective box')

    if max_nodes is not None:
        max_nodes = int(max_nodes)
        if max_nodes <= 0:
            raise LiftPatchError('max_nodes must be positive')

    # -----------------
    # Traversal
    # -----------------
    visited: Dict[NodeInst, int] = {seed: 0}
    q: deque[NodeInst] = deque([seed])

    def iter_weak_neighbors(inst: NodeInst) -> Iterator[NodeInst]:
        for v, s2 in G.neighbors_inst(inst, keys=False, data=False):
            yield v, s2
        if not G.is_undirected:
            for v, s2 in G.in_neighbors_inst(inst, keys=False, data=False):
                yield v, s2

    while q:
        cur = q.popleft()
        dcur = visited[cur]
        if radius is not None and dcur >= radius:
            continue

        for nb in iter_weak_neighbors(cur):
            _v, s2 = nb
            validate_tvec(s2, dim)
            if not _in_box(s2, eff_box):
                continue
            if nb in visited:
                continue
            visited[nb] = dcur + 1
            q.append(nb)
            if max_nodes is not None and len(visited) > max_nodes:
                raise LiftPatchError('max_nodes exceeded during traversal')

    # Canonical node order.
    nodes_list = list(visited.keys())
    if node_order is None:
        nodes = tuple(stable_sorted(nodes_list))
    else:
        nodes = tuple(sorted(nodes_list, key=node_order))

    patch_is_directed = not bool(G.is_undirected)

    # -----------------
    # Edge inclusion (no explicit tvec)
    # -----------------
    edges_out: List[Union[PatchEdgeRec, PatchMultiEdgeRec]] = []
    if include_edges:
        included_set = set(visited)

        if patch_is_directed:
            records: List[
                Tuple[NodeInst, NodeInst, int, Any, Dict[str, Any]]
            ] = []
            for inst in nodes:
                for v, s2, k, attrs in G.neighbors_inst(
                    inst, keys=True, data=True
                ):
                    nb = (v, s2)
                    if nb not in included_set:
                        continue
                    sel_key = (inst, nb, int(k))
                    sc = (
                        edge_order(sel_key)
                        if edge_order is not None
                        else sel_key
                    )
                    records.append((inst, nb, int(k), sc, dict(attrs)))

            try:
                records.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
            except TypeError:
                records.sort(
                    key=lambda r: (
                        fallback_key(r[3]),
                        fallback_key(r[0]),
                        fallback_key(r[1]),
                        r[2],
                    )
                )

            if G.is_multigraph:
                for u_inst, v_inst, kk, _sc, attrs in records:
                    edges_out.append((u_inst, v_inst, int(kk), dict(attrs)))
            else:
                for u_inst, v_inst, _kk, _sc, attrs in records:
                    edges_out.append((u_inst, v_inst, dict(attrs)))

        else:
            candidates: List[
                Tuple[NodeInst, NodeInst, int, Dict[str, Any]]
            ] = []
            for inst in nodes:
                for v, s2, k, attrs in G.neighbors_inst(
                    inst, keys=True, data=True
                ):
                    nb = (v, s2)
                    if nb not in included_set:
                        continue
                    candidates.append((inst, nb, int(k), dict(attrs)))

            # Canonicalize endpoints to undirected pairs.
            canon: List[Tuple[NodeInst, NodeInst, int, Dict[str, Any]]] = []
            for a, b, k, attrs in candidates:
                u_inst, v_inst = stable_sorted([a, b])
                canon.append((u_inst, v_inst, k, attrs))

            # Deduplicate reciprocal realizations deterministically.
            best: Dict[
                Tuple[NodeInst, NodeInst, Optional[int]],
                Tuple[Any, Dict[str, Any]],
            ] = {}
            for u_inst, v_inst, k, attrs in canon:
                if G.is_multigraph:
                    eid: Tuple[
                        NodeInst, NodeInst, Optional[int]
                    ] = (u_inst, v_inst, k)
                    sel_key = (u_inst, v_inst, k)
                else:
                    eid = (u_inst, v_inst, None)
                    sel_key = (u_inst, v_inst, k)

                score = (
                    edge_order(sel_key)
                    if edge_order is not None
                    else sel_key
                )

                if eid not in best:
                    best[eid] = (score, attrs)
                    continue
                prev_score, _prev_attrs = best[eid]
                try:
                    better = score < prev_score
                except TypeError:
                    better = fallback_key(score) < fallback_key(prev_score)
                if better:
                    best[eid] = (score, attrs)

            if G.is_multigraph:
                out_multi: List[Tuple[Any, Any, int, Any]] = []
                for (u_inst, v_inst, kk), (sc, attrs) in best.items():
                    assert kk is not None
                    out_multi.append((u_inst, v_inst, int(kk), (sc, attrs)))
                _try_sort_patch_edges(out_multi)
                for u_inst, v_inst, kk, payload in out_multi:
                    _sc, attrs = payload
                    edges_out.append((u_inst, v_inst, int(kk), dict(attrs)))
            else:
                out_simple: List[Tuple[Any, Any, int, Any]] = []
                for (u_inst, v_inst, _), (sc, attrs) in best.items():
                    out_simple.append((u_inst, v_inst, 0, (sc, attrs)))
                _try_sort_patch_edges(out_simple)
                for u_inst, v_inst, _kk, payload in out_simple:
                    _sc, attrs = payload
                    edges_out.append((u_inst, v_inst, dict(attrs)))
    return LiftPatch(
        nodes=nodes,
        edges=tuple(edges_out),
        seed=seed,
        radius=radius,
        box=eff_box,
        _is_multigraph=bool(G.is_multigraph),
        _is_directed=patch_is_directed,
    )
