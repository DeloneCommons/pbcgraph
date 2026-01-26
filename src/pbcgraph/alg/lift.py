"""Finite lifts of periodic graphs.

This module implements finite, non-periodic views derived from a periodic
quotient graph.

v0.1.2 adds two high-level operations:

1) ``lift_patch``: extract a finite undirected patch of the infinite lift
   around a seed instance.

2) ``canonical_lift`` (added in later steps of the v0.1.2 plan).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx

from pbcgraph.core.exceptions import CanonicalLiftError, LiftPatchError
from pbcgraph.core.ordering import fallback_key, stable_sorted
from pbcgraph.core.protocols import PeriodicDiGraphLike
from pbcgraph.core.types import (
    NodeId,
    NodeInst,
    TVec,
    add_tvec,
    sub_tvec,
    zero_tvec,
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
        records.sort(key=lambda r: (fallback_key(r[0]), fallback_key(r[1]), r[2]))


@dataclass(frozen=True)
class LiftPatch:
    """A finite undirected patch extracted from the infinite lift.

    Attributes:
        nodes: Node instances `(u, shift)` in canonical order.
        edges: Undirected edges between included node instances.
            - For simple containers: `(u_inst, v_inst, attrs)`.
            - For multigraph containers: `(u_inst, v_inst, key, attrs)`.
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

    def to_networkx(self) -> Union[nx.Graph, nx.MultiGraph]:
        """Export the patch as a NetworkX graph."""
        if self._is_multigraph:
            G: Union[nx.Graph, nx.MultiGraph] = nx.MultiGraph()
        else:
            G = nx.Graph()

        for node in self.nodes:
            G.add_node(node)

        if self._is_multigraph:
            for u, v, key, attrs in self.edges:  # type: ignore[misc]
                G.add_edge(u, v, key=int(key), **dict(attrs))
        else:
            for u, v, attrs in self.edges:  # type: ignore[misc]
                G.add_edge(u, v, **dict(attrs))
        return G


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
    """Extract a finite undirected patch of the lifted graph around a seed.

    The traversal uses weak connectivity in the infinite lift: from an instance
    it considers both outgoing and incoming quotient edges.

    Notes:
        The returned patch is undirected. When extracting from a directed
        periodic graph, distinct directed edges can map to the same undirected
        patch adjacency. In such cases, only one edge attribute snapshot is
        retained deterministically.

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
        raise LiftPatchError('at least one of radius, box, or box_rel is required')
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

    # -----------------
    # Edge inclusion (undirected, no explicit tvec)
    # -----------------
    edges_out: List[Union[PatchEdgeRec, PatchMultiEdgeRec]] = []
    if include_edges:
        included_set = set(visited)

        candidates: List[Tuple[NodeInst, NodeInst, int, Dict[str, Any]]] = []
        for inst in nodes:
            for v, s2, k, attrs in G.neighbors_inst(inst, keys=True, data=True):
                nb = (v, s2)
                if nb not in included_set:
                    continue
                candidates.append((inst, nb, int(k), dict(attrs)))
            for v, s2, k, attrs in G.in_neighbors_inst(inst, keys=True, data=True):
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
        best: Dict[Tuple[NodeInst, NodeInst, Optional[int]], Tuple[int, Dict[str, Any]]] = {}
        for u_inst, v_inst, k, attrs in canon:
            if G.is_multigraph:
                eid: Tuple[NodeInst, NodeInst, Optional[int]] = (u_inst, v_inst, k)
                sel_key = (u_inst, v_inst, k)
            else:
                eid = (u_inst, v_inst, None)
                sel_key = (u_inst, v_inst, k)

            if edge_order is not None:
                score = edge_order(sel_key)
            else:
                score = sel_key

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
    )


TreeEdgeRec = Tuple[NodeId, NodeId, TVec, int]


@dataclass(frozen=True)
class CanonicalLift:
    """A deterministic finite representation of a single strand.

    Attributes:
        nodes: Node instances `(u, shift)` in canonical order. Contains
            exactly one instance for every quotient node in the component.
        strand_key: Target strand (coset) key in `Z^d / L`.
        anchor_site: Quotient node chosen to be placed in `anchor_shift`.
        anchor_shift: Anchor cell translation vector.
        placement: Placement mode used to construct the lift.
        score: Placement score (smaller is better; 0 is best).
        tree_edges: Optional spanning-tree edge records for debugging.
    """

    nodes: Tuple[NodeInst, ...]
    strand_key: Hashable
    anchor_site: NodeId
    anchor_shift: TVec
    placement: str
    score: Union[int, float]
    tree_edges: Optional[Tuple[TreeEdgeRec, ...]] = None


def _sorted_nodes_by_key(
    nodes: Sequence[NodeId],
    node_order: Optional[Callable[[NodeId], Any]],
) -> Tuple[NodeId, ...]:
    seq = list(nodes)
    if not seq:
        return ()

    if node_order is None:
        return tuple(stable_sorted(seq))

    def k(u: NodeId) -> Any:
        return node_order(u)

    try:
        return tuple(sorted(seq, key=lambda u: (k(u), fallback_key(u))))
    except TypeError:
        return tuple(sorted(seq, key=lambda u: (fallback_key(k(u)), fallback_key(u))))


def _sorted_node_insts(
    insts: Sequence[NodeInst],
    node_order: Optional[Callable[[NodeId], Any]],
) -> Tuple[NodeInst, ...]:
    seq = list(insts)
    if not seq:
        return ()

    if node_order is None:
        try:
            return tuple(sorted(seq, key=lambda x: (x[0], x[1])))
        except TypeError:
            return tuple(sorted(seq, key=lambda x: (fallback_key(x[0]), x[1])))

    def k(u: NodeId) -> Any:
        return node_order(u)

    try:
        return tuple(sorted(seq, key=lambda x: (k(x[0]), x[1], fallback_key(x[0]))))
    except TypeError:
        return tuple(
            sorted(seq, key=lambda x: (fallback_key(k(x[0])), x[1], fallback_key(x[0])))
        )



def _compute_lift_score(
    snf: Any,
    rel_shifts: Dict[NodeId, TVec],
    nodes: Sequence[NodeId],
    score: Literal['l1', 'l2'],
) -> int:
    """Compute placement score for a lift.

    Args:
        snf: SNF decomposition of the component translation subgroup.
        rel_shifts: Per-node relative shifts with respect to the anchor site.
        nodes: Quotient node ids in the component.
        score: Score metric: 'l1' or 'l2'.

    Returns:
        The deterministic integer score (smaller is better).

    Raises:
        CanonicalLiftError: If the SNF decomposition is invalid.
    """
    r = int(snf.rank)
    total = 0
    for u in nodes:
        y = snf.apply_U(rel_shifts[u])
        node_mag = 0
        for i in range(r):
            di = int(snf.diag[i])
            if di == 0:
                raise CanonicalLiftError('invalid SNF diagonal entry')
            qi = int(y[i] // di)
            if score == 'l1':
                node_mag += abs(qi)
            else:
                node_mag += qi * qi
        total += node_mag
    return int(total)


def _compute_rel_abs_shifts(
    pot: Dict[NodeId, TVec],
    *,
    anchor_site: NodeId,
    anchor_shift: TVec,
) -> Tuple[Dict[NodeId, TVec], Dict[NodeId, TVec]]:
    """Compute relative and absolute shifts for a given anchor site."""
    pot_anchor = pot[anchor_site]
    rel: Dict[NodeId, TVec] = {}
    abs_s: Dict[NodeId, TVec] = {}
    for u, pu in pot.items():
        r = sub_tvec(pu, pot_anchor)
        rel[u] = r
        abs_s[u] = add_tvec(anchor_shift, r)
    return rel, abs_s


def _build_internal_adj(
    component: Any,
    abs_shift: Dict[NodeId, TVec],
) -> Dict[NodeId, FrozenSet[NodeId]]:
    """Build induced internal undirected adjacency on selected instances.

    An undirected adjacency between quotient nodes `u` and `v` exists if at
    least one directed periodic edge between them is consistent with the
    selected absolute shifts.

    Args:
        component: PeriodicComponent.
        abs_shift: Mapping `u -> shift` for exactly the component nodes.

    Returns:
        Dict mapping node id to a frozen set of adjacent node ids.
    """
    adj: Dict[NodeId, set[NodeId]] = {u: set() for u in component.nodes}
    for u in component.nodes:
        su = abs_shift[u]
        for v, t, _k in component.graph.neighbors(u, keys=True, data=False):
            if v not in component.nodes:
                continue
            if abs_shift[v] == add_tvec(su, t):
                adj[u].add(v)
                adj[v].add(u)
    return {u: frozenset(nbs) for u, nbs in adj.items()}


def _is_connected_undirected(
    adj: Dict[NodeId, FrozenSet[NodeId]],
    nodes_ordered: Sequence[NodeId],
    *,
    skip: Optional[NodeId] = None,
) -> bool:
    """Return True if the induced graph is connected (optionally skipping a node)."""
    nodes = [u for u in nodes_ordered if u != skip]
    if not nodes:
        return True

    start = nodes[0]
    seen: set[NodeId] = {start}
    q: deque[NodeId] = deque([start])

    while q:
        u = q.popleft()
        for v in stable_sorted(list(adj.get(u, frozenset()))):
            if v == skip:
                continue
            if v in seen:
                continue
            seen.add(v)
            q.append(v)
    return len(seen) == len(nodes)


def _boundary_deltas_for_node(
    component: Any,
    abs_shift: Dict[NodeId, TVec],
    u: NodeId,
) -> Tuple[TVec, ...]:
    """Enumerate per-node deltas induced by boundary periodic edges."""
    su = abs_shift[u]
    deltas: set[TVec] = set()

    for v, t, _k in component.graph.neighbors(u, keys=True, data=False):
        if v not in component.nodes:
            continue
        desired = add_tvec(su, t)
        if abs_shift[v] == desired:
            continue
        # Want: abs_shift[v] == (su + delta) + t
        delta = sub_tvec(sub_tvec(abs_shift[v], su), t)
        deltas.add(delta)

    for v, t_in, _k in component.graph.in_neighbors(u, keys=True, data=False):
        if v not in component.nodes:
            continue
        desired_u = add_tvec(abs_shift[v], t_in)
        if desired_u == su:
            continue
        # Want: (su + delta) == abs_shift[v] + t_in
        delta = sub_tvec(desired_u, su)
        deltas.add(delta)

    if not deltas:
        return ()

    try:
        return tuple(sorted(deltas))
    except TypeError:
        return tuple(sorted(deltas, key=fallback_key))


def canonical_lift(
    component: Any,
    *,
    strand_key: Optional[Hashable] = None,
    seed: Optional[NodeInst] = None,
    anchor_shift: Optional[TVec] = None,
    placement: Literal['tree', 'best_anchor', 'greedy_cut'] = 'tree',
    score: Literal['l1', 'l2'] = 'l1',
    return_tree: bool = False,
    node_order: Optional[Callable[[NodeId], Any]] = None,
    edge_order: Optional[Callable[[Tuple[Any, ...]], Any]] = None,
) -> CanonicalLift:
    """Construct a deterministic finite representation of one strand.

    v0.1.2 step4 implements `placement='tree'`, `placement='best_anchor'`, and
    `placement='greedy_cut'`.

    Args:
        component: A :class:`~pbcgraph.component.PeriodicComponent`.
        strand_key: Optional explicit strand key.
        seed: Optional seed instance `(u, shift)`.
        anchor_shift: Optional anchor cell shift.
        placement: Placement mode (`'tree'` in step2).
        score: Score metric: `'l1'` or `'l2'`.
        return_tree: If True, include spanning-tree edge records.
        node_order: Optional ordering key for quotient node ids.
        edge_order: Optional ordering key for periodic edges (reserved).

    Returns:
        A :class:`~pbcgraph.alg.lift.CanonicalLift`.

    Raises:
        CanonicalLiftError: On invalid inputs or if the requested strand does
            not intersect the anchor cell.
    """
    del edge_order  # Reserved for later placement modes.

    if placement not in ('tree', 'best_anchor', 'greedy_cut'):
        raise CanonicalLiftError(
            "canonical_lift placement must be one of 'tree', 'best_anchor', 'greedy_cut'"
        )

    dim = int(component.graph.dim)

    if seed is not None:
        u_seed, s_seed = seed
        validate_tvec(s_seed, dim)
    else:
        u_seed = None
        s_seed = None

    if anchor_shift is None:
        if s_seed is not None:
            anchor_shift = s_seed
        else:
            anchor_shift = zero_tvec(dim)
    else:
        validate_tvec(anchor_shift, dim)

    if strand_key is None:
        if seed is not None:
            try:
                K = component.inst_key(seed)
            except KeyError as e:
                raise CanonicalLiftError('seed does not belong to component') from e
        else:
            nodes_sorted = _sorted_nodes_by_key(list(component.nodes), node_order)
            if not nodes_sorted:
                raise CanonicalLiftError('component has no nodes')
            default_seed = (nodes_sorted[0], zero_tvec(dim))
            K = component.inst_key(default_seed)
    else:
        K = strand_key

    eligible: List[NodeId] = []
    for u in component.nodes:
        if component.inst_key((u, anchor_shift)) == K:
            eligible.append(u)

    if not eligible:
        raise CanonicalLiftError(
            'requested strand_key does not intersect the anchor cell'
        )

    pot = {u: component.potential(u) for u in component.nodes}

    snf = component._snf
    if snf is None:
        raise CanonicalLiftError('component has no SNF decomposition')

    if score not in ('l1', 'l2'):
        raise CanonicalLiftError("score must be 'l1' or 'l2'")

    nodes_list = list(component.nodes)
    eligible_sorted = _sorted_nodes_by_key(eligible, node_order)

    if placement == 'tree':
        anchor_site = eligible_sorted[0]
        rel_shift, abs_shift = _compute_rel_abs_shifts(
            pot,
            anchor_site=anchor_site,
            anchor_shift=anchor_shift,
        )
        total_score = _compute_lift_score(snf, rel_shift, nodes_list, score)
    else:
        best_anchor_site: Optional[NodeId] = None
        best_rel: Optional[Dict[NodeId, TVec]] = None
        best_abs: Optional[Dict[NodeId, TVec]] = None
        best_score: Optional[int] = None

        for a in eligible_sorted:
            rel_a, abs_a = _compute_rel_abs_shifts(
                pot,
                anchor_site=a,
                anchor_shift=anchor_shift,
            )
            s = _compute_lift_score(snf, rel_a, nodes_list, score)
            if best_score is None or s < best_score:
                best_score = int(s)
                best_anchor_site = a
                best_rel = rel_a
                best_abs = abs_a

        if best_anchor_site is None or best_rel is None or best_abs is None:
            raise CanonicalLiftError('failed to select anchor site')

        anchor_site = best_anchor_site
        rel_shift = best_rel
        abs_shift = best_abs
        total_score = int(best_score)

    if placement == 'greedy_cut':
        # Start from the best-anchor placement and perform local, per-node
        # moves by elements of the translation subgroup L that improve score
        # while keeping the induced internal graph connected.
        nodes_sorted = _sorted_nodes_by_key(list(component.nodes), node_order)
        abs_cur: Dict[NodeId, TVec] = dict(abs_shift)
        cur_score = int(total_score)

        while True:
            moved = False
            adj = _build_internal_adj(component, abs_cur)
            if not _is_connected_undirected(adj, nodes_sorted):
                raise CanonicalLiftError('internal induced graph is disconnected')

            for u in nodes_sorted:
                if u == anchor_site:
                    continue
                deltas = _boundary_deltas_for_node(component, abs_cur, u)
                if not deltas:
                    continue

                # Pre-filter: u must not be an articulation point of the
                # current internal graph.
                if not _is_connected_undirected(adj, nodes_sorted, skip=u):
                    continue

                best_move: Optional[Tuple[int, TVec]] = None
                old_s = abs_cur[u]

                for delta in deltas:
                    new_s = add_tvec(old_s, delta)
                    if component.inst_key((u, new_s)) != K:
                        continue

                    abs_cur[u] = new_s
                    new_adj = _build_internal_adj(component, abs_cur)
                    ok = True
                    if not new_adj.get(u, frozenset()):
                        ok = False
                    elif not _is_connected_undirected(new_adj, nodes_sorted):
                        ok = False

                    if ok:
                        rel_tmp = {
                            x: sub_tvec(abs_cur[x], abs_cur[anchor_site])
                            for x in component.nodes
                        }
                        s = _compute_lift_score(snf, rel_tmp, nodes_list, score)
                        if s < cur_score:
                            if best_move is None:
                                best_move = (int(s), delta)
                            else:
                                best_s, best_delta = best_move
                                if int(s) < best_s or (
                                    int(s) == best_s and delta < best_delta
                                ):
                                    best_move = (int(s), delta)

                    abs_cur[u] = old_s

                if best_move is not None:
                    best_s, best_delta = best_move
                    abs_cur[u] = add_tvec(abs_cur[u], best_delta)
                    cur_score = int(best_s)
                    moved = True
                    break

            if not moved:
                break

        abs_shift = abs_cur
        total_score = int(cur_score)

    insts = [(u, abs_shift[u]) for u in component.nodes]
    insts_sorted = _sorted_node_insts(insts, node_order)

    tree_edges: Optional[Tuple[TreeEdgeRec, ...]] = None
    if return_tree:
        recs: List[TreeEdgeRec] = []
        children = _sorted_nodes_by_key(list(component._tree_parent.keys()), node_order)
        for child in children:
            parent, _t, k = component._tree_parent[child]
            tvec = sub_tvec(abs_shift[child], abs_shift[parent])
            recs.append((parent, child, tvec, int(k)))
        tree_edges = tuple(recs)

    return CanonicalLift(
        nodes=insts_sorted,
        strand_key=K,
        anchor_site=anchor_site,
        anchor_shift=anchor_shift,
        placement=placement,
        score=int(total_score),
        tree_edges=tree_edges,
    )
