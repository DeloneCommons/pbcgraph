"""Directed periodic graph container."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import networkx as nx

if TYPE_CHECKING:
    from pbcgraph.alg.lift import LiftPatch
    from pbcgraph.component import PeriodicComponent

from pbcgraph.alg.components import components as _components
from pbcgraph.core.ordering import (
    fallback_key,
    stable_sorted,
    stable_tvec,
    try_sort_edges,
)
from pbcgraph.core.types import (
    EdgeKey,
    NodeId,
    NodeInst,
    TVec,
    add_tvec,
    neg_tvec,
    sub_tvec,
    validate_tvec,
)
from pbcgraph.graph.shared import (
    _TVEC_ATTR,
    _USER_ATTRS,
    base_key as _base_key,
    check_reserved_edge_attrs as _check_reserved_edge_attrs,
    ro as _ro,
    validate_edge_key as _validate_edge_key,
)


class PeriodicDiGraph:
    """Directed periodic graph on ``Z^d``.

    The quotient is stored as a NetworkX :class:`networkx.MultiDiGraph`, but
    this container enforces an important invariant:

    *For any fixed triple ``(u, v, tvec)``, at most one edge exists.*

    This means that the translation vector is treated as part of the edge
    identity. Parallel edges between the same ordered node pair are still
    possible as long as their translation vectors differ.

    Attributes:
        structural_version: Incremented when the quotient structure changes
            (nodes/edges added or removed).
        data_version: Incremented when user data changes without structural
            changes (node/edge attribute updates).

    Notes:
        - Quotient nodes are `NodeId` values.
        - Each directed edge stores a translation vector (``TVec``) that
          describes how the cell shift changes when traversing that edge in the
          infinite periodic lift.
    """

    def __init__(self, dim: int = 3):
        self._dim = int(dim)
        if self._dim <= 0:
            raise ValueError('dim must be positive')
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.structural_version: int = 0
        self.data_version: int = 0

    @property
    def dim(self) -> int:
        """Lattice dimension `d`."""
        return self._dim

    @property
    def is_undirected(self) -> bool:
        """Whether this container should be treated as undirected
        by algorithms."""
        return False

    @property
    def is_multigraph(self) -> bool:
        """Whether this container allows multiple edges per `(u, v, tvec)`."""
        return False

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def number_of_nodes(self) -> int:
        """Return the number of quotient nodes."""
        return self._g.number_of_nodes()

    def number_of_edges(self) -> int:
        """Return the number of directed quotient edges
        (counts parallel edges)."""
        return self._g.number_of_edges()

    # -----------------
    # Nodes
    # -----------------
    def add_node(self, u: NodeId, **attrs: Any) -> None:
        """Add a quotient node.

        Args:
            u: Node id.
            **attrs: User attributes.

        Notes:
            - Increments `structural_version` if the node is new.
            - If the node already exists and `attrs` are provided,
              this updates attributes and increments `data_version`.
            - If the node is new, attributes provided at creation do not
              increment `data_version` (pure structural change semantics).
        """
        exists = self._g.has_node(u)
        if not exists:
            self._g.add_node(u)
            self.structural_version += 1
        if attrs:
            self._g.nodes[u].update(attrs)
            if exists:
                self.data_version += 1

    def remove_node(self, u: NodeId) -> None:
        """Remove a quotient node and all incident edges.

        Args:
            u: Node id.

        Raises:
            KeyError: If `u` is not present.
        """
        if not self._g.has_node(u):
            raise KeyError(u)
        self._g.remove_node(u)
        self.structural_version += 1

    def has_node(self, u: NodeId) -> bool:
        """Return True if the node exists."""
        return self._g.has_node(u)

    def nodes(self, data: bool = False) -> Iterable:
        """Iterate quotient nodes in deterministic order.

        Args:
            data: If True, yield `(u, attrs)` where `attrs` is a read-only
                live view of the node attribute mapping.

        Returns:
            Iterable of node ids or `(node, attrs)` pairs.
        """
        nodes = stable_sorted(self._g.nodes)
        if not data:
            return iter(nodes)
        return ((u, _ro(self._g.nodes[u])) for u in nodes)

    def get_node_data(self, u: NodeId) -> MappingProxyType:
        """Return a read-only live view of the node attribute mapping.

        Args:
            u: Node id.

        Raises:
            KeyError: If node is missing.
        """
        return _ro(self._g.nodes[u])

    def set_node_attrs(self, u: NodeId, **attrs: Any) -> None:
        """Update node attributes and increment `data_version`.

        Args:
            u: Node id.
            **attrs: Attributes to set.

        Raises:
            KeyError: If node is missing.
        """
        if not self._g.has_node(u):
            raise KeyError(u)
        if attrs:
            self._g.nodes[u].update(attrs)
            self.data_version += 1

    # -----------------
    # Edges
    # -----------------
    def _alloc_key_directed(self, u: NodeId, v: NodeId) -> EdgeKey:
        """Allocate a new edge key for a directed edge (u -> v).

        Mirrors NetworkX's ``new_edge_key`` behavior: start from
        ``len(keys)``, then increment until unused.
        """
        kd = self._g.get_edge_data(u, v)
        if not kd:
            return 0
        k = len(kd)
        while k in kd:
            k += 1
        return int(k)

    def _alloc_key_undirected(self, u: NodeId, v: NodeId) -> EdgeKey:
        """Allocate a new edge key for an undirected edge between u and v.

        Keys are allocated in *public base-key* space. Undirected containers
        store directed realizations using private `_UKey` objects, so the
        internal MultiDiGraph keys are not necessarily ints.
        """
        kd_uv = self._g.get_edge_data(u, v) or {}
        kd_vu = self._g.get_edge_data(v, u) or {}
        used = {_base_key(k) for k in kd_uv} | {_base_key(k) for k in kd_vu}
        if not used:
            return 0
        k = len(used)
        while k in used:
            k += 1
        return int(k)

    def _key_for_tvec(
        self, u: NodeId, v: NodeId, tvec: TVec
    ) -> Optional[EdgeKey]:
        """Return an existing edge key for a given directed ``(u, v, tvec)``.

        Args:
            u: Source node.
            v: Target node.
            tvec: Translation vector.

        Returns:
            The corresponding edge key if an edge with this translation exists,
            otherwise None.
        """
        if not self._g.has_node(u):
            return None
        adj = self._g.adj[u]
        if v not in adj:
            return None
        want = stable_tvec(tvec)
        for k, ed in adj[v].items():
            if tuple(ed[_TVEC_ATTR]) == want:
                return _base_key(k)
        return None

    def _add_edge_impl(
        self,
        u: NodeId,
        v: NodeId,
        tvec: TVec,
        *,
        key: Optional[EdgeKey],
        attrs: Dict[str, Any],
    ) -> EdgeKey:
        """Implementation for adding a directed edge
        (no (u, v, tvec) checks)."""
        validate_tvec(tvec, self._dim)
        if not self._g.has_node(u):
            self.add_node(u)
        if not self._g.has_node(v):
            self.add_node(v)

        if key is None:
            key = self._alloc_key_directed(u, v)
        else:
            _validate_edge_key(key)

        # Disallow overwriting an existing directed edge id.
        if self._g.has_edge(u, v, key=key):
            raise KeyError((u, v, key))

        tvec_norm = stable_tvec(tvec)
        user_attrs: Dict[str, Any] = dict(attrs)
        _check_reserved_edge_attrs(user_attrs)
        self._g.add_edge(
            u, v, key=key, **{_TVEC_ATTR: tvec_norm, _USER_ATTRS: user_attrs}
        )
        self.structural_version += 1
        return key

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        tvec: TVec,
        key: Optional[EdgeKey] = None,
        **attrs: Any,
    ) -> EdgeKey:
        """Add a directed periodic edge.

        Args:
            u: Source node id.
            v: Target node id.
            tvec: Translation vector in Z^d.
            key: Optional explicit edge key. If None, a fresh deterministic
                key is assigned.
            **attrs: User attributes.

        Returns:
            The edge key used.

        Raises:
            ValueError: If `tvec` has wrong dimension.
        """
        validate_tvec(tvec, self._dim)

        existing = self._key_for_tvec(u, v, tvec)
        if existing is not None:
            raise ValueError(
                'edge already exists for (u, v, tvec): '
                f'({u!r}, {v!r}, {tuple(tvec)!r}); key={existing!r}'
            )

        return self._add_edge_impl(u, v, tvec, key=key, attrs=dict(attrs))

    def has_edge(
        self, u: NodeId, v: NodeId, key: Optional[EdgeKey] = None
    ) -> bool:
        """Return True if a directed edge exists.

        Args:
            u: Source node id.
            v: Target node id.
            key: If provided, check existence of that specific edge key.

        Returns:
            True if edge exists.
        """
        if key is None:
            return self._g.has_edge(u, v)
        return self._g.has_edge(u, v, key=key)

    def edge_tvec(self, u: NodeId, v: NodeId, key: EdgeKey) -> TVec:
        """Return the structural translation vector for an edge."""
        data = self._g.get_edge_data(u, v, key)
        if data is None:
            raise KeyError((u, v, key))
        return stable_tvec(data[_TVEC_ATTR])

    def get_edge_data(
        self, u: NodeId, v: NodeId, key: EdgeKey, default: Any = None
    ) -> Any:
        """Return a read-only live view of the user attribute mapping.

        Args:
            u: Source node id.
            v: Target node id.
            key: Edge key.
            default: Value to return if edge is missing.

        Returns:
            A read-only live view of the user attribute mapping, or `default`
            if missing.
        """
        data = self._g.get_edge_data(u, v, key)
        if data is None:
            return default
        return _ro(data[_USER_ATTRS])

    def set_edge_attrs(
        self, u: NodeId, v: NodeId, key: EdgeKey, **attrs: Any
    ) -> None:
        """Update user attributes for an edge and increment `data_version`."""
        data = self._g.get_edge_data(u, v, key)
        if data is None:
            raise KeyError((u, v, key))
        if attrs:
            _check_reserved_edge_attrs(attrs)
            data[_USER_ATTRS].update(attrs)
            self.data_version += 1

    def remove_edge(self, u: NodeId, v: NodeId, key: EdgeKey) -> None:
        """Remove a directed edge.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self._g.has_edge(u, v, key=key):
            raise KeyError((u, v, key))
        self._g.remove_edge(u, v, key=key)
        self.structural_version += 1

    def edges(
        self, keys: bool = False, data: bool = False, tvec: bool = False
    ) -> Iterable:
        """Iterate directed edges in deterministic order.

        Args:
            keys: If True, include the multiedge key.
            data: If True, include the read-only user attribute mapping.
            tvec: If True, include the translation vector.

        Returns:
            An iterable of:
                - `(u, v)`
                - `(u, v, attrs)`
                - `(u, v, key)`
                - `(u, v, tvec)`
                - `(u, v, tvec, key)`
                - `(u, v, key, attrs)`
                - `(u, v, tvec, attrs)`
                - `(u, v, tvec, key, attrs)`
        """
        # Streaming deterministic iteration:
        # iterate u, then v, then edges on (u, v) ordered by (tvec, key).
        for u in stable_sorted(list(self._g.nodes)):
            adj = self._g.adj[u]
            for v in stable_sorted(list(adj.keys())):
                kd = adj[v]
                items: List[Tuple[Tuple[int, ...], int, Any]] = []
                for ik, ed in kd.items():
                    items.append(
                        (
                            stable_tvec(ed[_TVEC_ATTR]),
                            _base_key(ik),
                            ed[_USER_ATTRS],
                        )
                    )
                items.sort(key=lambda r: (r[0], r[1]))
                for tv, k, attrs in items:
                    if data:
                        attrs_ro = _ro(attrs)
                    if tvec:
                        if keys:
                            if data:
                                yield u, v, tv, k, attrs_ro
                            else:
                                yield u, v, tv, k
                        else:
                            if data:
                                yield u, v, tv, attrs_ro
                            else:
                                yield u, v, tv
                    else:
                        if keys:
                            if data:
                                yield u, v, k, attrs_ro
                            else:
                                yield u, v, k
                        else:
                            if data:
                                yield u, v, attrs_ro
                            else:
                                yield u, v

    def undirected_edges_unique(
        self, keys: bool = False, data: bool = False, tvec: bool = False
    ) -> Iterable:
        """Iterate unique undirected edges in deterministic order.

        This iterator is only defined for undirected containers
        (`PeriodicGraph` / `PeriodicMultiGraph`). It returns each undirected
        quotient edge exactly once, in a canonical orientation.

        Canonicalization rules:
            - For `u != v`, the returned endpoints satisfy `u <= v` under the
              same ordering policy used elsewhere in pbcgraph.
            - For quotient self-loops with nonzero translation, the returned
              translation vector is canonicalized to `min(tvec, -tvec)`.

        Args:
            keys: If True, include the public base edge key.
            data: If True, include the read-only user attribute mapping.
            tvec: If True, include the translation vector.

        Returns:
            An iterable of:
                - `(u, v)`
                - `(u, v, attrs)`
                - `(u, v, key)`
                - `(u, v, tvec)`
                - `(u, v, tvec, key)`
                - `(u, v, key, attrs)`
                - `(u, v, tvec, attrs)`
                - `(u, v, tvec, key, attrs)`

        Raises:
            TypeError: If called on a directed container.
        """
        if not self.is_undirected:
            raise TypeError(
                'undirected_edges_unique is only available for '
                'undirected containers'
            )

        records: List[Tuple[Any, Any, Tuple[int, ...], int, Any]] = []
        seen: set[Tuple[Any, Any, Tuple[int, ...], int]] = set()

        for u, v, k, edata in self._g.edges(keys=True, data=True):
            base = int(_base_key(k))
            tv = stable_tvec(edata[_TVEC_ATTR])

            if u == v:
                tv_neg = stable_tvec(neg_tvec(tv))
                tv_abs = tv if tv <= tv_neg else tv_neg
                ident = (u, u, tv_abs, base)
                if ident in seen:
                    continue
                seen.add(ident)
                records.append((u, u, tv_abs, base, edata[_USER_ATTRS]))
                continue

            try:
                leq = u <= v  # type: ignore[operator]
            except TypeError:
                leq = fallback_key(u) <= fallback_key(v)

            if leq:
                a, b, tv_use = u, v, tv
            else:
                a, b, tv_use = v, u, stable_tvec(neg_tvec(tv))

            ident = (a, b, tv_use, base)
            if ident in seen:
                continue
            seen.add(ident)
            records.append((a, b, tv_use, base, edata[_USER_ATTRS]))

        try_sort_edges(records)

        for u, v, tv, k, attrs in records:
            if data:
                attrs_ro = _ro(attrs)
            if tvec:
                if keys:
                    if data:
                        yield u, v, tv, k, attrs_ro
                    else:
                        yield u, v, tv, k
                else:
                    if data:
                        yield u, v, tv, attrs_ro
                    else:
                        yield u, v, tv
            else:
                if keys:
                    if data:
                        yield u, v, k, attrs_ro
                    else:
                        yield u, v, k
                else:
                    if data:
                        yield u, v, attrs_ro
                    else:
                        yield u, v

    # -----------------
    # Neighborhoods
    # -----------------
    def neighbors(
        self, u: NodeId, keys: bool = False, data: bool = False
    ) -> Iterable:
        """Iterate outgoing periodic edges from quotient node `u`.

        Yields:
            Depending on flags:
            - `(v, tvec)`
            - `(v, tvec, key)`
            - `(v, tvec, attrs)`
            - `(v, tvec, key, attrs)`
        """
        if not self._g.has_node(u):
            raise KeyError(u)

        adj = self._g.adj[u]
        for v in stable_sorted(list(adj.keys())):
            kd = adj[v]
            items: List[Tuple[Tuple[int, ...], int, Any]] = []
            for ik, ed in kd.items():
                items.append(
                    (
                        stable_tvec(ed[_TVEC_ATTR]),
                        _base_key(ik),
                        ed[_USER_ATTRS],
                    )
                )
            items.sort(key=lambda r: (r[0], r[1]))
            for tv, k, attrs in items:
                if data:
                    attrs_ro = _ro(attrs)
                if keys:
                    if data:
                        yield v, tv, k, attrs_ro
                    else:
                        yield v, tv, k
                else:
                    if data:
                        yield v, tv, attrs_ro
                    else:
                        yield v, tv

    def in_neighbors(
        self, u: NodeId, keys: bool = False, data: bool = False
    ) -> Iterable:
        """Iterate incoming periodic edges into quotient node `u`.

        The returned translation vector is the one stored on the directed edge
        ``v -> u`` (i.e. *not* negated).

        Yields:
            Depending on flags:
            - `(v, tvec)`
            - `(v, tvec, key)`
            - `(v, tvec, attrs)`
            - `(v, tvec, key, attrs)`
        """
        if not self._g.has_node(u):
            raise KeyError(u)

        pred_adj = self._g.pred[u]
        for v in stable_sorted(list(pred_adj.keys())):
            kd = pred_adj[v]
            items: List[Tuple[Tuple[int, ...], int, Any]] = []
            for ik, ed in kd.items():
                items.append(
                    (
                        stable_tvec(ed[_TVEC_ATTR]),
                        _base_key(ik),
                        ed[_USER_ATTRS],
                    )
                )
            items.sort(key=lambda r: (r[0], r[1]))
            for tv, k, attrs in items:
                if data:
                    attrs_ro = _ro(attrs)
                if keys:
                    if data:
                        yield v, tv, k, attrs_ro
                    else:
                        yield v, tv, k
                else:
                    if data:
                        yield v, tv, attrs_ro
                    else:
                        yield v, tv

    def neighbors_inst(
        self, node_inst: NodeInst, keys: bool = False, data: bool = False
    ) -> Iterable:
        """Iterate outgoing lifted neighbors from a node instance.

        Args:
            node_inst: `(u, shift)`.

        Yields:
            Depending on flags:
            - `(v, shift + tvec)`
            - `(v, shift + tvec, key)`
            - `(v, shift + tvec, attrs)`
            - `(v, shift + tvec, key, attrs)`
        """
        u, shift = node_inst
        validate_tvec(shift, self._dim)

        def iter_lifted() -> Iterator[Tuple[NodeId, TVec, EdgeKey, Any]]:
            for item in self.neighbors(u, keys=True, data=True):
                v, tvec, k, attrs = item
                yield v, add_tvec(shift, tvec), k, attrs

        if not keys and not data:
            return ((v, s2) for (v, s2, _k, _a) in iter_lifted())
        if keys and not data:
            return ((v, s2, k) for (v, s2, k, _a) in iter_lifted())
        if (not keys) and data:
            return ((v, s2, a) for (v, s2, _k, a) in iter_lifted())
        return ((v, s2, k, a) for (v, s2, k, a) in iter_lifted())

    def in_neighbors_inst(
        self, node_inst: NodeInst, keys: bool = False, data: bool = False
    ) -> Iterable:
        """Iterate incoming lifted neighbors into a node instance.

        For an incoming edge ``v -> u`` with translation ``tvec``, the lifted
        neighbor instance for ``v`` is ``shift - tvec``.
        """
        u, shift = node_inst
        validate_tvec(shift, self._dim)

        def iter_lifted() -> Iterator[Tuple[NodeId, TVec, EdgeKey, Any]]:
            for item in self.in_neighbors(u, keys=True, data=True):
                v, tvec, k, attrs = item
                yield v, sub_tvec(shift, tvec), k, attrs

        if not keys and not data:
            return ((v, s2) for (v, s2, _k, _a) in iter_lifted())
        if keys and not data:
            return ((v, s2, k) for (v, s2, k, _a) in iter_lifted())
        if (not keys) and data:
            return ((v, s2, a) for (v, s2, _k, a) in iter_lifted())
        return ((v, s2, k, a) for (v, s2, k, a) in iter_lifted())

    def successors(self, u: NodeId) -> Iterable[NodeId]:
        """Return successor nodes (quotient) in deterministic order."""
        vs = {v for (v, _t) in self.neighbors(u, keys=False, data=False)}
        return stable_sorted(vs)

    def predecessors(self, u: NodeId) -> Iterable[NodeId]:
        """Return predecessor nodes (quotient) in deterministic order."""
        vs = {v for (v, _t) in self.in_neighbors(u, keys=False, data=False)}
        return stable_sorted(vs)

    # -----------------
    # Construction helpers
    # -----------------
    @classmethod
    def from_edges(
        cls,
        dim: int,
        nodes: Optional[Iterable[Any]] = None,
        edges: Optional[Iterable[Any]] = None,
    ) -> 'PeriodicDiGraph':
        """Construct a graph from nodes and edges.

        Args:
            dim: Lattice dimension.
            nodes: Optional iterable of node ids or `(node_id, attrs_dict)`
                pairs.
            edges: Optional iterable of edges, each one of:
                - `(u, v, tvec)`
                - `(u, v, tvec, attrs_dict)`
                - `(u, v, tvec, key, attrs_dict)`

        Returns:
            A graph instance of type `cls`.
        """
        G = cls(dim=dim)
        if nodes is not None:
            for item in nodes:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], dict)
                ):
                    u, ad = item
                    G.add_node(u, **ad)
                else:
                    G.add_node(item)

        if edges is not None:
            for e in edges:
                if len(e) == 3:
                    u, v, tvec = e
                    G.add_edge(u, v, tvec)
                elif len(e) == 4:
                    u, v, tvec, ad = e
                    if not isinstance(ad, dict):
                        raise ValueError(
                            '4-tuple edges must be (u, v, tvec, attrs_dict)'
                        )
                    G.add_edge(u, v, tvec, **ad)
                elif len(e) == 5:
                    u, v, tvec, key, ad = e
                    if not isinstance(ad, dict):
                        raise ValueError(
                            '5-tuple edges must be (u, v, tvec, key, '
                            'attrs_dict)'
                        )
                    G.add_edge(u, v, tvec, key=key, **ad)
                else:
                    raise ValueError('edge must have length 3, 4, or 5')
        return G

    # -----------------
    # Components
    # -----------------
    def components(self) -> List['PeriodicComponent']:
        """Return connected components as `PeriodicComponent` objects."""
        return _components(self)

    # -----------------
    # Finite lifts
    # -----------------
    def lift_patch(
        self,
        seed: NodeInst,
        *,
        radius: Optional[int] = None,
        box: Optional[Tuple[Tuple[int, int], ...]] = None,
        box_rel: Optional[Tuple[Tuple[int, int], ...]] = None,
        include_edges: bool = True,
        max_nodes: Optional[int] = None,
        node_order: Optional[Callable[[NodeInst], Any]] = None,
        edge_order: Optional[Callable[[Tuple[Any, ...]], Any]] = None,
    ) -> 'LiftPatch':
        """Extract a finite patch of the lifted graph.

        This is a thin wrapper over :func:`pbcgraph.alg.lift.lift_patch`.

        Notes:
            For directed containers this patch is directed by default (exported
            as `nx.DiGraph` / `nx.MultiDiGraph`). Use
            `patch.to_networkx(as_undirected=True, ...)` for undirected views.
        """
        from pbcgraph.alg.lift import lift_patch as _lift_patch

        return _lift_patch(
            self,
            seed,
            radius=radius,
            box=box,
            box_rel=box_rel,
            include_edges=include_edges,
            max_nodes=max_nodes,
            node_order=node_order,
            edge_order=edge_order,
        )


