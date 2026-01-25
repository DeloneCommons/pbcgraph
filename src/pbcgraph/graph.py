"""Periodic graph containers.

pbcgraph represents a periodic graph by a finite quotient graph, where each
directed quotient edge carries an integer translation vector in ``Z^d``.

Internally, quotient edges are stored in a NetworkX
:class:`networkx.MultiDiGraph`.
However, pbcgraph exposes *two* containers families:

- `PeriodicDiGraph` / `PeriodicGraph`: at most one edge per ``(u, v, tvec)``.
- `PeriodicMultiDiGraph` / `PeriodicMultiGraph`: allow multiple edges per
  ``(u, v, tvec)`` (distinguished by edge keys).

Exports:
    PeriodicDiGraph: Directed periodic graph on ``Z^d`` (unique per
        ``(u, v, tvec)``).
    PeriodicGraph: Undirected periodic graph implemented as a pair of directed
        realizations per undirected edge (unique per undirected
        ``{u, v, tvec}`` up to reversal).
    PeriodicMultiDiGraph: Directed periodic multigraph on ``Z^d``.
    PeriodicMultiGraph: Undirected periodic multigraph.

Attributes:
    _TVEC_ATTR: Internal edge-data key for translation vectors.
    _USER_ATTRS: Internal edge-data key for the live user-attributes mapping.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from types import MappingProxyType

import networkx as nx

if TYPE_CHECKING:
    from pbcgraph.component import PeriodicComponent

from pbcgraph.alg.components import components as _components

from pbcgraph.core.ordering import (
    stable_sorted,
    stable_tvec,
    stable_unique_sorted,
    try_sort_edges,
    try_sort_neighbor_edges,
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


_TVEC_ATTR = '_tvec'
_USER_ATTRS = '_attrs'


def _ro(mapping: Dict[str, Any]) -> MappingProxyType:
    """Return a read-only live view of a mapping."""
    return MappingProxyType(mapping)


def _validate_edge_key(key: EdgeKey) -> None:
    """Validate an edge key.

    Edge keys must be ints, but ``bool`` is rejected (even though it is a
    subclass of ``int``).
    """
    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError('edge key must be an int (bool is not allowed)')


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
            changes (edge attribute updates).

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
            - If the node already exists, this only updates attributes
              and increments `data_version` if any attributes are provided.
        """
        exists = self._g.has_node(u)
        if not exists:
            self._g.add_node(u)
            self.structural_version += 1
        if attrs:
            self._g.nodes[u].update(attrs)
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
        """Allocate a new edge key for an undirected edge between u and v."""
        kd_uv = self._g.get_edge_data(u, v) or {}
        kd_vu = self._g.get_edge_data(v, u) or {}
        used = set(kd_uv) | set(kd_vu)
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
                return k
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
        records: List[Tuple[Any, Any, Tuple[int, ...], int, Any]] = []
        for u, v, k, edata in self._g.edges(keys=True, data=True):
            records.append(
                (u, v, stable_tvec(edata[_TVEC_ATTR]), int(k), edata[_USER_ATTRS])
            )

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

        records: List[Tuple[Any, Tuple[int, ...], int, Any]] = []
        adj = self._g.adj[u]
        for v in adj:
            kd = adj[v]
            for k in kd:
                ed = kd[k]
                records.append(
                    (v, stable_tvec(ed[_TVEC_ATTR]), int(k), ed[_USER_ATTRS])
                )

        try_sort_neighbor_edges(records)

        for v, tv, k, attrs in records:
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

        records: List[Tuple[Any, Tuple[int, ...], int, Any]] = []
        pred_adj = self._g.pred[u]
        for v in pred_adj:
            kd = pred_adj[v]
            for k in kd:
                ed = kd[k]
                records.append(
                    (v, stable_tvec(ed[_TVEC_ATTR]), int(k), ed[_USER_ATTRS])
                )

        try_sort_neighbor_edges(records)

        for v, tv, k, attrs in records:
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


class PeriodicMultiDiGraph(PeriodicDiGraph):
    """Directed periodic multigraph on ``Z^d``.

    Unlike `PeriodicDiGraph`, this container allows multiple edges for the same
    directed triple ``(u, v, tvec)``. Such parallel edges are distinguished by
    their edge keys.
    """

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


class PeriodicGraph(PeriodicDiGraph):
    """Undirected periodic graph.

    Internally, an undirected periodic edge is represented by two directed
    realizations:

    - ``u -> v`` with translation ``tvec``
    - ``v -> u`` with translation ``-tvec``

    Both realizations share the same underlying user-attributes dict.
    The public API returns read-only live views of that mapping.

    In addition to the undirected-invariant pairing, this container enforces
    an invariant analogous to `PeriodicDiGraph`:

    *For any undirected triple ``{u, v, tvec}`` (up to reversal), at most one
    edge exists.*

    To allow multiple contacts for the same motif pair and translation, use
    `PeriodicMultiGraph`.

    Notes:
        `PeriodicGraph` is a subclass of `PeriodicDiGraph`, but restricts some
        operations (for example, directed connectivity modes in algorithms).
    """

    @property
    def is_undirected(self) -> bool:
        """Whether this container should be treated as undirected
        by algorithms."""
        return True

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        tvec: TVec,
        key: Optional[EdgeKey] = None,
        **attrs: Any,
    ) -> EdgeKey:
        validate_tvec(tvec, self._dim)

        existing = self._key_for_tvec(u, v, tvec)
        existing_rev = self._key_for_tvec(v, u, neg_tvec(tvec))
        if existing is not None or existing_rev is not None:
            raise ValueError(
                'undirected edge already exists for {u, v, tvec}: '
                f'({u!r}, {v!r}, {tuple(tvec)!r}); '
                f'key={existing if existing is not None else existing_rev!r}'
            )

        return self._add_undirected_impl(
            u, v, tvec, key=key, attrs=dict(attrs)
        )

    def _add_undirected_impl(
        self,
        u: NodeId,
        v: NodeId,
        tvec: TVec,
        *,
        key: Optional[EdgeKey],
        attrs: Dict[str, Any],
    ) -> EdgeKey:
        """Implementation for adding an undirected edge (no tvec checks)."""
        validate_tvec(tvec, self._dim)
        if not self._g.has_node(u):
            self.add_node(u)
        if not self._g.has_node(v):
            self.add_node(v)

        if key is None:
            key = self._alloc_key_undirected(u, v)
        else:
            _validate_edge_key(key)

        # Disallow overwriting existing keys in either direction.
        if self._g.has_edge(u, v, key=key) or self._g.has_edge(v, u, key=key):
            raise KeyError((u, v, key))

        tvec_norm = stable_tvec(tvec)
        user_attrs: Dict[str, Any] = dict(attrs)
        self._g.add_edge(
            u,
            v,
            key=key,
            **{_TVEC_ATTR: tvec_norm, _USER_ATTRS: user_attrs},
        )
        self._g.add_edge(
            v,
            u,
            key=key,
            **{
                _TVEC_ATTR: stable_tvec(neg_tvec(tvec)),
                _USER_ATTRS: user_attrs,
            },
        )
        self.structural_version += 1
        return key

    def has_edge(
        self, u: NodeId, v: NodeId, key: Optional[EdgeKey] = None
    ) -> bool:
        if key is None:
            return self._g.has_edge(u, v) and self._g.has_edge(v, u)
        return (
            self._g.has_edge(u, v, key=key) and
            self._g.has_edge(v, u, key=key)
        )

    def remove_edge(self, u: NodeId, v: NodeId, key: EdgeKey) -> None:
        if (
            not self._g.has_edge(u, v, key=key)
            or not self._g.has_edge(v, u, key=key)
        ):
            raise KeyError((u, v, key))
        self._g.remove_edge(u, v, key=key)
        self._g.remove_edge(v, u, key=key)
        self.structural_version += 1

    def check_invariants(self, *, strict: bool = False) -> Dict[str, Any]:
        """Check undirected pairing invariants.

        Returns a structured report and optionally raises on errors.

        Invariants checked:
            - For every (u, v, key) there is (v, u, key).
            - Translation vectors satisfy t(v,u,k) = -t(u,v,k).
            - The user-attributes dict is the *same object* for the paired
              directed realizations.

        Args:
            strict: If True, raise ValueError on the first violation.

        Returns:
            A dict with keys: `ok`, `errors`, `n_edges`.
        """
        errors: List[str] = []

        for u, v, k, ed in self._g.edges(keys=True, data=True):
            rev = self._g.get_edge_data(v, u, k)
            if rev is None:
                msg = f'missing reverse edge for ({u!r}, {v!r}, {k!r})'
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
                continue
            tv = stable_tvec(ed[_TVEC_ATTR])
            tv_rev = stable_tvec(rev[_TVEC_ATTR])
            if tv_rev != stable_tvec(neg_tvec(tv)):
                msg = (
                    'translation mismatch for paired edges: '
                    f'({u!r}->{v!r},k={k!r}) has {tv!r}, '
                    f'({v!r}->{u!r},k={k!r}) has {tv_rev!r}'
                )
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
            if ed[_USER_ATTRS] is not rev[_USER_ATTRS]:
                msg = (
                    'attribute mapping is not shared for paired edges: '
                    f'({u!r},{v!r},k={k!r})'
                )
                if strict:
                    raise ValueError(msg)
                errors.append(msg)

        return {
            'ok': len(errors) == 0,
            'errors': errors,
            'n_edges': int(self._g.number_of_edges()),
        }


class PeriodicMultiGraph(PeriodicGraph):
    """Undirected periodic multigraph.

    Unlike `PeriodicGraph`, this container allows multiple undirected edges for
    the same motif pair and translation (i.e. multiple edges for the same
    undirected ``{u, v, tvec}`` up to reversal). Parallel edges are
    distinguished by their edge keys.
    """

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
