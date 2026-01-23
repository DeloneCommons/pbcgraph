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

import networkx as nx

if TYPE_CHECKING:
    from pbcgraph.component import PeriodicComponent

from pbcgraph.alg.components import components as _components

from pbcgraph.core.types import (
    EdgeKey,
    NodeId,
    NodeInst,
    TVec,
    add_tvec,
    neg_tvec,
    validate_tvec,
)


_TVEC_ATTR = '_tvec'
_USER_ATTRS = '_attrs'


def _unique_in_order(items: Iterable[NodeId]) -> List[NodeId]:
    """Deduplicate an iterable while preserving the order
    of first occurrence."""
    seen: set[NodeId] = set()
    out: List[NodeId] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


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
        self._next_key: int = 0
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
        """Iterate quotient nodes.

        Args:
            data: If True, yield `(u, attrs)` where `attrs` is a live mapping.

        Returns:
            Iterable of node ids or `(node, attrs)` pairs.
        """
        return self._g.nodes(data=data)

    def get_node_data(self, u: NodeId) -> Dict[str, Any]:
        """Return the live node attribute mapping.

        Args:
            u: Node id.

        Raises:
            KeyError: If node is missing.
        """
        return self._g.nodes[u]

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
    def _fresh_key(self) -> int:
        key = self._next_key
        self._next_key += 1
        return key

    def _maybe_advance_key_counter(self, key: EdgeKey) -> None:
        if isinstance(key, int) and key >= self._next_key:
            self._next_key = key + 1

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
        want = tuple(tvec)
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
            key = self._fresh_key()
        else:
            self._maybe_advance_key_counter(key)

        # Disallow overwriting an existing directed edge id.
        if self._g.has_edge(u, v, key=key):
            raise KeyError((u, v, key))

        user_attrs: Dict[str, Any] = dict(attrs)
        self._g.add_edge(
            u, v, key=key, **{_TVEC_ATTR: tuple(tvec), _USER_ATTRS: user_attrs}
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
        return tuple(data[_TVEC_ATTR])

    def get_edge_data(
        self, u: NodeId, v: NodeId, key: EdgeKey, default: Any = None
    ) -> Any:
        """Return the live user attribute mapping for an edge.

        Args:
            u: Source node id.
            v: Target node id.
            key: Edge key.
            default: Value to return if edge is missing.

        Returns:
            The live user attribute dict, or `default` if missing.
        """
        data = self._g.get_edge_data(u, v, key)
        if data is None:
            return default
        return data[_USER_ATTRS]

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

    def edges(self, keys: bool = False, data: bool = False) -> Iterable:
        """Iterate directed edges.

        Args:
            keys: If True, include the multiedge key.
            data: If True, include the live user attribute mapping.

        Returns:
            An iterable of `(u, v)` / `(u, v, key)` / `(u, v, attrs)` /
            `(u, v, key, attrs)`.
        """
        if data:
            for u, v, k, edata in self._g.edges(keys=True, data=True):
                attrs = edata[_USER_ATTRS]
                if keys:
                    yield u, v, k, attrs
                else:
                    yield u, v, attrs
        else:
            for u, v, k in self._g.edges(keys=True, data=False):
                if keys:
                    yield u, v, k
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

        def iter_edges() -> Iterator[
            Tuple[NodeId, EdgeKey, TVec, Dict[str, Any]]
        ]:
            adj = self._g.adj[u]
            for v in adj:
                kd = adj[v]
                for k in kd:
                    ed = kd[k]
                    yield v, k, tuple(ed[_TVEC_ATTR]), ed[_USER_ATTRS]

        if not keys and not data:
            return ((v, t) for (v, _k, t, _a) in iter_edges())
        if keys and not data:
            return ((v, t, k) for (v, k, t, _a) in iter_edges())
        if (not keys) and data:
            return ((v, t, a) for (v, _k, t, a) in iter_edges())
        return ((v, t, k, a) for (v, k, t, a) in iter_edges())

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

        def iter_lifted() -> Iterator[
            Tuple[NodeId, TVec, EdgeKey, Dict[str, Any]]
        ]:
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

    def successors(self, u: NodeId) -> Iterable[NodeId]:
        """Return successor nodes (quotient) in insertion order
        (duplicates removed)."""
        return _unique_in_order(
            v for (v, _t) in self.neighbors(u, keys=False, data=False)
        )

    def predecessors(self, u: NodeId) -> Iterable[NodeId]:
        """Return predecessor nodes (quotient) in insertion order
        (duplicates removed)."""
        if not self._g.has_node(u):
            raise KeyError(u)
        preds: List[NodeId] = []
        seen: set[NodeId] = set()
        pred_adj = self._g.pred[u]
        for v in pred_adj:
            if v in seen:
                continue
            seen.add(v)
            preds.append(v)
        return preds

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

    Both realizations share the same live user attribute mapping.

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
            key = self._fresh_key()
        else:
            self._maybe_advance_key_counter(key)

        # Disallow overwriting existing keys in either direction.
        if self._g.has_edge(u, v, key=key) or self._g.has_edge(v, u, key=key):
            raise KeyError((u, v, key))

        user_attrs: Dict[str, Any] = dict(attrs)
        self._g.add_edge(
            u,
            v,
            key=key,
            **{_TVEC_ATTR: tuple(tvec), _USER_ATTRS: user_attrs},
        )
        self._g.add_edge(
            v,
            u,
            key=key,
            **{_TVEC_ATTR: tuple(neg_tvec(tvec)), _USER_ATTRS: user_attrs},
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
