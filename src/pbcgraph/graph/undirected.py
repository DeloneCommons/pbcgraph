"""Undirected periodic graph container."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from pbcgraph.core.ordering import stable_tvec
from pbcgraph.core.types import EdgeKey, NodeId, TVec, neg_tvec, validate_tvec
from pbcgraph.graph.directed import PeriodicDiGraph
from pbcgraph.graph.shared import (
    _TVEC_ATTR,
    _USER_ATTRS,
    UKey as _UKey,
    base_key as _base_key,
    check_reserved_edge_attrs as _check_reserved_edge_attrs,
    ro as _ro,
    validate_edge_key as _validate_edge_key,
)


class PeriodicGraph(PeriodicDiGraph):
    """Undirected periodic graph.

    Internally, an undirected periodic edge is represented by two directed
    realizations:

    - ``u -> v`` with translation ``tvec``
    - ``v -> u`` with translation ``-tvec``

    Both realizations share the same underlying user-attributes dict.
    The public API returns read-only live views of that mapping.

    **Important:** a crystallographically common pattern is a quotient
    self-loop with non-zero translation (``u == v`` and ``tvec != 0``),
    representing a bond to a periodic image of the same motif.

    NetworkX identifies multiedges by ``(u, v, key)``. For self-loops,
    the two directed realizations would collide if they shared the same key.

    To avoid this, `PeriodicGraph` stores directed realizations using a private
    internal key type `_UKey(base, dir)`, where `base` is the user-visible
    integer key and `dir` is `+1` / `-1`. The public API always exposes
    the base key.

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

    def edges(
        self, keys: bool = False, data: bool = False, tvec: bool = False
    ) -> Iterable:
        """Iterate directed realizations in deterministic order.

        This iterator yields *directed realizations* of undirected edges.

        Note:
            For self-loop periodic edges (``u == v`` and ``tvec != 0``), the
            two directed realizations share the same ``(u, v, key)`` triple and
            differ only by the translation vector. If `keys=True` but
            `tvec=False`, this may yield duplicate ``(u, u, key)`` records. Use
            `tvec=True` to disambiguate.

        See `PeriodicDiGraph.edges` for the record formats.
        """
        return super().edges(keys=keys, data=data, tvec=tvec)

    def _internal_keys_for_base(
        self, u: NodeId, v: NodeId, key: EdgeKey
    ) -> List[object]:
        """Return internal keys on (u -> v) whose public base key
        equals `key`."""
        kd = self._g.get_edge_data(u, v) or {}
        base = int(key)
        out: List[object] = []
        for ik in kd:
            if _base_key(ik) == base:
                out.append(ik)
        return out

    def _choose_internal_key(
        self, u: NodeId, v: NodeId, key: EdgeKey
    ) -> object:
        """Choose a deterministic internal key for accessing
        shared attrs/tvec."""
        keys = self._internal_keys_for_base(u, v, key)
        if not keys:
            raise KeyError((u, v, key))
        # Prefer the "forward" realization when present.
        for ik in keys:
            if isinstance(ik, _UKey) and ik.dir == 1:
                return ik
        # Otherwise choose a deterministic order.
        return sorted(
            keys,
            key=lambda x: (
                0 if isinstance(x, _UKey) else 1,
                getattr(x, 'dir', 0),
                repr(x),
            ),
        )[0]

    def _has_undirected_base(self, u: NodeId, v: NodeId, key: EdgeKey) -> bool:
        """Return True if an undirected edge with base key exists
        between u and v."""
        if u != v:
            return (
                len(self._internal_keys_for_base(u, v, key)) == 1 and
                len(self._internal_keys_for_base(v, u, key)) == 1
            )
        # Self-loop: both realizations live on (u -> u).
        return len(self._internal_keys_for_base(u, u, key)) == 2

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

        # Disallow overwriting an existing base key in either direction.
        if (
            self._internal_keys_for_base(u, v, key)
            or self._internal_keys_for_base(v, u, key)
        ):
            raise KeyError((u, v, key))

        tvec_norm = stable_tvec(tvec)
        user_attrs: Dict[str, Any] = dict(attrs)
        _check_reserved_edge_attrs(user_attrs)

        k_fwd = _UKey(int(key), 1)
        k_rev = _UKey(int(key), -1)

        self._g.add_edge(
            u,
            v,
            key=k_fwd,
            **{_TVEC_ATTR: tvec_norm, _USER_ATTRS: user_attrs},
        )
        self._g.add_edge(
            v,
            u,
            key=k_rev,
            **{
                _TVEC_ATTR: stable_tvec(neg_tvec(tvec)),
                _USER_ATTRS: user_attrs,
            },
        )
        self.structural_version += 1
        return int(key)

    def has_edge(
        self, u: NodeId, v: NodeId, key: Optional[EdgeKey] = None
    ) -> bool:
        if key is None:
            if u != v:
                return self._g.has_edge(u, v) and self._g.has_edge(v, u)
            kd = self._g.get_edge_data(u, u) or {}
            return len(kd) >= 2
        return self._has_undirected_base(u, v, key)

    def edge_tvec(self, u: NodeId, v: NodeId, key: EdgeKey) -> TVec:
        ik = self._choose_internal_key(u, v, key)
        data = self._g.get_edge_data(u, v, ik)
        if data is None:
            raise KeyError((u, v, key))
        return stable_tvec(data[_TVEC_ATTR])

    def get_edge_data(
        self, u: NodeId, v: NodeId, key: EdgeKey, default: Any = None
    ) -> Any:
        try:
            ik = self._choose_internal_key(u, v, key)
        except KeyError:
            return default
        data = self._g.get_edge_data(u, v, ik)
        if data is None:
            return default
        return _ro(data[_USER_ATTRS])

    def set_edge_attrs(
        self, u: NodeId, v: NodeId, key: EdgeKey, **attrs: Any
    ) -> None:
        if not self._has_undirected_base(u, v, key):
            raise KeyError((u, v, key))
        if not attrs:
            return
        ik = self._choose_internal_key(u, v, key)
        data = self._g.get_edge_data(u, v, ik)
        if data is None:
            raise KeyError((u, v, key))
        _check_reserved_edge_attrs(attrs)
        data[_USER_ATTRS].update(attrs)
        self.data_version += 1

    def remove_edge(self, u: NodeId, v: NodeId, key: EdgeKey) -> None:
        triples = set()
        for a, b in ((u, v), (v, u)):
            kd = self._g.get_edge_data(a, b) or {}
            for ik in kd:
                if _base_key(ik) == int(key):
                    triples.add((a, b, ik))
        if len(triples) != 2:
            raise KeyError((u, v, key))
        for a, b, ik in triples:
            self._g.remove_edge(a, b, key=ik)
        self.structural_version += 1

    def check_invariants(self, *, strict: bool = False) -> Dict[str, Any]:
        """Check undirected pairing invariants.

        Returns a structured report and optionally raises on errors.

        Invariants checked:
            - For every directed realization there is a paired reverse one.
            - Translation vectors satisfy t(v,u,rev) = -t(u,v,fwd).
            - The user-attributes dict is the *same object* for paired
              realizations.

        Args:
            strict: If True, raise ValueError on the first violation.

        Returns:
            A dict with keys: `ok`, `errors`, `n_edges`.
        """
        errors: List[str] = []

        for u, v, ik, ed in self._g.edges(keys=True, data=True):
            base = _base_key(ik)
            if isinstance(ik, _UKey):
                rev_key: object = _UKey(base, -ik.dir)
            else:
                rev_key = ik
            rev = self._g.get_edge_data(v, u, rev_key)
            if rev is None:
                msg = (
                    'missing reverse edge for '
                    f'({u!r}, {v!r}, base={base!r}, ik={ik!r})'
                )
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
                continue
            tv = stable_tvec(ed[_TVEC_ATTR])
            tv_rev = stable_tvec(rev[_TVEC_ATTR])
            if tv_rev != stable_tvec(neg_tvec(tv)):
                msg = (
                    'translation mismatch for paired edges: '
                    f'({u!r}->{v!r}, base={base!r}) has {tv!r}, '
                    f'({v!r}->{u!r}, base={base!r}) has {tv_rev!r}'
                )
                if strict:
                    raise ValueError(msg)
                errors.append(msg)
            if ed[_USER_ATTRS] is not rev[_USER_ATTRS]:
                msg = (
                    'attribute mapping is not shared for paired edges: '
                    f'({u!r},{v!r}, base={base!r})'
                )
                if strict:
                    raise ValueError(msg)
                errors.append(msg)

        return {
            'ok': len(errors) == 0,
            'errors': errors,
            'n_edges': int(self._g.number_of_edges()),
        }


