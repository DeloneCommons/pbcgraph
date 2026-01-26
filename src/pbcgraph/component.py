"""Periodic components.

A :class:`~pbcgraph.component.PeriodicComponent` represents a connected
quotient component and carries lattice invariants for the induced
translation subgroup ``L subset Z^d``.

Exports:
    PeriodicComponent: Connected quotient component with periodic-lift
        invariants (rank, translation generators, torsion invariants).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Callable,
    Any,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from pbcgraph.core.exceptions import StaleComponentError
from pbcgraph.core.types import (
    NodeId,
    NodeInst,
    TVec,
    add_tvec,
    neg_tvec,
    sub_tvec,
    zero_tvec,
)
from pbcgraph.core.protocols import PeriodicDiGraphLike
from pbcgraph.lattice.snf import SNFDecomposition, snf_decomposition
from pbcgraph.alg.lift import CanonicalLift


def _tvec_is_zero(t: TVec) -> bool:
    return all(x == 0 for x in t)


def _columns(mat: Tuple[Tuple[int, ...], ...]) -> List[Tuple[int, ...]]:
    if not mat:
        return []
    nrows = len(mat)
    ncols = len(mat[0])
    cols: List[Tuple[int, ...]] = []
    for j in range(ncols):
        cols.append(tuple(int(mat[i][j]) for i in range(nrows)))
    return cols


@dataclass(frozen=True)
class PeriodicComponent:
    """A connected quotient component with periodic-lift invariants.

    A component is computed on the quotient graph (nodes are templates
    living in the reference cell). In addition to the node set, it stores
    information needed to reason about the infinite periodic lift.

    Attributes:
        graph: Parent graph that produced this component. `PeriodicGraph` is
            accepted since it is a subclass of `PeriodicDiGraph`.
        nodes: Quotient node ids in this component.
        root: Deterministic root node used for spanning-tree potentials.
        created_structural_version: Structural version of `graph` at creation.

        rank: Rank of the translation subgroup ``L`` (the periodic dimension).
        translation_generators: A (possibly redundant) generator set for ``L``.
        torsion_invariants: Invariant factors ``d_i > 1`` describing the
            torsion part of ``Z^d / L``. An empty tuple means no torsion.

    Notes:
        This dataclass is frozen for safety. Internal caches are populated in
        `__post_init__` using `object.__setattr__`.
    """

    graph: PeriodicDiGraphLike
    nodes: FrozenSet[NodeId]
    root: NodeId
    created_structural_version: int

    # Public computed invariants.
    rank: int = 0
    translation_generators: Tuple[TVec, ...] = ()
    torsion_invariants: Tuple[int, ...] = ()

    # Private caches.
    _potentials: Dict[NodeId, TVec] = field(default_factory=dict, repr=False)
    _tree_parent: Dict[NodeId, Tuple[NodeId, TVec, int]] = field(
        default_factory=dict, repr=False
    )
    _snf: Optional[SNFDecomposition] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Compute potentials and lattice invariants eagerly for determinism.
        # The dataclass is frozen, so we must use `object.__setattr__` to
        # populate computed fields and caches during initialization.
        pot, parent = self._compute_potentials()
        gens = self._compute_generators(pot)
        dec = snf_decomposition(gens, self.graph.dim)

        object.__setattr__(self, '_potentials', pot)
        object.__setattr__(self, '_tree_parent', parent)
        object.__setattr__(self, '_snf', dec)
        object.__setattr__(self, 'rank', dec.rank)
        object.__setattr__(self, 'translation_generators', tuple(gens))
        object.__setattr__(
            self, 'torsion_invariants', tuple(di for di in dec.diag if di > 1)
        )

    # -----------------
    # Staleness
    # -----------------
    def is_stale(self) -> bool:
        """Return True if the parent graph has changed structurally since
        creation."""
        return self.graph.structural_version != self.created_structural_version

    def _require_fresh(self) -> None:
        if self.is_stale():
            raise StaleComponentError(
                'PeriodicComponent is stale: graph structure has changed'
            )

    @property
    def snf(self) -> SNFDecomposition:
        """Smith normal form decomposition for the translation subgroup.

        Raises:
            StaleComponentError: If the parent graph has changed structurally.
        """
        self._require_fresh()
        dec = self._snf
        assert dec is not None
        return dec

    def tree_parent_map(self) -> Mapping[NodeId, Tuple[NodeId, TVec, int]]:
        """Read-only spanning-tree parent mapping.

        The mapping records, for each non-root node `child`, a tuple
        `(parent, tvec, key)` describing the tree edge used to assign the
        node potential.

        Raises:
            StaleComponentError: If the parent graph has changed structurally.
        """
        self._require_fresh()
        return MappingProxyType(self._tree_parent)

    # -----------------
    # Potential
    # -----------------
    def potential(self, u: NodeId) -> TVec:
        """Return the spanning-tree potential `pot(u)`.

        Args:
            u: Quotient node id.

        Returns:
            Integer translation vector in Z^d.

        Raises:
            StaleComponentError: If component is stale.
            KeyError: If `u` is not in the component.
        """
        self._require_fresh()
        return self._potentials[u]

    # -----------------
    # Instance key and connectivity
    # -----------------
    def inst_key(self, node_inst: NodeInst) -> Hashable:
        """Return a canonical coset key for the lifted instance in Z^d / L.

        Representation (v0.1):
            Returns a tuple of ints with layout:
                (torsion_residues..., free_coords...)

            Torsion residues are reduced modulo the SNF diagonal invariants
            (>1) and free coordinates are unbounded integers.

        Args:
            node_inst: `(u, shift)` where `shift in Z^d`.

        Raises:
            StaleComponentError: If component is stale.
            KeyError: If `u` is not in the component.
        """
        self._require_fresh()
        u, shift = node_inst
        if u not in self.nodes:
            raise KeyError(u)
        # Absolute coordinate A(u, s) = s - pot(u)
        A = sub_tvec(shift, self._potentials[u])
        y = self._snf.apply_U(A)

        residues: List[int] = []
        for i in range(self._snf.rank):
            di = self._snf.diag[i]
            if di > 1:
                residues.append(int(y[i] % di))
        free_coords = [int(x) for x in y[self._snf.rank:]]
        return tuple(residues + free_coords)

    def same_fragment(self, inst_a: NodeInst, inst_b: NodeInst) -> bool:
        """Decide connectivity of two instances in the infinite lift.

        The criterion is:
            A(a) - A(b) in L
        where A(u, s) = s - pot(u) and L is the translation subgroup induced
        by cycles.

        Args:
            inst_a: `(u, shift)` instance.
            inst_b: `(v, shift)` instance.

        Returns:
            True if the two instances are in the same connected component of
            the infinite lift (with connectivity ignoring edge directions,
            consistent with v0.1 components).

        Raises:
            StaleComponentError: If component is stale.
            KeyError: If either quotient node is not in this component.
        """
        self._require_fresh()
        ua, sa = inst_a
        ub, sb = inst_b
        if ua not in self.nodes:
            raise KeyError(ua)
        if ub not in self.nodes:
            raise KeyError(ub)

        Aa = sub_tvec(sa, self._potentials[ua])
        Ab = sub_tvec(sb, self._potentials[ub])
        delta = sub_tvec(Aa, Ab)

        y = self._snf.apply_U(delta)
        r = self._snf.rank
        for i in range(r):
            di = self._snf.diag[i]
            if di != 0 and (y[i] % di) != 0:
                return False
        for i in range(r, self.graph.dim):
            if y[i] != 0:
                return False
        return True

    def transversal_basis(self) -> Dict[str, List[TVec]]:
        """Return a deterministic transversal basis description.

        Returns:
            A dict with keys:
                - 'free': list of vectors spanning the free part Z^(d-r)
                - 'torsion_dirs': list of vectors corresponding to torsion
                  coordinates
                - 'torsion_moduli': list of moduli (same length as
                  torsion_dirs)

        Notes:
            Direction vectors are given in the original coordinate basis
            of Z^d.
        """
        self._require_fresh()
        dim = self.graph.dim
        r = self._snf.rank

        cols = _columns(self._snf.U_inv)
        free: List[TVec] = []
        torsion_dirs: List[TVec] = []
        torsion_moduli: List[int] = []

        for i in range(r, dim):
            free.append(tuple(int(x) for x in cols[i]))

        for i in range(r):
            di = self._snf.diag[i]
            if di > 1:
                torsion_dirs.append(tuple(int(x) for x in cols[i]))
                torsion_moduli.append(int(di))

        return {
            'free': free,
            'torsion_dirs': torsion_dirs,
            'torsion_moduli': torsion_moduli,
        }

    # -----------------
    # Canonical lifts
    # -----------------
    def canonical_lift(
        self,
        *,
        strand_key: Hashable | None = None,
        seed: NodeInst | None = None,
        anchor_shift: TVec | None = None,
        placement: str = 'tree',
        score: str = 'l1',
        return_tree: bool = False,
        node_order: Callable[[NodeId], Any] | None = None,
        edge_order: Callable[[tuple], Any] | None = None,
    ) -> 'CanonicalLift':
        """Return a deterministic finite representation of a single strand.

        This is a thin wrapper over :func:`pbcgraph.alg.lift.canonical_lift`.

        Args:
            strand_key: Optional explicit strand (coset) key.
            seed: Optional seed instance used to determine `strand_key` and/or
                default `anchor_shift`.
            anchor_shift: Target anchor cell shift.
            placement: Placement mode. v0.1.2 step4 implements `'tree'`,
                `'best_anchor'`, and `'greedy_cut'`.
            score: Score metric, `'l1'` or `'l2'`.
            return_tree: If True, include spanning-tree edge records.
            node_order: Optional ordering key for quotient node ids.
            edge_order: Optional ordering key for periodic edges (reserved for
                later placement modes).

        Returns:
            A :class:`~pbcgraph.alg.lift.CanonicalLift`.
        """
        from pbcgraph.alg.lift import canonical_lift as _canonical_lift

        return _canonical_lift(
            self,
            strand_key=strand_key,
            seed=seed,
            anchor_shift=anchor_shift,
            placement=placement,
            score=score,
            return_tree=return_tree,
            node_order=node_order,
            edge_order=edge_order,
        )

    # -----------------
    # Internal computations
    # -----------------
    def _compute_potentials(
        self,
    ) -> Tuple[Dict[NodeId, TVec], Dict[NodeId, Tuple[NodeId, TVec, int]]]:
        dim = self.graph.dim
        pot: Dict[NodeId, TVec] = {self.root: zero_tvec(dim)}
        parent: Dict[NodeId, Tuple[NodeId, TVec, int]] = {}
        q = deque([self.root])

        while q:
            u = q.popleft()
            pu = pot[u]

            # Outgoing edges first (deterministic).
            for v, tvec, k in self.graph.neighbors(u, keys=True, data=False):
                if v not in self.nodes:
                    continue
                if v in pot:
                    continue
                pot[v] = add_tvec(pu, tvec)
                parent[v] = (u, tvec, int(k))
                q.append(v)

            # Incoming edges next (weak traversal).
            for v, t_in, k in self.graph.in_neighbors(
                u, keys=True, data=False
            ):
                if v not in self.nodes:
                    continue
                if v in pot:
                    continue
                pot[v] = sub_tvec(pu, t_in)
                parent[v] = (u, neg_tvec(t_in), int(k))
                q.append(v)

        if len(pot) != len(self.nodes):
            # This should never happen if component extraction is correct.
            missing = [u for u in self.nodes if u not in pot]
            raise RuntimeError(
                'component potential assignment incomplete, '
                f'missing: {missing}'
            )
        return pot, parent

    def _compute_generators(self, pot: Dict[NodeId, TVec]) -> List[TVec]:
        gens: List[TVec] = []

        if not self.graph.is_undirected:
            for u, v, t, _k in self.graph.edges(
                keys=True, data=False, tvec=True
            ):
                if u not in self.nodes or v not in self.nodes:
                    continue
                g = sub_tvec(add_tvec(pot[u], t), pot[v])
                if _tvec_is_zero(g):
                    continue
                gens.append(g)
            return gens

        for u, v, t, k in self.graph.undirected_edges_unique(
            keys=True, data=False, tvec=True
        ):
            if u not in self.nodes or v not in self.nodes:
                continue
            g = sub_tvec(add_tvec(pot[u], t), pot[v])
            if _tvec_is_zero(g):
                continue
            gens.append(g)
        return gens
