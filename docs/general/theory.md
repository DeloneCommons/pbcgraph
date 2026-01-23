# Theory

This page summarizes the mathematics behind `PeriodicComponent`.
The goal is practical: **exact instance-aware connectivity** in the infinite periodic lift, without
enumerating infinitely many nodes.

## Quotient connectivity is not enough

`pbcgraph` stores a finite **quotient graph** on templates (sites, molecules, net vertices), but the intended
semantics is the **infinite lift**: nodes are `(u, s)` with `s in Z^d`, and an edge `(u -> v, tvec)` maps
`(u, s) -> (v, s + tvec)`.

Two quotient nodes can be connected in the quotient but still split into multiple disconnected copies in the
lift. In crystallography this shows up as **interpenetration**: multiple congruent nets or polymer strands that
project to the same quotient component.

`PeriodicComponent` computes lattice invariants that let pbcgraph decide whether two lifted instances belong to
the same connected fragment.

## Spanning-tree potentials

Fix a quotient component and choose a deterministic root node `r`.
Build a spanning tree of the quotient component and assign each quotient node a **potential**
`pot(u) in Z^d` such that along each spanning-tree edge, potentials are consistent with translation labels.

Intuition: `pot(u)` says where the template `u` sits (in cell coordinates) relative to the root in a particular
tree traversal.

## Cycle translations and the subgroup L

For any directed quotient edge `u -> v` with translation vector `t`, define the induced **cycle translation**:

\[
g = \mathrm{pot}(u) + t - \mathrm{pot}(v)
\]

Every non-tree edge closes a quotient cycle; `g` is the net translation you accumulate by going around that cycle
in the lift. Collecting these `g` vectors generates a subgroup:

\[
L = \langle g_1, g_2, \dots \rangle \subset \mathbb{Z}^d
\]

`L` is the set of translations that are reachable by moving around cycles in the component. Its rank is the
**periodic dimension** of the component:

- rank 0: isolated motif replicated in each cell (no periodic connectivity)
- rank 1: chain / 1D polymer direction
- rank 2: layer
- rank 3: 3D net (in a 3D lattice)

## Smith Normal Form (SNF) and torsion

The quotient group `Z^d / L` describes how many distinct "lifted copies" exist *inside a single quotient
component*. SNF gives a canonical decomposition:

\[
\mathbb{Z}^d / L \cong \mathbb{Z}^{d-r} \oplus \mathbb{Z}/d_1\mathbb{Z} \oplus \cdots \oplus \mathbb{Z}/d_k\mathbb{Z}
\]

where `r = rank(L)` and the integers `d_i > 1` are the **torsion invariants**.

Interpretation:

- The free part `Z^{d-r}` corresponds to directions transverse to the component periodicity.
- Each torsion factor `Z/d_iZ` means there are `d_i` distinct cosets that cannot be connected by translations in
  `L`. In the lift, this manifests as `d_i` disconnected but congruent fragments (a common notion of
  interpenetration).

### Minimal example: two interpenetrating strands

Work in `d = 1`. Suppose cycle translations only generate `L = 2Z`.
Then `Z / 2Z` has torsion `Z/2Z`, i.e. two cosets: even and odd indices.

In the lift, `(A, 0)` is connected to `(A, 2)`, `(A, 4)`, ... but **not** to `(A, 1)`.
Both strands project to the same quotient component, but they are disconnected in the lift.

This is exactly what `PeriodicComponent.torsion_invariants == (2,)` reports.

## Canonical coset keys and `same_fragment`

Define the **absolute coordinate** of a lifted instance `(u, s)` as:

\[
A(u, s) = s - \mathrm{pot}(u)
\]

Two lifted instances are in the same lift-connected fragment iff:

\[
A(a) - A(b) \in L
\]

pbcgraph implements this criterion using the SNF decomposition. Practically:

- `inst_key((u, s))` returns a deterministic tuple encoding the coset of `(u, s)` in `Z^d / L`.
  The tuple layout is `(torsion_residues..., free_coords...)`.
- `same_fragment(a, b)` is the semantic predicate you should rely on.

If you are only interested in "are these two atoms/molecules in the same infinite fragment?", you can ignore the
details and use `same_fragment(...)` directly.