# Choosing a container

This page is a practical guide to selecting the right pbcgraph container for a problem.

The main knobs are:

- **Undirected vs directed**: do edges have intrinsic orientation?
- **Unique vs multi-edge**: do you need multiple distinct edges for the same `(u, v, tvec)`?

Quick decision rule:

- If your relations are naturally symmetric, start with an undirected container (`PeriodicGraph`).
- If you need multiple interactions per template pair (e.g., multiple contacts), use a multi-edge container.
- Only use directed containers when the direction is part of the semantics (not just a label).

!!! note
    In v0.1, `PeriodicComponent` invariants are computed on **weakly connected** quotient components
    even for directed containers (i.e., directions are ignored when building components). Instance
    connectivity via `same_fragment(...)` follows the same convention.

## `PeriodicGraph`

Use `PeriodicGraph` for undirected periodic graphs where there is **at most one edge** per `(u, v, tvec)`.

Typical examples:

- **Atomic graphs** (covalent / ionic / coordination bonds): a single bond relation between two sites.
- **Coarse-grained molecular graphs** where you treat “molecule A contacts molecule B in this neighbor cell”
  as a single edge, and store details (multiple atom-atom contacts, energy terms, etc.) in edge attributes.
- Periodic nets where parallel edges are not meaningful.

![PeriodicGraph schematic](../assets/graph_types/periodic_graph.svg)

Notes:

- `PeriodicGraph` is implemented as two directed realizations (`u -> v` with `tvec` and `v -> u` with `-tvec`)
  that share the same underlying user-attributes dict. The public API returns
  read-only live views of that mapping (update via `set_edge_attrs()`).
- Edge identity includes the translation vector: two edges between the same pair of nodes are allowed
  if their `tvec` differ.
- Self-loop periodic edges are supported: a quotient edge with `u == v` and `tvec != 0` represents a bond to a periodic image. Internally this uses private keys derived from the base key, but the public API still exposes only integer base keys.
- When iterating edges with `keys=True`, pass `tvec=True` for self-loop periodic edges to distinguish the paired realizations; otherwise `edges(keys=True)` can yield duplicate `(u, u, key)` records.

## `PeriodicMultiGraph`

Use `PeriodicMultiGraph` when the undirected relation is still natural, but you need **multiple distinct
edges** for the same `(u, v, tvec)`.

Typical examples:

- **Molecular contact graphs** where a molecule pair can have multiple physically distinct interactions
  (e.g., separate hydrogen bonds, pi-stacking, multiple close contacts).
- “Annotated” graphs where you intentionally keep separate edges for different interaction models
  (e.g., geometric contact vs energy-filtered contact) between the same templates.

![PeriodicMultiGraph schematic](../assets/graph_types/multiedge.svg)

Practical note: if you do not need to distinguish parallel edges algorithmically, it is often simpler
to start with `PeriodicGraph` and store a *list* of contact records in a single edge attribute.

## `PeriodicDiGraph`

Use `PeriodicDiGraph` when edges are intrinsically oriented and there should be **at most one edge** per
`(u, v, tvec)`.

Examples where direction is meaningful:

- **Directed transport / flow** models on a periodic lattice (biased random walk, directed percolation).
- **Oriented backbones** (e.g., polymer directionality) when algorithms want a chosen direction.
- Any periodic state-transition graph (Markov-like) where the relation is not symmetric.

![PeriodicDiGraph schematic](../assets/graph_types/directed.svg)

If you only want to *label* an interaction (e.g., donor/acceptor role), but the relation should still be
treated as symmetric for connectivity, an undirected container plus attributes is usually a better fit.

!!! note
    `lift_patch(...)` follows the container direction: patches extracted from
    `PeriodicDiGraph` / `PeriodicMultiDiGraph` are directed by default (while
    traversal still uses weak connectivity).

## `PeriodicMultiDiGraph`

Use `PeriodicMultiDiGraph` when edges are directed **and** multiple distinct edges may exist for the same
`(u, v, tvec)`.

Typical examples:

- **Molecular hydrogen-bond graphs** with explicit `donor -> acceptor` direction, where multiple distinct
  H-bonds can exist between the same pair of molecules (and the same neighbor cell relation).
- Multi-channel directed relations in coarse-grained models.

If you are unsure, start with the simplest container that can represent your semantics, then upgrade to a
multi-edge or directed container only when you have a concrete need.
