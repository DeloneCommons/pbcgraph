# pbcgraph

**pbcgraph** is a compact Python library for **translation-labeled periodic graphs** on the integer lattice \(\mathbb{Z}^d\).

It is meant for situations where *topology matters more than geometry*:

- molecular-crystal connectivity graphs (contacts, LoS graphs, Voronoi-derived graphs),
- periodic nets and frameworks (MOFs/COFs/zeolites),
- abstract periodic graphs in mathematics.

The key idea is simple and useful: you store a **finite quotient graph** (internally a NetworkX `MultiDiGraph`), but every edge carries an integer translation vector. This lets you reason about the **infinite lift** exactly, without enumerating infinitely many nodes.

## What you get in v0.1

- `PeriodicGraph` (undirected) and `PeriodicDiGraph` (directed), enforcing **one edge per (u, v, tvec)**.
- `PeriodicMultiGraph` and `PeriodicMultiDiGraph` when you really need **parallel edges for the same (u, v, tvec)**.
- A `PeriodicComponent` view that exposes lattice invariants of the component translation subgroup \(L\subset\mathbb{Z}^d\):
    - `rank` (periodic dimension of the component),
    - `torsion_invariants` (torsion / interpenetration signature),
    - deterministic `inst_key(...)` keys for lifted instances within a component.
- `same_fragment(...)`: exact “are these two lifted instances in the same connected fragment?” checks.
- `shortest_path_quotient(...)`: fast BFS in the quotient with `connectivity='directed'|'weak'`.
- `lift_patch(...)`: extract a finite patch of the infinite lift around a seed instance.
- `canonical_lift(...)`: pick a canonical set of lifted instances (one per quotient node) for a chosen strand.

## Design philosophy

This project tries to sit between two ecosystems:

- **NetworkX-style ergonomics** for graph manipulation,
- **crystallographic correctness** for periodicity (translation vectors, torsion, components).

The API is intentionally small. The roadmap page lists bigger ideas (ToposPro-like topology, richer invariants, interop).

## Where to start

- If you want a runnable overview: start with **Examples → Quickstart**.
- If you want to understand SNF/torsion and interpenetration: see **Examples → SNF and interpenetration**.
- If you are unsure which container you need: read **Choosing a container**.
- For exact signatures: browse the **API Reference** (docstrings-based).
