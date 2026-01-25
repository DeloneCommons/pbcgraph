# Changelog

This project follows a lightweight "keep a log" style.


## 0.1.1 - Refactoring

- **Deterministic iteration**
    - All public iteration APIs (`nodes`, `edges`, `neighbors`, `successors`, `predecessors`) now yield results in a deterministic order.
    - Ordering is lexicographic when objects are mutually comparable; otherwise a stable fallback order is used.

- **Read-only attribute views**
    - `get_node_data()` and `get_edge_data()` now return **read-only live views** (mapping proxies) instead of mutable dicts.
    - Use `set_node_attrs()` / `set_edge_attrs()` to update attributes (these bump `data_version`).

- **Edge keys**
    - Auto-generated edge keys are now deterministic and local to a `(u, v)` pair (mirrors NetworkX `new_edge_key`).
    - Explicit keys must be Python integers (bool is rejected).

- **New APIs**
    - `edges(..., tvec=True)` can include the structural translation vector in iteration records.
    - `in_neighbors(...)` and `in_neighbors_inst(...)` provide deterministic access to incoming periodic edges.
    - `PeriodicGraph.check_invariants()` validates undirected pairing invariants.

- **Lattice/SNF**
    - Removed the SymPy dependency by implementing exact inversion of unimodular matrices.

- **Version semantics**
    - Pure data-only `data_version` semantics: `data_version` increments only on user-attribute updates that do not change structure (e.g., `set_node_attrs`, `set_edge_attrs`, or `add_node` on an existing node).
    - Creating new nodes/edges with attributes does not increment `data_version` (structural change only).

- **Docs clarifications**
    - Clarified that component extraction and weak-neighbor helpers rely on deterministic (stable-sorted) iteration, not insertion order.
    - Documented an edge-iteration gotcha for self-loop periodic edges: use `tvec=True` with `keys=True` to disambiguate paired realizations.
    - Corrected `SNFDecomposition.diag` documentation (returned length is `rank`).

- **Performance**
    - Reduced redundant generator collection for undirected components by deduplicating paired directed realizations (no semantic change).

## 0.1.0 — Initial release

- First public release of **pbcgraph**, a lightweight Python library for periodic graphs built on top of **NetworkX**.
- Provides periodic graph containers with **integer translation vectors** on directed edges to represent connectivity between periodic images.
- Supports **directed/undirected** and **simple/multi** variants (NetworkX `DiGraph`/`MultiDiGraph`-style API).
- Includes core algorithms for **connected components**, **quotient shortest paths**, and basic periodic graph traversal utilities.
- Implements **periodic component** analysis (computing translation subgroup invariants via **Smith normal form**-based lattice reduction).
- Ships with initial tests and basic documentation.
