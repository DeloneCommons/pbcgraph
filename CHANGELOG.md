# Changelog

This project follows a lightweight "keep a log" style.


## 0.1.3 - Refactoring and robustness

- **Added**:
    - `undirected_edges_unique(...)` for undirected containers and use it internally for component generator extraction.
    - Read-only accessors `PeriodicComponent.snf` and `PeriodicComponent.tree_parent_map()` and updated `canonical_lift(...)` to avoid touching private component caches.

- **Changed**:
    - `LiftPatch.to_networkx(as_undirected=True, ...)` now stores direction and orig-edge snapshots under a single reserved edge attribute `__pbcgraph__` to avoid collisions with user attributes.
    - Minor: code that previously read `_pbc_*` or `orig_edges` from NetworkX exports must now read `__pbcgraph__`.

- **Performance**:
    - `lift_patch(...)` avoids redundant incoming-edge traversal for undirected containers (no semantic change).
    - Refactored `edges()`, `neighbors()`, and `in_neighbors()` to use streaming deterministic iteration (avoid building a full edge list just to sort it).

- **Refactors**:
    - Refactored `LiftPatch.to_networkx(...)` into small helpers (no behavior change).
    - Split `pbcgraph.alg.lift` into `_lift_patch` and `_canonical_lift` implementation modules, keeping the public API unchanged.
    - General refactors: introduce an internal constant `PBC_META_KEY` for NetworkX export metadata, simplify internal key filtering in undirected containers, and apply small style cleanups.


## 0.1.2 - Finite lifts and canonical lifts

- **Finite lift patches**
    - Added `lift_patch(...)`: extract a finite patch of the infinite lift around a seed instance, using either a BFS radius and/or absolute/relative cell-index bounding boxes.
    - Patch edges store **snapshot** attribute dicts.
    - For undirected containers, paired directed realizations are deduplicated deterministically.

- **Patch export and directed semantics**
    - For directed periodic containers, `lift_patch(...)` now produces a **directed** patch by default (exported as `nx.DiGraph` / `nx.MultiDiGraph`).
    - `LiftPatch.to_networkx(as_undirected=True, undirected_mode=...)` provides undirected views of directed patches:
        - `undirected_mode='multigraph'`: one undirected multiedge per directed edge (direction preserved in `_pbc_tail`/`_pbc_head`).
        - `undirected_mode='orig_edges'`: collapsed simple graph with `orig_edges=[...]` snapshots for each adjacency.

- **Canonical lifts (strand representatives)**
    - Added `canonical_lift(...)` to select one instance per quotient node for a chosen strand (coset in `Z^d/L`).
    - Implemented placements: `tree`, `best_anchor`, and `greedy_cut`.
    - Stored deterministic spanning-tree parent edges on `PeriodicComponent` to optionally return `tree_edges`.

- **Errors**
    - Added `CanonicalLiftError` and `LiftPatchError` for well-scoped failure modes.


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
