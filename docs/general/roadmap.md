# Roadmap

This page is a strategic direction for pbcgraph, not a promise or a fixed schedule.
Priorities can change based on real downstream use-cases.

- **v0.1.x**: stabilization + documentation + small ergonomics; plus early interop adapters.

Other directions (versioning intentionally omitted):

- **Interop and serialization**:
    - round-trip export/import of quotient graphs (including translation vectors and attributes)
    - adapters to common crystallography / materials toolchains (e.g., pymatgen-style graphs)

- **Topological analysis**:
    - translation-aware ring statistics and cycle classification
    - coordination sequences / growth series and simple net descriptors
    - better interpenetration summaries building on torsion signatures

- **Canonical forms and isomorphism**:
    - periodic-graph isomorphism tests (including translations)
    - canonical labeling / hashing for dataset building and deduplication

- **Optional geometry-aware helpers**:
    - embeddings (fractional coordinates) as attributes and consistency checks with `tvec`
    - geometry-derived weights (lengths/energies) without making geometry a core dependency

If you rely on any of these, treat them as collaboration targets: concrete use-cases and minimal reproducible
examples will shape what lands in pbcgraph versus what belongs in downstream packages.