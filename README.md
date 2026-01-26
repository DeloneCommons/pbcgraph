# pbcgraph

[![CI](https://github.com/IvanChernyshov/pbcgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/IvanChernyshov/pbcgraph/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://IvanChernyshov.github.io/pbcgraph/)
[![PyPI](https://img.shields.io/pypi/v/pbcgraph.svg)](https://pypi.org/project/pbcgraph/)
[![Python versions](https://img.shields.io/pypi/pyversions/pbcgraph.svg)](https://pypi.org/project/pbcgraph/)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

`pbcgraph` is a compact Python library for **translation-labeled periodic graphs** on the integer lattice `Z^d`.

You store a *finite quotient graph* (internally a NetworkX `MultiDiGraph`), but every directed edge carries an
integer translation vector. This gives the quotient an exact **infinite-lift semantics** and lets you do
instance-aware connectivity tests without enumerating the infinite graph.

What you get in v0.1:

- `PeriodicGraph` / `PeriodicDiGraph`: unique edge per `(u, v, tvec)`.
- `PeriodicMultiGraph` / `PeriodicMultiDiGraph`: parallel edges allowed for the same `(u, v, tvec)`.
- `PeriodicComponent`: lattice invariants (rank, SNF torsion) and exact instance connectivity via `same_fragment(...)`.
- `lift_patch(...)`: extract a finite (non-periodic) patch of the infinite lift around a seed instance.
- `canonical_lift(...)`: select one lifted instance per quotient node for a chosen strand (coset in `Z^d/L`).

## Status

`pbcgraph` is **alpha** (v0.1). The core containers and component invariants are implemented and covered by tests.
The API may still evolve, but the library is already useful for research code and prototyping.

## Install

Requires Python 3.10+. Latest stable version is usually published on PyPI:

```bash
python -m pip install pbcgraph
```

To install the latest version (or for the latest `dev` branch), install from GitHub:

```bash
python -m pip install git+https://github.com/IvanChernyshov/pbcgraph.git
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

```python
from pbcgraph import PeriodicGraph

# A quotient graph in Z^2.
G = PeriodicGraph(dim=2)

# Undirected edges are stored internally as two directed realizations
# with tvec and -tvec.

# Self-loop periodic edges are supported (quotient bond to a periodic image):
G1 = PeriodicGraph(dim=1)
G1.add_edge('A', 'A', tvec=(1,))

G.add_edge('A', 'B', tvec=(0, 0))
G.add_edge('B', 'C', tvec=(0, 0))
G.add_edge('C', 'A', tvec=(1, 0))  # closes a periodic cycle (rank-1 along x)

# Lifted nodes are (node_id, cell_shift).
neighbors = list(G.neighbors_inst(('A', (0, 0))))

comp = G.components()[0]
assert comp.same_fragment(('A', (0, 0)), ('A', (1, 0)))
assert not comp.same_fragment(('A', (0, 0)), ('A', (0, 1)))

# Extract a finite patch of the infinite lift around a seed instance.
patch = G.lift_patch(('A', (0, 0)), radius=2)
nx_patch = patch.to_networkx()  # nx.Graph / nx.MultiGraph for undirected sources

# For directed sources, patches are directed by default:
#   nx_patch = patch.to_networkx()  # nx.DiGraph / nx.MultiDiGraph
# and you can obtain undirected views via:
#   nx_u = patch.to_networkx(as_undirected=True, undirected_mode='multigraph')
#   nx_c = patch.to_networkx(as_undirected=True, undirected_mode='orig_edges')

# Canonical lift: pick one instance per quotient node for a strand.
lift = comp.canonical_lift(placement='tree')
assert len(lift.instances) == len(comp.nodes)
```

## Documentation

- Online docs (GitHub Pages): https://IvanChernyshov.github.io/pbcgraph/

Local docs build:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Notebooks in `docs/examples/` are executed during `mkdocs build` (MkDocs + `mkdocs-jupyter`).

## License

GNU LGPLv3 (or later). See `LICENSE` / `COPYING`.
