"""
pbcgraph: Periodic graphs on a translation lattice.

This package provides graph containers and algorithms for periodic
(crystallographic) graphs represented on a reference cell.
Nodes live in the reference cell, while each directed edge carries
an integer translation vector ``tvec`` in Z^d indicating how the
cell index changes when traversing the edge.

Core idea
---------
A periodic bond is represented as a directed edge ``u -> v``
labeled by ``tvec``. The corresponding realized neighbor
of ``v`` is the node instance ``(v, tvec)``.
Undirected periodic graphs are stored as two directed realizations:
``u -> v`` with ``tvec`` and ``v -> u`` with ``-tvec``.

Container families
------------------
Simple periodic graphs (default):
    - :class:`~pbcgraph.PeriodicDiGraph` / :class:`~pbcgraph.PeriodicGraph`
    - At most one edge is allowed per unique ``(u, v, tvec)`` combination.

Periodic multigraphs:
    - :class:`~pbcgraph.PeriodicMultiDiGraph` /
      :class:`~pbcgraph.PeriodicMultiGraph`
    - Multiple parallel edges are allowed for the same ``(u, v, tvec)``
      (distinguished by edge keys). This is useful when you want to store
      multiple contact types between the same motif nodes without aggregating
      them.

Example
-------
Create a 1D periodic chain with one node per cell and a bond to the next cell::

    from pbcgraph import PeriodicDiGraph

    G = PeriodicDiGraph(dim=1)
    G.add_node('A')
    G.add_edge('A', 'A', tvec=(1,))   # A(i) -> A(i+1)
    # For an undirected representation, use PeriodicGraph

    # A crystallographically common pattern is a quotient self-loop with
    # non-zero translation (bond to a periodic image):

    from pbcgraph import PeriodicGraph

    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'A', (1,), kind='bond')

    # (adds both directions).

Algorithms
----------
Algorithms are available under :mod:`pbcgraph.alg`, including connected
components, quotient shortest paths, and Smith normal form (SNF)
based lattice summaries for components.

See also
--------
- :mod:`pbcgraph.graph`      Graph containers and edge semantics.
- :mod:`pbcgraph.component`  Connected components and component invariants.
- :mod:`pbcgraph.alg`        Algorithms (components, paths, lattice/SNF).
"""


from pbcgraph.__about__ import __version__
from pbcgraph.graph import (
    PeriodicDiGraph,
    PeriodicGraph,
    PeriodicMultiDiGraph,
    PeriodicMultiGraph,
)
from pbcgraph.component import PeriodicComponent

from pbcgraph.core.constants import PBC_META_KEY
from pbcgraph.core.exceptions import PBCGraphError, StaleComponentError
from pbcgraph.core.types import (
    TVec,
    NodeId,
    EdgeKey,
    NodeInst,
    zero_tvec,
    validate_tvec,
    add_tvec,
    sub_tvec,
    neg_tvec,
)

__all__ = [
    '__version__',
    'PBC_META_KEY',
    # containers
    'PeriodicDiGraph',
    'PeriodicGraph',
    'PeriodicMultiDiGraph',
    'PeriodicMultiGraph',
    'PeriodicComponent',
    # exceptions
    'PBCGraphError',
    'StaleComponentError',
    # core types + tvec helpers
    'TVec',
    'NodeId',
    'EdgeKey',
    'NodeInst',
    'zero_tvec',
    'validate_tvec',
    'add_tvec',
    'sub_tvec',
    'neg_tvec',
]
