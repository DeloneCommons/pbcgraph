"""Periodic graph containers.

pbcgraph represents a periodic graph by a finite quotient graph, where each
directed quotient edge carries an integer translation vector in ``Z^d``.

Internally, quotient edges are stored in a NetworkX
:class:`networkx.MultiDiGraph`.

Two container families are provided:

- :class:`~pbcgraph.graph.PeriodicDiGraph` / :class:`~pbcgraph.graph.PeriodicGraph`:
  at most one edge per ``(u, v, tvec)``.
- :class:`~pbcgraph.graph.PeriodicMultiDiGraph` / :class:`~pbcgraph.graph.PeriodicMultiGraph`:
  allow multiple edges per ``(u, v, tvec)`` (distinguished by edge keys).
"""

from __future__ import annotations

from pbcgraph.graph.directed import PeriodicDiGraph
from pbcgraph.graph.multidirected import PeriodicMultiDiGraph
from pbcgraph.graph.undirected import PeriodicGraph
from pbcgraph.graph.multigraph import PeriodicMultiGraph

__all__ = [
    'PeriodicDiGraph',
    'PeriodicGraph',
    'PeriodicMultiDiGraph',
    'PeriodicMultiGraph',
]
