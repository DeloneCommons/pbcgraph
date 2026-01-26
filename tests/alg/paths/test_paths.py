import pytest

from pbcgraph import PeriodicDiGraph, PeriodicGraph
from pbcgraph.alg.paths import shortest_path_quotient


def test_shortest_path_directed():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,))
    G.add_edge('B', 'C', (0,))
    assert shortest_path_quotient(G, 'A', 'C') == ['A', 'B', 'C']


def test_shortest_path_weak_when_directed_fails():
    G = PeriodicDiGraph(dim=1)
    # B -> A and B -> C, so A -> C has no directed path.
    G.add_edge('B', 'A', (0,))
    G.add_edge('B', 'C', (0,))

    with pytest.raises(ValueError):
        shortest_path_quotient(G, 'A', 'C', connectivity='directed')

    assert shortest_path_quotient(
        G, 'A', 'C', connectivity='weak'
    ) == ['A', 'B', 'C']


def test_shortest_path_periodicgraph_defaults_to_weak_and_rejects_directed():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'B', (0,))
    G.add_edge('B', 'C', (0,))

    # Default is weak for PeriodicGraph.
    assert shortest_path_quotient(G, 'A', 'C') == ['A', 'B', 'C']

    with pytest.raises(ValueError):
        shortest_path_quotient(G, 'A', 'C', connectivity='directed')
