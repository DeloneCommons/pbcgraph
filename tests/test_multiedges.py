import pytest

from pbcgraph import (
    PeriodicDiGraph,
    PeriodicGraph,
    PeriodicMultiDiGraph,
    PeriodicMultiGraph,
)


def test_parallel_edges_distinct_tvec_and_iteration_order():
    G = PeriodicDiGraph(dim=1)
    k0 = G.add_edge('A', 'B', (0,), tag='e0')
    k1 = G.add_edge('A', 'B', (1,), tag='e1')

    assert k0 != k1
    assert G.has_edge('A', 'B', key=k0)
    assert G.has_edge('A', 'B', key=k1)

    edges = list(G.edges(keys=True, data=True))
    assert edges[0][0:3] == ('A', 'B', k0)
    assert edges[1][0:3] == ('A', 'B', k1)
    assert edges[0][3]['tag'] == 'e0'
    assert edges[1][3]['tag'] == 'e1'

    nbrs = list(G.neighbors('A', keys=True, data=False))
    assert nbrs[0] == ('B', (0,), k0)
    assert nbrs[1] == ('B', (1,), k1)


def test_simple_graph_disallows_duplicate_tvec_between_same_nodes():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,), tag='e0')
    with pytest.raises(ValueError):
        G.add_edge('A', 'B', (0,), tag='e1')


def test_multigraph_allows_duplicate_tvec_between_same_nodes():
    G = PeriodicMultiDiGraph(dim=1)
    k0 = G.add_edge('A', 'B', (0,), tag='e0')
    k1 = G.add_edge('A', 'B', (0,), tag='e1')
    assert k0 != k1

    nbrs = list(G.neighbors('A', keys=True, data=True))
    assert len(nbrs) == 2
    assert [n[1] for n in nbrs] == [(0,), (0,)]
    assert {n[3]['tag'] for n in nbrs} == {'e0', 'e1'}


def test_simple_undirected_graph_disallows_duplicate_undirected_edges():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'B', (1,), tag='e0')

    # Same direction and translation.
    with pytest.raises(ValueError):
        G.add_edge('A', 'B', (1,), tag='e1')

    # Reversed representation of the same undirected periodic edge.
    with pytest.raises(ValueError):
        G.add_edge('B', 'A', (-1,), tag='e2')


def test_multigraph_allows_duplicate_undirected_edges():
    G = PeriodicMultiGraph(dim=1)
    k0 = G.add_edge('A', 'B', (1,), tag='e0')
    k1 = G.add_edge('A', 'B', (1,), tag='e1')
    assert k0 != k1
    assert len(list(G.edges(keys=True, data=True))) == 4
