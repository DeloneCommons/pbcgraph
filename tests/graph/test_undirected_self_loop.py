import pytest

from pbcgraph import PeriodicGraph, PeriodicMultiGraph


def test_periodicgraph_supports_self_loop_periodic_edge():
    G = PeriodicGraph(dim=1)
    k = G.add_edge('A', 'A', (1,), kind='bond')

    # Stored as two directed realizations.
    assert G.number_of_edges() == 2
    assert G.has_edge('A', 'A') is True
    assert G.has_edge('A', 'A', key=k) is True

    edges = list(G.edges(keys=True, data=True, tvec=True))
    edge_quads = [(u, v, t, kk) for (u, v, t, kk, _a) in edges]
    assert ('A', 'A', (1,), k) in edge_quads
    assert ('A', 'A', (-1,), k) in edge_quads

    # Deterministic access: "forward" tvec is the one given in add_edge.
    assert G.edge_tvec('A', 'A', k) == (1,)

    attrs = G.get_edge_data('A', 'A', k)
    assert attrs['kind'] == 'bond'
    with pytest.raises(TypeError):
        attrs['x'] = 1

    G.set_edge_attrs('A', 'A', k, strength=7)
    assert G.get_edge_data('A', 'A', k)['strength'] == 7

    # Instance neighbors show the infinite-lift semantics.
    nbs = list(G.neighbors_inst(('A', (0,)), keys=True, data=False))
    assert ('A', (1,), k) in nbs
    assert ('A', (-1,), k) in nbs

    comp = G.components()[0]
    assert comp.rank == 1

    rep = G.check_invariants()
    assert rep['ok'] is True

    G.remove_edge('A', 'A', k)
    assert G.number_of_edges() == 0
    assert G.has_edge('A', 'A') is False


def test_periodicgraph_self_loop_rejects_duplicate_tvec_up_to_reversal():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'A', (2,))
    with pytest.raises(ValueError):
        G.add_edge('A', 'A', (2,))
    with pytest.raises(ValueError):
        G.add_edge('A', 'A', (-2,))


def test_periodicmultigraph_allows_parallel_self_loop_edges():
    G = PeriodicMultiGraph(dim=1)
    k1 = G.add_edge('A', 'A', (1,), label='x')
    k2 = G.add_edge('A', 'A', (1,), label='y')

    assert k1 != k2
    assert G.number_of_edges() == 4
    assert G.has_edge('A', 'A', key=k1) is True
    assert G.has_edge('A', 'A', key=k2) is True

    edges = list(G.edges(keys=True, data=True, tvec=True))
    t1 = sorted([t for (_u, _v, t, k, _a) in edges if k == k1])
    t2 = sorted([t for (_u, _v, t, k, _a) in edges if k == k2])

    assert t1 == [(-1,), (1,)]
    assert t2 == [(-1,), (1,)]

    assert G.check_invariants()['ok'] is True

    G.remove_edge('A', 'A', k1)
    assert G.number_of_edges() == 2
    assert G.has_edge('A', 'A', key=k1) is False
    assert G.has_edge('A', 'A', key=k2) is True
