from types import MappingProxyType

import pytest

from pbcgraph import PeriodicDiGraph, PeriodicGraph


def test_undirected_edges_unique_self_loop() -> None:
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'A', (1,), label='bond')

    recs = list(G.undirected_edges_unique(keys=True, data=True, tvec=True))
    assert len(recs) == 1

    u, v, t, k, attrs = recs[0]
    assert u == 'A'
    assert v == 'A'
    assert t == (-1,)
    assert k == 0
    assert isinstance(attrs, MappingProxyType)
    assert attrs['label'] == 'bond'


def test_undirected_edges_unique_non_loop() -> None:
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'B', (0,), label='ab')

    recs = list(G.undirected_edges_unique(keys=True, data=False, tvec=True))
    assert recs == [('A', 'B', (0,), 0)]


def test_undirected_edges_unique_directed_raises() -> None:
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,))

    with pytest.raises(TypeError):
        list(G.undirected_edges_unique())
