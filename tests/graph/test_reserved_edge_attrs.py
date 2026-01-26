import pytest

from pbcgraph import (
    PBC_META_KEY,
    PeriodicDiGraph,
    PeriodicGraph,
    PeriodicMultiDiGraph,
    PeriodicMultiGraph,
)


def test_reserved_meta_key_rejected_in_add_edge_directed():
    G = PeriodicDiGraph(dim=1)
    with pytest.raises(ValueError):
        G.add_edge('A', 'B', (0,), **{PBC_META_KEY: {'x': 1}})


def test_reserved_meta_key_rejected_in_add_edge_undirected():
    G = PeriodicGraph(dim=1)
    with pytest.raises(ValueError):
        G.add_edge('A', 'A', (1,), **{PBC_META_KEY: {'x': 1}})


def test_reserved_meta_key_rejected_in_set_edge_attrs_directed():
    G = PeriodicMultiDiGraph(dim=1)
    k = G.add_edge('A', 'B', (0,), kind='bond')
    with pytest.raises(ValueError):
        G.set_edge_attrs('A', 'B', k, **{PBC_META_KEY: {'x': 1}})


def test_reserved_meta_key_rejected_in_set_edge_attrs_undirected():
    G = PeriodicMultiGraph(dim=1)
    k = G.add_edge('A', 'A', (1,), kind='bond')
    with pytest.raises(ValueError):
        G.set_edge_attrs('A', 'A', k, **{PBC_META_KEY: {'x': 1}})
