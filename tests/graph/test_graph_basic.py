import pytest

from pbcgraph import PeriodicDiGraph, PeriodicGraph, PeriodicMultiDiGraph


def test_add_edge_autocreates_nodes_and_stores_tvec():
    G = PeriodicDiGraph(dim=2)
    eid = G.add_edge('A', 'B', (1, 0), kind='x')

    assert G.has_node('A')
    assert G.has_node('B')
    assert G.has_edge('A', 'B', key=eid)

    assert G.edge_tvec('A', 'B', eid) == (1, 0)

    attrs = G.get_edge_data('A', 'B', eid)
    assert attrs['kind'] == 'x'

    with pytest.raises(ValueError):
        G.add_edge('A', 'B', (1, 0, 0))


def test_edge_attr_updates_increment_data_version_only():
    G = PeriodicDiGraph(dim=1)
    eid = G.add_edge('A', 'B', (0,), w=1)
    sv0 = G.structural_version
    dv0 = G.data_version

    G.set_edge_attrs('A', 'B', eid, w=2, label='hi')
    assert G.structural_version == sv0
    assert G.data_version == dv0 + 1
    attrs = G.get_edge_data('A', 'B', eid)
    assert attrs['w'] == 2
    assert attrs['label'] == 'hi'


def test_remove_edge_increments_structural_version():
    G = PeriodicDiGraph(dim=1)
    eid = G.add_edge('A', 'B', (0,))
    sv0 = G.structural_version
    G.remove_edge('A', 'B', eid)
    assert G.structural_version == sv0 + 1
    assert not G.has_edge('A', 'B', key=eid)


def test_set_node_attrs_increments_data_version():
    G = PeriodicDiGraph(dim=1)
    G.add_node('A')
    dv0 = G.data_version
    G.set_node_attrs('A', foo=1)
    assert G.data_version == dv0 + 1
    assert G.get_node_data('A')['foo'] == 1


def test_add_node_with_attrs_data_version_semantics():
    G = PeriodicDiGraph(dim=1)
    sv0 = G.structural_version
    dv0 = G.data_version

    G.add_node('A', foo=1)
    assert G.structural_version == sv0 + 1
    assert G.data_version == dv0
    assert G.get_node_data('A')['foo'] == 1

    G.add_node('A', bar=2)
    assert G.structural_version == sv0 + 1
    assert G.data_version == dv0 + 1
    assert G.get_node_data('A')['bar'] == 2


def test_periodicgraph_stores_two_directions_and_shared_attrs():
    G = PeriodicGraph(dim=2)
    k = G.add_edge('A', 'B', (1, -1), kind='contact')

    assert G.has_edge('A', 'B', key=k) is True
    assert G.has_edge('B', 'A', key=k) is True

    assert G.edge_tvec('A', 'B', k) == (1, -1)
    assert G.edge_tvec('B', 'A', k) == (-1, 1)

    attrs_ab = G.get_edge_data('A', 'B', k)
    attrs_ba = G.get_edge_data('B', 'A', k)
    assert dict(attrs_ab) == dict(attrs_ba)
    assert attrs_ab is not attrs_ba

    # Returned mappings are read-only views.
    with pytest.raises(TypeError):
        attrs_ab['x'] = 1

    # Updates propagate because paired realizations share the same
    # underlying user-attrs dict.
    G.set_edge_attrs('A', 'B', k, foo=1)
    assert G.get_edge_data('B', 'A', k)['foo'] == 1
    assert G.check_invariants()['ok'] is True
    assert attrs_ab['kind'] == 'contact'

    G.set_edge_attrs('A', 'B', k, strength=5)
    assert G.get_edge_data('B', 'A', k)['strength'] == 5

    G.remove_edge('A', 'B', k)
    assert not G.has_edge('A', 'B', key=k)
    assert not G.has_edge('B', 'A', key=k)


def test_edge_key_rejects_bool():
    G = PeriodicMultiDiGraph(dim=1)
    with pytest.raises(TypeError):
        G.add_edge('A', 'B', (0,), key=True)
