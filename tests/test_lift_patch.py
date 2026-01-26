import networkx as nx
import pytest

from pbcgraph import PeriodicDiGraph, PeriodicGraph, PeriodicMultiGraph
from pbcgraph.core.exceptions import LiftPatchError


def test_lift_patch_radius_uses_weak_connectivity():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,))

    patch = G.lift_patch(('B', (0,)), radius=1)

    assert ('B', (0,)) in patch.nodes
    assert ('A', (0,)) in patch.nodes
    assert len(patch.edges) == 1


def test_lift_patch_box_rel_bounds_and_termination():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'A', (1,))

    patch = G.lift_patch(('A', (0,)), box_rel=((0, 3),))
    assert patch.nodes == (('A', (0,)), ('A', (1,)), ('A', (2,)))
    assert len(patch.edges) == 2


def test_lift_patch_multigraph_preserves_key_and_dedupes_reciprocals():
    G = PeriodicMultiGraph(dim=1)
    G.add_edge('A', 'A', (1,), key=7, kind='bond')

    patch = G.lift_patch(('A', (0,)), box=((0, 2),))
    assert patch.nodes == (('A', (0,)), ('A', (1,)))
    assert len(patch.edges) == 1

    u, v, key, attrs = patch.edges[0]
    assert {u, v} == {('A', (0,)), ('A', (1,))}
    assert key == 7
    assert attrs['kind'] == 'bond'

    nxG = patch.to_networkx()
    assert isinstance(nxG, nx.MultiGraph)


def test_lift_patch_directed_attr_collapse_is_deterministic():
    G = PeriodicDiGraph(dim=1)
    # Note: edge keys are only unique per (u, v) pair in NetworkX, so these
    # two directed edges can both receive key 0.
    G.add_edge('A', 'B', (0,), label='x')
    G.add_edge('B', 'A', (0,), label='y')

    patch = G.lift_patch(('A', (0,)), radius=1)
    assert len(patch.edges) == 1

    u, v, attrs = patch.edges[0]
    assert {u, v} == {('A', (0,)), ('B', (0,))}
    assert attrs['label'] == 'x'


def test_lift_patch_to_networkx_snapshots_edge_attrs():
    G = PeriodicGraph(dim=1)
    k = G.add_edge('A', 'A', (1,), kind='bond')

    patch = G.lift_patch(('A', (0,)), box=((0, 2),))
    nxG = patch.to_networkx()

    (u, v) = list(nxG.edges())[0]
    nxG.edges[u, v]['kind'] = 'modified'

    # Underlying pbcgraph edge attrs are unaffected.
    assert G.get_edge_data('A', 'A', k)['kind'] == 'bond'


def test_lift_patch_requires_a_finiteness_constraint():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'A', (1,))
    with pytest.raises(LiftPatchError):
        G.lift_patch(('A', (0,)))