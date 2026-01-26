from pbcgraph import PeriodicDiGraph, PeriodicGraph
from pbcgraph.alg.components import components


def test_components_split_and_root_order():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,))
    G.add_edge('C', 'D', (0,))

    comps = components(G)
    assert len(comps) == 2
    assert comps[0].root == 'A'
    assert comps[1].root == 'C'
    assert comps[0].rank == 0
    assert comps[1].rank == 0


def test_rank_full_in_1d_from_cycle_translation():
    G = PeriodicDiGraph(dim=1)
    # Weakly connected component on {A,B}; cycle translation = 1.
    G.add_edge('A', 'B', (0,))
    G.add_edge('B', 'A', (1,))

    c = components(G)[0]
    assert c.rank == 1
    assert c.torsion_invariants == ()  # Z / Z is trivial


def test_periodicgraph_rank_zero_for_simple_undirected_edge():
    G = PeriodicGraph(dim=1)
    G.add_edge('A', 'B', (0,))
    c = components(G)[0]
    assert c.rank == 0
