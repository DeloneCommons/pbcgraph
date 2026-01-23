from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components


def test_torsion_in_1d_loop_translation_2():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'A', (2,))

    c = components(G)[0]
    assert c.rank == 1
    assert c.torsion_invariants == (2,)

    assert c.inst_key(('A', (0,))) == (0,)
    assert c.inst_key(('A', (1,))) == (1,)
    assert c.inst_key(('A', (2,))) == (0,)

    assert c.same_fragment(('A', (0,)), ('A', (2,))) is True
    assert c.same_fragment(('A', (0,)), ('A', (1,))) is False
