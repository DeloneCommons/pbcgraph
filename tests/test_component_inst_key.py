from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components


def test_inst_key_rank_zero_is_absolute_coordinate():
    G = PeriodicDiGraph(dim=2)
    G.add_edge('A', 'B', (0, 0))
    c = components(G)[0]

    assert c.rank == 0
    assert c.inst_key(('A', (1, 2))) == (1, 2)
    assert c.inst_key(('B', (1, 2))) == (1, 2)

    assert c.same_fragment(('A', (0, 0)), ('B', (0, 0))) is True
    assert c.same_fragment(('A', (0, 0)), ('B', (1, 0))) is False
