from collections.abc import Mapping

from pbcgraph import PeriodicMultiDiGraph


def test_edges_tvec_flag_shapes_and_order():
    G = PeriodicMultiDiGraph(dim=2)
    k = G.add_edge('A', 'B', (1, 0), tag='x')

    assert list(G.edges(tvec=True)) == [('A', 'B', (1, 0))]
    assert list(G.edges(keys=True, tvec=True, data=False)) == [
        ('A', 'B', (1, 0), k)
    ]

    e = list(G.edges(keys=True, tvec=True, data=True))
    assert len(e) == 1
    u, v, tv, kk, attrs = e[0]
    assert (u, v, tv, kk) == ('A', 'B', (1, 0), k)
    assert isinstance(attrs, Mapping)
    assert attrs['tag'] == 'x'
