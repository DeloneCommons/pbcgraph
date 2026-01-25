from collections.abc import Mapping

import pytest

from pbcgraph import PeriodicDiGraph


def test_neighbors_inst_lifts_translations():
    G = PeriodicDiGraph(dim=2)
    k1 = G.add_edge('A', 'B', (1, 0), kind='ab')
    k2 = G.add_edge('A', 'C', (0, -1), kind='ac')

    out = list(G.neighbors_inst(('A', (5, 5))))
    assert ('B', (6, 5)) in out
    assert ('C', (5, 4)) in out

    out_k = list(G.neighbors_inst(('A', (5, 5)), keys=True, data=False))
    assert ('B', (6, 5), k1) in out_k
    assert ('C', (5, 4), k2) in out_k

    out_d = list(G.neighbors_inst(('A', (5, 5)), keys=False, data=True))
    # attrs are read-only live views
    for v, s2, attrs in out_d:
        assert isinstance(attrs, Mapping)
        assert 'kind' in attrs
        with pytest.raises(TypeError):
            attrs['x'] = 1
