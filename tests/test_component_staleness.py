import pytest

from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components
from pbcgraph import StaleComponentError


def test_component_staleness_structural_vs_data_versions():
    G = PeriodicDiGraph(dim=1)
    k = G.add_edge('A', 'A', (2,))
    c = components(G)[0]
    assert c.is_stale() is False

    # Data-only update does not stale.
    G.set_edge_attrs('A', 'A', k, foo=1)
    assert c.is_stale() is False
    assert c.inst_key(('A', (1,))) == (1,)

    # Structural update stales.
    G.add_edge('A', 'B', (0,))
    assert c.is_stale() is True
    with pytest.raises(StaleComponentError):
        c.inst_key(('A', (0,)))
