import pytest

from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components
from pbcgraph.core.exceptions import CanonicalLiftError
from pbcgraph.core.types import sub_tvec


def test_canonical_lift_tree_basic_properties_and_tree_edges():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (0,))
    G.add_edge('B', 'C', (0,))
    G.add_edge('C', 'A', (1,))

    c = components(G)[0]
    out = c.canonical_lift(
        seed=('B', (0,)), anchor_shift=(0,), return_tree=True
    )

    assert {u for u, _s in out.nodes} == {'A', 'B', 'C'}
    assert len(out.nodes) == 3
    assert out.anchor_site == 'A'
    assert out.anchor_shift == (0,)

    for u, s in out.nodes:
        assert c.inst_key((u, s)) == out.strand_key

    assert out.tree_edges is not None
    assert len(out.tree_edges) == 2

    shift_map = {u: s for u, s in out.nodes}
    children = set()
    for parent, child, tvec, key in out.tree_edges:
        assert parent in shift_map
        assert child in shift_map
        children.add(child)
        assert tvec == sub_tvec(shift_map[child], shift_map[parent])
        assert isinstance(key, int)
    assert children == {'B', 'C'}


def test_canonical_lift_tree_raises_when_strand_absent_in_anchor_cell():
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'A', (2,))

    c = components(G)[0]
    assert c.inst_key(('A', (0,))) == (0,)

    with pytest.raises(CanonicalLiftError):
        c.canonical_lift(strand_key=(1,), anchor_shift=(0,))
