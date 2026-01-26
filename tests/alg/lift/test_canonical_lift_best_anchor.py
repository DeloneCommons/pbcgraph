from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components


def test_canonical_lift_best_anchor_selects_min_score_anchor():
    # Build a 1D directed quotient where potentials are highly unbalanced:
    # pot(A)=0, pot(B)=2, pot(C)=100 (deterministic root is A).
    # Add an extra edge to make the translation subgroup L = Z so scoring is
    # sensitive to absolute displacements.
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (2,))
    G.add_edge('B', 'C', (98,))
    G.add_edge('C', 'A', (-99,))  # cycle generator = 1 -> L = Z

    c = components(G)[0]
    K0 = c.inst_key(('A', (0,)))
    assert c.inst_key(('B', (0,))) == K0
    assert c.inst_key(('C', (0,))) == K0

    out_tree = c.canonical_lift(
        anchor_shift=(0,), placement='tree', score='l1'
    )
    out_best = c.canonical_lift(
        anchor_shift=(0,), placement='best_anchor', score='l1'
    )

    assert out_tree.anchor_site == 'A'
    assert out_best.anchor_site == 'B'
    assert out_best.placement == 'best_anchor'
    assert out_best.score < out_tree.score

    # Still returns exactly one instance per quotient node.
    assert {u for u, _s in out_best.nodes} == {'A', 'B', 'C'}
    assert len(out_best.nodes) == 3
