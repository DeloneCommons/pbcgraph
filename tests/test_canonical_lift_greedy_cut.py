from collections import deque

from pbcgraph import PeriodicDiGraph
from pbcgraph.alg.components import components
from pbcgraph.core.types import add_tvec


def _internal_adj(component, shift_map):
    adj = {u: set() for u in component.nodes}
    for u in component.nodes:
        su = shift_map[u]
        for v, t, _k in component.graph.neighbors(u, keys=True, data=False):
            if v not in component.nodes:
                continue
            if shift_map[v] == add_tvec(su, t):
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _is_connected(adj, nodes):
    nodes = list(nodes)
    if not nodes:
        return True
    start = sorted(nodes, key=str)[0]
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in sorted(adj[u], key=str):
            if v in seen:
                continue
            seen.add(v)
            q.append(v)
    return len(seen) == len(nodes)


def test_canonical_lift_greedy_cut_improves_score_and_preserves_connectivity():
    # A 1D quotient where best_anchor is good but local redistribution can
    # further reduce the score while preserving internal connectivity.
    G = PeriodicDiGraph(dim=1)
    G.add_edge('A', 'B', (2,))
    G.add_edge('B', 'C', (98,))
    # Two distinct quotient edges between C and A. The spanning-tree
    # potentials use the first one in deterministic order, while
    # `greedy_cut` can locally switch to the other to reduce score.
    G.add_edge('C', 'A', (-100,))
    G.add_edge('C', 'A', (-99,))

    c = components(G)[0]

    out_best = c.canonical_lift(
        anchor_shift=(0,), placement='best_anchor', score='l1'
    )
    out_greedy = c.canonical_lift(
        anchor_shift=(0,), placement='greedy_cut', score='l1'
    )

    assert out_greedy.placement == 'greedy_cut'
    assert out_greedy.anchor_site == out_best.anchor_site
    assert out_greedy.anchor_shift == out_best.anchor_shift
    assert out_greedy.score <= out_best.score
    assert out_greedy.score < out_best.score

    shift_map = {u: s for u, s in out_greedy.nodes}
    assert set(shift_map) == set(c.nodes)

    for u, s in out_greedy.nodes:
        assert c.inst_key((u, s)) == out_greedy.strand_key

    adj = _internal_adj(c, shift_map)
    assert _is_connected(adj, c.nodes)
