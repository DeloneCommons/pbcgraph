import importlib
import pkgutil


def test_import_orders_do_not_cause_cycles():
    # Try several common "entry points" in different orders.
    import pbcgraph.alg  # noqa: F401
    import pbcgraph.graph  # noqa: F401
    import pbcgraph.component  # noqa: F401
    import pbcgraph.lattice  # noqa: F401
    import pbcgraph.core  # noqa: F401
    import pbcgraph  # noqa: F401


def test_walk_all_submodules_import_cleanly():
    import pbcgraph

    for mod in pkgutil.walk_packages(
        pbcgraph.__path__, pbcgraph.__name__ + '.'
    ):
        importlib.import_module(mod.name)
