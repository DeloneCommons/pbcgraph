"""Core building blocks.

This subpackage holds foundational types/protocols/exceptions that are intended
not to depend on higher-level modules.

Most users should import from the top-level `pbcgraph` namespace.
"""

from pbcgraph.core.constants import PBC_META_KEY

__all__ = [
    'PBC_META_KEY',
]
