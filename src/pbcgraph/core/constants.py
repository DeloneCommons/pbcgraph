"""Shared constants.

This module centralizes constant values used across pbcgraph.

Some constants are part of the public API when they define interoperability
contracts with external data structures.
"""

from __future__ import annotations


# Reserved key used to store pbcgraph export metadata inside external
# data structures (e.g. NetworkX edge attribute dictionaries).
PBC_META_KEY = '__pbcgraph__'
