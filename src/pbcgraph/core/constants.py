"""Internal constants.

These constants are not part of the public API. They are provided to keep
internal conventions consistent across modules.
"""

from __future__ import annotations


# Reserved key used to store pbcgraph export metadata inside external
# data structures (e.g. NetworkX edge attribute dictionaries).
PBC_META_KEY = '__pbcgraph__'
