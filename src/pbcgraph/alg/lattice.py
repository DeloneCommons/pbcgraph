"""Compatibility shim for lattice utilities.

The lattice/SNF implementation lives in :mod:`pbcgraph.lattice`.
"""

from pbcgraph.lattice.snf import (
    SNFDecomposition,
    smith_normal_form,
    smith_normal_form_with_transform,
    snf_decomposition,
)

__all__ = [
    'SNFDecomposition',
    'smith_normal_form',
    'smith_normal_form_with_transform',
    'snf_decomposition',
]
