"""Polar strain surfaces.

A :class:`StrainSurface` is a lightweight 3D polar surface whose height encodes
the plus polarization at retarded time. It is a quick, matplotlib-only
alternative to the full volume renderer, useful for diagnostic figures.
:func:`build_strain_surface` constructs one from a strain time series and
:func:`plot_strain_surface` draws it.
"""

from __future__ import annotations

from ._core import StrainSurface, build_strain_surface, plot_strain_surface

__all__ = [
    "StrainSurface",
    "build_strain_surface",
    "plot_strain_surface",
]
