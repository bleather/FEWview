"""Waveform generation and mode reconstruction.

This module produces the mode-resolved, full-sky waveform that every Fewview
visualization is built from. :func:`generate_relativistic_mode_waveform` runs
the fully relativistic ``FastKerrEccentricEquatorialFlux`` model and returns a
:class:`RelativisticModeWaveform`, which carries the individual
:math:`h_{\\ell m}(t)` modes together with the FEW inspiral trajectory.

The remaining helpers characterise a waveform for display: the local
polarization period and a sensible amount of history to map from the centre of
the sphere to its edge.
"""

from __future__ import annotations

from ._core import (
    RelativisticModeWaveform,
    choose_max_delay,
    estimate_waveform_period,
    generate_relativistic_mode_waveform,
    polarizations_from_complex,
)

__all__ = [
    "RelativisticModeWaveform",
    "generate_relativistic_mode_waveform",
    "polarizations_from_complex",
    "estimate_waveform_period",
    "choose_max_delay",
]
