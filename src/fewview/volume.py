"""Retarded-time volumes.

A :class:`RetardedTimeVolume` is a spherical scalar field sampled at retarded
time ``u = t_frame - r / c``, so that the gravitational wave is seen radiating
outward from the source. :func:`build_mode_retarded_time_volume` evaluates the
mode-resolved field of a :class:`~fewview.waveform.RelativisticModeWaveform`
over every sky direction; :func:`build_retarded_time_volume` offers a lighter
approximation directly from a one-dimensional strain time series.

:func:`to_pyvista` and :func:`save_volume` hand the field to PyVista/VTK for
rendering or for export to a ParaView-compatible ``.vti`` file.
"""

from __future__ import annotations

from ._core import (
    RetardedTimeVolume,
    build_mode_retarded_time_volume,
    build_retarded_time_volume,
    save_volume,
    to_pyvista,
)

__all__ = [
    "RetardedTimeVolume",
    "build_mode_retarded_time_volume",
    "build_retarded_time_volume",
    "to_pyvista",
    "save_volume",
]
