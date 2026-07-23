r"""Fewview: full-sky volume visualizations for FEW EMRI waveforms.

Fewview turns a `FastEMRIWaveforms <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms>`_
waveform into a spherical, retarded-time picture of the gravitational wave. The
fully relativistic ``FastKerrEccentricEquatorialFlux`` modes are reconstructed
into :math:`h_{\ell m}(t)`, combined with spin-weighted spherical harmonics over
the whole sky, and evaluated at ``u = t_frame - r / c`` throughout a sphere so
the wave is seen propagating outward.

A minimal still::

    import numpy as np
    from fewview import (
        generate_relativistic_mode_waveform,
        choose_max_delay,
        render_mode_frame,
    )

    wf = generate_relativistic_mode_waveform(
        M=1e6, mu=10.0, a=0.9, p0=12.0, e0=0.4, xI0=1.0,
        dt=10.0, T=0.01, lmax=10, nmax=55,
    )
    ref = wf.strain(theta=np.pi / 3.0, phi=0.0)
    max_delay = choose_max_delay(wf.time, np.real(ref), -np.imag(ref))
    render_mode_frame(
        wf, screenshot="emri.png",
        max_delay=max_delay, frame_time=wf.time[-1],
        waveform_start_time=wf.time[0], waveform_end_time=wf.time[-1],
        component="plus", opacity_profile="shells", color_scheme="rainbow",
    )

The public API is re-exported here for convenience and is also grouped into
topical submodules: :mod:`fewview.waveform`, :mod:`fewview.volume`,
:mod:`fewview.rendering`, :mod:`fewview.surface`, :mod:`fewview.plotting`
and :mod:`fewview.colormaps`.
"""

from __future__ import annotations

from ._version import __version__
from .colormaps import (
    OPACITY_PROFILES,
    PRESENTATIONS,
    available_color_schemes,
)
from .plotting import plot_volume_slice
from .rendering import (
    DEFAULT_TRAJECTORY_COLOR,
    render_mode_animation,
    render_mode_frame,
    render_volume,
)
from .surface import StrainSurface, build_strain_surface, plot_strain_surface
from .volume import (
    RetardedTimeVolume,
    build_mode_retarded_time_volume,
    build_retarded_time_volume,
    save_volume,
    to_pyvista,
)
from .waveform import (
    RelativisticModeWaveform,
    choose_max_delay,
    estimate_waveform_period,
    generate_relativistic_mode_waveform,
    polarizations_from_complex,
)

__all__ = [
    "__version__",
    # waveform
    "RelativisticModeWaveform",
    "generate_relativistic_mode_waveform",
    "polarizations_from_complex",
    "estimate_waveform_period",
    "choose_max_delay",
    # volume
    "RetardedTimeVolume",
    "build_mode_retarded_time_volume",
    "build_retarded_time_volume",
    "to_pyvista",
    "save_volume",
    # surface
    "StrainSurface",
    "build_strain_surface",
    "plot_strain_surface",
    # rendering
    "render_volume",
    "render_mode_frame",
    "render_mode_animation",
    "DEFAULT_TRAJECTORY_COLOR",
    # plotting
    "plot_volume_slice",
    # colormaps
    "available_color_schemes",
    "OPACITY_PROFILES",
    "PRESENTATIONS",
]
