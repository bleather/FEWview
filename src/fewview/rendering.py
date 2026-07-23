"""Volume rendering of EMRI waveforms.

Three entry points, in increasing order of capability:

* :func:`render_volume` renders a single prebuilt
  :class:`~fewview.volume.RetardedTimeVolume`.
* :func:`render_mode_frame` builds the volume for one instant of a mode
  waveform and renders it, optionally with the central bodies, the fading
  inspiral trail and a synchronized strain panel.
* :func:`render_mode_animation` sweeps a range of frame times to an MP4 or GIF,
  reusing the expensive angular basis across frames and holding the colour
  scale fixed for a flicker-free movie.

Appearance is controlled by an ``opacity_profile`` (``"soft"``, ``"bands"``,
``"shells"`` or ``"flux"``), a ``color_scheme`` (see
:mod:`fewview.colormaps`) and a ``presentation`` preset (``"balanced"``,
``"dramatic"`` or ``"shells_dramatic"``).
"""

from __future__ import annotations

from ._core import (
    DEFAULT_TRAJECTORY_COLOR,
    render_mode_animation,
    render_mode_frame,
    render_volume,
)

__all__ = [
    "render_volume",
    "render_mode_frame",
    "render_mode_animation",
    "DEFAULT_TRAJECTORY_COLOR",
]
