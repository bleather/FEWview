"""Matplotlib diagnostic plots.

:func:`plot_volume_slice` renders a 2D coordinate-plane slice through a
:class:`~fewview.volume.RetardedTimeVolume`, which is a fast way to check a
field before committing to a full volume render.
"""

from __future__ import annotations

from ._core import paper_style_file, plot_volume_slice

__all__ = ["plot_volume_slice", "paper_style_file"]
