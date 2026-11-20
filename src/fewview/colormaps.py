"""Colour palettes and opacity profiles.

Fewview accepts these Matplotlib palettes (``"magma"``, ``"viridis"``,
``"plasma"``, ``"inferno"``, ``"cividis"``, ``"cool"``, ``"blues"``) plus four
palettes tuned for volume rendering:

``rainbow``
    A perceptually smooth blue-cyan-green-gold-red sequence for signed-strain
    shells.
``aurora``
    A luminous magma variant that lifts weak values off black.
``cinematic``
    A cool-to-warm sequence with a bright core.

``ice``
    A navy-to-white sequence.

The **opacity profile** decides how scalar values map to transparency:

``soft``
    Broad, smooth fronts. A good default.
``bands``
    Symmetric scalar levels.
``shells``
    Nested translucent positive-strain sheets. Requires a ``plus`` or ``cross``
    strain component.
``flux``
    Logarithmically compressed energy flux. Requires the ``energy_flux``
    component.
"""

from __future__ import annotations

from ._core import _MATPLOTLIB_COLOR_SCHEMES, _VOLUME_COLOR_SCHEMES

#: Opacity profiles accepted by the renderers.
OPACITY_PROFILES: tuple[str, ...] = ("soft", "bands", "shells", "flux")

#: Presentation presets accepted by the renderers.
PRESENTATIONS: tuple[str, ...] = ("balanced", "dramatic", "shells_dramatic")


def available_color_schemes() -> tuple[str, ...]:
    """Return every accepted ``color_scheme`` name, sorted."""

    return tuple(sorted(_VOLUME_COLOR_SCHEMES))


def is_matplotlib_scheme(name: str) -> bool:
    """Whether ``name`` is a passthrough to a Matplotlib colormap."""

    return name in _MATPLOTLIB_COLOR_SCHEMES


__all__ = [
    "OPACITY_PROFILES",
    "PRESENTATIONS",
    "available_color_schemes",
    "is_matplotlib_scheme",
]
