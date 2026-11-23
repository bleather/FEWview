"""Transparent FEWview artwork for the lower-right corner of a render."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image


Logo = Literal["wordmark", "icon"]
_INSET = 0.015


@lru_cache(maxsize=3)
def _load_artwork(filename):
    package = Path(__file__).resolve().parent
    artwork = package / "brand" / filename
    if not artwork.is_file():
        # Hatch installs docs/img as fewview/brand when building a wheel.
        artwork = package.parents[1] / "docs" / "img" / filename
    with Image.open(artwork) as source:
        return source.convert("RGBA").crop(source.getbbox())


def _artwork(logo, background):
    if logo == "icon":
        return _load_artwork("FEWview-symbol.png")
    if logo != "wordmark":
        raise ValueError("logo must be 'wordmark' or 'icon'")
    red, green, blue = to_rgb(background)
    light = 0.299 * red + 0.587 * green + 0.114 * blue > 0.55
    return _load_artwork("FEWview-on-light.png" if light else "FEWview-on-dark.png")


def _display_size(image, width, height):
    # Both variants have the same height, even in a narrow portrait frame.
    mark_height = min(height * 0.075, width * 0.055)
    return mark_height * image.width / image.height, mark_height


def add_figure_credit(figure, logo):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    image = _artwork(logo, figure.get_facecolor())
    for artist in tuple(figure.artists):
        if artist.get_gid() == "fewview-render-credit":
            artist.remove()
    width, height = figure.get_size_inches() * 72  # OffsetImage zoom is in points.
    display_width, _ = _display_size(image, width, height)
    artist = AnnotationBbox(
        OffsetImage(np.asarray(image), zoom=display_width / image.width,
                    interpolation="hanning", resample=True),
        (1 - _INSET, _INSET), xycoords=figure.transFigure, box_alignment=(1, 0),
        frameon=False, pad=0, annotation_clip=False,
    )
    artist.set_gid("fewview-render-credit")
    figure.add_artist(artist)


def add_volume_credit(plotter, width, height, background, logo):
    import pyvista as pv

    image = _artwork(logo, background)
    mark_width, mark_height = _display_size(image, width, height)
    size = (mark_width / width, mark_height / height)
    widget = plotter.add_logo_widget(
        pv.Texture(np.asarray(image)).to_image(), position=(1 - _INSET - size[0], _INSET), size=size,
    )
    widget.GetRepresentation().SetShowBorderToOff()
    widget.ProcessEventsOff()


def prepare_frame_logo(width, height, background, logo):
    image = _artwork(logo, background)
    size = tuple(max(1, round(v)) for v in _display_size(image, width, height))
    return image.resize(size, Image.Resampling.LANCZOS)


def stamp_frame(frame, mark):
    """Composite into a writable frame, touching only the logo rectangle."""
    height, width = frame.shape[:2]
    right = width - round(width * _INSET)
    bottom = height - round(height * _INSET)
    target = frame[bottom - mark.height:bottom, right - mark.width:right]
    patch = Image.fromarray(target)
    patch.paste(mark, mask=mark.getchannel("A"))
    if not frame.flags.writeable:
        frame = frame.copy()
    frame[bottom - mark.height:bottom, right - mark.width:right] = np.asarray(patch)
    return frame
