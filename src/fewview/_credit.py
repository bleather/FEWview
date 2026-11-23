"""The FEWview render credit, using the supplied transparent wordmark."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont


def _light_background(color):
    red, green, blue = to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue > 0.55


@lru_cache(maxsize=2)
def _credit_image(light_background):
    from matplotlib.font_manager import findfont

    filename = "FEWview-on-light.png" if light_background else "FEWview-on-dark.png"
    package = Path(__file__).resolve().parent
    artwork = package / "brand" / filename
    if not artwork.is_file():
        # Hatch installs docs/img as fewview/brand when building a wheel.
        artwork = package.parents[1] / "docs" / "img" / filename

    roman = ImageFont.truetype(findfont("cmr10"), 96)
    # In cmr10 the © code point draws capital phi. STIX has the correct glyph.
    symbol = ImageFont.truetype(findfont("STIXGeneral"), 90)
    prefix, owner = "Rendered by", "Benjamin Leather"
    ink = "#26313c" if light_background else "#e6ebf2"
    with Image.open(artwork) as source:
        mark = source.convert("RGBA").crop(source.getbbox())
    mark_height = 210
    mark = mark.resize((round(mark.width * mark_height / mark.height), mark_height),
                       Image.Resampling.LANCZOS)
    gap = round(roman.getlength(" "))
    mark_x = round(roman.getlength(prefix)) + gap
    symbol_x = mark_x + mark.width + gap
    owner_x = symbol_x + round(symbol.getlength("©")) + gap
    image = Image.new("RGBA", (owner_x + round(roman.getlength(owner)) + 3, mark_height))
    image.alpha_composite(mark, (mark_x, 0))
    draw = ImageDraw.Draw(image)
    baseline = mark_height // 2 + 30
    for x, text, font in ((0, prefix, roman), (symbol_x, "©", symbol), (owner_x, owner, roman)):
        draw.text((x, baseline), text, font=font, fill=ink, anchor="ls")
    return image


def _display_width(width, height):
    return min(width * 0.94, max(230, height * 0.48))


def add_figure_credit(figure):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    if any(artist.get_gid() == "fewview-render-credit" for artist in figure.artists):
        return
    image = _credit_image(_light_background(figure.get_facecolor()))
    width, height = figure.get_size_inches() * 72  # OffsetImage zoom is in points.
    display_width = min(width * 0.90, max(155, height * 0.36))
    credit = AnnotationBbox(
        OffsetImage(np.asarray(image), zoom=display_width / image.width,
                    interpolation="hanning", resample=True),
        (0.985, 0.012), xycoords=figure.transFigure, box_alignment=(1, 0),
        frameon=False, pad=0, annotation_clip=False,
    )
    credit.set_gid("fewview-render-credit")
    figure.add_artist(credit)


def add_volume_credit(plotter, width, height, background):
    import pyvista as pv

    image = _credit_image(_light_background(background))
    display_width = _display_width(width, height)
    size = (display_width / width, display_width * image.height / image.width / height)
    widget = plotter.add_logo_widget(
        pv.Texture(np.asarray(image)).to_image(), position=(0.985 - size[0], 0.018), size=size,
    )
    widget.GetRepresentation().SetShowBorderToOff()
    widget.ProcessEventsOff()


def credit_strip(width, height, scale, background):
    """Reserve whole logical rows so the finished frame keeps its requested size."""
    image = _credit_image(_light_background(background))
    text_width = max(1, round(_display_width(width, height) * scale))
    text_height = max(1, round(text_width * image.height / image.width))
    padding = max(1, round(0.005 * height)) * scale
    rows = (text_height + 2 * padding + scale - 1) // scale
    strip = Image.new("RGB", (width * scale, rows * scale),
                      tuple(round(255 * value) for value in to_rgb(background)))
    text = image.resize((text_width, text_height), Image.Resampling.LANCZOS)
    strip.paste(text, (max(0, strip.width - text_width - padding), padding), text)
    return np.asarray(strip)
