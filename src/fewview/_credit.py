"""The FEWview render credit, using the supplied transparent wordmark."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFont


# In both supplied SVGs the wordmark baseline is y=166, scaled by 1.9
# and translated down 10 units in a 500-unit-high viewBox.
_WORDMARK_BASELINE = (166 * 1.9 + 10) / 500
_RIGHT_EDGE = 0.985


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
        bounds = source.getbbox()
        mark = source.convert("RGBA").crop(bounds)
        baseline_in_crop = _WORDMARK_BASELINE * source.height - bounds[1]
    mark_height = 210
    baseline = round(baseline_in_crop * mark_height / mark.height)
    mark = mark.resize((round(mark.width * mark_height / mark.height), mark_height),
                       Image.Resampling.LANCZOS)
    gap = round(roman.getlength(" "))
    mark_x = round(roman.getlength(prefix)) + gap
    symbol_x = mark_x + mark.width + gap
    owner_x = symbol_x + round(symbol.getlength("©")) + gap
    image = Image.new("RGBA", (owner_x + round(roman.getlength(owner)) + 3, mark_height))
    image.alpha_composite(mark, (mark_x, 0))
    draw = ImageDraw.Draw(image)
    for x, text, font in ((0, prefix, roman), (symbol_x, "©", symbol), (owner_x, owner, roman)):
        draw.text((x, baseline), text, font=font, fill=ink, anchor="ls")
    return image.crop(image.getbbox())


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
        (_RIGHT_EDGE, 0.012), xycoords=figure.transFigure, box_alignment=(1, 0),
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
        pv.Texture(np.asarray(image)).to_image(), position=(_RIGHT_EDGE - size[0], 0.018), size=size,
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
    strip.paste(text, (round(_RIGHT_EDGE * strip.width) - text_width, padding), text)
    return np.asarray(strip)
