"""Command-line entry points for Fewview."""

# Segments from older layouts must be rendered again before they can be resumed.
SEGMENT_RENDER_VERSION = "corner-logo-v1"


def add_logo_argument(parser):
    parser.add_argument(
        "--logo", choices=("wordmark", "icon"), default="wordmark",
        help="place the FEWview wordmark or orbit icon in the lower-right corner",
    )
