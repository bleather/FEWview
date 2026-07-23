#!/usr/bin/env python3
"""Losslessly concatenate completed EMRI animation segments with FFmpeg."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segments", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.segments < 1:
        raise SystemExit("--segments must be positive")

    segment_files = [
        args.segments_dir / f"segment_{index:04d}.mp4"
        for index in range(args.segments)
    ]
    marker_files = [
        args.segments_dir / f"segment_{index:04d}.done"
        for index in range(args.segments)
    ]
    missing = [
        str(path)
        for path in (*segment_files, *marker_files)
        if not path.is_file()
    ]
    if missing:
        formatted = "\n  ".join(missing)
        raise SystemExit(f"Cannot merge; files are missing:\n  {formatted}")

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit(
            "imageio-ffmpeg is required; install the visualization extra"
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.segments_dir / "segments.ffconcat"
    manifest.write_text(
        "ffconcat version 1.0\n"
        + "".join(
            f"file '{_ffmpeg_concat_path(path)}'\n" for path in segment_files
        ),
        encoding="utf-8",
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    sys.stdout.write(f"{args.output.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
