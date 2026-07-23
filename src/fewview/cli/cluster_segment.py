#!/usr/bin/env python3
"""Render one globally consistent segment of an EMRI animation.

This worker is intended to be launched as a Slurm array job.  Every array task
renders a disjoint group of the global frame times, while sharing the same
colour normalization, waveform-panel limits, and camera phase.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from fewview import (
    RelativisticModeWaveform,
    choose_max_delay,
    estimate_waveform_period,
    render_mode_animation,
)


LOGGER = logging.getLogger("fewview.cli.cluster_segment")


def _optional_array(data: np.lib.npyio.NpzFile, name: str):
    return np.asarray(data[name]) if name in data.files else None


def _optional_float(data: np.lib.npyio.NpzFile, name: str):
    return float(data[name]) if name in data.files else None


def load_mode_waveform(filename: Path) -> tuple[RelativisticModeWaveform, np.ndarray]:
    """Load the portable mode archive written by visualize_emri_waveform.py."""

    with np.load(filename, allow_pickle=False) as data:
        waveform = RelativisticModeWaveform(
            time=np.asarray(data["time"]),
            modes=np.asarray(data["modes"]),
            ell=np.asarray(data["ell"]),
            m=np.asarray(data["m"]),
            teukolsky_modes_used=int(data.get("teukolsky_modes_used", 0)),
            teukolsky_modes_available=int(
                data.get("teukolsky_modes_available", 0)
            ),
            retained_power_fraction=float(
                data.get("retained_power_fraction", 1.0)
            ),
            trajectory_p=_optional_array(data, "trajectory_p"),
            trajectory_e=_optional_array(data, "trajectory_e"),
            trajectory_xI=_optional_array(data, "trajectory_xI"),
            trajectory_phi_phi=_optional_array(data, "trajectory_phi_phi"),
            trajectory_phi_r=_optional_array(data, "trajectory_phi_r"),
            primary_mass=_optional_float(data, "primary_mass"),
            secondary_mass=_optional_float(data, "secondary_mass"),
            spin=_optional_float(data, "spin"),
        )
        if "h_plus_reference" in data and "h_cross_reference" in data:
            reference = np.asarray(data["h_plus_reference"]) - 1j * np.asarray(
                data["h_cross_reference"]
            )
        else:
            reference = waveform.strain(theta=np.pi / 3.0, phi=0.0)
    return waveform, reference


def segment_bounds(
    total_frames: int, segment_count: int, segment_index: int
) -> tuple[int, int]:
    """Return the inclusive global frame bounds for one balanced segment."""

    if total_frames < 2:
        raise ValueError("total_frames must be at least 2")
    if segment_count < 1 or segment_count > total_frames // 2:
        raise ValueError(
            "segment_count must be positive and leave at least two frames per segment"
        )
    if not 0 <= segment_index < segment_count:
        raise ValueError("segment_index is outside the configured job array")
    base, remainder = divmod(total_frames, segment_count)
    count = base + int(segment_index < remainder)
    first = segment_index * base + min(segment_index, remainder)
    return first, first + count - 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help="defaults to SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--component",
        choices=("plus", "cross", "amplitude", "energy_flux"),
        default="plus",
    )
    parser.add_argument(
        "--flux-mode-combination",
        choices=("coherent", "incoherent"),
        default="coherent",
    )
    parser.add_argument("--wave-cycles", type=float, default=1.5)
    parser.add_argument("--animation-cycles", type=float, default=8.0)
    parser.add_argument("--max-delay", type=float, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--normalization-time", type=float, default=None)
    parser.add_argument("--normalization-samples", type=int, default=9)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument(
        "--angular-sampling",
        choices=("cartesian", "spherical"),
        default="cartesian",
    )
    parser.add_argument("--polar-samples", type=int, default=200)
    parser.add_argument("--azimuthal-samples", type=int, default=200)
    parser.add_argument("--inner-window-fraction", type=float, default=0.10)
    parser.add_argument("--outer-window-fraction", type=float, default=0.10)
    parser.add_argument(
        "--opacity-profile",
        choices=("soft", "bands", "shells", "flux"),
        default="shells",
    )
    parser.add_argument(
        "--color-scheme",
        choices=(
            "magma",
            "aurora",
            "rainbow",
            "inferno",
            "plasma",
            "viridis",
            "cividis",
            "cool",
            "blues",
            "cinematic",
        ),
        default="rainbow",
    )
    parser.add_argument(
        "--presentation",
        choices=("balanced", "dramatic", "shells_dramatic"),
        default="shells_dramatic",
    )
    parser.add_argument("--color-exposure", type=float, default=None)
    parser.add_argument("--background-color", default=None)
    parser.add_argument("--opacity", type=float, default=0.30)
    parser.add_argument("--shell-count", type=int, default=7)
    parser.add_argument("--shell-min", type=float, default=0.10)
    parser.add_argument("--shell-max", type=float, default=0.92)
    parser.add_argument("--shell-width", type=float, default=0.075)
    parser.add_argument("--shell-opacity-floor", type=float, default=0.16)
    parser.add_argument("--shell-glow", type=float, default=0.12)
    parser.add_argument("--smooth-sigma", type=float, default=1.20)
    parser.add_argument("--opacity-unit-distance", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--image-scale", type=int, default=1)
    parser.add_argument(
        "--camera-view",
        choices=("oblique", "face_on"),
        default="oblique",
    )
    parser.add_argument("--camera-zoom", type=float, default=0.95)
    parser.add_argument("--camera-orbit", type=float, default=0.0)
    parser.add_argument(
        "--starfield", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--star-count", type=int, default=None)
    parser.add_argument(
        "--bodies", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trajectory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--waveform-panel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--trajectory-tail-cycles", type=float, default=2.0)
    parser.add_argument("--trajectory-line-width", type=float, default=1.6)
    parser.add_argument("--trajectory-color", default="#00b7ff")
    parser.add_argument("--body-exaggeration", type=float, default=2.0)
    parser.add_argument(
        "--trajectory-tube",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--orbit-display-radius", type=float, default=0.25)
    parser.add_argument("--waveform-fraction", type=float, default=0.22)
    parser.add_argument(
        "--waveform-font-style",
        choices=("latex", "sans"),
        default="latex",
    )
    parser.add_argument("--waveform-style-file", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerender a segment even when its completion marker exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    segment_index = args.segment_index
    if segment_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise SystemExit(
                "--segment-index is required outside a Slurm array job"
            )
        segment_index = int(value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"segment_{segment_index:04d}.mp4"
    complete = args.output_dir / f"segment_{segment_index:04d}.done"
    if complete.exists() and output.exists() and not args.force:
        LOGGER.info("Segment %d is already complete; skipping", segment_index)
        return 0

    waveform, reference = load_mode_waveform(args.modes_file)
    h_plus = np.real(reference)
    h_cross = -np.imag(reference)
    period = estimate_waveform_period(waveform.time, h_plus, h_cross)
    max_delay = (
        choose_max_delay(
            waveform.time,
            h_plus,
            h_cross,
            wave_cycles=args.wave_cycles,
        )
        if args.max_delay is None
        else args.max_delay
    )
    duration = float(waveform.time[-1] - waveform.time[0])
    if args.max_delay is None and max_delay >= 0.8 * duration:
        max_delay = 0.6 * duration

    global_end = float(
        waveform.time[-1] if args.end_time is None else args.end_time
    )
    earliest = float(waveform.time[0] + max_delay)
    global_start = float(
        max(earliest, global_end - args.animation_cycles * period)
        if args.start_time is None
        else args.start_time
    )
    all_times = np.linspace(global_start, global_end, args.frames)
    first, last = segment_bounds(args.frames, args.segments, segment_index)
    local_frames = last - first + 1
    LOGGER.info(
        "Rendering segment %d/%d: global frames %d-%d (%d frames)",
        segment_index + 1,
        args.segments,
        first,
        last,
        local_frames,
    )

    render_mode_animation(
        waveform,
        output,
        max_delay=max_delay,
        component=args.component,
        flux_mode_combination=args.flux_mode_combination,
        start_time=float(all_times[first]),
        end_time=float(all_times[last]),
        frames=local_frames,
        fps=args.fps,
        resolution=args.resolution,
        angular_sampling=args.angular_sampling,
        polar_samples=args.polar_samples,
        azimuthal_samples=args.azimuthal_samples,
        inner_window_fraction=args.inner_window_fraction,
        outer_window_fraction=args.outer_window_fraction,
        opacity_profile=args.opacity_profile,
        color_scheme=args.color_scheme,
        presentation=args.presentation,
        color_exposure=args.color_exposure,
        background_color=args.background_color,
        opacity=args.opacity,
        shell_count=args.shell_count,
        shell_min=args.shell_min,
        shell_max=args.shell_max,
        shell_width=args.shell_width,
        shell_opacity_floor=args.shell_opacity_floor,
        shell_glow=args.shell_glow,
        smooth_sigma=args.smooth_sigma,
        opacity_unit_distance=args.opacity_unit_distance,
        window_size=(args.width, args.height),
        image_scale=args.image_scale,
        camera_view=args.camera_view,
        camera_zoom=args.camera_zoom,
        camera_orbit_degrees=args.camera_orbit,
        starfield=args.starfield,
        star_count=args.star_count,
        show_bodies=args.bodies,
        show_trajectory=args.trajectory,
        show_waveform=args.waveform_panel,
        trajectory_tail_cycles=args.trajectory_tail_cycles,
        trajectory_line_width=args.trajectory_line_width,
        trajectory_color=args.trajectory_color,
        body_exaggeration=args.body_exaggeration,
        trajectory_as_tube=args.trajectory_tube,
        orbit_display_radius=args.orbit_display_radius,
        waveform_fraction=args.waveform_fraction,
        waveform_font_style=args.waveform_font_style,
        waveform_style_file=args.waveform_style_file,
        global_start_time=global_start,
        global_end_time=global_end,
        normalization_time=args.normalization_time,
        normalization_samples=args.normalization_samples,
    )
    complete.write_text("complete\n", encoding="utf-8")
    LOGGER.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
