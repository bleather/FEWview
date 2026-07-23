#!/usr/bin/env python3
"""Animate a saved relativistic FEW mode file as an MP4 or GIF."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from fewview import (
    RelativisticModeWaveform,
    choose_max_delay,
    estimate_waveform_period,
    render_mode_animation,
)


logger = logging.getLogger("fewview.cli.animate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load relativistic-modes.npz from the FEW visualization example "
            "and render a genuinely time-evolving full-sky volume."
        )
    )
    parser.add_argument(
        "modes_file",
        type=Path,
        help="relativistic-modes.npz produced by visualize_emri_waveform.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .mp4 or .gif; defaults beside the modes file",
    )
    parser.add_argument(
        "--component",
        choices=("plus", "cross", "amplitude", "energy_flux"),
        default="plus",
    )
    parser.add_argument(
        "--flux-mode-combination",
        choices=("coherent", "incoherent"),
        default="coherent",
        help=(
            "coherent keeps instantaneous modal interference; incoherent "
            "is a smoother phase-averaged energy-flux proxy"
        ),
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--wave-cycles", type=float, default=5.0)
    parser.add_argument(
        "--animation-cycles",
        type=float,
        default=8.0,
        help="physical wave cycles traversed during the movie",
    )
    parser.add_argument("--max-delay", type=float, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument(
        "--normalization-time",
        type=float,
        default=None,
        help="optional single time used for colour scaling",
    )
    parser.add_argument(
        "--normalization-samples",
        type=int,
        default=9,
        help="global frames sampled for stable colour scaling",
    )
    parser.add_argument("--grid-resolution", type=int, default=72)
    parser.add_argument(
        "--angular-sampling",
        choices=("cartesian", "spherical"),
        default="cartesian",
        help=(
            "direct Cartesian harmonic evaluation removes the phi seam; "
            "spherical uses less memory but interpolates an angular grid"
        ),
    )
    parser.add_argument("--polar-samples", type=int, default=48)
    parser.add_argument("--azimuthal-samples", type=int, default=96)
    parser.add_argument("--opacity", type=float, default=0.12)
    parser.add_argument("--shell-count", type=int, default=7)
    parser.add_argument("--shell-min", type=float, default=0.10)
    parser.add_argument("--shell-max", type=float, default=0.92)
    parser.add_argument("--shell-width", type=float, default=0.075)
    parser.add_argument("--shell-opacity-floor", type=float, default=0.16)
    parser.add_argument("--shell-glow", type=float, default=0.12)
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.65,
        help="display-only Cartesian Gaussian smoothing in voxel units",
    )
    parser.add_argument(
        "--opacity-unit-distance",
        type=float,
        default=None,
        help="physical opacity accumulation length; default is 0.04 radius",
    )
    parser.add_argument(
        "--opacity-profile",
        choices=("soft", "bands", "shells", "flux"),
        default="soft",
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
        default="aurora",
    )
    parser.add_argument(
        "--presentation",
        choices=("balanced", "dramatic", "shells_dramatic"),
        default="dramatic",
    )
    parser.add_argument("--color-exposure", type=float, default=None)
    parser.add_argument("--background-color", default=None)
    parser.add_argument(
        "--edge-taper",
        type=float,
        default=0.05,
        help="fractional outer-radius taper used to smooth the voxel boundary",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--image-scale", type=int, default=1)
    parser.add_argument(
        "--camera-view",
        choices=("oblique", "face_on"),
        default="oblique",
        help="face_on projects the signed-strain chart axis onto the source",
    )
    parser.add_argument("--camera-zoom", type=float, default=None)
    parser.add_argument(
        "--camera-orbit",
        type=float,
        default=0.0,
        help="optional total camera rotation in degrees",
    )
    parser.add_argument(
        "--starfield", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--star-count", type=int, default=None)
    parser.add_argument(
        "--bodies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the massive primary and orbiting compact secondary",
    )
    parser.add_argument(
        "--trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show a trailing segment of the FEW inspiral trajectory",
    )
    parser.add_argument(
        "--waveform-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add a synchronized h-plus strip below the spherical volume",
    )
    parser.add_argument(
        "--trajectory-tail-cycles",
        type=float,
        default=2.0,
        help="number of recent azimuthal orbits drawn behind the secondary",
    )
    parser.add_argument(
        "--trajectory-line-width",
        type=float,
        default=1.6,
        help="trajectory width in display pixels",
    )
    parser.add_argument(
        "--trajectory-color",
        default="#ffd36a",
        help="inspiral trail colour; any Matplotlib colour spec",
    )
    parser.add_argument(
        "--trajectory-tube",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render the orbit trail as a 3D tube instead of a thin line",
    )
    parser.add_argument(
        "--orbit-size",
        type=float,
        default=0.16,
        help="display orbit radius as a fraction of the wave sphere radius",
    )
    parser.add_argument(
        "--body-exaggeration",
        type=float,
        default=2.0,
        help="primary radius in horizon radii r+; the orbit sets the scale",
    )
    parser.add_argument(
        "--waveform-height",
        type=float,
        default=0.22,
        help="waveform panel height as a fraction of the output frame",
    )
    parser.add_argument(
        "--waveform-font-style",
        choices=("latex", "sans"),
        default="latex",
        help="use bundled LaTeX-like STIX fonts or the former sans-serif style",
    )
    parser.add_argument(
        "--waveform-style-file",
        type=Path,
        default=None,
        help="optional Matplotlib .mplstyle file for the waveform panel",
    )
    return parser


def _optional_array(data, name: str):
    return np.asarray(data[name]) if name in data.files else None


def _optional_float(data, name: str):
    return float(data[name]) if name in data.files else None


def load_mode_waveform(filename: Path) -> tuple[RelativisticModeWaveform, np.ndarray]:
    data = np.load(filename)
    waveform = RelativisticModeWaveform(
        time=np.asarray(data["time"]),
        modes=np.asarray(data["modes"]),
        ell=np.asarray(data["ell"]),
        m=np.asarray(data["m"]),
        teukolsky_modes_used=int(data.get("teukolsky_modes_used", 0)),
        teukolsky_modes_available=int(data.get("teukolsky_modes_available", 0)),
        retained_power_fraction=float(data.get("retained_power_fraction", 1.0)),
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
    total_duration = float(waveform.time[-1] - waveform.time[0])
    if args.max_delay is None and max_delay >= 0.8 * total_duration:
        max_delay = 0.6 * total_duration
        logger.warning(
            "The saved waveform is short, so the radial history was reduced "
            "to %.1f s to leave time for animation. Generate with a larger "
            "--T for a longer movie.",
            max_delay,
        )
    end_time = float(waveform.time[-1] if args.end_time is None else args.end_time)
    earliest = float(waveform.time[0] + max_delay)
    start_time = (
        max(earliest, end_time - args.animation_cycles * period)
        if args.start_time is None
        else args.start_time
    )
    output = (
        args.modes_file.with_name("relativistic-emri-animation.mp4")
        if args.output is None
        else args.output
    )
    logger.info(
        "Rendering %.1f s of waveform evolution to %s",
        end_time - start_time,
        output,
    )
    render_mode_animation(
        waveform,
        output,
        max_delay=max_delay,
        component=args.component,
        flux_mode_combination=args.flux_mode_combination,
        start_time=start_time,
        end_time=end_time,
        frames=args.frames,
        fps=args.fps,
        resolution=args.grid_resolution,
        angular_sampling=args.angular_sampling,
        polar_samples=args.polar_samples,
        azimuthal_samples=args.azimuthal_samples,
        outer_window_fraction=args.edge_taper,
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
        trajectory_as_tube=args.trajectory_tube,
        orbit_display_radius=args.orbit_size,
        body_exaggeration=args.body_exaggeration,
        waveform_fraction=args.waveform_height,
        waveform_font_style=args.waveform_font_style,
        waveform_style_file=args.waveform_style_file,
        normalization_time=args.normalization_time,
        normalization_samples=args.normalization_samples,
    )
    logger.info("Saved %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
