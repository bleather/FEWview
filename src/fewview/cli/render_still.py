#!/usr/bin/env python3
"""Render a full-sky relativistic FEW waveform as a shells-like volume."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fewview import (
    build_mode_retarded_time_volume,
    choose_max_delay,
    estimate_waveform_period,
    generate_relativistic_mode_waveform,
    plot_volume_slice,
    render_volume,
    save_volume,
)


logger = logging.getLogger("fewview.cli.render_still")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the fully relativistic FastKerrEccentricEquatorialFlux "
            "modes, evaluate their spin-weighted full-sky field at retarded "
            "time u=t-r/c, and render a smooth spherical volume."
        )
    )
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("emri-visualization"))

    source = parser.add_argument_group("equatorial eccentric Kerr EMRI")
    source.add_argument("--M", type=float, default=1.0e6, help="primary mass [Msun]")
    source.add_argument("--mu", type=float, default=10.0, help="secondary mass [Msun]")
    source.add_argument("--a", type=float, default=0.9, help="dimensionless spin")
    source.add_argument("--p0", type=float, default=12.0)
    source.add_argument("--e0", type=float, default=0.4)
    source.add_argument("--xI0", type=float, default=1.0)
    source.add_argument("--distance", type=float, default=1.0, help="distance [Gpc]")
    source.add_argument("--Phi-phi0", dest="Phi_phi0", type=float, default=0.0)
    source.add_argument("--Phi-r0", dest="Phi_r0", type=float, default=0.0)

    modes = parser.add_argument_group("relativistic mode reconstruction")
    modes.add_argument("--lmax", type=int, default=10)
    modes.add_argument("--nmax", type=int, default=55)
    modes.add_argument(
        "--mode-power-fraction",
        type=float,
        default=1.0,
        help="sky-integrated Teukolsky-mode power retained; 1 uses every mode",
    )
    modes.add_argument(
        "--max-teukolsky-modes",
        type=int,
        default=None,
        help="optional speed cap after power ranking; default uses every mode",
    )

    display = parser.add_argument_group("sampling and rendering")
    display.add_argument("--dt", type=float, default=10.0, help="sample spacing [s]")
    display.add_argument("--T", type=float, default=0.001, help="duration [yr]")
    display.add_argument(
        "--max-delay",
        type=float,
        default=None,
        help="history mapped from centre to edge [s]",
    )
    display.add_argument(
        "--wave-cycles",
        type=float,
        default=5.0,
        help="local wave cycles mapped from centre to edge",
    )
    display.add_argument("--grid-resolution", type=int, default=128)
    display.add_argument("--polar-samples", type=int, default=80)
    display.add_argument("--azimuthal-samples", type=int, default=160)
    display.add_argument(
        "--raw-strain", action="store_true", help="do not normalize for display"
    )
    display.add_argument(
        "--export-vti",
        action="store_true",
        help="export a ParaView/PyVista .vti volume",
    )
    display.add_argument(
        "--render-volume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the PyVista volume (enabled by default)",
    )
    display.add_argument(
        "--render-component",
        choices=("plus", "cross", "amplitude", "energy_flux"),
        default="plus",
    )
    display.add_argument(
        "--flux-mode-combination",
        choices=("coherent", "incoherent"),
        default="coherent",
        help=(
            "coherent is instantaneous |sum dh_lm Y_lm|^2; incoherent "
            "drops modal cross terms for a smoother phase-averaged proxy"
        ),
    )
    display.add_argument(
        "--render-style",
        choices=("cinematic", "contours"),
        default="cinematic",
    )
    display.add_argument(
        "--opacity-profile",
        choices=("soft", "bands", "shells", "flux"),
        default="soft",
        help=(
            "soft uses broad fronts; bands uses symmetric scalar levels; "
            "shells exposes positive signed-strain bands; flux uses log flux"
        ),
    )
    display.add_argument(
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
        help="volume colour palette; aurora is the presentation default",
    )
    display.add_argument(
        "--presentation",
        choices=("balanced", "dramatic", "shells_dramatic"),
        default="dramatic",
    )
    display.add_argument("--color-exposure", type=float, default=None)
    display.add_argument("--background-color", default=None)
    display.add_argument(
        "--edge-taper",
        type=float,
        default=0.05,
        help="fractional outer-radius taper used to smooth the voxel boundary",
    )
    display.add_argument("--opacity", type=float, default=0.12)
    display.add_argument("--shell-count", type=int, default=7)
    display.add_argument("--shell-min", type=float, default=0.10)
    display.add_argument("--shell-max", type=float, default=0.92)
    display.add_argument("--shell-width", type=float, default=0.075)
    display.add_argument("--shell-opacity-floor", type=float, default=0.16)
    display.add_argument("--shell-glow", type=float, default=0.12)
    display.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.65,
        help="display-only Cartesian Gaussian smoothing in voxel units",
    )
    display.add_argument(
        "--opacity-unit-distance",
        type=float,
        default=None,
        help="physical opacity accumulation length; default is 0.04 radius",
    )
    display.add_argument("--image-width", type=int, default=1600)
    display.add_argument("--image-height", type=int, default=900)
    display.add_argument("--image-scale", type=int, default=1)
    display.add_argument(
        "--camera-view",
        choices=("oblique", "face_on"),
        default="oblique",
        help="face_on hides the signed-strain chart axis behind the source",
    )
    display.add_argument("--camera-zoom", type=float, default=None)
    display.add_argument(
        "--starfield", action=argparse.BooleanOptionalAction, default=None
    )
    display.add_argument("--star-count", type=int, default=None)
    display.add_argument(
        "--source-marker", action=argparse.BooleanOptionalAction, default=False
    )
    display.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full relativistic visualization workflow."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating FastKerrEccentricEquatorialFlux modes")
    waveform = generate_relativistic_mode_waveform(
        args.M,
        args.mu,
        args.a,
        args.p0,
        args.e0,
        args.xI0,
        dist=args.distance,
        Phi_phi0=args.Phi_phi0,
        Phi_r0=args.Phi_r0,
        dt=args.dt,
        T=args.T,
        lmax=args.lmax,
        nmax=args.nmax,
        power_fraction=args.mode_power_fraction,
        max_teukolsky_modes=args.max_teukolsky_modes,
        force_backend=args.backend,
    )
    logger.info(
        "Reconstructed %d (l,m) modes from %d/%d Teukolsky modes (%.6f power)",
        waveform.modes.shape[1],
        waveform.teukolsky_modes_used,
        waveform.teukolsky_modes_available,
        waveform.retained_power_fraction,
    )

    # This direction is only used to estimate a display period and save a
    # familiar 1D waveform. The volume itself evaluates every sky direction.
    reference_strain = waveform.strain(theta=np.pi / 3.0, phi=0.0)
    h_plus = np.real(reference_strain)
    h_cross = -np.imag(reference_strain)
    np.savez(
        args.output_dir / "relativistic-modes.npz",
        time=waveform.time,
        modes=waveform.modes,
        ell=waveform.ell,
        m=waveform.m,
        teukolsky_modes_used=waveform.teukolsky_modes_used,
        teukolsky_modes_available=waveform.teukolsky_modes_available,
        retained_power_fraction=waveform.retained_power_fraction,
        trajectory_p=waveform.trajectory_p,
        trajectory_e=waveform.trajectory_e,
        trajectory_xI=waveform.trajectory_xI,
        trajectory_phi_phi=waveform.trajectory_phi_phi,
        trajectory_phi_r=waveform.trajectory_phi_r,
        primary_mass=waveform.primary_mass,
        secondary_mass=waveform.secondary_mass,
        spin=waveform.spin,
        h_plus_reference=h_plus,
        h_cross_reference=h_cross,
    )

    period = estimate_waveform_period(waveform.time, h_plus, h_cross)
    max_delay = (
        choose_max_delay(
            waveform.time, h_plus, h_cross, wave_cycles=args.wave_cycles
        )
        if args.max_delay is None
        else args.max_delay
    )
    logger.info(
        "Estimated local period %.1f s; displaying %.1f s (%.1f cycles)",
        period,
        max_delay,
        max_delay / period,
    )

    logger.info("Evaluating the mode-resolved retarded-time sphere")
    volume = build_mode_retarded_time_volume(
        waveform,
        frame_time=waveform.time[-1],
        max_delay=max_delay,
        resolution=args.grid_resolution,
        polar_samples=args.polar_samples,
        azimuthal_samples=args.azimuthal_samples,
        inner_window_fraction=0.025,
        outer_window_fraction=args.edge_taper,
        flux_mode_combination=args.flux_mode_combination,
        normalize=not args.raw_strain,
    )

    slice_figure, _ = plot_volume_slice(volume, component="plus", plane="xz")
    slice_path = args.output_dir / "relativistic-volume-slice.png"
    slice_figure.savefig(slice_path, dpi=180, bbox_inches="tight")

    if args.export_vti:
        volume_path = save_volume(volume, args.output_dir / "relativistic-volume.vti")
        logger.info("Saved %s", volume_path)
    if args.render_volume:
        render_path = args.output_dir / "relativistic-emri-volume.png"
        plotter = render_volume(
            volume,
            component=args.render_component,
            style=args.render_style,
            opacity_profile=args.opacity_profile,
            color_scheme=args.color_scheme,
            presentation=args.presentation,
            color_exposure=args.color_exposure,
            background_color=args.background_color,
            screenshot=render_path,
            show=args.show,
            opacity=args.opacity,
            shell_count=args.shell_count,
            shell_min=args.shell_min,
            shell_max=args.shell_max,
            shell_width=args.shell_width,
            shell_opacity_floor=args.shell_opacity_floor,
            shell_glow=args.shell_glow,
            smooth_sigma=args.smooth_sigma,
            opacity_unit_distance=args.opacity_unit_distance,
            window_size=(args.image_width, args.image_height),
            image_scale=args.image_scale,
            camera_view=args.camera_view,
            camera_zoom=args.camera_zoom,
            starfield=args.starfield,
            star_count=args.star_count,
            source_marker=args.source_marker,
        )
        if not args.show:
            plotter.close()
        logger.info("Saved %s", render_path)

    logger.info("Saved %s", slice_path)
    if args.show:
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
