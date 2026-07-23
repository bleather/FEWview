#!/usr/bin/env python3
"""Create an Iridis X Slurm array workflow for an EMRI animation."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import stat
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Partition:
    gpus_per_node: int
    default_concurrency: int
    restricted: bool = False
    preemptible: bool = False


PARTITIONS = {
    "a100": Partition(gpus_per_node=2, default_concurrency=8),
    "swarm_a100": Partition(
        gpus_per_node=4,
        default_concurrency=4,
        restricted=True,
    ),
    "swarm_h100": Partition(
        gpus_per_node=8,
        default_concurrency=8,
        restricted=True,
    ),
    "scavenger_4a100": Partition(
        gpus_per_node=4,
        default_concurrency=8,
        preemptible=True,
    ),
    "scavenger_8h100": Partition(
        gpus_per_node=8,
        default_concurrency=8,
        preemptible=True,
    ),
}


DEFAULT_FRAMES = 1000


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resolve(project_dir: Path, value: Path) -> Path:
    return (project_dir / value).resolve() if not value.is_absolute() else value


def _module_lines(modules: list[str]) -> str:
    if not modules:
        return ""
    return "\n".join(f"module load {shlex.quote(module)}" for module in modules)


def _activate_lines(venv: Path | None, conda_env: str | None) -> str:
    if conda_env is not None:
        # `conda activate` accepts either an environment name or a full prefix
        # path. In a non-interactive Slurm shell conda has usually not been
        # initialised, so source its hook first; `conda info --base` resolves
        # the install even when only the `conda` shell function is on PATH via
        # a loaded module.
        env = shlex.quote(conda_env)
        return textwrap.dedent(
            f"""
            if ! command -v conda >/dev/null 2>&1; then
                echo "conda not found; load the conda module first with --module" >&2
                exit 2
            fi
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate {env}
            """
        ).strip()
    activate = venv / "bin" / "activate"
    return textwrap.dedent(
        f"""
        if [[ ! -f {shlex.quote(str(activate))} ]]; then
            echo "Missing Python environment: {activate}" >&2
            exit 2
        fi
        source {shlex.quote(str(activate))}
        """
    ).strip()


def _optional_directive(flag: str, value: str | None) -> list[str]:
    return [f"#SBATCH --{flag}={value}"] if value else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modes_file",
        type=Path,
        help="relativistic-modes.npz on the cluster filesystem",
    )
    parser.add_argument(
        "--partition",
        choices=tuple(PARTITIONS),
        default="a100",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="working directory for the job (outputs and logs go here)",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=Path("iridisx-emri-job"),
    )
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv-iridisx"),
        help="virtualenv to source (default); ignored when --conda-env is set",
    )
    environment.add_argument(
        "--conda-env",
        default=None,
        help="conda environment name or prefix path to activate instead of a venv",
    )
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument(
        "--account",
        default=None,
        help="Slurm account authorized for the selected GPU partition",
    )
    parser.add_argument(
        "--merge-account",
        default=None,
        help="optional separate account for the CPU merge job",
    )
    parser.add_argument(
        "--merge-partition",
        default=None,
        help="optional CPU partition for the merge job",
    )
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--job-name", default="emri-render")
    parser.add_argument("--segments", type=_positive, default=10)
    parser.add_argument("--max-concurrent", type=_positive, default=None)
    parser.add_argument("--cpus-per-task", type=_positive, default=8)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument(
        "--headless-backend",
        choices=("egl", "auto", "xvfb"),
        default="egl",
    )

    render = parser.add_argument_group("render settings")
    length = render.add_mutually_exclusive_group()
    length.add_argument(
        "--frames",
        type=_positive,
        default=None,
        help=f"total rendered frames (default {DEFAULT_FRAMES})",
    )
    length.add_argument(
        "--duration",
        type=_positive_float,
        default=None,
        help=(
            "movie playback length in seconds; sets --frames to duration x fps. "
            "This is how long the movie runs, not how much inspiral it covers, "
            "which is --animation-cycles"
        ),
    )
    render.add_argument("--fps", type=_positive, default=30)
    render.add_argument(
        "--component",
        choices=("plus", "cross", "amplitude", "energy_flux"),
        default="plus",
    )
    render.add_argument(
        "--flux-mode-combination",
        choices=("coherent", "incoherent"),
        default="coherent",
    )
    render.add_argument("--wave-cycles", type=float, default=1.5)
    render.add_argument("--animation-cycles", type=float, default=8.0)
    render.add_argument("--max-delay", type=float, default=None)
    render.add_argument("--start-time", type=float, default=None)
    render.add_argument("--end-time", type=float, default=None)
    render.add_argument("--normalization-time", type=float, default=None)
    render.add_argument("--normalization-samples", type=_positive, default=9)
    render.add_argument("--resolution", type=_positive, default=300)
    render.add_argument(
        "--angular-sampling",
        choices=("cartesian", "spherical"),
        default="cartesian",
    )
    render.add_argument("--polar-samples", type=_positive, default=200)
    render.add_argument("--azimuthal-samples", type=_positive, default=200)
    render.add_argument("--inner-window-fraction", type=float, default=0.10)
    render.add_argument("--outer-window-fraction", type=float, default=0.10)
    render.add_argument(
        "--opacity-profile",
        choices=("soft", "bands", "shells", "flux"),
        default="shells",
    )
    render.add_argument(
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
    render.add_argument(
        "--presentation",
        choices=("balanced", "dramatic", "shells_dramatic"),
        default="shells_dramatic",
    )
    render.add_argument("--color-exposure", type=float, default=None)
    render.add_argument("--background-color", default=None)
    render.add_argument("--opacity", type=float, default=0.30)
    render.add_argument("--shell-count", type=_positive, default=7)
    render.add_argument("--shell-min", type=float, default=0.10)
    render.add_argument("--shell-max", type=float, default=0.92)
    render.add_argument("--shell-width", type=float, default=0.075)
    render.add_argument("--shell-opacity-floor", type=float, default=0.16)
    render.add_argument("--shell-glow", type=float, default=0.12)
    render.add_argument("--smooth-sigma", type=float, default=1.20)
    render.add_argument("--opacity-unit-distance", type=float, default=None)
    render.add_argument("--width", type=_positive, default=1280)
    render.add_argument("--height", type=_positive, default=720)
    render.add_argument("--image-scale", type=_positive, default=1)
    render.add_argument(
        "--camera-view",
        choices=("oblique", "face_on"),
        default="oblique",
    )
    render.add_argument("--camera-zoom", type=float, default=0.95)
    render.add_argument("--camera-orbit", type=float, default=0.0)
    render.add_argument(
        "--starfield", action=argparse.BooleanOptionalAction, default=None
    )
    render.add_argument("--star-count", type=_positive, default=None)
    render.add_argument(
        "--bodies", action=argparse.BooleanOptionalAction, default=True
    )
    render.add_argument(
        "--trajectory", action=argparse.BooleanOptionalAction, default=True
    )
    render.add_argument(
        "--waveform-panel", action=argparse.BooleanOptionalAction, default=True
    )
    render.add_argument("--trajectory-tail-cycles", type=float, default=2.0)
    render.add_argument("--trajectory-line-width", type=float, default=1.6)
    render.add_argument("--trajectory-color", default="#00b7ff")
    render.add_argument("--body-exaggeration", type=float, default=2.0)
    render.add_argument(
        "--trajectory-tube",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    render.add_argument("--orbit-display-radius", type=float, default=0.25)
    render.add_argument("--waveform-fraction", type=float, default=0.22)
    render.add_argument(
        "--waveform-font-style",
        choices=("latex", "sans"),
        default="latex",
    )
    render.add_argument("--waveform-style-file", type=Path, default=None)
    render.add_argument(
        "--output-name",
        default="relativistic-emri-animation.mp4",
    )
    return parser


def _worker_arguments(
    args: argparse.Namespace, modes: Path, segments_dir: Path
) -> list[str]:
    values = [
        str(modes),
        "--output-dir",
        str(segments_dir),
        "--segments",
        str(args.segments),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--component",
        args.component,
        "--flux-mode-combination",
        args.flux_mode_combination,
        "--wave-cycles",
        str(args.wave_cycles),
        "--animation-cycles",
        str(args.animation_cycles),
        "--resolution",
        str(args.resolution),
        "--angular-sampling",
        args.angular_sampling,
        "--polar-samples",
        str(args.polar_samples),
        "--azimuthal-samples",
        str(args.azimuthal_samples),
        "--inner-window-fraction",
        str(args.inner_window_fraction),
        "--outer-window-fraction",
        str(args.outer_window_fraction),
        "--opacity-profile",
        args.opacity_profile,
        "--color-scheme",
        args.color_scheme,
        "--presentation",
        args.presentation,
        "--opacity",
        str(args.opacity),
        "--shell-count",
        str(args.shell_count),
        "--shell-min",
        str(args.shell_min),
        "--shell-max",
        str(args.shell_max),
        "--shell-width",
        str(args.shell_width),
        "--shell-opacity-floor",
        str(args.shell_opacity_floor),
        "--shell-glow",
        str(args.shell_glow),
        "--smooth-sigma",
        str(args.smooth_sigma),
        "--normalization-samples",
        str(args.normalization_samples),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--image-scale",
        str(args.image_scale),
        "--camera-view",
        args.camera_view,
        "--camera-orbit",
        str(args.camera_orbit),
        "--trajectory-tail-cycles",
        str(args.trajectory_tail_cycles),
        "--trajectory-line-width",
        str(args.trajectory_line_width),
        "--trajectory-color",
        str(args.trajectory_color),
        "--body-exaggeration",
        str(args.body_exaggeration),
        "--orbit-display-radius",
        str(args.orbit_display_radius),
        "--waveform-fraction",
        str(args.waveform_fraction),
        "--waveform-font-style",
        args.waveform_font_style,
        "--bodies" if args.bodies else "--no-bodies",
        "--trajectory" if args.trajectory else "--no-trajectory",
        "--trajectory-tube" if args.trajectory_tube else "--no-trajectory-tube",
        "--waveform-panel" if args.waveform_panel else "--no-waveform-panel",
    ]
    if args.starfield is not None:
        values.append("--starfield" if args.starfield else "--no-starfield")
    if args.waveform_style_file is not None:
        values.extend(("--waveform-style-file", str(args.waveform_style_file)))
    for name in (
        "max_delay",
        "start_time",
        "end_time",
        "normalization_time",
        "opacity_unit_distance",
        "color_exposure",
        "background_color",
        "camera_zoom",
        "star_count",
    ):
        value = getattr(args, name)
        if value is not None:
            values.extend((f"--{name.replace('_', '-')}", str(value)))
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolve the movie length down to a frame count before it reaches the run
    # fingerprint and the worker, so --duration 60 and the --frames it implies
    # describe the same render and reuse the same segment directory.
    if args.duration is not None:
        args.frames = round(args.duration * args.fps)
    elif args.frames is None:
        args.frames = DEFAULT_FRAMES
    project = args.project_dir.resolve()
    modes = _resolve(project, args.modes_file)
    job_dir = _resolve(project, args.job_dir)
    # A conda env is activated by name/prefix, so there is no venv path to
    # resolve or require on disk.
    venv = None if args.conda_env else _resolve(project, args.venv)
    if args.waveform_style_file is not None:
        args.waveform_style_file = _resolve(project, args.waveform_style_file)
    # The render and merge steps are Fewview console scripts, resolved from the
    # activated environment's PATH rather than from a source checkout.
    worker = "fewview-cluster-segment"
    merger = "fewview-cluster-merge"
    required_paths = [modes]
    if args.waveform_style_file is not None:
        required_paths.append(args.waveform_style_file)
    for required in required_paths:
        if not required.exists():
            raise SystemExit(f"Required path does not exist: {required}")
    if args.frames < 2 * args.segments:
        raise SystemExit(
            f"Use at least two frames per segment: {args.frames} frames at "
            f"{args.fps} fps do not cover {args.segments} segments"
        )
    if args.width % 2 or args.height % 2:
        raise SystemExit("MP4 width and height must both be even")

    partition = PARTITIONS[args.partition]
    concurrency = min(
        args.segments,
        args.max_concurrent or partition.default_concurrency,
    )
    render_keys = (
        "segments",
        "frames",
        "fps",
        "component",
        "flux_mode_combination",
        "wave_cycles",
        "animation_cycles",
        "max_delay",
        "start_time",
        "end_time",
        "normalization_time",
        "normalization_samples",
        "resolution",
        "angular_sampling",
        "polar_samples",
        "azimuthal_samples",
        "inner_window_fraction",
        "outer_window_fraction",
        "opacity_profile",
        "color_scheme",
        "presentation",
        "color_exposure",
        "background_color",
        "opacity",
        "shell_count",
        "shell_min",
        "shell_max",
        "shell_width",
        "shell_opacity_floor",
        "shell_glow",
        "smooth_sigma",
        "opacity_unit_distance",
        "width",
        "height",
        "image_scale",
        "camera_view",
        "camera_zoom",
        "camera_orbit",
        "starfield",
        "star_count",
        "bodies",
        "trajectory",
        "waveform_panel",
        "trajectory_tail_cycles",
        "trajectory_line_width",
        "trajectory_color",
        "body_exaggeration",
        "trajectory_tube",
        "orbit_display_radius",
        "waveform_fraction",
        "waveform_font_style",
        "waveform_style_file",
    )
    mode_stat = modes.stat()
    fingerprint_source = repr(
        (
            str(modes),
            mode_stat.st_size,
            mode_stat.st_mtime_ns,
            tuple((key, getattr(args, key)) for key in render_keys),
        )
    )
    run_id = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    logs_dir = job_dir / "logs"
    segments_dir = job_dir / "segments" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    output = job_dir / args.output_name

    module_lines = _module_lines(args.module)
    activate_lines = _activate_lines(venv, args.conda_env)
    render_account = _optional_directive("account", args.account)
    merge_account = _optional_directive("account", args.merge_account)
    mail = (
        [
            f"#SBATCH --mail-user={args.mail_user}",
            "#SBATCH --mail-type=END,FAIL",
        ]
        if args.mail_user
        else []
    )
    requeue = ["#SBATCH --requeue"] if partition.preemptible else []
    backend = ""
    runner_prefix = ""
    if args.headless_backend == "egl":
        backend = "export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow"
    elif args.headless_backend == "xvfb":
        runner_prefix = "xvfb-run -a "

    worker_values = _worker_arguments(args, modes, segments_dir)
    worker_command = (
        f"srun --ntasks=1 {runner_prefix}{shlex.quote(worker)} "
        f"{shlex.join(worker_values)} "
        '--segment-index "${SLURM_ARRAY_TASK_ID}"'
    )
    render_directives = [
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --array=0-{args.segments - 1}%{concurrency}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.memory}",
        "#SBATCH --gres=gpu:1",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={logs_dir}/render_%A_%a.out",
        f"#SBATCH --error={logs_dir}/render_%A_%a.err",
        *render_account,
        *mail,
        *requeue,
    ]
    render_body = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(project))}",
        module_lines,
        activate_lines,
        "",
        "export PYVISTA_OFF_SCREEN=true",
        backend,
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
        'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
        'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
        'export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
        "",
        'echo "Node: ${HOSTNAME}"',
        'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        "nvidia-smi -L",
        "python -c 'import pyvista as pv; print(pv.Report(gpu=True))'",
        worker_command,
    ]
    render_script = "\n".join(
        ["#!/bin/bash", *render_directives, "", *render_body, ""]
    )

    merge_command = shlex.join(
        [
            merger,
            str(segments_dir),
            str(output),
            "--segments",
            str(args.segments),
        ]
    )
    merge_directives = [
        f"#SBATCH --job-name={args.job_name}-merge",
        *_optional_directive("partition", args.merge_partition),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=4G",
        "#SBATCH --time=00:30:00",
        f"#SBATCH --output={logs_dir}/merge_%j.out",
        f"#SBATCH --error={logs_dir}/merge_%j.err",
        *merge_account,
        *mail,
    ]
    merge_body = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(project))}",
        module_lines,
        activate_lines,
        merge_command,
    ]
    merge_script = "\n".join(
        ["#!/bin/bash", *merge_directives, "", *merge_body, ""]
    )

    render_path = job_dir / "render_array.slurm"
    merge_path = job_dir / "merge.slurm"
    submit_path = job_dir / "submit.sh"
    render_path.write_text(render_script, encoding="utf-8")
    merge_path.write_text(merge_script, encoding="utf-8")
    merge_submit = (
        'merge_id=$(sbatch --parsable --dependency="afterok:${render_id}" '
        f"{shlex.quote(str(merge_path))})"
    )
    submit_path.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail
            render_id=$(sbatch --parsable {shlex.quote(str(render_path))})
            render_id=${{render_id%%;*}}
            {merge_submit}
            merge_id=${{merge_id%%;*}}
            echo "Render array job: ${{render_id}}"
            echo "Merge job: ${{merge_id}}"
            echo "Final movie: {output}"
            """
        ),
        encoding="utf-8",
    )
    submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)

    summary = [
        f"Created {render_path}",
        f"Created {merge_path}",
        f"Created {submit_path}",
        f"Partition: {args.partition} ({partition.gpus_per_node} GPUs/node)",
        f"Array: {args.segments} segments, up to {concurrency} concurrent",
        f"Movie: {args.frames} frames at {args.fps} fps "
        f"({args.frames / args.fps:.1f} s)",
    ]
    if partition.restricted:
        summary.append(
            "Note: this partition is restricted to eligible ECS/ORC staff and PGRs."
        )
    if partition.preemptible:
        summary.append(
            "Note: this scavenger partition may preempt and requeue render tasks."
        )
    summary.append(f"Submit with: {submit_path}")
    sys.stdout.write("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
