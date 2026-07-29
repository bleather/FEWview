#!/bin/bash
#
# Iridis (Southampton HPC) wrapper for a "rise-and-return" EMRI flythrough.
#
# This is a SITE-SPECIFIC convenience script. It is deliberately separate from
# the package: the package ships the generic `fewview-cluster-job` tool (and a
# generic docs/cluster_rendering.md), and this file just fills in the Iridis
# partition / account / module / env and the flythrough camera settings, then
# submits. Copy it to Iridis, edit the four site values, and run:
#
#     bash render_flythrough.sh [path/to/relativistic-modes.npz]
#
# It writes a Slurm array + a dependent merge job under $JOBDIR and submits both.
set -euo pipefail

# ---- Iridis / site settings (EDIT THESE) ---------------------------------
PARTITION="gpu"                    # GPU partition name; list with `sinfo -s`
ACCOUNT=""                         # Slurm account for the GPU partition ("" = your default)
CONDA_MODULE="conda/py3-latest"    # find yours with `module avail` (anaconda/miniforge/...)
CONDA_ENV="fewview"                # env with fewview + FastEMRIWaveforms installed
JOBDIR="$HOME/emri/flythrough-job"

# Mode file written by `fewview-render` (arg 1 overrides the default path).
MODES="${1:-$HOME/emri/relativistic-modes.npz}"

# ---- movie + camera settings (the flythrough) ----------------------------
# A full 360° loop is 3× the 120° shot, so 3× the frames keeps the same glide
# speed; CYCLES = 38*3 keeps the inspiral playing at its old pace too.
FRAMES=2880
FPS=20
CYCLES=114
RES=300
WIDTH=3840
HEIGHT=2160
SEGMENTS=48                        # ~60 frames per task
MAX_CONCURRENT=24                  # raise as high as your allocation allows

# ---- assemble optional directives ----------------------------------------
ACCOUNT_ARG=()
[ -n "$ACCOUNT" ] && ACCOUNT_ARG=(--account "$ACCOUNT")

if [ ! -f "$MODES" ]; then
    echo "Mode file not found: $MODES" >&2
    echo "Generate it once with 'fewview-render ...' and copy it across." >&2
    exit 1
fi

# If Iridis reports a software renderer instead of the GPU in the job log,
# swap '--headless-backend egl' for '--headless-backend auto' (xvfb fallback).
# For a preemptible partition, add '--preemptible' so requeued tasks resume.
fewview-cluster-job "$MODES" \
    --job-dir "$JOBDIR" \
    --partition "$PARTITION" "${ACCOUNT_ARG[@]}" \
    --conda-env "$CONDA_ENV" --module "$CONDA_MODULE" \
    --headless-backend egl \
    --segments "$SEGMENTS" --max-concurrent "$MAX_CONCURRENT" \
    --time 01:00:00 --memory 24G --cpus-per-task 4 \
    --frames "$FRAMES" --fps "$FPS" --animation-cycles "$CYCLES" \
    --resolution "$RES" --width "$WIDTH" --height "$HEIGHT" \
    --component plus --opacity-profile shells --color-scheme cool \
    --presentation shells_dramatic \
    --camera-latitude 10 --camera-longitude 0 --camera-latitude-end 70 --camera-loop \
    --waveform-panel --waveform-transparent \
    --bodies --trajectory --trajectory-color '#00b7ff' \
    --output-name flythrough.mp4

echo
echo "Submitting the array + merge jobs..."
bash "$JOBDIR/submit.sh"
echo
echo "Watch with:  squeue -u \"\$USER\""
echo "Final movie: $JOBDIR/flythrough.mp4"
