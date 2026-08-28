#!/bin/bash
#
# Iridis (Southampton HPC) wrapper for an oblique "push-in" EMRI movie:
# a fixed oblique camera that zooms in over the course of the inspiral.
#
# Site-specific convenience wrapper around the generic `fewview-cluster-job`
# (see ../docs/cluster_rendering.md). Edit the four site values, then:
#
#     bash render_zoom_inspiral.sh [path/to/relativistic-modes.npz]
#
# Writes a Slurm array + dependent merge job under $JOBDIR and submits both.
set -euo pipefail

# ---- Iridis / site settings (EDIT THESE) ---------------------------------
PARTITION="a100"                   # a100 (immediate, ~serial) or scavenger_4a100 (parallel when free)
ACCOUNT="normal"                   # Slurm account ("" = your default)
CONDA_ENV="few_env"                # env with fewview + FastEMRIWaveforms
CONDA_MODULE=""                    # leave "" to use your own conda already on PATH
JOBDIR="$HOME/emri/zoom-inspiral-job"
MERGE_PARTITION=""                 # "" uses your default Slurm partition; set a non-amd CPU partition or merge manually

# Mode file written by `fewview-render` (arg 1 overrides the default path).
MODES="${1:-$HOME/emri/relativistic-modes.npz}"

# ---- movie + camera settings (the zoom-in) -------------------------------
FRAMES=1200
FPS=20
CYCLES=38                          # how much inspiral the movie covers
RES=300
WIDTH=3840
HEIGHT=2160
SEGMENTS=48
MAX_CONCURRENT=24

# Fixed oblique view; zoom pushes in from ZOOM_START to ZOOM_END over the movie.
CAM_LAT=22                         # oblique latitude above the equatorial plane
CAM_LON=25                         # oblique azimuth
ZOOM_START=0.9                     # wide at the start
ZOOM_END=2.2                       # magnified at merger (larger = closer-in)

# ---- assemble optional directives ----------------------------------------
ACCOUNT_ARG=(); [ -n "$ACCOUNT" ] && ACCOUNT_ARG=(--account "$ACCOUNT")
MODULE_ARG=();  [ -n "$CONDA_MODULE" ] && MODULE_ARG=(--module "$CONDA_MODULE")
MERGE_ARG=();   [ -n "$MERGE_PARTITION" ] && MERGE_ARG=(--merge-partition "$MERGE_PARTITION")

if [ ! -f "$MODES" ]; then
    echo "Mode file not found: $MODES" >&2
    echo "Generate it once with 'fewview-render --no-render-volume ...' and copy it across." >&2
    exit 1
fi

fewview-cluster-job "$MODES" \
    --job-dir "$JOBDIR" \
    --partition "$PARTITION" "${ACCOUNT_ARG[@]}" \
    --conda-env "$CONDA_ENV" "${MODULE_ARG[@]}" "${MERGE_ARG[@]}" \
    --headless-backend egl \
    --segments "$SEGMENTS" --max-concurrent "$MAX_CONCURRENT" \
    --time 01:00:00 --memory 24G --cpus-per-task 4 \
    --frames "$FRAMES" --fps "$FPS" --animation-cycles "$CYCLES" \
    --resolution "$RES" --width "$WIDTH" --height "$HEIGHT" \
    --component plus --opacity-profile shells --color-scheme cool \
    --presentation shells_dramatic \
    --camera-latitude "$CAM_LAT" --camera-longitude "$CAM_LON" \
    --camera-zoom "$ZOOM_START" --camera-zoom-end "$ZOOM_END" \
    --waveform-panel --waveform-transparent \
    --bodies --trajectory --trajectory-color '#00b7ff' \
    --output-name zoom-inspiral.mp4

echo
echo "Submitting the array + merge jobs..."
bash "$JOBDIR/submit.sh"
echo
echo "Watch:  squeue -u \"\$USER\""
echo "Movie:  $JOBDIR/zoom-inspiral.mp4"
# NOTE: the merge crashes on the 'amd' partition (bundled ffmpeg hits an illegal
# instruction there). If it fails, merge on the login node instead:
#   fewview-cluster-merge "$JOBDIR/segments/<run-id>" "$JOBDIR/zoom-inspiral.mp4" --segments $SEGMENTS
