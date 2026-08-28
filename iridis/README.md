# Iridis flythrough scripts

Site-specific helpers for rendering long Fewview movies on Iridis (Southampton
HPC). These are **not** part of the installable package; they wrap the generic
`fewview-cluster-job` tool (see [`../docs/cluster_rendering.md`](../docs/cluster_rendering.md))
with Iridis partition/account/module details and a fixed camera shot.

## One-time setup on Iridis

1. Build the environment on a login node (not in a job):
   ```bash
   module load <your-conda-module>      # `module avail` to find it
   conda create -n fewview python=3.11
   conda activate fewview
   # install your FastEMRIWaveforms build, then:
   pip install fewview
   ```
2. Copy across the mode file written locally by `fewview-render`:
   ```bash
   rsync -av relativistic-modes.npz <user>@iridis:~/emri/
   ```

## Render the flythrough

Edit the four `EDIT THESE` values at the top of `render_flythrough.sh`
(`PARTITION`, `ACCOUNT`, `CONDA_MODULE`, `CONDA_ENV`), then:

```bash
bash render_flythrough.sh                    # uses ~/emri/relativistic-modes.npz
bash render_flythrough.sh path/to/modes.npz  # or point at another mode file
```

It generates and submits a 48-task Slurm array (2880 frames, 4K, a full 360°
rise-and-return loop with the transparent waveform panel) plus a dependent merge
job. The finished movie lands at `$JOBDIR/flythrough.mp4`.

## Doing several movies

Each distinct set of render/camera settings gets its own segment directory (the
job tool fingerprints them), so you can launch different shots into the same
`--job-dir` without them colliding. Copy `render_flythrough.sh` per shot and
change the camera block, or parameterise it further. Raise `MAX_CONCURRENT` to
cut wall-clock: total work is fixed, so wall time is roughly total ÷ concurrent.
