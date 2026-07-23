# Rendering an EMRI animation on a GPU cluster

For long or high-resolution movies the per-frame cost dominates, and the work is
serial within a task, so the fastest route is to split the movie across many
Slurm array tasks. `fewview-cluster-job` generates that workflow: it renders
independent, globally numbered movie segments as a Slurm array and submits a
dependent CPU job to concatenate them. Each render task requests one GPU. The
segment renderer shares waveform-panel limits, camera timing, and colour
normalization so the joined movie is continuous.

This guide assumes a generic Slurm cluster with GPU nodes. Substitute your
cluster's own partition name, modules, and account wherever a placeholder
appears.

## What you need on the cluster

1. **Fewview installed in an environment on the cluster**, alongside your
   FastEMRIWaveforms install (see below).
2. **The `relativistic-modes.npz` file** written by `fewview-render` (or the
   `visualize_emri_waveform` workflow). It already contains the waveform modes
   and trajectory, so nothing else needs to be transferred.

Copy the mode file across, for example:

```bash
rsync -av relativistic-modes.npz <username>@<login-host>:~/emri/
```

A local (for example macOS) virtual environment is not portable to a Linux
cluster and must not be copied; create the environment on the cluster instead.

## Create the environment

Fewview renders from a saved mode file and does not need the FEW CUDA backend,
so a CPU FEW install is enough. Build the environment on a login node, not in a
render job. A conda environment is convenient:

```bash
module avail                      # find your cluster's conda/python module
module load <conda-module>
conda create -n fewview python=3.11
conda activate fewview
# install your existing FastEMRIWaveforms build, then:
pip install fewview
```

The generated jobs activate this environment on each node. Pass
`--conda-env fewview` (and `--module <conda-module>`) to `fewview-cluster-job`,
or `--venv <path>` for a virtualenv instead.

The jobs use EGL for headless GPU rendering. Before a long run, confirm the
installed VTK build reports the allocated GPU rather than a software renderer;
the render log prints a `pyvista` report near the top. If `vtkEGLRenderWindow`
is unavailable on your cluster, ask your admins which VTK/EGL module or
container to use, or generate the job with `--headless-backend auto` (falls back
to `xvfb`).

## Generate and submit

The defaults render 1,000 frames of a 300-cubed volume with the `shells`
signed-strain look, bodies, a fading trajectory trail, and a waveform panel. At
30 fps that is a 33.3 second movie. To ask for a length directly, pass
`--duration 60` instead of `--frames`; that sets the frame count from the
seconds and frame rate. `--duration` is how long the movie *runs*;
`--animation-cycles` is how much of the inspiral it *covers*.

```bash
conda activate fewview

fewview-cluster-job \
  ~/emri/relativistic-modes.npz \
  --partition <your-gpu-partition> \
  --conda-env fewview \
  --module <conda-module> \
  --segments 20 \
  --max-concurrent 8 \
  --duration 60 --animation-cycles 60

bash fewview-render-job/submit.sh
```

Set `--partition` to your cluster's GPU partition name (the default is `gpu`).
Add `--preemptible` if that partition can preempt jobs: it adds
`#SBATCH --requeue`, and because each finished segment leaves a completion
marker, a requeued task skips the segments it already rendered.

For a fuller flux (energy-flux) movie instead of signed strain, add:

```bash
--component energy_flux --flux-mode-combination coherent \
--opacity-profile flux --color-scheme aurora --presentation dramatic \
--opacity 0.12 --smooth-sigma 0.65
```

The `shells` layer shape is tunable with `--shell-count`, `--shell-min`,
`--shell-max`, `--shell-width`, `--shell-opacity-floor`, and `--shell-glow`;
appearance presets are `--presentation` and `--color-scheme`, and
`--star-count`, `--background-color`, `--color-exposure`, `--camera-zoom`, and
`--no-starfield` override individual choices.

Add `--account <allocation>` if your default Slurm account is not authorized for
the GPU partition. If the CPU merge job needs a different account or partition,
use `--merge-account` and `--merge-partition`.

The generator creates, under `--job-dir` (default `fewview-render-job/`):

- `render_array.slurm` — the GPU job array;
- `merge.slurm` — the dependent segment-concatenation job;
- `submit.sh` — submits both with the correct dependency;
- `segments/<run-id>/` — restartable MP4 segments and completion markers;
- `logs/` — one output and error log per array task.

The finished movie is written to `fewview-render-job/relativistic-emri-animation.mp4`
(or your `--output-name`).

## Sizing the job

The render is CPU-bound and single-threaded per task, so the GPU sits nearly
idle and extra `--cpus-per-task` does not help. Two levers matter:

- **`--segments`** splits the frames into contiguous chunks, one per task. More
  segments means less work per task and less chance of hitting the `--time`
  limit. Keep at least a few dozen frames per segment so the per-task setup cost
  amortizes.
- **`--max-concurrent`** caps how many tasks run at once. Since each task is
  light (one barely-used GPU, ~12-20 GB host memory at `--resolution 300`),
  raise this as high as your queue allows to cut wall-clock time; the total work
  is fixed, so wall time is roughly `total / concurrent`.

`--resolution` drives cost as the cube: 400 is ~2.4x the work of 300 and needs
correspondingly more memory (`--memory`). Do a short throwaway run
(`--segments 2 --duration 2`) first to measure the real per-frame time on your
hardware and confirm EGL is on the GPU before committing to a long array.

Monitor with:

```bash
squeue -u "$USER"
tail -f fewview-render-job/logs/render_<job-id>_<task-id>.out
```
