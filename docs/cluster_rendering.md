# Rendering an EMRI animation on Iridis X

The cluster workflow renders independent, globally numbered movie segments as
a Slurm array and submits a dependent CPU job to concatenate them. Each render
task requests one GPU. The segment renderer uses common waveform-panel limits,
camera timing, and colour normalization so that the joined movie is continuous.

## What to copy

Copy these items to a shared Iridis filesystem:

1. This FastEMRIWaveforms checkout, including `pyproject.toml`, `src/`, and
   `examples/`.
2. The `relativistic-modes.npz` file written by
   `examples/visualize_emri_waveform.py`.

The saved mode file already contains the waveform modes and trajectory needed
for rendering. The original notebooks, existing MP4 files, and the macOS Python
environment are not needed. A macOS virtual environment is not portable to the
Linux cluster and must not be copied.

For example, from the laptop (replace both placeholders):

```bash
rsync -av --exclude=.git --exclude='*.mp4' --exclude='__pycache__' \
  FastEMRIWaveforms/ <username>@<iridis-login-host>:~/FastEMRIWaveforms/

rsync -av emri-visualization/relativistic-modes.npz \
  <username>@<iridis-login-host>:~/FastEMRIWaveforms/emri-visualization/
```

## Create the cluster environment

Run environment installation on an Iridis login node, not in a render job.
Use the Python module currently recommended by the service:

Fewview renders from a saved mode file and does not need the FEW CUDA backend.
Install it into the same environment as your FastEMRIWaveforms installation. A
conda environment is recommended on the cluster:

```bash
module avail python
module load <conda-module>
conda create -n few_env python=3.11
conda activate few_env
# your existing FastEMRIWaveforms install, then:
pip install fewview
```

The generated jobs activate this environment on each node. Pass
`--conda-env few_env` (and `--module <conda-module>`) to `fewview-cluster-job`,
or `--venv <path>` for a virtualenv instead.

The FEW CUDA backend is not required when rendering an existing NPZ file. VTK
uses the allocated GPU for volume ray casting; the FEW CUDA package is relevant
only if the relativistic modes are regenerated on an NVIDIA node.

The generated jobs use EGL for headless GPU rendering. Before a long run, test
that the installed VTK build reports the allocated NVIDIA GPU rather than a
software renderer. If `vtkEGLRenderWindow` is unavailable, ask the Iridis team
which VTK/EGL module or container they recommend, or generate the job with
`--headless-backend auto`.

## Generate and submit the jobs

The defaults reproduce the high-resolution settings requested for the current
movie: 1,000 frames, a 300-cubed volume, direct Cartesian harmonic evaluation,
bodies, a fading two-orbit trajectory line, and a LaTeX-style waveform panel.
At 30 fps that is a 33.3 second movie. To ask for a length directly, pass
`--duration 60` instead of `--frames`, which sets the frame count from the
requested seconds and the frame rate. That controls how long the movie runs;
how much of the inspiral it covers is `--animation-cycles`.
Add one of the presentation regimes shown below for a brighter navy-and-
starfield presentation. The renderer also uses display smoothing, a
resolution-independent opacity distance, soft lighting, and a common scale
sampled at nine times across the complete movie. Direct evaluation removes the
spherical interpolation seam but uses substantially more host memory than
``--angular-sampling spherical``.

```bash
cd ~/FastEMRIWaveforms
source .venv-iridisx/bin/activate

fewview-cluster-job \
  emri-visualization/relativistic-modes.npz \
  --partition a100 \
  --segments 10 \
  --max-concurrent 8 \
  --module <python-module>

bash iridisx-emri-job/submit.sh
```

For a shells-style signed-strain movie, add:

```bash
--component plus --opacity-profile shells --camera-view oblique \
--color-scheme rainbow --presentation shells_dramatic \
--opacity 0.30 --smooth-sigma 0.80 \
--shells-num-layers 7 --shells-layer-min 0.10 --shells-layer-max 0.92 \
--shells-layer-width 0.075 --shells-opacity-floor 0.16 --shells-glow 0.12
```

For a fuller flux-style energy-flux movie, add:

```bash
--component energy_flux --flux-mode-combination coherent \
--opacity-profile flux --camera-view oblique \
--color-scheme aurora --presentation dramatic \
--opacity 0.12 --smooth-sigma 0.65
```

The flux profile logarithmically compresses four decades of flux before volume
rendering. This prevents the large dynamic range of `|dh/dt|^2` from leaving
only the narrow, brightest wavefronts visible.
The coherent option retains the scientifically faithful instantaneous angular
lobes. Use `--flux-mode-combination incoherent` only when an explicitly
phase-averaged modal-power proxy is desired. Increase
`--normalization-samples` for a more exhaustive global scale, or set
`--normalization-time` to deliberately normalize at one waveform time.

The `shells_dramatic` preset enables a deterministic 6,500-star field, a deep
navy background, 1.30 colour exposure, emissive-looking unshaded colour, and a
1.12 camera zoom. Its seven
transfer-function layers progress from blue/cyan through green and gold to red,
with increasing opacity at larger positive strain. The weak broad halo rounds
their edges without filling the gaps between wavefronts. Five displayed wave
cycles gives a good balance between layered shells and negative space.

The magma `dramatic` preset remains available for the flux render. It uses
4,000 stars, 2.25 colour exposure, and a 1.10 camera zoom. Colour exposure is
independent of opacity in both presets, so brightness does not turn weaker
modes into solid surfaces. Override individual choices with `--star-count`,
`--background-color`, `--color-exposure`, or `--camera-zoom`; use
`--no-starfield` if a clean black background is required.

Add `--account <gpu-allocation>` when your default Slurm account is not
authorized for the selected GPU partition. If the later CPU merge job also
requires an explicit account or partition, use `--merge-account` and
`--merge-partition` separately. The generator creates:

- `render_array.slurm`: the GPU job array;
- `merge.slurm`: the dependent segment-concatenation job;
- `submit.sh`: submits both jobs with the correct dependency;
- `segments/<run-id>/`: restartable MP4 segments and completion markers;
- `logs/`: one output and error log per array task.

The finished movie is written to
`iridisx-emri-job/relativistic-emri-animation.mp4`.

## Choosing a partition

- `a100` is the recommended first choice: it is non-preemptible and the
  rendering workload does not benefit from H100-specific tensor hardware.
- `swarm_a100` and `swarm_h100` can be selected by eligible ECS/ORC staff and
  PGR users.
- `scavenger_4a100` and `scavenger_8h100` may start sooner but can be preempted.
  Generated scavenger jobs request Slurm requeueing, and completed segments are
  skipped when a task restarts.

Every array task requests one GPU with `--gres=gpu:1`. NVLink and NVSwitch are
not used because segments are independent; throughput comes from running
several one-GPU tasks concurrently.

Monitor the workflow with:

```bash
squeue -u "$USER"
tail -f iridisx-emri-job/logs/render_<job-id>_<task-id>.out
```
