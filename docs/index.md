# FEWview

FEWview renders the gravitational waves from an extreme mass-ratio inspiral.
It takes the radiative modes from FastEMRIWaveforms (FEW) and evaluates the
strain in every direction around the binary. Radius measures look-back time.
A crest near the edge left the source earlier than one near the centre.
The orbit at the centre has its own display scale.

[![Four ice-blue views of an eccentric Kerr inspiral above an hour-long strain trace. Four dots mark the snapshots in the shaded interval at the right.](images/ice-orbit-waveform.png)](images/ice-orbit-waveform.png)

The four snapshots span one cycle of the plus polarisation, from one crest to
the next, seen from 18° above the orbital plane. The trace shows an hour of
strain along that line of sight. The shaded interval is the cycle pictured above.

## Installation

Use Python 3.12 or later. On macOS, the public FEW wheels require Apple Silicon
and macOS 14 or later. Use a native arm64 Python environment there.

```bash
python -m pip install fewview
```

Pip installs the public CPU build of FEW 2.2.x. On x86-64 Linux it selects
2.2.1.post1 or later. FEW downloads and caches the data for each waveform model
on first use. The circular example below needs a few MB. The eccentric Kerr
model needs about 5 GB for its amplitude table.

To install the development version, use
`python -m pip install "git+https://github.com/bleather/FEWview.git"`,
or run `python -m pip install .` from a checkout.

## First render

On a headless Ubuntu machine, install Mesa's software renderer before making
a 3D still or movie.

```bash
sudo apt-get update
sudo apt-get install libegl1 libegl-mesa0 libgl1-mesa-dri
```

This example follows a ten-solar-mass secondary on a circular orbit around a
million-solar-mass primary using the post-adiabatic `Circ1PAT1R` model.

```bash
fewview-render --model Circ1PAT1R --M 1e6 --mu 10 --a 0.00001 \
  --p0 10 --e0 0 --chi2 0.3 --T 0.0002 --dt 10 \
  --grid-resolution 64 --opacity-profile shells --output-dir circular
```

Open `circular/relativistic-emri-volume.png` for the still or
`circular/relativistic-volume-slice.png` for a cut through the orbital axis.
On a machine without an off-screen VTK/OpenGL backend, add `--no-render-volume`
to generate just the slice and the mode archive.

Use the saved modes to make a two-second ice-blue movie with a glowing orbital
trail and the strain trace underneath.

```bash
fewview-animate circular/relativistic-modes.npz --output circular/orbit.mp4 \
  --frames 48 --fps 24 --animation-cycles 2 --grid-resolution 64 \
  --opacity-profile shells --presentation shells_dramatic --color-scheme ice \
  --camera-latitude 18 --camera-longitude -35 \
  --source-glow --waveform-glow --no-waveform-transparent
```

For an eccentric Kerr inspiral, run `fewview-render --output-dir kerr`.
The [Python tutorial](tutorial.md) covers the retarded-time window, viewing
angle and strain panel. The [model and API guide](reference.md) lists the
available fields and the [Slurm guide](cluster_rendering.md) covers long movies.

## Citation

If you use FEWview for a figure or movie, cite the release you used. For v0.2.2, use this reference.

> Leather, B., & Witzany, V. (2026). *FEWview* (v0.2.2) [Software]. [GitHub release](https://github.com/bleather/FEWview/releases/tag/v0.2.2).

For BibTeX, select **Cite this repository** on the [GitHub repository](https://github.com/bleather/FEWview).
Please also cite FEW and the waveform model you used. [FEW citation guide](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms#citation)

### Contributors

- **Benjamin Leather** created and maintains FEWview.
- **Vojtěch Witzany** corrected the sign of negative-*m* modes for odd ℓ in
  FEWview’s waveform reconstruction.
- **Zachary Nasipak** developed fewviz and highlighted FEWview in its
  documentation. [@znasipak](https://github.com/znasipak)

## Related projects

- [**fewviz**](https://github.com/znasipak/fewviz) by Zachary Nasipak animates
  an inspiral orbit alongside its waveform from FastEMRIWaveforms. It can also
  render the outgoing gravitational-wave strain in 3D.

## Project artwork

Figures and movies carry the FEWview wordmark in the lower-right corner.
Use `--logo icon` or `logo="icon"` in Python for the orbit symbol alone.
Download the wordmark for a [light background](img/FEWview-on-light.svg)
or a [dark background](img/FEWview-on-dark.svg), or the [symbol](img/FEWview-symbol.svg).
The package includes the artwork and [DM Sans font licence](img/DM-Sans-OFL.txt)
under `fewview/brand/`.
