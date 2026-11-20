# FEWview

FEWview renders the gravitational waves from an extreme mass-ratio inspiral.
It takes the radiative modes from FastEMRIWaveforms (FEW) and evaluates the
strain in every direction around the binary. Radius measures look-back time.
In a still, a crest near the edge left the source earlier than one near the centre.

![Oblique view of an eccentric Kerr inspiral, showing successive wavefronts in the plus polarisation](https://raw.githubusercontent.com/bleather/FEWview/main/docs/images/gallery-magma-oblique.png)

## Installation

Use Python 3.12 or later. On macOS, FEW supplies wheels for Apple Silicon
(macOS 14 or later). Use a native arm64 Python environment there. From a checkout, run

```bash
python -m pip install .
```

To install directly from GitHub, run

```bash
python -m pip install "git+https://github.com/bleather/FEWview.git"
```

Pip installs the public CPU build of FEW 2.2.x. On x86-64 Linux the minimum
is 2.2.1.post1, which fixes CPU compatibility in the published wheels.
FEW downloads the data for each model
on first use. The default eccentric Kerr model needs about 5 GB for its
amplitude table. The circular example below uses a few MB.

## Render a waveform

This example follows a ten-solar-mass secondary on a circular orbit around a
million-solar-mass primary. It uses the post-adiabatic `Circ1PAT1R` model.

```bash
fewview-render --model Circ1PAT1R --M 1e6 --mu 10 --a 0.00001 \
  --p0 10 --e0 0 --chi2 0.3 --T 0.0002 --dt 10 \
  --grid-resolution 64 --opacity-profile shells --output-dir circular
```

Open `circular/relativistic-emri-volume.png` for the still or
`circular/relativistic-volume-slice.png` for a cut through the orbital axis.
The same run saves the modes, so you can change the camera or make a movie
without generating the inspiral again.

```bash
fewview-animate circular/relativistic-modes.npz --output circular/orbit.mp4 \
  --frames 48 --fps 24 --animation-cycles 2 --grid-resolution 64
```

For an eccentric Kerr inspiral, `fewview-render --output-dir kerr` uses
`FastKerrEccentricEquatorialFlux` with spin 0.9 and initial eccentricity 0.4.
The tutorial works through this case in Python, including the choice of
retarded-time window, polarisation and viewing angle.

[Python tutorial](https://github.com/bleather/FEWview/blob/main/docs/tutorial.md) ·
[Example notebook](https://github.com/bleather/FEWview/blob/main/examples/tutorial.ipynb) ·
[Model and API guide](https://github.com/bleather/FEWview/blob/main/docs/reference.md) ·
[Rendering on Slurm](https://github.com/bleather/FEWview/blob/main/docs/cluster_rendering.md)

## Acknowledgements

The outgoing-wave visualisation follows the approach used for numerical
relativity waveforms in gwpv (Nils Vu) and ROSE. FEW supplies the inspiral
trajectories and amplitudes. If you use these figures in a paper, please cite
FEW and the waveform model you used. [gwpv](https://github.com/nilsvu/gwpv) ·
[FEW and its citations](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms#citation)

FEWview is released under the [MIT licence](https://github.com/bleather/FEWview/blob/main/LICENSE).
