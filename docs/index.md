# FEWview

FEWview draws an EMRI waveform on a sphere. At each point the field is a sum of
radiative modes weighted by spin-weighted spherical harmonics, evaluated at
retarded time. The radius of the sphere sets how much waveform history fits
on screen. The colour shows the radiative waveform evaluated at that delay.
The orbit drawn in the centre has a separate display scale.

![Plus polarisation of an eccentric Kerr inspiral, viewed obliquely](images/gallery-magma-oblique.png)

## First render

In a Python 3.12 or later environment, install the GitHub release with

```bash
python -m pip install https://github.com/bleather/FEWview/releases/download/v0.2.0/fewview-0.2.0-py3-none-any.whl
```

For the development version, run `python -m pip install .` from a checkout.
On macOS use a native arm64 Python on macOS 14 or later, the platform covered
by the [public FEW wheels](https://pypi.org/project/fastemriwaveforms/#files).
On a headless Ubuntu machine, install Mesa's software renderer before making
a 3D still or movie:

```bash
sudo apt-get update
sudo apt-get install libegl1 libegl-mesa0 libgl1-mesa-dri
```

```bash
fewview-render --model Circ1PAT1R --a 0.00001 --chi2 0.3 --p0 10 --e0 0 \
  --T 0.0002 --grid-resolution 64 --output-dir circular
```

The output directory contains a PNG of the volume, a slice through it and a
portable NumPy archive of the modes. FEW downloads the model data on first
use and caches it locally. An off-screen VTK/OpenGL backend is needed for the
3D still. On a machine without one, add `--no-render-volume` to generate the
slice and mode archive.

- [Generate the modes, choose a time window and render a movie in Python](tutorial.md)
- [Look up waveform models, units, data products and commands](reference.md)
- [Split a long animation across a Slurm array](cluster_rendering.md)

## Project artwork

Use the wordmark on a [light background](img/FEWview-on-light.svg) or a
[dark background](img/FEWview-on-dark.svg), or use the [symbol](img/FEWview-symbol.svg)
where space is limited. The installed Python package includes the SVGs, PNGs,
icons and [DM Sans font licence](img/DM-Sans-OFL.txt) under `fewview/brand/`.
