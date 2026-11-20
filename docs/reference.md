# Models and API

## Equatorial waveform models

| FEW class | Orbit | Parameters to set |
| --- | --- | --- |
| `FastKerrEccentricEquatorialFlux` (default) | Eccentric Kerr | Primary spin `a`, initial semi-latus rectum `p0`, eccentricity `e0`, inclination cosine `xI0` |
| `FastSchwarzschildEccentricFlux` | Eccentric Schwarzschild | Set `a=0`. Its amplitude table extends to radial harmonic 30. |
| `Circ1PAT1R` | Quasicircular, slowly spinning primary, first post-adiabatic order | Set `e0=0` and `xI0=1`. Pass the dimensionless secondary spin as `chi2` (default 0). |

FEWview 0.1 targets the public FEW 2.2 series. FEW supplies the
allowed parameter bounds for each model. The exact functions and citations
for a model are in the [FEW documentation](https://fastemriwaveforms.readthedocs.io/en/stable/).

```python
import fewview as fv

wave = fv.generate_relativistic_mode_waveform(
    M=1e6, mu=10., a=1e-5, p0=10., e0=0., chi2=0.3,
    model="Circ1PAT1R", dist=1., T=0.0002, dt=10.,
)
h = wave.strain(theta=1., phi=0.3)
h_plus, h_cross = fv.polarizations_from_complex(h)
```

`M` and `mu` are the primary and secondary masses in solar masses.
`p0` is in FEW's geometric units, `T` is in years, `dt` in seconds and
luminosity distance `dist` in Gpc. With `dist=None` the modes have FEW's
unscaled amplitude. Angles in `strain` are in radians, measured in the
source frame. The sum follows the underlying FEW model class's strain
convention. The high-level FEW detector response also applies a polarisation
rotation and distance scaling.

`wave.time` contains the uniformly sampled times in seconds. Column `j` of
`wave.modes` contains the complex mode labelled by `wave.ell[j]` and
`wave.m[j]`, summed over radial harmonics. Use `lmax` and `nmax` to limit
that basis. For a faster preview, `power_fraction` selects harmonics by
their mean squared amplitudes, counting both azimuthal partners.
`max_teukolsky_modes` caps the number retained.

## From modes to a figure

| Function | Result |
| --- | --- |
| `choose_max_delay` | Length of waveform history to display, in seconds |
| `build_mode_retarded_time_volume` | Cartesian grid of plus, cross, amplitude and flux proxy |
| `plot_volume_slice` | Matplotlib figure and axes for a cut through the volume |
| `save_volume` | VTK volume for inspection in ParaView |
| `render_mode_frame` | Still at a specified frame time |
| `render_mode_animation` | MP4 or GIF over a specified time interval |
| `available_color_schemes` | Accepted palette names |
| `paper_style_file` | Path to the bundled Matplotlib style |

Import these names from `fewview`. Use `help(fv.render_mode_animation)` for
the rendering arguments. The [Python tutorial](tutorial.md) has complete calls.
For the coherent flux proxy the field is the squared magnitude of the strain
time derivative. Normalisation and radial tapering control the display, so the
brightness in the rendered image has no absolute flux units.

## Command-line tools

| Command | Input and output |
| --- | --- |
| `fewview-render` | Source parameters to a PNG, a slice and `relativistic-modes.npz`. Add `--export-vti` for ParaView. |
| `fewview-animate` | Saved NPZ to an MP4 or GIF. |
| `fewview-cluster-job` | Saved NPZ and Slurm settings to submission scripts. |
| `fewview-cluster-segment` | One frame range to an MP4 segment. |
| `fewview-cluster-merge` | Completed segments to a single MP4. |

Each command accepts `--help`. Mode archives store arrays by name and can be
opened with `numpy.load(path, allow_pickle=False)`. Existing archives retain
the mode amplitudes and conventions of the FEWview version that generated
them. Regenerate an older archive to pick up corrections to mode reconstruction.
