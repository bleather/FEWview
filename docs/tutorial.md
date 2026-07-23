# Tutorial

This walkthrough renders a still and a short movie from a FEW EMRI waveform. It
mirrors [`examples/tutorial.ipynb`](../examples/tutorial.ipynb); run that
notebook for an executable version with embedded images.

Fewview reads FastEMRIWaveforms models directly, so install it into the same
environment as your existing FEW install:

```bash
pip install fewview
```

## 1. Generate a waveform

`generate_relativistic_mode_waveform` runs the fully relativistic
`FastKerrEccentricEquatorialFlux` model and returns a `RelativisticModeWaveform`
carrying the individual $h_{\ell m}(t)$ modes and the FEW inspiral trajectory.
This is the one slow step; everything downstream is fast.

```python
import numpy as np
import fewview as fv

waveform = fv.generate_relativistic_mode_waveform(
    M=1.0e6, mu=10.0, a=0.9, p0=12.0, e0=0.4, xI0=1.0,
    dt=10.0, T=0.01, lmax=10, nmax=55,
)
```

The amplitude model is trained up to `nmax=55`. Requesting a larger value emits
a warning and falls back to the trained set, so at high eccentricity some
high-overtone content is unavoidably missing.

## 2. Choose how much history to show

The sphere maps waveform history from its centre (the current frame) to its
edge. `choose_max_delay` picks a span holding a resolvable number of wave
cycles; a small number keeps the shells cleanly separated.

```python
reference = waveform.strain(theta=np.pi / 3.0, phi=0.0)
h_plus, h_cross = np.real(reference), -np.imag(reference)

period = fv.estimate_waveform_period(waveform.time, h_plus, h_cross)
max_delay = fv.choose_max_delay(waveform.time, h_plus, h_cross, wave_cycles=1.5)
end = float(waveform.time[-1])
```

## 3. Render a still

`render_mode_frame` builds the volume for one instant and renders it. Three
choices set the look:

- **`component`** — `plus` has no equatorial node and is the usual choice.
- **`opacity_profile`** — `shells` draws nested translucent signed-strain
  sheets.
- **`color_scheme`** — `rainbow`, a smooth blue-to-red sequence tuned for the
  shells.

```python
fv.render_mode_frame(
    waveform,
    screenshot="tutorial-shells.png",
    max_delay=max_delay,
    frame_time=end,
    waveform_start_time=end - 8 * period,
    waveform_end_time=end,
    resolution=200,
    component="plus",
    opacity_profile="shells",
    color_scheme="rainbow",
    presentation="shells_dramatic",
    camera_view="oblique",
    camera_zoom=0.95,
    shell_count=7, shell_min=0.10, shell_max=0.92,
    smooth_sigma=1.2,
    inner_window_fraction=0.10, outer_window_fraction=0.10,
    trajectory_color="#00b7ff",
    window_size=(1000, 620),
)
```

The central bodies are drawn on the orbit's own scale — the primary spans a few
horizon radii $r_+$ — with a fading inspiral trail behind the secondary
(`trajectory_color` accepts any Matplotlib colour).

## 4. A different look: face-on

`camera_view="face_on"` looks straight down the spin axis, revealing the spiral
wavefront pattern. Any palette works; `cool` gives a calm blue-to-magenta
sequence. This is the same `shells` profile as before — only the camera and
colours change.

```python
fv.render_mode_frame(
    waveform, screenshot="tutorial-faceon.png",
    max_delay=max_delay, frame_time=end,
    waveform_start_time=end - 8 * period, waveform_end_time=end,
    resolution=200,
    component="plus", opacity_profile="shells", color_scheme="cool",
    presentation="shells_dramatic", camera_view="face_on", camera_zoom=1.1,
    shell_count=7, shell_min=0.10, shell_max=0.92, smooth_sigma=1.2,
    inner_window_fraction=0.10, outer_window_fraction=0.10,
    trajectory_color="#00b7ff", window_size=(760, 760),
)
```

For energy flux instead of strain, set `component="energy_flux"` with
`opacity_profile="flux"`, which logarithmically compresses $|\dot h|^2$ so
faint, broad emission stays visible. It reads best at higher eccentricity,
where the flux develops strong spiral structure.

## 5. Diagnostic slice

Before a full render, a coordinate-plane slice is a fast way to inspect the
field:

```python
volume = fv.build_mode_retarded_time_volume(
    waveform, frame_time=end, max_delay=max_delay, resolution=160,
    inner_window_fraction=0.10, outer_window_fraction=0.10,
)
fig, _ = fv.plot_volume_slice(volume, component="plus", plane="xz")
```

## 6. A short animation

`render_mode_animation` sweeps a range of frame times to an MP4, reusing the
angular basis across frames and holding the colour scale fixed so the movie does
not flicker.

```python
anim_end = end
anim_start = max(float(waveform.time[0] + max_delay), anim_end - 3.0 * period)

fv.render_mode_animation(
    waveform, "tutorial.mp4",
    max_delay=max_delay, start_time=anim_start, end_time=anim_end,
    frames=24, fps=12, resolution=110,
    component="plus", opacity_profile="shells", color_scheme="rainbow",
    presentation="shells_dramatic", camera_zoom=0.95,
    shell_count=7, smooth_sigma=1.2,
    inner_window_fraction=0.10, outer_window_fraction=0.10,
    trajectory_color="#00b7ff", show_waveform=True,
    window_size=(640, 400), normalization_samples=3,
)
```

`--animation-cycles` (via `start_time`/`end_time` here) sets how much inspiral
is traversed; the frame count sets how smoothly. Aim for at least ~20 frames per
wave cycle, or the wave pattern strobes.

## Choosing settings

| knob | options | notes |
| --- | --- | --- |
| `component` | `plus`, `cross`, `amplitude`, `energy_flux` | `cross` is zero on the equatorial plane; `amplitude` is basis-independent |
| `opacity_profile` | `soft`, `bands`, `shells`, `flux` | `shells` needs `plus`/`cross`; `flux` needs `energy_flux` |
| `color_scheme` | Matplotlib names + `rainbow`, `aurora`, `cinematic` | `fv.available_color_schemes()` lists them |
| `presentation` | `balanced`, `dramatic`, `shells_dramatic` | preset lighting, exposure, starfield |
| `camera_view` | `oblique`, `face_on` | `face_on` looks down the spin axis |

## Two physical caveats

- **The polar axis** carries a faint, unavoidable seam. A spin-2 field has no
  continuous scalar representation over the poles, so `plus`/`cross` are
  multivalued there. It is faint obliquely and softened by `smooth_sigma`;
  raising `resolution` makes it sharper, not fainter. `amplitude` and
  `energy_flux` are immune.
- **The equatorial gap** for `component="cross"` is physical, not a bug:
  $h_\times \propto \cos\theta$ vanishes on the equatorial plane for an
  equatorial orbit. Use `plus` to avoid it.

## Cluster rendering

For long, high-resolution movies the per-frame CPU cost dominates and the work
is serial within a task, so split across many Slurm array tasks with
`fewview-cluster-job`. See [cluster_rendering.md](cluster_rendering.md).
