r"""Retarded-time visualizations for time-domain EMRI waveforms.

The mode-resolved path in this module uses the fully relativistic
``FastKerrEccentricEquatorialFlux`` model.  Its Teukolsky amplitudes and orbital
phases are reconstructed into :math:`h_{\ell m}(t)` and combined with FEW's
spin-weighted spherical harmonics over the whole sky.  The resulting field is
then evaluated at retarded time ``u = t_frame - r / c`` throughout a sphere, so
that a still or movie shows the gravitational wave propagating outward.

The older line-of-sight helpers remain available for lightweight diagnostic
plots.  They are not used by the full-sky example or notebook.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from numba import njit, prange

from few.utils.ylm import GetYlms, _ylm_kernel_inner


AngularPattern = Literal["line_of_sight", "quadrupole", "mode_resolved"]
VolumeComponent = Literal["plus", "cross", "amplitude", "energy_flux"]
VolumeRenderStyle = Literal["cinematic", "contours"]
VolumeOpacityProfile = Literal["soft", "bands", "shells", "flux"]
VolumeCameraView = Literal["oblique", "face_on"]
VolumePresentation = Literal["balanced", "dramatic", "shells_dramatic"]
ModeAngularSampling = Literal["cartesian", "spherical"]
FluxModeCombination = Literal["coherent", "incoherent"]
WaveformFontStyle = Literal["latex", "sans"]
VolumeColorScheme = Literal[
    "magma",
    "aurora",
    "rainbow",
    "inferno",
    "plasma",
    "viridis",
    "cividis",
    "cool",
    "blues",
    "cinematic",
]

# Palettes handed straight to Matplotlib. The values are the Matplotlib names,
# which are case sensitive, so the lower-case scheme names used throughout this
# module need an explicit mapping.
_MATPLOTLIB_COLOR_SCHEMES = {
    "magma": "magma",
    "aurora": "magma",
    "inferno": "inferno",
    "plasma": "plasma",
    "viridis": "viridis",
    "cividis": "cividis",
    "cool": "cool",
    "blues": "Blues",
}

# Every accepted scheme: the Matplotlib passthroughs plus the two hand-built
# sequences assembled in ``_volume_colormap``.
_VOLUME_COLOR_SCHEMES = frozenset(_MATPLOTLIB_COLOR_SCHEMES) | {
    "rainbow",
    "cinematic",
}
PathLike = Union[str, Path]

# Display halo radius as a multiple of the primary's drawn radius.
_PRIMARY_HALO_SCALE = 1.18
# Fraction of the display periapsis the halo and secondary may together fill.
_BODY_CLEARANCE_FRACTION = 0.8
# Colour of the fading inspiral trail. Any Matplotlib colour spec is accepted.
DEFAULT_TRAJECTORY_COLOR = "#ffd36a"
# Drawn secondary radius as a fraction of the drawn primary radius. The
# secondary's own horizon is ~1e-5 of the primary's, so it is a legibility
# choice, not a scale; this ratio matches the hand-tuned circular-orbit look.
_SECONDARY_BODY_SCALE = 0.33


@dataclass(frozen=True)
class _ResolvedVolumePresentation:
    """Concrete display settings after applying a presentation preset."""

    color_exposure: float
    background_color: str
    camera_zoom: float
    star_count: int
    starfield: bool
    ambient: float
    diffuse: float
    specular: float
    shade: bool


@dataclass(frozen=True)
class StrainSurface:
    """A polar surface carrying an outgoing gravitational waveform.

    The surface geometry is intentionally exaggerated.  Color contains the
    normalized (or raw) plus polarization after a spin-2 rotation around the
    source, while ``z`` is a display displacement derived from the same field.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    plus: np.ndarray
    cross: np.ndarray
    delay: np.ndarray
    frame_time: float
    max_delay: float
    normalization: float
    normalized: bool


@dataclass(frozen=True)
class RetardedTimeVolume:
    """Strain and strain-rate-squared fields on a uniform Cartesian grid."""

    axis: np.ndarray
    plus: np.ndarray
    cross: np.ndarray
    amplitude: np.ndarray
    energy_flux: np.ndarray
    frame_time: float
    max_delay: float
    radius: float
    normalization: float
    energy_flux_normalization: float
    normalized: bool
    angular_pattern: AngularPattern
    model: Optional[str] = None
    mode_count: int = 0

    def component(self, name: VolumeComponent = "plus") -> np.ndarray:
        """Return a named scalar field."""

        if name == "plus":
            return self.plus
        if name == "cross":
            return self.cross
        if name == "amplitude":
            return self.amplitude
        if name == "energy_flux":
            return self.energy_flux
        raise ValueError(
            "component must be 'plus', 'cross', 'amplitude', or 'energy_flux'"
        )


@dataclass(frozen=True)
class RelativisticModeWaveform:
    r"""Uniformly sampled, full-sky FEW radiative modes.

    ``modes[:, k]`` is :math:`h_{\ell m}(t)` before multiplication by
    :math:`{}_{-2}Y_{\ell m}(\theta,\phi)`.  Radial harmonics ``n`` have already
    been summed, and both positive and negative ``m`` modes are present using
    the same parity convention as :class:`few.utils.ylm.GetYlms`.
    """

    time: np.ndarray
    modes: np.ndarray
    ell: np.ndarray
    m: np.ndarray
    teukolsky_modes_used: int
    teukolsky_modes_available: int
    retained_power_fraction: float
    model: str = "FastKerrEccentricEquatorialFlux"
    trajectory_p: Optional[np.ndarray] = None
    trajectory_e: Optional[np.ndarray] = None
    trajectory_xI: Optional[np.ndarray] = None
    trajectory_phi_phi: Optional[np.ndarray] = None
    trajectory_phi_r: Optional[np.ndarray] = None
    primary_mass: Optional[float] = None
    secondary_mass: Optional[float] = None
    spin: Optional[float] = None

    @property
    def horizon_radius(self) -> Optional[float]:
        r"""Return the outer Kerr horizon :math:`r_+/M=1+\sqrt{1-a^2}`.

        ``None`` when the waveform carries no spin, which is the case for mode
        files written before the spin was recorded.
        """

        if self.spin is None:
            return None
        spin = float(self.spin)
        if abs(spin) > 1.0:
            raise ValueError("spin must satisfy |a| <= 1")
        return 1.0 + np.sqrt(1.0 - spin**2)

    def strain(self, theta: float, phi: float) -> np.ndarray:
        """Return :math:`h_+-i h_\times` for one source-frame direction."""

        ylms = GetYlms(force_backend="cpu")(self.ell, self.m, theta, phi)
        return self.modes @ _as_numpy(ylms)

    @property
    def has_trajectory(self) -> bool:
        """Whether the waveform carries FEW's uniformly sampled inspiral."""

        return all(
            value is not None
            for value in (
                self.trajectory_p,
                self.trajectory_e,
                self.trajectory_xI,
                self.trajectory_phi_phi,
                self.trajectory_phi_r,
            )
        )

    def orbital_position(self) -> np.ndarray:
        r"""Return the osculating equatorial trajectory in units of ``M``.

        The Boyer--Lindquist radial coordinate is reconstructed from FEW's
        relativistic trajectory as
        :math:`r=p/(1+e\cos\Phi_r)`.  The azimuth is FEW's accumulated
        :math:`\Phi_\phi`; this is a display coordinate transformation, not a
        separate approximate waveform model.
        """

        if not self.has_trajectory:
            raise ValueError(
                "This mode waveform does not include a FEW trajectory. "
                "Regenerate it with generate_relativistic_mode_waveform."
            )
        _validate_mode_waveform(self)
        p = np.asarray(self.trajectory_p, dtype=float)
        e = np.asarray(self.trajectory_e, dtype=float)
        phi = np.asarray(self.trajectory_phi_phi, dtype=float)
        phi_r = np.asarray(self.trajectory_phi_r, dtype=float)
        radial_position = p / (1.0 + e * np.cos(phi_r))
        return np.column_stack(
            (
                radial_position * np.cos(phi),
                radial_position * np.sin(phi),
                np.zeros_like(radial_position),
            )
        )


@dataclass(frozen=True)
class _ModeVolumeSampling:
    """Geometry and interpolants reused across animation frames."""

    waveform: RelativisticModeWaveform
    mode_spline: object
    ylms: np.ndarray
    angular_sampling: ModeAngularSampling
    radii: np.ndarray
    inner_window: np.ndarray
    outer_window: np.ndarray
    axis: np.ndarray
    coordinates: np.ndarray
    cartesian_shape: tuple[int, int, int]
    inside: np.ndarray
    radius: float
    polar_samples: int
    azimuthal_samples: int
    phi_padding: int
    inside_indices: np.ndarray
    cartesian_radius_fraction: np.ndarray
    cartesian_window: np.ndarray


def polarizations_from_complex(
    strain: np.ndarray, *, convention: Literal["few", "summation"] = "few"
) -> tuple[np.ndarray, np.ndarray]:
    r"""Split a complex waveform into plus and cross polarizations.

    Args:
        strain: One-dimensional complex waveform.
        convention: ``"few"`` for the public ``GenerateEMRIWaveform`` output
            :math:`h_+ - i h_\times`, or ``"summation"`` for the internal
            summation convention :math:`h_+ + i h_\times`.

    Returns:
        A tuple ``(h_plus, h_cross)`` of NumPy arrays.
    """

    strain_array = _as_numpy(strain)
    if strain_array.ndim != 1:
        raise ValueError("strain must be one-dimensional")
    if convention == "few":
        return np.real(strain_array), -np.imag(strain_array)
    if convention == "summation":
        return np.real(strain_array), np.imag(strain_array)
    raise ValueError("convention must be 'few' or 'summation'")


def generate_relativistic_mode_waveform(
    M: float,
    mu: float,
    a: float,
    p0: float,
    e0: float,
    xI0: float = 1.0,
    *,
    dist: Optional[float] = 1.0,
    Phi_phi0: float = 0.0,
    Phi_r0: float = 0.0,
    dt: float = 10.0,
    T: float = 0.001,
    lmax: int = 10,
    nmax: int = 55,
    power_fraction: float = 1.0,
    max_teukolsky_modes: Optional[int] = None,
    interpolation_chunk_size: int = 256,
    force_backend: str = "cpu",
) -> RelativisticModeWaveform:
    """Generate full-sky modes with FEW's relativistic Kerr flux model.

    This follows :class:`few.waveform.FastKerrEccentricEquatorialFlux` exactly:
    a relativistic Kerr flux trajectory supplies the phases, the Teukolsky
    interpolant supplies complex ``(l,m,n)`` amplitudes, and the radial
    harmonics are combined into uniformly sampled ``(l,m)`` modes.  With the
    default ``power_fraction=1`` and no mode cap, all available modes are used.

    ``power_fraction`` and ``max_teukolsky_modes`` are optional rendering-speed
    controls.  Selection is based on sky-integrated mode power, including the
    negative-``m`` partner of every positive-``m`` mode.
    """

    if not 0.0 < power_fraction <= 1.0:
        raise ValueError("power_fraction must be in the interval (0, 1]")
    if max_teukolsky_modes is not None and max_teukolsky_modes < 1:
        raise ValueError("max_teukolsky_modes must be positive or None")
    if interpolation_chunk_size < 1:
        raise ValueError("interpolation_chunk_size must be positive")
    if dt <= 0.0 or T <= 0.0:
        raise ValueError("dt and T must be positive")
    if dist is not None and dist <= 0.0:
        raise ValueError("dist must be positive or None")

    from scipy.interpolate import CubicSpline

    from few.utils.constants import Gpc, MRSUN_SI
    from few.waveform.waveform import FastKerrEccentricEquatorialFlux

    generator = FastKerrEccentricEquatorialFlux(
        lmax=lmax,
        nmax=nmax,
        force_backend=force_backend,
    )
    a, xI0 = generator.sanity_check_init(M, mu, a, p0, e0, xI0)
    trajectory = generator.inspiral_generator(
        M,
        mu,
        a,
        p0,
        e0,
        xI0,
        Phi_phi0=Phi_phi0,
        Phi_theta0=0.0,
        Phi_r0=Phi_r0,
        T=T,
        dt=dt,
        **generator.inspiral_kwargs,
    )
    t_sparse, p, e, xI, _, _, _ = (_as_numpy(value) for value in trajectory)
    generator.sanity_check_traj(a, p, e, xI)

    amplitude_generator = generator.amplitude_generator
    amplitudes = _as_numpy(amplitude_generator(a, p, e, xI0))
    # Label the amplitude columns from the amplitude module's own index arrays,
    # not the outer generator's. The amplitude model is trained to a fixed
    # ``nmax`` (currently 55), so a larger requested ``nmax`` leaves the
    # generator's ``*_arr_no_mask`` describing more modes than the amplitude
    # module actually returns, and the two no longer line up.
    ell_all = _as_numpy(amplitude_generator.l_arr_no_mask).astype(
        np.int32, copy=False
    )
    m_all = _as_numpy(amplitude_generator.m_arr_no_mask).astype(
        np.int32, copy=False
    )
    n_all = _as_numpy(amplitude_generator.n_arr_no_mask).astype(
        np.int32, copy=False
    )
    available = amplitudes.shape[1]
    if ell_all.shape[0] != available:
        raise ValueError(
            "amplitude mode-index arrays do not match the amplitude output "
            f"({ell_all.shape[0]} labels for {available} modes)"
        )
    amplitude_nmax = int(np.abs(n_all).max())
    if nmax > amplitude_nmax:
        warnings.warn(
            f"The amplitude model is trained to nmax={amplitude_nmax}, so the "
            f"requested nmax={nmax} was capped; {available} Teukolsky modes "
            "are available.",
            stacklevel=2,
        )

    # The factor of two accounts for the symmetry-related negative-m partner.
    power = np.mean(np.abs(amplitudes) ** 2, axis=0)
    power *= np.where(m_all > 0, 2.0, 1.0)
    order = np.argsort(power)[::-1]
    target = power_fraction * float(np.sum(power))
    count = int(np.searchsorted(np.cumsum(power[order]), target, side="left") + 1)
    # searchsorted can land one past the end when the cumulative power reaches
    # the target only at the final mode (e.g. power_fraction=1.0 with rounding),
    # which would otherwise report more modes used than exist.
    count = min(count, available)
    if max_teukolsky_modes is not None:
        count = min(count, max_teukolsky_modes)
    selected = np.sort(order[:count])
    retained_power = float(np.sum(power[selected]) / np.sum(power))

    ell_selected = ell_all[selected]
    m_selected = m_all[selected]
    n_selected = n_all[selected]
    positive_lm = sorted(set(zip(ell_selected.tolist(), m_selected.tolist())))
    all_lm = sorted(
        set(positive_lm)
        | {(ell, -m) for ell, m in positive_lm if m > 0}
    )
    lm_index = {mode: index for index, mode in enumerate(all_lm)}
    ell = np.asarray([mode[0] for mode in all_lm], dtype=np.int32)
    m = np.asarray([mode[1] for mode in all_lm], dtype=np.int32)

    number_of_samples = int((float(t_sparse[-1]) - float(t_sparse[0])) / dt) + 1
    time = float(t_sparse[0]) + np.arange(number_of_samples, dtype=float) * dt
    dense_trajectory = (
        generator.inspiral_generator.inspiral_generator.eval_integrator_spline(time)
    )
    dense_p = np.asarray(dense_trajectory[:, 0], dtype=float)
    dense_e = np.asarray(dense_trajectory[:, 1], dtype=float)
    dense_xI = np.asarray(dense_trajectory[:, 2], dtype=float)
    Phi_phi = np.asarray(dense_trajectory[:, 3], dtype=float)
    Phi_r = np.asarray(dense_trajectory[:, 5], dtype=float)
    if a > 0.0:
        Phi_phi *= np.sign(xI0)

    modes = np.zeros((time.size, len(all_lm)), dtype=np.complex128)
    for start in range(0, count, interpolation_chunk_size):
        stop = min(start + interpolation_chunk_size, count)
        indices = selected[start:stop]
        amplitude_chunk = CubicSpline(
            t_sparse, amplitudes[:, indices], axis=0
        )(time)
        phase = (
            m_all[indices][None, :] * Phi_phi[:, None]
            + n_all[indices][None, :] * Phi_r[:, None]
        )
        positive_modes = amplitude_chunk * np.exp(-1j * phase)
        for local_index, global_index in enumerate(indices):
            ell_here = int(ell_all[global_index])
            m_here = int(m_all[global_index])
            modes[:, lm_index[(ell_here, m_here)]] += positive_modes[:, local_index]
            if m_here > 0:
                modes[:, lm_index[(ell_here, -m_here)]] += np.conj(
                    positive_modes[:, local_index]
                )

    if dist is not None:
        distance_dimensionless = (dist * Gpc) / (mu * MRSUN_SI)
        modes /= distance_dimensionless

    return RelativisticModeWaveform(
        time=time,
        modes=modes,
        ell=ell,
        m=m,
        teukolsky_modes_used=count,
        teukolsky_modes_available=available,
        retained_power_fraction=retained_power,
        trajectory_p=dense_p,
        trajectory_e=dense_e,
        trajectory_xI=dense_xI,
        trajectory_phi_phi=Phi_phi,
        trajectory_phi_r=Phi_r,
        primary_mass=float(M),
        secondary_mass=float(mu),
        spin=float(a),
    )


def _prepare_mode_volume_sampling(
    waveform: RelativisticModeWaveform,
    *,
    resolution: int,
    radius: float,
    polar_samples: int,
    azimuthal_samples: int,
    inner_window_fraction: float,
    outer_window_fraction: float,
    angular_sampling: ModeAngularSampling,
) -> _ModeVolumeSampling:
    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if angular_sampling not in ("cartesian", "spherical"):
        raise ValueError("angular_sampling must be 'cartesian' or 'spherical'")
    if angular_sampling == "spherical" and (
        polar_samples < 8 or azimuthal_samples < 16
    ):
        raise ValueError("polar_samples >= 8 and azimuthal_samples >= 16 are required")
    if not 0.0 <= inner_window_fraction < 0.5:
        raise ValueError("inner_window_fraction must be in the interval [0, 0.5)")
    if not 0.0 <= outer_window_fraction < 0.5:
        raise ValueError("outer_window_fraction must be in the interval [0, 0.5)")
    _validate_mode_waveform(waveform)

    from scipy.interpolate import CubicSpline

    radii = np.linspace(0.0, radius, resolution)
    if inner_window_fraction > 0.0:
        inner_window = _smoothstep(
            (radii / radius) / inner_window_fraction
        )[:, None, None]
    else:
        inner_window = np.ones((resolution, 1, 1), dtype=float)

    axis = np.linspace(-radius, radius, resolution)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    radial_coordinate = np.sqrt(x * x + y * y + z * z)
    safe_radius = np.where(radial_coordinate > 0.0, radial_coordinate, 1.0)
    theta = np.broadcast_to(
        np.arccos(np.clip(z / safe_radius, -1.0, 1.0)),
        radial_coordinate.shape,
    )
    phi = np.broadcast_to(
        np.mod(np.arctan2(y, x), 2.0 * np.pi),
        radial_coordinate.shape,
    )
    if outer_window_fraction > 0.0:
        outer_window = _smoothstep(
            (radius - radial_coordinate) / (radius * outer_window_fraction)
        )
    else:
        outer_window = np.ones(radial_coordinate.shape, dtype=float)
    inside = radial_coordinate <= radius
    inside_indices = np.flatnonzero(inside.ravel())
    cartesian_radius_fraction = (
        radial_coordinate.ravel()[inside_indices] / radius
    )
    if inner_window_fraction > 0.0:
        cartesian_inner_window = _smoothstep(
            cartesian_radius_fraction / inner_window_fraction
        )
    else:
        cartesian_inner_window = np.ones_like(cartesian_radius_fraction)
    cartesian_window = (
        cartesian_inner_window * outer_window.ravel()[inside_indices]
    )

    ell = np.asarray(waveform.ell, dtype=np.int32)
    m = np.asarray(waveform.m, dtype=np.int32)
    if angular_sampling == "cartesian":
        # Evaluate the spin-weighted harmonics at the actual VTK voxel
        # directions.  Storing them as complex64 halves the reusable basis
        # memory without affecting visible rendering precision.  Crucially,
        # there is no interpolated phi coordinate and therefore no 0/2pi seam.
        ylms = _evaluate_ylm_grid(
            ell,
            m,
            theta.ravel()[inside_indices],
            phi.ravel()[inside_indices],
            dtype=np.complex64,
        )
        phi_padding = 0
        coordinates = np.empty((3, 0), dtype=float)
    else:
        theta_values = np.linspace(0.0, np.pi, polar_samples)
        phi_values = np.linspace(
            0.0, 2.0 * np.pi, azimuthal_samples, endpoint=False
        )
        theta_flat = np.repeat(theta_values, azimuthal_samples)
        phi_flat = np.tile(phi_values, polar_samples)
        ylms = _evaluate_ylm_grid(
            ell,
            m,
            theta_flat,
            phi_flat,
            dtype=np.complex128,
        )
        # Retain the former spherical-grid path as a lower-memory fallback.
        # Cubic B-spline prefiltering has a short but non-local footprint, so
        # wrapped samples keep its boundary away from the physical phi seam.
        phi_padding = 8
        coordinates = np.vstack(
            [
                (radial_coordinate.ravel() / radius) * (resolution - 1),
                (theta.ravel() / np.pi) * (polar_samples - 1),
                (phi.ravel() / (2.0 * np.pi)) * azimuthal_samples
                + phi_padding,
            ]
        )
    return _ModeVolumeSampling(
        waveform=waveform,
        mode_spline=CubicSpline(waveform.time, waveform.modes, axis=0),
        ylms=ylms,
        angular_sampling=angular_sampling,
        radii=radii,
        inner_window=inner_window,
        outer_window=outer_window,
        axis=axis,
        coordinates=coordinates,
        cartesian_shape=radial_coordinate.shape,
        inside=inside,
        radius=radius,
        polar_samples=polar_samples,
        azimuthal_samples=azimuthal_samples,
        phi_padding=phi_padding,
        inside_indices=inside_indices,
        cartesian_radius_fraction=cartesian_radius_fraction,
        cartesian_window=cartesian_window,
    )


def _sample_mode_complex_field(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    derivative_order: int = 0,
) -> np.ndarray:
    if sampling.angular_sampling == "cartesian":
        return _sample_direct_cartesian_mode_field(
            sampling,
            frame_time=frame_time,
            max_delay=max_delay,
            derivative_order=derivative_order,
        )

    from scipy.ndimage import map_coordinates

    retarded_time = frame_time - sampling.radii * (
        max_delay / sampling.radius
    )
    retarded_modes = sampling.mode_spline(retarded_time, derivative_order)
    spherical_field = (retarded_modes @ sampling.ylms.T).reshape(
        sampling.radii.size,
        sampling.polar_samples,
        sampling.azimuthal_samples,
    )
    spherical_field *= sampling.inner_window
    pad = sampling.phi_padding
    spherical_field = np.concatenate(
        [
            spherical_field[:, :, -pad:],
            spherical_field,
            spherical_field[:, :, :pad],
        ],
        axis=2,
    )
    cartesian_field = _sample_complex_spherical_field(
        spherical_field,
        sampling.coordinates,
        sampling.cartesian_shape,
        map_coordinates,
    )
    return np.where(
        sampling.inside,
        cartesian_field * sampling.outer_window,
        0.0,
    )


def _sample_mode_power_field(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    derivative_order: int = 1,
) -> np.ndarray:
    """Sum modal powers without coherent cross terms.

    This is a phase-averaged display proxy: it computes
    ``sum_lm |d h_lm/dt * Y_lm|^2`` rather than
    ``|sum_lm d h_lm/dt * Y_lm|^2``.  It removes rotating interference lobes
    while retaining each mode's sky-dependent angular power.
    """

    if sampling.angular_sampling == "cartesian":
        return _sample_direct_cartesian_mode_power(
            sampling,
            frame_time=frame_time,
            max_delay=max_delay,
            derivative_order=derivative_order,
        )

    from scipy.ndimage import map_coordinates

    retarded_time = frame_time - sampling.radii * (
        max_delay / sampling.radius
    )
    retarded_modes = sampling.mode_spline(retarded_time, derivative_order)
    spherical_power = (
        np.abs(retarded_modes) ** 2 @ np.abs(sampling.ylms.T) ** 2
    ).reshape(
        sampling.radii.size,
        sampling.polar_samples,
        sampling.azimuthal_samples,
    )
    spherical_power *= sampling.inner_window**2
    pad = sampling.phi_padding
    spherical_power = np.concatenate(
        [
            spherical_power[:, :, -pad:],
            spherical_power,
            spherical_power[:, :, :pad],
        ],
        axis=2,
    )
    cartesian_power = map_coordinates(
        spherical_power,
        sampling.coordinates,
        order=3,
        mode="nearest",
        prefilter=True,
    ).reshape(sampling.cartesian_shape)
    return np.where(
        sampling.inside,
        np.maximum(cartesian_power, 0.0) * sampling.outer_window**2,
        0.0,
    )


def _volume_from_mode_sampling(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    normalize: bool,
    flux_mode_combination: FluxModeCombination,
) -> RetardedTimeVolume:
    strain = _sample_mode_complex_field(
        sampling, frame_time=frame_time, max_delay=max_delay
    )
    plus = np.real(strain)
    cross = -np.imag(strain)

    normalization = float(np.max(np.hypot(plus, cross)))
    if normalization == 0.0:
        raise ValueError("the mode waveform produces an all-zero volume")
    if normalize:
        plus /= normalization
        cross /= normalization
    else:
        normalization = 1.0
    amplitude = np.hypot(plus, cross)
    if flux_mode_combination == "incoherent":
        energy_flux = _sample_mode_power_field(
            sampling,
            frame_time=frame_time,
            max_delay=max_delay,
            derivative_order=1,
        )
        if normalize:
            energy_flux /= normalization**2
    else:
        derivative = _sample_mode_complex_field(
            sampling,
            frame_time=frame_time,
            max_delay=max_delay,
            derivative_order=1,
        )
        if normalize:
            derivative /= normalization
        energy_flux = np.abs(derivative) ** 2
    energy_flux_normalization = float(np.max(energy_flux))
    if normalize and energy_flux_normalization > 0.0:
        energy_flux /= energy_flux_normalization
    else:
        energy_flux_normalization = 1.0

    return RetardedTimeVolume(
        axis=sampling.axis,
        plus=plus,
        cross=cross,
        amplitude=amplitude,
        energy_flux=energy_flux,
        frame_time=frame_time,
        max_delay=max_delay,
        radius=sampling.radius,
        normalization=normalization,
        energy_flux_normalization=energy_flux_normalization,
        normalized=normalize,
        angular_pattern="mode_resolved",
        model=sampling.waveform.model,
        mode_count=sampling.waveform.modes.shape[1],
    )


def _component_from_mode_sampling(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    component: VolumeComponent,
    flux_mode_combination: FluxModeCombination = "coherent",
) -> np.ndarray:
    if component == "energy_flux" and flux_mode_combination == "incoherent":
        return _sample_mode_power_field(
            sampling,
            frame_time=frame_time,
            max_delay=max_delay,
            derivative_order=1,
        )
    derivative_order = 1 if component == "energy_flux" else 0
    field = _sample_mode_complex_field(
        sampling,
        frame_time=frame_time,
        max_delay=max_delay,
        derivative_order=derivative_order,
    )
    if component == "plus":
        return np.real(field)
    if component == "cross":
        return -np.imag(field)
    if component == "amplitude":
        return np.abs(field)
    if component == "energy_flux":
        return np.abs(field) ** 2
    raise ValueError(
        "component must be 'plus', 'cross', 'amplitude', or 'energy_flux'"
    )


def _validate_frame_window(
    time: np.ndarray, frame_time: float, max_delay: float
) -> None:
    if frame_time < time[0] or frame_time > time[-1]:
        raise ValueError("frame_time must fall within the waveform time range")
    if max_delay <= 0.0 or max_delay > frame_time - float(time[0]):
        raise ValueError(
            "max_delay must be positive and no larger than frame_time - time[0]"
        )


def build_mode_retarded_time_volume(
    waveform: RelativisticModeWaveform,
    *,
    frame_time: Optional[float] = None,
    max_delay: Optional[float] = None,
    resolution: int = 128,
    radius: float = 1.0,
    polar_samples: int = 80,
    azimuthal_samples: int = 160,
    angular_sampling: ModeAngularSampling = "cartesian",
    inner_window_fraction: float = 0.025,
    outer_window_fraction: float = 0.05,
    flux_mode_combination: FluxModeCombination = "coherent",
    normalize: bool = True,
) -> RetardedTimeVolume:
    r"""Map relativistic FEW modes into a spherical retarded-time volume.

    At every point, this evaluates

    .. math::

        h(t,r,\theta,\phi) = \sum_{\ell m} h_{\ell m}(t-r/c)
        {}_{-2}Y_{\ell m}(\theta,\phi).

    By default, the spin-weighted harmonics and retarded mode splines are
    evaluated directly at every in-sphere Cartesian voxel.  This avoids the
    azimuthal branch cut introduced by interpolating a spherical grid across
    :math:`\phi=0=2\pi`.  ``angular_sampling="spherical"`` retains the older,
    lower-memory interpolation path; ``polar_samples`` and
    ``azimuthal_samples`` apply only to that fallback.  A narrow smooth outer
    taper avoids exposing the staircase boundary of the Cartesian voxel grid.

    ``flux_mode_combination="coherent"`` returns the instantaneous
    :math:`|\sum_{\ell m}\dot h_{\ell m}Y_{\ell m}|^2` flux proxy.  The
    ``"incoherent"`` option drops cross terms and returns the smoother
    phase-averaged proxy
    :math:`\sum_{\ell m}|\dot h_{\ell m}Y_{\ell m}|^2`.
    """

    _validate_flux_mode_combination(flux_mode_combination)
    sampling = _prepare_mode_volume_sampling(
        waveform,
        resolution=resolution,
        radius=radius,
        polar_samples=polar_samples,
        azimuthal_samples=azimuthal_samples,
        inner_window_fraction=inner_window_fraction,
        outer_window_fraction=outer_window_fraction,
        angular_sampling=angular_sampling,
    )
    time = np.asarray(waveform.time, dtype=float)
    frame = float(time[-1] if frame_time is None else frame_time)
    available_delay = frame - float(time[0])
    delay_max = available_delay if max_delay is None else float(max_delay)
    _validate_frame_window(time, frame, delay_max)
    return _volume_from_mode_sampling(
        sampling,
        frame_time=frame,
        max_delay=delay_max,
        normalize=normalize,
        flux_mode_combination=flux_mode_combination,
    )


def estimate_waveform_period(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    frame_time: Optional[float] = None,
    analysis_fraction: float = 0.25,
) -> float:
    """Estimate the local polarization period near a requested frame.

    The estimate comes from the unwrapped phase of
    :math:`h_+ - i h_\times`. Low-amplitude samples and the most extreme phase
    increments are discarded, which makes the estimate robust to eccentric
    amplitude modulation.

    This period is primarily a display aid. It lets a volume show a controlled
    number of resolvable wavefronts instead of compressing an entire waveform
    into a small Cartesian grid.
    """

    if not 0.0 < analysis_fraction <= 1.0:
        raise ValueError("analysis_fraction must be in the interval (0, 1]")

    t, hp, hc, frame, available_delay, _ = _prepare_waveform(
        time,
        h_plus,
        h_cross,
        frame_time=frame_time,
        max_delay=None,
        normalize=False,
    )
    analysis_start = frame - analysis_fraction * available_delay
    selection = (t >= analysis_start) & (t <= frame)
    t_window = t[selection]
    strain = hp[selection] - 1j * hc[selection]
    if t_window.size < 4:
        raise ValueError("at least four waveform samples are needed")

    phase = np.unwrap(np.angle(strain))
    phase_rate = np.abs(np.diff(phase) / np.diff(t_window))
    midpoint_amplitude = 0.5 * (np.abs(strain[:-1]) + np.abs(strain[1:]))
    amplitude_floor = 0.15 * float(np.max(midpoint_amplitude))
    valid = (
        np.isfinite(phase_rate)
        & (phase_rate > np.finfo(float).eps)
        & (midpoint_amplitude >= amplitude_floor)
    )
    rates = phase_rate[valid]
    if rates.size < 3:
        raise ValueError("could not estimate a stable waveform period")

    lower, upper = np.quantile(rates, [0.1, 0.9])
    rates = rates[(rates >= lower) & (rates <= upper)]
    angular_frequency = float(np.median(rates))
    return 2.0 * np.pi / angular_frequency


def choose_max_delay(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    frame_time: Optional[float] = None,
    wave_cycles: float = 8.0,
) -> float:
    """Choose a display history containing a resolvable number of cycles."""

    if wave_cycles <= 0.0:
        raise ValueError("wave_cycles must be positive")
    t = _as_numpy(time).astype(float, copy=False)
    frame = float(t[-1] if frame_time is None else frame_time)
    available_delay = frame - float(t[0])
    period = estimate_waveform_period(
        time, h_plus, h_cross, frame_time=frame_time
    )
    minimum_delay = 2.0 * float(np.median(np.diff(t)))
    return min(available_delay, max(minimum_delay, wave_cycles * period))


def build_strain_surface(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    frame_time: Optional[float] = None,
    max_delay: Optional[float] = None,
    radial_samples: int = 320,
    angular_samples: int = 180,
    height: float = 0.12,
    radial_warp: float = 0.025,
    window_fraction: float = 0.025,
    normalize: bool = True,
) -> StrainSurface:
    """Build a lightweight 3D polar strain surface from a FEW waveform.

    Retarded time runs from the center (the current frame) to the edge (the
    oldest displayed sample).  The polarizations are rotated by twice the
    azimuthal angle, as required for a spin-2 field.

    Args:
        time: Waveform sample times in seconds.
        h_plus: Plus polarization samples.
        h_cross: Cross polarization samples.
        frame_time: Source time shown at the center. Defaults to the final
            waveform sample.
        max_delay: Waveform history shown from center to edge, in seconds.
            Defaults to all available history before ``frame_time``.
        radial_samples: Number of samples from center to edge.
        angular_samples: Number of azimuthal samples.
        height: Vertical display displacement in units of the outer radius.
        radial_warp: Radial display displacement in the same units.
        window_fraction: Fraction of the radius smoothly tapered at both ends.
        normalize: Divide both polarizations by their joint maximum amplitude.

    Returns:
        A :class:`StrainSurface` ready for :func:`plot_strain_surface`.
    """

    if radial_samples < 2:
        raise ValueError("radial_samples must be at least 2")
    if angular_samples < 4:
        raise ValueError("angular_samples must be at least 4")
    if height < 0.0 or radial_warp < 0.0:
        raise ValueError("height and radial_warp must be non-negative")

    t, hp, hc, frame, delay_max, normalization = _prepare_waveform(
        time,
        h_plus,
        h_cross,
        frame_time=frame_time,
        max_delay=max_delay,
        normalize=normalize,
    )

    delay = np.linspace(0.0, delay_max, radial_samples)
    radius = delay / delay_max
    phi = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=True)

    hp_retarded, hc_retarded = _sample_retarded(t, hp, hc, frame, delay)
    cos_2phi = np.cos(2.0 * phi)[None, :]
    sin_2phi = np.sin(2.0 * phi)[None, :]
    plus = hp_retarded[:, None] * cos_2phi - hc_retarded[:, None] * sin_2phi
    cross = hp_retarded[:, None] * sin_2phi + hc_retarded[:, None] * cos_2phi

    window = _radial_window(radius, window_fraction)[:, None]
    plus *= window
    cross *= window

    displaced_radius = radius[:, None] + radial_warp * plus
    x = displaced_radius * np.cos(phi)[None, :]
    y = displaced_radius * np.sin(phi)[None, :]
    z = height * plus

    return StrainSurface(
        x=x,
        y=y,
        z=z,
        plus=plus,
        cross=cross,
        delay=delay,
        frame_time=frame,
        max_delay=delay_max,
        normalization=normalization,
        normalized=normalize,
    )


def build_retarded_time_volume(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    frame_time: Optional[float] = None,
    max_delay: Optional[float] = None,
    resolution: int = 64,
    radius: float = 1.0,
    angular_pattern: AngularPattern = "quadrupole",
    window_fraction: float = 0.025,
    normalize: bool = True,
) -> RetardedTimeVolume:
    """Evaluate a time-domain waveform on a retarded-time 3D volume.

    The source is at the origin.  At display radius ``r``, the waveform is
    sampled at

    ``u = frame_time - (r / radius) * max_delay``.

    Args:
        time: Waveform sample times in seconds.
        h_plus: Plus polarization samples.
        h_cross: Cross polarization samples.
        frame_time: Source time at the origin. Defaults to the final sample.
        max_delay: Waveform history mapped to the outer radius, in seconds.
        resolution: Points per Cartesian dimension.
        radius: Outer display radius.
        angular_pattern: ``"line_of_sight"`` copies the observed waveform
            onto every sphere. ``"quadrupole"`` applies an illustrative
            dominant-quadrupole envelope and spin-2 polarization rotation.
        window_fraction: Fraction of the radius smoothly tapered at the center
            and outer boundary.
        normalize: Divide both polarizations by their joint maximum amplitude.

    Returns:
        A :class:`RetardedTimeVolume` containing plus, cross, amplitude, and a
        strain-rate-squared energy-flux proxy.
    """

    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if angular_pattern not in ("line_of_sight", "quadrupole"):
        raise ValueError("angular_pattern must be 'line_of_sight' or 'quadrupole'")

    t, hp, hc, frame, delay_max, normalization = _prepare_waveform(
        time,
        h_plus,
        h_cross,
        frame_time=frame_time,
        max_delay=max_delay,
        normalize=normalize,
    )

    axis = np.linspace(-radius, radius, resolution)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    radial_coordinate = np.sqrt(x * x + y * y + z * z)
    delay = radial_coordinate * (delay_max / radius)
    hp_retarded, hc_retarded = _sample_retarded(t, hp, hc, frame, delay.ravel())
    hp_retarded = hp_retarded.reshape(radial_coordinate.shape)
    hc_retarded = hc_retarded.reshape(radial_coordinate.shape)
    dhp = np.gradient(hp, t)
    dhc = np.gradient(hc, t)
    dhp_retarded, dhc_retarded = _sample_retarded(t, dhp, dhc, frame, delay.ravel())
    dhp_retarded = dhp_retarded.reshape(radial_coordinate.shape)
    dhc_retarded = dhc_retarded.reshape(radial_coordinate.shape)

    if angular_pattern == "quadrupole":
        safe_radius = np.where(radial_coordinate == 0.0, 1.0, radial_coordinate)
        cos_theta = np.broadcast_to(z / safe_radius, radial_coordinate.shape)
        phi = np.broadcast_to(np.arctan2(y, x), radial_coordinate.shape)
        envelope = 0.5 * (1.0 + cos_theta * cos_theta)
        cos_2phi = np.cos(2.0 * phi)
        sin_2phi = np.sin(2.0 * phi)
        cross_envelope = cos_theta
        plus = envelope * (hp_retarded * cos_2phi - hc_retarded * sin_2phi)
        cross = cross_envelope * (
            hp_retarded * sin_2phi + hc_retarded * cos_2phi
        )
        dplus = envelope * (dhp_retarded * cos_2phi - dhc_retarded * sin_2phi)
        dcross = cross_envelope * (
            dhp_retarded * sin_2phi + dhc_retarded * cos_2phi
        )
    else:
        plus = hp_retarded
        cross = hc_retarded
        dplus = dhp_retarded
        dcross = dhc_retarded

    normalized_radius = radial_coordinate / radius
    window = _radial_window(normalized_radius, window_fraction)
    window = np.where(normalized_radius <= 1.0, window, 0.0)
    plus *= window
    cross *= window
    dplus *= window
    dcross *= window
    amplitude = np.hypot(plus, cross)
    energy_flux = dplus * dplus + dcross * dcross
    energy_flux_normalization = float(np.max(energy_flux))
    if normalize and energy_flux_normalization > 0.0:
        energy_flux /= energy_flux_normalization
    else:
        energy_flux_normalization = 1.0

    return RetardedTimeVolume(
        axis=axis,
        plus=plus,
        cross=cross,
        amplitude=amplitude,
        energy_flux=energy_flux,
        frame_time=frame,
        max_delay=delay_max,
        radius=radius,
        normalization=normalization,
        energy_flux_normalization=energy_flux_normalization,
        normalized=normalize,
        angular_pattern=angular_pattern,
    )


def plot_strain_surface(
    surface: StrainSurface,
    *,
    ax=None,
    cmap: str = "coolwarm",
    colorbar: bool = True,
    view_elevation: float = 34.0,
    view_azimuth: float = -58.0,
):
    """Plot a :class:`StrainSurface` with Matplotlib.

    Returns:
        ``(figure, axes)`` so callers can further customize or save the plot.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if ax is None:
        figure = plt.figure(figsize=(9.0, 7.0))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure

    limit = _symmetric_limit(surface.plus)
    norm = Normalize(vmin=-limit, vmax=limit)
    colors = plt.get_cmap(cmap)(norm(surface.plus))
    ax.plot_surface(
        surface.x,
        surface.y,
        surface.z,
        facecolors=colors,
        linewidth=0,
        antialiased=True,
        shade=True,
        rcount=min(surface.x.shape[0], 260),
        ccount=min(surface.x.shape[1], 180),
    )
    ax.view_init(elev=view_elevation, azim=view_azimuth)
    ax.set_box_aspect((1.0, 1.0, 0.32))
    ax.set_xlabel(r"$x/r_{\mathrm{max}}$")
    ax.set_ylabel(r"$y/r_{\mathrm{max}}$")
    ax.set_zlabel("display displacement")
    ax.set_title(
        "FEW strain at retarded time "
        + rf"$u=t_{{\rm frame}}-r/c$ ({surface.max_delay:g} s history)"
    )

    if colorbar:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        label = r"normalized $h_+$" if surface.normalized else r"$h_+$"
        figure.colorbar(mappable, ax=ax, shrink=0.66, pad=0.08, label=label)

    return figure, ax


def plot_volume_slice(
    volume: RetardedTimeVolume,
    *,
    component: VolumeComponent = "plus",
    plane: Literal["xy", "xz", "yz"] = "xz",
    ax=None,
    cmap: str = "coolwarm",
    colorbar: bool = True,
):
    """Plot a central slice through a retarded-time volume with Matplotlib."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    field = volume.component(component)
    display_name = component.replace("_", " ")
    if plane == "xy":
        image = _zero_coordinate_plane(field, volume.axis, fixed_axis=2).T
        labels = (r"$x/r_{\mathrm{max}}$", r"$y/r_{\mathrm{max}}$")
    elif plane == "xz":
        image = _zero_coordinate_plane(field, volume.axis, fixed_axis=1).T
        labels = (r"$x/r_{\mathrm{max}}$", r"$z/r_{\mathrm{max}}$")
    else:
        image = _zero_coordinate_plane(field, volume.axis, fixed_axis=0).T
        labels = (r"$y/r_{\mathrm{max}}$", r"$z/r_{\mathrm{max}}$")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.2, 6.2))
    else:
        figure = ax.figure

    if component in ("amplitude", "energy_flux"):
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(image)), 1e-15))
    else:
        limit = _symmetric_limit(image)
        norm = Normalize(vmin=-limit, vmax=limit)

    artist = ax.imshow(
        image,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    quantity_name = (
        "Energy-flux proxy" if component == "energy_flux" else f"{display_name} strain"
    )
    ax.set_title(
        f"{quantity_name.capitalize()}, {plane} slice "
        + rf"($\Delta u={volume.max_delay:g}$ s)"
    )
    if colorbar:
        label = f"normalized {display_name}" if volume.normalized else display_name
        figure.colorbar(artist, ax=ax, shrink=0.82, label=label)
    return figure, ax


def _zero_coordinate_plane(
    field: np.ndarray,
    coordinates: np.ndarray,
    *,
    fixed_axis: int,
) -> np.ndarray:
    """Return a true coordinate-zero plane for odd or even voxel grids."""

    coordinate = np.asarray(coordinates, dtype=float)
    upper = int(np.searchsorted(coordinate, 0.0))
    if upper < coordinate.size and np.isclose(coordinate[upper], 0.0):
        return np.take(field, upper, axis=fixed_axis)
    if upper == 0 or upper == coordinate.size:
        raise ValueError("volume.axis must bracket zero")
    lower = upper - 1
    fraction = -coordinate[lower] / (coordinate[upper] - coordinate[lower])
    lower_plane = np.take(field, lower, axis=fixed_axis)
    upper_plane = np.take(field, upper, axis=fixed_axis)
    return (1.0 - fraction) * lower_plane + fraction * upper_plane


def to_pyvista(volume: RetardedTimeVolume):
    """Convert a volume to ``pyvista.ImageData``.

    PyVista is optional. Install the ``visualization`` extra to use this
    function.
    """

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "PyVista is required for volume rendering. Install FEW with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    grid = pv.ImageData()
    grid.dimensions = volume.plus.shape
    spacing = float(volume.axis[1] - volume.axis[0])
    grid.spacing = (spacing, spacing, spacing)
    grid.origin = (volume.axis[0], volume.axis[0], volume.axis[0])
    grid.point_data["plus"] = volume.plus.ravel(order="F")
    grid.point_data["cross"] = volume.cross.ravel(order="F")
    grid.point_data["amplitude"] = volume.amplitude.ravel(order="F")
    grid.point_data["energy_flux"] = volume.energy_flux.ravel(order="F")
    return grid


def _validate_volume_presentation(
    component: VolumeComponent,
    opacity_profile: VolumeOpacityProfile,
    camera_view: VolumeCameraView,
) -> None:
    if opacity_profile not in ("soft", "bands", "shells", "flux"):
        raise ValueError(
            "opacity_profile must be 'soft', 'bands', 'shells', or 'flux'"
        )
    if opacity_profile == "shells" and component not in ("plus", "cross"):
        raise ValueError("the shells opacity profile requires plus or cross strain")
    if opacity_profile == "flux" and component != "energy_flux":
        raise ValueError("the flux opacity profile requires energy_flux")
    if camera_view not in ("oblique", "face_on"):
        raise ValueError("camera_view must be 'oblique' or 'face_on'")


def _validate_flux_mode_combination(value: FluxModeCombination) -> None:
    if value not in ("coherent", "incoherent"):
        raise ValueError(
            "flux_mode_combination must be 'coherent' or 'incoherent'"
        )


def _resolve_volume_presentation(
    presentation: VolumePresentation,
    *,
    color_exposure: Optional[float],
    background_color: Optional[str],
    camera_zoom: Optional[float],
    star_count: Optional[int],
    starfield: Optional[bool],
) -> _ResolvedVolumePresentation:
    """Resolve a restrained or publication-oriented volume presentation."""

    if presentation not in ("balanced", "dramatic", "shells_dramatic"):
        raise ValueError(
            "presentation must be 'balanced', 'dramatic', or 'shells_dramatic'"
        )
    if presentation == "shells_dramatic":
        defaults = dict(
            color_exposure=1.30,
            background_color="#010522",
            camera_zoom=1.12,
            star_count=6500,
            starfield=True,
            ambient=0.94,
            diffuse=0.26,
            specular=0.05,
            shade=False,
        )
    elif presentation == "dramatic":
        defaults = dict(
            color_exposure=2.25,
            background_color="#01051b",
            camera_zoom=1.10,
            star_count=4000,
            starfield=True,
            ambient=1.00,
            diffuse=0.22,
            specular=0.04,
            shade=True,
        )
    else:
        defaults = dict(
            color_exposure=1.0,
            background_color="#000000",
            camera_zoom=1.0,
            star_count=620,
            starfield=False,
            ambient=0.72,
            diffuse=0.34,
            specular=0.03,
            shade=True,
        )

    resolved_exposure = float(
        defaults["color_exposure"] if color_exposure is None else color_exposure
    )
    resolved_background = str(
        defaults["background_color"]
        if background_color is None
        else background_color
    )
    resolved_zoom = float(
        defaults["camera_zoom"] if camera_zoom is None else camera_zoom
    )
    resolved_stars = int(
        defaults["star_count"] if star_count is None else star_count
    )
    resolved_starfield = bool(
        defaults["starfield"] if starfield is None else starfield
    )
    if resolved_exposure <= 0.0:
        raise ValueError("color_exposure must be positive")
    if resolved_zoom <= 0.0:
        raise ValueError("camera_zoom must be positive")
    if resolved_stars < 0:
        raise ValueError("star_count must be non-negative")
    from matplotlib.colors import is_color_like

    if not is_color_like(resolved_background):
        raise ValueError("background_color must be a valid Matplotlib colour")
    return _ResolvedVolumePresentation(
        color_exposure=resolved_exposure,
        background_color=resolved_background,
        camera_zoom=resolved_zoom,
        star_count=resolved_stars,
        starfield=resolved_starfield,
        ambient=float(defaults["ambient"]),
        diffuse=float(defaults["diffuse"]),
        specular=float(defaults["specular"]),
        shade=bool(defaults["shade"]),
    )


def render_volume(
    volume: RetardedTimeVolume,
    *,
    component: VolumeComponent = "plus",
    screenshot: Optional[PathLike] = None,
    show: bool = True,
    style: VolumeRenderStyle = "cinematic",
    opacity_profile: VolumeOpacityProfile = "soft",
    color_scheme: VolumeColorScheme = "magma",
    presentation: VolumePresentation = "balanced",
    color_exposure: Optional[float] = None,
    background_color: Optional[str] = None,
    opacity: float = 0.11,
    shell_count: int = 7,
    shell_min: float = 0.10,
    shell_max: float = 0.92,
    shell_width: float = 0.075,
    shell_opacity_floor: float = 0.16,
    shell_glow: float = 0.12,
    smooth_sigma: float = 0.65,
    opacity_unit_distance: Optional[float] = None,
    window_size: tuple[int, int] = (1600, 900),
    image_scale: int = 1,
    camera_view: VolumeCameraView = "oblique",
    camera_zoom: Optional[float] = None,
    starfield: Optional[bool] = None,
    star_count: Optional[int] = None,
    source_marker: bool = False,
    show_scalar_bar: bool = False,
):
    """Render a retarded-time volume using PyVista's GPU ray caster.

    The default ``"cinematic"`` style uses direct shaded volume rendering,
    smooth scalar interpolation, a dark background, and a presentation camera.
    ``opacity_profile="shells"`` exposes positive signed-strain crests in narrow
    coloured bands. ``opacity_profile="flux"`` logarithmically compresses the
    energy-flux proxy and gives it a broad opacity ramp. Unlike a stack of
    extracted contour meshes, the translucent wavefronts remain continuous
    when viewed obliquely.

    Args:
        volume: Volume returned by :func:`build_retarded_time_volume`.
        component: Scalar field to render.
        screenshot: Optional PNG path.
        show: Open the interactive PyVista window.
        style: ``"cinematic"`` for ray-cast volume rendering or ``"contours"``
            for the former diagnostic isosurface view.
        opacity_profile: ``"soft"`` or ``"bands"`` for the generic mappings,
            ``"shells"`` for positive signed-strain bands, or ``"flux"`` for a
            logarithmically compressed energy-flux volume.
        color_scheme: Magma-based or alternative Matplotlib colour palette.
        presentation: ``"balanced"`` preserves the restrained scientific
            styling; ``"dramatic"`` adds magma-oriented glow;
            ``"shells_dramatic"`` uses a brighter rainbow-oriented exposure,
            deeper navy, denser stars, and a slightly closer camera.
        color_exposure: Colour-transfer exposure independent of opacity. Values
            above one brighten the palette without making the volume more solid.
        background_color: Optional Matplotlib-compatible render background.
        opacity: Maximum opacity in the transfer function.
        shell_count: Number of translucent positive-strain level sheets.
        shell_min: Normalized strain at the first shell.
        shell_max: Normalized strain at the final shell.
        shell_width: Normalized decay width behind each shell.
        shell_opacity_floor: First-shell opacity as a fraction of ``opacity``.
        shell_glow: Fractional soft halo added around each shell.
        smooth_sigma: Gaussian smoothing in voxel units for display only.
        opacity_unit_distance: Physical distance over which opacity accumulates.
            The default is four percent of the volume radius, so changing the
            Cartesian grid resolution does not change the apparent solidity.
        window_size: Render-window dimensions in pixels.
        image_scale: Integer screenshot supersampling factor.
        camera_view: ``"oblique"`` for the presentation view or ``"face_on"``
            to look down the spin-frame axis. The face-on view collapses the
            unavoidable signed-polarization chart axis onto the source marker.
        camera_zoom: Optional camera zoom multiplier.
        starfield: Add a deterministic background star field. The dramatic
            preset enables it unless explicitly disabled.
        star_count: Number of deterministic stars distributed around the scene.
        source_marker: Add a small dark sphere at the source.
        show_scalar_bar: Include a scalar bar in the render.

    Returns:
        The configured ``pyvista.Plotter``.
    """

    if not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in the interval (0, 1]")
    if style not in ("cinematic", "contours"):
        raise ValueError("style must be 'cinematic' or 'contours'")
    _validate_volume_presentation(component, opacity_profile, camera_view)
    if color_scheme not in _VOLUME_COLOR_SCHEMES:
        raise ValueError(
            "unsupported volume color_scheme; expected one of "
            + ", ".join(sorted(_VOLUME_COLOR_SCHEMES))
        )
    resolved_presentation = _resolve_volume_presentation(
        presentation,
        color_exposure=color_exposure,
        background_color=background_color,
        camera_zoom=camera_zoom,
        star_count=star_count,
        starfield=starfield,
    )
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative")
    resolved_opacity_distance = _resolve_opacity_unit_distance(
        volume.radius, opacity_unit_distance
    )
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("window_size must contain two positive integers")
    if image_scale < 1:
        raise ValueError("image_scale must be at least 1")
    grid = to_pyvista(volume)
    try:
        import pyvista as pv
    except ImportError:  # pragma: no cover - guarded by to_pyvista
        raise

    field = volume.component(component)
    maximum = float(np.max(np.abs(field)))
    if maximum == 0.0:
        raise ValueError("cannot render an all-zero field")

    plotter = pv.Plotter(off_screen=not show, window_size=window_size)
    plotter.set_background(resolved_presentation.background_color)
    plotter.enable_anti_aliasing("ssaa")
    render_name = None
    volume_actor = None

    if style == "contours":
        _add_contour_render(
            plotter,
            grid,
            field,
            component=component,
            color_scheme=color_scheme,
            color_exposure=resolved_presentation.color_exposure,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
        )
    else:
        display_field = _smooth_render_field(field, smooth_sigma)
        display_field, clim = _normalize_render_field(
            display_field,
            component,
            opacity_profile=opacity_profile,
        )
        render_name = f"{component}_render"
        grid.point_data[render_name] = display_field.ravel(order="F")
        n_colors = 512
        opacity_transfer = _wavefront_opacity_transfer(
            component,
            n_colors=n_colors,
            maximum_opacity=opacity,
            profile=opacity_profile,
            shell_count=shell_count,
            shell_min=shell_min,
            shell_max=shell_max,
            shell_width=shell_width,
            shell_opacity_floor=shell_opacity_floor,
            shell_glow=shell_glow,
        )
        actor = plotter.add_volume(
            grid,
            scalars=render_name,
            clim=clim,
            cmap=_volume_colormap(
                component,
                color_scheme,
                color_exposure=resolved_presentation.color_exposure,
            ),
            opacity=opacity_transfer,
            n_colors=n_colors,
            blending="composite",
            mapper="gpu",
            opacity_unit_distance=resolved_opacity_distance,
            shade=resolved_presentation.shade,
            ambient=resolved_presentation.ambient,
            diffuse=resolved_presentation.diffuse,
            specular=resolved_presentation.specular,
            specular_power=12.0,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": component.replace("_", " ")},
        )
        actor.prop.interpolation_type = "linear"
        # VTK's GPU ray caster defaults to a sample distance of 1.0 *world*
        # unit while this volume spans only 2.0, so it takes about two samples
        # per ray, misses the tapered outer shell, and clips the ball into a
        # lopsided blob. Deriving the step from the grid spacing keeps the
        # sampling correct at any ``resolution``.
        actor.mapper.SetLockSampleDistanceToInputSpacing(1)
        volume_actor = actor

    if resolved_presentation.starfield and resolved_presentation.star_count > 0:
        _add_starfield(
            plotter,
            volume.radius,
            count=resolved_presentation.star_count,
        )
    if source_marker:
        source = pv.Sphere(
            radius=0.042 * volume.radius,
            theta_resolution=64,
            phi_resolution=64,
        )
        plotter.add_mesh(
            source,
            color="#010104",
            smooth_shading=True,
            ambient=0.04,
            diffuse=0.22,
            specular=0.95,
            specular_power=80.0,
        )

    r = volume.radius
    if camera_view == "face_on":
        plotter.camera_position = [
            (0.0, 0.0, 4.75 * r),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    else:
        plotter.camera_position = [
            (2.00 * r, -4.20 * r, 1.00 * r),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
    plotter.camera.parallel_projection = False
    plotter.camera.view_angle = 26.5
    plotter.camera.zoom(resolved_presentation.camera_zoom)

    screenshot_path = None if screenshot is None else str(Path(screenshot))
    if show:
        plotter.show(auto_close=False)
        if screenshot_path is not None:
            plotter.screenshot(screenshot_path, scale=image_scale)
    elif screenshot_path is not None:
        plotter.show(auto_close=False)
        plotter.screenshot(screenshot_path, scale=image_scale)
    plotter._fewview_volume_grid = grid
    plotter._fewview_render_name = render_name
    plotter._fewview_volume_actor = volume_actor
    return plotter


def _prepare_display_trajectory(
    waveform: RelativisticModeWaveform,
    *,
    radius: float,
    orbit_display_radius: float,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, float, float]:
    """Scale FEW's central orbit for visibility and estimate its period.

    The returned scale converts a Boyer--Lindquist radius in ``M`` into display
    units, so the bodies can be drawn on the same scale as the orbit.
    """

    trajectory = waveform.orbital_position()
    physical_radius = np.linalg.norm(trajectory[:, :2], axis=1)
    reference_radius = float(np.quantile(physical_radius, 0.995))
    if reference_radius <= 0.0:
        raise ValueError("the FEW trajectory has zero orbital radius")
    display_scale = orbit_display_radius * radius / reference_radius
    display = trajectory * display_scale

    time = np.asarray(waveform.time, dtype=float)
    phi = np.unwrap(np.asarray(waveform.trajectory_phi_phi, dtype=float))
    omega = np.abs(np.gradient(phi, time))
    window = (time >= start_time) & (time <= end_time) & (omega > 0.0)
    usable = omega[window] if np.any(window) else omega[omega > 0.0]
    if usable.size == 0:
        orbital_period = end_time - start_time
    else:
        orbital_period = 2.0 * np.pi / float(np.median(usable))
    return display, orbital_period, display_scale


def _interpolate_trajectory_position(
    time: np.ndarray, trajectory: np.ndarray, frame_time: float
) -> np.ndarray:
    return np.asarray(
        [np.interp(frame_time, time, trajectory[:, index]) for index in range(3)]
    )


def _trajectory_tail_points(
    time: np.ndarray,
    trajectory: np.ndarray,
    *,
    frame_time: float,
    duration: float,
) -> np.ndarray:
    mask = (time >= frame_time - duration) & (time <= frame_time)
    points = trajectory[mask]
    current = _interpolate_trajectory_position(time, trajectory, frame_time)
    if points.size == 0 or not np.allclose(points[-1], current):
        points = np.vstack((points, current))
    if len(points) == 1:
        points = np.vstack((points, points))
    return points


def _trajectory_rgb(color: str) -> np.ndarray:
    """Return a trajectory colour as 0-255 RGB.

    Any Matplotlib colour specification is accepted, so ``"#4dc8ff"``,
    ``"deepskyblue"`` and ``"tab:blue"`` all work.
    """

    from matplotlib.colors import to_rgb

    try:
        rgb = to_rgb(color)
    except ValueError as error:
        raise ValueError(
            f"trajectory_color {color!r} is not a valid colour"
        ) from error
    return np.rint(255.0 * np.asarray(rgb)).astype(np.uint8)


def _set_polyline_points(
    polyline, points: np.ndarray, *, color: str = DEFAULT_TRAJECTORY_COLOR
) -> None:
    polyline.points = points
    polyline.lines = np.concatenate(
        ([len(points)], np.arange(len(points), dtype=np.int64))
    )
    rgba = np.empty((len(points), 4), dtype=np.uint8)
    rgba[:, :3] = _trajectory_rgb(color)
    age = np.linspace(0.0, 1.0, len(points))
    rgba[:, 3] = np.rint(255.0 * (0.06 + 0.88 * age**0.7)).astype(np.uint8)
    polyline.point_data["trajectory_rgba"] = rgba
    polyline.Modified()


def _resolve_body_radii(
    waveform: RelativisticModeWaveform,
    *,
    radius: float,
    display_scale: float,
    display_trajectory: np.ndarray,
    primary_body_radius: float,
    secondary_body_radius: float,
    body_exaggeration: float,
) -> tuple[float, float]:
    r"""Return the drawn primary and secondary radii in display units.

    The orbit is drawn in rescaled ``M``, so whenever the spin is known the
    primary is drawn on that same scale, as ``body_exaggeration`` horizon radii
    :math:`r_+`, and the secondary as a fixed fraction of the primary.  Both
    then track the orbit under any change of :math:`(a,p,e)` instead of staying
    pinned to the unrelated wave-sphere radius.  A waveform without a spin falls
    back to the legacy wave-sphere fractions.
    """

    horizon = waveform.horizon_radius
    if horizon is None:
        primary_radius = primary_body_radius * radius
        secondary_radius = secondary_body_radius * radius
    else:
        primary_radius = body_exaggeration * horizon * display_scale
        secondary_radius = _SECONDARY_BODY_SCALE * primary_radius

    # r_+ always sits well inside periapsis, but an exaggerated r_+ need not, so
    # keep a visible gap between the halo and the secondary at closest approach.
    display_periapsis = float(
        np.min(np.linalg.norm(display_trajectory[:, :2], axis=1))
    )
    occupied = _PRIMARY_HALO_SCALE * primary_radius + secondary_radius
    clearance = _BODY_CLEARANCE_FRACTION * display_periapsis
    if occupied > clearance:
        primary_radius *= clearance / occupied
        secondary_radius *= clearance / occupied
    return primary_radius, secondary_radius


def _add_source_overlays(
    plotter,
    waveform: RelativisticModeWaveform,
    *,
    frame_time: float,
    radius: float,
    start_time: float,
    end_time: float,
    show_bodies: bool,
    show_trajectory: bool,
    trajectory_tail_cycles: float,
    trajectory_line_width: float,
    trajectory_as_tube: bool,
    trajectory_color: str,
    orbit_display_radius: float,
    primary_body_radius: float,
    secondary_body_radius: float,
    body_exaggeration: float,
):
    """Add the shared bodies and fading FEW trajectory to a VTK scene."""

    if not (show_bodies or show_trajectory):
        return None, None, None, None, None

    import pyvista as pv

    trajectory_time = np.asarray(waveform.time, dtype=float)
    display_trajectory, orbital_period, display_scale = _prepare_display_trajectory(
        waveform,
        radius=radius,
        orbit_display_radius=orbit_display_radius,
        start_time=start_time,
        end_time=end_time,
    )
    trajectory_tail_duration = trajectory_tail_cycles * orbital_period
    current_position = _interpolate_trajectory_position(
        trajectory_time, display_trajectory, frame_time
    )

    trajectory_polyline = None
    if show_trajectory:
        tail = _trajectory_tail_points(
            trajectory_time,
            display_trajectory,
            frame_time=frame_time,
            duration=trajectory_tail_duration,
        )
        trajectory_polyline = pv.PolyData()
        _set_polyline_points(trajectory_polyline, tail, color=trajectory_color)
        plotter.add_mesh(
            trajectory_polyline,
            scalars="trajectory_rgba",
            rgba=True,
            opacity=1.0,
            line_width=trajectory_line_width,
            render_lines_as_tubes=trajectory_as_tube,
            lighting=False,
        )

    secondary_actor = None
    if show_bodies:
        primary_radius, secondary_radius = _resolve_body_radii(
            waveform,
            radius=radius,
            display_scale=display_scale,
            display_trajectory=display_trajectory,
            primary_body_radius=primary_body_radius,
            secondary_body_radius=secondary_body_radius,
            body_exaggeration=body_exaggeration,
        )
        primary_halo = pv.Sphere(
            radius=_PRIMARY_HALO_SCALE * primary_radius,
            theta_resolution=64,
            phi_resolution=64,
        )
        plotter.add_mesh(
            primary_halo,
            color="#4dc8ff",
            opacity=0.16,
            smooth_shading=True,
            ambient=0.9,
            diffuse=0.1,
            specular=0.0,
        )
        primary = pv.Sphere(
            radius=primary_radius,
            theta_resolution=80,
            phi_resolution=80,
        )
        plotter.add_mesh(
            primary,
            color="#010207",
            smooth_shading=True,
            ambient=0.06,
            diffuse=0.20,
            specular=1.0,
            specular_power=95.0,
        )
        secondary = pv.Sphere(
            radius=secondary_radius,
            theta_resolution=64,
            phi_resolution=64,
        )
        secondary_actor = plotter.add_mesh(
            secondary,
            color="#fff3bd",
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.70,
            specular=1.0,
            specular_power=70.0,
        )
        secondary_actor.position = tuple(current_position)

    return (
        trajectory_time,
        display_trajectory,
        trajectory_tail_duration,
        secondary_actor,
        trajectory_polyline,
    )


class _WaveformPanelRenderer:
    """Small Agg-rendered waveform strip composited below a VTK frame."""

    def __init__(
        self,
        waveform: RelativisticModeWaveform,
        *,
        start_time: float,
        end_time: float,
        width: int,
        height: int,
        theta: float,
        phi: float,
        font_style: WaveformFontStyle,
        style_file: Optional[PathLike],
        background_color: str,
    ) -> None:
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.font_style = font_style
        self.style_file = None if style_file is None else str(Path(style_file))
        self.font_rc = (
            {
                "font.family": "serif",
                "font.serif": ["STIXGeneral"],
                "mathtext.fontset": "stix",
                "mathtext.default": "regular",
                "text.usetex": False,
            }
            if font_style == "latex"
            else {"text.usetex": False}
        )

        reference = waveform.strain(theta=theta, phi=phi)
        h_plus = np.real(reference)
        time = np.asarray(waveform.time, dtype=float)
        mask = (time >= start_time) & (time <= end_time)
        if np.count_nonzero(mask) < 2:
            raise ValueError("the waveform panel needs at least two time samples")
        self.time = time[mask]
        self.relative_time = self.time - start_time
        scale = float(np.max(np.abs(h_plus[mask])))
        self.h_plus = h_plus[mask] / max(scale, np.finfo(float).tiny)
        self.start_time = start_time

        with self._style_context(mpl):
            dpi = 100.0
            figure = Figure(
                figsize=(width / dpi, height / dpi),
                dpi=dpi,
                facecolor=background_color,
            )
            self.canvas = FigureCanvasAgg(figure)
            self.axis = figure.add_axes((0.055, 0.27, 0.92, 0.66))
            self.axis.set_facecolor(background_color)
            self.axis.axhline(0.0, color="#20313a", linewidth=0.7, zorder=0)
            self.axis.plot(
                self.relative_time,
                self.h_plus,
                color="#245464",
                linewidth=0.85,
                alpha=0.72,
            )
            (self.active_line,) = self.axis.plot(
                [], [], color="#7cecff", linewidth=1.35
            )
            self.marker = self.axis.axvline(
                0.0, color="#ffd66d", linewidth=1.1, alpha=0.95
            )
            self.time_label = self.axis.text(
                0.995,
                0.88,
                "",
                transform=self.axis.transAxes,
                color="#dfeef2",
                ha="right",
                va="top",
                fontsize=8.5,
            )
            waveform_label = (
                r"$h_+\;\mathrm{(normalised)}$"
                if font_style == "latex"
                else r"$h_+$  (normalised)"
            )
            self.axis.text(
                0.005,
                0.88,
                waveform_label,
                transform=self.axis.transAxes,
                color="#dfeef2",
                ha="left",
                va="top",
                fontsize=8.5,
            )
            self.axis.set_xlim(0.0, end_time - start_time)
            self.axis.set_ylim(-1.12, 1.12)
            self.axis.set_yticks([])
            xlabel = (
                r"$t \;[\mathrm{s}]$"
                if font_style == "latex"
                else "source time from animation start  [s]"
            )
            self.axis.set_xlabel(
                xlabel,
                color="#91a8af",
                fontsize=8,
                labelpad=2,
            )
            self.axis.tick_params(
                axis="x",
                colors="#718991",
                labelsize=7.5,
                length=2.5,
                width=0.6,
            )
            for side in ("left", "right", "top"):
                self.axis.spines[side].set_visible(False)
            self.axis.spines["bottom"].set_color("#38515a")
            self.axis.spines["bottom"].set_linewidth(0.7)

    def _style_context(self, mpl):
        if self.style_file is not None:
            from matplotlib import style

            return style.context(self.style_file)
        return mpl.rc_context(self.font_rc)

    def render(self, frame_time: float) -> np.ndarray:
        import matplotlib as mpl

        relative = frame_time - self.start_time
        mask = self.relative_time <= relative
        self.active_line.set_data(self.relative_time[mask], self.h_plus[mask])
        self.marker.set_xdata([relative, relative])
        if self.font_style == "latex":
            self.time_label.set_text(rf"$t = {relative:.1f}\,\mathrm{{s}}$")
        else:
            self.time_label.set_text(f"t = {relative:,.1f} s")
        with self._style_context(mpl):
            self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def render_mode_frame(
    waveform: RelativisticModeWaveform,
    screenshot: PathLike,
    *,
    max_delay: float,
    frame_time: Optional[float] = None,
    component: VolumeComponent = "plus",
    flux_mode_combination: FluxModeCombination = "coherent",
    waveform_start_time: Optional[float] = None,
    waveform_end_time: Optional[float] = None,
    resolution: int = 96,
    radius: float = 1.0,
    polar_samples: int = 48,
    azimuthal_samples: int = 96,
    angular_sampling: ModeAngularSampling = "cartesian",
    inner_window_fraction: float = 0.025,
    outer_window_fraction: float = 0.05,
    opacity_profile: VolumeOpacityProfile = "soft",
    color_scheme: VolumeColorScheme = "magma",
    presentation: VolumePresentation = "balanced",
    color_exposure: Optional[float] = None,
    background_color: Optional[str] = None,
    opacity: float = 0.10,
    shell_count: int = 7,
    shell_min: float = 0.10,
    shell_max: float = 0.92,
    shell_width: float = 0.075,
    shell_opacity_floor: float = 0.16,
    shell_glow: float = 0.12,
    smooth_sigma: float = 0.65,
    opacity_unit_distance: Optional[float] = None,
    window_size: tuple[int, int] = (1280, 720),
    image_scale: int = 1,
    camera_view: VolumeCameraView = "oblique",
    camera_zoom: Optional[float] = None,
    starfield: Optional[bool] = None,
    star_count: Optional[int] = None,
    source_marker: bool = False,
    show_bodies: bool = True,
    show_trajectory: bool = True,
    show_waveform: bool = True,
    trajectory_tail_cycles: float = 2.0,
    trajectory_line_width: float = 1.6,
    trajectory_color: str = DEFAULT_TRAJECTORY_COLOR,
    trajectory_as_tube: bool = False,
    orbit_display_radius: float = 0.16,
    primary_body_radius: float = 0.043,
    secondary_body_radius: float = 0.014,
    body_exaggeration: float = 2.0,
    waveform_fraction: float = 0.22,
    waveform_theta: float = np.pi / 3.0,
    waveform_phi: float = 0.0,
    waveform_font_style: WaveformFontStyle = "latex",
    waveform_style_file: Optional[PathLike] = None,
) -> Path:
    """Render one full-sky mode frame with the animation presentation layers.

    For ``component="energy_flux"``, ``flux_mode_combination="coherent"``
    retains instantaneous interference lobes. Use ``"incoherent"`` for an
    explicitly phase-averaged display proxy.
    """

    output = Path(screenshot)
    _validate_volume_presentation(component, opacity_profile, camera_view)
    _validate_flux_mode_combination(flux_mode_combination)
    resolved_presentation = _resolve_volume_presentation(
        presentation,
        color_exposure=color_exposure,
        background_color=background_color,
        camera_zoom=camera_zoom,
        star_count=star_count,
        starfield=starfield,
    )
    if output.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("screenshot filename must end in .png, .jpg, or .jpeg")
    if image_scale < 1:
        raise ValueError("image_scale must be at least 1")
    if not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in the interval (0, 1]")
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative")
    if trajectory_tail_cycles <= 0.0:
        raise ValueError("trajectory_tail_cycles must be positive")
    if trajectory_line_width <= 0.0:
        raise ValueError("trajectory_line_width must be positive")
    _trajectory_rgb(trajectory_color)
    if not 0.0 < orbit_display_radius < 0.5:
        raise ValueError("orbit_display_radius must be in the interval (0, 0.5)")
    if primary_body_radius <= 0.0 or secondary_body_radius <= 0.0:
        raise ValueError("body radii must be positive")
    if body_exaggeration <= 0.0:
        raise ValueError("body_exaggeration must be positive")
    if show_waveform and not 0.1 <= waveform_fraction < 0.5:
        raise ValueError("waveform_fraction must be in the interval [0.1, 0.5)")
    if (show_bodies or show_trajectory) and not waveform.has_trajectory:
        raise ValueError(
            "Body and trajectory rendering requires the FEW trajectory arrays. "
            "Regenerate relativistic-modes.npz with visualize_emri_waveform.py."
        )

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Still export requires imageio. Install FEW with "
            "`pip install -e '.[visualization]'`."
        ) from exc

    time = np.asarray(waveform.time, dtype=float)
    frame = float(time[-1] if frame_time is None else frame_time)
    _validate_frame_window(time, frame, max_delay)
    panel_start = float(
        time[0] + max_delay
        if waveform_start_time is None
        else waveform_start_time
    )
    panel_end = float(time[-1] if waveform_end_time is None else waveform_end_time)
    if panel_start < time[0] or panel_end > time[-1] or panel_end <= panel_start:
        raise ValueError("the waveform panel interval must lie inside waveform.time")
    if not panel_start <= frame <= panel_end:
        raise ValueError("frame_time must lie inside the waveform panel interval")

    total_width, total_height = window_size
    panel_height = (
        int(round(total_height * waveform_fraction / 2.0) * 2)
        if show_waveform
        else 0
    )
    main_window_size = (total_width, total_height - panel_height)
    if min(main_window_size) <= 0:
        raise ValueError("waveform_fraction leaves no space for the volume")

    sampling = _prepare_mode_volume_sampling(
        waveform,
        resolution=resolution,
        radius=radius,
        polar_samples=polar_samples,
        azimuthal_samples=azimuthal_samples,
        inner_window_fraction=inner_window_fraction,
        outer_window_fraction=outer_window_fraction,
        angular_sampling=angular_sampling,
    )
    field = _component_from_mode_sampling(
        sampling,
        frame_time=frame,
        max_delay=max_delay,
        component=component,
        flux_mode_combination=flux_mode_combination,
    )
    zero = np.zeros_like(field)
    component_fields = {
        "plus": zero.copy(),
        "cross": zero.copy(),
        "amplitude": zero.copy(),
        "energy_flux": zero.copy(),
    }
    component_fields[component] = field
    volume = RetardedTimeVolume(
        axis=sampling.axis,
        plus=component_fields["plus"],
        cross=component_fields["cross"],
        amplitude=component_fields["amplitude"],
        energy_flux=component_fields["energy_flux"],
        frame_time=frame,
        max_delay=max_delay,
        radius=radius,
        normalization=1.0,
        energy_flux_normalization=1.0,
        normalized=False,
        angular_pattern="mode_resolved",
        model=waveform.model,
        mode_count=waveform.modes.shape[1],
    )
    plotter = render_volume(
        volume,
        component=component,
        show=False,
        opacity_profile=opacity_profile,
        color_scheme=color_scheme,
        presentation=presentation,
        color_exposure=color_exposure,
        background_color=background_color,
        opacity=opacity,
        shell_count=shell_count,
        shell_min=shell_min,
        shell_max=shell_max,
        shell_width=shell_width,
        shell_opacity_floor=shell_opacity_floor,
        shell_glow=shell_glow,
        smooth_sigma=smooth_sigma,
        opacity_unit_distance=opacity_unit_distance,
        window_size=main_window_size,
        camera_view=camera_view,
        camera_zoom=camera_zoom,
        starfield=starfield,
        star_count=star_count,
        source_marker=source_marker and not show_bodies,
    )
    _add_source_overlays(
        plotter,
        waveform,
        frame_time=frame,
        radius=radius,
        start_time=panel_start,
        end_time=panel_end,
        show_bodies=show_bodies,
        show_trajectory=show_trajectory,
        trajectory_tail_cycles=trajectory_tail_cycles,
        trajectory_line_width=trajectory_line_width,
        trajectory_color=trajectory_color,
        trajectory_as_tube=trajectory_as_tube,
        orbit_display_radius=orbit_display_radius,
        primary_body_radius=primary_body_radius,
        secondary_body_radius=secondary_body_radius,
        body_exaggeration=body_exaggeration,
    )
    waveform_panel = (
        _WaveformPanelRenderer(
            waveform,
            start_time=panel_start,
            end_time=panel_end,
            width=total_width * image_scale,
            height=panel_height * image_scale,
            theta=waveform_theta,
            phi=waveform_phi,
            font_style=waveform_font_style,
            style_file=waveform_style_file,
            background_color=resolved_presentation.background_color,
        )
        if show_waveform
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.show(auto_close=False)
        plotter.render()
        image = np.asarray(
            plotter.screenshot(return_img=True, scale=image_scale)
        )[..., :3]
        if waveform_panel is not None:
            panel = waveform_panel.render(frame)
            if panel.shape[1] != image.shape[1]:
                raise RuntimeError("waveform panel width does not match VTK render")
            image = np.vstack((image, panel))
        imageio.imwrite(output, image)
    finally:
        plotter.close()
    return output


def render_mode_animation(
    waveform: RelativisticModeWaveform,
    filename: PathLike,
    *,
    max_delay: float,
    component: VolumeComponent = "plus",
    flux_mode_combination: FluxModeCombination = "coherent",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    frames: int = 120,
    fps: int = 30,
    resolution: int = 72,
    radius: float = 1.0,
    polar_samples: int = 48,
    azimuthal_samples: int = 96,
    angular_sampling: ModeAngularSampling = "cartesian",
    inner_window_fraction: float = 0.025,
    outer_window_fraction: float = 0.05,
    opacity_profile: VolumeOpacityProfile = "soft",
    color_scheme: VolumeColorScheme = "magma",
    presentation: VolumePresentation = "balanced",
    color_exposure: Optional[float] = None,
    background_color: Optional[str] = None,
    opacity: float = 0.10,
    shell_count: int = 7,
    shell_min: float = 0.10,
    shell_max: float = 0.92,
    shell_width: float = 0.075,
    shell_opacity_floor: float = 0.16,
    shell_glow: float = 0.12,
    smooth_sigma: float = 0.65,
    opacity_unit_distance: Optional[float] = None,
    window_size: tuple[int, int] = (1280, 720),
    image_scale: int = 1,
    camera_view: VolumeCameraView = "oblique",
    camera_zoom: Optional[float] = None,
    camera_orbit_degrees: float = 0.0,
    starfield: Optional[bool] = None,
    star_count: Optional[int] = None,
    source_marker: bool = False,
    show_bodies: bool = False,
    show_trajectory: bool = False,
    show_waveform: bool = False,
    trajectory_tail_cycles: float = 2.0,
    trajectory_line_width: float = 1.6,
    trajectory_color: str = DEFAULT_TRAJECTORY_COLOR,
    trajectory_as_tube: bool = False,
    orbit_display_radius: float = 0.16,
    primary_body_radius: float = 0.043,
    secondary_body_radius: float = 0.014,
    body_exaggeration: float = 2.0,
    waveform_fraction: float = 0.22,
    waveform_theta: float = np.pi / 3.0,
    waveform_phi: float = 0.0,
    waveform_font_style: WaveformFontStyle = "latex",
    waveform_style_file: Optional[PathLike] = None,
    global_start_time: Optional[float] = None,
    global_end_time: Optional[float] = None,
    normalization_time: Optional[float] = None,
    normalization_samples: int = 9,
) -> Path:
    r"""Render an evolving full-sky EMRI volume to MP4 or GIF.

    The expensive Cartesian angular harmonics and waveform splines are built
    once and reused.  Only the retarded-time mode values and volume scalar field
    change from frame to frame.  ``angular_sampling="spherical"`` retains the
    older, lower-memory interpolated angular grid as a fallback.  When requested,
    the central orbit is reconstructed from FEW's relativistic
    :math:`(p,e,\Phi_r,\Phi_\phi)` trajectory.  Its overall spatial scale is a
    presentation choice set by ``orbit_display_radius``, but the bodies are then
    drawn on that same scale: the primary spans ``body_exaggeration`` horizon
    radii :math:`r_+`, so its size relative to the orbit stays faithful as
    :math:`(a,p,e)` change.

    ``global_start_time`` and ``global_end_time`` keep the waveform panel,
    camera motion, and colour normalization continuous when disjoint frame
    ranges are rendered by a cluster job array.  When ``normalization_time`` is
    omitted, a robust common scale is measured at ``normalization_samples``
    times spanning that global interval.  This avoids clipping stronger late
    inspiral frames against the first frame's scale.
    For energy flux, ``flux_mode_combination="incoherent"`` removes modal
    cross terms and therefore suppresses rapidly rotating angular lobes.
    """

    output = Path(filename)
    _validate_volume_presentation(component, opacity_profile, camera_view)
    _validate_flux_mode_combination(flux_mode_combination)
    resolved_presentation = _resolve_volume_presentation(
        presentation,
        color_exposure=color_exposure,
        background_color=background_color,
        camera_zoom=camera_zoom,
        star_count=star_count,
        starfield=starfield,
    )
    suffix = output.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError("animation filename must end in .mp4 or .gif")
    if frames < 2:
        raise ValueError("frames must be at least 2")
    if fps < 1:
        raise ValueError("fps must be positive")
    if image_scale < 1:
        raise ValueError("image_scale must be at least 1")
    if not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in the interval (0, 1]")
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative")
    if normalization_samples < 1:
        raise ValueError("normalization_samples must be positive")
    if angular_sampling not in ("cartesian", "spherical"):
        raise ValueError("angular_sampling must be 'cartesian' or 'spherical'")
    if waveform_font_style not in ("latex", "sans"):
        raise ValueError("waveform_font_style must be 'latex' or 'sans'")
    if trajectory_tail_cycles <= 0.0:
        raise ValueError("trajectory_tail_cycles must be positive")
    if trajectory_line_width <= 0.0:
        raise ValueError("trajectory_line_width must be positive")
    _trajectory_rgb(trajectory_color)
    if not 0.0 < orbit_display_radius < 0.5:
        raise ValueError("orbit_display_radius must be in the interval (0, 0.5)")
    if primary_body_radius <= 0.0 or secondary_body_radius <= 0.0:
        raise ValueError("body radii must be positive")
    if body_exaggeration <= 0.0:
        raise ValueError("body_exaggeration must be positive")
    if show_waveform and not 0.1 <= waveform_fraction < 0.5:
        raise ValueError("waveform_fraction must be in the interval [0.1, 0.5)")
    if (show_bodies or show_trajectory) and not waveform.has_trajectory:
        raise ValueError(
            "Body and trajectory rendering requires the FEW trajectory arrays. "
            "Regenerate relativistic-modes.npz with visualize_emri_waveform.py."
        )
    if suffix == ".mp4" and any(size % 2 for size in window_size):
        raise ValueError("MP4 window dimensions must both be even")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Animation export requires imageio and imageio-ffmpeg. Install "
            "FEW with `pip install -e '.[visualization]'`."
        ) from exc

    time = np.asarray(waveform.time, dtype=float)
    start = float(time[0] + max_delay if start_time is None else start_time)
    end = float(time[-1] if end_time is None else end_time)
    _validate_frame_window(time, start, max_delay)
    _validate_frame_window(time, end, max_delay)
    if end <= start:
        raise ValueError("end_time must be greater than start_time")
    global_start = float(start if global_start_time is None else global_start_time)
    global_end = float(end if global_end_time is None else global_end_time)
    _validate_frame_window(time, global_start, max_delay)
    _validate_frame_window(time, global_end, max_delay)
    if global_end <= global_start:
        raise ValueError("global_end_time must be greater than global_start_time")
    if start < global_start or end > global_end:
        raise ValueError(
            "the local animation interval must lie inside the global interval"
        )
    normalization_frames = _normalization_frame_times(
        global_start,
        global_end,
        normalization_time=normalization_time,
        samples=normalization_samples,
    )
    for normalization_frame in normalization_frames:
        _validate_frame_window(time, float(normalization_frame), max_delay)
    frame_times = np.linspace(start, end, frames)

    total_width, total_height = window_size
    panel_height = (
        int(round(total_height * waveform_fraction / 2.0) * 2)
        if show_waveform
        else 0
    )
    main_window_size = (total_width, total_height - panel_height)
    if min(main_window_size) <= 0:
        raise ValueError("waveform_fraction leaves no space for the volume")

    sampling = _prepare_mode_volume_sampling(
        waveform,
        resolution=resolution,
        radius=radius,
        polar_samples=polar_samples,
        azimuthal_samples=azimuthal_samples,
        inner_window_fraction=inner_window_fraction,
        outer_window_fraction=outer_window_fraction,
        angular_sampling=angular_sampling,
    )
    first_field = _component_from_mode_sampling(
        sampling,
        frame_time=float(frame_times[0]),
        max_delay=max_delay,
        component=component,
        flux_mode_combination=flux_mode_combination,
    )
    zero = np.zeros_like(first_field)
    component_fields = {
        "plus": zero.copy(),
        "cross": zero.copy(),
        "amplitude": zero.copy(),
        "energy_flux": zero.copy(),
    }
    component_fields[component] = first_field
    first_volume = RetardedTimeVolume(
        axis=sampling.axis,
        plus=component_fields["plus"],
        cross=component_fields["cross"],
        amplitude=component_fields["amplitude"],
        energy_flux=component_fields["energy_flux"],
        frame_time=float(frame_times[0]),
        max_delay=max_delay,
        radius=radius,
        normalization=1.0,
        energy_flux_normalization=1.0,
        normalized=False,
        angular_pattern="mode_resolved",
        model=waveform.model,
        mode_count=waveform.modes.shape[1],
    )
    render_scales = []
    for normalization_frame in normalization_frames:
        if np.isclose(
            normalization_frame,
            float(frame_times[0]),
            rtol=0.0,
            atol=np.finfo(float).eps * max(1.0, abs(float(frame_times[0]))),
        ):
            normalization_field = first_field
        else:
            normalization_field = _component_from_mode_sampling(
                sampling,
                frame_time=float(normalization_frame),
                max_delay=max_delay,
                component=component,
                flux_mode_combination=flux_mode_combination,
            )
        normalization_field = _smooth_render_field(
            normalization_field, smooth_sigma
        )
        render_scales.append(_render_field_scale(normalization_field, component))
        if normalization_field is not first_field:
            del normalization_field
    render_scale = max(render_scales)
    plotter = render_volume(
        first_volume,
        component=component,
        show=False,
        opacity_profile=opacity_profile,
        color_scheme=color_scheme,
        presentation=presentation,
        color_exposure=color_exposure,
        background_color=background_color,
        opacity=opacity,
        shell_count=shell_count,
        shell_min=shell_min,
        shell_max=shell_max,
        shell_width=shell_width,
        shell_opacity_floor=shell_opacity_floor,
        shell_glow=shell_glow,
        smooth_sigma=smooth_sigma,
        opacity_unit_distance=opacity_unit_distance,
        window_size=main_window_size,
        camera_view=camera_view,
        camera_zoom=camera_zoom,
        starfield=starfield,
        star_count=star_count,
        source_marker=source_marker and not show_bodies,
    )
    grid = plotter._fewview_volume_grid
    render_name = plotter._fewview_render_name
    volume_actor = plotter._fewview_volume_actor
    if render_name is None:  # pragma: no cover - defensive
        plotter.close()
        raise RuntimeError("animation requires the cinematic volume renderer")

    (
        trajectory_time,
        display_trajectory,
        trajectory_tail_duration,
        secondary_actor,
        trajectory_polyline,
    ) = _add_source_overlays(
        plotter,
        waveform,
        frame_time=float(frame_times[0]),
        radius=radius,
        start_time=global_start,
        end_time=global_end,
        show_bodies=show_bodies,
        show_trajectory=show_trajectory,
        trajectory_tail_cycles=trajectory_tail_cycles,
        trajectory_line_width=trajectory_line_width,
        trajectory_color=trajectory_color,
        trajectory_as_tube=trajectory_as_tube,
        orbit_display_radius=orbit_display_radius,
        primary_body_radius=primary_body_radius,
        secondary_body_radius=secondary_body_radius,
        body_exaggeration=body_exaggeration,
    )

    waveform_panel = (
        _WaveformPanelRenderer(
            waveform,
            start_time=global_start,
            end_time=global_end,
            width=total_width * image_scale,
            height=panel_height * image_scale,
            theta=waveform_theta,
            phi=waveform_phi,
            font_style=waveform_font_style,
            style_file=waveform_style_file,
            background_color=resolved_presentation.background_color,
        )
        if show_waveform
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".mp4":
        writer = imageio.get_writer(
            output,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            pixelformat="yuv420p",
        )
    else:
        writer = imageio.get_writer(output, mode="I", fps=fps, loop=0)

    base_camera_position = np.asarray(plotter.camera.position, dtype=float)
    render_array = grid.point_data[render_name]
    try:
        plotter.show(auto_close=False)
        for frame_index, frame_time in enumerate(frame_times):
            if frame_index == 0:
                field = first_field
            else:
                field = _component_from_mode_sampling(
                    sampling,
                    frame_time=float(frame_time),
                    max_delay=max_delay,
                    component=component,
                    flux_mode_combination=flux_mode_combination,
                )
            display_source = _smooth_render_field(field, smooth_sigma)
            display_field, _ = _normalize_render_field(
                display_source,
                component,
                scale=render_scale,
                opacity_profile=opacity_profile,
            )
            render_array[:] = display_field.ravel(order="F")
            render_array.VTKObject.Modified()
            grid.GetPointData().GetArray(render_name).Modified()
            grid.Modified()
            volume_actor.mapper.Update()
            if display_trajectory is not None and trajectory_time is not None:
                current_position = _interpolate_trajectory_position(
                    trajectory_time, display_trajectory, float(frame_time)
                )
                if secondary_actor is not None:
                    secondary_actor.position = tuple(current_position)
                if trajectory_polyline is not None:
                    tail = _trajectory_tail_points(
                        trajectory_time,
                        display_trajectory,
                        frame_time=float(frame_time),
                        duration=float(trajectory_tail_duration),
                    )
                    _set_polyline_points(
                        trajectory_polyline, tail, color=trajectory_color
                    )
            if camera_orbit_degrees != 0.0:
                global_fraction = (float(frame_time) - global_start) / (
                    global_end - global_start
                )
                angle = np.deg2rad(camera_orbit_degrees * global_fraction)
                cosine = np.cos(angle)
                sine = np.sin(angle)
                x, y, z = base_camera_position
                plotter.camera.position = (
                    cosine * x - sine * y,
                    sine * x + cosine * y,
                    z,
                )
            plotter.render()
            image = plotter.screenshot(
                return_img=True,
                scale=image_scale,
            )
            image = np.asarray(image)[..., :3]
            if waveform_panel is not None:
                panel = waveform_panel.render(float(frame_time))
                if panel.shape[1] != image.shape[1]:
                    raise RuntimeError(
                        "waveform panel width does not match the VTK render"
                    )
                image = np.vstack((image, panel))
            writer.append_data(image)
    finally:
        writer.close()
        plotter.close()
    return output


def _add_contour_render(
    plotter,
    grid,
    field: np.ndarray,
    *,
    component: VolumeComponent,
    color_scheme: VolumeColorScheme,
    color_exposure: float,
    opacity: float,
    show_scalar_bar: bool,
) -> None:
    maximum = float(np.max(np.abs(field)))
    if component in ("amplitude", "energy_flux"):
        levels = maximum * np.array([0.22, 0.42, 0.68])
        clim = (0.0, maximum)
    else:
        levels = maximum * np.array([-0.65, -0.35, 0.35, 0.65])
        clim = (-maximum, maximum)

    contours = grid.contour(isosurfaces=levels, scalars=component)
    plotter.add_mesh(
        contours,
        scalars=component,
        cmap=_volume_colormap(
            component, color_scheme, color_exposure=color_exposure
        ),
        clim=clim,
        opacity=opacity,
        smooth_shading=True,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title": component.replace("_", " ")},
    )


def _resolve_opacity_unit_distance(
    radius: float, requested: Optional[float]
) -> float:
    """Return a resolution-independent volume-opacity accumulation length."""

    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("volume radius must be positive")
    distance = 0.04 * radius if requested is None else float(requested)
    if distance <= 0.0:
        raise ValueError("opacity_unit_distance must be positive")
    return distance


def _smooth_render_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply optional display-only Cartesian smoothing to a scalar field."""

    if sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative")
    display = np.asarray(field, dtype=float)
    if sigma == 0.0:
        return display
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(display, sigma=float(sigma), mode="nearest")


def _normalization_frame_times(
    global_start: float,
    global_end: float,
    *,
    normalization_time: Optional[float],
    samples: int,
) -> np.ndarray:
    """Choose common scale-sampling times for a full or segmented movie."""

    if samples < 1:
        raise ValueError("normalization_samples must be positive")
    if global_end <= global_start:
        raise ValueError("global_end must be greater than global_start")
    if normalization_time is not None:
        return np.asarray([float(normalization_time)])
    if samples == 1:
        return np.asarray([float(global_end)])
    return np.linspace(float(global_start), float(global_end), samples)


def _normalize_render_field(
    field: np.ndarray,
    component: VolumeComponent,
    *,
    scale: Optional[float] = None,
    opacity_profile: VolumeOpacityProfile = "soft",
) -> tuple[np.ndarray, tuple[float, float]]:
    if scale is None:
        scale = _render_field_scale(field, component)
    scale = max(float(scale), np.finfo(float).tiny)

    # The shells profile exposes a set of positive strain levels. Mapping those
    # levels onto [0, 1] lets a sequential palette such as magma use its
    # complete colour range, while the negative half-cycle remains transparent
    # in the opacity transfer function.
    if opacity_profile == "shells":
        return np.clip(field / scale, 0.0, 1.0), (0.0, 1.0)

    # The flux profile displays a logarithmic energy flux. Four decades keep
    # faint, spatially broad emission visible instead of restricting the render
    # to the thinnest peak-flux shells.
    if opacity_profile == "flux":
        scaled = np.clip(field / scale, 0.0, 1.0)
        decades = 4.0
        floor = 10.0 ** (-decades)
        compressed = np.zeros_like(scaled)
        positive = scaled > floor
        compressed[positive] = (
            np.log10(scaled[positive]) + decades
        ) / decades
        return np.clip(compressed, 0.0, 1.0), (0.0, 1.0)

    if component in ("amplitude", "energy_flux"):
        return np.clip(field / scale, 0.0, 1.0), (
            0.0,
            1.0,
        )

    return np.clip(field / scale, -1.0, 1.0), (
        -1.0,
        1.0,
    )


def _render_field_scale(field: np.ndarray, component: VolumeComponent) -> float:
    if component in ("amplitude", "energy_flux"):
        nonzero = field[field > 0.0]
    else:
        magnitude = np.abs(field)
        nonzero = magnitude[magnitude > 0.0]
    return float(np.quantile(nonzero, 0.999)) if nonzero.size else 1.0


def _wavefront_opacity_transfer(
    component: VolumeComponent,
    *,
    n_colors: int,
    maximum_opacity: float,
    profile: VolumeOpacityProfile,
    shell_count: int = 7,
    shell_min: float = 0.10,
    shell_max: float = 0.92,
    shell_width: float = 0.075,
    shell_opacity_floor: float = 0.16,
    shell_glow: float = 0.12,
) -> np.ndarray:
    """Build generic, layered signed-strain, or logarithmic flux opacity maps."""

    if profile == "shells":
        if shell_count < 2:
            raise ValueError("shell_count must be at least 2")
        if not 0.0 <= shell_min < shell_max <= 1.0:
            raise ValueError(
                "shell limits must satisfy 0 <= shell_min < shell_max <= 1"
            )
        if not 0.0 < shell_width <= 0.5:
            raise ValueError("shell_width must be in the interval (0, 0.5]")
        if not 0.0 <= shell_opacity_floor <= 1.0:
            raise ValueError("shell_opacity_floor must be in the interval [0, 1]")
        if not 0.0 <= shell_glow <= 1.0:
            raise ValueError("shell_glow must be in the interval [0, 1]")

        values = np.linspace(0.0, 1.0, n_colors)
        centers = np.linspace(
            shell_min, shell_max, shell_count
        )
        layer_opacities = maximum_opacity * np.linspace(
            shell_opacity_floor, 1.0, shell_count
        )

        # The shells transfer function gives each level a very sharp leading
        # edge, a longer transparent decay, and progressively higher opacity.
        # This reads as nested translucent sheets rather than a set of equally
        # bright contour lines. A weak Gaussian halo softens the sheet
        # edges without filling the transparent gaps between wavefronts.
        rise_width = max(
            shell_width * 0.08,
            2.0 / max(n_colors - 1, 1),
        )
        displacement = values[:, None] - centers[None, :]
        rise = _smoothstep((displacement + rise_width) / rise_width)
        decay = 1.0 - _smoothstep(displacement / shell_width)
        sheets = rise * decay
        sheet_alpha = np.max(sheets * layer_opacities[None, :], axis=1)

        halo_width = 1.8 * shell_width
        halos = np.exp(-0.5 * (displacement / halo_width) ** 2)
        halo_alpha = shell_glow * np.max(
            halos * layer_opacities[None, :], axis=1
        )
        alpha = np.clip(sheet_alpha + halo_alpha, 0.0, maximum_opacity)
        alpha *= _smoothstep(
            (values - max(0.0, shell_min - 2.0 * rise_width))
            / max(rise_width, np.finfo(float).eps)
        )
        return np.asarray(
            np.clip(255.0 * alpha, 0.0, 255.0), dtype=np.uint8
        )

    if profile == "flux":
        values = np.linspace(0.0, 1.0, n_colors)
        # The input has already been compressed logarithmically. A broad,
        # saturating opacity ramp lets colour carry the power variation. Strong
        # angular lobes therefore remain bright without also becoming much more
        # solid than the surrounding radiation.
        visibility = _smoothstep((values - 0.015) / 0.20)
        plateau = 0.78 + 0.22 * _smoothstep((values - 0.20) / 0.55)
        alpha = maximum_opacity * visibility * plateau
        return np.asarray(
            np.clip(255.0 * alpha, 0.0, 255.0), dtype=np.uint8
        )

    if component in ("amplitude", "energy_flux"):
        values = np.linspace(0.0, 1.0, n_colors)
        if profile == "soft":
            alpha = maximum_opacity * _smoothstep((values - 0.08) / 0.78) ** 1.15
        else:
            centers = np.linspace(0.18, 0.94, 7)
            width = 0.040
            bands = np.exp(
                -0.5 * ((values[:, None] - centers[None, :]) / width) ** 2
            )
            alpha = maximum_opacity * np.max(bands, axis=1)
    else:
        values = np.linspace(-1.0, 1.0, n_colors)
        if profile == "soft":
            centers = np.asarray([-0.58, 0.58])
            width = 0.20
        else:
            centers = np.asarray([-0.86, -0.52, -0.20, 0.20, 0.52, 0.86])
            width = 0.082
        bands = np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2)
        alpha = maximum_opacity * np.max(bands, axis=1)
        if profile == "bands":
            alpha *= _smoothstep((np.abs(values) - 0.045) / 0.055)
    return np.asarray(np.clip(255.0 * alpha, 0.0, 255.0), dtype=np.uint8)


def _volume_colormap(
    component: VolumeComponent,
    color_scheme: VolumeColorScheme,
    *,
    color_exposure: float = 1.0,
):
    from matplotlib.colors import LinearSegmentedColormap

    if color_exposure <= 0.0:
        raise ValueError("color_exposure must be positive")
    if color_scheme in ("cinematic", "rainbow"):
        if color_scheme == "rainbow":
            # A perceptually smooth blue-cyan-green-gold-red sequence, a gentler
            # alternative to a hard rainbow for signed-strain shells.
            colors = [
                "#092b8f",
                "#0078cf",
                "#00c5db",
                "#2cce72",
                "#9fd447",
                "#f5df4d",
                "#f79a42",
                "#e54155",
            ]
        elif component in ("amplitude", "energy_flux"):
            colors = [
                "#081536",
                "#0069a8",
                "#00b8b0",
                "#2ad14f",
                "#d5ec35",
                "#ffad32",
                "#ec315e",
            ]
        else:
            colors = [
                "#0a1e70",
                "#005fc4",
                "#00a9ed",
                "#00dfc2",
                "#32e85a",
                "#bde83b",
                "#ffe144",
                "#ff8f30",
                "#f23570",
            ]
        base = LinearSegmentedColormap.from_list(
            f"few_{color_scheme}_base", colors, N=512
        )
        colors = base(np.linspace(0.0, 1.0, 512))
    else:
        from matplotlib import colormaps

        base = colormaps[_MATPLOTLIB_COLOR_SCHEMES[color_scheme]]
        if color_scheme == "aurora":
            # Retain magma's violet-red-gold identity while skipping its nearly
            # black foot and biasing low scalar values toward luminous colours.
            values = np.linspace(0.0, 1.0, 512)
            coordinates = 0.16 + 0.83 * values**0.72
        else:
            # Avoid the near-black first few percent, which disappear against
            # the render background.
            coordinates = np.linspace(0.07, 0.98, 512)
        colors = base(coordinates)

    # Exposure is applied to colour only. Opacity remains controlled entirely
    # by the transfer function, so a brighter render need not become more solid.
    colors[:, :3] = 1.0 - (1.0 - colors[:, :3]) ** color_exposure
    return LinearSegmentedColormap.from_list(
        f"few_{color_scheme}_exposure_{color_exposure:g}", colors, N=512
    )


def _add_starfield(
    plotter,
    radius: float,
    *,
    count: int = 4000,
    seed: int = 937,
) -> None:
    import pyvista as pv

    if count < 0:
        raise ValueError("star count must be non-negative")
    if count == 0:
        return
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    distances = radius * rng.uniform(5.7, 6.5, size=count)
    points = directions * distances[:, None]
    hero_count = min(count, max(1, int(round(0.018 * count))))
    remaining = count - hero_count
    bright_count = (
        min(remaining, max(1, int(round(0.14 * count))))
        if remaining > 0
        else 0
    )
    faint_count = count - bright_count - hero_count
    groups = (
        (points[:faint_count], "#a9c9ff", 1.9, 0.74),
        (
            points[faint_count : faint_count + bright_count],
            "#f5f3df",
            3.1,
            0.96,
        ),
        (points[faint_count + bright_count :], "#fff7d2", 4.8, 0.99),
    )
    for group, color, point_size, opacity in groups:
        if group.size == 0:
            continue
        plotter.add_mesh(
            pv.PolyData(group),
            color=color,
            point_size=point_size,
            opacity=opacity,
            render_points_as_spheres=True,
            lighting=False,
        )


def save_volume(volume: RetardedTimeVolume, filename: PathLike) -> Path:
    """Save a retarded-time volume as VTK ImageData (``.vti``)."""

    output = Path(filename)
    if output.suffix.lower() != ".vti":
        raise ValueError("volume filename must use the .vti extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    to_pyvista(volume).save(output)
    return output


def _as_numpy(values: np.ndarray) -> np.ndarray:
    """Copy NumPy/CuPy-like values to a NumPy array."""

    if hasattr(values, "get"):
        values = values.get()
    return np.asarray(values)


def _prepare_waveform(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    frame_time: Optional[float],
    max_delay: Optional[float],
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    t = _as_numpy(time).astype(float, copy=False)
    hp = _as_numpy(h_plus).astype(float, copy=False)
    hc = _as_numpy(h_cross).astype(float, copy=False)

    if t.ndim != 1 or hp.ndim != 1 or hc.ndim != 1:
        raise ValueError("time, h_plus, and h_cross must be one-dimensional")
    if len(t) < 2 or hp.shape != t.shape or hc.shape != t.shape:
        raise ValueError("time, h_plus, and h_cross must have the same length >= 2")
    if (
        not np.all(np.isfinite(t))
        or not np.all(np.isfinite(hp))
        or not np.all(np.isfinite(hc))
    ):
        raise ValueError("waveform arrays must contain only finite values")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time samples must be strictly increasing")

    frame = float(t[-1] if frame_time is None else frame_time)
    if frame < t[0] or frame > t[-1]:
        raise ValueError("frame_time must fall within the waveform time range")
    available_delay = frame - float(t[0])
    delay_max = available_delay if max_delay is None else float(max_delay)
    if delay_max <= 0.0 or delay_max > available_delay:
        raise ValueError(
            "max_delay must be positive and no larger than frame_time - time[0]"
        )

    normalization = float(np.max(np.hypot(hp, hc)))
    if normalization == 0.0:
        raise ValueError("the waveform is identically zero")
    if normalize:
        hp = hp / normalization
        hc = hc / normalization
    else:
        normalization = 1.0
    return t, hp, hc, frame, delay_max, normalization


def _sample_retarded(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    frame_time: float,
    delay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    retarded_time = frame_time - delay
    plus = np.interp(retarded_time, time, h_plus, left=0.0, right=0.0)
    cross = np.interp(retarded_time, time, h_cross, left=0.0, right=0.0)
    return plus, cross


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _radial_window(radius: np.ndarray, window_fraction: float) -> np.ndarray:
    if not 0.0 <= window_fraction < 0.5:
        raise ValueError("window_fraction must be in the interval [0, 0.5)")
    if window_fraction == 0.0:
        return np.ones_like(radius, dtype=float)
    inner = _smoothstep(radius / window_fraction)
    outer = _smoothstep((1.0 - radius) / window_fraction)
    return inner * outer


def _symmetric_limit(values: np.ndarray) -> float:
    maximum = float(np.max(np.abs(values)))
    return max(maximum, np.finfo(float).eps)


def _validate_mode_waveform(waveform: RelativisticModeWaveform) -> None:
    time = np.asarray(waveform.time)
    modes = np.asarray(waveform.modes)
    ell = np.asarray(waveform.ell)
    m = np.asarray(waveform.m)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("waveform.time must be a strictly increasing 1D array")
    if modes.ndim != 2 or modes.shape != (time.size, ell.size):
        raise ValueError("waveform.modes must have shape (len(time), len(ell))")
    if ell.ndim != 1 or m.shape != ell.shape:
        raise ValueError("waveform.ell and waveform.m must be matching 1D arrays")
    if not np.all(np.isfinite(modes)):
        raise ValueError("waveform modes must contain only finite values")
    trajectory_values = (
        waveform.trajectory_p,
        waveform.trajectory_e,
        waveform.trajectory_xI,
        waveform.trajectory_phi_phi,
        waveform.trajectory_phi_r,
    )
    present = [value is not None for value in trajectory_values]
    if any(present) and not all(present):
        raise ValueError("all FEW trajectory arrays must be provided together")
    if all(present):
        for value in trajectory_values:
            array = np.asarray(value)
            if array.shape != time.shape or not np.all(np.isfinite(array)):
                raise ValueError(
                    "each FEW trajectory array must be finite and match waveform.time"
                )


@njit(parallel=True, fastmath=False)
def _fill_ylm_grid(
    output: np.ndarray,
    ell: np.ndarray,
    m: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> None:
    """Evaluate FEW's spin-weighted harmonics for many sky directions."""

    for angle_index in prange(theta.size):
        for mode_index in range(ell.size):
            output[angle_index, mode_index] = _ylm_kernel_inner(
                ell[mode_index],
                m[mode_index],
                theta[angle_index],
                phi[angle_index],
            )


def _evaluate_ylm_grid(
    ell: np.ndarray,
    m: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    dtype=np.complex128,
) -> np.ndarray:
    """Evaluate and cache FEW harmonics with selectable storage precision."""

    output = np.empty((theta.size, ell.size), dtype=dtype)
    _fill_ylm_grid(output, ell, m, theta, phi)
    return output


@njit(parallel=True, fastmath=False)
def _contract_cartesian_modes(
    retarded_modes: np.ndarray,
    ylms: np.ndarray,
    window: np.ndarray,
) -> np.ndarray:
    """Contract cached harmonics with modes in parallel over voxels."""

    output = np.empty(retarded_modes.shape[0], dtype=np.complex128)
    for point_index in prange(retarded_modes.shape[0]):
        value = 0.0j
        for mode_index in range(retarded_modes.shape[1]):
            value += (
                retarded_modes[point_index, mode_index]
                * ylms[point_index, mode_index]
            )
        output[point_index] = value * window[point_index]
    return output


@njit(parallel=True, fastmath=False)
def _contract_cartesian_mode_power(
    retarded_modes: np.ndarray,
    ylms: np.ndarray,
    window: np.ndarray,
) -> np.ndarray:
    """Sum individual modal powers, omitting coherent cross terms."""

    output = np.empty(retarded_modes.shape[0], dtype=np.float64)
    for point_index in prange(retarded_modes.shape[0]):
        value = 0.0
        for mode_index in range(retarded_modes.shape[1]):
            term = (
                retarded_modes[point_index, mode_index]
                * ylms[point_index, mode_index]
            )
            value += term.real * term.real + term.imag * term.imag
        output[point_index] = value * window[point_index] ** 2
    return output


def _sample_direct_cartesian_mode_field(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    derivative_order: int,
) -> np.ndarray:
    """Evaluate the mode sum directly at every in-sphere Cartesian voxel."""

    point_count = sampling.inside_indices.size
    mode_count = sampling.waveform.modes.shape[1]
    # Bound the temporary CubicSpline result to roughly 128 MB for complex128
    # modes.  The precomputed Cartesian harmonic basis remains reusable across
    # every animation frame.
    chunk_size = max(1, min(point_count, 8_000_000 // max(mode_count, 1)))
    output = np.zeros(int(np.prod(sampling.cartesian_shape)), dtype=np.complex128)
    for first in range(0, point_count, chunk_size):
        last = min(first + chunk_size, point_count)
        retarded_time = frame_time - max_delay * sampling.cartesian_radius_fraction[
            first:last
        ]
        retarded_modes = sampling.mode_spline(retarded_time, derivative_order)
        field = _contract_cartesian_modes(
            retarded_modes,
            sampling.ylms[first:last],
            sampling.cartesian_window[first:last],
        )
        output[sampling.inside_indices[first:last]] = field
    return output.reshape(sampling.cartesian_shape)


def _sample_direct_cartesian_mode_power(
    sampling: _ModeVolumeSampling,
    *,
    frame_time: float,
    max_delay: float,
    derivative_order: int,
) -> np.ndarray:
    """Evaluate the modal-incoherent power at every Cartesian voxel."""

    point_count = sampling.inside_indices.size
    mode_count = sampling.waveform.modes.shape[1]
    chunk_size = max(1, min(point_count, 8_000_000 // max(mode_count, 1)))
    output = np.zeros(int(np.prod(sampling.cartesian_shape)), dtype=np.float64)
    for first in range(0, point_count, chunk_size):
        last = min(first + chunk_size, point_count)
        retarded_time = frame_time - max_delay * sampling.cartesian_radius_fraction[
            first:last
        ]
        retarded_modes = sampling.mode_spline(retarded_time, derivative_order)
        power = _contract_cartesian_mode_power(
            retarded_modes,
            sampling.ylms[first:last],
            sampling.cartesian_window[first:last],
        )
        output[sampling.inside_indices[first:last]] = power
    return output.reshape(sampling.cartesian_shape)


def _sample_complex_spherical_field(
    field: np.ndarray,
    coordinates: np.ndarray,
    shape: tuple[int, int, int],
    map_coordinates,
) -> np.ndarray:
    real = map_coordinates(
        np.real(field), coordinates, order=3, mode="nearest", prefilter=True
    )
    imaginary = map_coordinates(
        np.imag(field), coordinates, order=3, mode="nearest", prefilter=True
    )
    return (real + 1j * imaginary).reshape(shape)


__all__ = [
    "AngularPattern",
    "FluxModeCombination",
    "ModeAngularSampling",
    "RelativisticModeWaveform",
    "RetardedTimeVolume",
    "StrainSurface",
    "VolumeCameraView",
    "VolumeComponent",
    "VolumeRenderStyle",
    "VolumeColorScheme",
    "VolumeOpacityProfile",
    "VolumePresentation",
    "WaveformFontStyle",
    "build_mode_retarded_time_volume",
    "build_retarded_time_volume",
    "build_strain_surface",
    "choose_max_delay",
    "estimate_waveform_period",
    "generate_relativistic_mode_waveform",
    "plot_strain_surface",
    "plot_volume_slice",
    "polarizations_from_complex",
    "render_mode_animation",
    "render_mode_frame",
    "render_volume",
    "save_volume",
    "to_pyvista",
]
