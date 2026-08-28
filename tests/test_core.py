import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fewview._core import (
    RelativisticModeWaveform,
    _PRESET_CAMERA_DISTANCE,
    _PRIMARY_HALO_SCALE,
    _SECONDARY_BODY_SCALE,
    _apply_camera_offset,
    _camera_angle_vectors,
    _combine_frame_and_panel,
    _normalization_frame_times,
    _normalize_render_field,
    _prepare_display_trajectory,
    _resolve_body_radii,
    _resolve_opacity,
    _resolve_opacity_unit_distance,
    _resolve_volume_presentation,
    _smooth_render_field,
    _validate_volume_presentation,
    _volume_colormap,
    _wavefront_opacity_transfer,
    _zero_coordinate_plane,
    build_mode_retarded_time_volume,
    build_retarded_time_volume,
    build_strain_surface,
    choose_max_delay,
    estimate_waveform_period,
    generate_relativistic_mode_waveform,
    polarizations_from_complex,
)


class VisualizationTest(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 4.0, 9)
        self.h_plus = self.time + 1.0
        self.h_cross = 0.5 * (self.time + 1.0)

    def test_public_complex_convention(self):
        strain = self.h_plus - 1j * self.h_cross
        h_plus, h_cross = polarizations_from_complex(strain, convention="few")
        np.testing.assert_allclose(h_plus, self.h_plus)
        np.testing.assert_allclose(h_cross, self.h_cross)

    def test_internal_complex_convention(self):
        strain = self.h_plus + 1j * self.h_cross
        h_plus, h_cross = polarizations_from_complex(strain, convention="summation")
        np.testing.assert_allclose(h_plus, self.h_plus)
        np.testing.assert_allclose(h_cross, self.h_cross)

    def test_retarded_time_volume_samples_outgoing_wave(self):
        volume = build_retarded_time_volume(
            self.time,
            self.h_plus,
            self.h_cross,
            frame_time=4.0,
            max_delay=4.0,
            resolution=5,
            angular_pattern="line_of_sight",
            window_fraction=0.0,
            normalize=False,
        )

        # The origin sees u=4 s; the +x outer edge sees u=0 s.
        self.assertAlmostEqual(volume.plus[2, 2, 2], 5.0)
        self.assertAlmostEqual(volume.plus[4, 2, 2], 1.0)
        self.assertAlmostEqual(volume.cross[2, 2, 2], 2.5)
        self.assertAlmostEqual(volume.energy_flux[2, 2, 2], 1.25)
        self.assertEqual(volume.plus.shape, (5, 5, 5))

    def test_quadrupole_pattern_is_spin_two(self):
        volume = build_retarded_time_volume(
            self.time,
            self.h_plus,
            np.zeros_like(self.h_cross),
            frame_time=4.0,
            max_delay=4.0,
            resolution=5,
            angular_pattern="quadrupole",
            window_fraction=0.0,
            normalize=False,
        )

        # At theta=pi/2 the envelope is 1/2. A pi/2 azimuthal rotation
        # reverses a spin-2 plus pattern.
        self.assertAlmostEqual(volume.plus[4, 2, 2], 0.5)
        self.assertAlmostEqual(volume.plus[2, 4, 2], -0.5)

    def test_even_volume_slice_interpolates_to_zero_coordinate(self):
        coordinates = np.array([-1.0, -0.25, 0.25, 1.0])
        field = np.broadcast_to(
            coordinates[None, :, None], (3, coordinates.size, 2)
        )

        plane = _zero_coordinate_plane(field, coordinates, fixed_axis=1)

        np.testing.assert_allclose(plane, 0.0, atol=1e-15)

    def test_strain_surface_has_closed_finite_surface(self):
        flux = build_strain_surface(
            self.time,
            self.h_plus,
            self.h_cross,
            radial_samples=12,
            angular_samples=16,
        )
        self.assertEqual(flux.plus.shape, (12, 16))
        self.assertTrue(np.all(np.isfinite(flux.x)))
        np.testing.assert_allclose(flux.x[:, 0], flux.x[:, -1])
        np.testing.assert_allclose(flux.y[:, 0], flux.y[:, -1], atol=1e-15)
        np.testing.assert_allclose(flux.z[:, 0], flux.z[:, -1], atol=1e-15)

    def test_rejects_delay_outside_waveform(self):
        with self.assertRaisesRegex(ValueError, "max_delay"):
            build_retarded_time_volume(
                self.time,
                self.h_plus,
                self.h_cross,
                frame_time=2.0,
                max_delay=3.0,
            )

    def test_wavefront_window_uses_requested_number_of_cycles(self):
        time = np.linspace(0.0, 40.0, 2001)
        period = 2.0
        phase = 2.0 * np.pi * time / period
        h_plus = np.cos(phase)
        h_cross = np.sin(phase)

        self.assertAlmostEqual(
            estimate_waveform_period(time, h_plus, h_cross), period, places=3
        )
        self.assertAlmostEqual(
            choose_max_delay(
                time, h_plus, h_cross, wave_cycles=6.0
            ),
            6.0 * period,
            places=3,
        )

    def test_mode_waveform_projects_with_few_harmonics(self):
        time = np.linspace(0.0, 4.0, 17)
        mode = np.exp(-1j * time)[:, None]
        waveform = RelativisticModeWaveform(
            time=time,
            modes=mode,
            ell=np.array([2]),
            m=np.array([2]),
            teukolsky_modes_used=1,
            teukolsky_modes_available=1,
            retained_power_fraction=1.0,
        )
        strain = waveform.strain(theta=0.0, phi=0.0)
        expected_ylm = np.sqrt(5.0 / np.pi) / 2.0
        np.testing.assert_allclose(strain, expected_ylm * mode[:, 0])

    def test_mode_volume_has_spherical_boundary(self):
        time = np.linspace(0.0, 4.0, 65)
        phase = 2.0 * np.pi * time / 2.0
        waveform = RelativisticModeWaveform(
            time=time,
            modes=np.exp(-1j * phase)[:, None],
            ell=np.array([2]),
            m=np.array([2]),
            teukolsky_modes_used=1,
            teukolsky_modes_available=1,
            retained_power_fraction=1.0,
        )
        volume = build_mode_retarded_time_volume(
            waveform,
            max_delay=4.0,
            resolution=9,
            polar_samples=12,
            azimuthal_samples=24,
            inner_window_fraction=0.0,
            outer_window_fraction=0.0,
        )
        self.assertEqual(volume.angular_pattern, "mode_resolved")
        self.assertEqual(volume.mode_count, 1)
        self.assertEqual(volume.plus.shape, (9, 9, 9))
        self.assertEqual(volume.plus[0, 0, 0], 0.0)
        self.assertGreater(volume.amplitude[-1, 4, 4], 0.0)

    def test_cartesian_mode_volume_matches_direct_harmonic_evaluation(self):
        time = np.linspace(0.0, 4.0, 17)
        waveform = RelativisticModeWaveform(
            time=time,
            modes=np.ones((time.size, 1), dtype=complex),
            ell=np.array([2]),
            m=np.array([2]),
            teukolsky_modes_used=1,
            teukolsky_modes_available=1,
            retained_power_fraction=1.0,
        )
        volume = build_mode_retarded_time_volume(
            waveform,
            frame_time=4.0,
            max_delay=4.0,
            resolution=5,
            angular_sampling="cartesian",
            inner_window_fraction=0.0,
            outer_window_fraction=0.0,
            normalize=False,
        )

        expected = waveform.strain(theta=np.pi / 2.0, phi=0.0)[0]
        self.assertAlmostEqual(volume.plus[4, 2, 2], np.real(expected), places=6)
        self.assertAlmostEqual(volume.cross[4, 2, 2], -np.imag(expected), places=6)

    def test_mode_waveform_reconstructs_few_orbital_position(self):
        time = np.array([0.0, 1.0, 2.0])
        waveform = RelativisticModeWaveform(
            time=time,
            modes=np.ones((3, 1), dtype=complex),
            ell=np.array([2]),
            m=np.array([2]),
            teukolsky_modes_used=1,
            teukolsky_modes_available=1,
            retained_power_fraction=1.0,
            trajectory_p=np.full(3, 10.0),
            trajectory_e=np.zeros(3),
            trajectory_xI=np.ones(3),
            trajectory_phi_phi=np.array([0.0, np.pi / 2.0, np.pi]),
            trajectory_phi_r=np.zeros(3),
        )

        np.testing.assert_allclose(
            waveform.orbital_position(),
            np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [-10.0, 0.0, 0.0]]),
            atol=1e-14,
        )

    def test_shells_profile_uses_positive_strain_and_full_colour_range(self):
        field = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
        display, limits = _normalize_render_field(
            field,
            "plus",
            scale=1.0,
            opacity_profile="shells",
        )

        np.testing.assert_allclose(display, [0.0, 0.0, 0.0, 0.2, 1.0])
        self.assertEqual(limits, (0.0, 1.0))
        opacity = _wavefront_opacity_transfer(
            "plus",
            n_colors=512,
            maximum_opacity=0.2,
            profile="shells",
        )
        self.assertEqual(opacity[0], 0)
        self.assertGreater(int(np.max(opacity)), 0)

    def test_flux_profile_gamma_lifts_and_threshold_gates(self):
        field = np.array([0.0, 1.0e-4, 1.0e-2, 1.0])
        display, limits = _normalize_render_field(
            field,
            "energy_flux",
            scale=1.0,
            opacity_profile="flux",
        )

        # ``flux_gamma`` below one lifts the dim inter-crest values into the
        # bright half of the colour map while the crest stays at 1 and the
        # floor stays at 0.
        np.testing.assert_allclose(display, field ** 0.6, rtol=1e-6)
        self.assertEqual(display[0], 0.0)
        self.assertEqual(display[-1], 1.0)
        self.assertGreater(display[1], field[1])
        self.assertEqual(limits, (0.0, 1.0))

        opacity = _wavefront_opacity_transfer(
            "energy_flux",
            n_colors=512,
            maximum_opacity=0.5,
            profile="flux",
        )
        values = np.linspace(0.0, 1.0, 512)
        # The threshold gate keeps the dim troughs transparent while the bright
        # crests turn solid; this is what turns a filled ball into flux shells.
        self.assertEqual(int(opacity[0]), 0)
        self.assertEqual(int(opacity[np.searchsorted(values, 0.20)]), 0)
        self.assertGreater(int(opacity[np.searchsorted(values, 0.70)]), 0)
        self.assertTrue(np.all(np.diff(opacity.astype(int)) >= 0))
        self.assertGreaterEqual(int(opacity[-1]), int(0.5 * 255) - 1)

    def test_flux_profile_threshold_and_gamma_validation(self):
        with self.assertRaisesRegex(ValueError, "flux_gamma"):
            _normalize_render_field(
                np.array([0.0, 1.0]), "energy_flux", scale=1.0,
                opacity_profile="flux", flux_gamma=0.0,
            )
        with self.assertRaisesRegex(ValueError, "flux_threshold"):
            _wavefront_opacity_transfer(
                "energy_flux", n_colors=64, maximum_opacity=0.5,
                profile="flux", flux_threshold=1.0,
            )

    def test_flux_profile_has_a_higher_default_opacity(self):
        self.assertAlmostEqual(_resolve_opacity(None, "flux"), 0.55)
        self.assertAlmostEqual(_resolve_opacity(None, "shells"), 0.30)
        self.assertAlmostEqual(_resolve_opacity(None, "soft"), 0.11)
        self.assertAlmostEqual(_resolve_opacity(0.42, "flux"), 0.42)

    def test_apply_camera_offset_rotates_only_when_requested(self):
        class _Camera:
            def __init__(self):
                self.calls = []

            def Azimuth(self, angle):
                self.calls.append(("azimuth", angle))

            def Elevation(self, angle):
                self.calls.append(("elevation", angle))

            def OrthogonalizeViewUp(self):
                self.calls.append(("orthogonalize",))

        class _Plotter:
            def __init__(self):
                self.camera = _Camera()

        # A zero offset must leave the preset camera untouched.
        idle = _Plotter()
        _apply_camera_offset(idle, 0.0, 0.0)
        self.assertEqual(idle.camera.calls, [])

        # Both angles: orbit, then tilt, then re-level the horizon.
        both = _Plotter()
        _apply_camera_offset(both, 30.0, -15.0)
        self.assertEqual(
            both.camera.calls,
            [("azimuth", 30.0), ("elevation", -15.0), ("orthogonalize",)],
        )

        # Only elevation requested: azimuth is skipped.
        tilt = _Plotter()
        _apply_camera_offset(tilt, 0.0, 20.0)
        self.assertEqual(
            tilt.camera.calls, [("elevation", 20.0), ("orthogonalize",)]
        )

    def test_camera_angle_vectors_place_the_camera_at_absolute_angles(self):
        radius = 2.0
        position, view_up = _camera_angle_vectors(radius, 45.0, 30.0)
        position = np.asarray(position)
        distance = np.linalg.norm(position)

        latitude = np.degrees(np.arcsin(position[2] / distance))
        longitude = np.degrees(np.arctan2(position[1], position[0]))
        self.assertAlmostEqual(latitude, 45.0, places=6)
        self.assertAlmostEqual(longitude, 30.0, places=6)
        # Distance scales with the volume radius so framing tracks the presets.
        self.assertAlmostEqual(distance, _PRESET_CAMERA_DISTANCE * radius, places=6)
        # Away from the poles the spin axis is the view-up reference.
        self.assertEqual(tuple(view_up), (0.0, 0.0, 1.0))

        # At the equator the camera lies in the equatorial plane.
        equator, _ = _camera_angle_vectors(1.0, 0.0, 90.0)
        self.assertAlmostEqual(equator[2], 0.0, places=6)

        # Near a pole a horizontal up replaces the (degenerate) spin axis.
        _, polar_up = _camera_angle_vectors(1.0, 90.0, 0.0)
        self.assertAlmostEqual(polar_up[2], 0.0, places=6)

    def test_combine_frame_and_panel_stacks_opaque_but_composites_transparent(self):
        image = np.full((20, 8, 3), 100, dtype=np.uint8)

        # An opaque (RGB) panel is stacked beneath the scene, growing the frame.
        opaque = np.full((5, 8, 3), 200, dtype=np.uint8)
        stacked = _combine_frame_and_panel(image, opaque)
        self.assertEqual(stacked.shape, (25, 8, 3))
        np.testing.assert_array_equal(stacked[:20], image)
        np.testing.assert_array_equal(stacked[20:], opaque)

        # A transparent (RGBA) panel is alpha-composited over the bottom rows of
        # a full-height render, so the frame keeps its height and the scene
        # shows through wherever the panel is transparent.
        panel = np.zeros((5, 8, 4), dtype=np.uint8)
        panel[..., :3] = 240
        panel[2, :, 3] = 255  # one fully opaque row; the rest stay transparent
        out = _combine_frame_and_panel(image, panel)
        self.assertEqual(out.shape, (20, 8, 3))
        np.testing.assert_array_equal(out[:15], image[:15])  # untouched above
        np.testing.assert_array_equal(
            out[17], np.full((8, 3), 240, dtype=np.uint8)
        )  # opaque panel row overwrites the scene
        np.testing.assert_array_equal(out[15], image[15])  # transparent row kept

        with self.assertRaisesRegex(RuntimeError, "width"):
            _combine_frame_and_panel(image, np.zeros((5, 9, 4), dtype=np.uint8))

    def test_opacity_accumulation_distance_is_resolution_independent(self):
        self.assertAlmostEqual(_resolve_opacity_unit_distance(2.5, None), 0.1)
        self.assertAlmostEqual(_resolve_opacity_unit_distance(2.5, 0.2), 0.2)
        with self.assertRaisesRegex(ValueError, "opacity_unit_distance"):
            _resolve_opacity_unit_distance(1.0, 0.0)

    def test_display_smoothing_softens_voxels_without_moving_the_peak(self):
        field = np.zeros((9, 9, 9))
        field[4, 4, 4] = 1.0

        smoothed = _smooth_render_field(field, 0.65)

        self.assertEqual(
            np.unravel_index(np.argmax(smoothed), smoothed.shape), (4, 4, 4)
        )
        self.assertLess(smoothed[4, 4, 4], 1.0)
        self.assertAlmostEqual(float(np.sum(smoothed)), 1.0, places=10)

    def test_global_normalization_samples_cover_the_complete_movie(self):
        sampled = _normalization_frame_times(
            10.0, 20.0, normalization_time=None, samples=5
        )
        np.testing.assert_allclose(sampled, [10.0, 12.5, 15.0, 17.5, 20.0])
        explicit = _normalization_frame_times(
            10.0, 20.0, normalization_time=18.0, samples=5
        )
        np.testing.assert_allclose(explicit, [18.0])

    def test_named_profiles_require_the_matching_physical_field(self):
        _validate_volume_presentation("plus", "shells", "face_on")
        _validate_volume_presentation("energy_flux", "flux", "oblique")
        with self.assertRaisesRegex(ValueError, "requires plus or cross"):
            _validate_volume_presentation("energy_flux", "shells", "face_on")
        with self.assertRaisesRegex(ValueError, "requires energy_flux"):
            _validate_volume_presentation("plus", "flux", "oblique")

    def test_incoherent_flux_removes_modal_cross_terms(self):
        time = np.linspace(0.0, 4.0, 17)
        modes = np.column_stack((time, -time)).astype(complex)
        waveform = RelativisticModeWaveform(
            time=time,
            modes=modes,
            ell=np.array([2, 2]),
            m=np.array([2, -2]),
            teukolsky_modes_used=2,
            teukolsky_modes_available=2,
            retained_power_fraction=1.0,
        )
        common = dict(
            frame_time=4.0,
            max_delay=4.0,
            resolution=5,
            angular_sampling="cartesian",
            inner_window_fraction=0.0,
            outer_window_fraction=0.0,
            normalize=False,
        )
        coherent = build_mode_retarded_time_volume(
            waveform,
            flux_mode_combination="coherent",
            **common,
        )
        incoherent = build_mode_retarded_time_volume(
            waveform,
            flux_mode_combination="incoherent",
            **common,
        )

        self.assertAlmostEqual(coherent.energy_flux[4, 2, 2], 0.0, places=12)
        self.assertGreater(incoherent.energy_flux[4, 2, 2], 0.0)

    def test_dramatic_presentation_enables_the_dense_starfield(self):
        dramatic = _resolve_volume_presentation(
            "dramatic",
            color_exposure=None,
            background_color=None,
            camera_zoom=None,
            star_count=None,
            starfield=None,
        )

        self.assertGreater(dramatic.color_exposure, 1.0)
        self.assertEqual(dramatic.background_color, "#01051b")
        self.assertGreater(dramatic.camera_zoom, 1.0)
        self.assertTrue(dramatic.starfield)
        self.assertEqual(dramatic.star_count, 4000)

    def test_shells_dramatic_presentation_uses_reference_like_defaults(self):
        dramatic = _resolve_volume_presentation(
            "shells_dramatic",
            color_exposure=None,
            background_color=None,
            camera_zoom=None,
            star_count=None,
            starfield=None,
        )

        self.assertEqual(dramatic.color_exposure, 1.30)
        self.assertEqual(dramatic.background_color, "#010522")
        self.assertEqual(dramatic.camera_zoom, 1.12)
        self.assertTrue(dramatic.starfield)
        self.assertEqual(dramatic.star_count, 6500)
        self.assertFalse(dramatic.shade)

    def test_presentation_values_can_be_overridden(self):
        resolved = _resolve_volume_presentation(
            "dramatic",
            color_exposure=1.25,
            background_color="#123456",
            camera_zoom=1.04,
            star_count=321,
            starfield=False,
        )

        self.assertEqual(resolved.color_exposure, 1.25)
        self.assertEqual(resolved.background_color, "#123456")
        self.assertEqual(resolved.camera_zoom, 1.04)
        self.assertEqual(resolved.star_count, 321)
        self.assertFalse(resolved.starfield)

    def test_aurora_brightens_colour_without_changing_alpha(self):
        magma = np.asarray(_volume_colormap("plus", "magma")(0.2))
        glow = np.asarray(
            _volume_colormap(
                "plus", "aurora", color_exposure=1.65
            )(0.2)
        )
        luminance = np.array([0.2126, 0.7152, 0.0722])

        self.assertGreater(
            float(np.dot(glow[:3], luminance)),
            float(np.dot(magma[:3], luminance)),
        )
        self.assertEqual(glow[3], magma[3])

    def test_shells_are_separate_and_increase_in_opacity(self):
        transfer = _wavefront_opacity_transfer(
            "plus",
            n_colors=1001,
            maximum_opacity=0.5,
            profile="shells",
            shell_count=7,
            shell_min=0.10,
            shell_max=0.92,
            shell_width=0.055,
            shell_opacity_floor=0.20,
            shell_glow=0.0,
        )
        centers = np.linspace(0.10, 0.92, 7)
        peak_alpha = transfer[np.rint(1000 * centers).astype(int)]
        first_gap = transfer[int(round(1000 * np.mean(centers[:2])))]

        self.assertTrue(np.all(np.diff(peak_alpha.astype(int)) > 0))
        self.assertLess(int(first_gap), int(peak_alpha[0]))

    def test_rainbow_has_a_green_middle_layer(self):
        middle = np.asarray(
            _volume_colormap(
                "plus", "rainbow", color_exposure=1.0
            )(0.5)
        )

        self.assertGreater(middle[1], middle[0])
        self.assertGreater(middle[1], middle[2])

    def _body_waveform(self, *, p0, e0, spin):
        time = np.linspace(0.0, 4.0e4, 2001)
        return RelativisticModeWaveform(
            time=time,
            modes=np.zeros((time.size, 1), dtype=complex),
            ell=np.array([2]),
            m=np.array([2]),
            teukolsky_modes_used=1,
            teukolsky_modes_available=1,
            retained_power_fraction=1.0,
            trajectory_p=np.full_like(time, p0),
            trajectory_e=np.full_like(time, e0),
            trajectory_xI=np.ones_like(time),
            trajectory_phi_phi=2.0 * np.pi * time / 3.0e3,
            trajectory_phi_r=2.0 * np.pi * time / 8.0e3,
            primary_mass=1.0e6,
            secondary_mass=10.0,
            spin=spin,
        )

    def _drawn_bodies(self, waveform, *, orbit_display_radius=0.2, exaggeration=2.0):
        """Return the drawn radii alongside the display periapsis and scale."""

        display, _, display_scale = _prepare_display_trajectory(
            waveform,
            radius=1.0,
            orbit_display_radius=orbit_display_radius,
            start_time=float(waveform.time[0]),
            end_time=float(waveform.time[-1]),
        )
        primary, secondary = _resolve_body_radii(
            waveform,
            radius=1.0,
            display_scale=display_scale,
            display_trajectory=display,
            primary_body_radius=0.043,
            secondary_body_radius=0.014,
            body_exaggeration=exaggeration,
        )
        periapsis = float(np.min(np.linalg.norm(display[:, :2], axis=1)))
        return primary, secondary, periapsis, display_scale

    def test_horizon_radius_matches_the_kerr_horizon(self):
        for spin in (0.0, 0.5, 0.9, 0.99, 1.0):
            waveform = self._body_waveform(p0=12.0, e0=0.0, spin=spin)
            self.assertAlmostEqual(
                waveform.horizon_radius, 1.0 + np.sqrt(1.0 - spin**2)
            )
        self.assertIsNone(
            self._body_waveform(p0=12.0, e0=0.0, spin=None).horizon_radius
        )
        with self.assertRaises(ValueError):
            self._body_waveform(p0=12.0, e0=0.0, spin=1.5).horizon_radius

    def test_primary_is_drawn_on_the_orbit_scale(self):
        for spin, e0 in ((0.0, 0.3), (0.9, 0.0), (0.99, 0.8)):
            waveform = self._body_waveform(p0=12.0, e0=e0, spin=spin)
            primary, secondary, _, scale = self._drawn_bodies(waveform)
            self.assertAlmostEqual(primary, 2.0 * waveform.horizon_radius * scale)
            self.assertAlmostEqual(secondary, _SECONDARY_BODY_SCALE * primary)

    def test_a_less_spinning_primary_is_drawn_larger(self):
        radii = [
            self._drawn_bodies(self._body_waveform(p0=12.0, e0=0.0, spin=spin))[0]
            for spin in (0.0, 0.9, 0.99)
        ]
        self.assertGreater(radii[0], radii[1])
        self.assertGreater(radii[1], radii[2])

    def test_eccentric_bodies_stay_clear_of_periapsis(self):
        # A periapsis drawn inside the primary swallowed the secondary whole.
        for e0 in (0.0, 0.4, 0.6, 0.8, 0.9):
            for spin in (None, 0.99):
                primary, secondary, periapsis, _ = self._drawn_bodies(
                    self._body_waveform(p0=12.0, e0=e0, spin=spin)
                )
                self.assertLess(
                    _PRIMARY_HALO_SCALE * primary + secondary,
                    periapsis,
                    f"bodies reach the orbit at e={e0}, spin={spin}",
                )

    def test_bodies_without_a_spin_use_wave_sphere_fractions(self):
        waveform = self._body_waveform(p0=12.0, e0=0.0, spin=None)
        primary, secondary, _, _ = self._drawn_bodies(waveform)
        self.assertAlmostEqual(primary, 0.043)
        self.assertAlmostEqual(secondary, 0.014)

    def test_clamped_bodies_keep_their_size_ratio(self):
        waveform = self._body_waveform(p0=12.0, e0=0.8, spin=None)
        primary, secondary, _, _ = self._drawn_bodies(waveform)
        self.assertLess(primary, 0.043)
        self.assertAlmostEqual(secondary / primary, 0.014 / 0.043)


class PostAdiabaticBranchTest(unittest.TestCase):
    """Test the Trajectory1PAT1R (post-adiabatic) branch in generate_relativistic_mode_waveform."""

    def _build_mock_generator(self, n_sparse=10, n_modes=2):
        """Build a mock FEW generator that mimics a Waveform1PAT1R model."""

        t_sparse = np.linspace(0.0, 100.0, n_sparse)
        p_sparse = np.linspace(10.0, 9.0, n_sparse)
        e_sparse = np.zeros(n_sparse)
        xI_sparse = np.ones(n_sparse)
        Phi_phi_sparse = np.linspace(0.0, 6.0, n_sparse)
        Phi_theta_sparse = np.zeros(n_sparse)
        Phi_r_sparse = np.linspace(0.0, 3.0, n_sparse)
        delta_m1_sparse = np.linspace(0.0, 0.01, n_sparse)
        chit_sparse = np.linspace(0.0, 0.001, n_sparse)  # stored as delta

        # Nine-column trajectory for post-adiabatic models
        trajectory = [
            t_sparse, p_sparse, e_sparse, xI_sparse,
            Phi_phi_sparse, Phi_theta_sparse, Phi_r_sparse,
            delta_m1_sparse, chit_sparse,
        ]

        # A mock class whose __name__ is "Trajectory1PAT1R"
        class Trajectory1PAT1R:
            args = {"chit1": 0.9, "nu": 1e-5, "chit2": 0.0}

        # Build the mock generator hierarchy
        generator = MagicMock()
        type(generator).__name__ = "Waveform1PAT1R"
        generator.inspiral_kwargs = {}
        generator.sanity_check_init.return_value = (0.9, 1.0)
        generator.sanity_check_traj.return_value = None
        generator.inspiral_generator.return_value = trajectory
        generator.inspiral_generator.func = Trajectory1PAT1R()

        # Dense trajectory: columns are p, e, xI, Phi_phi, Phi_theta, Phi_r
        n_dense = int((t_sparse[-1] - t_sparse[0]) / 10.0) + 1
        dense = np.zeros((n_dense, 6))
        dense[:, 0] = np.linspace(10.0, 9.0, n_dense)  # p
        dense[:, 2] = 1.0  # xI
        dense[:, 3] = np.linspace(0.0, 6.0, n_dense)  # Phi_phi
        dense[:, 5] = np.linspace(0.0, 3.0, n_dense)  # Phi_r
        generator.inspiral_generator.inspiral_generator.eval_integrator_spline.return_value = dense

        # Amplitude generator with get_amplitudes for the post-adiabatic path
        amp_gen = generator.amplitude_generator
        amplitudes = np.random.default_rng(42).random((n_sparse, n_modes))
        amp_gen.get_amplitudes.return_value = amplitudes
        amp_gen.l_arr_no_mask = np.array([2, 2])
        amp_gen.m_arr_no_mask = np.array([2, 1])
        amp_gen.n_arr_no_mask = np.array([0, 0])

        return generator

    @patch("fewview._core._resolve_few_model")
    def test_post_adiabatic_trajectory_is_unpacked(self, mock_resolve):
        """The nine-column trajectory is correctly unpacked and amplitudes use evolving spin."""

        mock_generator = self._build_mock_generator()
        mock_resolve.return_value = mock_generator

        # Mock the constants imported inside the function
        constants_mock = MagicMock()
        constants_mock.Gpc = 3.086e25
        constants_mock.MRSUN_SI = 1477.0

        with patch.dict("sys.modules", {"few.utils.constants": constants_mock}):
            waveform = generate_relativistic_mode_waveform(
                M=1e6, mu=10.0, a=0.9, p0=10.0, e0=0.0, T=0.001, dt=10.0,
                model="Waveform1PAT1R",
            )

        self.assertIsInstance(waveform, RelativisticModeWaveform)
        self.assertEqual(waveform.model, "Waveform1PAT1R")
        # Verify the post-adiabatic amplitude path was used
        mock_generator.amplitude_generator.get_amplitudes.assert_called_once()
        call_kwargs = mock_generator.amplitude_generator.get_amplitudes.call_args
        # Should have nu, chit2, chit, delta_m1 keyword arguments
        self.assertIn("nu", call_kwargs.kwargs)
        self.assertIn("chit2", call_kwargs.kwargs)
        self.assertIn("chit", call_kwargs.kwargs)
        self.assertIn("delta_m1", call_kwargs.kwargs)
        # chit should be delta + chit1 baseline
        chit_passed = call_kwargs.kwargs["chit"]
        expected_chit1 = 0.9
        self.assertTrue(np.all(chit_passed >= expected_chit1 - 1e-10))

    @patch("fewview._core._resolve_few_model")
    def test_post_adiabatic_waveform_has_modes(self, mock_resolve):
        """Post-adiabatic branch produces a waveform with the expected mode structure."""

        mock_generator = self._build_mock_generator()
        mock_resolve.return_value = mock_generator

        constants_mock = MagicMock()
        constants_mock.Gpc = 3.086e25
        constants_mock.MRSUN_SI = 1477.0

        with patch.dict("sys.modules", {"few.utils.constants": constants_mock}):
            waveform = generate_relativistic_mode_waveform(
                M=1e6, mu=10.0, a=0.9, p0=10.0, e0=0.0, T=0.001, dt=10.0,
                model="Waveform1PAT1R",
            )

        # Should have modes with negative-m partners included
        self.assertGreater(waveform.modes.shape[1], 0)
        self.assertEqual(waveform.time.shape[0], waveform.modes.shape[0])
        # Trajectory should be populated
        self.assertTrue(waveform.has_trajectory)


if __name__ == "__main__":
    unittest.main()
