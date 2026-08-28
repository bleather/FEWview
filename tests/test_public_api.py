"""Smoke tests for the public Fewview surface and the renamed palettes."""

import inspect
import unittest

import fewview
from fewview._core import _resolve_few_model


class PublicApiTest(unittest.TestCase):
    def test_version_is_exposed(self):
        self.assertRegex(fewview.__version__, r"^\d+\.\d+")

    def test_generate_accepts_model_parameter(self):
        sig = inspect.signature(fewview.generate_relativistic_mode_waveform)
        self.assertIn("model", sig.parameters)
        self.assertEqual(
            sig.parameters["model"].default, "FastKerrEccentricEquatorialFlux"
        )

    def test_resolve_few_model_passes_through_instances(self):
        sentinel = object()  # an already-instantiated model is returned as-is
        self.assertIs(_resolve_few_model(sentinel, 10, 55, "cpu"), sentinel)

    def test_resolve_few_model_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            _resolve_few_model("NotARealFewModel", 10, 55, "cpu")

    def test_all_names_are_importable(self):
        for name in fewview.__all__:
            self.assertTrue(hasattr(fewview, name), name)

    def test_submodules_reexport_core(self):
        from fewview import rendering, surface, volume, waveform

        self.assertIs(
            waveform.generate_relativistic_mode_waveform,
            fewview.generate_relativistic_mode_waveform,
        )
        self.assertIs(volume.build_mode_retarded_time_volume,
                      fewview.build_mode_retarded_time_volume)
        self.assertIs(rendering.render_mode_animation,
                      fewview.render_mode_animation)
        self.assertIs(surface.StrainSurface, fewview.StrainSurface)

    def test_renamed_palettes_present_and_old_names_gone(self):
        schemes = set(fewview.available_color_schemes())
        self.assertLessEqual({"rainbow", "aurora", "cinematic"}, schemes)
        self.assertNotIn("gwpv_rainbow", schemes)
        self.assertNotIn("magma_glow", schemes)

    def test_opacity_profiles_and_presentations(self):
        self.assertEqual(
            fewview.OPACITY_PROFILES, ("soft", "bands", "shells", "flux")
        )
        self.assertIn("shells_dramatic", fewview.PRESENTATIONS)


if __name__ == "__main__":
    unittest.main()
