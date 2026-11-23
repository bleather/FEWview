import inspect
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import fewview
from fewview.cli import animate, cluster_job, cluster_segment, render_still
from fewview._credit import add_figure_credit, prepare_frame_logo, stamp_frame


class RenderCreditTests(unittest.TestCase):
    def test_logos_share_an_inset_and_are_composited_over_the_scene(self):
        for width, height in ((640, 400), (400, 640), (1280, 800)):
            word = prepare_frame_logo(width, height, "black", "wordmark")
            icon = prepare_frame_logo(width, height, "black", "icon")
            self.assertEqual(word.height, icon.height)
            self.assertGreater(word.width, 2 * icon.width)
            for mark in (word, icon):
                frame = np.full((height, width, 3), 42, dtype=np.uint8)
                frame.setflags(write=False)
                output = stamp_frame(frame, mark)
                x = width - round(width * 0.015) - mark.width
                y = height - round(height * 0.015) - mark.height
                region = output[y:y + mark.height, x:x + mark.width]
                alpha = np.asarray(mark)[..., 3]
                self.assertTrue(np.any(region != 42))
                self.assertTrue(np.all(region[alpha == 0] == 42))
                self.assertTrue(np.all(output[:y] == 42))
                self.assertTrue(np.all(output[:, :x] == 42))
                self.assertTrue(np.all(frame == 42))

    def test_wordmark_contrast_and_invalid_choice(self):
        on_dark = prepare_frame_logo(640, 400, "black", "wordmark")
        on_light = prepare_frame_logo(640, 400, "white", "wordmark")
        self.assertFalse(np.array_equal(on_dark, on_light))
        with self.assertRaisesRegex(ValueError, "wordmark.*icon"):
            prepare_frame_logo(640, 400, "black", "none")

    def test_public_interface_defaults_to_wordmark(self):
        for render in (fewview.plot_strain_surface, fewview.plot_volume_slice, fewview.render_volume,
                       fewview.render_mode_frame, fewview.render_mode_animation):
            self.assertEqual(inspect.signature(render).parameters["logo"].default, "wordmark")
        for command in (animate, cluster_job, cluster_segment, render_still):
            option = command.build_parser()._option_string_actions["--logo"]
            self.assertEqual(option.default, "wordmark")
            self.assertEqual(option.choices, ("wordmark", "icon"))

    def test_multipanel_figure_has_one_logo(self):
        fig = Figure(figsize=(7, 6))
        canvas = FigureCanvasAgg(fig)
        fig.subplots(2)
        add_figure_credit(fig, "wordmark")
        add_figure_credit(fig, "icon")
        canvas.draw()
        self.assertEqual([a.get_gid() for a in fig.artists], ["fewview-render-credit"])
