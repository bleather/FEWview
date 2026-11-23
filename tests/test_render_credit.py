import inspect
import unittest

import fewview
from fewview.cli import animate, cluster_job, cluster_segment, render_still

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fewview._credit import add_figure_credit, credit_strip


class RenderCreditTests(unittest.TestCase):
    def test_strip_scales_and_uses_a_contrasting_wordmark(self):
        for scale in (1, 2):
            for background, light_text in (("black", True), ("white", False)):
                strip = credit_strip(640, 400, scale, background)
                self.assertEqual(strip.shape[1:], (640 * scale, 3))
                self.assertEqual(strip.shape[0] % scale, 0)
                right = strip[:, -110 * scale:, :]
                grey = right.mean(axis=2)
                self.assertGreater(np.count_nonzero(grey > 120 if light_text else grey < 120), 40)
                self.assertTrue(np.any(strip[..., 2].astype(float) - strip[..., 0] > 35))

    def test_renderers_offer_no_credit_switch(self):
        for render in (fewview.plot_strain_surface, fewview.plot_volume_slice, fewview.render_volume,
                       fewview.render_mode_frame, fewview.render_mode_animation):
            self.assertNotIn("render_credit", inspect.signature(render).parameters)
        for command in (animate, cluster_job, cluster_segment, render_still):
            self.assertNotIn("--no-render-credit", command.build_parser()._option_string_actions)

    def test_multipanel_figure_has_one_credit(self):
        fig = Figure(figsize=(7, 6))
        canvas = FigureCanvasAgg(fig)
        fig.subplots(2)
        add_figure_credit(fig)
        add_figure_credit(fig)
        canvas.draw()
        self.assertEqual([a.get_gid() for a in fig.artists], ["fewview-render-credit"])

    def test_footer_right_edge_stays_put_between_portrait_and_landscape(self):
        for scale in (1, 2):
            edges = []
            for height in (400, 960):
                strip = credit_strip(640, height, scale, "black")
                columns = np.flatnonzero(np.any(strip > 100, axis=(0, 2)))
                edges.append(columns[-1])
            for edge in edges:
                self.assertAlmostEqual(edge, 0.985 * 640 * scale, delta=3)
