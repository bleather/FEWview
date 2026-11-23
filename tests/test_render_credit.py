import unittest

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

    def test_multipanel_figure_has_one_credit(self):
        fig = Figure(figsize=(7, 6))
        canvas = FigureCanvasAgg(fig)
        fig.subplots(2)
        add_figure_credit(fig)
        add_figure_credit(fig)
        canvas.draw()
        self.assertEqual([a.get_gid() for a in fig.artists], ["fewview-render-credit"])
