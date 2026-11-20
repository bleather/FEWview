"""End-to-end check of the reconstructed modes against FEW's own waveform.

Needs a working FastEMRIWaveforms CPU backend; skipped otherwise.
"""

import unittest

import numpy as np

from fewview import generate_relativistic_mode_waveform


def _few_cpu_waveform_model():
    try:
        from few.waveform import FastKerrEccentricEquatorialFlux

        return FastKerrEccentricEquatorialFlux(lmax=10, nmax=55, force_backend="cpu")
    except Exception as error:  # missing FEW, backend or data files
        raise unittest.SkipTest(f"FEW CPU backend unavailable: {error}")


class FewWaveformAgreementTest(unittest.TestCase):
    def test_off_equator_strain_matches_few(self):
        """Odd-l modes and their -m partners must reproduce FEW off the equator.

        At theta = pi/3 a wrong sign on the odd-l negative-m partners gives a
        ~7% mismatch; the correct reconstruction agrees to round-off.
        """

        from few.utils.ylm import GetYlms

        generator = _few_cpu_waveform_model()
        M, mu, a, p0, e0, dt, T = 1e6, 10.0, 0.9, 10.0, 0.3, 10.0, 0.002
        theta, phi = np.pi / 3.0, 0.7

        reference = np.asarray(
            generator(
                M, mu, a, p0, e0, 1.0, theta=theta, phi=phi, dt=dt, T=T,
                mode_selection="all",
            )
        )
        waveform = generate_relativistic_mode_waveform(
            M, mu, a, p0, e0, 1.0, dist=None, dt=dt, T=T, model=generator
        )
        ylm = GetYlms(force_backend="cpu")
        harmonics = np.array(
            [ylm(int(l), int(m), theta, phi) for l, m in zip(waveform.ell, waveform.m)]
        )
        strain = waveform.modes @ harmonics

        n = min(reference.size, strain.size)
        mismatch = np.linalg.norm(reference[:n] - strain[:n]) / np.linalg.norm(
            reference[:n]
        )
        self.assertLess(mismatch, 1e-10)


if __name__ == "__main__":
    unittest.main()
