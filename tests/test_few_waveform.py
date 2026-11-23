"""End-to-end check of the reconstructed modes against FEW's own waveform.

Run with the opt-in few_data tests and a working FEW CPU backend.
"""

import unittest

import numpy as np
import pytest

from fewview import generate_relativistic_mode_waveform


@pytest.mark.few_data
class FewWaveformAgreementTest(unittest.TestCase):
    def test_off_equator_strain_matches_few(self):
        """Odd-l modes and their -m partners must reproduce FEW off the equator.

        At theta = pi/3 a wrong sign on the odd-l negative-m partners gives a
        ~7% mismatch; the correct reconstruction agrees to round-off.
        """

        from few.utils.ylm import GetYlms
        from few.waveform import FastKerrEccentricEquatorialFlux

        generator = FastKerrEccentricEquatorialFlux(lmax=10, nmax=55, force_backend="cpu")
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

        self.assertEqual(strain.shape, reference.shape)
        mismatch = np.linalg.norm(reference - strain) / np.linalg.norm(reference)
        self.assertLess(mismatch, 1e-10)


if __name__ == "__main__":
    unittest.main()
