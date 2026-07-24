"""Tests for the exact Boyer-Lindquist trajectory reconstruction.

The reconstruction (Lynch & Burke, arXiv:2411.04955) is validated against a
direct numerical integration of the equatorial Kerr geodesic equations: feeding
the integrated BL-time radial phase back through the conversion must return the
integrated radius to near machine precision.
"""

import unittest

import numpy as np
from scipy.integrate import solve_ivp

from few.utils.utility import (
    get_fundamental_frequencies,
    get_kerr_geo_constants_of_motion,
)

from fewview.geodesic import boyer_lindquist_equatorial


def _integrate_geodesic(a, p, e, n_points=2000):
    """Directly integrate an equatorial Kerr geodesic; return (r, phi, Phi_r)."""

    E, L, Q = (float(np.asarray(v)) for v in get_kerr_geo_constants_of_motion(a, p, e, 1.0))
    omega_phi, _, omega_r = (
        float(np.asarray(v)) for v in get_fundamental_frequencies(a, p, e, 1.0)
    )
    ra, rp = p / (1 - e), p / (1 + e)
    inv = 1 / (1 - E * E)
    r3 = inv - 0.5 * (ra + rp) + np.sqrt((inv - 0.5 * (ra + rp)) ** 2 - a * a * Q / (ra * rp * (1 - E * E)))
    r4 = a * a * Q / (ra * rp * r3 * (1 - E * E)) if r3 > 0 else 0.0
    p3, p4 = r3 * (1 - e), r4 * (1 + e)

    def rhs(_lam, y):
        psi, t, phi = y
        r = p / (1 + e * np.cos(psi))
        fr = (1 / (1 - e * e)) * np.sqrt(
            (1 - E * E)
            * max((p - p3) - e * (p + p3 * np.cos(psi)), 0.0)
            * max((p - p4) + e * (p - p4 * np.cos(psi)), 0.0)
        )
        Delta = r * r - 2 * r + a * a
        B = E * (r * r + a * a) - a * L
        ft = (r * r + a * a) * B / Delta - a * a * E + a * L
        fphi = a * B / Delta + L - a * E
        return [fr, ft, fphi]

    lam_max = 3 * 2 * np.pi / rhs(np.pi, [np.pi, 0, 0])[0]
    sol = solve_ivp(rhs, [0, lam_max], [0, 0, 0], dense_output=True, rtol=1e-11, atol=1e-12)
    lam = np.linspace(0, lam_max, n_points)
    psi, t, phi = sol.sol(lam)
    return p / (1 + e * np.cos(psi)), phi, omega_r * t, omega_phi * t


class GeodesicTest(unittest.TestCase):
    CASES = [(0.9, 12.0, 0.4), (0.9, 12.0, 0.7), (0.998, 8.0, 0.5), (0.0, 15.0, 0.6)]

    def test_matches_direct_integration(self):
        for a, p, e in self.CASES:
            r_ode, phi_ode, Phi_r, Phi_phi = _integrate_geodesic(a, p, e)
            r_rec, phi_rec = boyer_lindquist_equatorial(a, p, e, Phi_r, Phi_phi)
            radial_range = p / (1 - e) - p / (1 + e)
            self.assertLess(
                np.max(np.abs(r_rec - r_ode)) / radial_range, 1e-8,
                f"radius mismatch at a={a}, p={p}, e={e}",
            )
            dphi = np.abs(((phi_rec - phi_ode) + np.pi) % (2 * np.pi) - np.pi)
            self.assertLess(np.max(dphi), 1e-6, f"phi mismatch at a={a}, p={p}, e={e}")

    def test_turning_points(self):
        # Phi_r = 0 -> periapsis; Phi_r = pi -> apoapsis.
        a, p, e = 0.9, 12.0, 0.5
        r, _ = boyer_lindquist_equatorial(a, p, e, np.array([0.0, np.pi]), np.array([0.0, 0.0]))
        self.assertAlmostEqual(r[0], p / (1 + e), places=6)
        self.assertAlmostEqual(r[1], p / (1 - e), places=6)

    def test_differs_from_naive_approximation(self):
        # The exact reconstruction should meaningfully depart from r = p/(1+e cos Phi_r).
        a, p, e = 0.9, 12.0, 0.5
        Phi_r = np.linspace(0, 4 * np.pi, 400)
        r_exact, _ = boyer_lindquist_equatorial(a, p, e, Phi_r, np.zeros_like(Phi_r))
        r_naive = p / (1 + e * np.cos(Phi_r))
        radial_range = p / (1 - e) - p / (1 + e)
        self.assertGreater(np.max(np.abs(r_exact - r_naive)) / radial_range, 0.1)

    def test_stays_within_radial_range(self):
        a, p, e = 0.998, 9.0, 0.6
        Phi_r = np.linspace(0, 20.0, 1000)
        r, _ = boyer_lindquist_equatorial(a, p, e, Phi_r, np.zeros_like(Phi_r))
        self.assertGreaterEqual(r.min(), p / (1 + e) - 1e-6)
        self.assertLessEqual(r.max(), p / (1 - e) + 1e-6)


if __name__ == "__main__":
    unittest.main()
