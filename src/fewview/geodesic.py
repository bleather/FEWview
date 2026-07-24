r"""Exact Boyer-Lindquist trajectory for equatorial Kerr geodesics.

FastEMRIWaveforms returns *Boyer-Lindquist-time action angles*
:math:`(\Phi_r, \Phi_\phi)` for its Kerr eccentric-equatorial trajectory. These
have no closed-form relation to the secondary's actual position. This module
implements the conversion of Lynch & Burke, *A note on the conversion of orbital
angles for extreme mass ratio inspirals* (`arXiv:2411.04955
<https://arxiv.org/abs/2411.04955>`_), specialised to the equatorial plane, to
recover the true BL coordinates :math:`(r, \phi)`.

The scheme (their Section 3), for an equatorial orbit where the polar
oscillations vanish identically:

1. From FEW's radial action angle :math:`\Phi_r`, numerically invert
   :math:`\Phi_r = q_r + \Omega_r\, t_r(q_r)` (Eq. 22) for the Mino-time action
   angle :math:`q_r`. :math:`t_r` is the oscillatory time function of Appendix A.
2. Evaluate the radius directly from :math:`q_r` (Eq. 13a).
3. Correct the azimuth, :math:`\phi = \Phi_\phi - \Omega_\phi\, t_r(q_r) +
   \phi_r(q_r)` (Eqs. 22, 23, 16b), with :math:`\phi_r` from Appendix A.

Equation numbers refer to the paper. Geometric units with :math:`M=m_1=1` are
used throughout; ``r`` is returned in units of ``M``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipj, elliprd, elliprf, elliprj

__all__ = ["boyer_lindquist_equatorial"]


# --- Elliptic integral of the third kind via Carlson symmetric forms ---------
# scipy lacks Pi directly, but Pi(n; phi | m) = sin(phi) R_F(cos^2 phi,
# 1 - m sin^2 phi, 1) + (n/3) sin^3 phi R_J(...), with m the *parameter* (not
# modulus), matching the paper's K(m) = F(pi/2 | m) convention.


def _ellippi_complete(n, m):
    """Complete elliptic integral of the third kind ``Pi(n | m)``."""

    return elliprf(0.0, 1.0 - m, 1.0) + (n / 3.0) * elliprj(0.0, 1.0 - m, 1.0, 1.0 - n)


def _ellippi_principal(n, phi, m):
    """``Pi(n; phi | m)`` for ``phi`` in ``[-pi/2, pi/2]`` (Carlson forms)."""

    sin_phi = np.sin(phi)
    sin2 = sin_phi * sin_phi
    cos2 = np.cos(phi) ** 2
    return sin_phi * elliprf(cos2, 1.0 - m * sin2, 1.0) + (n / 3.0) * sin_phi * sin2 * elliprj(
        cos2, 1.0 - m * sin2, 1.0, 1.0 - n * sin2
    )


def _ellippi_incomplete(n, phi, m):
    """Incomplete elliptic integral of the third kind ``Pi(n; phi | m)``.

    The Carlson form is valid only on the principal branch ``[-pi/2, pi/2]``.
    General ``phi`` is reduced with the quasi-period
    ``Pi(n; phi + k pi | m) = Pi(n; phi | m) + 2 k Pi(n | m)``.
    """

    phi = np.asarray(phi, dtype=float)
    k = np.round(phi / np.pi)
    phi_red = phi - k * np.pi
    return 2.0 * k * _ellippi_complete(n, m) + _ellippi_principal(n, phi_red, m)


# --- Incomplete elliptic integral of the second kind via Carlson forms -------
# scipy's ellipeinc has isolated accuracy glitches for some arguments in
# vectorised calls, so it is reimplemented with the robust Carlson R_D form.


def _ellipe_complete(m):
    """Complete elliptic integral of the second kind ``E(m)``."""

    return elliprf(0.0, 1.0 - m, 1.0) - (m / 3.0) * elliprd(0.0, 1.0 - m, 1.0)


def _ellipe_principal(phi, m):
    """``E(phi | m)`` for ``phi`` in ``[-pi/2, pi/2]`` (Carlson forms)."""

    sin_phi = np.sin(phi)
    sin2 = sin_phi * sin_phi
    cos2 = np.cos(phi) ** 2
    return sin_phi * elliprf(cos2, 1.0 - m * sin2, 1.0) - (m / 3.0) * sin_phi * sin2 * elliprd(
        cos2, 1.0 - m * sin2, 1.0
    )


def _ellipe_incomplete(phi, m):
    """Incomplete elliptic integral of the second kind ``E(phi | m)``."""

    phi = np.asarray(phi, dtype=float)
    k = np.round(phi / np.pi)
    phi_red = phi - k * np.pi
    return 2.0 * k * _ellipe_complete(m) + _ellipe_principal(phi_red, m)


def _radial_roots(a, p, e, E, L, Q):
    """Return the four radial-potential roots ``(ra, rp, r3, r4)`` (Eqs. 7, 8)."""

    ra = p / (1.0 - e)
    rp = p / (1.0 + e)
    one_minus_E2 = 1.0 - E * E
    half_sum = 0.5 * (ra + rp)
    inv = 1.0 / one_minus_E2  # M / (1 - E^2) with M = 1
    disc = (inv - half_sum) ** 2 - a * a * Q / (ra * rp * one_minus_E2)
    r3 = inv - half_sum + np.sqrt(np.clip(disc, 0.0, None))
    # r4 -> 0 in the equatorial plane (Q = 0); the general form is kept.
    r4 = np.where(r3 > 0.0, a * a * Q / (ra * rp * np.where(r3 == 0.0, 1.0, r3) * one_minus_E2), 0.0)
    return ra, rp, r3, r4


def _build_context(a, E, L, ra, rp, r3, r4):
    """Precompute all radial-phase-independent quantities for Eqs. 13a, A.4, A.5.

    Everything that does not depend on ``q_r`` (roots, moduli, complete elliptic
    integrals, prefactors) is evaluated once so the bisection loop only touches
    the incomplete integrals.
    """

    from scipy.special import ellipk

    r_plus = 1.0 + np.sqrt(1.0 - a * a)
    r_minus = 1.0 - np.sqrt(1.0 - a * a)
    kr = (ra - rp) * (r3 - r4) / ((ra - r3) * (rp - r4))  # Eq. 14a (parameter m)
    hr = (ra - rp) / (ra - r3)  # Eq. A.2a
    h_plus = hr * (r3 - r_plus) / (rp - r_plus)  # Eq. A.2b
    h_minus = hr * (r3 - r_minus) / (rp - r_minus)

    return {
        "a": a, "E": E, "L": L, "ra": ra, "rp": rp, "r3": r3, "r4": r4,
        "r_plus": r_plus, "r_minus": r_minus,
        "kr": kr, "hr": hr, "h_plus": h_plus, "h_minus": h_minus,
        "Kk": ellipk(kr),
        "pic_hr": _ellippi_complete(hr, kr),
        "pic_hp": _ellippi_complete(h_plus, kr),
        "pic_hm": _ellippi_complete(h_minus, kr),
        "ec": _ellipe_complete(kr),
        "t_prefac": -E / np.sqrt((1.0 - E * E) * (ra - r3) * (rp - r4)),
        "phi_prefac": 2.0 * a * E / ((r_plus - r_minus) * np.sqrt((1.0 - E * E) * (ra - r3) * (rp - r4))),
    }


def _sin_amplitude(q_r, c):
    """Return ``(sn^2, xi_r)`` at radial phase ``q_r`` (Jacobi sn and amplitude)."""

    sn, _cn, _dn, xi_r = ellipj(q_r * c["Kk"] / np.pi, c["kr"])
    return sn * sn, xi_r


def _t_r(q_r, c):
    """Oscillating time function ``t_r(q_r)`` (Eq. A.4), the bisection integrand."""

    kr, hr = c["kr"], c["hr"]
    ra, rp, r3, r4 = c["ra"], c["rp"], c["r3"], c["r4"]
    r_plus, r_minus = c["r_plus"], c["r_minus"]
    a, E, L = c["a"], c["E"], c["L"]
    q_over_pi = q_r / np.pi
    _sn2, xi = _sin_amplitude(q_r, c)

    pair_hr = q_over_pi * c["pic_hr"] - _ellippi_incomplete(hr, xi, kr)
    pair_hp = q_over_pi * c["pic_hp"] - _ellippi_incomplete(c["h_plus"], xi, kr)
    pair_hm = q_over_pi * c["pic_hm"] - _ellippi_incomplete(c["h_minus"], xi, kr)

    plus_term = (r_plus * (4.0 - a * L / E) - 2.0 * a * a) * pair_hp / ((rp - r_plus) * (r3 - r_plus))
    minus_term = (r_minus * (4.0 - a * L / E) - 2.0 * a * a) * pair_hm / ((rp - r_minus) * (r3 - r_minus))

    e_pair = q_over_pi * c["ec"] - _ellipe_incomplete(xi, kr)
    sin_xi = np.sin(xi)
    cos_xi = np.cos(xi)
    boundary = hr * sin_xi * cos_xi * np.sqrt(1.0 - kr * sin_xi * sin_xi) / (1.0 - hr * sin_xi * sin_xi)

    return c["t_prefac"] * (
        4.0 * (rp - r3) * pair_hr
        - 4.0 * (rp - r3) / (r_plus - r_minus) * (plus_term - minus_term)
        + (rp - r3) * (ra + rp + r3 + r4) * pair_hr
        + (ra - r3) * (rp - r4) * (boundary + e_pair)
    )


def _phi_r(q_r, c):
    """Oscillating azimuth function ``phi_r(q_r)`` (Eq. A.5)."""

    kr = c["kr"]
    rp, r3 = c["rp"], c["r3"]
    r_plus, r_minus = c["r_plus"], c["r_minus"]
    a, E, L = c["a"], c["E"], c["L"]
    q_over_pi = q_r / np.pi
    _sn2, xi = _sin_amplitude(q_r, c)
    pair_hp = q_over_pi * c["pic_hp"] - _ellippi_incomplete(c["h_plus"], xi, kr)
    pair_hm = q_over_pi * c["pic_hm"] - _ellippi_incomplete(c["h_minus"], xi, kr)
    return c["phi_prefac"] * (
        (rp - r3) * (2.0 * r_plus - a * L / E) * pair_hp / ((rp - r_plus) * (r3 - r_plus))
        - (rp - r3) * (2.0 * r_minus - a * L / E) * pair_hm / ((rp - r_minus) * (r3 - r_minus))
    )


def _radius_of_qr(q_r, c):
    """Boyer-Lindquist radius at radial phase ``q_r`` (Eq. 13a)."""

    ra, rp, r3 = c["ra"], c["rp"], c["r3"]
    sn2, _xi = _sin_amplitude(q_r, c)
    return (r3 * (ra - rp) * sn2 - rp * (ra - r3)) / ((ra - rp) * sn2 - (ra - r3))


def _invert_phi_r(phi_r_target, omega_r, c, *, iters=50):
    r"""Solve ``Phi_r = q_r + Omega_r t_r(q_r)`` for ``q_r`` (Eq. 22).

    ``t_r`` is 2-pi-periodic with zero mean and ``t_r(0)=0``, so the integer
    number of radial cycles ("bulk") is shared by ``Phi_r`` and ``q_r``; only
    the remainder in ``[0, 2 pi]`` is solved. The map ``q -> q + Omega_r t_r(q)``
    is strictly increasing there and brackets ``[0, 2 pi]``, so vectorised
    bisection converges unconditionally (no derivative, no division).
    """

    two_pi = 2.0 * np.pi
    bulk = np.floor(phi_r_target / two_pi) * two_pi
    rem = phi_r_target - bulk

    lo = np.zeros_like(rem)
    hi = np.full_like(rem, two_pi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = (mid + omega_r * _t_r(mid, c) - rem) < 0.0
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return bulk + 0.5 * (lo + hi)


def boyer_lindquist_equatorial(a, p, e, phi_r, phi_phi):
    r"""Reconstruct equatorial Kerr geodesic coordinates from FEW action angles.

    Args:
        a: Dimensionless Kerr spin (prograde, ``0 <= a < 1``).
        p, e: Semi-latus rectum (in ``M``) and eccentricity, per trajectory
            sample. Scalars or 1D arrays.
        phi_r, phi_phi: FEW's Boyer-Lindquist-time radial and azimuthal action
            angles, same shape as ``p``/``e``.

    Returns:
        ``(r, phi)`` in units of ``M`` and radians, matching the input shape:
        the true osculating-geodesic radius and azimuth, replacing the
        ``r = p/(1 + e cos Phi_r)`` approximation.
    """

    from few.utils.utility import (
        get_fundamental_frequencies,
        get_kerr_geo_constants_of_motion,
    )

    p = np.atleast_1d(np.asarray(p, dtype=float))
    e = np.atleast_1d(np.asarray(e, dtype=float))
    phi_r = np.atleast_1d(np.asarray(phi_r, dtype=float))
    phi_phi = np.atleast_1d(np.asarray(phi_phi, dtype=float))
    x = np.ones_like(p)
    a_arr = np.full_like(p, float(a))

    E, L, Q = (np.asarray(v, dtype=float) for v in get_kerr_geo_constants_of_motion(a_arr, p, e, x))
    omega_phi, _omega_theta, omega_r = (
        np.asarray(v, dtype=float) for v in get_fundamental_frequencies(a_arr, p, e, x)
    )

    ra, rp, r3, r4 = _radial_roots(float(a), p, e, E, L, Q)
    context = _build_context(float(a), E, L, ra, rp, r3, r4)

    q_r = _invert_phi_r(phi_r, omega_r, context)
    r = _radius_of_qr(q_r, context)
    phi = phi_phi - omega_phi * _t_r(q_r, context) + _phi_r(q_r, context)
    return r, phi
