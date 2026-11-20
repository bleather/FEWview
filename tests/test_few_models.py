"""Compare source-frame modes to the public FEW summation at the same sky position."""

import numpy as np
import pytest
from few import waveform as models

from fewview import generate_relativistic_mode_waveform


@pytest.mark.few_data
@pytest.mark.parametrize(
    "model_name,spin,eccentricity,inclination",
    [
        ("Circ1PAT1R", 1e-5, 0., 1.),
        ("FastSchwarzschildEccentricFlux", 0., 0.3, 1.),
        ("FastKerrEccentricEquatorialFlux", 0.9, 0.3, 1.),
        ("FastKerrEccentricEquatorialFlux", 0.9, 0.3, -1.),
    ],
)
def test_strain_matches_few(model_name, spin, eccentricity, inclination):
    model = getattr(models, model_name)(force_backend="cpu")
    circular = model_name == "Circ1PAT1R"
    source = (1e6, 10., spin, 10. if circular else 12., eccentricity, inclination)
    options = dict(dt=10., T=0.0001, dist=1.)
    theta, phi = 2., 0.3
    chi2 = 0.3 if circular else 0.
    waveform = generate_relativistic_mode_waveform(
        *source, model=model, chi2=chi2, **options,
        nmax=30 if model_name == "FastSchwarzschildEccentricFlux" else 55,
    )
    if circular:
        args = (*source[:4], theta, phi, chi2)
    elif model_name == "FastSchwarzschildEccentricFlux":
        args = (*source[:2], source[3], source[4], theta, phi)
    else:
        args = (*source, theta, phi)
    reference = model(*args, mode_selection="all", **options)
    strain = waveform.strain(theta, phi)
    assert strain.shape == reference.shape
    assert np.linalg.norm(strain - reference) / np.linalg.norm(reference) < 1e-7
    assert np.isfinite(waveform.orbital_position()).all()
    assert waveform.teukolsky_modes_used == waveform.teukolsky_modes_available


@pytest.mark.few_data
def test_legacy_model_name_and_mode_limits():
    waveform = generate_relativistic_mode_waveform(
        M=1e6, mu=10., a=1e-5, p0=10., e0=0., model="Waveform1PAT1R",
        T=0.0001, lmax=2, nmax=0,
    )
    assert waveform.model == "Circ1PAT1R"
    assert set(waveform.ell) == {2}
    assert waveform.teukolsky_modes_used < waveform.teukolsky_modes_available
