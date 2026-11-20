"""Read saved FEWview mode archives, including early archives without metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .waveform import RelativisticModeWaveform


def _optional_float(data: np.lib.npyio.NpzFile, name: str):
    return float(data[name]) if name in data.files else None


def load_mode_waveform(filename: Path) -> tuple[RelativisticModeWaveform, np.ndarray]:
    """Load the portable mode archive written by fewview-render."""

    with np.load(filename, allow_pickle=False) as data:
        waveform = RelativisticModeWaveform(
            time=np.asarray(data["time"]),
            modes=np.asarray(data["modes"]),
            ell=np.asarray(data["ell"]),
            m=np.asarray(data["m"]),
            teukolsky_modes_used=int(data.get("teukolsky_modes_used", 0)),
            teukolsky_modes_available=int(
                data.get("teukolsky_modes_available", 0)
            ),
            retained_power_fraction=float(
                data.get("retained_power_fraction", 1.0)
            ),
            trajectory_p=data.get("trajectory_p"),
            trajectory_e=data.get("trajectory_e"),
            trajectory_xI=data.get("trajectory_xI"),
            trajectory_phi_phi=data.get("trajectory_phi_phi"),
            trajectory_phi_r=data.get("trajectory_phi_r"),
            primary_mass=_optional_float(data, "primary_mass"),
            secondary_mass=_optional_float(data, "secondary_mass"),
            spin=_optional_float(data, "spin"),
            model=str(data.get("model", "FastKerrEccentricEquatorialFlux")),
        )
        if "h_plus_reference" in data and "h_cross_reference" in data:
            reference = np.asarray(data["h_plus_reference"]) - 1j * np.asarray(
                data["h_cross_reference"]
            )
        else:
            reference = waveform.strain(theta=np.pi / 3.0, phi=0.0)
    return waveform, reference
