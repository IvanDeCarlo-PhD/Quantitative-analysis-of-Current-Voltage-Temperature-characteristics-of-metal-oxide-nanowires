###############################################################################
# Copyright (C) 2026 by
# Ivan De Carlo
# 
# This work is licensed under a  
# 
# Creative Commons Attribution Non-Commercial 4.0 License
# ( https://creativecommons.org/licenses/by-nc/4.0/ )
# 
#
###############################################################################
"""Plot a simulated MSM I-V curve from manually specified model parameters.

Put this script next to msm_model.py, edit the USER INPUT section, and run it
in Spyder or with ``python run_model.py``.
No experimental data or fitting script is needed.

Requirements: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from msm_model import M0, solve_curve

# =============================================================================
# USER INPUT: replace these example numbers with the parameters for your device.
# All values use SI units, except energies (eV) and temperature (K).
# =============================================================================
PARAMS = {
    "T": 300.0,                       # temperature [K]
    "S1": 60e-9 * 550e-9,             # contact 1 area [m^2]
    "S2": 60e-9 * 550e-9,             # contact 2 area [m^2]
    "m_star": 0.78 * M0,               # effective mass [kg]
    "Rnw": 1.0e9,                     # nanowire resistance [ohm]
    "Rsh1": 1.0e10,                   # contact 1 shunt resistance [ohm]
    "Rsh2": 1.0e10,                   # contact 2 shunt resistance [ohm]
    "phi1_eV": 0.43,                  # contact 1 Schottky barrier [eV]
    "phi2_eV": 0.49,                  # contact 2 Schottky barrier [eV]
    "eta1": 3.0,                      # contact 1 ideality factor
    "eta2": 3.0,                      # contact 2 ideality factor
    "xi_eV": 0.20,                    # TFE energy offset [eV]
    "E00_eV": 6.7e-3,                 # TFE characteristic energy [eV]
}

V_MIN = -10.0                         # minimum bias [V]
V_MAX = 10.0                          # maximum bias [V]
N_POINTS = 201                        # number of simulated voltage points

SAVE_PNG = True
SAVE_CSV = True
# =============================================================================


def simulate_and_plot(params, v_min=V_MIN, v_max=V_MAX, n_points=N_POINTS):
    """Return (voltage [V], current [A], Matplotlib figure) for one curve."""
    required = (
        "T", "S1", "S2", "m_star", "Rnw", "Rsh1", "Rsh2",
        "phi1_eV", "phi2_eV", "eta1", "eta2", "xi_eV", "E00_eV",
    )
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError("Missing model parameters: " + ", ".join(missing))
    if not all(np.isfinite(params[name]) for name in required):
        raise ValueError("Every model parameter must be a finite number")
    strictly_positive = (
        "T", "S1", "S2", "m_star", "Rnw", "Rsh1", "Rsh2",
        "eta1", "eta2", "E00_eV",
    )
    if any(params[name] <= 0 for name in strictly_positive):
        raise ValueError("Temperature, areas, mass, resistances, ideality factors, and E00_eV must be positive")
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min >= v_max:
        raise ValueError("Set finite V_MIN < V_MAX")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("N_POINTS must be an integer of at least 2")

    voltage = np.linspace(v_min, v_max, n_points)
    _, _, current = solve_curve(voltage, params)
    if not np.all(np.isfinite(current)):
        raise RuntimeError("The MSM solver produced a nonfinite current; check your parameters")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(voltage, current, color="tab:blue", linewidth=1.8, label="MSM model")
    ax.axhline(0, color="0.4", linewidth=0.7)
    ax.axvline(0, color="0.4", linewidth=0.7)
    ax.set(xlabel="Bias voltage (V)", ylabel="Current (A)", title=f"MSM I-V curve at {params['T']:g} K")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return voltage, current, fig


def main():
    voltage, current, fig = simulate_and_plot(PARAMS, V_MIN, V_MAX, N_POINTS)
    output_folder = Path(__file__).resolve().parent
    if SAVE_PNG:
        path = output_folder / "msm_simulated_curve.png"
        fig.savefig(path, dpi=250)
        print(f"Saved plot: {path}")
    if SAVE_CSV:
        path = output_folder / "msm_simulated_curve.csv"
        np.savetxt(
            path, np.column_stack((voltage, current)), delimiter=",",
            header="V_bias_V,I_A", comments="", fmt="%.12g",
        )
        print(f"Saved simulated data: {path}")
    plt.show()


if __name__ == "__main__":
    main()
