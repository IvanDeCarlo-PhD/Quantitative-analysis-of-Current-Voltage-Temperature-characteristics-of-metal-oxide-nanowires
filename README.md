# MSM I–V model: guide to the three scripts

These scripts model current through a nanowire between two Schottky contacts. You can either calculate a curve from known parameters or fit parameters from measured curves.


 `msm_model.py` - Implements the transport equations and solves the circuit at each voltage. 
  `fit_msm_global.py` - Finds model parameters by fitting measured I–V curves at several temperatures. 
 `run_model.py` - Calculates and plots one I–V curve using parameters you enter. 

## Set up

Use Python 3.10 or later and install the dependencies in the Python environment used by Spyder or your terminal:

```bash
python -m pip install numpy scipy matplotlib
```

Place `msm_model.py`, `fit_msm_global.py`, and `run_model.py` in the same folder. The fitting and plotting scripts both import `msm_model.py`.

## 1. `msm_model.py`: the numerical model

This file describes two back-to-back metal–semiconductor contacts connected by a nanowire resistance `Rnw`. Each contact has a parallel shunt resistance; the model uses thermionic emission (TE) for forward bias and thermionic-field emission (TFE) for reverse bias. At every applied voltage, it numerically solves for the current and the voltage drops across the contacts.

Its main function is `solve_curve(V_bias, params)`. It takes an array of bias voltages in volts and a dictionary containing the 13 parameters listed below. It returns `(V1, V2, I)`: the **positive magnitudes** of the two contact voltage drops and the **signed** current in amperes, in the same order as the input voltages. `run_model.py` uses the returned current to draw the graph; `fit_msm_global.py` calls the same function repeatedly during fitting.

If you use the model from your own script, define `params` as in Section 2 and call it like this:

```python
import numpy as np
from msm_model import solve_curve

voltages = np.linspace(-10.0, 10.0, 201)
V1_magnitude, V2_magnitude, current_A = solve_curve(voltages, params)
```

You normally do not run `msm_model.py` itself. It does not read experimental files or produce a plot on its own.

## 2. `run_model.py`: plot from known parameters

Use this script when you already have parameter values. It needs `msm_model.py`, but it does **not** need experimental data or `fit_msm_global.py`.

1. Open `run_model.py` in Spyder or a text editor.
2. In the **USER INPUT** section, replace every example value in `PARAMS` with the corresponding value for your device. In particular, set the desired temperature `T`.
3. Set `V_MIN`, `V_MAX`, and `N_POINTS` for the desired voltage sweep. The defaults are −10 to +10 V with 201 points.
4. Run the file in Spyder (F5) or from a terminal:

   ```bash
   python run_model.py
   ```

The graph is displayed and, by default, the following files are saved beside `run_model.py`:

- `msm_simulated_curve.png`: I–V plot.
- `msm_simulated_curve.csv`: two columns, `V_bias_V` and `I_A`, suitable for further plotting or analysis.

Set `SAVE_PNG` or `SAVE_CSV` to `False` if you do not want the respective file. Each run overwrites the previous files with those names. The values already present in `PARAMS` are **illustrative**, not fitted values for your device.

### Model parameters


 `T` - Temperature of this curve , (K) 
 `S1`, `S2` - Areas of contacts 1 and 2, (m^2)
 `m_star` - Effective carrier mass; `M0` imported from `msm_model` is the free electron mass, (kg)
 `Rnw` - Nanowire series resistance, (Ω) 
 `Rsh1`, `Rsh2` - Shunt resistance at each contact, (Ω)
 `phi1_eV`, `phi2_eV` - Schottky barrier heights, (eV)
 `eta1`, `eta2` - Contact ideality factors
 `xi_eV` - TFE energy offset, (eV)
 `E00_eV` - TFE characteristic energy, (eV)

All 13 entries must be present in `PARAMS`. Resistances, contact areas, effective mass, temperature, ideality factors, and `E00_eV` must be positive. 

Here is the complete editable block, with **example numbers only**:

```python
PARAMS = {
    "T": 300.0,                     # K
    "S1": 60e-9 * 550e-9,           # m^2
    "S2": 60e-9 * 550e-9,           # m^2
    "m_star": 0.78 * M0,            # kg; M0 is imported from msm_model
    "Rnw": 1.0e9,                   # ohm
    "Rsh1": 1.0e10,                 # ohm
    "Rsh2": 1.0e10,                 # ohm
    "phi1_eV": 0.43,                # eV
    "phi2_eV": 0.49,                # eV
    "eta1": 3.0,                    # dimensionless
    "eta2": 3.0,                    # dimensionless
    "xi_eV": 0.20,                  # eV
    "E00_eV": 6.7e-3,               # eV
}
```

For use in another Python script, `simulate_and_plot(params, v_min, v_max, n_points)` returns the voltage array, current array, and Matplotlib figure.

## 3. `fit_msm_global.py`: determine parameters from measurements

Use this script if you have experimental I–V curves at several temperatures and want to estimate the model parameters. It imports `msm_model.py` and runs a two-stage global fit; `run_model.py` is not involved in fitting.

Prepare **one** tab-delimited `.txt` file with repeated, adjacent three-column blocks:

```text
Temperature    Current    Voltage    Temperature    Current    Voltage    ...
[K]            [A]        [V]        [K]            [A]        [V]        ...
```

Each block is one I–V curve. The first two rows give names and units. The loader skips any later metadata above the first numeric row of each block, then reads the numeric temperature, current, and voltage values. Currents must be in amperes and voltages in volts.

1. In the **USER CONFIGURATION** section of `fit_msm_global.py`, leave `DATA_FILE = None` to select the combined text file in a graphical dialog, or give its full path, for example `DATA_FILE = r"C:\path\to\Figure 3.txt"`.
2. Set the nanowire diameter, the two contact overlap lengths, and the effective mass to your device values. Review the parameter bounds and the preliminary low- and high-bias windows.
3. Run the script in Spyder (F5) or use `python fit_msm_global.py` from a terminal.

For each curve, the fitter estimates the shunt resistance from the low-bias slope and a starting value for `Rnw` from the high-bias slope. It uses the same shunt estimate for both contacts by default. It then fits `phi1_eV`, `phi2_eV`, `E00_eV`, and `xi_eV` across **all** temperatures. `eta1` and `eta2` are fitted separately for each curve; `Rnw` is fixed at its preliminary estimate in Stage 1 and also fitted separately for each curve in Stage 2.

Results appear in `fit_global_output` beside the input text file. The main files are `global_parameters.csv` (four shared fitted parameters), `temperature_dependent_parameters.csv` (temperature and per-curve parameters), `fit_report.txt`, and individual simulated CSV files and fit plots. The default preliminary windows are −2 to +2 V (low bias) and |V| from 8 to 10 V (high bias); each linear estimate requires at least five points in its window. Adjust those windows if your data do not cover them.

## Citation

If you use these codes in a publication, cite the associated publication and Zenodo code archive DOI.
