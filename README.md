# MSM I-V-T model and fitting code

Python code to simulate and fit temperature-dependent I-V characteristics of metal-semiconductor-metal (MSM) nanowire devices.

The two scripts have different roles:

- `msm_model.py` implements the MSM transport model and numerically solves the circuit. Given an array of applied voltages and a set of parameters, including temperature, `solve_curve(V_bias, params)` simulates a single I-V curve.
- `fit_msm_global.py` imports and repeatedly calls the solver in `msm_model.py` to fit experimental I-V curves at multiple temperatures. It reads all curves from one file, estimates the resistances, performs a two-stage global fit, and reports the extracted parameters, uncertainties, and simulated curves.

## Requirements

Install the required packages:

```bash
pip install numpy scipy matplotlib
```

## Input data

Provide one tab-delimited `.txt` file containing all the measured I-V curves, as in `Figure 3.txt`. Each curve occupies three adjacent columns, repeated from left to right:

```text
Temperature    Current    Voltage    Temperature    Current    Voltage    ...
[K]            [A]        [V]        [K]            [A]        [V]        ...
```

The first row names the columns and the second gives their units. For each curve, the script automatically finds the first row with three numeric values and skips any preceding metadata rows (such as the `T = ... K` labels in `Figure 3.txt`). After a curve starts, nonnumeric or partially filled rows raise an error; blank triplets are allowed when curves have different numbers of points. The temperature assigned to each curve is the mean of its numeric temperature column. Curves are numbered `curve_001`, `curve_002`, etc. from left to right; for fitting, the script keeps the sweep between the minimum and maximum voltage of each curve.

## Fitting procedure

The parameters are treated as follows:

- `Rsh1`, `Rsh2`: estimated from the low-bias linear region and kept fixed.
- `phi1_eV`, `phi2_eV`, `E00_eV`, `xi_eV`: global parameters shared by all temperatures.
- `eta1`, `eta2`, `Rnw`: temperature-dependent parameters fitted for each I-V curve.

The optimization is performed in two stages:

1. `Rnw` is fixed to the value estimated from the high-bias linear region.
2. The result of Stage 1 is used as the initial condition and `Rnw` is released.

`USE_SPARSE_JACOBIAN = True` speeds up both stages by using the fact that each curve depends on the global parameters and only its own `eta1`, `eta2`, and `Rnw`. It leaves the residuals and bounds unchanged, but SciPy uses an iterative sparse trust-region solver, so the optimization path can differ. Set it to `False` to use the previous dense solver.

The TFE contact solver searches from zero voltage: below the TFE onset the TFE current is zero, while the parallel shunt can still carry current.

## How to run

To fit measured curves, keep both scripts in the same folder, open `fit_msm_global.py`, and edit the **USER CONFIGURATION** section if necessary. The files must be named `msm_model.py` and `fit_msm_global.py` (remove any `(1)` suffix added when downloading). To simulate a single curve with specified parameters, use `from msm_model import solve_curve` in your own Python script.

After replacing `msm_model.py` while Spyder is open, restart its IPython console before rerunning the fit so Python loads the updated model.

To select the combined data file through a graphical window, leave:

```python
DATA_FILE = None
```

and run the script, for example from Spyder.

Alternatively, specify the file directly:

```python
DATA_FILE = r"C:\path\to\Figure 3.txt"
```

The main settings that may need adjustment are:

- nanowire diameter and contact geometry;
- high-bias and low-bias fitting windows;
- parameter initial values and bounds;
- Jacobian mode (`USE_SPARSE_JACOBIAN`);
- residual weighting (`VOLTAGE_WEIGHT_ALPHA`).

Set `VOLTAGE_WEIGHT_ALPHA = 0.0` for equal weighting of all voltage points.

## Console progress

With `SHOW_PROGRESS = True` (the default), the console shows the current fitting stage, model evaluation number, elapsed time, and a bar for the curves processed in that evaluation. The bar restarts for each evaluation; it is not a percentage of the entire fit, because the optimizer decides how many evaluations it needs. Set `PROGRESS_REFRESH_S` to change the update interval. Set `SHOW_PROGRESS = False` to use SciPy's `VERBOSE` output instead.

## Output

`fit_msm_global.py` creates a `fit_global_output` folder alongside the input file containing:

- fitted global parameters and uncertainties;
- temperature-dependent `eta1`, `eta2`, and `Rnw`;
- covariance matrix;
- fitting report;
- simulated I-V curves and fit plots;
- temperature-dependent parameter plots.

The reported uncertainties correspond to local one-standard-deviation (`1 sigma`) estimates obtained from the covariance matrix of the nonlinear least-squares fit.

## Citation

If you use this code, please cite the associated publication and the Zenodo code archive DOI.
