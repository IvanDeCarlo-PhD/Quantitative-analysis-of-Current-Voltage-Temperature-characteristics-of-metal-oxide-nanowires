# MSM I-V-T fitting code

Python code used to fit temperature-dependent I-V characteristics of metal-semiconductor-metal (MSM) nanowire devices.

The repository contains two scripts:

- `msm_model.py` - MSM transport model and numerical solver.
- `fit_msm_global.py` - data loading, preliminary resistance estimates, two-stage fitting, uncertainty calculation, and output generation.

## Requirements

Install the required packages:

```bash
pip install numpy scipy matplotlib
```

## Input data

Place all experimental `.txt` files in the same folder.

Each file must contain three columns:

```text
Temperature_K    Current_A    Voltage_V
```

The first row is treated as a header. The temperature assigned to each I-V curve is the mean value of the first column.

## Fitting procedure

The parameters are treated as follows:

- `Rsh1`, `Rsh2`: estimated from the low-bias linear region and kept fixed.
- `phi1_eV`, `phi2_eV`, `E00_eV`, `xi_eV`: global parameters shared by all temperatures.
- `eta1`, `eta2`, `Rnw`: temperature-dependent parameters fitted for each I-V curve.

The optimization is performed in two stages:

1. `Rnw` is fixed to the value estimated from the high-bias linear region.
2. The result of Stage 1 is used as the initial condition and `Rnw` is released.

## How to run

Open `fit_msm_global.py` and edit the **USER CONFIGURATION** section if necessary.

To select the data folder through a graphical window, leave:

```python
DATA_FOLDER = None
```

and run the script, for example from Spyder.

Alternatively, specify the folder directly:

```python
DATA_FOLDER = r"C:\path\to\data"
```

The main settings that may need adjustment are:

- nanowire diameter and contact geometry;
- high-bias and low-bias fitting windows;
- parameter initial values and bounds;
- residual weighting (`VOLTAGE_WEIGHT_ALPHA`).

Set `VOLTAGE_WEIGHT_ALPHA = 0.0` for equal weighting of all voltage points.

## Output

The script creates a `fit_global_output` folder containing:

- fitted global parameters and uncertainties;
- temperature-dependent `eta1`, `eta2`, and `Rnw`;
- covariance matrix;
- fitting report;
- simulated I-V curves and fit plots;
- temperature-dependent parameter plots.

The reported uncertainties correspond to local one-standard-deviation (`1 sigma`) estimates obtained from the covariance matrix of the nonlinear least-squares fit.

## Citation

If you use this code, please cite the associated publication and the Zenodo code archive DOI.
