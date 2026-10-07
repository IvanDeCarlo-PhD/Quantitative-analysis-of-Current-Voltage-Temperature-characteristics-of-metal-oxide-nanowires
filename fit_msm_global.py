"""Global two-stage fit of temperature-dependent MSM I-V characteristics.

The parameterization follows the fitting strategy described in the manuscript:

Fixed from a preliminary fit for each I-V curve
    Rsh1, Rsh2  : estimated from the low-bias linear region. By default the
                   same effective low-bias estimate is assigned to both contacts.

Global parameters shared by the complete I-V-T dataset
    phi1_eV, phi2_eV, E00_eV, xi_eV

Temperature-dependent (curve-by-curve) parameters
    eta1, eta2, Rnw

Two fitting stages
    Stage 1: Rnw is fixed to the 8-10 V high-bias linear estimate.
             Global parameters and eta1/eta2 are optimized.
    Stage 2: initialized from Stage 1; Rnw is released and optimized together
             with the same global parameters and eta1/eta2.

Input text files must contain at least three whitespace-separated columns:
    temperature_K, current_A, voltage_V
with one header row by default.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from msm_model import M0, solve_curve, signed_voltage_drops


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Set DATA_FOLDER to a folder path, or leave it as None to select a folder with
# a graphical dialog when the script starts (useful when running from Spyder).
DATA_FOLDER = None
FILE_PATTERN = "*.txt"
SKIP_HEADER_ROWS = 1

# Device geometry and material constants.
# Set S1 and S2 independently if the two contact overlap areas are known.
NW_DIAMETER_M = 60e-9
CONTACT_OVERLAP_LENGTH_1_M = 550e-9
CONTACT_OVERLAP_LENGTH_2_M = 550e-9
S1_M2 = NW_DIAMETER_M * CONTACT_OVERLAP_LENGTH_1_M
S2_M2 = NW_DIAMETER_M * CONTACT_OVERLAP_LENGTH_2_M
M_STAR_KG = 0.78 * M0

# Preliminary linear fits.
HIGH_BIAS_WINDOW_V = (8.0, 10.0)       # uses |V| in this range
LOW_BIAS_WINDOW_V = (-2.0, 2.0)        # effective low-bias slope
MIN_POINTS_LINEAR_FIT = 5

# Global parameter initial guesses and bounds.
GLOBAL_INITIAL = {
    "phi1_eV": 0.43,
    "phi2_eV": 0.49,
    "E00_eV": 6.7e-3,
    "xi_eV": 0.20,
}
GLOBAL_LOWER = {
    "phi1_eV": 0.30,
    "phi2_eV": 0.30,
    "E00_eV": 1.0e-3,
    "xi_eV": 0.10,
}
GLOBAL_UPPER = {
    "phi1_eV": 0.60,
    "phi2_eV": 0.60,
    "E00_eV": 1.0e-2,
    "xi_eV": 1.00,
}

# Curve-by-curve initial guesses for eta. Initial values are interpolated
# between the points below; this affects only optimizer initialization, not the
# physical model or the final temperature dependence.
ETA1_INITIAL_T = ([140.0, 300.0], [8.0, 3.0])
ETA2_INITIAL_T = ([140.0, 300.0], [15.0, 7.0])
ETA_BOUNDS = (1.0, 20.0)
RNW_BOUNDS_OHM = (1.0e7, 1.0e11)

# Global least-squares settings.
MAX_NFEV = 20000
XTOL = 1e-12
FTOL = 1e-12
GTOL = 1e-12
VERBOSE = 1

# Each I-V curve is normalized by its maximum |I| so that high-current
# temperatures do not dominate the simultaneous fit. An optional voltage
# weight can emphasize the high-bias region; 0.0 means no extra weighting.
VOLTAGE_WEIGHT_ALPHA = 0.0

OUTPUT_FOLDER_NAME = "fit_global_output"

GLOBAL_NAMES = ("phi1_eV", "phi2_eV", "E00_eV", "xi_eV")
STAGE1_CURVE_NAMES = ("eta1", "eta2")
STAGE2_CURVE_NAMES = ("eta1", "eta2", "Rnw")


# =============================================================================
# I/O and preliminary linear estimates
# =============================================================================
def select_folder() -> str:
    """Return DATA_FOLDER or open a graphical folder selector."""
    if DATA_FOLDER:
        return str(DATA_FOLDER)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select folder containing I-V-T .txt files")
        root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "DATA_FOLDER is None and the graphical folder selector could not be opened. "
            "Set DATA_FOLDER manually near the top of fit_msm_global.py."
        ) from exc
    if not folder:
        raise RuntimeError("No data folder selected")
    return folder


def load_iv_file(path: str | Path) -> dict:
    data = np.loadtxt(path, skiprows=SKIP_HEADER_ROWS)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns (T, I, V)")
    data = data[np.all(np.isfinite(data[:, :3]), axis=1)]
    if data.shape[0] < 3:
        raise ValueError(f"{path}: not enough finite data points")

    T = data[:, 0].astype(float)
    I = data[:, 1].astype(float)
    V = data[:, 2].astype(float)

    # Keep the sweep between the voltage extrema and sort it in increasing V.
    i_max = int(np.argmax(V))
    i_min = int(np.argmin(V))
    lo, hi = sorted((i_min, i_max))
    V = V[lo:hi + 1]
    I = I[lo:hi + 1]
    order = np.argsort(V)

    return {
        "path": str(path),
        "name": Path(path).stem,
        "T_mean": float(np.mean(T)),
        "T_std": float(np.std(T)),
        "V": V[order],
        "I": I[order],
    }


def linear_resistance(V, I, vmin, vmax, use_abs=False, min_points=5):
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    if use_abs:
        mask = (np.abs(V) >= vmin) & (np.abs(V) <= vmax)
    else:
        mask = (V >= vmin) & (V <= vmax)
    if np.count_nonzero(mask) < min_points:
        raise ValueError(
            f"Only {np.count_nonzero(mask)} points in linear-fit window; need >= {min_points}."
        )
    slope, intercept = np.polyfit(V[mask], I[mask], 1)
    if not np.isfinite(slope) or abs(slope) < 1e-30:
        raise ValueError("Invalid slope in preliminary linear fit")
    return {
        "R_ohm": abs(1.0 / slope),
        "slope_A_per_V": float(slope),
        "intercept_A": float(intercept),
        "n_points": int(np.count_nonzero(mask)),
    }


def preliminary_estimates(curve: dict) -> dict:
    high = linear_resistance(
        curve["V"], curve["I"], HIGH_BIAS_WINDOW_V[0], HIGH_BIAS_WINDOW_V[1],
        use_abs=True, min_points=MIN_POINTS_LINEAR_FIT,
    )
    low = linear_resistance(
        curve["V"], curve["I"], LOW_BIAS_WINDOW_V[0], LOW_BIAS_WINDOW_V[1],
        use_abs=False, min_points=MIN_POINTS_LINEAR_FIT,
    )
    return {"Rnw_linear": high["R_ohm"], "Rsh_effective": low["R_ohm"], "high": high, "low": low}


def interp_initial(T, spec):
    temps, values = spec
    return float(np.interp(T, np.asarray(temps, float), np.asarray(values, float)))


# =============================================================================
# Parameter vector helpers
# =============================================================================
def pack_stage1(global_params, curve_params, order):
    x = [global_params[p] for p in GLOBAL_NAMES]
    for name in order:
        x.extend(curve_params[name][p] for p in STAGE1_CURVE_NAMES)
    return np.asarray(x, dtype=float)


def unpack_stage1(theta, order):
    theta = np.asarray(theta, dtype=float)
    out = {"global": dict(zip(GLOBAL_NAMES, theta[:len(GLOBAL_NAMES)])), "curves": {}}
    idx = len(GLOBAL_NAMES)
    for name in order:
        out["curves"][name] = {
            "eta1": float(theta[idx]),
            "eta2": float(theta[idx + 1]),
        }
        idx += 2
    return out


def bounds_stage1(order):
    lb = [GLOBAL_LOWER[p] for p in GLOBAL_NAMES]
    ub = [GLOBAL_UPPER[p] for p in GLOBAL_NAMES]
    for _ in order:
        lb.extend([ETA_BOUNDS[0], ETA_BOUNDS[0]])
        ub.extend([ETA_BOUNDS[1], ETA_BOUNDS[1]])
    return np.asarray(lb), np.asarray(ub)


def pack_stage2(global_params, curve_params, order):
    x = [global_params[p] for p in GLOBAL_NAMES]
    for name in order:
        x.extend(curve_params[name][p] for p in STAGE2_CURVE_NAMES)
    return np.asarray(x, dtype=float)


def unpack_stage2(theta, order):
    theta = np.asarray(theta, dtype=float)
    out = {"global": dict(zip(GLOBAL_NAMES, theta[:len(GLOBAL_NAMES)])), "curves": {}}
    idx = len(GLOBAL_NAMES)
    for name in order:
        out["curves"][name] = {
            "eta1": float(theta[idx]),
            "eta2": float(theta[idx + 1]),
            "Rnw": float(theta[idx + 2]),
        }
        idx += 3
    return out


def bounds_stage2(order):
    lb = [GLOBAL_LOWER[p] for p in GLOBAL_NAMES]
    ub = [GLOBAL_UPPER[p] for p in GLOBAL_NAMES]
    for _ in order:
        lb.extend([ETA_BOUNDS[0], ETA_BOUNDS[0], RNW_BOUNDS_OHM[0]])
        ub.extend([ETA_BOUNDS[1], ETA_BOUNDS[1], RNW_BOUNDS_OHM[1]])
    return np.asarray(lb), np.asarray(ub)


def model_params(curve, decoded, fixed, stage):
    g = decoded["global"]
    c = decoded["curves"][curve["name"]]
    return {
        "S1": S1_M2,
        "S2": S2_M2,
        "m_star": M_STAR_KG,
        "T": curve["T_mean"],
        "Rsh1": fixed[curve["name"]]["Rsh1"],
        "Rsh2": fixed[curve["name"]]["Rsh2"],
        "phi1_eV": g["phi1_eV"],
        "phi2_eV": g["phi2_eV"],
        "E00_eV": g["E00_eV"],
        "xi_eV": g["xi_eV"],
        "eta1": c["eta1"],
        "eta2": c["eta2"],
        "Rnw": fixed[curve["name"]]["Rnw_stage1"] if stage == 1 else c["Rnw"],
    }


# =============================================================================
# Residuals, fitting, statistics
# =============================================================================
def residual_vector(theta, curves, order, fixed, stage):
    decoded = unpack_stage1(theta, order) if stage == 1 else unpack_stage2(theta, order)
    residuals = []
    for name in order:
        curve = curves[name]
        params = model_params(curve, decoded, fixed, stage)
        _, _, I_fit = solve_curve(curve["V"], params)

        # Normalize each temperature trace so that all curves contribute on a
        # comparable scale to the simultaneous global optimization.
        current_scale = max(np.max(np.abs(curve["I"])), 1e-30)
        vmax = max(np.max(np.abs(curve["V"])), 1e-30)
        weight = 1.0 + VOLTAGE_WEIGHT_ALPHA * np.abs(curve["V"]) / vmax
        residuals.append(weight * (curve["I"] - I_fit) / current_scale)
    return np.concatenate(residuals)


def covariance_from_result(result):
    """Local covariance approximation C = s^2 (J^T J)^-1."""
    r = np.asarray(result.fun, dtype=float)
    J = np.asarray(result.jac, dtype=float)
    dof = max(J.shape[0] - J.shape[1], 1)
    rss = float(np.dot(r, r))
    s2 = rss / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return cov, se, rss, dof


def r_squared(y, yfit):
    y = np.asarray(y, float)
    yfit = np.asarray(yfit, float)
    sse = float(np.sum((y - yfit)**2))
    sst = float(np.sum((y - np.mean(y))**2))
    return np.nan if sst == 0 else 1.0 - sse / sst


def load_dataset(folder):
    output_dir = Path(folder) / OUTPUT_FOLDER_NAME
    files = sorted(Path(folder).glob(FILE_PATTERN))
    curves = {}
    for path in files:
        if output_dir in path.parents:
            continue
        try:
            curve = load_iv_file(path)
            curves[curve["name"]] = curve
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if not curves:
        raise FileNotFoundError(f"No valid {FILE_PATTERN} files found in {folder}")
    order = sorted(curves, key=lambda n: curves[n]["T_mean"])
    return curves, order


def fit_dataset(folder):
    curves, order = load_dataset(folder)
    output_dir = Path(folder) / OUTPUT_FOLDER_NAME
    output_dir.mkdir(exist_ok=True)

    fixed = {}
    curve_init = {}
    for name in order:
        est = preliminary_estimates(curves[name])
        # The manuscript treats Rsh as a low-bias pre-fit quantity. With only
        # the two-terminal curve available, the same effective estimate is used
        # for the two junctions unless the code is modified to provide separate
        # independently measured values.
        fixed[name] = {
            "Rsh1": est["Rsh_effective"],
            "Rsh2": est["Rsh_effective"],
            "Rnw_stage1": est["Rnw_linear"],
            "linear": est,
        }
        curve_init[name] = {
            "eta1": interp_initial(curves[name]["T_mean"], ETA1_INITIAL_T),
            "eta2": interp_initial(curves[name]["T_mean"], ETA2_INITIAL_T),
            "Rnw": est["Rnw_linear"],
        }

    # ---- Stage 1: global phi/E00/xi + eta(T), with Rnw fixed ----
    x01 = pack_stage1(GLOBAL_INITIAL, curve_init, order)
    b1 = bounds_stage1(order)
    result1 = least_squares(
        residual_vector, x01, bounds=b1, method="trf",
        args=(curves, order, fixed, 1),
        loss="linear", max_nfev=MAX_NFEV, xtol=XTOL, ftol=FTOL, gtol=GTOL,
        verbose=VERBOSE,
    )
    d1 = unpack_stage1(result1.x, order)

    # ---- Stage 2: same global model + eta(T) + released Rnw(T) ----
    curve_init2 = {}
    for name in order:
        curve_init2[name] = {
            "eta1": d1["curves"][name]["eta1"],
            "eta2": d1["curves"][name]["eta2"],
            "Rnw": fixed[name]["Rnw_stage1"],
        }
    x02 = pack_stage2(d1["global"], curve_init2, order)
    b2 = bounds_stage2(order)
    result2 = least_squares(
        residual_vector, x02, bounds=b2, method="trf",
        args=(curves, order, fixed, 2),
        loss="linear", max_nfev=MAX_NFEV, xtol=XTOL, ftol=FTOL, gtol=GTOL,
        verbose=VERBOSE,
    )

    decoded = unpack_stage2(result2.x, order)
    cov, se, rss_norm, dof = covariance_from_result(result2)

    names = list(GLOBAL_NAMES)
    for name in order:
        names.extend([f"{name}:{p}" for p in STAGE2_CURVE_NAMES])
    se_map = dict(zip(names, se))

    results = {
        "folder": str(folder),
        "output_dir": str(output_dir),
        "curves": curves,
        "order": order,
        "fixed": fixed,
        "decoded": decoded,
        "covariance": cov,
        "se_map": se_map,
        "rss_normalized": rss_norm,
        "dof": dof,
        "stage1": result1,
        "stage2": result2,
        "curve_results": {},
    }

    for name in order:
        p = model_params(curves[name], decoded, fixed, 2)
        V1, V2, I_fit = solve_curve(curves[name]["V"], p)
        results["curve_results"][name] = {
            "params": p,
            "I_fit": I_fit,
            "V1_mag": V1,
            "V2_mag": V2,
            "R2": r_squared(curves[name]["I"], I_fit),
            "eta1_se": se_map[f"{name}:eta1"],
            "eta2_se": se_map[f"{name}:eta2"],
            "Rnw_se": se_map[f"{name}:Rnw"],
        }

    save_results(results)
    return results


# =============================================================================
# Outputs
# =============================================================================
def save_results(results):
    out = Path(results["output_dir"])
    order = results["order"]
    d = results["decoded"]
    se = results["se_map"]

    # Global parameter table: global quantities are intentionally reported
    # once rather than repeated as artificial temperature series.
    with open(out / "global_parameters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "standard_error_1sigma"])
        for p in GLOBAL_NAMES:
            writer.writerow([p, d["global"][p], se[p]])

    with open(out / "temperature_dependent_parameters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "curve", "T_mean_K", "T_std_K", "eta1", "eta1_se", "eta2", "eta2_se",
            "Rnw_ohm", "Rnw_se_ohm", "Rnw_high_bias_ohm", "Rsh_fixed_ohm", "R2"
        ])
        for name in order:
            c = results["curves"][name]
            cr = results["curve_results"][name]
            writer.writerow([
                name, c["T_mean"], c["T_std"], d["curves"][name]["eta1"], cr["eta1_se"],
                d["curves"][name]["eta2"], cr["eta2_se"], d["curves"][name]["Rnw"],
                cr["Rnw_se"], results["fixed"][name]["Rnw_stage1"],
                results["fixed"][name]["Rsh1"], cr["R2"],
            ])

    np.savetxt(out / "covariance_matrix.csv", results["covariance"], delimiter=",")

    with open(out / "fit_report.txt", "w", encoding="utf-8") as f:
        f.write("MSM GLOBAL TWO-STAGE FIT\n")
        f.write("========================\n\n")
        f.write(f"Curves: {len(order)}\n")
        f.write(f"Stage 1 normalized RSS: {np.sum(results['stage1'].fun**2):.8g}\n")
        f.write(f"Stage 2 normalized RSS: {np.sum(results['stage2'].fun**2):.8g}\n")
        f.write(f"Final dof: {results['dof']}\n\n")
        f.write("Global parameters (value +/- 1 sigma):\n")
        for p in GLOBAL_NAMES:
            f.write(f"  {p:10s} = {d['global'][p]:.8g} +/- {se[p]:.3g}\n")
        f.write("\nNOTE: covariance-derived uncertainties are local linearized standard errors.\n")
        f.write("A large uncertainty (e.g. for xi_eV) indicates weak parameter identifiability.\n")
        f.write("\nCurve-by-curve parameters:\n")
        for name in order:
            c = results["curves"][name]
            cr = results["curve_results"][name]
            cp = d["curves"][name]
            f.write(
                f"  {name}: T={c['T_mean']:.3f} K, "
                f"eta1={cp['eta1']:.6g}+/-{cr['eta1_se']:.2g}, "
                f"eta2={cp['eta2']:.6g}+/-{cr['eta2_se']:.2g}, "
                f"Rnw={cp['Rnw']:.6g}+/-{cr['Rnw_se']:.2g} ohm, R2={cr['R2']:.6f}\n"
            )

    # Per-curve simulated data and I-V plots.
    for name in order:
        c = results["curves"][name]
        cr = results["curve_results"][name]
        cp = d["curves"][name]
        p = cr["params"]

        V_dense = np.linspace(np.min(c["V"]), np.max(c["V"]), 500)
        V1m, V2m, I_dense = solve_curve(V_dense, p)
        V1, Vnw, V2 = signed_voltage_drops(V_dense, V1m, V2m, I_dense, cp["Rnw"])
        np.savetxt(
            out / f"{name}_simulated.csv",
            np.column_stack([V_dense, I_dense, V1, Vnw, V2]), delimiter=",",
            header="V_bias_V,I_fit_A,V1_V,Vnw_V,V2_V", comments="",
        )

        plt.figure(figsize=(7, 5))
        plt.plot(c["V"], c["I"], "o", ms=3, label="experiment")
        plt.plot(V_dense, I_dense, "-", lw=1.5, label="MSM fit")
        plt.xlabel("Bias voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"{c['T_mean']:.1f} K | R2={cr['R2']:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{name}_fit.png", dpi=250)
        plt.close()

    # Only genuine temperature-dependent parameters are plotted versus T.
    T = np.array([results["curves"][n]["T_mean"] for n in order])
    for p_name, se_name, ylabel in [
        ("eta1", "eta1_se", r"$\eta_1$"),
        ("eta2", "eta2_se", r"$\eta_2$"),
        ("Rnw", "Rnw_se", r"$R_{NW}$ ($\Omega$)"),
    ]:
        y = np.array([d["curves"][n][p_name] for n in order])
        yerr = np.array([results["curve_results"][n][se_name] for n in order])
        plt.figure(figsize=(7, 5))
        plt.errorbar(T, y, yerr=yerr, fmt="o", capsize=3)
        plt.xlabel("Temperature (K)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out / f"{p_name}_vs_temperature.png", dpi=250)
        plt.close()

    # Rnw comparison with the preliminary high-bias estimate.
    R_fit = np.array([d["curves"][n]["Rnw"] for n in order])
    R_fit_se = np.array([results["curve_results"][n]["Rnw_se"] for n in order])
    R_lin = np.array([results["fixed"][n]["Rnw_stage1"] for n in order])
    plt.figure(figsize=(7, 5))
    plt.errorbar(T, R_fit, yerr=R_fit_se, fmt="o", capsize=3, label="MSM model")
    plt.plot(T, R_lin, "o", label="high-bias estimate")
    plt.xlabel("Temperature (K)")
    plt.ylabel(r"$R_{NW}$ ($\Omega$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "Rnw_MSM_vs_high_bias.png", dpi=250)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    folder = select_folder()
    results = fit_dataset(folder)
    print("\nFit complete.")
    print(f"Output folder: {results['output_dir']}")
    print("\nGlobal parameters (value +/- 1 sigma):")
    for p in GLOBAL_NAMES:
        print(f"  {p:10s}: {results['decoded']['global'][p]:.6g} +/- {results['se_map'][p]:.3g}")


if __name__ == "__main__":
    main()
