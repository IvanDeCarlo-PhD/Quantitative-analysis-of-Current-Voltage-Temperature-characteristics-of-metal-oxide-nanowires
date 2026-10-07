"""Metal-semiconductor-metal (MSM) current-voltage model.

This module implements the transport model used for two back-to-back
metal/semiconductor Schottky junctions connected by a nanowire resistance.
Forward-biased contacts are described with thermionic emission (TE), while
reverse-biased contacts are described with thermionic-field emission (TFE).
A fixed shunt resistance is included in parallel with each contact.

The public entry point is ``solve_curve(V_bias, params)``.

Units
-----
SI units are used unless explicitly stated otherwise:
    voltage              V
    current              A
    resistance           ohm
    contact area         m^2
    temperature          K
    effective mass       kg
    phi1_eV, phi2_eV     eV
    E00_eV, xi_eV        eV

Notes
-----
``xi_eV`` is an energy offset expressed in electronvolts. Numerically, an
energy of xi_eV eV corresponds to q*xi_eV joules, hence terms of the form
q*(V - xi_eV) are dimensionally consistent when V is in volts.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# Exact SI constants
Q = 1.602176634e-19
H = 6.62607015e-34
KB = 1.380649e-23
PI = np.pi
M0 = 9.1093837015e-31

# exp(700) is close to the floating-point limit; a smaller value leaves
# headroom for multiplication by prefactors.
_EXP_CLIP = 650.0


def _safe_exp(x: float | np.ndarray) -> float | np.ndarray:
    """Exponentiation protected against floating-point overflow."""
    return np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP))


def richardson_constant(m_star: float) -> float:
    """Effective Richardson constant A* in A m^-2 K^-2."""
    return 4.0 * PI * m_star * Q * KB**2 / H**3


def current_density_te(V: float, phi_eV: float, eta: float, T: float, m_star: float) -> float:
    """Forward-bias thermionic-emission current density."""
    if eta <= 0 or T <= 0:
        raise ValueError("eta and T must be positive")

    phi_J = phi_eV * Q
    A_star = richardson_constant(m_star)
    return float(
        A_star
        * T**2
        * _safe_exp(-phi_J / (KB * T))
        * _safe_exp(Q * V / (eta * KB * T))
        * (1.0 - _safe_exp(-Q * V / (KB * T)))
    )


def _tfe_energy_terms(phi_eV: float, E00_eV: float, T: float) -> tuple[float, float]:
    """Return E0 [J] and cosh(E00/kBT) used by the TFE expression."""
    if E00_eV <= 0 or T <= 0:
        raise ValueError("E00_eV and T must be positive")

    E00_J = E00_eV * Q
    x = E00_J / (KB * T)
    tanh_x = np.tanh(x)
    if abs(tanh_x) < 1e-30:
        tanh_x = np.copysign(1e-30, tanh_x if tanh_x != 0 else 1.0)
    E0_J = E00_J / tanh_x
    cosh_x = np.cosh(np.clip(x, -300.0, 300.0))
    return float(E0_J), float(cosh_x)


def current_density_tfe(
    V: float,
    phi_eV: float,
    xi_eV: float,
    E00_eV: float,
    T: float,
    m_star: float,
) -> float:
    """Reverse-bias thermionic-field-emission current density.

    The Padovani-Stratton prefactor is evaluated only when its square-root
    argument is positive. The zero-bias value is subtracted so that this
    branch contributes zero net current at V=0 in the circuit model.
    """
    phi_J = phi_eV * Q
    E00_J = E00_eV * Q
    E0_J, cosh_x = _tfe_energy_terms(phi_eV, E00_eV, T)
    A_star = richardson_constant(m_star)

    def raw(v: float) -> float:
        arg_J = Q * (v - xi_eV) + phi_J / (cosh_x**2)
        if arg_J <= 0.0:
            return 0.0
        value = (
            A_star
            * (T / KB)
            * np.sqrt(PI * E00_J)
            * np.sqrt(arg_J)
            * _safe_exp(-phi_J / E0_J)
            * _safe_exp(Q * v * (1.0 / (KB * T) - 1.0 / E0_J))
        )
        return float(value) if np.isfinite(value) else np.inf

    out = raw(V) - raw(0.0)
    return float(out) if np.isfinite(out) else np.inf


def _bracket_root(
    func,
    x_floor: float,
    x_previous: float | None = None,
    local_width: float = 1e-6,
    local_tries: int = 20,
    global_start: float = 1.0,
    global_growth: float = 2.0,
    global_tries: int = 50,
) -> tuple[float, float]:
    """Find a finite sign-changing interval, using continuation first."""
    floor = float(x_floor)

    if x_previous is not None and np.isfinite(x_previous):
        center = max(floor, float(x_previous))
        width = max(float(local_width), 1e-15)
        for _ in range(local_tries):
            a = max(floor, center - width)
            b = center + width
            try:
                fa, fb = func(a), func(b)
            except Exception:
                fa, fb = np.nan, np.nan
            if np.isfinite(fa) and np.isfinite(fb):
                if fa == 0.0:
                    return a, a
                if fb == 0.0:
                    return b, b
                if fa * fb < 0.0:
                    return a, b
            width *= 2.0

    a = floor
    b = max(floor + 1e-12, float(global_start))
    try:
        fa = func(a)
    except Exception:
        fa = np.nan

    for _ in range(global_tries):
        try:
            fb = func(b)
        except Exception:
            fb = np.nan
        if np.isfinite(fa) and np.isfinite(fb):
            if fa == 0.0:
                return a, a
            if fb == 0.0:
                return b, b
            if fa * fb < 0.0:
                return a, b
        b *= global_growth

    raise RuntimeError(f"Unable to bracket a root on [{floor}, {b:g}]")


def _solve_contact_voltage(
    current_mag: float,
    transport: str,
    area: float,
    Rsh: float,
    phi_eV: float,
    eta: float,
    xi_eV: float,
    E00_eV: float,
    T: float,
    m_star: float,
    V_previous: float | None = None,
    xtol: float = 1e-12,
    maxiter: int = 1000,
) -> float:
    """Solve S*J(V) + V/Rsh = |I| for a single contact."""
    if area <= 0 or Rsh <= 0:
        raise ValueError("Contact area and Rsh must be positive")

    if transport == "TE":
        def f(v: float) -> float:
            return area * current_density_te(v, phi_eV, eta, T, m_star) + v / Rsh - current_mag
        v_floor = 0.0

    elif transport == "TFE":
        _, cosh_x = _tfe_energy_terms(phi_eV, E00_eV, T)
        # Positive square-root argument requires
        # V > xi - phi/cosh^2(E00/kBT), with phi and xi both in eV/V numerically.
        v_phys_min = xi_eV - phi_eV / (cosh_x**2)
        v_floor = max(0.0, float(v_phys_min))

        def f(v: float) -> float:
            return area * current_density_tfe(v, phi_eV, xi_eV, E00_eV, T, m_star) + v / Rsh - current_mag
    else:
        raise ValueError("transport must be 'TE' or 'TFE'")

    a, b = _bracket_root(
        f,
        x_floor=v_floor,
        x_previous=V_previous,
        local_width=1e-6,
        global_start=max(v_floor + 1e-6, 1.0),
    )
    if a == b:
        return float(a)
    return float(brentq(f, a, b, xtol=xtol, rtol=1e-12, maxiter=maxiter))


def solve_point(
    V_bias: float,
    params: dict,
    I_previous_mag: float = 0.0,
    V1_previous: float | None = None,
    V2_previous: float | None = None,
    xtol: float = 1e-12,
    maxiter: int = 1000,
) -> tuple[float, float, float]:
    """Solve one MSM operating point.

    Returns positive magnitudes of V1 and V2 and a signed total current.
    The sign of each contact voltage for plotting is obtained by multiplying
    the returned V1/V2 by sign(V_bias).
    """
    if abs(V_bias) < 1e-15:
        return 0.0, 0.0, 0.0

    required = [
        "Rnw", "Rsh1", "Rsh2", "S1", "S2", "phi1_eV", "phi2_eV",
        "eta1", "eta2", "xi_eV", "E00_eV", "m_star", "T",
    ]
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing model parameter(s): {', '.join(missing)}")

    V_abs = abs(float(V_bias))
    Rnw = float(params["Rnw"])
    if Rnw <= 0:
        raise ValueError("Rnw must be positive")

    if V_bias > 0:
        left_transport, right_transport = "TFE", "TE"
        reverse_is_left = True
    else:
        left_transport, right_transport = "TE", "TFE"
        reverse_is_left = False

    def v1_of(i_mag: float) -> float:
        return _solve_contact_voltage(
            i_mag, left_transport, params["S1"], params["Rsh1"],
            params["phi1_eV"], params["eta1"], params["xi_eV"],
            params["E00_eV"], params["T"], params["m_star"],
            V_previous=V1_previous, xtol=xtol, maxiter=maxiter,
        )

    def v2_of(i_mag: float) -> float:
        return _solve_contact_voltage(
            i_mag, right_transport, params["S2"], params["Rsh2"],
            params["phi2_eV"], params["eta2"], params["xi_eV"],
            params["E00_eV"], params["T"], params["m_star"],
            V_previous=V2_previous, xtol=xtol, maxiter=maxiter,
        )

    def circuit_residual(i_mag: float) -> float:
        return v1_of(i_mag) + v2_of(i_mag) + i_mag * Rnw - V_abs

    # If the reverse branch is outside its physically valid TFE domain at
    # zero current, treat the operating point as blocked and place the full
    # voltage on that contact.
    g0 = circuit_residual(0.0)
    if np.isfinite(g0) and g0 > 0.0:
        if reverse_is_left:
            return V_abs, 0.0, 0.0
        return 0.0, V_abs, 0.0

    a, b = _bracket_root(
        circuit_residual,
        x_floor=0.0,
        x_previous=max(float(I_previous_mag), 0.0),
        local_width=max(1e-12, 0.2 * max(float(I_previous_mag), 1e-12)),
        global_start=max(10.0 * float(I_previous_mag), 1e-12),
    )
    if a == b:
        I_mag = float(a)
    else:
        I_mag = float(brentq(circuit_residual, a, b, xtol=1e-18, rtol=1e-12, maxiter=maxiter))

    V1 = v1_of(I_mag)
    V2 = v2_of(I_mag)
    return float(V1), float(V2), float(np.sign(V_bias) * I_mag)


def solve_curve(
    V_bias: np.ndarray,
    params: dict,
    xtol: float = 1e-12,
    maxiter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve an entire I-V curve using branch-wise continuation.

    Parameters
    ----------
    V_bias : array-like
        Applied voltages in volts. Input order is preserved.
    params : dict
        Model parameters described in :func:`solve_point`.

    Returns
    -------
    V1, V2, I : ndarray
        Contact-voltage magnitudes and signed current, in input order.
    """
    V_bias = np.asarray(V_bias, dtype=float)
    V1 = np.zeros_like(V_bias)
    V2 = np.zeros_like(V_bias)
    current = np.zeros_like(V_bias)

    neg = np.where(V_bias < 0.0)[0]
    zero = np.where(np.isclose(V_bias, 0.0))[0]
    pos = np.where(V_bias > 0.0)[0]

    for indices in (neg[np.argsort(np.abs(V_bias[neg]))], pos[np.argsort(np.abs(V_bias[pos]))]):
        I_prev = 0.0
        V1_prev = None
        V2_prev = None
        for i in indices:
            v1, v2, cur = solve_point(
                V_bias[i], params, I_prev, V1_prev, V2_prev,
                xtol=xtol, maxiter=maxiter,
            )
            V1[i], V2[i], current[i] = v1, v2, cur
            I_prev = abs(cur)
            V1_prev, V2_prev = v1, v2

    V1[zero] = 0.0
    V2[zero] = 0.0
    current[zero] = 0.0
    return V1, V2, current


def signed_voltage_drops(V_bias: np.ndarray, V1_mag: np.ndarray, V2_mag: np.ndarray, current: np.ndarray, Rnw: float):
    """Return signed contact and nanowire voltage drops for plotting/saving."""
    V_bias = np.asarray(V_bias, dtype=float)
    sign = np.sign(V_bias)
    V1 = np.asarray(V1_mag, dtype=float) * sign
    V2 = np.asarray(V2_mag, dtype=float) * sign
    Vnw = np.asarray(current, dtype=float) * float(Rnw)
    return V1, Vnw, V2
