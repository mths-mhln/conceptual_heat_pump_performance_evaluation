"""
Spinodal curve plotting with CoolProp
Demonstrates two methods:
  1. build_spinodal() via the low-level AbstractState interface (mixture-capable)
  2. Manual rootfinding: locate where dP/drho|T = 0 along isotherms (pure fluids)
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

# ── Configuration ──────────────────────────────────────────────────────────────
FLUID = "Water"          # change to any HEOS fluid, e.g. "CO2", "Propane"
BACKEND = "HEOS"

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — build_spinodal() (low-level interface, pure fluid treated as
#             pseudo-mixture so the method resolves)
# ══════════════════════════════════════════════════════════════════════════════

def get_spinodal_build(fluid, backend="HEOS"):
    """
    Use AbstractState.build_spinodal() + get_spinodal_data().
    Returns arrays of (T [K], rho [kg/m³], P [Pa]).
    """
    AS = AbstractState(backend, fluid)
    try:
        AS.build_spinodal()
        data = AS.get_spinodal_data()

        # data contains reduced coordinates; convert back to SI
        rho_r = AS.rhomass_critical()   # kg/m³
        T_r   = AS.T_critical()         # K

        tau_arr   = np.array(data.tau)    # τ = T_r / T
        delta_arr = np.array(data.delta)  # δ = ρ / ρ_r

        T_arr   = T_r / tau_arr
        rho_arr = delta_arr * rho_r

        # Compute P at each (T, rho) point
        P_arr = []
        for T_val, rho_val in zip(T_arr, rho_arr):
            try:
                AS.update(CP.DmassT_INPUTS, rho_val, T_val)
                P_arr.append(AS.p())
            except Exception:
                P_arr.append(np.nan)

        return T_arr, rho_arr, np.array(P_arr)

    except Exception as e:
        print(f"build_spinodal() failed: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Manual isotherm rootfinding: dP/drho|T = 0
#             Works robustly for any pure HEOS fluid
# ══════════════════════════════════════════════════════════════════════════════

def dpdrho_T(AS, rho, T):
    """Return dP/dρ|T at a given (ρ, T) using CoolProp's derivative interface."""
    AS.update(CP.DmassT_INPUTS, rho, T)
    return AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)


def find_spinodal_manual(fluid, backend="HEOS", n_T=80):
    """
    For each isotherm between T_triple and T_critical, find the two densities
    where dP/dρ|T = 0 (liquid spinodal and vapour spinodal).
    Returns separate arrays for liquid and vapour branches.
    """
    AS = AbstractState(backend, fluid)

    T_c   = AS.T_critical()
    rho_c = AS.rhomass_critical()

    # Triple-point temperature (approximate lower bound)
    T_min = AS.Tmin() + 1.0
    T_max = T_c - 0.5   # stay just below critical point

    T_vals = np.linspace(T_min, T_max, n_T)

    liq_T, liq_rho, liq_P = [], [], []
    vap_T, vap_rho, vap_P = [], [], []

    from scipy.optimize import brentq

    for T in T_vals:
        try:
            # Get saturation densities as bracket bounds
            AS.update(CP.QT_INPUTS, 0.0, T)
            rho_liq_sat = AS.rhomass()
            AS.update(CP.QT_INPUTS, 1.0, T)
            rho_vap_sat = AS.rhomass()

            # ── Liquid spinodal: dP/dρ|T = 0 between rho_c and rho_liq_sat ──
            rho_lo, rho_hi = rho_c * 1.01, rho_liq_sat * 0.999
            if dpdrho_T(AS, rho_lo, T) * dpdrho_T(AS, rho_hi, T) < 0:
                rho_sp = brentq(lambda r: dpdrho_T(AS, r, T), rho_lo, rho_hi)
                AS.update(CP.DmassT_INPUTS, rho_sp, T)
                liq_T.append(T);   liq_rho.append(rho_sp);  liq_P.append(AS.p())

            # ── Vapour spinodal: dP/dρ|T = 0 between rho_vap_sat and rho_c ──
            rho_lo, rho_hi = rho_vap_sat * 1.001, rho_c * 0.99
            if dpdrho_T(AS, rho_lo, T) * dpdrho_T(AS, rho_hi, T) < 0:
                rho_sp = brentq(lambda r: dpdrho_T(AS, r, T), rho_lo, rho_hi)
                AS.update(CP.DmassT_INPUTS, rho_sp, T)
                vap_T.append(T);   vap_rho.append(rho_sp);  vap_P.append(AS.p())

        except Exception:
            continue

    return (np.array(liq_T),  np.array(liq_rho),  np.array(liq_P),
            np.array(vap_T),  np.array(vap_rho),  np.array(vap_P))


# ══════════════════════════════════════════════════════════════════════════════
# Saturation (binodal) curve for reference
# ══════════════════════════════════════════════════════════════════════════════

def get_saturation_curve(fluid, backend="HEOS", n=200):
    AS = AbstractState(backend, fluid)
    T_c = AS.T_critical()
    T_min = AS.Tmin() + 1.0
    T_vals = np.linspace(T_min, T_c - 0.1, n)

    liq_rho, vap_rho, liq_T, vap_T = [], [], [], []
    for T in T_vals:
        try:
            AS.update(CP.QT_INPUTS, 0.0, T);  liq_rho.append(AS.rhomass()); liq_T.append(T)
            AS.update(CP.QT_INPUTS, 1.0, T);  vap_rho.append(AS.rhomass()); vap_T.append(T)
        except Exception:
            continue
    return np.array(liq_T), np.array(liq_rho), np.array(vap_T), np.array(vap_rho)


# ══════════════════════════════════════════════════════════════════════════════
# Plot — T-ρ diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_spinodal(fluid):
    AS = AbstractState(BACKEND, fluid)
    T_c, rho_c = AS.T_critical(), AS.rhomass_critical()

    # Saturation curve
    lT, lRho, vT, vRho = get_saturation_curve(fluid)

    # Spinodal via manual method (most reliable for pure fluids)
    liq_T, liq_rho, _, vap_T, vap_rho, _ = find_spinodal_manual(fluid)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Binodal
    ax.plot(lRho, lT, 'b-',  lw=2, label='Binodal (sat. liquid)')
    ax.plot(vRho, vT, 'b--', lw=2, label='Binodal (sat. vapour)')

    # Spinodal
    ax.plot(liq_rho, liq_T, 'r-',  lw=2, label='Spinodal (liquid branch)')
    ax.plot(vap_rho, vap_T, 'r--', lw=2, label='Spinodal (vapour branch)')

    # Critical point
    ax.plot(rho_c, T_c, 'ko', ms=8, zorder=5, label=f'Critical point')

    ax.set_xlabel(r'Density  $\rho$  [kg m$^{-3}$]')
    ax.set_ylabel('Temperature  $T$  [K]')
    ax.set_title(f'Binodal & Spinodal curves — {fluid}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'spinodal_{fluid}.png', dpi=150)
    plt.show()
    print(f"Saved spinodal_{fluid}.png")


if __name__ == "__main__":
    plot_spinodal(FLUID)