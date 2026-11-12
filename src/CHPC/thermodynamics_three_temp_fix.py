import io
import contextlib
import numpy as np
from scipy.optimize import minimize_scalar

from CHPC.logger import setup_logger
from CHPC.thermodynamics import (_cp_props, _vapour_quality_scaler, _specific_heat_from_isobar_path,
                                 _sample_isobar_ts_uniform_arc, _cop_turb)

logger = setup_logger()



# Cycle calculation method of thermodynamics_legacy_4: the temperature differences ΔT_pp_1 (evaporator outlet), ΔT_sh 
# (superheat), ΔT_pp_3 (condenser outlet) and ΔT_pp_4 (evaporator inlet) are fixed, such that PR is the only variable 
# that is optimised. The refrigerant mass flow follows from the evaporator, hence Q_out_req is not used. 

def _quiet_minimize_scalar(func, bounds, tol=1e-9, maxiter=80):
    """Run scipy minimize_scalar without terminal output."""
    options = {
        "xatol": tol,
        "maxiter": maxiter,
        "disp": 0,
    }
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return minimize_scalar(
            func,
            bounds=bounds,
            method="bounded",
            tol=tol,
            options=options,
        )



# Evaluate cycle for specific pressure ratio
def evaluate_cycle_PR(cycle_config, general_config, PR, verbose=True):
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]
    ṁ_c = cycle_config["ṁ_c"]
    ṁ_h = cycle_config["ṁ_h"]
    cp_c = cycle_config["cp_c"]
    cp_h = cycle_config["cp_h"]
    η_compr = cycle_config["η_compr"]
    η_turb = cycle_config["η_turb"]
    ΔT_pp_1 = cycle_config["ΔT_pp_1"]
    ΔT_pp_3 = cycle_config["ΔT_pp_3"]
    ΔT_pp_4 = cycle_config["ΔT_pp_4"]
    ΔT_sh = cycle_config["ΔT_sh"]
    fluid = f"REFPROP::{refrigerant}"

    # Station 1 - compressor inlet
    T_ev = T_c_in - ΔT_pp_1 - ΔT_sh
    p_ev = _cp_props("P", "T", T_ev, "Q", 0, fluid)
    p_ref_1 = p_ev
    T_ref_1 = T_c_in - ΔT_pp_1
    h_ref_1, s_ref_1, Q_ref_1 = _cp_props(("H", "S", "Q"), "T", T_ref_1, "P", p_ref_1, fluid)
    Q_ref_1 = _vapour_quality_scaler(Q_ref_1)

    # Station 2 - compressor outlet
    p_ref_2 = PR * p_ref_1
    h_ref_2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, fluid)
    h_ref_2 = (h_ref_2_is - h_ref_1) / η_compr + h_ref_1
    T_ref_2, Q_ref_2, s_ref_2 = _cp_props(("T", "Q", "S"), "P", p_ref_2, "H", h_ref_2, fluid)
    Q_ref_2 = _vapour_quality_scaler(Q_ref_2)

    # Station 3 - turbine inlet (pinch ΔT_pp_3 fixed)
    T_ref_3 = T_h_in + ΔT_pp_3
    p_ref_3 = p_ref_2
    h_ref_3, s_ref_3, Q_ref_3 = _cp_props(("H", "S", "Q"), "T", T_ref_3, "P", p_ref_3, fluid)
    Q_ref_3 = _vapour_quality_scaler(Q_ref_3)

    # Determine cycle type
    Pcrit = _cp_props("Pcrit", fluid)
    Tcrit = _cp_props("Tcrit", fluid)
    s_crit = _cp_props("S", "P", Pcrit, "T", Tcrit, fluid)
    supercritical_cycle = p_ref_2 > Pcrit or Q_ref_2 == 0 # latter is not supercritical, but same calculaiton procedure holds

    # Impossible cycle checks
    if T_ref_2 < T_ref_3 * 0.99:
        if verbose:
            logger.info(f"Station 2 temperature (T_ref_2 = {T_ref_2:.2f} K) is significantly lower than station 3 temperature (T_ref_3 = {T_ref_3:.2f} K).")
        return None
    if s_ref_3 > s_crit * 0.95:
        if verbose:
            logger.info("Station 3 is saturated vapour.")
            logger.info(f"Station 3 properties: T_ref_3 = {T_ref_3:.2f} K, p_ref_3 = {p_ref_3:.2f} Pa, h_ref_3 = {h_ref_3:.2f} J/kg, s_ref_3 = {s_ref_3:.2f} J/kg/K, s_crit = {s_crit:.2f} J/kg/K")
        return None
    
    p_ref_4 = p_ref_1
    h_ref_4_is = _cp_props("H", "P", p_ref_4, "S", s_ref_3, fluid)
    h_ref_4 = (h_ref_4_is - h_ref_3) * η_turb + h_ref_3
    T_ref_4, Q_ref_4, s_ref_4 = _cp_props(("T", "Q", "S"), "P", p_ref_4, "H", h_ref_4, fluid)
    Q_ref_4 = _vapour_quality_scaler(Q_ref_4)
    Q_ref_4_isenth = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_4, "H", h_ref_3, fluid))

    # Latent heat (evaporator always two-phase)
    Δh_ev = (_cp_props("H", "P", p_ref_1, "Q", 1, fluid) -
             _cp_props("H", "P", p_ref_1, "Q", 0, fluid))
    
    # Condenser saturation temperature & latent heat (only for subcritical)
    if not supercritical_cycle:
        T_cond = _cp_props("T", "P", p_ref_2, "Q", 1, fluid)
        Δh_cond = (_cp_props("H", "P", p_ref_2, "Q", 1, fluid) -
                   _cp_props("H", "P", p_ref_2, "Q", 0, fluid))
    else:
        T_cond = None
        Δh_cond = 0.0

    # ṁ_ref from evaporator pinch ΔT_pp_4
    T_h_pp_4 = T_ev + ΔT_pp_4
    denom_ev = (Δh_ev * (Q_ref_1 - Q_ref_4) +
                _specific_heat_from_isobar_path(T_ev, T_ref_1, p_ref_1, general_config, cycle_config,
                                                supercritical_cycle=supercritical_cycle))
    ṁ_ref = (T_c_in - T_h_pp_4) * ṁ_c * cp_c / denom_ev

    # Outlet temperatures
    if not supercritical_cycle:
        Q_out_ref = (_specific_heat_from_isobar_path(T_ref_2, T_cond, p_ref_2, general_config, cycle_config,
                                                     supercritical_cycle=supercritical_cycle) * ṁ_ref +
                     (Q_ref_2 - Q_ref_3) * Δh_cond * ṁ_ref +
                     _specific_heat_from_isobar_path(T_cond, T_ref_3, p_ref_2, general_config, cycle_config,
                                                     supercritical_cycle=supercritical_cycle) * ṁ_ref)
        T_h_out = T_h_in + Q_out_ref / (ṁ_h * cp_h)
    else:
        Q_out_ref = _specific_heat_from_isobar_path(T_ref_3, T_ref_2, p_ref_2, general_config, cycle_config,
                                                    supercritical_cycle=supercritical_cycle) * ṁ_ref
        T_h_out = T_h_in + Q_out_ref / (ṁ_h * cp_h)
    T_c_out = T_c_in - ((_specific_heat_from_isobar_path(T_ref_1, T_ev, p_ref_1, general_config, cycle_config,
                                                         supercritical_cycle=supercritical_cycle) * ṁ_ref +
                         (Q_ref_1 - Q_ref_4) * Δh_ev * ṁ_ref +
                         _specific_heat_from_isobar_path(T_ev, T_ref_4, p_ref_1, general_config, cycle_config,
                                                         supercritical_cycle=supercritical_cycle) * ṁ_ref) /
                        (ṁ_c * cp_c))
   
    # Heat sink temperature at the condenser pinch, for plotting
    _, T_ref_arr, h_ref_arr = _sample_isobar_ts_uniform_arc(
        h_ref_3, h_ref_2, p_ref_2, cycle_config, num_points=500
    )
    T_c_arr = T_h_in + (np.asarray(h_ref_arr) - h_ref_3) * ṁ_ref / (ṁ_h * cp_h)
    T_diff_arr = np.asarray(T_ref_arr) - T_c_arr
    valid = np.isfinite(T_diff_arr)
    if np.any(valid):
        min_idx = np.argmin(T_diff_arr[valid])
        T_h_pp_2 = float(T_c_arr[valid][min_idx])
    else:
        T_h_pp_2 = T_h_in + (T_h_out - T_h_in) * 0.5

    # Store cycle data
    cycle_data = dict(
        p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
        p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
        p_ref_3=p_ref_3, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3, Q_ref_3=Q_ref_3,
        p_ref_4=p_ref_4, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4,
        T_ev=T_ev, Δh_ev=Δh_ev,
        T_h_out=T_h_out, T_c_out=T_c_out,
        Q_ref_4_isenth=Q_ref_4_isenth,
        ṁ_ref=ṁ_ref,
        supercritical_cycle=supercritical_cycle,
        T_cond=T_cond,
        Δh_cond=Δh_cond,
        T_h_pp_2=T_h_pp_2,
        h_ref_2_is=h_ref_2_is, h_ref_4_is=h_ref_4_is
    )
    return cycle_data

# counter flow condenser minimum ΔT calculation
def _compute_min_condenser_deltaT(cycle_data, cycle_config):
    s = cycle_data
    p2 = s["p_ref_2"]
    h3 = s["h_ref_3"]
    h2 = s["h_ref_2"]
    m_ref = s["ṁ_ref"]
    m_h = cycle_config["ṁ_h"]
    cp_h = cycle_config["cp_h"]
    T_h_in = cycle_config["T_h_in"]
    _, T_ref_arr, h_ref_arr = _sample_isobar_ts_uniform_arc(
        h3, h2, p2, cycle_config, num_points=500
    )
    T_c_arr = T_h_in + (h_ref_arr - h3) * m_ref / (m_h * cp_h)
    T_diff_arr = np.asarray(T_ref_arr) - np.asarray(T_c_arr)
    valid = np.isfinite(T_diff_arr)
    if not np.any(valid):
        return np.nan
    return float(np.min(T_diff_arr[valid]))

def solve_cycle_three_temp_fix(cycle_config, general_config, verbose=True):
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    ΔT_pp_1 = cycle_config["ΔT_pp_1"]
    ΔT_sh = cycle_config["ΔT_sh"]
    T_ev = T_c_in - ΔT_pp_1 - ΔT_sh
    p_ev = _cp_props("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    PR_min = max(1.01, (p_ev + 1.0) / p_ev)
    PR_max = 30
    Pcrit = _cp_props("Pcrit", f"REFPROP::{refrigerant}")
    PR_crit = Pcrit / p_ev

    def objective(PR):
        """Single objective used for BOTH regions (returns -COP or huge penalty)."""
        cycle_data = evaluate_cycle_PR(cycle_config, general_config, PR, verbose=False)
        if cycle_data is None:
            return 1e9
        min_dt = _compute_min_condenser_deltaT(cycle_data, cycle_config)
        if min_dt < cycle_config["ΔT_pp_3"] - 1e-3:
            return 1e9
        return -_cop_turb(cycle_data, cycle_config)
    
    # The COP can have a maximum both below and above the critical pressure, hence 
    # the PR is optimised for both regions separately, after which the best is selected.
    # 3a. Subcritical optimisation (below critical point)
    PR_max_sub = PR_crit * 0.999
    best_sub_PR = None
    best_sub_COP = -np.inf
    sub_nfev = 0
    if PR_min < PR_max_sub:
        res_sub = _quiet_minimize_scalar(
            objective,
            bounds=(PR_min, PR_max_sub),
            tol=1e-9,
            maxiter=80,
        )
        if res_sub.fun < 0:                     # feasible solution found
            best_sub_PR = res_sub.x
            best_sub_COP = -res_sub.fun
            sub_nfev = res_sub.nfev

    # 3b. Supercritical optimisation (above critical point)
    PR_min_super = max(PR_crit + 1e-6, PR_min)
    best_super_PR = None
    best_super_COP = -np.inf
    super_nfev = 0
    if PR_min_super < PR_max:
        res_super = _quiet_minimize_scalar(
            objective,
            bounds=(PR_min_super, PR_max),
            tol=1e-9,
            maxiter=80,
        )
        if res_super.fun < 0:
            best_super_PR = res_super.x
            best_super_COP = -res_super.fun
            super_nfev = res_super.nfev

    # Pick the overall best (subcritical or supercritical)
    if best_sub_PR is None and best_super_PR is None:
        raise ValueError("No feasible PR found that satisfies all pinch constraints.")
    
    # Choose the one with higher COP
    if best_sub_COP >= best_super_COP:
        best_PR = best_sub_PR
        best_COP = best_sub_COP
        region = "SUBCRITICAL"
        total_evals = sub_nfev
    else:
        best_PR = best_super_PR
        best_COP = best_super_COP
        region = "SUPERCRITICAL"
        total_evals = super_nfev

    # Final evaluation of the chosen PR (full cycle data)
    best_cycle_data = evaluate_cycle_PR(cycle_config, general_config, best_PR, verbose=False)
    best_cycle_data["PR"] = best_PR
    best_cycle_data["COP_turb"] = best_COP
    if verbose:
        p2_final = best_cycle_data["p_ref_2"]
        min_dt_final = _compute_min_condenser_deltaT(best_cycle_data, cycle_config)
        logger.info(
            f"Optimised cycle found: PR = {best_PR:.6f} ({region}), "
            f"COP_turb = {best_COP:.4f}, "
            f"p2 = {p2_final/1e6:.2f} MPa, min condenser ΔT = {min_dt_final:.4f} K, "
            f"ṁ_ref = {best_cycle_data['ṁ_ref']:.4f} kg/s, "
            f"(total evaluations: {total_evals})"
        )
    return best_cycle_data
