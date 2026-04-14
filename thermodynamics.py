import sys
sys.path.append('./verification/')

import math
import itertools
import io
import contextlib
import numpy as np
import CoolProp.CoolProp as CP

from logger import setup_logger
from scipy.optimize import minimize_scalar

logger = setup_logger()



# Fluid properties
_ABSTRACT_STATES = {}
def _parse_backend_fluid(fluid_spec):
    if "::" in fluid_spec:
        backend, fluid = fluid_spec.split("::", 1)
        return backend, fluid
    return "REFPROP", fluid_spec

def _get_abstract_state(fluid_spec):
    backend, fluid = _parse_backend_fluid(fluid_spec)
    key = (backend, fluid)
    if key not in _ABSTRACT_STATES:
        _ABSTRACT_STATES[key] = CP.AbstractState(backend, fluid)
    return _ABSTRACT_STATES[key]

def _update_state_from_pair(state, in1, val1, in2, val2):
    pair = (in1, in2)
    if pair == ("P", "T"):
        state.update(CP.PT_INPUTS, val1, val2)
    elif pair == ("T", "P"):
        state.update(CP.PT_INPUTS, val2, val1)
    elif pair == ("P", "Q"):
        state.update(CP.PQ_INPUTS, val1, val2)
    elif pair == ("Q", "P"):
        state.update(CP.PQ_INPUTS, val2, val1)
    elif pair == ("T", "Q"):
        state.update(CP.QT_INPUTS, val2, val1)
    elif pair == ("Q", "T"):
        state.update(CP.QT_INPUTS, val1, val2)
    elif pair == ("P", "H"):
        state.update(CP.HmassP_INPUTS, val2, val1)
    elif pair == ("H", "P"):
        state.update(CP.HmassP_INPUTS, val1, val2)
    elif pair == ("P", "S"):
        state.update(CP.PSmass_INPUTS, val1, val2)
    elif pair == ("S", "P"):
        state.update(CP.PSmass_INPUTS, val2, val1)
    else:
        raise NotImplementedError(f"Unsupported input pair for AbstractState: {pair}")
    
def _cp_props(*args):
    try:
        if len(args) == 2:
            output, fluid = args
            state = _get_abstract_state(fluid)
            if output == "Pcrit":
                return state.p_critical()
            return CP.PropsSI(output, fluid)
        if len(args) == 6:
            output, in1, val1, in2, val2, fluid = args
            state = _get_abstract_state(fluid)
            _update_state_from_pair(state, in1, val1, in2, val2)
            if output == "T":
                return state.T()
            if output == "P":
                return state.p()
            if output == "H":
                return state.hmass()
            if output == "S":
                return state.smass()
            if output == "Q":
                return state.Q()
            if output == "C":
                return state.cpmass()
            if output == "Pcrit":
                return state.p_critical()
            return CP.PropsSI(output, in1, val1, in2, val2, fluid)
        raise TypeError("PropsSI wrapper received unsupported signature")
    except RuntimeError as err:
        raise ValueError(str(err)) from err
    


# Helpers
def _vapour_quality_scaler(Q):
    if Q > 1:
        return 1
    elif Q < 0:
        return 0
    return Q

def _isobar_segment(s_start, s_end, p, cycle_config, general_config):
    resolution = general_config["resolution"]
    refrigerant = cycle_config["refrigerant"]
    num_points = 150 if resolution == "low" else 1000
    s_range = np.linspace(s_start, s_end, num=num_points)
    T_range = np.zeros(num_points)
    for i, s in enumerate(s_range):
        try:
            T_range[i] = _cp_props("T", "P", p, "S", s, f"REFPROP::{refrigerant}")
        except ValueError:
            pass
    return s_range.tolist(), T_range.tolist()

def _isobar_h_segment(h_start, h_end, p):
    h_range = np.linspace(h_start, h_end, num=30)
    p_range = np.full(30, p)
    return h_range.tolist(), p_range.tolist()

def _specific_heat_from_isobar_path(
    T_start, T_end, p, general_config, cycle_config, supercritical_cycle=False, uniform_sampling=False,
):
    num_points = 150 if general_config["resolution"] == "low" else 1000
    if supercritical_cycle or uniform_sampling:
        h_start = _cp_props("H", "P", p, "T", T_start, f"REFPROP::{cycle_config['refrigerant']}")
        h_end = _cp_props("H", "P", p, "T", T_end, f"REFPROP::{cycle_config['refrigerant']}")
        _, T_range, _ = _sample_isobar_ts_uniform_arc(
            h_start, h_end, p, cycle_config, num_points=num_points,
        )
    else:
        T_range = np.linspace(T_start, T_end, num=num_points)
    if len(T_range) < 2:
        return 0
    heat = 0
    for T_1, T_2 in zip(T_range[:-1], T_range[1:]):
        try:
            cp = _cp_props("C", "P", p, "T", T_1, f"REFPROP::{cycle_config['refrigerant']}")
            heat += cp * np.abs(T_2 - T_1)
        except ValueError:
            pass
    return heat

def _sample_isobar_ts_uniform_arc(h_start, h_end, p, cycle_config, num_points=2000):
    refrigerant = cycle_config["refrigerant"]
    n = max(num_points, 400)
    h_arr = np.linspace(h_start, h_end, n)
    T_arr = np.full_like(h_arr, np.nan, dtype=float)
    s_arr = np.full_like(h_arr, np.nan, dtype=float)
    state = CP.AbstractState("REFPROP", refrigerant)
    for i, h in enumerate(h_arr):
        try:
            state.update(CP.HmassP_INPUTS, h, p)
            T_arr[i] = state.T()
            s_arr[i] = state.smass()
        except (ValueError, RuntimeError):
            pass
    valid = np.isfinite(T_arr) & np.isfinite(s_arr) & np.isfinite(h_arr)
    if valid.sum() < 4:
        return s_arr[valid], T_arr[valid], h_arr[valid]
    T_valid = T_arr[valid]
    s_valid = s_arr[valid]
    h_valid = h_arr[valid]
    T_scale = max(np.max(T_valid) - np.min(T_valid), 1e-12)
    s_scale = max(np.max(s_valid) - np.min(s_valid), 1e-12)
    T_norm = (T_valid - np.min(T_valid)) / T_scale
    s_norm = (s_valid - np.min(s_valid)) / s_scale
    seg_len = np.sqrt(np.diff(T_norm) ** 2 + np.diff(s_norm) ** 2)
    arc = np.concatenate(([0.0], np.cumsum(seg_len)))
    if arc[-1] <= 0:
        return s_valid, T_valid, h_valid
    arc_uniform = np.linspace(0.0, arc[-1], num_points)
    T_uniform = np.interp(arc_uniform, arc, T_valid)
    s_uniform = np.interp(arc_uniform, arc, s_valid)
    h_uniform = np.interp(arc_uniform, arc, h_valid)
    return s_uniform, T_uniform, h_uniform

def _check_second_derivative_sign(s_arr, T_arr):
    dT_ds = np.gradient(T_arr, s_arr)
    d2T_ds2 = np.gradient(dT_ds, s_arr)
    if np.all(d2T_ds2 > 0):
        return 1
    elif np.all(d2T_ds2 < 0):
        return -1
    else:
        return 0
    
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

    # Station 1 — compressor inlet
    T_ev = T_c_in - ΔT_pp_1 - ΔT_sh
    p_ev = _cp_props("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    p_ref_1 = p_ev
    T_ref_1 = T_c_in - ΔT_pp_1
    h_ref_1 = _cp_props("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    s_ref_1 = _cp_props("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    Q_ref_1 = _vapour_quality_scaler(_cp_props("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}"))

    # Station 2 — compressor outlet
    p_ref_2 = PR * p_ref_1
    h_ref_2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
    h_ref_2 = (h_ref_2_is - h_ref_1) / η_compr + h_ref_1
    T_ref_2 = _cp_props("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
    Q_ref_2 = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}"))
    s_ref_2 = _cp_props("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")

    # Station 3 — turbine inlet (pinch ΔT_pp_3 fixed)
    T_ref_3 = T_h_in + ΔT_pp_3
    p_ref_3 = p_ref_2
    h_ref_3 = _cp_props("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    s_ref_3 = _cp_props("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    Q_ref_3 = _vapour_quality_scaler(_cp_props("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}"))

    # Determine cycle type
    Pcrit = _cp_props("Pcrit", f"REFPROP::{refrigerant}")
    Tcrit = _cp_props("Tcrit", f"REFPROP::{refrigerant}")
    s_crit = _cp_props("S", "P", Pcrit, "T", Tcrit, f"REFPROP::{refrigerant}")
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
    h_ref_4_is = _cp_props("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
    h_ref_4 = (h_ref_4_is - h_ref_3) * η_turb + h_ref_3
    T_ref_4 = _cp_props("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
    Q_ref_4 = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}"))
    Q_ref_4_isenth = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_4, "H", h_ref_3, f"REFPROP::{refrigerant}"))
    s_ref_4 = _cp_props("S", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")

    # Latent heat (evaporator always two-phase)
    Δh_ev = (_cp_props("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}") -
             _cp_props("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}"))
    
    # Condenser saturation temperature & latent heat (only for subcritical)
    if not supercritical_cycle:
        T_cond = _cp_props("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
        Δh_cond = (_cp_props("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") -
                   _cp_props("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}"))
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
   
    # T_c_pp_2 for plotting
    _, T_ref_arr, h_ref_arr = _sample_isobar_ts_uniform_arc(
        h_ref_3, h_ref_2, p_ref_2, cycle_config, num_points=500
    )
    T_c_arr = T_h_in + (np.asarray(h_ref_arr) - h_ref_3) * ṁ_ref / (ṁ_h * cp_h)
    T_diff_arr = np.asarray(T_ref_arr) - T_c_arr
    valid = np.isfinite(T_diff_arr)
    if np.any(valid):
        min_idx = np.argmin(T_diff_arr[valid])
        T_c_pp_2 = float(T_c_arr[valid][min_idx])
    else:
        T_c_pp_2 = False

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
        T_c_pp_2=T_c_pp_2,
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

def solve_cycle(cycle_config, general_config, verbose=True):
    if "PR" in cycle_config:
        PR = cycle_config["PR"]
        cycle_data = evaluate_cycle_PR(cycle_config, general_config, PR, verbose=verbose)
        if cycle_data is None:
            raise ValueError("Specified PR leads to an impossible cycle.")
        min_dt = _compute_min_condenser_deltaT(cycle_data, cycle_config)
        if min_dt < cycle_config["ΔT_pp_3"] - 1e-3:
            logger.warning(f"Fixed PR = {PR} violates condenser pinch (min ΔT = {min_dt:.3f} K < ΔT_pp_3).")
        cycle_data["PR"] = PR
        return cycle_data
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    ΔT_pp_1 = cycle_config["ΔT_pp_1"]
    ΔT_sh = cycle_config["ΔT_sh"]
    T_ev = T_c_in - ΔT_pp_1 - ΔT_sh
    p_ev = _cp_props("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    PR_min = max(1.01, (p_ev + 1.0) / p_ev)
    PR_max = 30

    # Any supercritical point with p2 ∈ (Pcrit, Pcrit × critical_buffer_ratio]
    # is treated as infeasible → optimiser automatically selects the
    # second global maximum (the first physically reliable peak). This was
    # necessary due to unreasonable behavior near the critical point by CoolProp.
    critical_buffer_ratio = float(cycle_config.get("near_critical_buffer_ratio", 1.05))
    if critical_buffer_ratio <= 1.0:
        raise ValueError("'near_critical_buffer_ratio' must be > 1.0")
    Pcrit = _cp_props("Pcrit", f"REFPROP::{refrigerant}")
    PR_crit = Pcrit / p_ev
    PR_buffer = (Pcrit * critical_buffer_ratio) / p_ev
    def objective(PR):
        """Single objective used for BOTH regions (returns -COP or huge penalty)."""
        cycle_data = evaluate_cycle_PR(cycle_config, general_config, PR, verbose=False)
        if cycle_data is None:
            return 1e9
        min_dt = _compute_min_condenser_deltaT(cycle_data, cycle_config)
        if min_dt < cycle_config["ΔT_pp_3"] - 1e-3:
            return 1e9
        
        # Near-critical buffer (only relevant in supercritical region, the moment we go subcritical no strange THDY property stuff occurs)
        p2 = cycle_data["p_ref_2"]
        if Pcrit < p2 <= Pcrit * critical_buffer_ratio:
            return 1e9
        perf = compute_performance(cycle_data, cycle_config, general_config)
        return -perf["COP_turb"]
    
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

    # supercritical optimisation (above buffer)
    PR_min_super = max(PR_buffer + 1e-6, PR_min)
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

    # Pick the overall best (subcritical or safe-supercritical)
    if best_sub_PR is None and best_super_PR is None:
        raise ValueError(
            "No feasible PR found that satisfies all pinch constraints "
            "and stays outside the near-critical buffer zone."
        )
    
    # Choose the one with higher COP
    if best_sub_COP >= best_super_COP:
        best_PR = best_sub_PR
        best_COP = best_sub_COP
        region = "SUBCRITICAL"
        total_evals = sub_nfev
    else:
        best_PR = best_super_PR
        best_COP = best_super_COP
        region = f"SUPERCRITICAL (safe, buffer = {critical_buffer_ratio})"
        total_evals = super_nfev

    # Final evaluation of the chosen PR (full cycle data)
    best_cycle_data = evaluate_cycle_PR(cycle_config, general_config, best_PR, verbose=False)
    best_cycle_data["PR"] = best_PR
    best_cycle_data["COP_turb"] = best_COP
    if verbose:
        p2_final = best_cycle_data["p_ref_2"]
        min_dt_final = _compute_min_condenser_deltaT(best_cycle_data, cycle_config)
        logger.info(
            f"Optimised cycle found → PR = {best_PR:.6f} ({region}), "
            f"COP_turb = {best_COP:.4f}, "
            f"p2 = {p2_final/1e6:.2f} MPa, min condenser ΔT = {min_dt_final:.4f} K "
            f"(total evaluations: {total_evals})"
        )
    return best_cycle_data



# Performance metrics
# ===================
def compute_performance(cycle_data, cycle_config, general_config):
    s = cycle_data
    ṁ_h = cycle_config["ṁ_h"]
    cp_h = cycle_config["cp_h"]
    T_h_in = cycle_config["T_h_in"]
    ɳ_shaft = cycle_config["ɳ_shaft"]
    refrigerant = cycle_config["refrigerant"]
    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out   = ṁ_h * cp_h * (s["T_h_out"] - T_h_in)
    Q_in    = s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
              s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4"]) + \
              s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ev"], s["T_ref_4"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])
    Q_in_isenth = s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4_isenth"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_4"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])
    COP_turb = Q_out / (Ẇ_comp - Ẇ_turb * ɳ_shaft)
    h_ref_4_is = _cp_props("H", "P", s["p_ref_4"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
    Ẇ_turb_is  = s["ṁ_ref"] * (s["h_ref_3"] - h_ref_4_is)
    COP_is     = Q_out / (Ẇ_comp - Ẇ_turb_is * ɳ_shaft)
    COP_isenth = Q_out / Ẇ_comp
    return dict(
        Ẇ_turb=Ẇ_turb, Ẇ_comp=Ẇ_comp, Q_out=Q_out, Q_in=Q_in,
        COP_turb=COP_turb, COP_is=COP_is, COP_isenth=COP_isenth,
        Q_in_isenth=Q_in_isenth, PR=s["p_ref_2"] / s["p_ref_1"],
        ṁ_ref=s["ṁ_ref"]
    )



# Diagram data preparation
# ========================
def build_ts_data(cycle_data, cycle_config, general_config):
    s = cycle_data
    refrigerant = cycle_config["refrigerant"]
    T_h_in = cycle_config["T_h_in"]
    T_c_in = cycle_config["T_c_in"]
    p1, p2 = s["p_ref_1"], s["p_ref_2"]
    if not s["supercritical_cycle"]:
        s_ref_23_v_inflection = _cp_props("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        s_ref_23_l_inflection = _cp_props("S", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    s_ref_41_v_inflection = _cp_props("S", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_41_l_inflection = _cp_props("S", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")
    
    # Condenser path 2 -> 3:
    if s["supercritical_cycle"]:
        seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
        s_23_chain, T_23_chain = seg_s, seg_T
    else:
        s_23_chain, T_23_chain = [], []
        if s["s_ref_2"] > s_ref_23_v_inflection:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s_ref_23_v_inflection, p2, cycle_config, general_config)
            s_23_chain += seg_s + [s_ref_23_v_inflection]
            T_23_chain += seg_T + [s["T_cond"]]
        if s["s_ref_3"] < s_ref_23_l_inflection:
            seg_s, seg_T = _isobar_segment(s_ref_23_l_inflection, s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain += [s_ref_23_l_inflection] + seg_s
            T_23_chain += [s["T_cond"]] + seg_T
        if not s_23_chain:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain, T_23_chain = seg_s, seg_T

    # Evaporator path 4 -> 1:
    s_41_chain, T_41_chain = [], []
    if s["s_ref_4"] < s_ref_41_l_inflection:
        seg_s, seg_T = _isobar_segment(s["s_ref_4"], s_ref_41_l_inflection, p1, cycle_config, general_config)
        s_41_chain += seg_s + [s_ref_41_l_inflection]
        T_41_chain += seg_T + [s["T_ev"]]
    if s["s_ref_1"] > s_ref_41_v_inflection:
        seg_s, seg_T = _isobar_segment(s_ref_41_v_inflection, s["s_ref_1"], p1, cycle_config, general_config)
        s_41_chain += [s_ref_41_v_inflection] + seg_s
        T_41_chain += [s["T_ev"]] + seg_T
    if not s_41_chain:
        seg_s, seg_T = _isobar_segment(s["s_ref_4"], s["s_ref_1"], p1, cycle_config, general_config)
        s_41_chain, T_41_chain = seg_s, seg_T
    s_ref_lst = list(itertools.chain(
        [s["s_ref_1"]], [s["s_ref_2"]], s_23_chain, [s["s_ref_3"]],
        [s["s_ref_4"]], s_41_chain
    ))
    T_ref_lst = list(itertools.chain(
        [s["T_ref_1"]], [s["T_ref_2"]], T_23_chain, [s["T_ref_3"]],
        [s["T_ref_4"]], T_41_chain
    ))
    s_ref_23_v_inflection = _cp_props("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
    T_c_pp_2 = s.get("T_c_pp_2", False)
    if T_c_pp_2 is False:
        s_c_out = s["s_ref_2"]
    elif s["Q_ref_2"] >= 1 and np.isfinite(T_c_pp_2) and not np.isclose(T_c_pp_2, T_h_in):
        s_pp_anchor = s_ref_23_v_inflection
        s_c_out = s["s_ref_3"] + (s_pp_anchor - s["s_ref_3"]) / (T_c_pp_2 - T_h_in) * (s["T_h_out"] - T_h_in)
    else:
        s_c_out = s["s_ref_2"]
    return dict(
        major={"s": [s["s_ref_1"], s["s_ref_2"], s["s_ref_3"], s["s_ref_4"], s["s_ref_1"]],
               "T": [s["T_ref_1"], s["T_ref_2"], s["T_ref_3"], s["T_ref_4"], s["T_ref_1"]]},
        minor={"s": s_ref_lst, "T": T_ref_lst},
        coolant={"s": [s["s_ref_3"], s_c_out], "T": [T_h_in, s["T_h_out"]]},
        heating={"s": [s["s_ref_1"], s["s_ref_4"]], "T": [T_c_in, s["T_c_out"]]},
    )

def build_ph_data(cycle_data, cycle_config):
    s = cycle_data
    refrigerant = cycle_config["refrigerant"]
    p1, p2 = s["p_ref_1"], s["p_ref_2"]
    if not s["supercritical_cycle"]:
        h_ref_23_v_inflection = _cp_props("H", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        h_ref_23_l_inflection = _cp_props("H", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    h_ref_41_v_inflection = _cp_props("H", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_41_l_inflection = _cp_props("H", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

    # Condenser path 2 -> 3:
    if s["supercritical_cycle"]:
        seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
        h_23_chain, p_23_chain = seg_h, seg_p
    else:
        h_23_chain, p_23_chain = [], []
        if s["h_ref_2"] > h_ref_23_v_inflection:
            seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], h_ref_23_v_inflection, p2)
            h_23_chain += seg_h + [h_ref_23_v_inflection]
            p_23_chain += seg_p + [p2]
        if s["h_ref_3"] < h_ref_23_l_inflection:
            seg_h, seg_p = _isobar_h_segment(h_ref_23_l_inflection, s["h_ref_3"], p2)
            h_23_chain += [h_ref_23_l_inflection] + seg_h
            p_23_chain += [p2] + seg_p
        if not h_23_chain:
            seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
            h_23_chain, p_23_chain = seg_h, seg_p

    # Evaporator path 4 -> 1:
    h_41_chain, p_41_chain = [], []
    if s["h_ref_4"] < h_ref_41_l_inflection:
        seg_h, seg_p = _isobar_h_segment(s["h_ref_4"], h_ref_41_l_inflection, p1)
        h_41_chain += seg_h + [h_ref_41_l_inflection]
        p_41_chain += seg_p + [p1]
    if s["h_ref_1"] > h_ref_41_v_inflection:
        seg_h, seg_p = _isobar_h_segment(h_ref_41_v_inflection, s["h_ref_1"], p1)
        h_41_chain += [h_ref_41_v_inflection] + seg_h
        p_41_chain += [p1] + seg_p
    if not h_41_chain:
        seg_h, seg_p = _isobar_h_segment(s["h_ref_4"], s["h_ref_1"], p1)
        h_41_chain, p_41_chain = seg_h, seg_p
    h_ref_lst = list(itertools.chain(
        [s["h_ref_1"]], [s["h_ref_2"]], h_23_chain, [s["h_ref_3"]],
        [s["h_ref_4"]], h_41_chain
    ))
    p_ref_lst = list(itertools.chain(
        [p1], [p2], p_23_chain, [s["p_ref_3"]],
        [s["p_ref_4"]], p_41_chain
    ))
    return dict(
        major={"h": [s["h_ref_1"], s["h_ref_2"], s["h_ref_3"], s["h_ref_4"], s["h_ref_1"]],
               "p": [p1, p2, s["p_ref_3"], s["p_ref_4"], p1]},
        minor={"h": h_ref_lst, "p": p_ref_lst},
    )