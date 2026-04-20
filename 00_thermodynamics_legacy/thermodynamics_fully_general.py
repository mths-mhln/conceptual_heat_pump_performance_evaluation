"""
Problem Encountered:
- the optimizer pushes to the lowest possible ṁ_ref to satisfy the pinch constraints, which leads to near-zero cooling capacity, which is not meaningful for the COP improvements possible 
  in typical heat pump systems that we will actually build. Connected to this is the optimizer's desire to place the pinch points at ref_1 and re_3, which is made possible by the low ṁ_ref.
    - reason: the optimizer aims for largest COP = (h2-h3) + Δh_cond / (h2-h1). Which is largest the lowest possible we keep the PR. See notes on paper, but this happens if the ṁ_ref
      is so small that your pp near 2' (the inflection point) is nearly the same ΔT (but slightly smaller) than you experience for ref_3. For the evaporation, the lowest PR is achieved by 
      placing the pp at ref_1. Note that this is very similar behavior to the brayton cycle where if we try to maximize the cycle on our metric of η_th, we end up with a super thin cycle
      that maximizes the cooling curve, at the expense of the cycle not doing any actual net work, simply because that metric accounts for one of the heating curves and one of the work curves. 
      This is also the reason why for the HP cycle, no subcooling will be present, since that is thermodynamically unfavorable (you press for larger PR for very little heating enthalpy gain). 
    - The underlying reason for why this is thermodyanmically feasible is because the varaition in thdy properties is simply as represented on the map. That is how our reality functions. 
    - For this reason, instead of a fully general optimization, you can account for the fact that the optimizer will push for the PP to occur at T_ref1 and T_pp_2 anyway by simply placing the pp
      there from the start. However, you can of course never be fully sure that this is the best option, what about subcooling, what about supercritical cycles. Hence I will just leave this as is, 
      implement th feedback of Carlo and be done with this shit tbh. 
"""



import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/')
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/verification/')
import math
import itertools
import numpy as np
import CoolProp.CoolProp as CP
import scipy.optimize as opt
from logger import setup_logger

logger = setup_logger()

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
    """AbstractState-backed PropsSI replacement for this module.
    Supports the signatures used in this file:
    - _cp_props(output, fluid)
    - _cp_props(output, in1, val1, in2, val2, fluid)
    """
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
# =======

def _vapour_quality_scaler(Q):
    """Clamp CoolProp quality to [0, 1] for single-phase regions."""
    if Q > 1:
        return 1
    elif Q < 0:
        return 0
    return Q

def _isobar_segment(s_start, s_end, p, cycle_config, general_config):
    """Return (s_range, T_range) along a constant-pressure path (TS diagram)."""
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
    """Return (h_range, p_range) along a constant-pressure path (PH diagram)."""
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

# New helpers for true minimum approach (counter-flow pinch anywhere)
def _compute_min_approach_cond(ṁ_ref, h_ref_2, h_ref_3, p_ref_2, T_c_in, cp_c, ṁ_c, refrigerant):
    if ṁ_ref <= 0:
        return np.inf
    Q_cond = ṁ_ref * (h_ref_2 - h_ref_3)
    num_points = 200
    q_arr = np.linspace(0, Q_cond, num_points)
    delta_t = np.full(num_points, np.nan)
    state = CP.AbstractState("REFPROP", refrigerant)
    for i, q in enumerate(q_arr):
        h = h_ref_2 - q / ṁ_ref
        try:
            state.update(CP.HmassP_INPUTS, h, p_ref_2)
            T_ref = state.T()
            T_ext = T_c_in + q / (ṁ_c * cp_c)
            delta_t[i] = T_ref - T_ext
        except:
            pass
    return np.nanmin(delta_t)

def _compute_min_approach_evap(ṁ_ref, h_ref_4, h_ref_1, p_ref_1, T_h_in, cp_h, ṁ_h, refrigerant):
    if ṁ_ref <= 0:
        return np.inf
    Q_evap = ṁ_ref * (h_ref_1 - h_ref_4)
    num_points = 200
    q_arr = np.linspace(0, Q_evap, num_points)
    delta_t = np.full(num_points, np.nan)
    state = CP.AbstractState("REFPROP", refrigerant)
    for i, q in enumerate(q_arr):
        h = h_ref_4 + q / ṁ_ref
        try:
            state.update(CP.HmassP_INPUTS, h, p_ref_1)
            T_ref = state.T()
            T_ext = T_h_in - (Q_evap - q) / (ṁ_h * cp_h)
            delta_t[i] = T_ext - T_ref
        except:
            pass
    return np.nanmin(delta_t)

def _find_max_mref(compute_min_approach, target_delta, bounds=(1e-3, 100), tol=1e-6):
    low, high = bounds
    while high - low > tol:
        mid = (low + high) / 2
        if compute_min_approach(mid) >= target_delta:
            low = mid
        else:
            high = mid
    return low

# Cycle solver (FULL OPTIMISATION as requested)
def solve_cycle(cycle_config, general_config, verbose=True):
    """Full optimisation over PR, T_ref_1 and T_ref_3.
    - Maximises specific COP = (h2-h3) / w_net_per_kg
    - Pinch points are strict inequality constraints (min ΔT anywhere ≥ ΔT_pp_min_xxx)
    - End temperature differences are allowed to be larger than the minimum (the optimiser will choose the best trade-off)
    - Only the two physically meaningful pinch constraints are used.
    """
    refrigerant = cycle_config["refrigerant"]
    T_h_in = cycle_config["T_h_in"]
    T_c_in = cycle_config["T_c_in"]
    ṁ_h = cycle_config["ṁ_h"]
    ṁ_c = cycle_config["ṁ_c"]
    cp_h = cycle_config["cp_h"]
    cp_c = cycle_config["cp_c"]
    η_compr = cycle_config["η_compr"]
    η_turb = cycle_config["η_turb"]
    ΔT_sh = cycle_config["ΔT_sh"]
    ΔT_pp_min_evap = cycle_config["ΔT_pp_min_evap"]
    ΔT_pp_min_cond = cycle_config["ΔT_pp_min_cond"]

    def objective(x):
        PR, T_ref_1, T_ref_3 = x
        if PR <= 1.0 or T_ref_1 > T_h_in - ΔT_pp_min_evap + 1e-3 or T_ref_3 < T_c_in + ΔT_pp_min_cond - 1e-3:
            return 1e6
        try:
            T_ev = T_ref_1 - ΔT_sh
            p_ref_1 = _cp_props("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
            p_ref_2 = PR * p_ref_1

            # Station 1
            h_ref_1 = _cp_props("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
            s_ref_1 = _cp_props("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")

            # Station 2
            h2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
            h_ref_2 = h_ref_1 + (h2_is - h_ref_1) / η_compr

            # Station 3 (free)
            h_ref_3 = _cp_props("H", "T", T_ref_3, "P", p_ref_2, f"REFPROP::{refrigerant}")
            s_ref_3 = _cp_props("S", "T", T_ref_3, "P", p_ref_2, f"REFPROP::{refrigerant}")

            # Station 4
            h4_is = _cp_props("H", "P", p_ref_1, "S", s_ref_3, f"REFPROP::{refrigerant}")
            h_ref_4 = h_ref_3 - η_turb * (h_ref_3 - h4_is)

            q_out_per = h_ref_2 - h_ref_3
            w_net_per = (h_ref_2 - h_ref_1) - (h_ref_3 - h_ref_4) * cycle_config.get("ɳ_shaft", 1.0)
            if w_net_per <= 0 or q_out_per <= 0:
                return 1e6

            # Feasibility check (pinch inequality)
            def min_app_evap(mr):
                return _compute_min_approach_evap(mr, h_ref_4, h_ref_1, p_ref_1, T_h_in, cp_h, ṁ_h, refrigerant)
            def min_app_cond(mr):
                return _compute_min_approach_cond(mr, h_ref_2, h_ref_3, p_ref_2, T_c_in, cp_c, ṁ_c, refrigerant)

            mref_max_evap = _find_max_mref(min_app_evap, ΔT_pp_min_evap)
            mref_max_cond = _find_max_mref(min_app_cond, ΔT_pp_min_cond)

            if mref_max_evap < 1e-6 or mref_max_cond < 1e-6:
                return 1e6

            return -(q_out_per / w_net_per)   # negative for maximisation
        except Exception:
            return 1e6

    # Global optimisation (3 variables)
    bounds = [
        (1.01, 30),                                      # PR
        (T_h_in - 150, T_h_in - ΔT_pp_min_evap),         # T_ref_1 (can be lower → larger ΔT at station 1)
        (T_c_in + ΔT_pp_min_cond, T_c_in + 150)          # T_ref_3 (can be higher → larger ΔT at station 3)
    ]
    res = opt.differential_evolution(
        objective,
        bounds,
        tol=1e-5,
        atol=1e-5,
        popsize=20,
        workers=1,
        disp=False
    )

    if res.fun >= 0:
        raise ValueError("No feasible cycle found that satisfies the pinch constraints.")

    optimal_PR, optimal_T_ref_1, optimal_T_ref_3 = res.x
    if verbose:
        logger.info(f"Optimal PR = {optimal_PR:.4f} | Optimal T_ref_1 = {optimal_T_ref_1:.2f} K (ΔT_end_evap possibly > min) | "
                    f"Optimal T_ref_3 = {optimal_T_ref_3:.2f} K (ΔT_end_cond possibly > min) | Max COP = {-res.fun:.4f}")

    # Re-build full states with optimal values
    T_ev = optimal_T_ref_1 - ΔT_sh
    p_ref_1 = _cp_props("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    p_ref_2 = optimal_PR * p_ref_1

    h_ref_1 = _cp_props("H", "T", optimal_T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    s_ref_1 = _cp_props("S", "T", optimal_T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")

    h2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
    h_ref_2 = h_ref_1 + (h2_is - h_ref_1) / η_compr
    T_ref_2 = _cp_props("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
    Q_ref_2 = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}"))
    s_ref_2 = _cp_props("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")

    h_ref_3 = _cp_props("H", "T", optimal_T_ref_3, "P", p_ref_2, f"REFPROP::{refrigerant}")
    s_ref_3 = _cp_props("S", "T", optimal_T_ref_3, "P", p_ref_2, f"REFPROP::{refrigerant}")
    Q_ref_3 = _vapour_quality_scaler(_cp_props("Q", "T", optimal_T_ref_3, "P", p_ref_2, f"REFPROP::{refrigerant}"))

    h4_is = _cp_props("H", "P", p_ref_1, "S", s_ref_3, f"REFPROP::{refrigerant}")
    h_ref_4 = h_ref_3 - η_turb * (h_ref_3 - h4_is)
    T_ref_4 = _cp_props("T", "P", p_ref_1, "H", h_ref_4, f"REFPROP::{refrigerant}")
    Q_ref_4 = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_1, "H", h_ref_4, f"REFPROP::{refrigerant}"))
    Q_ref_4_isenth = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_1, "H", h_ref_3, f"REFPROP::{refrigerant}"))
    s_ref_4 = _cp_props("S", "P", p_ref_1, "H", h_ref_4, f"REFPROP::{refrigerant}")

    supercritical_cycle = p_ref_2 > _cp_props("Pcrit", f"REFPROP::{refrigerant}")

    # Final feasible ṁ_ref
    def min_app_evap(mr):
        return _compute_min_approach_evap(mr, h_ref_4, h_ref_1, p_ref_1, T_h_in, cp_h, ṁ_h, refrigerant)
    def min_app_cond(mr):
        return _compute_min_approach_cond(mr, h_ref_2, h_ref_3, p_ref_2, T_c_in, cp_c, ṁ_c, refrigerant)

    mref_max_evap = _find_max_mref(min_app_evap, ΔT_pp_min_evap)
    mref_max_cond = _find_max_mref(min_app_cond, ΔT_pp_min_cond)
    ṁ_ref = min(mref_max_evap, mref_max_cond)

    # Outlet temperatures
    Q_out = ṁ_ref * (h_ref_2 - h_ref_3)
    T_c_out = T_c_in + Q_out / (ṁ_c * cp_c)
    Q_in = ṁ_ref * (h_ref_1 - h_ref_4)
    T_h_out = T_h_in - Q_in / (ṁ_h * cp_h)

    # Legacy field for plotting compatibility
    T_c_pp_2 = T_c_in + (T_c_out - T_c_in) * 0.5   # approximate (exact pinch location not required for COP)

    cycle_data = dict(
        p_ref_1=p_ref_1,
        T_ref_1=optimal_T_ref_1,
        h_ref_1=h_ref_1,
        s_ref_1=s_ref_1,
        Q_ref_1=_vapour_quality_scaler(_cp_props("Q", "T", optimal_T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")),
        p_ref_2=p_ref_2,
        T_ref_2=T_ref_2,
        h_ref_2=h_ref_2,
        s_ref_2=s_ref_2,
        Q_ref_2=Q_ref_2,
        p_ref_3=p_ref_2,
        T_ref_3=optimal_T_ref_3,
        h_ref_3=h_ref_3,
        s_ref_3=s_ref_3,
        Q_ref_3=Q_ref_3,
        p_ref_4=p_ref_1,
        T_ref_4=T_ref_4,
        h_ref_4=h_ref_4,
        s_ref_4=s_ref_4,
        Q_ref_4=Q_ref_4,
        Q_ref_4_isenth=Q_ref_4_isenth,
        T_ev=T_ev,
        supercritical_cycle=supercritical_cycle,
        ṁ_ref=ṁ_ref,
        T_c_out=T_c_out,
        T_h_out=T_h_out,
        T_c_pp_2=T_c_pp_2,
        T_cond=_cp_props("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") if not supercritical_cycle else None,
        Δh_ev=(_cp_props("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}") -
               _cp_props("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}")),
        Δh_cond=(_cp_props("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") -
                 _cp_props("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")) if not supercritical_cycle else 0,
    )

    if verbose:
        logger.info(f"Final ṁ_ref = {ṁ_ref:.4f} kg/s | Pinch exactly at the limit somewhere along the curves")

    return cycle_data

# Performance metrics (unchanged)
def compute_performance(cycle_data, cycle_config, general_config):
    s = cycle_data
    ṁ_c = cycle_config["ṁ_c"]
    cp_c = cycle_config["cp_c"]
    T_c_in = cycle_config["T_c_in"]
    ɳ_shaft = cycle_config.get("ɳ_shaft", 1.0)
    refrigerant = cycle_config["refrigerant"]
    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out = ṁ_c * cp_c * (s["T_c_out"] - T_c_in)
    Q_in = s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
           s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4"]) + \
           s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ev"], s["T_ref_4"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])
    Q_in_isenth = s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4_isenth"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_4"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])
    COP_turb = Q_out / (Ẇ_comp - Ẇ_turb * ɳ_shaft)
    h_ref_4_is = _cp_props("H", "P", s["p_ref_4"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
    Ẇ_turb_is = s["ṁ_ref"] * (s["h_ref_3"] - h_ref_4_is)
    COP_is = Q_out / (Ẇ_comp - Ẇ_turb_is * ɳ_shaft)
    COP_isenth = Q_out / Ẇ_comp
    return dict(
        Ẇ_turb=Ẇ_turb,
        Ẇ_comp=Ẇ_comp,
        Q_out=Q_out,
        Q_in=Q_in,
        COP_turb=COP_turb,
        COP_is=COP_is,
        COP_isenth=COP_isenth,
        Q_in_isenth=Q_in_isenth,
        PR=s["p_ref_2"] / s["p_ref_1"],
        ṁ_ref=s["ṁ_ref"]
    )

# Diagram data preparation (unchanged – fully compatible)
def build_ts_data(cycle_data, cycle_config, general_config):
    s = cycle_data
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]
    p1, p2 = s["p_ref_1"], s["p_ref_2"]

    if not s["supercritical_cycle"]:
        s_ref_23_v_inflection = _cp_props("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        s_ref_23_l_inflection = _cp_props("S", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
        s_ref_41_v_inflection = _cp_props("S", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
        s_ref_41_l_inflection = _cp_props("S", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

        # Condenser path 2 -> 3
        s_23_chain, T_23_chain = [], []
        if s["s_ref_2"] > s_ref_23_v_inflection:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s_ref_23_v_inflection, p2, cycle_config, general_config)
            s_23_chain += seg_s + [s_ref_23_v_inflection]
            T_23_chain += seg_T + [s.get("T_cond", s["T_ref_3"])]
        if s["s_ref_3"] < s_ref_23_l_inflection:
            seg_s, seg_T = _isobar_segment(s_ref_23_l_inflection, s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain += [s_ref_23_l_inflection] + seg_s
            T_23_chain += [s.get("T_cond", s["T_ref_3"])] + seg_T
        if not s_23_chain:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain, T_23_chain = seg_s, seg_T

        # Evaporator path 4 -> 1
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
    else:
        seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
        s_23_chain, T_23_chain = seg_s, seg_T
        seg_s, seg_T = _isobar_segment(s["s_ref_4"], s["s_ref_1"], p1, cycle_config, general_config)
        s_41_chain, T_41_chain = seg_s, seg_T

    s_ref_lst = list(itertools.chain(
        [s["s_ref_1"]], [s["s_ref_2"]], s_23_chain, [s["s_ref_3"]], [s["s_ref_4"]], s_41_chain
    ))
    T_ref_lst = list(itertools.chain(
        [s["T_ref_1"]], [s["T_ref_2"]], T_23_chain, [s["T_ref_3"]], [s["T_ref_4"]], T_41_chain
    ))

    if not s["supercritical_cycle"]:
        s_ref_23_v_inflection = _cp_props("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        if s["Q_ref_2"] >= 1:
            s_pp_anchor = s_ref_23_v_inflection
            s_c_out = s["s_ref_3"] + (s_pp_anchor - s["s_ref_3"]) / (s["T_c_pp_2"] - T_c_in) * (s["T_c_out"] - T_c_in)
        else:
            s_c_out = s["s_ref_2"]
    else:
        s_c_out = s["s_ref_2"]

    return dict(
        major={"s": [s["s_ref_1"], s["s_ref_2"], s["s_ref_3"], s["s_ref_4"], s["s_ref_1"]],
               "T": [s["T_ref_1"], s["T_ref_2"], s["T_ref_3"], s["T_ref_4"], s["T_ref_1"]]},
        minor={"s": s_ref_lst, "T": T_ref_lst},
        coolant={"s": [s["s_ref_3"], s_c_out], "T": [T_c_in, s["T_c_out"]]},
        heating={"s": [s["s_ref_1"], s["s_ref_4"]], "T": [T_h_in, s["T_h_out"]]},
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
    else:
        seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
        h_23_chain, p_23_chain = seg_h, seg_p
        seg_h, seg_p = _isobar_h_segment(s["h_ref_4"], s["h_ref_1"], p1)
        h_41_chain, p_41_chain = seg_h, seg_p

    h_ref_lst = list(itertools.chain(
        [s["h_ref_1"]], [s["h_ref_2"]], h_23_chain, [s["h_ref_3"]], [s["h_ref_4"]], h_41_chain
    ))
    p_ref_lst = list(itertools.chain(
        [p1], [p2], p_23_chain, [s["p_ref_3"]], [s["p_ref_4"]], p_41_chain
    ))

    return dict(
        major={"h": [s["h_ref_1"], s["h_ref_2"], s["h_ref_3"], s["h_ref_4"], s["h_ref_1"]],
               "p": [p1, p2, s["p_ref_3"], s["p_ref_4"], p1]},
        minor={"h": h_ref_lst, "p": p_ref_lst},
    )



