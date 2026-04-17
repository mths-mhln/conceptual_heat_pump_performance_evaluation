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
    elif pair == ("H", "S"):
        state.update(CP.HmassSmass_INPUTS, val1, val2)
    elif pair == ("S", "H"):
        state.update(CP.HmassSmass_INPUTS, val2, val1)
    elif pair == ("T", "H"):
        state.update(CP.HmassT_INPUTS, val2, val1)
    elif pair == ("H", "T"):
        state.update(CP.HmassT_INPUTS, val1, val2)
    else:
        raise NotImplementedError(f"Unsupported input pair for AbstractState: {pair}")

def _cp_props(*args):
    """AbstractState-backed PropsSI replacement for this module."""
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
def _compute_min_approach_cond(ṁ_ref, h_ref_2, h_ref_3, p_ref_2, T_h_in, cp_h, ṁ_h, refrigerant):
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
            T_ext = T_h_in + q / (ṁ_h * cp_h)
            delta_t[i] = T_ref - T_ext
        except:
            pass
    return np.nanmin(delta_t)

def _compute_min_approach_evap(ṁ_ref, h_ref_4, h_ref_1, p_ref_1, T_c_in, cp_c, ṁ_c, refrigerant):
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
            T_ext = T_c_in - (Q_evap - q) / (ṁ_c * cp_c)
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

# ===================================================================
# EVOLUTIONARY GLOBAL OPTIMIZER (Differential Evolution)
# ===================================================================
def _run_global_de(objective, bounds, general_config):
    """Differential Evolution – replaces SHGO for true global search with many iterations.
    Configurable via general_config keys:
        de_popsize, de_maxiter, de_tol, de_strategy, de_init
    """
    de_popsize = int(general_config.get("de_popsize", 30))
    de_maxiter = int(general_config.get("de_maxiter", 1000))
    de_tol = float(general_config.get("de_tol", 1e-8))
    de_strategy = general_config.get("de_strategy", "best1bin")
    de_init = general_config.get("de_init", "latinhypercube")

    result = opt.differential_evolution(
        objective,
        bounds,
        strategy=de_strategy,
        popsize=de_popsize,
        maxiter=de_maxiter,
        tol=de_tol,
        init=de_init,
        workers=1,          # CoolProp is not always thread-safe
        disp=True,
        polish=True,
        seed=42,
    )

    logger.info(f"Differential Evolution finished – nfev = {result.nfev:,}, "
                f"success = {result.success}, message = {result.message}")
    return result, result.nfev

# Cycle solver (FULL OPTIMISATION as requested – now using DE)
def _candidate_bounds(cycle_config):
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]
    ΔT_pp_min_evap = cycle_config["ΔT_pp_min_evap"]
    ΔT_pp_min_cond = cycle_config["ΔT_pp_min_cond"]
    T_ref_1_min = T_c_in - 50.0
    print(T_ref_1_min)
    T_ref_1_max = T_c_in - ΔT_pp_min_evap
    p_ref_1_min = _cp_props("P", "T", T_ref_1_min, "Q", 1, f"REFPROP::{refrigerant}")    
    p_ref_1_max = _cp_props("P", "T", T_c_in, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_central = _cp_props("H", "T", T_ref_1_max, "Q", 1, f"REFPROP::{refrigerant}")
    print(T_ref_1_max, p_ref_1_max, refrigerant)
    print({
        "PR": (1.01, 30.0),
        "h_ref_1": (h_ref_central*0.9, h_ref_central*1.1),
        "p_ref_1": (p_ref_1_min, p_ref_1_max),
        "T_ref_3": (T_h_in + ΔT_pp_min_cond, T_h_in + 150.0),
    })
    return {
        "PR": (1.01, 30.0),
        "h_ref_1": (h_ref_central*0.9, h_ref_central*1.1),
        "p_ref_1": (p_ref_1_min, p_ref_1_max),
        "T_ref_3": (T_h_in + ΔT_pp_min_cond, T_h_in + 150.0),
    }

def _refine_bounds_around_point(bounds, point, shrink_fraction=0.15):
    """Kept for potential future use (not needed with DE)."""
    refined_bounds = []
    for (lower, upper), value in zip(bounds, point):
        span = upper - lower
        half_width = max(span * shrink_fraction, 1e-9)
        refined_lower = max(lower, value - half_width)
        refined_upper = min(upper, value + half_width)
        if refined_upper <= refined_lower:
            refined_lower, refined_upper = lower, upper
        refined_bounds.append((refined_lower, refined_upper))
    return refined_bounds

def evaluate_cycle_candidate(cycle_config, general_config, PR, h_ref_1_var, p_ref_1_var, T_ref_3, verbose=True):
    """Evaluate one candidate cycle and return cycle data or None if it is impossible."""
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]
    ṁ_c = cycle_config["ṁ_c"]
    ṁ_h = cycle_config["ṁ_h"]
    cp_c = cycle_config["cp_c"]
    cp_h = cycle_config["cp_h"]
    η_compr = cycle_config["η_compr"]
    η_turb = cycle_config["η_turb"]
    ΔT_pp_min_evap = cycle_config["ΔT_pp_min_evap"]
    ΔT_pp_min_cond = cycle_config["ΔT_pp_min_cond"]
    Q_out_req = cycle_config.get("Q_out_req")
    if Q_out_req is None or Q_out_req <= 0:
        raise ValueError("cycle_config must contain Q_out_req > 0")
    if PR <= 1.0 or T_ref_3 < T_h_in + ΔT_pp_min_cond - 1e-3:
        print(f"Candidate rejected due to PR or T_ref_3 constraints: PR={PR}, T_ref_3={T_ref_3:.2f} K")
        return None
    fluid = f"REFPROP::{refrigerant}"
    try:
        p_ref_1 = p_ref_1_var
        T_ref_1 = _cp_props("T", "H", h_ref_1_var, "P", p_ref_1, fluid)
        if T_ref_1 > T_c_in - ΔT_pp_min_evap + 1e-3:
            print(f"Candidate rejected due to T_ref_1 constraint: T_ref_1={T_ref_1:.2f} K")
            return None
        p_ref_2 = PR * p_ref_1
        Q_ref_1 = _vapour_quality_scaler(_cp_props("Q", "T", T_ref_1, "P", p_ref_1, fluid))
        if Q_ref_1 < 0.999:
            print(f"Candidate rejected due to ref_1 quality constraint: Q_ref_1={Q_ref_1:.4f}")
            return None
        T_ev = _cp_props("T", "P", p_ref_1, "Q", 0, fluid)
        h_ref_1 = _cp_props("H", "T", T_ref_1, "P", p_ref_1, fluid)
        s_ref_1 = _cp_props("S", "T", T_ref_1, "P", p_ref_1, fluid)
        h2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, fluid)
        h_ref_2 = h_ref_1 + (h2_is - h_ref_1) / η_compr
        T_ref_2 = _cp_props("T", "P", p_ref_2, "H", h_ref_2, fluid)
        Pcrit = _cp_props("Pcrit", fluid)
        Q_ref_2 = 0.0 if p_ref_2 > Pcrit else _vapour_quality_scaler(_cp_props("Q", "P", p_ref_2, "H", h_ref_2, fluid))
        s_ref_2 = _cp_props("S", "P", p_ref_2, "H", h_ref_2, fluid)
        h_ref_3 = _cp_props("H", "T", T_ref_3, "P", p_ref_2, fluid)
        s_ref_3 = _cp_props("S", "T", T_ref_3, "P", p_ref_2, fluid)
        Tcrit = _cp_props("Tcrit", fluid)
        s_crit = _cp_props("S", "P", Pcrit, "T", Tcrit, fluid)
        if T_ref_2 < T_ref_3 * 0.99 or s_ref_3 > s_crit * 0.95:
            print(f"Candidate rejected due to ref_2/ref_3 constraints: T_ref_2={T_ref_2:.2f} K, T_ref_3={T_ref_3:.2f} K, s_ref_3={s_ref_3:.4f}, s_crit={s_crit:.4f}")
            return None
        h4_is = _cp_props("H", "P", p_ref_1, "S", s_ref_3, fluid)
        h_ref_4 = h_ref_3 - η_turb * (h_ref_3 - h4_is)
        T_ref_4 = _cp_props("T", "P", p_ref_1, "H", h_ref_4, fluid)
        Q_ref_4 = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_1, "H", h_ref_4, fluid))
        Q_ref_4_isenth = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_1, "H", h_ref_3, fluid))
        s_ref_4 = _cp_props("S", "P", p_ref_1, "H", h_ref_4, fluid)
        q_out_per = h_ref_2 - h_ref_3
        w_net_per = (h_ref_2 - h_ref_1) - (h_ref_3 - h_ref_4) * cycle_config.get("ɳ_shaft", 1.0)
        if q_out_per <= 0 or w_net_per <= 0:
            print(f"Candidate rejected due to negative performance values: q_out_per={q_out_per:.4f}, w_net_per={w_net_per:.4f}")
            return None
        mref_req = Q_out_req / q_out_per
        if mref_req <= 0:
            print(f"Candidate rejected due to non-positive mass flow requirement: mref_req={mref_req:.4f}")
            return None
        min_app_evap_val = _compute_min_approach_evap(mref_req, h_ref_4, h_ref_1, p_ref_1, T_c_in, cp_c, ṁ_c, refrigerant)
        min_app_cond_val = _compute_min_approach_cond(mref_req, h_ref_2, h_ref_3, p_ref_2, T_h_in, cp_h, ṁ_h, refrigerant)
        if min_app_evap_val < ΔT_pp_min_evap or min_app_cond_val < ΔT_pp_min_cond:
            print(f"Candidate rejected due to minimum approach constraints: min_app_evap_val={min_app_evap_val:.2f} K, min_app_cond_val={min_app_cond_val:.2f} K")
            return None
        supercritical_cycle = p_ref_2 > Pcrit or Q_ref_2 == 0
        Δh_ev = _cp_props("H", "P", p_ref_1, "Q", 1, fluid) - _cp_props("H", "P", p_ref_1, "Q", 0, fluid)
        Δh_cond = (
            _cp_props("H", "P", p_ref_2, "Q", 1, fluid) - _cp_props("H", "P", p_ref_2, "Q", 0, fluid)
        ) if not supercritical_cycle else 0.0
        T_cond = _cp_props("T", "P", p_ref_2, "Q", 1, fluid) if not supercritical_cycle else None
        T_h_out = T_h_in + Q_out_req / (ṁ_h * cp_h)
        Q_in = mref_req * (h_ref_1 - h_ref_4)
        T_c_out = T_c_in - Q_in / (ṁ_c * cp_c)
        T_h_pp_2 = T_h_in + (T_h_out - T_h_in) * 0.5
        return dict(
            PR=PR,
            p_ref_1=p_ref_1,
            T_ref_1=T_ref_1,
            h_ref_1=h_ref_1,
            s_ref_1=s_ref_1,
            Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2,
            T_ref_2=T_ref_2,
            h_ref_2=h_ref_2,
            s_ref_2=s_ref_2,
            Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_2,
            T_ref_3=T_ref_3,
            h_ref_3=h_ref_3,
            s_ref_3=s_ref_3,
            Q_ref_3=0.0 if supercritical_cycle else _vapour_quality_scaler(_cp_props("Q", "T", T_ref_3, "P", p_ref_2, fluid)),
            p_ref_4=p_ref_1,
            T_ref_4=T_ref_4,
            h_ref_4=h_ref_4,
            s_ref_4=s_ref_4,
            Q_ref_4=Q_ref_4,
            Q_ref_4_isenth=Q_ref_4_isenth,
            T_ev=T_ev,
            supercritical_cycle=supercritical_cycle,
            ṁ_ref=mref_req,
            T_h_out=T_h_out,
            T_c_out=T_c_out,
            T_h_pp_2=T_h_pp_2,
            T_cond=T_cond,
            Δh_ev=Δh_ev,
            Δh_cond=Δh_cond,
            Q_out_req=Q_out_req,
            Q_out=Q_out_req,
        )
    except Exception as e:
        logger.error(f"An error occurred while evaluating the cycle candidate: {e}")
        return None

def _optimize_cycle(cycle_config, general_config, fixed_PR=None, verbose=True):
    """Optimize the cycle or, when PR is supplied, optimize the remaining variables at that PR.
    Now uses Differential Evolution + anti-degenerate m_ref penalty."""
    bounds = _candidate_bounds(cycle_config)
    min_mref_threshold = general_config.get("min_mref_threshold", 1e-6)   # <<< prevents near-zero cooling capacity

    if fixed_PR is None:
        # Full optimisation (PR + h_ref_1 + p_ref_1 + T_ref_3)
        search_bounds = [bounds["PR"], bounds["h_ref_1"], bounds["p_ref_1"], bounds["T_ref_3"]]

        def raw_objective(x):
            cycle_data = evaluate_cycle_candidate(cycle_config, general_config, x[0], x[1], x[2], x[3], verbose=False)
            if cycle_data is None:
                return 1e6 + np.random.normal(loc=0.0, scale=500.0)
            return -compute_performance(cycle_data, cycle_config, general_config)["COP_turb"]

        def objective(x):
            score = raw_objective(x)
            if score >= 1e3:          # invalid candidate
                return score
            # Re-evaluate to get ṁ_ref for penalty
            cycle_data = evaluate_cycle_candidate(cycle_config, general_config, x[0], x[1], x[2], x[3], verbose=False)
            mref = cycle_data.get("ṁ_ref", 0.0) if cycle_data is not None else 0.0
            if mref < min_mref_threshold:
                penalty = 1e7 * (min_mref_threshold - mref) / min_mref_threshold
                return 1e6 + penalty
            return score

        res, total_nfev = _run_global_de(objective, search_bounds, general_config)

        if (not np.isfinite(res.fun)) or (res.fun >= 0):
            raise ValueError("No feasible cycle found that satisfies the pinch constraints at the requested Q_out_req.")
        best_PR, best_h_ref_1, best_p_ref_1, best_T_ref_3 = res.x
        best_cycle_data = evaluate_cycle_candidate(cycle_config, general_config, best_PR, best_h_ref_1, best_p_ref_1, best_T_ref_3, verbose=False)
        if best_cycle_data is None:
            raise ValueError("No feasible cycle found that satisfies the pinch constraints at the requested Q_out_req.")
        best_cycle_data["PR"] = best_PR
        best_cycle_data["COP_turb"] = -res.fun
        if verbose:
            logger.info(
                f"Optimised cycle found → PR = {best_PR:.6f}, h_ref_1 = {best_h_ref_1:.2f} J/kg, p_ref_1 = {best_p_ref_1:.2f} Pa, "
                f"T_ref_3 = {best_T_ref_3:.2f} K, COP_turb = {-res.fun:.4f}, Q_out_req = {cycle_config['Q_out_req']:.2f} W, m_ref = {best_cycle_data['ṁ_ref']:.6f} kg/s"
            )
        logger.info(f"Objective function evaluations: {total_nfev}")
        return best_cycle_data

    else:
        # Fixed PR optimisation
        search_bounds = [bounds["h_ref_1"], bounds["p_ref_1"], bounds["T_ref_3"]]

        def raw_objective(x):
            cycle_data = evaluate_cycle_candidate(cycle_config, general_config, fixed_PR, x[0], x[1], x[2], verbose=False)
            if cycle_data is None:
                return 1e6 + np.random.normal(loc=0.0, scale=500.0)
            return -compute_performance(cycle_data, cycle_config, general_config)["COP_turb"]

        def objective(x):
            score = raw_objective(x)
            if score >= 1e3:          # invalid candidate
                return score
            cycle_data = evaluate_cycle_candidate(cycle_config, general_config, fixed_PR, x[0], x[1], x[2], verbose=False)
            mref = cycle_data.get("ṁ_ref", 0.0) if cycle_data is not None else 0.0
            if mref < min_mref_threshold:
                penalty = 1e7 * (min_mref_threshold - mref) / min_mref_threshold
                return 1e6 + penalty
            return score

        res, total_nfev = _run_global_de(objective, search_bounds, general_config)

        if (not np.isfinite(res.fun)) or (res.fun >= 0):
            raise ValueError("Specified PR leads to an impossible cycle.")
        best_h_ref_1, best_p_ref_1, best_T_ref_3 = res.x
        best_cycle_data = evaluate_cycle_candidate(cycle_config, general_config, fixed_PR, best_h_ref_1, best_p_ref_1, best_T_ref_3, verbose=False)
        if best_cycle_data is None:
            raise ValueError("Specified PR leads to an impossible cycle.")
        best_cycle_data["PR"] = fixed_PR
        best_cycle_data["COP_turb"] = -res.fun
        if verbose:
            logger.info(
                f"Fixed PR cycle found → PR = {fixed_PR:.6f}, h_ref_1 = {best_h_ref_1:.2f} J/kg, p_ref_1 = {best_p_ref_1:.2f} Pa, "
                f"T_ref_3 = {best_T_ref_3:.2f} K, COP_turb = {-res.fun:.4f}, Q_out_req = {cycle_config['Q_out_req']:.2f} W, m_ref = {best_cycle_data['ṁ_ref']:.6f} kg/s"
            )
            logger.info(f"Objective function evaluations: {total_nfev}")
        return best_cycle_data

def solve_cycle(cycle_config, general_config, verbose=True):
    """Optimize the cycle or, when PR is supplied, optimize the remaining variables at that PR."""
    if "PR" in cycle_config:
        return _optimize_cycle(cycle_config, general_config, fixed_PR=float(cycle_config["PR"]), verbose=verbose)
    return _optimize_cycle(cycle_config, general_config, fixed_PR=None, verbose=verbose)

# Performance metrics (unchanged)
def compute_performance(cycle_data, cycle_config, general_config):
    s = cycle_data
    ṁ_h = cycle_config["ṁ_h"]
    cp_h = cycle_config["cp_h"]
    T_h_in = cycle_config["T_h_in"]
    ɳ_shaft = cycle_config.get("ɳ_shaft", 1.0)
    refrigerant = cycle_config["refrigerant"]
    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out = ṁ_h * cp_h * (s["T_h_out"] - T_h_in)
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
    T_h_in = cycle_config["T_h_in"]
    T_c_in = cycle_config["T_c_in"]
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
            s_h_out = s["s_ref_3"] + (s_pp_anchor - s["s_ref_3"]) / (s["T_h_pp_2"] - T_h_in) * (s["T_h_out"] - T_h_in)
        else:
            s_h_out = s["s_ref_2"]
    else:
        s_h_out = s["s_ref_2"]
    return dict(
        major={"s": [s["s_ref_1"], s["s_ref_2"], s["s_ref_3"], s["s_ref_4"], s["s_ref_1"]],
               "T": [s["T_ref_1"], s["T_ref_2"], s["T_ref_3"], s["T_ref_4"], s["T_ref_1"]]},
        minor={"s": s_ref_lst, "T": T_ref_lst},
        coolant={"s": [s["s_ref_3"], s_h_out], "T": [T_h_in, s["T_h_out"]]},
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