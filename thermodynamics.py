import os
import sys
cwd = os.getcwd()
sys.path.append(f'{cwd}/verification/')

import functools
import itertools
import numpy as np
import CoolProp.CoolProp as CP
import scipy.optimize as opt
from logger import setup_logger
logger = setup_logger()

_ABSTRACT_STATES = {}

def _parse_backend_fluid(fluid_spec):
    """Split a CoolProp fluid spec into backend and fluid name.

    Args:
        fluid_spec: Fluid identifier such as 'REFPROP::R1234ze(E)'.

    Returns:
        A (backend, fluid) tuple.
    """
    if "::" in fluid_spec:
        backend, fluid = fluid_spec.split("::", 1)
        return backend, fluid
    return "REFPROP", fluid_spec

def _get_abstract_state(fluid_spec):
    """Return a cached CoolProp AbstractState for a fluid specification.

    Args:
        fluid_spec: Fluid identifier with an optional backend prefix.

    Returns:
        Cached CoolProp AbstractState instance.
    """
    backend, fluid = _parse_backend_fluid(fluid_spec)
    key = (backend, fluid)
    if key not in _ABSTRACT_STATES:
        _ABSTRACT_STATES[key] = CP.AbstractState(backend, fluid)
    return _ABSTRACT_STATES[key]

def _update_state_from_pair(state, in1, val1, in2, val2):
    """Update a CoolProp AbstractState from a supported input pair.

    Args:
        state: CoolProp AbstractState to update.
        in1: First input key, such as 'P' or 'T'.
        val1: Value paired with in1.
        in2: Second input key.
        val2: Value paired with in2.

    Returns:
        None.
    """
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
    elif pair == ("T", "S"):
        state.update(CP.SmassT_INPUTS, val2, val1)
    elif pair == ("S", "T"):
        state.update(CP.SmassT_INPUTS, val1, val2)
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
    """Clamp vapor quality to [0, 1] for single-phase regions.

    Args:
        Q: Raw vapor quality value from CoolProp.

    Returns:
        Quality clipped to the physically meaningful range.
    """
    if Q > 1:
        return 1
    elif Q < 0:
        return 0
    return Q

def _isobar_segment(s_start, s_end, p, cycle_config, general_config):
    """Build a TS curve segment along a constant-pressure path.

    Args:
        s_start: Starting entropy value.
        s_end: Ending entropy value.
        p: Constant pressure for the segment.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime and plotting configuration.

    Returns:
        Tuple of entropy and temperature lists for the path.
    """
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
    """Build a PH curve segment along a constant-pressure path.

    Args:
        h_start: Starting enthalpy value.
        h_end: Ending enthalpy value.
        p: Constant pressure for the segment.

    Returns:
        Tuple of enthalpy and pressure lists for the path.
    """
    h_range = np.linspace(h_start, h_end, num=30)
    p_range = np.full(30, p)
    return h_range.tolist(), p_range.tolist()

def _specific_heat_from_isobar_path(
    T_start, T_end, p, general_config, cycle_config, supercritical_cycle=False, uniform_sampling=False,
):
    """Estimate heat transferred along an isobaric path.

    Args:
        T_start: Starting temperature.
        T_end: Ending temperature.
        p: Constant pressure along the path.
        general_config: General runtime and plotting configuration.
        cycle_config: Cycle configuration dictionary.
        supercritical_cycle: Whether the cycle runs above the critical point.
        uniform_sampling: Whether to use arc-length-based TS sampling.

    Returns:
        Estimated specific heat transfer in J/kg.
    """
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
    """Sample an isobar using near-uniform arc-length spacing in TS space.

    Args:
        h_start: Starting enthalpy.
        h_end: Ending enthalpy.
        p: Constant pressure.
        cycle_config: Cycle configuration dictionary.
        num_points: Requested number of output samples.

    Returns:
        Tuple of entropy, temperature, and enthalpy arrays.
    """
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

# New helpers for true minimum approach (counter-flow pinch anywhere)
def _compute_min_approach_cond(ṁ_ref, h_ref_2, h_ref_3, p_ref_2, T_h_in, cp_h, ṁ_h, refrigerant):
    """Compute the minimum condenser approach temperature for a candidate cycle.

    Args:
        ṁ_ref: Refrigerant mass flow rate.
        h_ref_2: Refrigerant enthalpy at condenser inlet.
        h_ref_3: Refrigerant enthalpy at condenser outlet.
        p_ref_2: High-side refrigerant pressure.
        T_h_in: Hot-side inlet temperature.
        cp_h: Hot-side specific heat capacity.
        ṁ_h: Hot-side mass flow rate.
        refrigerant: Refrigerant name.

    Returns:
        Minimum temperature approach across the condenser, in K.
    """
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
    """Compute the minimum evaporator approach temperature for a candidate cycle.

    Args:
        ṁ_ref: Refrigerant mass flow rate.
        h_ref_4: Refrigerant enthalpy at evaporator inlet.
        h_ref_1: Refrigerant enthalpy at evaporator outlet.
        p_ref_1: Low-side refrigerant pressure.
        T_c_in: Cold-side inlet temperature.
        cp_c: Cold-side specific heat capacity.
        ṁ_c: Cold-side mass flow rate.
        refrigerant: Refrigerant name.

    Returns:
        Minimum temperature approach across the evaporator, in K.
    """
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

# ===================================================================
# EVOLUTIONARY GLOBAL OPTIMIZER (Differential Evolution)
# ===================================================================
def _run_global_de(objective, bounds, general_config, callback=None):
    """Run the global differential-evolution optimizer.

    Args:
        objective: Objective function to minimize.
        bounds: Search bounds for each decision variable.
        general_config: General runtime configuration.
        callback: Optional optimizer callback.

    Returns:
        SciPy optimization result object.
    """
    de_popsize = int(general_config.get("de_popsize", 30))
    de_maxiter = int(general_config.get("de_maxiter", 1000))
    de_tol = float(general_config.get("de_tol", 1e-3))
    de_strategy = general_config.get("de_strategy", "best1bin")
    de_init = general_config.get("de_init", "latinhypercube")
    if general_config.get("analysis_type") == "COP_vs_eff_investigation":
        de_workers = -1
        de_updating = "deferred"
    else:
        de_workers = 1
        de_updating = "immediate"

    result = opt.differential_evolution(
        objective,
        bounds,
        strategy=de_strategy,
        popsize=de_popsize,
        maxiter=de_maxiter,
        tol=de_tol,
        init=de_init,
        workers=de_workers,
        updating=de_updating,
        disp=False,
        polish=False,
        callback=callback,
        seed=42,
    )

    if general_config.get("analysis_type") != "COP_vs_eff_investigation":
        log_msg = f"Differential Evolution finished: {result.nfev:,} function evaluations, success = {result.success}"
        if not result.success:
            log_msg += f", optimizer final message = {result.message}"
        logger.info(log_msg)
    return result

def _create_optimization_trace():
    """Create an empty optimization trace container.

    Returns:
        Dictionary used to track objective evaluations and best values.
    """
    return {
        "eval_idx": [],
        "objective": [],
        "failed": [],
        "best_so_far": [],
        "iter_idx": [],
        "iter_best": [],
    }

def _record_trace_point(optimization_trace, state, score, failed):
    """Record one objective evaluation in the optimization trace.

    Args:
        optimization_trace: Mutable trace dictionary.
        state: Mutable optimizer state counters.
        score: Objective value for the current evaluation.
        failed: Whether the candidate was infeasible.

    Returns:
        None.
    """
    state["eval_counter"] += 1
    if np.isfinite(score) and score < state["best_score_seen"]:
        state["best_score_seen"] = score
    optimization_trace["eval_idx"].append(state["eval_counter"])
    optimization_trace["objective"].append(float(score))
    optimization_trace["failed"].append(bool(failed))
    optimization_trace["best_so_far"].append(float(state["best_score_seen"]))

def _de_trace_callback(optimization_trace, *args, **kwargs):
    """Record the best value seen after a differential-evolution iteration.

    Args:
        optimization_trace: Mutable trace dictionary.
        *args: Positional callback arguments from SciPy.
        **kwargs: Keyword callback arguments from SciPy.

    Returns:
        False to keep the optimizer running.
    """
    intermediate_result = kwargs.get("intermediate_result", None)
    if intermediate_result is None and len(args) > 0 and hasattr(args[0], "fun"):
        intermediate_result = args[0]
    if intermediate_result is None:
        return False
    optimization_trace["iter_idx"].append(len(optimization_trace["iter_idx"]) + 1)
    optimization_trace["iter_best"].append(float(intermediate_result.fun))
    return False

def _raw_objective(cycle_config, general_config, decision_variables, verbose, fixed_PR=None):
    """Evaluate the negative COP objective for a cycle candidate.

    Args:
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        decision_variables: Candidate variables for optimization.
        verbose: Whether to log rejection reasons.
        fixed_PR: Optional pressure ratio to hold fixed.

    Returns:
        Tuple of objective score, invalid-cycle flag, and cycle data or None.
    """
    if fixed_PR is None:
        PR, T_ref_1_var, s_ref_1_var, T_ref_3 = decision_variables
    else:
        PR = fixed_PR
        T_ref_1_var, s_ref_1_var, T_ref_3 = decision_variables
    cycle_data = evaluate_cycle_candidate(
        cycle_config,
        general_config,
        PR,
        T_ref_1_var,
        s_ref_1_var,
        T_ref_3,
        verbose=verbose,
    )
    if cycle_data is None:
        return np.random.uniform(low=0.05, high=0.15, size=None), True, None
    return -compute_performance(cycle_data, cycle_config, general_config)["COP_turb"], False, cycle_data

def _objective_with_trace(x, cycle_config, general_config, verbose, optimization_trace, trace_state, fixed_PR=None):
    """Evaluate a candidate while updating the optimization trace.

    Args:
        x: Candidate decision vector.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        verbose: Whether to log rejection reasons.
        optimization_trace: Trace container to update.
        trace_state: Mutable counters for the trace.
        fixed_PR: Optional pressure ratio to hold fixed.

    Returns:
        Objective value for SciPy minimization.
    """
    score, invalid_cycle, _ = _raw_objective(cycle_config, general_config, x, verbose, fixed_PR=fixed_PR)
    if score >= 0.05:  # invalid candidate
        _record_trace_point(optimization_trace, trace_state, score, True)
        return score
    _record_trace_point(optimization_trace, trace_state, score, invalid_cycle)
    return score

def _objective_de(x, cycle_config, general_config, verbose, optimization_trace, trace_state, fixed_PR=None):
    """Adapter that forwards a candidate to the traced objective.

    Args:
        x: Candidate decision vector.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        verbose: Whether to log rejection reasons.
        optimization_trace: Trace container to update.
        trace_state: Mutable counters for the trace.
        fixed_PR: Optional pressure ratio to hold fixed.

    Returns:
        Objective value for SciPy minimization.
    """
    return _objective_with_trace(
        x,
        cycle_config,
        general_config,
        verbose,
        optimization_trace,
        trace_state,
        fixed_PR=fixed_PR,
    )

def _de_callback_with_trace(*args, optimization_trace, **kwargs):
    """Forward the SciPy callback to the internal trace recorder.

    Args:
        *args: Positional callback arguments from SciPy.
        optimization_trace: Trace container to update.
        **kwargs: Keyword callback arguments from SciPy.

    Returns:
        False to keep the optimizer running.
    """
    return _de_trace_callback(optimization_trace, *args, **kwargs)

# Cycle solver (FULL OPTIMISATION as requested – now using DE)
def _candidate_bounds(cycle_config):
    """Build optimization bounds from the cycle configuration.

    Args:
        cycle_config: Cycle configuration dictionary.

    Returns:
        Dictionary mapping decision-variable names to lower/upper bounds.
    """
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]
    ΔT_pp_min_evap = cycle_config["ΔT_pp_min_evap"]
    ΔT_pp_min_cond = cycle_config["ΔT_pp_min_cond"]
    T_ref_1_min = T_c_in - 30.0
    T_ref_1_max = T_c_in - ΔT_pp_min_evap
    s_ref_1_min = _cp_props("S", "T", T_ref_1_max, "Q", 1, f"REFPROP::{refrigerant}") - 0.2*_cp_props("S", "T", T_ref_1_min, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_1_max = _cp_props("S", "T", T_ref_1_max, "Q", 1, f"REFPROP::{refrigerant}") + 0.2*_cp_props("S", "T", T_ref_1_max, "Q", 1, f"REFPROP::{refrigerant}")
    return {
        "PR": (1.01, 90.0),
        "s_ref_1": (s_ref_1_min, s_ref_1_max),
        "T_ref_1": (T_ref_1_min, T_ref_1_max),
        "T_ref_3": (T_h_in + ΔT_pp_min_cond, T_h_in + 50.0),
    }

def evaluate_cycle_candidate(cycle_config, general_config, PR, T_ref_1_var, s_ref_1_var, T_ref_3, verbose=True):
    """Evaluate one cycle candidate and reject infeasible states.

    Args:
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        PR: Candidate pressure ratio.
        T_ref_1_var: Candidate inlet temperature for state 1.
        s_ref_1_var: Candidate inlet entropy for state 1.
        T_ref_3: Candidate condenser outlet temperature.
        verbose: Whether to log rejection reasons.

    Returns:
        Cycle-state dictionary for feasible candidates, otherwise None.
    """
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
        return None
    fluid = f"REFPROP::{refrigerant}"
    try:
        p_ref_1 = _cp_props("P", "T", T_ref_1_var, "S", s_ref_1_var, fluid)
        T_ref_1 = T_ref_1_var
        if T_ref_1 > T_c_in - ΔT_pp_min_evap + 1e-3:
            if verbose:
                logger.info(f"Candidate rejected due to T_ref_1 constraint: T_ref_1={T_ref_1:.2f} K, T_c_in={T_c_in:.2f} K, ΔT_pp_min_evap={ΔT_pp_min_evap:.2f} K")
            return None
        p_ref_2 = PR * p_ref_1
        Q_ref_1 = _vapour_quality_scaler(_cp_props("Q", "T", T_ref_1, "P", p_ref_1, fluid))
        if Q_ref_1 < 0.999:
            if verbose:
                logger.info(f"Candidate rejected due to subcooled/ref_1 constraint: Q_ref_1={Q_ref_1:.4f}")
            return None
        T_ev = _cp_props("T", "P", p_ref_1, "Q", 1, fluid)
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
            if verbose:
                logger.info(f"Candidate rejected due to ref_2/ref_3 constraints: T_ref_2={T_ref_2:.2f} K, T_ref_3={T_ref_3:.2f} K, s_ref_3={s_ref_3:.4f}, s_crit={s_crit:.4f}")
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
            if verbose:
                logger.info(f"Candidate rejected due to negative performance values: q_out_per={q_out_per:.4f}, w_net_per={w_net_per:.4f}")
            return None
        mref_req = Q_out_req / q_out_per
        if mref_req <= 0:
            if verbose:
                logger.info(f"Candidate rejected due to non-positive mass flow requirement: mref_req={mref_req:.4f}")
            return None
        min_app_evap_val = _compute_min_approach_evap(mref_req, h_ref_4, h_ref_1, p_ref_1, T_c_in, cp_c, ṁ_c, refrigerant)
        min_app_cond_val = _compute_min_approach_cond(mref_req, h_ref_2, h_ref_3, p_ref_2, T_h_in, cp_h, ṁ_h, refrigerant)
        if min_app_evap_val < ΔT_pp_min_evap or min_app_cond_val < ΔT_pp_min_cond:
            if verbose:
                logger.info(f"Candidate rejected due to minimum approach constraints: min_app_evap_val={min_app_evap_val:.2f} K, min_app_cond_val={min_app_cond_val:.2f} K")
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
            p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_2, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3,
            Q_ref_3=0.0 if supercritical_cycle else _vapour_quality_scaler(_cp_props("Q", "T", T_ref_3, "P", p_ref_2, fluid)),
            p_ref_4=p_ref_1, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4, Q_ref_4_isenth=Q_ref_4_isenth,
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
        if verbose:
            logger.exception(f"An error occurred while evaluating the cycle candidate: {e}")
        return None

def _optimize_cycle(cycle_config, general_config, fixed_PR=None, verbose=True):
    """Optimize the heat-pump cycle using differential evolution.

    Args:
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        fixed_PR: Optional pressure ratio to keep fixed.
        verbose: Whether to log rejection reasons.

    Returns:
        Best feasible cycle-state dictionary.
    """
    bounds = _candidate_bounds(cycle_config)
    optimization_trace = _create_optimization_trace()
    trace_state = {
        "eval_counter": 0,
        "best_score_seen": np.inf,
    }
    objective = functools.partial(
        _objective_de,
        cycle_config=cycle_config,
        general_config=general_config,
        verbose=verbose,
        optimization_trace=optimization_trace,
        trace_state=trace_state,
        fixed_PR=fixed_PR,
    )
    de_callback = functools.partial(
        _de_callback_with_trace,
        optimization_trace=optimization_trace,
    )

    if fixed_PR is None:
        # Full optimisation (PR + T_ref_1 + s_ref_1 + T_ref_3)
        search_bounds = [bounds["PR"], bounds["T_ref_1"], bounds["s_ref_1"], bounds["T_ref_3"]]

        res = _run_global_de(objective, search_bounds, general_config, callback=de_callback)

        if (not np.isfinite(res.fun)) or (res.fun >= 0):
            raise ValueError("No feasible cycle found that satisfies the pinch constraints at the requested Q_out_req.")
        best_PR, best_T_ref_1, best_s_ref_1, best_T_ref_3 = res.x
        best_cycle_data = evaluate_cycle_candidate(cycle_config, general_config, best_PR, best_T_ref_1, best_s_ref_1, best_T_ref_3, verbose=verbose)
        if best_cycle_data is None:
            raise ValueError("No feasible cycle found that satisfies the pinch constraints at the requested Q_out_req.")
        best_cycle_data["PR"] = best_PR
        best_cycle_data["COP_turb"] = -res.fun
        best_cycle_data["optimization_trace"] = optimization_trace
        return best_cycle_data

    else:
        # Fixed PR optimisation
        search_bounds = [bounds["T_ref_1"], bounds["s_ref_1"], bounds["T_ref_3"]]

        res = _run_global_de(objective, search_bounds, general_config, callback=de_callback)

        if (not np.isfinite(res.fun)) or (res.fun >= 0):
            raise ValueError("Specified PR leads to an impossible cycle.")
        best_T_ref_1, best_s_ref_1, best_T_ref_3 = res.x
        best_cycle_data = evaluate_cycle_candidate(cycle_config, general_config, fixed_PR, best_T_ref_1, best_s_ref_1, best_T_ref_3, verbose=verbose)
        if best_cycle_data is None:
            raise ValueError("Specified PR leads to an impossible cycle.")
        best_cycle_data["PR"] = fixed_PR
        best_cycle_data["COP_turb"] = -res.fun
        best_cycle_data["optimization_trace"] = optimization_trace
        return best_cycle_data

def solve_cycle(cycle_config, general_config, verbose=True):
    """Solve the configured cycle optimization problem.

    Args:
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        verbose: Whether to log rejection reasons.

    Returns:
        Best feasible cycle-state dictionary.
    """
    if "PR" in cycle_config:
        return _optimize_cycle(cycle_config, general_config, fixed_PR=float(cycle_config["PR"]), verbose=verbose)
    return _optimize_cycle(cycle_config, general_config, fixed_PR=None, verbose=verbose)

# Performance metrics (unchanged)
def compute_performance(cycle_data, cycle_config, general_config):
    """Compute performance metrics from a solved cycle.

    Args:
        cycle_data: Solved cycle-state dictionary.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.

    Returns:
        Dictionary of performance metrics such as COP and power flows.
    """
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
    """Build T-s diagram data for the solved cycle.

    Args:
        cycle_data: Solved cycle-state dictionary.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.

    Returns:
        Dictionary with major and minor curve data for the TS diagram.
    """
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
    """Build P-h diagram data for the solved cycle.

    Args:
        cycle_data: Solved cycle-state dictionary.
        cycle_config: Cycle configuration dictionary.

    Returns:
        Dictionary with major and minor curve data for the PH diagram.
    """
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