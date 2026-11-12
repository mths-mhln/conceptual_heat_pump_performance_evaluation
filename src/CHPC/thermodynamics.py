import functools
import itertools
import numpy as np
import scipy.optimize as opt
from thermoplot import CoolPropAbstractState
from CHPC.logger import setup_logger
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
    """Return a cached CoolPropAbstractState for a fluid specification.

    Args:
        fluid_spec: Fluid identifier with an optional backend prefix.

    Returns:
        Cached CoolPropAbstractState instance.
    """
    backend, fluid = _parse_backend_fluid(fluid_spec)
    key = (backend, fluid)
    if key not in _ABSTRACT_STATES:
        _ABSTRACT_STATES[key] = CoolPropAbstractState(backend, fluid)
    return _ABSTRACT_STATES[key]

def _cp_props(output, *args):
    """PropsSI replacement making use of the CoolPropAbstractState class.

    Args:
        output: Output key such as 'T', or a tuple of output keys such as ('H', 'S') at the same state.
        *args: Either the fluid only, e.g. 'REFPROP::R1234ze(E)', for constants such as 'Pcrit',
            or in1, val1, in2, val2 and the fluid, as for PropsSI.

    Returns:
        Property value, or a tuple of property values if output is a tuple.

    Raises:
        ValueError: If the state cannot be evaluated. The class returns nan in that case, whereas
            the cycle calculations rely on an error to reject infeasible candidates.
    """
    *inputs, fluid = args
    state = _get_abstract_state(fluid)
    outputs = output if isinstance(output, tuple) else (output,)
    outputs = tuple("Cpmass" if o == "C" else o for o in outputs)   # PropsSI syntax "C" is "Cpmass" for the class
    if inputs:
        values = state.PropsSI(outputs, *inputs)
    else:
        values = tuple(state.PropsSI(o) for o in outputs)
    values = tuple(float(v) for v in values)
    if any(np.isnan(v) for v in values):
        raise ValueError(f"Unable to evaluate {output} for inputs {inputs} of {fluid}")
    return values if isinstance(output, tuple) else values[0]

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
        general_config: General runtime configuration.

    Returns:
        Tuple of entropy and temperature lists for the path.
    """
    refrigerant = cycle_config["refrigerant"]
    num_points = general_config.get("n_pts", 150)
    s_range = np.linspace(s_start, s_end, num=num_points)
    T_range = _get_abstract_state(f"REFPROP::{refrigerant}").PropsSI("T", "P", p, "S", s_range)
    T_range = np.nan_to_num(T_range, nan=0.0)
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
        general_config: General runtime configuration.
        cycle_config: Cycle configuration dictionary.
        supercritical_cycle: Whether the cycle runs above the critical point.
        uniform_sampling: Whether to use arc-length-based TS sampling.

    Returns:
        Estimated specific heat transfer in J/kg.
    """
    num_points = general_config.get("n_pts", 150)
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
    cp_range = _get_abstract_state(f"REFPROP::{cycle_config['refrigerant']}").PropsSI("Cpmass", "P", p, "T", np.asarray(T_range[:-1]))
    heat = 0
    for cp, T_1, T_2 in zip(cp_range, T_range[:-1], T_range[1:]):
        if np.isfinite(cp):
            heat += cp * np.abs(T_2 - T_1)
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
    T_arr, s_arr = _get_abstract_state(f"REFPROP::{refrigerant}").PropsSI(("T", "S"), "H", h_arr, "P", p)
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

# Minimum approach temperatures (pinch anywhere in the heat exchanger)
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
    h_arr = h_ref_2 - q_arr / ṁ_ref
    T_ref = _get_abstract_state(f"REFPROP::{refrigerant}").PropsSI("T", "H", h_arr, "P", p_ref_2)
    T_ext = T_h_in + q_arr / (ṁ_h * cp_h)
    delta_t = T_ref - T_ext
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
    h_arr = h_ref_4 + q_arr / ṁ_ref
    T_ref = _get_abstract_state(f"REFPROP::{refrigerant}").PropsSI("T", "H", h_arr, "P", p_ref_1)
    T_ext = T_c_in - (Q_evap - q_arr) / (ṁ_c * cp_c)
    delta_t = T_ext - T_ref
    return np.nanmin(delta_t)

# ===================================================================
# EVOLUTIONARY GLOBAL OPTIMIZER (Differential Evolution)
# ===================================================================
def _run_global_de(objective, bounds, general_config):
    """Run the global differential-evolution optimizer.

    For the COP vs efficiency investigation the population is evaluated in parallel. For a single 
    configuration the candidates are evaluated one by one, such that the optimization trace can be recorded.

    Args:
        objective: Objective function to minimize.
        bounds: Search bounds for each decision variable.
        general_config: General runtime configuration.

    Returns:
        SciPy optimization result object.
    """
    de_popsize = int(general_config.get("de_popsize", 30))
    de_maxiter = int(general_config.get("de_maxiter", 1000))
    de_tol = float(general_config.get("de_tol", 1e-3))
    de_strategy = general_config.get("de_strategy", "best1bin")
    de_init = general_config.get("de_init", "latinhypercube")
    if general_config.get("analysis_type") == "COP_vs_eff":
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
        seed=42,
    )

    if general_config.get("analysis_type") != "COP_vs_eff":
        log_msg = f"Differential Evolution finished: {result.nfev:,} function evaluations, success = {result.success}"
        if not result.success:
            log_msg += f", optimizer final message = {result.message}"
        logger.info(log_msg)
    return result

def _objective(x, cycle_config, general_config, fixed_PR=None, trace=None, verbose=False):
    """Evaluate the negative COP_turb of a cycle candidate, which is minimized by the optimizer.

    Infeasible candidates receive a random penalty between 0.05 and 0.15 rather than a single value,
    such that they remain distinguishable for the optimizer.

    Args:
        x: Candidate decision vector [PR, T_ref_1, s_ref_1, T_ref_3], or [T_ref_1, s_ref_1, T_ref_3] if PR is fixed.
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        fixed_PR: Optional pressure ratio to hold fixed.
        trace: Optional dictionary in which the objective value and feasibility of each evaluation are stored.
        verbose: Whether to log rejection reasons.

    Returns:
        Objective value for SciPy minimization.
    """
    PR, T_ref_1_var, s_ref_1_var, T_ref_3 = x if fixed_PR is None else (fixed_PR, *x)
    cycle_data = evaluate_cycle_candidate(cycle_config, PR, T_ref_1_var, s_ref_1_var, T_ref_3, verbose=verbose)
    if cycle_data is None:
        score = np.random.uniform(low=0.05, high=0.15, size=None)
    else:
        score = -_cop_turb(cycle_data, cycle_config)
    if trace is not None:
        trace["objective"].append(float(score))
        trace["failed"].append(cycle_data is None)
    return score

# Cycle solver
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

def evaluate_cycle_candidate(cycle_config, PR, T_ref_1_var, s_ref_1_var, T_ref_3, verbose=True):
    """Evaluate one cycle candidate and reject infeasible states.

    Args:
        cycle_config: Cycle configuration dictionary.
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
        h_ref_1, s_ref_1, Q_ref_1 = _cp_props(("H", "S", "Q"), "T", T_ref_1, "P", p_ref_1, fluid)
        Q_ref_1 = _vapour_quality_scaler(Q_ref_1)
        if Q_ref_1 < 0.999:
            if verbose:
                logger.info(f"Candidate rejected due to subcooled/ref_1 constraint: Q_ref_1={Q_ref_1:.4f}")
            return None
        T_ev = _cp_props("T", "P", p_ref_1, "Q", 1, fluid)
        h2_is = _cp_props("H", "P", p_ref_2, "S", s_ref_1, fluid)
        h_ref_2 = h_ref_1 + (h2_is - h_ref_1) / η_compr
        T_ref_2, s_ref_2, Q_ref_2 = _cp_props(("T", "S", "Q"), "P", p_ref_2, "H", h_ref_2, fluid)
        Pcrit = _cp_props("Pcrit", fluid)
        Q_ref_2 = 0.0 if p_ref_2 > Pcrit else _vapour_quality_scaler(Q_ref_2)
        h_ref_3, s_ref_3 = _cp_props(("H", "S"), "T", T_ref_3, "P", p_ref_2, fluid)
        Tcrit = _cp_props("Tcrit", fluid)
        s_crit = _cp_props("S", "P", Pcrit, "T", Tcrit, fluid)
        if T_ref_2 < T_ref_3 * 0.99 or s_ref_3 > s_crit * 0.95:
            if verbose:
                logger.info(f"Candidate rejected due to ref_2/ref_3 constraints: T_ref_2={T_ref_2:.2f} K, T_ref_3={T_ref_3:.2f} K, s_ref_3={s_ref_3:.4f}, s_crit={s_crit:.4f}")
            return None
        h4_is = _cp_props("H", "P", p_ref_1, "S", s_ref_3, fluid)
        h_ref_4 = h_ref_3 - η_turb * (h_ref_3 - h4_is)
        T_ref_4, Q_ref_4, s_ref_4 = _cp_props(("T", "Q", "S"), "P", p_ref_1, "H", h_ref_4, fluid)
        Q_ref_4 = _vapour_quality_scaler(Q_ref_4)
        Q_ref_4_isenth = _vapour_quality_scaler(_cp_props("Q", "P", p_ref_1, "H", h_ref_3, fluid))
        q_out_per = h_ref_2 - h_ref_3
        w_net_per = (h_ref_2 - h_ref_1) - (h_ref_3 - h_ref_4) * cycle_config.get("η_shaft", 1.0)
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
            logger.info(f"An error occurred while evaluating the cycle candidate: {e}")
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
    # Full optimisation (PR + T_ref_1 + s_ref_1 + T_ref_3), or T_ref_1 + s_ref_1 + T_ref_3 at a fixed PR
    bounds = _candidate_bounds(cycle_config)
    search_bounds = [bounds["T_ref_1"], bounds["s_ref_1"], bounds["T_ref_3"]]
    if fixed_PR is None:
        search_bounds.insert(0, bounds["PR"])
        error_message = "No feasible cycle found that satisfies the pinch constraints at the requested Q_out_req."
    else:
        error_message = "Specified PR leads to an impossible cycle."

    optimization_trace = {"objective": [], "failed": []}
    objective = functools.partial(
        _objective,
        cycle_config=cycle_config,
        general_config=general_config,
        fixed_PR=fixed_PR,
        trace=optimization_trace,
        verbose=verbose,
    )
    res = _run_global_de(objective, search_bounds, general_config)

    if (not np.isfinite(res.fun)) or (res.fun >= 0):
        raise ValueError(error_message)
    best_PR, best_T_ref_1, best_s_ref_1, best_T_ref_3 = res.x if fixed_PR is None else (fixed_PR, *res.x)
    best_cycle_data = evaluate_cycle_candidate(cycle_config, best_PR, best_T_ref_1, best_s_ref_1, best_T_ref_3, verbose=verbose)
    if best_cycle_data is None:
        raise ValueError(error_message)
    best_cycle_data["PR"] = best_PR
    best_cycle_data["COP_turb"] = -res.fun
    best_cycle_data["optimization_trace"] = optimization_trace
    return best_cycle_data

def solve_cycle(cycle_config, general_config, verbose=True):
    """Solve the cycle with the calculation method specified in the general configuration.

    full_optimization optimizes PR, T_ref_1, s_ref_1 and T_ref_3, fixed_PR optimizes T_ref_1, s_ref_1 
    and T_ref_3 at the PR in the cycle configuration, and three_temp_fix fixes the temperature 
    differences ΔT_pp_1, ΔT_pp_3, ΔT_pp_4 and ΔT_sh and only optimizes PR (thermodynamics_legacy_4).

    Args:
        cycle_config: Cycle configuration dictionary.
        general_config: General runtime configuration.
        verbose: Whether to log rejection reasons.

    Returns:
        Best feasible cycle-state dictionary.
    """
    calculation_method = general_config.get("calculation_method", "full_optimization")
    if calculation_method == "three_temp_fix":
        # imported here, since thermodynamics_three_temp_fix imports the property functions of this module
        from CHPC.thermodynamics_three_temp_fix import solve_cycle_three_temp_fix
        return solve_cycle_three_temp_fix(cycle_config, general_config, verbose=verbose)
    if calculation_method == "fixed_PR":
        return _optimize_cycle(cycle_config, general_config, fixed_PR=float(cycle_config["PR"]), verbose=verbose)
    return _optimize_cycle(cycle_config, general_config, fixed_PR=None, verbose=verbose)

# Performance metrics
def _cop_turb(cycle_data, cycle_config):
    """Compute the COP of the cycle including the turbine expansion, the objective of the optimization.

    Args:
        cycle_data: Cycle-state dictionary.
        cycle_config: Cycle configuration dictionary.

    Returns:
        COP_turb.
    """
    s = cycle_data
    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out = cycle_config["ṁ_h"] * cycle_config["cp_h"] * (s["T_h_out"] - cycle_config["T_h_in"])
    return Q_out / (Ẇ_comp - Ẇ_turb * cycle_config.get("η_shaft", 1.0))

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
    η_shaft = cycle_config.get("η_shaft", 1.0)
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
    COP_turb = _cop_turb(cycle_data, cycle_config)
    h_ref_4_is = _cp_props("H", "P", s["p_ref_4"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
    Ẇ_turb_is = s["ṁ_ref"] * (s["h_ref_3"] - h_ref_4_is)
    COP_is = Q_out / (Ẇ_comp - Ẇ_turb_is * η_shaft)
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

# Diagram data preparation
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
