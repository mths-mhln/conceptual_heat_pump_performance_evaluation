import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/')
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/verification/')

import math
import itertools
import numpy as np
from CoolProp.CoolProp import PropsSI
from logger import setup_logger
import matplotlib.pyplot as plt

logger = setup_logger()



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
            T_range[i] = PropsSI("T", "P", p, "S", s, f"REFPROP::{refrigerant}")
        except ValueError:
            pass
    return s_range.tolist(), T_range.tolist()


def _isobar_h_segment(h_start, h_end, p):
    """Return (h_range, p_range) along a constant-pressure path (PH diagram)."""
    h_range = np.linspace(h_start, h_end, num=30)
    p_range = np.full(30, p)
    return h_range.tolist(), p_range.tolist()


def _specific_heat_from_isobar_path(
    T_start,
    T_end,
    p,
    general_config,
    cycle_config,
    supercritical_cycle=False,
    uniform_sampling=False,
):
    """Return specific heat transfer [J/kg] along an isobaric path from T_start to T_end.

    Uses enthalpy-based arc-length uniform isobar sampling (via
    _sample_isobar_ts_uniform_arc) when supercritical_cycle or uniform_sampling
    is True. The enthalpy-based approach gives better point distribution near the
    critical region where T is nearly flat as a function of H. Falls back to plain
    linspace temperature sampling for standard subcritical paths.
    """
    num_points = 150 if general_config["resolution"] == "low" else 1000

    if supercritical_cycle or uniform_sampling:
        # Derive enthalpy bounds from the temperature endpoints, then delegate to
        # the enthalpy-based sampler.  The returned T_range is used directly for
        # the cp integration so no extra PropsSI calls are needed here.
        h_start = PropsSI("H", "P", p, "T", T_start, f"REFPROP::{cycle_config['refrigerant']}")
        h_end   = PropsSI("H", "P", p, "T", T_end,   f"REFPROP::{cycle_config['refrigerant']}")
        _, T_range, _ = _sample_isobar_ts_uniform_arc(
            h_start,
            h_end,
            p,
            cycle_config,
            num_points=num_points,
        )
    else:
        T_range = np.linspace(T_start, T_end, num=num_points)

    if len(T_range) < 2:
        return 0

    heat = 0
    for T_1, T_2 in zip(T_range[:-1], T_range[1:]):
        try:
            cp = PropsSI("C", "P", p, "T", T_1, f"REFPROP::{cycle_config['refrigerant']}")
            heat += cp * np.abs(T_2 - T_1)
        except ValueError:
            pass
    return heat


def _sample_isobar_ts_uniform_arc(h_start, h_end, p, cycle_config, num_points=2000):
    """
    Sample a TS isobar uniformly in arc length between two enthalpy values.

    Enthalpy is used as the independent variable instead of temperature because
    near the critical point the isobar T(H) curve is nearly flat, meaning a
    uniform temperature grid clusters most points in a tiny region while the
    enthalpy range remains well-spread.

    Returns:
        s_uniform  : entropy array [J/(kg·K)], near-uniform arc-length spacing
        T_uniform  : temperature array [K], near-uniform arc-length spacing
        h_uniform  : enthalpy array [J/kg], near-uniform arc-length spacing
    """
    refrigerant = cycle_config["refrigerant"]
    n = max(num_points, 400)

    h_arr = np.linspace(h_start, h_end, n)
    T_arr = np.full_like(h_arr, np.nan, dtype=float)
    s_arr = np.full_like(h_arr, np.nan, dtype=float)
    for i, h in enumerate(h_arr):
        try:
            T_arr[i] = PropsSI("T", "P", p, "H", h, f"REFPROP::{refrigerant}")
            s_arr[i] = PropsSI("S", "P", p, "H", h, f"REFPROP::{refrigerant}")
        except ValueError:
            pass

    valid = np.isfinite(T_arr) & np.isfinite(s_arr) & np.isfinite(h_arr)
    if valid.sum() < 4:
        # Fallback: return the raw (nan-filtered) arrays without arc resampling.
        return s_arr[valid], T_arr[valid], h_arr[valid]

    T_valid = T_arr[valid]
    s_valid = s_arr[valid]
    h_valid = h_arr[valid]

    # Normalise all three axes before computing arc length so that no single
    # physical unit dominates the distance metric.
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
    """
    Check the sign of the second derivative d²T/ds² along a TS curve.

    Returns:
        sign: +1 if positive all over
                -1 if negative all over
                0 if it changes sign
    """
    dT_ds = np.gradient(T_arr, s_arr)
    d2T_ds2 = np.gradient(dT_ds, s_arr)
    if np.all(d2T_ds2 > 0):
        return 1
    elif np.all(d2T_ds2 < 0):
        return -1
    else:
        return 0



# Cycle solver (bisection on pressure ratio)
# ==========================================
def solve_cycle(cycle_config, general_config):
    # Extract parameters from cycle_config
    refrigerant = cycle_config["refrigerant"]
    T_h_in      = cycle_config["T_h_in"]
    T_c_in      = cycle_config["T_c_in"]
    ṁ_h         = cycle_config["ṁ_h"]
    ṁ_c         = cycle_config["ṁ_c"]
    cp_h        = cycle_config["cp_h"]
    cp_c        = cycle_config["cp_c"]
    η_compr     = cycle_config["η_compr"]
    η_turb      = cycle_config["η_turb"]
    ΔT_pp_1     = cycle_config["ΔT_pp_1"]
    ΔT_pp_2     = cycle_config["ΔT_pp_2"]
    ΔT_pp_3     = cycle_config["ΔT_pp_3"]
    ΔT_pp_4     = cycle_config["ΔT_pp_4"]
    ΔT_sh       = cycle_config["ΔT_sh"]

    # Station 1 — compressor inlet (fixed by user inputs)
    T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
    p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    p_ref_1 = p_ev

    # Pressure ratio bounds
    p_lim_lower = p_ref_1 + 1
    PR_bisection_range = [p_lim_lower / p_ref_1, 30]
    PR_guess = sum(PR_bisection_range) / 2

    ΔT_pp_4_calculated = 0
    p_ref_2_conv = 0
    state = {}

    while not math.isclose(ΔT_pp_4_calculated, ΔT_pp_4, rel_tol=1e-3):
        # Station 1
        T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
        p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
        p_ref_1 = p_ev
        T_ref_1 = T_h_in - ΔT_pp_1
        h_ref_1 = PropsSI("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
        s_ref_1 = PropsSI("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
        Q_ref_1 = _vapour_quality_scaler(PropsSI("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}"))

        # Station 2 — condenser inlet
        p_ref_2 = PR_guess * p_ref_1
        h_ref_2_is = PropsSI("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
        h_ref_2 = (h_ref_2_is - h_ref_1) / η_compr + h_ref_1
        T_ref_2 = PropsSI("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
        Q_ref_2 = _vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}"))
        s_ref_2 = PropsSI("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")

        if math.isclose(p_ref_2, p_ref_2_conv, rel_tol=1e-6):
            logger.info(
                "Convergence stagnated for ΔT_pp_4_calculated. Reason to believe heat pump cycle for specifications is impossible without"
                "the cycle occurring fully on the right side of the critical point."
            )
            sys.exit()
        p_ref_2_conv = p_ref_2

        if p_ref_2 > PropsSI("Pcrit", f"REFPROP::{refrigerant}"):
            supercritical_cycle = True
        else:
            supercritical_cycle = False

        # Station 3 — turbine inlet
        T_ref_3 = T_c_in + ΔT_pp_3
        p_ref_3 = p_ref_2
        h_ref_3 = PropsSI("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
        s_ref_3 = PropsSI("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
        Q_ref_3 = _vapour_quality_scaler(PropsSI("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}"))
        if not supercritical_cycle:
            T_cond = PropsSI("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")

        if T_ref_2 < T_ref_3 * 0.99:
            logger.info(
                f"Station 2 temperature (T_ref_2 = {T_ref_2:.2f} K) is significantly lower than station 3 "
                f"temperature (T_ref_3 = {T_ref_3:.2f} K). Adjusting pressure ratio bounds for bisection."
            )
            ΔT_pp_4_calculated = np.inf
            if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                PR_bisection_range[1] = PR_guess
            else:
                PR_bisection_range[0] = PR_guess
            PR_guess = sum(PR_bisection_range) / 2
            continue

        if Q_ref_3 == 1:
            logger.info(
                "Station 3 is saturated vapour. Adjusting pressure ratio bounds for bisection."
            )
            if supercritical_cycle:
                ΔT_pp_4_calculated = -np.inf
            if not supercritical_cycle:
                ΔT_pp_4_calculated = np.inf
            if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                PR_bisection_range[1] = PR_guess
            else:
                PR_bisection_range[0] = PR_guess
            PR_guess = sum(PR_bisection_range) / 2
            continue

        # Sample isobar between station 3 and 2 using enthalpy as independent variable.
        s_ref_arr, T_ref_arr, _ = _sample_isobar_ts_uniform_arc(
            h_ref_3,
            h_ref_2,
            p_ref_2,
            cycle_config,
            num_points=150,
        )
        sign = _check_second_derivative_sign(s_ref_arr, T_ref_arr)
        if sign > 0:
            logger.info(
                "Second derivative d²T/ds² along the isobar between station 3 and 2 is positive. "
                "Adjusting pressure ratio bounds for bisection."
            )
            ΔT_pp_4_calculated = -np.inf
            if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                PR_bisection_range[1] = PR_guess
            else:
                PR_bisection_range[0] = PR_guess
            PR_guess = sum(PR_bisection_range) / 2
            continue

        # Station 4 — evaporator inlet
        p_ref_4 = p_ref_1
        h_ref_4_is = PropsSI("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
        h_ref_4 = (h_ref_4_is - h_ref_3) * η_turb + h_ref_3
        T_ref_4 = PropsSI("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
        Q_ref_4 = _vapour_quality_scaler(PropsSI("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}"))
        Q_ref_4_isenth = _vapour_quality_scaler(PropsSI("Q", "P", p_ref_4, "H", h_ref_3, f"REFPROP::{refrigerant}"))
        s_ref_4 = PropsSI("S", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")

        if Q_ref_2 < 0:
            s_arr, T_arr = _isobar_segment(s_ref_3, s_ref_2, p_ref_2, cycle_config, general_config)
            sign = _check_second_derivative_sign(s_arr, T_arr)
            if sign >= 0:
                logger.info(
                    "Second derivative d²T/ds² along the isobar between station 3 and 2 is non-negative. "
                    "Adjusting pressure ratio bounds for bisection."
                )
                ΔT_pp_4_calculated = -np.inf
                if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                    PR_bisection_range[1] = PR_guess
                else:
                    PR_bisection_range[0] = PR_guess
                PR_guess = sum(PR_bisection_range) / 2
                continue
            else:
                supercritical_cycle = True

        # Latent heats
        if not supercritical_cycle:
            Δh_cond = (PropsSI("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") -
                       PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}"))
        Δh_ev = (PropsSI("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}") -
                 PropsSI("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}"))

        if supercritical_cycle:
            ṁ_ref_bisection_range = [1e-5, 100]
            ṁ_ref_guess = sum(ṁ_ref_bisection_range) / 2
            ṁ_ref_conv = 0
            ΔT_pp_2_calculated = 0
            stagnation = False
            while not math.isclose(ΔT_pp_2_calculated, ΔT_pp_2, rel_tol=1e-6):
                if math.isclose(ṁ_ref_guess, ṁ_ref_conv, rel_tol=1e-8):
                    logger.info(
                        "Convergence stagnated for ΔT_pp_2_calculated. "
                        "Increasing PR to push for a viable supercritical isobar."
                    )
                    stagnation = True
                    break
                ṁ_ref_conv = ṁ_ref_guess

                # Sample isobar between station 3 and 2 using enthalpy as independent variable.
                s_ref_arr, T_ref_arr, h_ref_arr = _sample_isobar_ts_uniform_arc(
                    h_ref_3,
                    h_ref_2,
                    p_ref_2,
                    cycle_config,
                    num_points=80,
                )
                T_c_arr = np.zeros_like(T_ref_arr)
                for i, (T_ref, s_ref, h_ref) in enumerate(zip(T_ref_arr, s_ref_arr, h_ref_arr)):
                    heat_transferred = 0
                    Q_ref = _vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "H", h_ref, f"REFPROP::{refrigerant}"))
                    if not supercritical_cycle:
                        if Q_ref < 0:
                            heat_transferred += _specific_heat_from_isobar_path(T_ref_3, T_ref, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref_guess
                        if Q_ref > 0 and Q_ref < 1:
                            Δh = PropsSI("H", "P", p_ref_2, "Q", Q_ref, f"REFPROP::{refrigerant}") - PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")
                            heat_transferred += Δh * ṁ_ref_guess
                        if Q_ref > 1:
                            heat_transferred += _specific_heat_from_isobar_path(T_ref, T_cond, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref_guess
                    if supercritical_cycle:
                        # Use the enthalpy difference directly — more accurate than cp·ΔT integration
                        # in the supercritical region where cp peaks sharply near the pseudocritical point.
                        heat_transferred += (h_ref - h_ref_3) * ṁ_ref_guess
                    T_c_arr[i] = T_c_in + heat_transferred / (ṁ_c * cp_c)
                T_diff_arr = T_ref_arr - T_c_arr
                negative_slope_points = np.where(np.diff(T_diff_arr) < 0)[0]
                if len(negative_slope_points) == 0:
                    logger.info(
                        f"No negative slope points found for ṁ_ref_guess = {ṁ_ref_guess:.6f} kg/s. "
                        "Adjusting mass flow rate bounds for bisection."
                    )
                    ΔT_pp_2_calculated = np.inf
                    if (ΔT_pp_2_calculated - ΔT_pp_2) < 0:
                        ṁ_ref_bisection_range[1] = ṁ_ref_guess
                    else:
                        ṁ_ref_bisection_range[0] = ṁ_ref_guess
                    ṁ_ref_guess = sum(ṁ_ref_bisection_range) / 2
                    continue

                closest_point_index = negative_slope_points[np.argmin(T_diff_arr[negative_slope_points])]
                T_c_pp_2_calculated = T_c_arr[closest_point_index]
                ΔT_pp_2_calculated = T_ref_arr[closest_point_index] - T_c_pp_2_calculated

                if ΔT_pp_2_calculated < ΔT_pp_2:
                    ṁ_ref_bisection_range[1] = ṁ_ref_guess
                else:
                    ṁ_ref_bisection_range[0] = ṁ_ref_guess
                ṁ_ref_guess = sum(ṁ_ref_bisection_range) / 2
            ṁ_ref = ṁ_ref_guess
            T_c_pp_2 = T_c_pp_2_calculated
            if stagnation:
                logger.info(
                    "Stagnation occurred during pinch point 2 mass flow rate bisection. "
                    "Adjusting pressure ratio bounds for bisection."
                )
                ΔT_pp_4_calculated = np.inf
                if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                    PR_bisection_range[1] = PR_guess
                else:
                    PR_bisection_range[0] = PR_guess
                PR_guess = sum(PR_bisection_range) / 2
                continue

        if not supercritical_cycle:
            T_c_pp_2 = T_cond - ΔT_pp_2
            ṁ_ref = ((T_c_pp_2 - T_c_in) * ṁ_c * cp_c /
                    (Δh_cond * (Q_ref_2 - Q_ref_3) + _specific_heat_from_isobar_path(T_cond, T_ref_3, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle)))

        # Outlet temperatures
        if supercritical_cycle:
            T_c_out = T_c_in + _specific_heat_from_isobar_path(T_ref_3, T_ref_2, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref / (ṁ_c * cp_c)
        if not supercritical_cycle:
            T_c_out = T_c_in + (
                _specific_heat_from_isobar_path(T_ref_2, T_cond, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref
                + (Q_ref_2 - Q_ref_3) * Δh_cond * ṁ_ref
                + _specific_heat_from_isobar_path(T_cond, T_ref_3, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref
            ) / (ṁ_c * cp_c)

        T_h_out = T_h_in - (
            _specific_heat_from_isobar_path(T_ref_1, T_ev, p_ref_1, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref
            + (Q_ref_1 - Q_ref_4) * Δh_ev * ṁ_ref
            + _specific_heat_from_isobar_path(T_ev, T_ref_4, p_ref_1, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref
        ) / (ṁ_h * cp_h)

        T_h_pp_4 = _specific_heat_from_isobar_path(T_ref_4, T_ev, p_ref_1, general_config, cycle_config, supercritical_cycle=supercritical_cycle) / (ṁ_h * cp_h) + T_h_out
        ΔT_pp_4_calculated = T_h_pp_4 - T_ev
        if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
            PR_bisection_range[1] = PR_guess
        else:
            PR_bisection_range[0] = PR_guess
        PR_guess = sum(PR_bisection_range) / 2

    # Cache converged state
    if supercritical_cycle:
        state = dict(
            p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_3, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3, Q_ref_3=Q_ref_3,
            p_ref_4=p_ref_4, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4,
            T_ev=T_ev, Δh_ev=Δh_ev, T_c_out=T_c_out, T_h_out=T_h_out,
            Q_ref_4_isenth=Q_ref_4_isenth, ṁ_ref=ṁ_ref, T_c_pp_2=T_c_pp_2, supercritical_cycle=supercritical_cycle
        )
    if not supercritical_cycle:
        state = dict(
            p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_3, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3, Q_ref_3=Q_ref_3,
            p_ref_4=p_ref_4, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4,
            T_cond=T_cond, T_ev=T_ev, Δh_cond=Δh_cond, Δh_ev=Δh_ev, T_c_out=T_c_out, T_h_out=T_h_out,
            Q_ref_4_isenth=Q_ref_4_isenth, ṁ_ref=ṁ_ref, T_c_pp_2=T_c_pp_2, supercritical_cycle=supercritical_cycle
        )
    return state





def evaluate_subcrtical_cycle_pp(cycle_config, general_config, PR_guess):
    """
    evaluates the subcritical thermodynamic cycle which is fully constrained by the pinch point specifications. 
    """












def evaluate_supercritical_cycle_pp(cycle_config, general_config, PR_guess):
    """
    evaluates the supercritical thermodynamic cycle which is fully constrained by the pinch point specifications. 
    """





def evaluate_supercritical_cycle_PR(cycle_config, general_config):
    """
    evaluates the supercritical thermodynamic cycle which is fully constrained by the three pp and a pressure ratio specification. 
    """











# Performance metrics
# ===================
def compute_performance(state, cycle_config, general_config):
    s = state
    ṁ_c = cycle_config["ṁ_c"]
    cp_c = cycle_config["cp_c"]
    T_c_in = cycle_config["T_c_in"]
    ɳ_shaft = cycle_config["ɳ_shaft"]
    refrigerant = cycle_config["refrigerant"]

    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out   = ṁ_c * cp_c * (s["T_c_out"] - T_c_in)
    Q_in    = s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
              s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4"]) + \
              s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ev"], s["T_ref_4"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])

    Q_in_isenth = s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4_isenth"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_4"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])

    COP_turb = Q_out / (Ẇ_comp - Ẇ_turb * ɳ_shaft)

    h_ref_4_is = PropsSI("H", "P", s["p_ref_4"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
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
def build_ts_data(state, cycle_config, general_config):
    s = state
    refrigerant = cycle_config["refrigerant"]
    T_c_in = cycle_config["T_c_in"]
    T_h_in = cycle_config["T_h_in"]

    p1, p2 = s["p_ref_1"], s["p_ref_2"]

    if not s["supercritical_cycle"]:
        s_ref_23_v_inflection = PropsSI("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        s_ref_23_l_inflection = PropsSI("S", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    s_ref_41_v_inflection = PropsSI("S", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_41_l_inflection = PropsSI("S", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

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

    s_ref_23_v_inflection = PropsSI("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
    if s["Q_ref_2"] >= 1:
        s_pp_anchor = s_ref_23_v_inflection
        s_c_out = s["s_ref_3"] + (s_pp_anchor - s["s_ref_3"]) / (s["T_c_pp_2"] - T_c_in) * (s["T_c_out"] - T_c_in)
    else:
        s_c_out = s["s_ref_2"]

    return dict(
        major={"s": [s["s_ref_1"], s["s_ref_2"], s["s_ref_3"], s["s_ref_4"], s["s_ref_1"]],
               "T": [s["T_ref_1"], s["T_ref_2"], s["T_ref_3"], s["T_ref_4"], s["T_ref_1"]]},
        minor={"s": s_ref_lst, "T": T_ref_lst},
        coolant={"s": [s["s_ref_3"], s_c_out], "T": [T_c_in, s["T_c_out"]]},
        heating={"s": [s["s_ref_1"], s["s_ref_4"]], "T": [T_h_in, s["T_h_out"]]},
    )


def build_ph_data(state, cycle_config):
    s = state
    refrigerant = cycle_config["refrigerant"]
    p1, p2 = s["p_ref_1"], s["p_ref_2"]

    if not s["supercritical_cycle"]:
        h_ref_23_v_inflection = PropsSI("H", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        h_ref_23_l_inflection = PropsSI("H", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    h_ref_41_v_inflection = PropsSI("H", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_41_l_inflection = PropsSI("H", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

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