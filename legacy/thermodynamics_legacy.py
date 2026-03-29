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

    If uniform_sampling is True, uses arc-length uniform TS sampling via
    _sample_isobar_ts_uniform_arc for better point distribution near critical
    regions. Otherwise falls back to plain linspace temperature sampling.
    Note: supercritical_cycle=True implicitly enables uniform sampling.
    """
    num_points = 150 if general_config["resolution"] == "low" else 1000

    if supercritical_cycle or uniform_sampling:
        # Arc-length uniform TS sampling gives better point distribution,
        # especially near the critical point where the isobar slope can be near-horizontal.
        _, T_range = _sample_isobar_ts_uniform_arc(
            T_start,
            T_end,
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


def _sample_isobar_ts_uniform_arc(T_start, T_end, p, cycle_config, num_points=2000):
    """
    Sample a TS isobar uniformly in arc length between two temperatures.
    Very similar purpose as _sample_isobar, however this time it does so more uniformally
    which can matter a lot for the condensing path for supercritical cycles since the slope 
    can become near horizontal. 

    Returns:
        s_uniform, T_uniform : arrays with near-uniform spacing along the TS curve.
    """
    refrigerant = cycle_config["refrigerant"]
    n = max(num_points, 400)

    T_arr = np.linspace(T_start, T_end, n)
    s_arr = np.full_like(T_arr, np.nan, dtype=float)
    for i, T_ref in enumerate(T_arr):
        try:
            s_arr[i] = PropsSI("S", "P", p, "T", T_ref, f"REFPROP::{refrigerant}")
        except ValueError:
            pass

    valid = np.isfinite(T_arr) & np.isfinite(s_arr)
    if valid.sum() < 4:
        # Fallback to plain temperature sampling if arc-length reconstruction fails.
        T_uniform = np.linspace(T_start, T_end, num_points)
        s_uniform = np.full_like(T_uniform, np.nan, dtype=float)
        for i, T_ref in enumerate(T_uniform):
            try:
                s_uniform[i] = PropsSI("S", "P", p, "T", T_ref, f"REFPROP::{refrigerant}")
            except ValueError:
                pass
        return s_uniform, T_uniform

    T_valid = T_arr[valid]
    s_valid = s_arr[valid]

    # Normalise axes to avoid units dominating arc-length calculation.
    T_scale = max(np.max(T_valid) - np.min(T_valid), 1e-12)
    s_scale = max(np.max(s_valid) - np.min(s_valid), 1e-12)
    T_norm = (T_valid - np.min(T_valid)) / T_scale
    s_norm = (s_valid - np.min(s_valid)) / s_scale

    seg_len = np.sqrt(np.diff(T_norm) ** 2 + np.diff(s_norm) ** 2)
    arc = np.concatenate(([0.0], np.cumsum(seg_len)))
    if arc[-1] <= 0:
        return s_valid, T_valid

    arc_uniform = np.linspace(0.0, arc[-1], num_points)
    T_uniform = np.interp(arc_uniform, arc, T_valid)
    s_uniform = np.interp(arc_uniform, arc, s_valid)
    return s_uniform, T_uniform


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
    p_lim_lower = p_ref_1 + 1 # semi arbitrary, but not equal to p_ref_1 as that may give issues
    PR_bisection_range = [p_lim_lower / p_ref_1, 30]  # arbitrary upper bound, I have yet to see compressors achieving such numbers, but I am inexperienced so who knows...
    # arbitrary upper bound due to absent limit for supercritical cycle other than what is reasonable for compressor design. 
    PR_guess = sum(PR_bisection_range) / 2

    ΔT_pp_4_calculated = 0
    p_ref_2_conv = 0
    state = {}

    while not math.isclose(ΔT_pp_4_calculated, ΔT_pp_4, rel_tol=1e-3):
        # Note: continue statement is used in the loop!
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

        # break cycle upon convergence of pp 4 without exiting the loop
        if math.isclose(p_ref_2, p_ref_2_conv, rel_tol=1e-6):
            logger.critical("Convergence stagnated for ΔT_pp_4_calculated. Reason to believe heat pump cycle for specifications is impossible without" \
            "the cycle occuring fully on the right side of the critical point.")
            sys.exit()
        p_ref_2_conv = p_ref_2

        # check supercriticality or for a compression process that ends in a p_ref_2 which is fully in the liquid domain
        # the treatment of the refrigerant cooling will be the same. 
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

        if T_ref_2 < T_ref_3*0.99: # allow some numarical error margin, maybe they are both in the two-phase dome
            logger.warning(f"Station 2 temperature (T_ref_2 = {T_ref_2:.2f} K) is significantly lower than station 3 temperature (T_ref_3 = {T_ref_3:.2f} K). This is an unusual cycle configuration and may indicate that the pressure ratio guess is too low. Adjusting pressure ratio bounds for bisection.")
            # Bisection update
            # rationale: if the calculated T_ref_2 is lower then T_ref_3, there is no condensation path for which a T_ref_3 on the left
            # of the critical point, the pressure ratio must be increased for ref_2 to achieve higher temperatures. Remember that the T_ref3
            # is decided through the pinch point. 
            ΔT_pp_4_calculated = np.inf
            if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                PR_bisection_range[1] = PR_guess
            else:
                PR_bisection_range[0] = PR_guess
            PR_guess = sum(PR_bisection_range) / 2
            continue

        if Q_ref_3 == 1:
            logger.warning(f"Station 3 is saturated vapour. This is an unusual cycle configuration and may indicate that the pressure ratio guess is too low. Adjusting pressure ratio bounds for bisection.")
            # Bisection update
            # some edge case occured where this converged, this is not a reasonable cycle and carries very low efficiency. Higher pressure ratio is required
            # however, for supercritical, the other edge case is that the PR is just so large that along the isobar the only ref_3 possible is a Q>1 one, in that
            # case the pressure ratio should be lowered. 
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

        s_ref_arr, T_ref_arr = _sample_isobar_ts_uniform_arc(
            T_ref_3,
            T_ref_2,
            p_ref_2,
            cycle_config,
            num_points=80,
        )
        print(T_ref_arr)
        print(s_ref_arr)
        print(f"Station 3: T_ref_3 = {T_ref_3:.2f} K, Q_ref_3 = {Q_ref_3:.4f}, s_ref_3 = {s_ref_3:.4f}")
        print(f"Station 2: T_ref_2 = {T_ref_2:.2f} K, Q_ref_2 = {Q_ref_2:.4f}, s_ref_2 = {s_ref_2:.4f}")
        sign = _check_second_derivative_sign(s_ref_arr, T_ref_arr)
        if sign > 0:
            logger.warning(f"Second derivative d²T/ds² along the isobar between station 3 and 2 is positive. This indicates that there is only one pinch point along this path, which is an unusual cycle configuration and may indicate that the pressure ratio guess is too low. Adjusting pressure ratio bounds for bisection.")
            # Bisection update
            # rationale: if the second derivative is positive, the isobar is one corresponding to very high pressure, hence the 
            # PR should be lowered
            # It is true that the second derivative of the difference between the isobar and the cooling curve is what matters, but if the isobar itself 
                # has a positive second derivative, it is unlikely that the difference has a negative one. 
                # Hence I will not allow for dry-wet compression cycles if they do not promise two pinch points
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

        # Check if we have dry to wet expansion with only a single pinch point (see elaborate explanation a couple lines down)
        if Q_ref_2 < 0:
            s_arr, T_arr = _isobar_segment(s_ref_3, s_ref_2, p_ref_2, cycle_config, general_config)
            sign = _check_second_derivative_sign(s_arr, T_arr)
            if sign >= 0:
                logger.warning(f"Second derivative d²T/ds² along the isobar between station 3 and 2 is non-negative. This indicates that there is at most one pinch point along this path, which is an unusual cycle configuration for a dry-wet compression and may indicate that the pressure ratio guess is too low. Adjusting pressure ratio bounds for bisection.")
                # Bisection update 
                # rationale: The cycle is a dry-wet compression. This is fine, but likely yields an isobar segment with positive second derivative
                # It is true that the second derivative of the difference between the isobar and the cooling curve is what matters, but if the isobar itself 
                # has a positive second derivative, it is unlikely that the difference has a negative one. 
                # Hence I will not allow for dry-wet compression cycles if they do not promise two pinch points
                # the pressure ratio should be lowered
                ΔT_pp_4_calculated = -np.inf
                if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
                    PR_bisection_range[1] = PR_guess
                else:
                    PR_bisection_range[0] = PR_guess
                PR_guess = sum(PR_bisection_range) / 2
                continue
            else:
                # use same routine as supercritical cycle
                supercritical_cycle = True
    
        # Latent heats
        if not supercritical_cycle:
            Δh_cond = (PropsSI("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") -
                       PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}"))
        Δh_ev   = (PropsSI("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}") -
                   PropsSI("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}"))

        # Pinch-point 2 heat balance → ṁ_ref
        # For non-supercritical cycle, one can assume that the superheating near ref_2 has a slope large enough for this method to make sense
        # T_c_pp_2 = T_cond - ΔT_pp_2
        # ṁ_ref = ((T_c_pp_2 - T_c_in) * ṁ_c * cp_c /
        #         (Δh_cond * (Q_ref_2 - Q_ref_3) + _specific_heat_from_isobar_path(T_cond, T_ref_3, p_ref_2, general_config, cycle_config)))
        # however there is a more universal, albeit more computationally intensive, iterative method. 
        # for supercritical cycles, specifying the pressure ratio makes more sense, but I will code this nonetheless
        if supercritical_cycle:
            # In this case there are some particularities regarding the pinch point calculation. Noting that this routine also should evaluate 
            # refrigerant cooling for vapour-liquid compressions. 3 scenario's can occur: the second derivative of the difference between the isobar
            # and the cooling curve can be positive, negative or change sign. If it is positive, there is only one pinch point along the entirety of the 
            # curve, and our classical calculation routine fails. That is, T_ref_3 cannot simply be calculated from the pinch point requirement. There are, 
            # however, an infinite amount refrigerant curves (the cooling curve is fixed at T_c_i so its more constrained) for which the single pinch point can
            # meet the requirement, hence there is a choice to the user. My choice, will be to pick the refrigerant and cooling curve pair for which the
            # pinch point criterion (an avg of the ones prescribed for 2 and 3) is met, and the differences between ref and coolant inlet and exit temperatures
            # are equal. This is somewhat arbitrary, but it is a choice nonetheless. If the second derivative is negative, there are two pinch points, and the 
            # classical calculation routine holds. The same story holds for the case where the second derivative changes sign. I actually realize that in the 
            # first case (positive second derivative) I can change the cycle requirements, but that means that I get a cycle not according to specs, since I 
            # asked fro 2 pp. THe cycle would be closed and the pressure iteration stops, but that is becasue I changed the requirements mid-run. 
            # So instead I should, upon realizing the compression goes from dry-wet (which not necessarily goes to a supercritical p), I should pass
            # ΔT_pp_4_calculated = infinity and let the iterator reduce the pressure ratio. In hindsight, I will allow wet-dry expansions only if they promise
            # two pinch points. 
            
            ṁ_ref_bisection_range = [1e-5, 100]  # kg/s [lower_bound, upper_bound] for mass flow rate through the cycle
            ṁ_ref_guess = sum(ṁ_ref_bisection_range) / 2
            ṁ_ref_conv = 0
            ΔT_pp_2_calculated = 0
            stagnation = False
            while not math.isclose(ΔT_pp_2_calculated, ΔT_pp_2, rel_tol=1e-6):
                if math.isclose(ṁ_ref_guess, ṁ_ref_conv, rel_tol=1e-8):
                    logger.critical("Convergence stagnated for ΔT_pp_2_calculated. Reason to believe the pp requirement is not possible for the current isobar. We would like to have a supercritical cycle, increasing PR")
                    stagnation = True
                    break
                ṁ_ref_conv = ṁ_ref_guess
                # Arc-length TS resampling on the p_ref_2 isobar gives more uniform
                # point distribution than plain uniform T sampling, while remaining
                # robust near p_ref_2 ~ p_crit.
                s_ref_arr, T_ref_arr = _sample_isobar_ts_uniform_arc(
                    T_ref_3,
                    T_ref_2,
                    p_ref_2,
                    cycle_config,
                    num_points=80,
                )
                T_c_arr = np.zeros_like(T_ref_arr)
                for i, (T_ref, s_ref) in enumerate(zip(T_ref_arr, s_ref_arr)):
                    heat_transferred = 0
                    Q_ref = _vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "S", s_ref, f"REFPROP::{refrigerant}"))
                    if not supercritical_cycle:
                        if Q_ref < 0:
                            heat_transferred += _specific_heat_from_isobar_path(T_ref_3, T_ref, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref_guess
                        if Q_ref > 0 and Q_ref < 1:
                            Δh = PropsSI("H", "P", p_ref_2, "Q", Q_ref, f"REFPROP::{refrigerant}") - PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")
                            heat_transferred += Δh * ṁ_ref_guess
                        if Q_ref > 1:
                            heat_transferred += _specific_heat_from_isobar_path(T_ref, T_cond, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref_guess
                    if supercritical_cycle:
                        heat_transferred += _specific_heat_from_isobar_path(T_ref_3, T_ref, p_ref_2, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref_guess
                    T_c_arr[i] = T_c_in + heat_transferred / (ṁ_c * cp_c)
                T_diff_arr = T_ref_arr - T_c_arr
                negative_slope_points = np.where(np.diff(T_diff_arr) < 0)[0]
                # Bisection update
                # Rationale:if there are no negative slope points, the mass flow rate is simply too low for the difference between the two to lead to 
                # negative slope points, hence we need higher mass flow rate
                # plot the arrays on a TS diagram for debugging using the visualization utilities
                # plt.plot(s_ref_arr, T_ref_arr)
                # plt.scatter(s_ref_arr[negative_slope_points], T_ref_arr[negative_slope_points], color='red', label='Negative slope points')
                # plt.plot(s_ref_arr, T_c_arr, label='Coolant temperature')
                # plt.show()
                if len(negative_slope_points) == 0:
                    logger.warning(f"No negative slope points found for ṁ_ref_guess = {ṁ_ref_guess:.6f} kg/s. This indicates that the mass flow rate guess is too low to achieve the required ΔT_pp_2. Adjusting mass flow rate bounds for bisection.")
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
                logger.info(f"ṁ_ref_guess: {ṁ_ref_guess:.6f} kg/s, ΔT_pp_2_calculated: {ΔT_pp_2_calculated:.4f} K, supercritical_cycle: {supercritical_cycle}")
                if ΔT_pp_2_calculated < ΔT_pp_2:
                    ṁ_ref_bisection_range[1] = ṁ_ref_guess
                else:                
                    ṁ_ref_bisection_range[0] = ṁ_ref_guess
                ṁ_ref_guess = sum(ṁ_ref_bisection_range) / 2
            ṁ_ref = ṁ_ref_guess
            T_c_pp_2 = T_c_pp_2_calculated
            if stagnation:
                logger.warning("Stagnation occurred during pinch point 2 mass flow rate bisection. This likely indicates that the pinch point requirements are not achievable for the current isobar, which may be due to the cycle being supercritical with a pressure ratio that is too low. Adjusting pressure ratio bounds for bisection.")
                # Bisection update
                # Rationale: the pressure ratio could also be reduced to push for a subcritical cycle, but we would like to keep a supercritical one for CO2, hence pushing for higher pressure ratio. 
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

        # evaluate heating stream at pinch point 4
        T_h_pp_4 = _specific_heat_from_isobar_path(T_ref_4, T_ev, p_ref_1, general_config, cycle_config, supercritical_cycle=supercritical_cycle) * ṁ_ref / (ṁ_h * cp_h) + T_h_out
        ΔT_pp_4_calculated = T_h_pp_4 - T_ev
        # Bisection update
        # Reasoning: For the heating stream, the fixed value is the inlet temperature, depending on the refrigerant mass flow rate of the refrigerant, the 
        # outlet temperature necessary to achieve the necessary heat transfer will be decided. The larger the refrigerant mass flow rate, the lower
        # the outlet temperature. high mass flow rate results from the pinch point requirements during the condensation step. The necessary mass flow rate 
        # to achieve the pinch point requirements at pp2 and pp3 will be higher for higher pressure ratio's. This is because the larger the PR, the lower
        # the specific latent heat exchange, hence the larger the mass flow rate must be to increase the temperature of the cooling flow enough for the pinch 
        # point requirements to be met. The inverse occurs for lower pressure ratio's
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
            Q_ref_4_isenth=Q_ref_4_isenth, ṁ_ref=ṁ_ref, T_c_pp_2 = T_c_pp_2, supercritical_cycle=supercritical_cycle
        )
    if not supercritical_cycle:
        state = dict(
            p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_3, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3, Q_ref_3=Q_ref_3,
            p_ref_4=p_ref_4, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4,
            T_cond=T_cond, T_ev=T_ev, Δh_cond=Δh_cond, Δh_ev=Δh_ev, T_c_out=T_c_out, T_h_out=T_h_out, 
            Q_ref_4_isenth=Q_ref_4_isenth, ṁ_ref=ṁ_ref, T_c_pp_2 = T_c_pp_2, supercritical_cycle=supercritical_cycle
        )
    return state



# Performance metrics
# ===================
def compute_performance(state, cycle_config, general_config):
    s = state
    ṁ_c = cycle_config["ṁ_c"]
    cp_c = cycle_config["cp_c"]
    T_c_in = cycle_config["T_c_in"]
    ɳ_shaft = cycle_config["ɳ_shaft"]
    refrigerant = cycle_config["refrigerant"]
    # ṁ_h = cycle_config["ṁ_h"]
    # cp_h = cycle_config["cp_h"]
    # T_h_in = cycle_config["T_h_in"]

    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out   = ṁ_c * cp_c * (s["T_c_out"] - T_c_in)
    Q_in    = s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
              s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4"]) + \
              s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ev"], s["T_ref_4"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])
    # Q_in    = ṁ_h * cp_h * (T_h_in - s["T_h_out"])
    
    # isenthalpic Q_in
    Q_in_isenth = s["ṁ_ref"] * s["Δh_ev"] * (s["Q_ref_1"] - s["Q_ref_4_isenth"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_1"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"]) + \
                  s["ṁ_ref"] * _specific_heat_from_isobar_path(s["T_ref_4"], s["T_ev"], s["p_ref_1"], general_config, cycle_config, supercritical_cycle=s["supercritical_cycle"])

    # COP with actual turbine expansion
    COP_turb = Q_out / (Ẇ_comp - Ẇ_turb * ɳ_shaft)

    # Isentropic expansion reference (ideal turbine, η_turb = 1)
    h_ref_4_is = PropsSI("H", "P", s["p_ref_4"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
    Ẇ_turb_is  = s["ṁ_ref"] * (s["h_ref_3"] - h_ref_4_is)
    COP_is     = Q_out / (Ẇ_comp - Ẇ_turb_is * ɳ_shaft)

    # Isenthalpic expansion reference (throttle valve, no work recovery)
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

    # Saturation entropy values at both pressures
    if not s["supercritical_cycle"]:
        s_ref_23_v_inflection = PropsSI("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        s_ref_23_l_inflection = PropsSI("S", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    s_ref_41_v_inflection = PropsSI("S", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_41_l_inflection = PropsSI("S", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

    # Condenser path 2 -> 3:
    if s["supercritical_cycle"]:
        # Supercritical: no phase transition, just a simple isobar segment
        seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
        s_23_chain, T_23_chain = seg_s, seg_T
    else:
        # Non-supercritical: handle phase transitions with inflection points
        # Only insert the vapour-dome inflection if point 2 is superheated (s2 > s_dew).
        # Only insert the liquid inflection if point 3 is subcooled (s3 < s_bubble).
        s_23_chain, T_23_chain = [], []
        if s["s_ref_2"] > s_ref_23_v_inflection:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s_ref_23_v_inflection, p2, cycle_config, general_config)
            s_23_chain += seg_s + [s_ref_23_v_inflection]
            T_23_chain += seg_T + [s["T_cond"]]
        if s["s_ref_3"] < s_ref_23_l_inflection:
            seg_s, seg_T = _isobar_segment(s_ref_23_l_inflection, s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain += [s_ref_23_l_inflection] + seg_s
            T_23_chain += [s["T_cond"]] + seg_T
        # If neither branch triggered, the whole 2->3 leg is inside the dome
        if not s_23_chain:
            seg_s, seg_T = _isobar_segment(s["s_ref_2"], s["s_ref_3"], p2, cycle_config, general_config)
            s_23_chain, T_23_chain = seg_s, seg_T

    # Evaporator path 4 -> 1:
    # Only insert the liquid inflection if point 4 is subcooled (s4 < s_bubble).
    # Only insert the vapour inflection if point 1 is superheated (s1 > s_dew).
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

    # Coolant flow visual anchor:
    # The pinch point (pp2) sits at the dew point on the refrigerant side.
    # If point 2 is still inside the dome (Q_ref_2 < 1), the dew point is never
    # reached during condensation, so use s_ref_2 directly as the anchor — no
    # extrapolation needed.  If point 2 is superheated, the pp is at the dew
    # point inflection (s_ref_23_v_inflection), and we extrapolate from there to
    # represent the additional heat given up in the superheated section.
    s_ref_23_v_inflection = PropsSI("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
    if s["Q_ref_2"] >= 1:
        # Point 2 is superheated: pp is at dew point, extrapolate s_c_out
        s_pp_anchor = s_ref_23_v_inflection
        s_c_out = s["s_ref_3"] + (s_pp_anchor - s["s_ref_3"]) / (s["T_c_pp_2"] - T_c_in) * (s["T_c_out"] - T_c_in)
    else:
        # Point 2 is inside the dome: pp is at point 2 itself, no extrapolation
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

    # Saturation enthalpy values at both pressures
    if not s["supercritical_cycle"]:
        h_ref_23_v_inflection = PropsSI("H", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
        h_ref_23_l_inflection = PropsSI("H", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    h_ref_41_v_inflection = PropsSI("H", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_41_l_inflection = PropsSI("H", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

    # Condenser path 2 -> 3:
    if s["supercritical_cycle"]:
        # Supercritical: no phase transition, just a simple isobar segment
        seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
        h_23_chain, p_23_chain = seg_h, seg_p
    else:
        # Non-supercritical: handle phase transitions with inflection points
        # Only insert the vapour inflection if point 2 is superheated (h2 > h_dew).
        # Only insert the liquid inflection if point 3 is subcooled (h3 < h_bubble).
        h_23_chain, p_23_chain = [], []
        if s["h_ref_2"] > h_ref_23_v_inflection:
            seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], h_ref_23_v_inflection, p2)
            h_23_chain += seg_h + [h_ref_23_v_inflection]
            p_23_chain += seg_p + [p2]
        if s["h_ref_3"] < h_ref_23_l_inflection:
            seg_h, seg_p = _isobar_h_segment(h_ref_23_l_inflection, s["h_ref_3"], p2)
            h_23_chain += [h_ref_23_l_inflection] + seg_h
            p_23_chain += [p2] + seg_p
        # If neither triggered, the whole 2->3 leg is inside the dome
        if not h_23_chain:
            seg_h, seg_p = _isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
            h_23_chain, p_23_chain = seg_h, seg_p

    # Evaporator path 4 -> 1:
    # Only insert the liquid inflection if point 4 is subcooled (h4 < h_bubble).
    # Only insert the vapour inflection if point 1 is superheated (h1 > h_dew).
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