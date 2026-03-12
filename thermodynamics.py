# thermodynamics.py
# =================
# Cycle evaluation (bisection solver), performance metrics, and
# TS/PH diagram data preparation.

import math
import itertools
import numpy as np
from CoolProp.CoolProp import PropsSI

from config import (
    T_c_in, T_h_in, ṁ_c, cp_c, ṁ_h, cp_h,
    η_turb, η_compr, ΔT_pp_1, ΔT_pp_2, ΔT_pp_3, ΔT_pp_4,
    ΔT_sh, ɳ_shaft, refrigerant, resolution
)


# Helpers
# =======
def vapour_quality_scaler(Q):
    """Clamp CoolProp quality to [0, 1] for single-phase regions."""
    if Q > 1:
        return 1
    elif Q < 0:
        return 0
    return Q


def isobar_segment(s_start, s_end, p):
    """Return (s_range, T_range) along a constant-pressure path (TS diagram)."""
    num_points = 150 if resolution == "low" else 1000
    s_range = np.linspace(s_start, s_end, num=num_points)
    T_range = np.zeros(num_points)
    for i, s in enumerate(s_range):
        try:
            T_range[i] = PropsSI("T", "P", p, "S", s, f"REFPROP::{refrigerant}")
        except ValueError:
            pass
    return s_range.tolist(), T_range.tolist()


def isobar_h_segment(h_start, h_end, p):
    """Return (h_range, p_range) along a constant-pressure path (PH diagram)."""
    h_range = np.linspace(h_start, h_end, num=30)
    p_range = np.full(30, p)
    return h_range.tolist(), p_range.tolist()


# Cycle solver (bisection on pressure ratio)
# ==========================================
def solve_cycle():
    # Station 1 — compressor inlet (fixed by user inputs)
    T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
    p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
    p_ref_1 = p_ev

    # Pressure ratio bounds
    T_ref_3_fixed = T_c_in + ΔT_pp_3
    p_lim_lower = PropsSI("P", "T", T_ref_3_fixed, "Q", 0, f"REFPROP::{refrigerant}")
    p_crit = PropsSI("Pcrit", f"REFPROP::{refrigerant}")
    PR_bisection_range = [p_lim_lower / p_ref_1, p_crit / p_ref_1]
    PR_guess = sum(PR_bisection_range) / 2

    ΔT_pp_4_calculated = 0
    state = {}

    while not math.isclose(ΔT_pp_4_calculated, ΔT_pp_4, rel_tol=1e-3):
        # Station 1
        T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
        p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
        p_ref_1 = p_ev
        T_ref_1 = T_h_in - ΔT_pp_1
        h_ref_1 = PropsSI("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
        s_ref_1 = PropsSI("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
        Q_ref_1 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}"))

        # Station 2 — condenser inlet
        p_ref_2 = PR_guess * p_ref_1
        h_ref_2_is = PropsSI("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
        h_ref_2 = (h_ref_2_is - h_ref_1) / η_compr + h_ref_1
        T_ref_2 = PropsSI("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
        Q_ref_2 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}"))
        s_ref_2 = PropsSI("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")

        # Station 3 — turbine inlet
        T_ref_3 = T_c_in + ΔT_pp_3
        p_ref_3 = p_ref_2
        h_ref_3 = PropsSI("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
        s_ref_3 = PropsSI("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
        Q_ref_3 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}"))
        T_cond = PropsSI("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")

        # Station 4 — evaporator inlet
        p_ref_4 = p_ref_1
        h_ref_4_is = PropsSI("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
        h_ref_4 = (h_ref_4_is - h_ref_3) * η_turb + h_ref_3
        T_ref_4 = PropsSI("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
        Q_ref_4 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}"))
        s_ref_4 = PropsSI("S", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")

        # Specific heats
        cp_ref_1 = PropsSI("C", "P", p_ref_1, "T", T_ref_1, f"REFPROP::{refrigerant}")
        cp_ref_2 = PropsSI("C", "P", p_ref_2, "T", T_ref_2, f"REFPROP::{refrigerant}")
        cp_ref_3 = PropsSI("C", "P", p_ref_3, "T", T_ref_3, f"REFPROP::{refrigerant}")
        cp_ref_4 = PropsSI("C", "P", p_ref_4, "T", T_ref_4, f"REFPROP::{refrigerant}")

        # Latent heats
        Δh_cond = (PropsSI("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}") -
                   PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}"))
        Δh_ev   = (PropsSI("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}") -
                   PropsSI("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}"))

        # Pinch-point 2 heat balance → ṁ_ref
        T_c_pp_2 = T_cond - ΔT_pp_2
        ṁ_ref = ((T_c_pp_2 - T_c_in) * ṁ_c * cp_c /
                 (Δh_cond * (Q_ref_2 - Q_ref_3) + (T_cond - T_ref_3) * cp_ref_3))

        # Outlet temperatures
        T_c_out = T_c_in + (
            (T_ref_2 - T_cond) * ṁ_ref * cp_ref_2
            + (Q_ref_2 - Q_ref_3) * Δh_cond * ṁ_ref
            + (T_cond - T_ref_3) * cp_ref_3 * ṁ_ref
        ) / (ṁ_c * cp_c)

        T_h_out = T_h_in - (
            (T_ref_1 - T_ev) * ṁ_ref * cp_ref_1
            + (Q_ref_1 - Q_ref_4) * Δh_ev * ṁ_ref
            + (T_ref_4 - T_ev) * cp_ref_4 * ṁ_ref
        ) / (ṁ_h * cp_h)

        ΔT_pp_4_calculated = T_h_out - T_ref_4

        # Bisection update
        if (ΔT_pp_4_calculated - ΔT_pp_4) < 0:
            PR_bisection_range[1] = PR_guess
        else:
            PR_bisection_range[0] = PR_guess
        PR_guess = sum(PR_bisection_range) / 2

        # Cache converged state
        state = dict(
            p_ref_1=p_ref_1, T_ref_1=T_ref_1, h_ref_1=h_ref_1, s_ref_1=s_ref_1, Q_ref_1=Q_ref_1,
            p_ref_2=p_ref_2, T_ref_2=T_ref_2, h_ref_2=h_ref_2, s_ref_2=s_ref_2, Q_ref_2=Q_ref_2,
            p_ref_3=p_ref_3, T_ref_3=T_ref_3, h_ref_3=h_ref_3, s_ref_3=s_ref_3, Q_ref_3=Q_ref_3,
            p_ref_4=p_ref_4, T_ref_4=T_ref_4, h_ref_4=h_ref_4, s_ref_4=s_ref_4, Q_ref_4=Q_ref_4,
            T_cond=T_cond, T_ev=T_ev,
            T_c_out=T_c_out, T_h_out=T_h_out, T_c_pp_2=T_c_pp_2,
            ṁ_ref=ṁ_ref, cp_ref_2=cp_ref_2,
        )

    return state


# Performance metrics
# ===================
def compute_performance(state):
    s = state
    Ẇ_turb = s["ṁ_ref"] * (s["h_ref_3"] - s["h_ref_4"])
    Ẇ_comp = s["ṁ_ref"] * (s["h_ref_2"] - s["h_ref_1"])
    Q_out   = ṁ_c * cp_c * (s["T_c_out"] - T_c_in)
    Q_in    = ṁ_h * cp_h * (T_h_in - s["T_h_out"])

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
    )


# Diagram data preparation
# ========================
def build_ts_data(state):
    s = state
    p1, p2 = s["p_ref_1"], s["p_ref_2"]

    # Saturation entropy values at both pressures
    s_ref_23_v_inflection = PropsSI("S", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_23_l_inflection = PropsSI("S", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    s_ref_41_v_inflection = PropsSI("S", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_41_l_inflection = PropsSI("S", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

    # Condenser path 2 -> 3:
    # Only insert the vapour-dome inflection if point 2 is superheated (s2 > s_dew).
    # Only insert the liquid inflection if point 3 is subcooled (s3 < s_bubble).
    s_23_chain, T_23_chain = [], []
    if s["s_ref_2"] > s_ref_23_v_inflection:
        seg_s, seg_T = isobar_segment(s["s_ref_2"], s_ref_23_v_inflection, p2)
        s_23_chain += seg_s + [s_ref_23_v_inflection]
        T_23_chain += seg_T + [s["T_cond"]]
    if s["s_ref_3"] < s_ref_23_l_inflection:
        seg_s, seg_T = isobar_segment(s_ref_23_l_inflection, s["s_ref_3"], p2)
        s_23_chain += [s_ref_23_l_inflection] + seg_s
        T_23_chain += [s["T_cond"]] + seg_T
    # If neither branch triggered, the whole 2->3 leg is inside the dome
    if not s_23_chain:
        seg_s, seg_T = isobar_segment(s["s_ref_2"], s["s_ref_3"], p2)
        s_23_chain, T_23_chain = seg_s, seg_T

    # Evaporator path 4 -> 1:
    # Only insert the liquid inflection if point 4 is subcooled (s4 < s_bubble).
    # Only insert the vapour inflection if point 1 is superheated (s1 > s_dew).
    s_41_chain, T_41_chain = [], []
    if s["s_ref_4"] < s_ref_41_l_inflection:
        seg_s, seg_T = isobar_segment(s["s_ref_4"], s_ref_41_l_inflection, p1)
        s_41_chain += seg_s + [s_ref_41_l_inflection]
        T_41_chain += seg_T + [s["T_ev"]]
    if s["s_ref_1"] > s_ref_41_v_inflection:
        seg_s, seg_T = isobar_segment(s_ref_41_v_inflection, s["s_ref_1"], p1)
        s_41_chain += [s_ref_41_v_inflection] + seg_s
        T_41_chain += [s["T_ev"]] + seg_T
    if not s_41_chain:
        seg_s, seg_T = isobar_segment(s["s_ref_4"], s["s_ref_1"], p1)
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


def build_ph_data(state):
    s = state
    p1, p2 = s["p_ref_1"], s["p_ref_2"]

    # Saturation enthalpy values at both pressures
    h_ref_23_v_inflection = PropsSI("H", "P", p2, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_23_l_inflection = PropsSI("H", "P", p2, "Q", 0, f"REFPROP::{refrigerant}")
    h_ref_41_v_inflection = PropsSI("H", "P", p1, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_41_l_inflection = PropsSI("H", "P", p1, "Q", 0, f"REFPROP::{refrigerant}")

    # Condenser path 2 -> 3:
    # Only insert the vapour inflection if point 2 is superheated (h2 > h_dew).
    # Only insert the liquid inflection if point 3 is subcooled (h3 < h_bubble).
    h_23_chain, p_23_chain = [], []
    if s["h_ref_2"] > h_ref_23_v_inflection:
        seg_h, seg_p = isobar_h_segment(s["h_ref_2"], h_ref_23_v_inflection, p2)
        h_23_chain += seg_h + [h_ref_23_v_inflection]
        p_23_chain += seg_p + [p2]
    if s["h_ref_3"] < h_ref_23_l_inflection:
        seg_h, seg_p = isobar_h_segment(h_ref_23_l_inflection, s["h_ref_3"], p2)
        h_23_chain += [h_ref_23_l_inflection] + seg_h
        p_23_chain += [p2] + seg_p
    # If neither triggered, the whole 2->3 leg is inside the dome
    if not h_23_chain:
        seg_h, seg_p = isobar_h_segment(s["h_ref_2"], s["h_ref_3"], p2)
        h_23_chain, p_23_chain = seg_h, seg_p

    # Evaporator path 4 -> 1:
    # Only insert the liquid inflection if point 4 is subcooled (h4 < h_bubble).
    # Only insert the vapour inflection if point 1 is superheated (h1 > h_dew).
    h_41_chain, p_41_chain = [], []
    if s["h_ref_4"] < h_ref_41_l_inflection:
        seg_h, seg_p = isobar_h_segment(s["h_ref_4"], h_ref_41_l_inflection, p1)
        h_41_chain += seg_h + [h_ref_41_l_inflection]
        p_41_chain += seg_p + [p1]
    if s["h_ref_1"] > h_ref_41_v_inflection:
        seg_h, seg_p = isobar_h_segment(h_ref_41_v_inflection, s["h_ref_1"], p1)
        h_41_chain += [h_ref_41_v_inflection] + seg_h
        p_41_chain += [p1] + seg_p
    if not h_41_chain:
        seg_h, seg_p = isobar_h_segment(s["h_ref_4"], s["h_ref_1"], p1)
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