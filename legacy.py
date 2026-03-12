import CoolProp
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import PropertyPlot
from matplotlib.lines import Line2D



# Cycle specification
# ===================
T_c_in = 353.15             # [K] - 80 degC
T_h_in = 287.15             # [K] - 15 degC (outside temp)
ṁ_c = 42                    # [kg\s] - BOTE calculation using typical refrigerant mass flow with 50 deg sensible heat and latent heat of evap of R1234ze  https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2445&context=iracc/1000 and [P]_optimal_cycle_and_turbine_design_for_MW_scale_waste_heat_recovery_orc_with_partial_evaporation
cp_c = 1885                 # [J/kg/K] - steam at 250 degrees https://www.engineeringtoolbox.com/water-vapor-d_979.html
ṁ_h = 40                    # [kg\s] arbitrarily chosen based on the prior 
cp_h = 1006                 # [J/kg/K] - air at 30 degrees and I presume atmospheric pressure
η_turb = 0.87               # [-] - from googling turbine maps
η_compr = 0.78              # [-] - from googling compressor maps and remembering how awful designing them is due to the adverse pressure gradient
ΔT_pp_1 = 7                 # [-] inspired by [P]_optimal_cycle_and_turbine_design_for_MW_scale_waste_heat_recovery_orc_with_partial_evaporation
ΔT_pp_2 = 7                 # [-]
ΔT_pp_3 = 7                 # [-]
ΔT_pp_4 = 7                 # [-]
ΔT_sh = 5                   # [-]
ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
refrigerant = "R1234ze(Z)"  # [-]



# Visualization method
# ====================
visualization_method = "CoolProp"  # "CoolProp" or "NiceProp"   -  NiceProp not implemented yet.
diagram_type = "TS" # "TS" OR "PH"
resolution = "low" # "high" OR "low"  -  changes the amount of runtime



# Scale quality to correct range
# ==============================
# coolprop can give massively negative or positive qualities for if fluid is 
# in single phase region. Scale to [0-1]
def vapour_quality_scaler(Q):
    if Q>1:
        Q = 1
    elif Q < 0:
        Q = 0
    else:
        pass
    return Q



# Cycle Evaluation
# ================
# does not work since the T_ref_3 is obtained from the PP, but if the pressure updater goes to a p_ref_2 that would yield a T_ref_3 below that, then 
# this lower T_ref_3 is not accounted for, eventually the h_ref_3 became larger than that of the h_ref_2 which makes no sense. Current implementation
# does not have this problem since the fixed T_ref_3 is used at the start => necessity of choosing fixed parameters at the same point!
# this is actually not the full picture, the new method still ran into the same issue with point 1. The issue is the bisection method. 
# you took some notes on this, the paper where you made the extrapolated entropy calculation as well (has two T-S diagrams on one side)

# properties at station 1 (fixed for the given user inputs)
T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")
p_ref_1 = p_ev

# lower pressure limit
T_ref_3 = T_c_in + ΔT_pp_3
p_lim_lower = PropsSI("P", "T", T_ref_3, "Q", 0, f"REFPROP::{refrigerant}")
PR_lower_limit = p_lim_lower/p_ref_1

# upper pressure limit
p_crit = PropsSI("Pcrit", f"REFPROP::{refrigerant}")
p_lim_upper = p_crit
PR_upper_limit = p_lim_upper/p_ref_1

# bisection method
PR_bisection_range = [PR_lower_limit, PR_upper_limit]
PR_guess = sum(PR_bisection_range)/len(PR_bisection_range)
ΔT_pp_4_calculated = 0
while not math.isclose(ΔT_pp_4_calculated, ΔT_pp_4, rel_tol=1e-3):
    # evaporation properties
    T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
    p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")

    # compressor inlet properties
    p_ref_1 = p_ev
    T_ref_1 = T_h_in - ΔT_pp_1
    h_ref_1 = PropsSI("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    s_ref_1 = PropsSI("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    Q_ref_1 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")) # vapour quality

    # condenser inlet properties
    p_ref_2 = PR_guess*p_ref_1
    h_ref_2_is = PropsSI("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
    h_ref_2 = (h_ref_2_is - h_ref_1)/η_compr + h_ref_1
    T_ref_2 = PropsSI("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
    Q_ref_2 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")) # vapour quality
    s_ref_2 = PropsSI("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}") # for completeness

    # Turbine inlet properties
    T_ref_3 = T_c_in + ΔT_pp_3
    p_ref_3 = p_ref_2
    h_ref_3 = PropsSI("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    s_ref_3 = PropsSI("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    Q_ref_3 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}"))
    T_cond = PropsSI("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")

    # Evaporator inlet properties
    p_ref_4 = p_ref_1
    h_ref_4_is = PropsSI("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
    h_ref_4 = (h_ref_4_is - h_ref_3)*η_turb + h_ref_3
    T_ref_4 = PropsSI("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
    Q_ref_4 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")) # vapour quality
    s_ref_4 = PropsSI("S", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}") # for completeness

    # specify specific heats (assumed constant)
    cp_ref_1 = PropsSI("C", "P", p_ref_1, "T", T_ref_1, f"REFPROP::{refrigerant}")
    cp_ref_2 = PropsSI("C", "P", p_ref_2, "T", T_ref_2, f"REFPROP::{refrigerant}")
    cp_ref_3 = PropsSI("C", "P", p_ref_3, "T", T_ref_3, f"REFPROP::{refrigerant}")
    cp_ref_4 = PropsSI("C", "P", p_ref_4, "T", T_ref_4, f"REFPROP::{refrigerant}") 

    # latent heat of condensation
    h_vap = PropsSI("H","P",p_ref_2,"Q",1,f"REFPROP::{refrigerant}")
    h_liq = PropsSI("H","P",p_ref_2,"Q",0,f"REFPROP::{refrigerant}")
    Δh_cond = h_vap-h_liq

    # latent heat of evaporation
    h_vap = PropsSI("H","P",p_ref_1,"Q",1,f"REFPROP::{refrigerant}")
    h_liq = PropsSI("H","P",p_ref_1,"Q",0,f"REFPROP::{refrigerant}")
    Δh_ev = h_vap-h_liq

    # compute T_c_pp_2 from pp requirement
    T_cond = PropsSI("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
    T_c_pp_2 = T_cond - ΔT_pp_2

    # T_c_pp_2 heat balance equation to obtain ṁ_ref
    ṁ_ref = (T_c_pp_2 - T_c_in)*ṁ_c*cp_c / (Δh_cond*(Q_ref_2 - Q_ref_3) + (T_cond - T_ref_3)*cp_ref_3)

    # compute T_c_out from condensation heat balance
    T_c_out = T_c_in + ((T_ref_2 - T_cond)*ṁ_ref*cp_ref_2 + (Q_ref_2 - Q_ref_3)*Δh_cond*ṁ_ref + (T_cond - T_ref_3)*cp_ref_3*ṁ_ref)/(ṁ_c*cp_c)

    # compute T_h_out from condensation heat balance
    T_h_out = T_h_in - ((T_ref_1 - T_ev)*ṁ_ref*cp_ref_1 + (Q_ref_1 - Q_ref_4)*Δh_ev*ṁ_ref + (T_ref_4 - T_ev)*cp_ref_4*ṁ_ref)/(ṁ_h*cp_h)

    # compute ΔT_pp_4 and check against the requirement
    ΔT_pp_4_calculated = T_h_out - T_ref_4

    # bisection method logic
    if (ΔT_pp_4_calculated - ΔT_pp_4) < 0: 
        PR_bisection_range[1] = PR_guess
    else:
        PR_bisection_range[0] = PR_guess
    PR_guess = sum(PR_bisection_range)/len(PR_bisection_range)
    


# Cycle performance 
Ẇ_turb = ṁ_ref*(h_ref_3 - h_ref_4)
Ẇ_comp = ṁ_ref*(h_ref_2 - h_ref_1)
Q_out = ṁ_c*cp_c*(T_c_out - T_c_in) # water vapour, so enthalpy change is directly related to temperature change.
Q_in = ṁ_h*cp_h*(T_h_in - T_h_out) # air, so enthalpy change is directly related to temperature change.
COP = Q_out/((Ẇ_comp - Ẇ_turb*ɳ_shaft))



# TS-Diagram data preparation
# ===========================
if diagram_type == "TS":
    s_ref_23_v_inflection = PropsSI("S", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
    s_ref_23_l_inflection = PropsSI("S", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")
    s_ref_41_v_inflection = PropsSI("S", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}")
    # Compute isobar segments to give them a nice curve
    def isobar_segment(s_start, s_end, p):
        if resolution == "low":
            num_points = 150
        else:            
            num_points = 1000
        s_range = np.linspace(s_start, s_end, num = num_points)
        T_range = np.zeros(shape = (1, num_points))
        for i, s in enumerate(s_range):
            try:
                T_range[0, i] = PropsSI("T", "P", p, "S", s, f"REFPROP::{refrigerant}")
            except ValueError:
                pass
        return s_range.tolist(), T_range.tolist()[0]
    s_ref_23_v, T_ref_23_v = isobar_segment(s_ref_2, s_ref_23_v_inflection, p_ref_2)
    s_ref_23_l, T_ref_23_l = isobar_segment(s_ref_23_l_inflection, s_ref_3, p_ref_2)
    s_ref_41_v, T_ref_41_v = isobar_segment(s_ref_41_v_inflection, s_ref_1, p_ref_1)    

    if s_ref_4 < PropsSI("S", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}"):
        s_ref_41_l_inflection = PropsSI("S", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}")
        s_ref_41_l, T_ref_41_l = isobar_segment(s_ref_4, s_ref_41_l_inflection, p_ref_1)
    else:
        s_ref_41_l_inflection = s_ref_4
        s_ref_41_l, T_ref_41_l = [s_ref_4], [T_ref_4]

    # Concatenate data
    s_ref_lst = list(itertools.chain(
        [s_ref_1], [s_ref_2], s_ref_23_v, [s_ref_23_v_inflection], [s_ref_23_l_inflection], s_ref_23_l, [s_ref_3], 
        [s_ref_4], s_ref_41_l, [s_ref_41_l_inflection], [s_ref_41_v_inflection], s_ref_41_v
        ))
    T_ref_lst = list(itertools.chain(
        [T_ref_1], [T_ref_2], T_ref_23_v, [T_cond], [T_cond], T_ref_23_l, [T_ref_3], 
        [T_ref_4], T_ref_41_l, [T_ev], [T_ev], T_ref_41_v
    ))

    # Property dictionary
    HP_cycle_props = {
        "point_1": [p_ref_1, T_ref_1, h_ref_1, s_ref_1, Q_ref_1], 
        "point_2": [p_ref_2, T_ref_2, h_ref_2, s_ref_2, Q_ref_2], 
        "point_3": [p_ref_3, T_ref_3, h_ref_3, s_ref_3, Q_ref_3], 
        "point_4": [p_ref_4, T_ref_4, h_ref_4, s_ref_4, Q_ref_4]
    }
    HP_cycle_plot_major = {
        "s": [s_ref_1, s_ref_2, s_ref_3, s_ref_4, s_ref_1],
        "T": [T_ref_1, T_ref_2, T_ref_3, T_ref_4, T_ref_1]
    }
    HP_cycle_plot_minor = {
        "s": s_ref_lst,
        "T": T_ref_lst
    }

    # Coolant and heating flows
    s_c_in = s_ref_3 # not true, but for visualization purposes
    s_h_in = s_ref_1 # same as above
    s_h_out = s_ref_4 # same as above
    # for correct plotting of the pinch point at location 2, there is a particularity:
    # Since we plot the coolant flow for the entropy of the refrigerant, the entropy point
    # of the PP will not necessarily look right. That is, the pp location will actually be
    # very close to T_c_out, thus at larger s_ref on the plot, causing the visual difference
    # solution: plot such that the pp is obvious.
    s_ref_pp_2 = PropsSI("S", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
    s_c_out = s_ref_3 + (s_ref_pp_2 - s_ref_3)/(T_c_pp_2 - T_c_in)*(T_c_out-T_c_in)
    coolant_flow = {
        "s": [s_c_in, s_c_out],
        "T": [T_c_in, T_c_out]
    }
    heating_flow = {
        "s": [s_h_in, s_h_out],
        "T": [T_h_in, T_h_out]
    }



# PH-Diagram data preparation
# ===========================
def isobar_h_segment(h_start, h_end, p):
    """Constant-pressure segment for PH plots (horizontal lines in P-H)."""
    h_range = np.linspace(h_start, h_end, num=30)
    p_range = np.full(30, p)
    return h_range.tolist(), p_range.tolist()
HP_cycle_plot_major_PH = None
HP_cycle_plot_minor_PH = None

if diagram_type == "PH":
    h_ref_23_v_inflection = PropsSI("H", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
    h_ref_23_l_inflection = PropsSI("H", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")
    h_ref_41_v_inflection = PropsSI("H", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}")

    # Condenser (high pressure, h decreasing)
    h_ref_23_v, p_ref_23_v = isobar_h_segment(h_ref_2, h_ref_23_v_inflection, p_ref_2)
    h_ref_23_l, p_ref_23_l = isobar_h_segment(h_ref_23_l_inflection, h_ref_3, p_ref_2)

    # Evaporator (low pressure)
    h_liq_ev = PropsSI("H", "P", p_ref_1, "Q", 0, f"REFPROP::{refrigerant}")
    if h_ref_4 < h_liq_ev:
        h_ref_41_l_inflection = h_liq_ev
        h_ref_41_l, p_ref_41_l = isobar_h_segment(h_ref_4, h_ref_41_l_inflection, p_ref_1)
    else:
        h_ref_41_l_inflection = h_ref_4
        h_ref_41_l, p_ref_41_l = [h_ref_4], [p_ref_4]

    h_ref_41_v, p_ref_41_v = isobar_h_segment(h_ref_41_v_inflection, h_ref_1, p_ref_1)

    # Concatenate PH path (mirrors exactly how TS minor path was built)
    h_ref_lst = list(itertools.chain(
        [h_ref_1], [h_ref_2], h_ref_23_v, [h_ref_23_v_inflection],
        [h_ref_23_l_inflection], h_ref_23_l, [h_ref_3],
        [h_ref_4], h_ref_41_l, [h_ref_41_l_inflection],
        [h_ref_41_v_inflection], h_ref_41_v
    ))
    p_ref_lst = list(itertools.chain(
        [p_ref_1], [p_ref_2], p_ref_23_v, [p_ref_2],
        [p_ref_2], p_ref_23_l, [p_ref_3],
        [p_ref_4], p_ref_41_l, [p_ref_1],
        [p_ref_1], p_ref_41_v
    ))

    HP_cycle_plot_major_PH = {
        "h": [h_ref_1, h_ref_2, h_ref_3, h_ref_4, h_ref_1],
        "p": [p_ref_1, p_ref_2, p_ref_3, p_ref_4, p_ref_1]
    }
    HP_cycle_plot_minor_PH = {
        "h": h_ref_lst,
        "p": p_ref_lst
    }



# Coolprop integrated visualization
# =================================
if visualization_method == "CoolProp":
    # change plot styling to LaTeX
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "text.latex.preamble": r"\usepackage{siunitx}\sisetup{group-separator={\,},group-minimum-digits=4}"
    })

    # draw isolines using the built-in coolprop method
    import warnings
    warnings.filterwarnings("ignore")
    coolplot = PropertyPlot(f"REFPROP::{refrigerant}", diagram_type, unit_system='SI', tp_limits = 'ACHP')
    if diagram_type == "TS":
        if resolution == "low":
            coolplot.calc_isolines(CoolProp.iQ, num=10, points = 150) # num indicating the number of isolines you want to display          was 1500   3000 3000
            coolplot.calc_isolines(CoolProp.iP, num=12, points = 300)
            coolplot.calc_isolines(CoolProp.iHmass, num=20, points = 300)
        if resolution == "high":
            coolplot.calc_isolines(CoolProp.iQ, num=10, points = 1500) # num indicating the number of isolines you want to display          was 1500   3000 3000
            coolplot.calc_isolines(CoolProp.iP, num=12, points = 3000)
            coolplot.calc_isolines(CoolProp.iHmass, num=20, points = 3000)
        # plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=True)
    if diagram_type == "PH":
        if resolution == "low":
            coolplot.calc_isolines(CoolProp.iQ, num=10, points = 150) # num indicating the number of isolines you want to display          was 1500   3000 3000
            coolplot.calc_isolines(CoolProp.iSmass, num=12, points = 300)
            coolplot.calc_isolines(CoolProp.iT, num=20, points = 300)
        if resolution == "high":
            coolplot.calc_isolines(CoolProp.iQ, num=10, points = 1500) # num indicating the number of isolines you want to display          was 1500   3000 3000
            coolplot.calc_isolines(CoolProp.iSmass, num=12, points = 3000)
            coolplot.calc_isolines(CoolProp.iT, num=20, points = 3000)

    def format_isoline_label(prop_key, value, include_name=False):
        if prop_key == CoolProp.iP:
            value_str = f"{value/1e6:.2f}"
            return rf"$p={value_str}\,\mathrm{{MPa}}$" if include_name else rf"${value_str}$"
        if prop_key == CoolProp.iQ:
            value_str = f"{value:.2f}"
            return rf"$x={value_str}$" if include_name else rf"${value_str}$"
        if prop_key == CoolProp.iHmass:
            value_str = f"{value/1e3:.0f}"
            return rf"$h={value_str}\,\mathrm{{kJ/kg}}$" if include_name else rf"${value_str}$"

        # === NEW FOR PH ===
        if prop_key == CoolProp.iSmass:
            value_str = f"{value/1000:.2f}"
            return rf"$s={value_str}\,\mathrm{{kJ/kg·K}}$" if include_name else rf"${value_str}$"
        if prop_key == CoolProp.iT:
            value_str = f"{value:.0f}"
            return rf"$T={value_str}\,\mathrm{{K}}$" if include_name else rf"${value_str}$"

        value_str = f"{value:.2f}"
        return rf"${value_str}$"
    
    def find_intersection_with_isotherm(x_data, y_data, T_target=270.0):
        """Return (s_intersect, T_target) where the isoline crosses T_target.
        If no crossing, returns the closest point."""
        x = np.array(x_data)
        y = np.array(y_data)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2:
            return None

        # Find crossings
        crossings = np.where(np.diff(np.sign(y - T_target)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            dy = y[idx + 1] - y[idx]
            if abs(dy) < 1e-9:
                frac = 0.5
            else:
                frac = (T_target - y[idx]) / dy
            s_int = x[idx] + frac * (x[idx + 1] - x[idx])
            return s_int, T_target
        else:
            # no crossing → closest point
            idx = np.argmin(np.abs(y - T_target))
            return x[idx], y[idx]

    def add_isoline_labels(ax_obj, coolplot_obj, prop_key, color, side):
        if prop_key not in coolplot_obj.isolines:
            return
        iso_lines = coolplot_obj.isolines[prop_key]
        if len(iso_lines) == 0:
            return

        for k, iso in enumerate(iso_lines):
            x_data = np.array(iso.x)
            y_data = np.array(iso.y)
            valid = np.isfinite(x_data) & np.isfinite(y_data)
            x_data = x_data[valid]
            y_data = y_data[valid]

            # Clip to visible plot bounds
            xlim = ax_obj.get_xlim()
            ylim = ax_obj.get_ylim()
            inside = (x_data >= xlim[0]) & (x_data <= xlim[1]) & \
                    (y_data >= ylim[0]) & (y_data <= ylim[1])
            if np.sum(inside) < 3:
                continue
            x_data = x_data[inside]
            y_data = y_data[inside]

            # ────── VAPOUR QUALITY ──────
            if prop_key == CoolProp.iQ:
                if diagram_type == "TS":
                    inter = find_intersection_with_isotherm(x_data, y_data, T_target=265.0)
                    if inter is None:
                        i_label = len(x_data) // 2
                        x_target = x_data[i_label]
                        y_target = y_data[i_label]
                    else:
                        x_target, y_target = inter
                        i_label = np.argmin(np.abs(x_data - x_target))
                    # FIX: left-to-right ordering for EVERY quality line → all labels have identical orientation
                    # (no more flipped second-to-last or last label)
                    sort_idx = np.argsort(x_data)
                    x_data = x_data[sort_idx]
                    y_data = y_data[sort_idx]
                    i_label = np.argmin(np.abs(x_data - x_target))  # re-locate after sort
                else:  # PH
                    i_label = len(x_data) // 2
                    x_target = x_data[i_label]
                    y_target = y_data[i_label]

            # ────── PRESSURE, ENTHALPY, ENTROPY, TEMPERATURE ──────
            else:
                x_min, x_max = np.min(x_data), np.max(x_data)

                if diagram_type == "TS" and prop_key == CoolProp.iHmass:
                    target_k = int(len(iso_lines) * 0.75)
                    frac = 0.88 if k == target_k else 0.93
                elif diagram_type == "TS" and prop_key == CoolProp.iP:
                    frac = 0.92
                else:
                    frac = 0.92 if side == "right" else 0.08

                x_target = x_min + frac * (x_max - x_min)
                i_label = np.argmin(np.abs(x_data - x_target))
                y_target = y_data[i_label]

            # safe slope calculation
            i_label = max(1, min(len(x_data) - 2, i_label))
            p_prev = ax_obj.transData.transform((x_data[i_label-1], y_data[i_label-1]))
            p_next = ax_obj.transData.transform((x_data[i_label+1], y_data[i_label+1]))
            angle = np.degrees(np.arctan2(p_next[1] - p_prev[1], p_next[0] - p_prev[0]))
            if angle > 90:   angle -= 180
            if angle < -90:  angle += 180

            # which line gets the full name
            if diagram_type == "TS" and prop_key == CoolProp.iHmass:
                target_k = int(len(iso_lines) * 0.75)
                include_name = (k == target_k)
            else:
                include_name = (k == len(iso_lines) // 2)

            label = format_isoline_label(prop_key, iso.value, include_name=include_name)

            ax_obj.text(
                x_target, y_target,
                label,
                color=color,
                fontsize=7,
                rotation=angle,
                rotation_mode='anchor',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
                zorder=6,
                clip_on=True
            )

    def add_mid_arrow(ax_obj, x_vals, y_vals, color, frac=0.18):
        if len(x_vals) < 2 or len(y_vals) < 2:
            return
        x_in, x_out = x_vals[0], x_vals[1]
        y_in, y_out = y_vals[0], y_vals[1]
        dx = x_out - x_in
        dy = y_out - y_in
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            return

        x_mid = x_in + 0.5 * dx
        y_mid = y_in + 0.5 * dy
        u = frac * dx
        v = frac * dy

        # Pivot in the middle so the arrow is visually centered on the line.
        ax_obj.quiver(
            x_mid,
            y_mid,
            u,
            v,
            angles='xy',
            scale_units='xy',
            scale=1,
            pivot='middle',
            color=color,
            width=0.003,
            headwidth=4.5,
            headlength=6,
            headaxislength=5,
            zorder=5
        )

    def add_endpoint_temperature_labels(ax_obj, x_vals, y_vals, start_label, end_label, color, side=1):
        if len(x_vals) < 2 or len(y_vals) < 2:
            return

        x_start, x_end = x_vals[0], x_vals[1]
        y_start, y_end = y_vals[0], y_vals[1]

        p_start = np.array(ax_obj.transData.transform((x_start, y_start)), dtype=float)
        p_end = np.array(ax_obj.transData.transform((x_end, y_end)), dtype=float)
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        if np.isclose(norm, 0.0):
            tangent = np.array([1.0, 0.0])
            normal = np.array([0.0, 1.0])
        else:
            tangent = vec / norm
            normal = np.array([-tangent[1], tangent[0]])
        normal = side * normal

        # Keep lateral separation but reduce vertical/perpendicular lift.
        offset_normal = 6.0
        offset_along = 15.0
        start_offset = normal * offset_normal - tangent * offset_along
        end_offset = normal * offset_normal + tangent * offset_along

        ax_obj.annotate(
            start_label,
            xy=(x_start, y_start),
            xytext=(start_offset[0], start_offset[1]),
            textcoords='offset points',
            fontsize=8,
            color=color,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6,
            clip_on=True
        )
        ax_obj.annotate(
            end_label,
            xy=(x_end, y_end),
            xytext=(end_offset[0], end_offset[1]),
            textcoords='offset points',
            fontsize=8,
            color=color,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6,
            clip_on=True
        )

    # see documentation, calc_isolines does not link data to figure object yet. Idk how exactly this works but you can find
    # that coolprop saves the data to propertyplot.isolines https://github.com/CoolProp/CoolProp/blob/master/wrappers/Python/CoolProp/Plots/Plots.py
    # so you can extract the data from there
    isolines = {}
    for key in list(coolplot.isolines.keys()):
        isolines[key] = {}
        isolines[key]["x"] = []
        isolines[key]["y"] = []
        isolines[key]["value"] = []
        for isoline in coolplot.isolines[key]:
            isolines[key]["x"].append(isoline.x)
            isolines[key]["y"].append(isoline.y)
            isolines[key]["value"].append(isoline.value)

    # Extract figure object
    fig = coolplot.figure
    ax = fig.gca()

    # draw isolines (unchanged)
    n_segments = len(list(isolines.keys()))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(n_segments)]
    i = 0
    for key, col in zip(list(isolines.keys()), colors):
        for j in range(len(isolines[key]["x"])):
            ax.plot(isolines[key]["x"][j], isolines[key]["y"][j], color = col, zorder = i, linewidth = 0.6)
        i+=1

    # add critical point
    p_crit = PropsSI("Pcrit", f"REFPROP::{refrigerant}")
    T_crit = PropsSI("Tcrit", f"REFPROP::{refrigerant}")
    if diagram_type == "TS":
        s_crit = PropsSI("S", "P", p_crit, "T", T_crit, f"REFPROP::{refrigerant}")
        ax.plot(s_crit, T_crit, marker = "o", markerfacecolor = "yellow", markersize = 5, markeredgecolor = 'black')
    else:  # PH
        h_crit = PropsSI("H", "T", T_crit, "P", p_crit, f"REFPROP::{refrigerant}")
        ax.plot(h_crit, p_crit, marker = "o", markerfacecolor = "yellow", markersize = 5, markeredgecolor = 'black')

    # add cycle
    if diagram_type == "TS":
        ax.scatter(HP_cycle_plot_major["s"], HP_cycle_plot_major["T"], 
                   color='orange', marker='o', s=5, zorder=8)
        ax.plot(HP_cycle_plot_minor["s"], HP_cycle_plot_minor["T"], 
                color='green', linewidth=1.5, zorder=7)
    else:  # PH
        ax.scatter(HP_cycle_plot_major_PH["h"], HP_cycle_plot_major_PH["p"], 
                   color='orange', marker='o', s=5, zorder=8)
        ax.plot(HP_cycle_plot_minor_PH["h"], HP_cycle_plot_minor_PH["p"], 
                color='green', linewidth=1.5, zorder=7)
        
    # Isentropic & isenthalpic expansion lines
    # === Isentropic & isenthalpic expansion lines — slightly more green + bit transparent + beige legend box (on top of labels) ===
    if diagram_type in ["TS", "PH"]:
        p_exp_range = np.linspace(p_ref_3, p_ref_1, num=100)

        # Slightly more green + still a bit transparent
        line_color = "#02220E"
        line_alpha = 0.75

        if diagram_type == "PH":
            # PH: isenthalpic = constant h
            h_isenth = np.full_like(p_exp_range, h_ref_3)
            line_isenth, = ax.plot(h_isenth, p_exp_range, color=line_color, linestyle="-",
                                   linewidth=1.5, alpha=line_alpha, zorder=4,
                                   label=r"$\mathrm{isenthalpic\ expansion}$")

            # PH: isentropic = constant s
            h_isen = np.zeros_like(p_exp_range, dtype=float)
            for i, pp in enumerate(p_exp_range):
                try:
                    h_isen[i] = PropsSI("H", "P", pp, "S", s_ref_3, f"REFPROP::{refrigerant}")
                except ValueError:
                    h_isen[i] = np.nan
            valid = np.isfinite(h_isen)
            line_isen, = ax.plot(h_isen[valid], p_exp_range[valid], color=line_color, linestyle="--",
                                 linewidth=1.5, alpha=line_alpha, zorder=4,
                                 label=r"$\mathrm{isentropic\ expansion}$")

        else:  # TS diagram
            # TS: isentropic = constant s (vertical)
            T_isen_exit = PropsSI("T", "P", p_ref_1, "S", s_ref_3, f"REFPROP::{refrigerant}")
            T_isen_range = np.linspace(T_ref_3, T_isen_exit, num=100)
            s_isen = np.full_like(T_isen_range, s_ref_3)
            line_isen, = ax.plot(s_isen, T_isen_range, color=line_color, linestyle="--",
                                 linewidth=1.5, alpha=line_alpha, zorder=4,
                                 label=r"$\mathrm{isentropic\ expansion}$")

            # TS: isenthalpic = constant h (curved)
            s_isenth = np.zeros_like(p_exp_range, dtype=float)
            T_isenth = np.zeros_like(p_exp_range, dtype=float)
            for i, pp in enumerate(p_exp_range):
                try:
                    s_isenth[i] = PropsSI("S", "P", pp, "H", h_ref_3, f"REFPROP::{refrigerant}")
                    T_isenth[i] = PropsSI("T", "P", pp, "H", h_ref_3, f"REFPROP::{refrigerant}")
                except ValueError:
                    s_isenth[i] = np.nan
                    T_isenth[i] = np.nan
            valid = np.isfinite(s_isenth)
            line_isenth, = ax.plot(s_isenth[valid], T_isenth[valid], color=line_color, linestyle="-",
                                   linewidth=1.5, alpha=line_alpha, zorder=4,
                                   label=r"$\mathrm{isenthalpic\ expansion}$")

        # Legend — lower right, beige style exactly like your performance box, and forced on top of all isoline label.
        line_turb = Line2D([0], [0], color='green', linewidth=1.5,
                        label=r"$\mathrm{turbine\ expansion}$")
        leg = ax.legend(handles=[line_turb, line_isen, line_isenth],
            loc="lower right", bbox_to_anchor=(0.9875, 0.015),
            fontsize=8, framealpha=0.85)

        # Apply the exact beige styling (same colours/padding/edge as performance box)
        frame = leg.get_frame()
        frame.set_facecolor((0.96, 0.92, 0.84, 0.72))
        frame.set_edgecolor('#9C7B53')
        frame.set_linewidth(1.2)
        frame.set_boxstyle('round,pad=0.5')

        # Force legend on top of every isoline label and cycle
        leg.set_zorder(11)
        
    # add coolant and heating flow ONLY for TS (PH has no sensible meaning here)
    if diagram_type == "TS":
        ax.plot(coolant_flow["s"], coolant_flow["T"], color="blue", marker="o", markersize=2, zorder=12)
        ax.plot(heating_flow["s"], heating_flow["T"], color="red", marker="o", markersize=2, zorder=12)
        add_mid_arrow(ax, coolant_flow["s"], coolant_flow["T"], color="blue")
        add_mid_arrow(ax, heating_flow["s"], heating_flow["T"], color="red")
        add_endpoint_temperature_labels(ax, coolant_flow["s"], coolant_flow["T"], 
                                        r"$T_{c,\mathrm{in}}$", r"$T_{c,\mathrm{out}}$", 
                                        color="blue", side=1)
        add_endpoint_temperature_labels(ax, heating_flow["s"], heating_flow["T"], 
                                        r"$T_{h,\mathrm{in}}$", r"$T_{h,\mathrm{out}}$", 
                                        color="red", side=-1)

    # isobar (TS) / critical isotherm (PH) — dotted, high resolution, full span
    if diagram_type == "TS":
        s_range = ax.get_xlim()
        s_crit_isobar, T_crit_isobar = isobar_segment(s_range[0], s_range[1], p_crit)
        ax.plot(s_crit_isobar, T_crit_isobar, color="black", linestyle=":", linewidth=1.0, zorder=1)
    else:  # PH diagram → critical isotherm at T = T_crit
        h_min, h_max = ax.get_xlim()
        p_min, p_max = ax.get_ylim()
        
        # Vapor branch (right side, high h, low p) — reaches right edge
        p_vap = np.linspace(max(1e3, p_min*0.05), p_crit, num=500)
        h_vap = np.full_like(p_vap, np.nan, dtype=float)
        for i, pp in enumerate(p_vap):
            try:
                h_vap[i] = PropsSI("H", "P", pp, "T", T_crit, f"REFPROP::{refrigerant}")
            except ValueError:
                pass
        valid_v = np.isfinite(h_vap) & (h_vap >= h_min) & (h_vap <= h_max)
        if np.any(valid_v):
            ax.plot(h_vap[valid_v], p_vap[valid_v], color="black", linestyle=":", linewidth=1.0, zorder=1)
        
        # Supercritical/liquid branch (left side, low h, high p) — reaches left edge
        p_liq = np.linspace(p_crit, min(10*p_crit, p_max*1.1), num=500)
        h_liq = np.full_like(p_liq, np.nan, dtype=float)
        for i, pp in enumerate(p_liq):
            try:
                h_liq[i] = PropsSI("H", "P", pp, "T", T_crit, f"REFPROP::{refrigerant}")
            except ValueError:
                pass
        valid_l = np.isfinite(h_liq) & (h_liq >= h_min) & (h_liq <= h_max)
        if np.any(valid_l):
            ax.plot(h_liq[valid_l], p_liq[valid_l], color="black", linestyle=":", linewidth=1.0, zorder=1)

    # label isolines (different keys for PH)
    fig.canvas.draw()
    if CoolProp.iQ in isolines:
        q_col = colors[list(isolines.keys()).index(CoolProp.iQ)]
        add_isoline_labels(ax, coolplot, CoolProp.iQ, q_col, side="left")

    if diagram_type == "TS":
        if CoolProp.iP in isolines:
            p_col = colors[list(isolines.keys()).index(CoolProp.iP)]
            add_isoline_labels(ax, coolplot, CoolProp.iP, p_col, side="right")
        if CoolProp.iHmass in isolines:
            h_col = colors[list(isolines.keys()).index(CoolProp.iHmass)]
            add_isoline_labels(ax, coolplot, CoolProp.iHmass, h_col, side="right")
    else:  # PH
        if CoolProp.iSmass in isolines:
            s_col = colors[list(isolines.keys()).index(CoolProp.iSmass)]
            add_isoline_labels(ax, coolplot, CoolProp.iSmass, s_col, side="right")
        if CoolProp.iT in isolines:
            t_col = colors[list(isolines.keys()).index(CoolProp.iT)]
            add_isoline_labels(ax, coolplot, CoolProp.iT, t_col, side="right")

    # axis labels
    if diagram_type == "TS":
        ax.set_xlabel("$s [J/kg/K]$")
        ax.set_ylabel("$T [K]$")
    else:
        ax.set_xlabel("$h [kJ/kg]$")
        ax.set_ylabel("$p [Pa]$")
        # make all enthalpy ticks display in kJ/kg
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}$"))

    # performance box and title (unchanged)
    textstr = rf'''$\begin{{array}}{{lrl}}\multicolumn{{3}}{{c}}{{\mathrm{{Performance}}}} \\\hline \mathrm{{COP}} & \num{{{COP:.1f}}} & [-] \\\dot{{W}}_{{turb}} & \num{{{Ẇ_turb:.0f}}} & [\mathrm{{W}}] \\\dot{{W}}_{{compr}} & \num{{{Ẇ_comp:.0f}}} & [\mathrm{{W}}] \\\dot{{Q}}_{{in}} & \num{{{Q_in:.0f}}} & [\mathrm{{W}}]\end{{array}}$'''
    ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', zorder=10,
        bbox=dict(facecolor=(0.96, 0.92, 0.84, 0.72), edgecolor='#9C7B53', linewidth=1.2, boxstyle='round,pad=0.5'))

    ax.set_title(f"$Conceptual$ $Heat$ $Pump$ $Cycle$   $-$   ${refrigerant}$")
    fig.savefig(f"Conceptual HP Cycle - {refrigerant} - {diagram_type}.pdf", dpi=1000, bbox_inches="tight")









# Waste Bin
# =========
# the only wrong thing about this is the pressure updating code, which you noted down on paper how to fix
# this is hence just for reference...
# PR_bisection_range = [1, PropsSI("Pcrit", f"REFPROP::{refrigerant}")/10e4]
# PR = sum(PR_bisection_range)/len(PR_bisection_range)
# ΔT_pp_4_calculated = 0
# max_cycles = 30
# cycle_index = 0
# while not math.isclose(ΔT_pp_4_calculated, ΔT_pp_4, rel_tol=1e-6) and cycle_index < max_cycles:
#     # condensation properties:
#     T_cond =  T_c_in + ΔT_pp_3 + ΔT_sc
#     p_cond =  PropsSI("P", "T", T_cond, "Q", 1, f"REFPROP::{refrigerant}")

#     # turbine inlet properties
#     T_ref_3 = T_c_in + ΔT_pp_3
#     p_ref_3 = p_cond
#     h_ref_3 = PropsSI("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
#     s_ref_3 = PropsSI("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
#     Q_ref_3 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}"))

#     # evaporator inlet properties
#     p_ref_4 = p_ref_3/PR
#     h_ref_4_is = PropsSI("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
#     h_ref_4 = (h_ref_4_is - h_ref_3)*η_turb + h_ref_3
#     T_ref_4 = PropsSI("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
#     Q_ref_4 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")) # vapour quality
#     s_ref_4 = PropsSI("S", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}") # for completeness

#     # compressor inlet properties
#     p_ref_1 = p_ref_4
#     T_ref_1 = T_h_in - ΔT_pp_1
#     h_ref_1 = PropsSI("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
#     s_ref_1 = PropsSI("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
#     Q_ref_1 = vapour_quality_scaler(PropsSI("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")) # vapour quality

#     # condenser inlet properties
#     p_ref_2 = PR*p_ref_1
#     h_ref_2_is = PropsSI("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
#     h_ref_2 = (h_ref_2_is - h_ref_1)/η_compr + h_ref_1
#     T_ref_2 = PropsSI("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
#     Q_ref_2 = vapour_quality_scaler(PropsSI("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")) # vapour quality
#     s_ref_2 = PropsSI("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}") # for completeness

#     # specify specific heats (assumed constant)
#     cp_ref_1 = PropsSI("C", "P", p_ref_1, "T", T_ref_1, f"REFPROP::{refrigerant}")
#     cp_ref_2 = PropsSI("C", "P", p_ref_2, "T", T_ref_2, f"REFPROP::{refrigerant}")
#     cp_ref_3 = PropsSI("C", "P", p_ref_3, "T", T_ref_3, f"REFPROP::{refrigerant}")
#     cp_ref_4 = PropsSI("C", "P", p_ref_4, "T", T_ref_4, f"REFPROP::{refrigerant}") 

#     # latent heat of condensation
#     h_vap = PropsSI("H","P",p_ref_2,"Q",1,f"REFPROP::{refrigerant}")
#     h_liq = PropsSI("H","P",p_ref_2,"Q",0,f"REFPROP::{refrigerant}")
#     Δh_cond = h_vap-h_liq

#     # latent heat of evaporation
#     h_vap = PropsSI("H","P",p_ref_1,"Q",1,f"REFPROP::{refrigerant}")
#     h_liq = PropsSI("H","P",p_ref_1,"Q",0,f"REFPROP::{refrigerant}")
#     Δh_ev = h_vap-h_liq

#     # compute T_c_pp_2 from pp requirement
#     T_c_pp_2 = T_cond - ΔT_pp_2

#     # T_c_pp_2 heat balance equation to obtain ṁ_ref
#     ṁ_ref = (T_c_pp_2 - T_c_in)*ṁ_c*cp_c / (Δh_cond*(Q_ref_2 - Q_ref_3) + (T_cond - T_ref_3)*cp_ref_3)

#     # compute T_c_out from condensation heat balance
#     T_c_out = T_c_in + ((T_ref_2 - T_cond)*ṁ_ref*cp_ref_2 + (Q_ref_2 - Q_ref_3)*Δh_cond*ṁ_ref + (T_cond - T_ref_3)*cp_ref_3*ṁ_ref)/(ṁ_c*cp_c)

#     # compute T_h_out from condensation heat balance
#     T_ev =  PropsSI("T", "P", p_ref_4, "Q", 0, f"REFPROP::{refrigerant}")
#     T_h_out = T_h_in - ((T_ref_1 - T_ev)*ṁ_ref*cp_ref_1 + (Q_ref_1 - Q_ref_4)*Δh_ev*ṁ_ref + (T_ref_4 - T_ev)*cp_ref_4*ṁ_ref)/(ṁ_h*cp_h)

#     # compute ΔT_pp_4 and check against the requirement
#     ΔT_pp_4_calculated = T_h_out - T_ref_4

#     # bisection method logic
#     if (ΔT_pp_4_calculated - ΔT_pp_4)/2 > 0: 
#         PR_bisection_range[1] = PR
#     else:
#         PR_bisection_range[0] = PR
#     PR = sum(PR_bisection_range)/len(PR_bisection_range)
#     cycle_index += 1

# # Cycle performance 
# Ẇ_turb = ṁ_ref*(cp_ref_3+cp_ref_4)/2*(h_ref_3 - h_ref_4)
# Ẇ_comp = ṁ_ref*(cp_ref_1+cp_ref_2)/2*(h_ref_2 - h_ref_1)
# Q_out = ṁ_c*cp_c*(T_c_out - T_c_in)
# Q_in = ṁ_h*cp_h*(T_h_in - T_h_out)
# COP = Q_out/((Ẇ_comp - Ẇ_turb*ɳ_shaft))
