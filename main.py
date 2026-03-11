import CoolProp
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import PropertyPlot



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
ΔT_sc = 5                   # [-]
ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
refrigerant = "R1234ze(Z)"  # [-]



# Visualization method
# ====================
visualization_method = "CoolProp"



# Cycle Evaluation
# ================
PR_bisection_range = [1, 50]
PR_guess = 25
ΔT_sc_calculated = 0
while not math.isclose(ΔT_sc_calculated, ΔT_sc, rel_tol=1e-3):
    # evaporation properties
    T_ev = T_h_in - ΔT_pp_1 - ΔT_sh
    p_ev = PropsSI("P", "T", T_ev, "Q", 0, f"REFPROP::{refrigerant}")

    # compressor inlet properties
    p_ref_1 = p_ev
    T_ref_1 = T_h_in - ΔT_pp_1
    h_ref_1 = PropsSI("H", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    s_ref_1 = PropsSI("S", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}")
    Q_ref_1 = PropsSI("Q", "T", T_ref_1, "P", p_ref_1, f"REFPROP::{refrigerant}") # vapour quality

    # condenser inlet properties
    p_ref_2 = PR_guess*p_ref_1
    h_ref_2_is = PropsSI("H", "P", p_ref_2, "S", s_ref_1, f"REFPROP::{refrigerant}")
    h_ref_2 = (h_ref_2_is - h_ref_1)/η_compr + h_ref_1
    T_ref_2 = PropsSI("T", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}")
    Q_ref_2 = PropsSI("Q", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}") # vapour quality
    s_ref_2 = PropsSI("S", "P", p_ref_2, "H", h_ref_2, f"REFPROP::{refrigerant}") # for completeness

    # Turbine inlet properties
    T_ref_3 = T_c_in + ΔT_pp_3
    p_ref_3 = p_ref_2
    h_ref_3 = PropsSI("H", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    s_ref_3 = PropsSI("S", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    Q_ref_3 = PropsSI("Q", "T", T_ref_3, "P", p_ref_3, f"REFPROP::{refrigerant}")
    T_cond = PropsSI("T", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")

    # compute ΔT_sc and check against the requirement
    ΔT_sc_calculated = T_cond - T_ref_3

    # bisection method logic
    if (ΔT_sc_calculated - ΔT_sc)/2 > 0: 
        PR_bisection_range[1] = PR_guess
    else:
        PR_bisection_range[0] = PR_guess
    PR_guess = sum(PR_bisection_range)/len(PR_bisection_range)
    
# Evaporator inlet properties
p_ref_4 = p_ref_1
h_ref_4_is = PropsSI("H", "P", p_ref_4, "S", s_ref_3, f"REFPROP::{refrigerant}")
h_ref_4 = (h_ref_4_is - h_ref_3)*η_turb + h_ref_3
T_ref_4 = PropsSI("T", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}")
Q_ref_4 = PropsSI("Q", "P", p_ref_4, "H", h_ref_4, f"REFPROP::{refrigerant}") # vapour quality
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

# Cycle performance 
Ẇ_turb = ṁ_ref*(cp_ref_3+cp_ref_4)/2*(h_ref_3 - h_ref_4)
Ẇ_comp = ṁ_ref*(cp_ref_1+cp_ref_2)/2*(h_ref_2 - h_ref_1)
Q_out = ṁ_c*cp_c*(T_c_out - T_c_in)
Q_in = ṁ_h*cp_h*(T_h_in - T_h_out)
COP = Q_out/((Ẇ_comp - Ẇ_turb*ɳ_shaft))



# Fill out points for plotting
# ============================
s_ref_23_v_inflection = PropsSI("S", "P", p_ref_2, "Q", 1, f"REFPROP::{refrigerant}")
s_ref_23_l_inflection = PropsSI("S", "P", p_ref_2, "Q", 0, f"REFPROP::{refrigerant}")
s_ref_41_v_inflection = PropsSI("S", "P", p_ref_1, "Q", 1, f"REFPROP::{refrigerant}")
# Compute isobar segments to give them a nice curve
def isobar_segment(s_end, s_start, p):
    s_range = np.linspace(s_end, s_start, num = 30)
    T_range = np.zeros(shape = (1, 30))
    for i, s in enumerate(s_range):
        T_range[0, i] = PropsSI("T", "P", p, "S", s, f"REFPROP::{refrigerant}")
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
# ================
s_ref_lst = list(itertools.chain(
    [s_ref_1], [s_ref_2], s_ref_23_v, [s_ref_23_v_inflection], [s_ref_23_l_inflection], s_ref_23_l, [s_ref_3], 
    [s_ref_4], s_ref_41_l, [s_ref_41_l_inflection], [s_ref_41_v_inflection], s_ref_41_v
    ))
T_ref_lst = list(itertools.chain(
    [T_ref_1], [T_ref_2], T_ref_23_v, [T_cond], [T_cond], T_ref_23_l, [T_ref_3], 
    [T_ref_4], T_ref_41_l, [T_ev], [T_ev], T_ref_41_v
))



# Property dictionary
# ===================
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



# Coolprop integrated visualization
# =================================
if visualization_method == "CoolProp":
    import warnings
    warnings.filterwarnings("ignore")
    myPlot = PropertyPlot(f"REFPROP::{refrigerant}", 'TS', unit_system='SI', tp_limits = 'ACHP')
    myPlot.calc_isolines(CoolProp.iQ, num=15) # num indicating the number of isolines you want to display
    myPlot.calc_isolines(CoolProp.iP, num=30)
    # plot.calc_isolines(CoolProp.iP, iso_range=[1,50], num=10, rounding=True)

    # add cycle
    plt.scatter(HP_cycle_plot_major["s"], HP_cycle_plot_major["T"], color='orange', marker = 'o', s = 5, zorder = 2)
    plt.plot(HP_cycle_plot_minor["s"], HP_cycle_plot_minor["T"], color='green', zorder = 1)

    # add labels
    myPlot.xlabel("Specific entropy, [J/kg/K]")
    myPlot.ylabel("Temperature, [K]")

    # add cycle performance
    textstr = '\n'.join((
    r'$COP=%.1f$' % (COP, ),
    r'$\dot{W}_turb=%.1f$' % (Ẇ_turb, ),
    r'$\dot{W}_compr=%.1f$' % (, )))
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # add title and save
    plt.title("Conceptual HP Cycle   -   R1234ze(Z)")
    myPlot.savefig("Conceptual HP Cycle - R1234ze(Z).pdf", dpi=1000, bbox_inches="tight")



# NiceProp Visualization
# ======================