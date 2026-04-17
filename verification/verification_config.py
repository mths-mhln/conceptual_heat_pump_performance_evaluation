import numpy as np



# Analysis Type
# =============
analysis_type = "single_configuration"



# Cycle Specifications (thermodynamic inputs)
# ===========================================
T_h_in = 353.15             # [K] - 80 degC
T_c_in = 287.15             # [K] - 15 degC (outside temp)
ṁ_h = 42                    # [kg/s] - BOTE calculation using typical refrigerant mass flow
cp_h = 1885                 # [J/kg/K] - steam at 250 degrees
ṁ_c = 40                    # [kg/s] - arbitrarily chosen
cp_c = 1006                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
η_turb = 0                  # [-] - enforce an isenthalpic expansion given MTW code is unable to simulate turbine expansion processes. 
η_compr = 0.78              # [-] - from compressor maps
ΔT_pp_1 = 10                # [K] - pinch point 1
ΔT_pp_3 = 10                # [K] - pinch point 3
ΔT_pp_4 = 10                # [K] - pinch point 4
ΔT_sh = 5                   # [K] - superheat
ɳ_shaft = 1.00              # [-] - not accounted for by MTW code
near_critical_buffer_ratio = 1.05  # [-] thermodynamic properties near the critical point can be unreliable, hence I force the
                                    # optimizer to stay away from this region
ΔT_pp_min_evap = 10
ΔT_pp_min_cond = 10
Q_out_req = 200000

cycle_config = {
    "T_h_in": T_h_in,
    "T_c_in": T_c_in,
    "ṁ_c": ṁ_c,
    "cp_c": cp_c,
    "ṁ_h": ṁ_h,
    "cp_h": cp_h,
    "η_turb": η_turb,
    "η_compr": η_compr,
    "ΔT_pp_1": ΔT_pp_1,
    "ΔT_pp_3": ΔT_pp_3,
    "ΔT_pp_4": ΔT_pp_4,
    "ΔT_sh": ΔT_sh,
    "ɳ_shaft": ɳ_shaft,
    "near_critical_buffer_ratio": near_critical_buffer_ratio,
    "ΔT_pp_min_evap": ΔT_pp_min_evap,
    "ΔT_pp_min_cond": ΔT_pp_min_cond,
    "Q_out_req": Q_out_req    
}



# General / runtime / plotting settings
# =====================================
if analysis_type == "single_configuration":
    general_config = {
        "analysis_type": analysis_type,
        "visualization_method": "CoolProp",  # "CoolProp" or "NiceProp" (NiceProp not implemented)
        "resolution": "low",                 # "low" or "high"
        "ignore_coolprop_warnings": True,
        "ts_margin_s_left": 0.2,
        "ts_margin_s_right": 0.2,
        "ts_margin_T_bot": 0.2,
        "ts_margin_T_top": 0.15,
        "ph_margin_h_left": 0.3,
        "ph_margin_h_right": 0.15,
        "ph_margin_p_bot": 0.1,
        "ph_margin_p_top": 0.2,
        "de_popsize": 50,          # more individuals = better global coverage
        "de_maxiter": 10000,         # ← this is your "many more iterations"
        "de_tol": 1e-9,
        "de_strategy": "best1bin",
        "min_mref_threshold": 0.005,   # adjust to your typical system size (kg/s)
    }
