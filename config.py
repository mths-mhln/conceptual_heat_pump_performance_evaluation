# Analysis Type
# =============
analysis_type = "substance_thermodynamic_diagrams"  # "single_configuration", "COP_vs_eff_investigation", or "substance_thermodynamic_diagrams"
# single_configuration       | evaluates conceptual heat pump cycle according to specifications
# COP_vs_eff_investigation   | evaluates COP variation for different values of turbine and compressor efficiencies
# substance_thermodynamic_diagrams | generates empty TS/PH diagrams for selected substances



# refrigerant selection - depending on specification different specifications are necessary
refrigerant = "R1234ze(E)"  # "R1234ze(Z)", "MM", "R1234ze(E)", "R1233zd(E)", "CO2"



# Substances to plot in "substance_thermodynamic_diagrams" analysis type
substances_to_plot = ["R1234ze(E)"]
show_diagrams = True
show_spinodal = False 



# Cycle Specifications (thermodynamic inputs)
# ===========================================
if refrigerant != "CO2":
    T_h_in = 353.15             # [K] - 80 degC
    T_c_in = 287.15             # [K] - 15 degC (outside temp)
    ṁ_h = 40                    # [kg/s] - BOTE calculation using typical refrigerant mass flow
    cp_h = 1885                 # [J/kg/K] - steam at 250 degrees
    ṁ_c = 40                    # [kg/s] - arbitrarily chosen
    cp_c = 1006                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
    η_turb = 0.80               # [-] - from turbine maps
    η_compr = 0.78              # [-] - from compressor maps
    ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
    near_critical_buffer_ratio = 1.05   # [-] thermodynamic properties near the critical point can be unreliable, hence I force the 
                                        # optimizer to stay away from this region
    ΔT_pp_min_evap = 10
    ΔT_pp_min_cond = 10
    Q_out_req = 200000

# if refrigerant != "CO2":
#     T_h_in = 320             # [K] - 80 degC  353.15
#     T_c_in = 290             # [K] - 15 degC (outside temp)
#     ṁ_h = 40                    # [kg/s] - BOTE calculation using typical refrigerant mass flow
#     cp_h = 1885                 # [J/kg/K] - steam at 250 degrees
#     ṁ_c = 40                    # [kg/s] - arbitrarily chosen
#     cp_c = 1006                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
#     η_turb = 0.87               # [-] - from turbine maps
#     η_compr = 0.78              # [-] - from compressor maps
#     ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
#     near_critical_buffer_ratio = 1.05  # [-] thermodynamic properties near the critical point can be unreliable, hence I force the
#                                         # optimizer to stay away from this region
#     ΔT_pp_min_evap = 10
#     ΔT_pp_min_cond = 10
#     Q_out_req = 200000

if refrigerant == "CO2":
    T_h_in = 320             # [K] - 80 degC  353.15
    T_c_in = 290             # [K] - 15 degC (outside temp)
    ṁ_h = 40                    # [kg/s] - BOTE calculation using typical refrigerant mass flow
    cp_h = 1885                 # [J/kg/K] - steam at 250 degrees
    ṁ_c = 80                    # [kg/s] - arbitrarily chosen
    cp_c = 1006                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
    η_turb = 0.87               # [-] - from turbine maps
    η_compr = 0.78              # [-] - from compressor maps
    ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
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
    "ɳ_shaft": ɳ_shaft,
    "near_critical_buffer_ratio": near_critical_buffer_ratio,
    "ΔT_pp_min_evap": ΔT_pp_min_evap,
    "ΔT_pp_min_cond": ΔT_pp_min_cond,
    "refrigerant": refrigerant,
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
        "opt_progress_fig_height": 3,
        "ts_margin_s_left": 0.2,
        "ts_margin_s_right": 0.2,
        "ts_margin_T_bot": 0.25,
        "ts_margin_T_top": 0.35, # 0.15
        "ph_margin_h_left": 0.3,
        "ph_margin_h_right": 0.15,
        "ph_margin_p_bot": 0.1,
        "ph_margin_p_top": 0.2,
    }
elif analysis_type == "COP_vs_eff_investigation":
    general_config = {
        "analysis_type": analysis_type,
        "visualization_method": "CoolProp",
        "resolution": "low",                 # "low" or "high"
        "ignore_coolprop_warnings": True,
        # Behavior when a complete COP sweep file already exists:
        # "prompt" asks interactively, "abort" skips run, "recompute" restarts automatically.
        "existing_complete_sweep_action": "prompt",
    }
elif analysis_type == "substance_thermodynamic_diagrams":
    general_config = {
        "analysis_type": analysis_type,
        "visualization_method": "CoolProp",
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
        "substances_to_plot": substances_to_plot,
        "show_diagrams": show_diagrams,
        "show_spinodal": show_spinodal,
    }
else:
    raise ValueError(
        "Invalid analysis_type. Use 'single_configuration', 'COP_vs_eff_investigation', or 'substance_thermodynamic_diagrams'."
    )
 