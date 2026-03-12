# config.py
# =========
# All cycle specification parameters, fluid selection, and visualization settings.
 
 
# Cycle specification
# ===================
T_c_in = 353.15             # [K] - 80 degC
T_h_in = 287.15             # [K] - 15 degC (outside temp)
ṁ_c = 42                    # [kg/s] - BOTE calculation using typical refrigerant mass flow
cp_c = 1885                 # [J/kg/K] - steam at 250 degrees
ṁ_h = 40                    # [kg/s] - arbitrarily chosen
cp_h = 1006                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
η_turb = 0.87               # [-] - from turbine maps
η_compr = 0.78              # [-] - from compressor maps
ΔT_pp_1 = 10                # [K] - pinch point 1
ΔT_pp_2 = 10                # [K] - pinch point 2
ΔT_pp_3 = 10                # [K] - pinch point 3
ΔT_pp_4 = 10                # [K] - pinch point 4
ΔT_sh = 5                   # [K] - superheat
ɳ_shaft = 0.98              # [-] - turbine/compressor shaft connection efficiency
refrigerant = "R1234ze(Z)"  # [-] - "R1234ze(Z)", "MM" "R1234ze(E)"
 
 
# Visualization settings
# ======================
visualization_method = "CoolProp"   # "CoolProp" or "NiceProp"  (NiceProp not implemented)
resolution = "high"                  # "low" or "high"  (affects runtime)

# Diagram margin scaling - you can adapt for aesthetics of the plot
ts_margin_s_left  = 0.2   # fraction of s-span added left
ts_margin_s_right = 0.2   # fraction of s-span added right
ts_margin_T_bot   = 0.2   # fraction of T-span added bottom
ts_margin_T_top   = 0.15  # fraction of T-span added top - for R1234ze(Z)/MM best is 0.15, for R1234ze(E) best is 0.25

ph_margin_h_left  = 0.3   # fraction of h-span added left
ph_margin_h_right = 0.15  # fraction of h-span added right
ph_margin_p_bot   = 0.1   # log-decades added below p_lo
ph_margin_p_top   = 0.2   # log-decades added above p_hi
 

