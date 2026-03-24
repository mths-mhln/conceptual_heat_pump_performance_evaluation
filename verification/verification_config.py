import numpy as np



# Analysis Type
# =============
analysis_type = "single_configuration"


# Cycle Specifications (thermodynamic inputs)
# ===========================================
T_c_in = 373                # [K] - 80 degC
T_h_in = 323                # [K] - 15 degC (outside temp)
ṁ_c = 1                     # [kg/s] - BOTE calculation using typical refrigerant mass flow
cp_c = 4200                 # [J/kg/K] - steam at 250 degrees
ṁ_h = 1                     # [kg/s] - arbitrarily chosen
cp_h = 4200                 # [J/kg/K] - air at 30 degrees and atmospheric pressure
η_turb = 0               # [-] - Martin T. White's case has expansion valve: isenthalpic expansion; so this is just for aesthetics
η_compr = 0.70              # [-] - from compressor maps
ΔT_pp_1 = 5.                # [K] - pinch point 1
ΔT_pp_4 = 5.                # [K] - pinch point 4
ΔT_sh = 5                   # [K] - superheat
ɳ_shaft = 1              # [-] - turbine/compressor shaft connection efficiency

cycle_config_R1233zd_E = {
    "T_c_in": T_c_in,
    "T_h_in": T_h_in,
    "ṁ_c": ṁ_c,
    "cp_c": cp_c,
    "ṁ_h": ṁ_h,
    "cp_h": cp_h,
    "η_turb": η_turb,
    "η_compr": η_compr,
    "ΔT_pp_1": ΔT_pp_1,
    "ΔT_pp_2": 39.01426188,
    "ΔT_pp_3": 45.44755455,
    "ΔT_pp_4": ΔT_pp_4,
    "ΔT_sh": ΔT_sh,
    "ɳ_shaft": ɳ_shaft,
    "refrigerant": "R1233zd(E)",
}

cycle_config_n_pentane = {
    "T_c_in": T_c_in,
    "T_h_in": T_h_in,
    "ṁ_c": ṁ_c,
    "cp_c": cp_c,
    "ṁ_h": ṁ_h,
    "cp_h": cp_h,
    "η_turb": η_turb,
    "η_compr": η_compr,
    "ΔT_pp_1": ΔT_pp_1,
    "ΔT_pp_2": 34.33426735,
    "ΔT_pp_3": 40.85796314,
    "ΔT_pp_4": ΔT_pp_4,
    "ΔT_sh": ΔT_sh,
    "ɳ_shaft": ɳ_shaft,
    "refrigerant": "n-pentane",
}

cycle_config_MM = {
    "T_c_in": T_c_in,
    "T_h_in": T_h_in,
    "ṁ_c": ṁ_c,
    "cp_c": cp_c,
    "ṁ_h": ṁ_h,
    "cp_h": cp_h,
    "η_turb": η_turb,
    "η_compr": η_compr,
    "ΔT_pp_1": ΔT_pp_1,
    "ΔT_pp_2": 5.56576495,
    "ΔT_pp_3": 8.58320032,
    "ΔT_pp_4": ΔT_pp_4,
    "ΔT_sh": ΔT_sh,
    "ɳ_shaft": ɳ_shaft,
    "refrigerant": "MM",
}


# resulting thermodynamic cycles
# ==============================
MTW_cycle = {
    "R1233zd(E)": np.array(
    [[3.18000000e+02, 4.31125113e+02, 4.23447555e+02, 4.23447555e+02, 4.18447555e+02, 4.18447555e+02, 3.13000000e+02, 3.13000000e+02, 3.18000000e+02, 3.18000000e+02],
 [2.14518277e+05, 2.75058327e+06, 2.75058327e+06, 2.75058327e+06, 2.75058327e+06, 2.75058327e+06, 2.14518277e+05, 2.14518277e+05, 2.14518277e+05, 2.14518277e+05],
 [4.35686014e+05, 5.00671644e+05, 4.86122910e+05, 4.06602279e+05, 3.96486107e+05, 3.96486107e+05, 3.96486107e+05, 4.31343897e+05, 4.35686014e+05, 4.35686014e+05],
 [1.76306061e+03, 1.80880821e+03, 1.77474318e+03, 1.58694984e+03, 1.56292137e+03, 1.56292137e+03, 1.63793103e+03, 1.74929777e+03, 1.76306061e+03, 1.76306061e+03],
 [1.13752093e+01, 1.67792437e+02, 1.93247486e+02, 8.03083541e+02, 8.50131838e+02, 8.50131838e+02, 1.43089622e+01, 1.16112723e+01, 1.13752093e+01, 1.13752093e+01]]
),
"n-pentane": np.array(
    [[3.18000000e+02, 4.18857963e+02, 4.18857963e+02, 4.18857963e+02, 4.13857963e+02, 4.13857963e+02, 3.13000000e+02, 3.13000000e+02, 3.18000000e+02, 3.18000000e+02],
 [1.15111251e+05, 1.47597252e+06, 1.47597252e+06, 1.47597252e+06, 1.47597252e+06, 1.47597252e+06, 1.15111251e+05, 1.15111251e+05, 1.15111251e+05, 1.15111251e+05],
 [3.72594600e+05, 4.92584187e+05, 5.27837336e+05, 2.96382246e+05, 2.80630174e+05, 2.80630174e+05, 2.80630174e+05, 3.63652792e+05, 3.72594600e+05, 3.72594600e+05],
 [1.19027478e+03, 1.27621531e+03, 1.36038024e+03, 8.07794135e+02, 7.69961645e+02, 7.69961645e+02, 8.96684900e+02, 1.16193288e+03, 1.19027478e+03, 1.19027478e+03],
 [3.29102540e+00, 5.00836829e+01, 4.31588083e+01, 4.68039361e+02, 4.77797683e+02, 4.77797683e+02, 4.37045267e+00, 3.35294335e+00, 3.29102540e+00, 3.29102540e+00]]
),
"MM": np.array(
    [[3.18000000e+02, 3.86583200e+02, 3.86583200e+02, 3.86583200e+02, 3.81583200e+02, 3.81583200e+02, 3.13000000e+02, 3.13000000e+02, 3.18000000e+02, 3.18000000e+02],
 [1.13286167e+04, 1.45257104e+05, 1.45257104e+05, 1.45257104e+05, 1.45257104e+05, 1.45257104e+05, 1.13286167e+04, 1.13286167e+04, 1.13286167e+04, 1.13286167e+04],
 [1.06872724e+05, 1.61179358e+05, 2.13356142e+05, 2.76273229e+04, 1.68846598e+04, 1.68846598e+04, 1.68846598e+04, 9.92281401e+04, 1.06872724e+05, 1.06872724e+05],
 [3.75834410e+02, 4.17977965e+02, 5.52947058e+02, 7.25101978e+01, 4.45404070e+01, 4.45404070e+01, 8.85259233e+01, 3.51604135e+02, 3.75834410e+02, 3.75834410e+02],
 [7.04536656e-01, 1.10611327e+01, 7.99141814e+00, 6.58887103e+02, 6.65205337e+02, 6.65205337e+02, 1.13775597e+00, 7.16400376e-01, 7.04536656e-01, 7.04536656e-01]]
)}

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
    }

# Cycle Calculation results according to Martin T. White HP calculation tool
# https://github.com/ElsevierSoftwareX/SOFTX-D-24-00232
# paper: https://www.sciencedirect.com/science/article/pii/S2352711024001778?via%3Dihub
# cop, Wc, Qh, Qc, Qr, mdot
# MTW = Martin T. White
MTW_cycle_performances = {
    "R1233zd(E)": [1.60320886, 34.81381253*1e3, 21.*1e3, 55.81381253, 0.53571555], # Martin T. White also has 
    "n-pentane": [1.76643672, 27.39952229*1e3, 21.*1e3, 48.39952229, 0.22834917], 
    "MM": [2.65703632, 12.67322858*1e3, 21.*1e3, 33.67322858, 0.23336428]
}


MTW_cycle_performances = {
    "R1233zd(E)": {
        "COP_isenth": 1.60320886,
        "ṁ_ref": 0.53571555,
        "Ẇ_comp": 34.81381253*1e3,
        "Q_in": 21.*1e3,
        "Q_out": 55.81381253*1e3
    },
    "n-pentane": {
        "COP_isenth": 1.76643672,
        "ṁ_ref": 0.22834917,
        "Ẇ_comp": 27.39952229*1e3,
        "Q_in": 21.*1e3,
        "Q_out": 48.39952229*1e3
    },
    "MM": {
        "COP_isenth": 2.65703632,
        "ṁ_ref": 0.23336428,
        "Ẇ_comp": 12.67322858*1e3,
        "Q_in": 21.*1e3,
        "Q_out": 33.67322858*1e3
    }
}