# python script that serves multiple purposes:
#  1) in the first instance, this script serves as an example
#     of how the 'thermo_props.py' and 'hp_simulator.py' modules
#     can be used outside of the pocketTHERM environment to 
#     model and simulate heat pump systems
#  2) secondly, this script demonstrates how to use the same 
#     'hp_simulator.py' module, but rely on CoolProp for 
#     carrying out thermodynamic property calculations, rather
#     than using the Peng-Robinson model
#  3) conduct a direct comparison between cycle predictions 
#     obtained using the Peng-Robinson model and using CoolProp
#
# Note:
#   in order for this code to work, the CoolProp package needs
#   to be imported within the 'thermo_props.py' module. 
#
# Martin T. White, University of Sussex, 05/06/2024

import os
import sys
cwd = os.getcwd()
sys.path.append(f'{cwd}/verification/SOFTX-D-24-00232-main/python_modules')
import thermo_props
import hp_simulator as hp

def MTW_HP_calculator(MTW_cycle_config):
    # heat exchanger discretisation:
    n_hxc = [2,2,2,5,2,2,2,2]

    # initialize result dictionaries:
    cycle_performance_params = {}
    cycle_state_params = {}

    # fluid properties:
    fluids = [ "R1233zd(E)", "n-pentane",     "MM" ]		# fluid names
    Tc     = [  439.6,        469.7,        518.7     ]	# critical temperature, K
    Pc     = [  3.6237e6,     3.3675e6,     1.93113e6 ]	# critical pressure, Pa
    om     = [  0.3025,       0.2510,       0.4180    ]	# acentric factor
    wm     = [  130.4962,     72.1488,      162.3768  ]	# molecular weight, g/mol
    cp1    = [  33.3490,      12.9055,      64.6367   ]	# polynomial coefficients: 
    cp2    = [  0.2823,       0.3906,       0.6695    ]     #   c_p(T) = cp1 + ... 
    cp3    = [ -0.1523e-3,   -0.1036e-3,   -0.2895e-3 ]     #     cp2*T + cp3*T**2

    # heat source and heat sink parameters:
    cld = [MTW_cycle_config["T_c_in"], MTW_cycle_config["ṁ_c"], MTW_cycle_config["cp_c"]]  # inlet temp. K, flow rate kg/s, specific heat J/kg K
    hot = [MTW_cycle_config["T_h_in"], MTW_cycle_config["ṁ_h"], MTW_cycle_config["cp_h"]]  # inlet temp. K, flow rate kg/s, specific heat J/kg K

    # fixed ORC parameters:
    Tevap  = MTW_cycle_config["T_evap"]	# evaporation temperature, K
    dtsh   = MTW_cycle_config["dtsh"]	# compressore superheating, K
    dtsc   = MTW_cycle_config["dtsc"]      # amount of subcooling before expansion valve, K
    dt_ihe = 0	# internal heat exchanger temperature difference, K
    Tco    = MTW_cycle_config["T_co"]	# heat-source outlet temperature, K

    # isentropic efficiency (compressor):
    eta = [MTW_cycle_config["eta"]]

    # number of points in array:
    n = 25

    # get index of current refrigerant:
    i = fluids.index(MTW_cycle_config["refrigerant"])

    # setup Peng-Robinson fluid:
    cp = [ cp1[i], cp2[i], cp3[i] ] 
    PR = thermo_props.pr_fluid(MTW_cycle_config["refrigerant"],Tc[i],Pc[i],om[i],cp,300,0.01,wm[i])

    # setup CoolProp fluid:
    CP = thermo_props.coolprop_fluid(MTW_cycle_config["refrigerant"],'REFPROP')

    # create array of inputs:
    x = [Tevap,dtsh,MTW_cycle_config["PR"],dtsc,dt_ihe,Tco]

    # run model using Peng-Robinson model:
    [props,cycle_out,_,pp,*_] = hp.simulate_HP(PR,x,eta,hot,cld,n_hxc,0)

    # run model using CoolProp:
    try:
        [props,cycle_out,_,pp,*_] = hp.simulate_HP(CP,x,eta,hot,cld,n_hxc,0)
        x_i = [Tevap,dtsh,MTW_cycle_config["PR"],dtsc,dt_ihe,Tco]
        [props_i,cycle_out_i,_,pp_i,*_] = hp.simulate_HP(CP,x_i,eta,hot,cld,n_hxc,0)
    except:
        pass
    # initialize subdictionary
    # print(MTW_cycle_config["refrigerant"])
    cycle_performance_params[MTW_cycle_config["refrigerant"]] = {}
    for cycle_performance_param, value in zip(["COP_isenth", "Ẇ_comp", "Q_out", "Q_in", "Q_rec", "ṁ_ref",], cycle_out):     #COP_turb = COP_isenth given the config. 
        cycle_performance_params[MTW_cycle_config["refrigerant"]][cycle_performance_param] = value
    del cycle_performance_params[MTW_cycle_config["refrigerant"]]["Q_rec"]
    cycle_state_params[MTW_cycle_config["refrigerant"]] = props
    # print(MTW_cycle_config["refrigerant"])
    # cycle output: cop, Wc, Qh, Qc, Qr, mdot
    # print(cycle_out_i)
    # property outputs. Rows are different properties (T, P, h, s, rho) and columns different thermodynamic states.
    # print(props_i)
    # the below prints: [a pp that does not matter, pp_2, pp_3, pp_4, pp_1]
    # print(pp_i)

    return cycle_performance_params, cycle_state_params
    


# plt.plot(Prat,COP_CP[0,:],'k-',label='R1233zd(E) - HEOS')
# plt.plot(Prat,COP_CP[1,:],'r-',label='n-pentane - HEOS')
# plt.plot(Prat,COP_CP[2,:],'b-',label='MM - HEOS')
# plt.plot(Prat,COP_PR[0,:],'k--',label='R1233zd(E) - PR')
# plt.plot(Prat,COP_PR[1,:],'r--',label='n-pentane - PR')
# plt.plot(Prat,COP_PR[2,:],'b--',label='MM - PR')
# plt.legend()
# plt.xlabel('Pressure ratio')
# plt.ylabel('Coefficient of performance')
# plt.show()

# End of file

































































































# # python script that serves multiple purposes:
# #  1) in the first instance, this script serves as an example
# #     of how the 'thermo_props.py' and 'hp_simulator.py' modules
# #     can be used outside of the pocketTHERM environment to 
# #     model and simulate heat pump systems
# #  2) secondly, this script demonstrates how to use the same 
# #     'hp_simulator.py' module, but rely on CoolProp for 
# #     carrying out thermodynamic property calculations, rather
# #     than using the Peng-Robinson model
# #  3) conduct a direct comparison between cycle predictions 
# #     obtained using the Peng-Robinson model and using CoolProp
# #
# # Note:
# #   in order for this code to work, the CoolProp package needs
# #   to be imported within the 'thermo_props.py' module. 
# #
# # Martin T. White, University of Sussex, 05/06/2024

# from itertools import cycle
# import sys
# sys.path.append('d:\\nexus\\02_learning\\00_university_education\\04_MSc_TUDelft\\05_thesis_nexus\\05_conceptual_heat_pump_performance_evaluation\\verification\\SOFTX-D-24-00232-main\\python_modules')
# # had to extend this because I think using relative paths made python (installed on C drive) look further on C drive rather than on D drive where the files are located.
# import thermo_props
# import hp_simulator as hp
# import matplotlib.pyplot as plt
# import numpy as np

# def MTW_HP_calculator(MTW_cycle_config):
#     # heat exchanger discretisation:
#     n_hxc = [2,2,2,5,2,2,2,2]

#     # initialize result dictionaries:
#     cycle_performance_params = {}
#     cycle_state_params = {}

#     # fluid properties:
#     fluids = [ "R1233zd(E)", "Pentane",     "MM" ]		# fluid names
#     Tc     = [  439.6,        469.7,        518.7     ]	# critical temperature, K
#     Pc     = [  3.6237e6,     3.3675e6,     1.93113e6 ]	# critical pressure, Pa
#     om     = [  0.3025,       0.2510,       0.4180    ]	# acentric factor
#     wm     = [  130.4962,     72.1488,      162.3768  ]	# molecular weight, g/mol
#     cp1    = [  33.3490,      12.9055,      64.6367   ]	# polynomial coefficients: 
#     cp2    = [  0.2823,       0.3906,       0.6695    ]     #   c_p(T) = cp1 + ... 
#     cp3    = [ -0.1523e-3,   -0.1036e-3,   -0.2895e-3 ]     #     cp2*T + cp3*T**2


#     # heat source and heat sink parameters:
#     cld = [323,1,4200]  # inlet temp. K, flow rate kg/s, specific heat J/kg K
#     hot = [373,1,4200]  # inlet temp. K, flow rate kg/s, specific heat J/kg K

#     # fixed ORC parameters:
#     Tevap  = 313	# evaporation temperature, K
#     dtsh   = 5	# compressore superheating, K
#     dtsc   = 5      # amount of subcooling before expansion valve, K
#     dt_ihe = 0	# internal heat exchanger temperature difference, K
#     Tco    = 318	# heat-source outlet temperature, K

#     # isentropic efficiency (compressor):
#     eta = [0.7]

#     # number of points in array:
#     n = 25

#     # pressure ratio array:
#     Prat = np.linspace(2,10,n)

#     # initialise arrays for data storage:
#     COP_PR = np.zeros([3,n])
#     COP_CP = np.zeros([3,n])

#     # run parametric study:
#     for i in range(len(fluids)):
#         # print(fluids[i])
#         # setup Peng-Robinson fluid:
#         cp = [ cp1[i], cp2[i], cp3[i] ] 
#         PR = thermo_props.pr_fluid(fluids[i],Tc[i],Pc[i],om[i],cp,300,0.01,wm[i])

#         # setup CoolProp fluid:
#         CP = thermo_props.coolprop_fluid(fluids[i],'REFPROP')

#         for j in range(len(Prat)):

#             # create array of inputs:
#             x = [Tevap,dtsh,12.822139469035617385218755537047,dtsc,dt_ihe,Tco]

#             # run model using Peng-Robinson model:
#             [props,cycle_out,_,pp,*_] = hp.simulate_HP(PR,x,eta,hot,cld,n_hxc,0)
#             COP_PR[i,j] = cycle_out[0]

#             # run model using CoolProp:
#             try:
#                 [props,cycle_out,_,pp,*_] = hp.simulate_HP(CP,x,eta,hot,cld,n_hxc,0)
#                 x_i = [Tevap,dtsh,12.822139469035617385218755537047,dtsc,dt_ihe,Tco]
#                 [props_i,cycle_out_i,_,pp_i,*_] = hp.simulate_HP(CP,x_i,eta,hot,cld,n_hxc,0)
#                 COP_CP[i,j] = cycle_out[0]
#             except:
#                 COP_CP[i,j] = np.nan
#             # initialize subdictionary
#             cycle_performance_params[fluids[i]] = {}
#             for cycle_performance_param, value in zip(["COP_isenth", "ṁ_ref", "Ẇ_comp", "Q_in", "Q_out"], cycle_out):     #COP_turb = COP_isenth given the config. 
#                 cycle_performance_params[fluids[i]][cycle_performance_param] = value
#             cycle_state_params[fluids[i]] = props
#         # print(fluids[i])
#         # cycle output: cop, Wc, Qh, Qc, Qr, mdot
#         # print(cycle_out_i)
#         # property outputs. Rows are different properties (T, P, h, s, rho) and columns different thermodynamic states.
#         # print(props_i)
#         # the below prints: [a pp that does not matter, pp_2, pp_3, pp_4, pp_1]
#         # print(pp_i)
#         return cycle_performance_params, cycle_state_params
    


# # plt.plot(Prat,COP_CP[0,:],'k-',label='R1233zd(E) - HEOS')
# # plt.plot(Prat,COP_CP[1,:],'r-',label='n-pentane - HEOS')
# # plt.plot(Prat,COP_CP[2,:],'b-',label='MM - HEOS')
# # plt.plot(Prat,COP_PR[0,:],'k--',label='R1233zd(E) - PR')
# # plt.plot(Prat,COP_PR[1,:],'r--',label='n-pentane - PR')
# # plt.plot(Prat,COP_PR[2,:],'b--',label='MM - PR')
# # plt.legend()
# # plt.xlabel('Pressure ratio')
# # plt.ylabel('Coefficient of performance')
# # plt.show()

# # End of file

# if __name__ == "__main__":
#     print(MTW_HP_calculator(None))