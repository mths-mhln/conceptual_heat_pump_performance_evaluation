import matplotlib.pyplot as plt
import numpy as np

from CHPC.visualization import make_cycle_plots, make_COP_vs_eff_plot, make_empty_thdy_plot



# T-s, p-h and optimizer progress diagrams of a CHPC analysis. The results file name carries the calculation 
# method if it is not full_optimization, e.g. CHPC_R1234ze(E)_three_temp_fix.npz. The margins around the cycle and 
# the size of the optimizer progress figure can be adapted, e.g. ts_margins={"T_top": 0.5}, see visualization.py.
make_cycle_plots("results/heat_pump_thermodynamic_diagrams/CO2/data/CHPC_CO2.npz")
plt.show()



# COP maps (contour plot and 3D surface) of a COP vs efficiency investigation
# data = np.load("results/COP_investigations/R1234ze(E)/data/COP_vs_eff_R1234ze(E)_COP_turb.npz")
# make_COP_vs_eff_plot(data["X"], data["Y"], data["Z"], refrigerant="R1234ze(E)")
# plt.show()



# Empty T-s and p-h diagrams of a substance, optionally with its spinodal curves
# make_empty_thdy_plot("TS", "R1233zd(E)", show_spinodal=False)
# make_empty_thdy_plot("PH", "R1233zd(E)", show_spinodal=False)
# plt.show()



# Thesis figures: effect of the compressor and turbine efficiency on the cycle of a CHPC config file, 
# and the COP maps of two refrigerants side by side
# from CHPC.effect_of_eff_on_COP import plot_single_cycle_custom_ts
# from CHPC.COP_vs_eff_comparison_plots import plot_side_by_side_contours
# plot_single_cycle_custom_ts("config/R1234ze(E)/CHPC.ini")
# plot_side_by_side_contours(refrigerants=("CO2", "R1234ze(E)"))
