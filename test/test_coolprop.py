import CoolProp.CoolProp as CP
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from CoolProp.CoolProp import PropsSI

# print(CP.PropsSI("H", "S", 1200, "T", 320, "REFPROP::R1234ze(E)"))

# # 1732.4058913958568


# print(CP.PropsSI("A", "S", 1260, "T", 260, "REFPROP::R1234ze(E)"))
# print(PropsSI("A", "S", 1260, "T", 260, "REFPROP::R1234ze(E)"))


# print(PropsSI('T', 'D', 519.20049536, 'U', 259012.115143168, "HEOS::R1234ze(E)"))
print(PropsSI('P', 'D', 519.20049536+0.1, 'U', 259012.115143168, "HEOS::R1234ze(E)"))
print(PropsSI('P', 'D', 519.20049536-0.1, 'U', 259012.115143168, "HEOS::R1234ze(E)"))
# print(PropsSI('T', 'D', 519.30813209, 'U', 259007.95005252, "HEOS::R1234ze(E)"))
# print(PropsSI('T', 'D', 509.80352476, 'U', 258990.3212446, "HEOS::R1234ze(E)"))
# print(PropsSI('S', 'D', 509.80352476, 'U', 258990.3212446, "HEOS::R1234ze(E)"))

# print(PropsSI('T', 'D', 529.00671099, 'U', 259033.71749167, "HEOS::R1234ze(E)"))
# print(PropsSI('S', 'D', 529.00671099, 'U', 259033.71749167, "HEOS::R1234ze(E)"))

# print(PropsSI('P', 'D', 3.8808791, 'U', -2314531.40885433, "HEOS::R1234ze(E)"))

AS = CP.AbstractState("REFPROP", "Air")
print(AS.T_critical())
AS.update(CP.DmassUmass_INPUTS, 519.20049536, 259012.115143168)
print(AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT))
print(AS.T())


AS.update(CP.SmassT_INPUTS, 1499.93, 376.2619149)
print(AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT))





# Input arrays:
AS = CP.AbstractState("REFPROP", "R1234ze(E)")
P_arr = np.array([6833552.88180982, 6870378.2427647])
Dmass_arr = np.array([1126.11569909, 1126.25698619])


print(PropsSI('T', 'D', Dmass_arr, 'P', P_arr, "HEOS::R1234ze(E)"))


print(AS.update(CP.DmassP_INPUTS, Dmass_arr[0], P_arr[0]))
print(AS.T())
print(AS.update(CP.DmassP_INPUTS, Dmass_arr[1], P_arr[1]))
print(AS.T())


print(PropsSI("d(P)/d(D)|T", 'D', 519.20049536+0.1, 'U', 259012.115143168, "HEOS::R1234ze(E)"))
AS.update(CP.DmassP_INPUTS, 519.20049536+0.1, 750077.39)
print(AS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT))


@np.vectorize(otypes=[float])
def update_and_get(D, P, output='T'):
    AS.update(CP.DmassP_INPUTS, D, P)
    return getattr(AS, output)()          # .T(), .Hmass(), .Smass(), etc.


# # Vectorize it properly
# vectorized_get = np.vectorize(update_and_get, otypes=[float])

T_arr = update_and_get(Dmass_arr, P_arr, output='T')
print(T_arr)


# vectorized_AS_updater = np.vectorize(lambda D, P: AS.update(CP.DmassP_INPUTS, D, P))
# extract = np.vectorize(lambda object: object.T())
# print(extract(vectorized_AS_updater(Dmass_arr, P_arr)))





# ax.scatter(1203.6835821063594 ,312.2138484973677, s = 0.1)
# ax.scatter(1203.6515242080186 ,312.504148610755, s = 0.1)


# cs = CubicSpline([509.80352476, 519.20049536, 529.00671099], [258990.3212446, 259012.115143168, 259033.71749167])
# Density_lst = np.linspace(509.80352476, 529.00671099, 10000)
# Energy_lst = cs(Density_lst)
# print(Density_lst)
# print(Energy_lst)
# Pressure_lst = []
# for D, U in zip(Density_lst, Energy_lst):
#     try:
#         Pressure_lst.append(PropsSI('P', 'D', D, 'U', U, "HEOS::R1234ze(E)"))
#     except:
#         Pressure_lst.append(np.nan)

# for i in range(len(Pressure_lst)):
#     print(Pressure_lst[i])
# plt.plot(Density_lst, Pressure_lst)
# plt.scatter(519.2018, 750077.39)
# plt.show()


# # Density_lst = np.linspace(509.80352476, 529.00671099, 2000)
# # Energy_lst = np.linspace(258990.3212446, 259033.71749167, 2000)
# Density_lst = np.linspace(519.2018-0.2, 519.2018+0.2, 2000)
# Energy_lst = np.linspace(259012.115143168 - 100, 259012.115143168 + 100, 2000)

# U, D  = np.meshgrid(Energy_lst, Density_lst)
# print(D)
# print(U)
# total_arrays = len(D)
# Pressure_lst = []
# for Density, Energy in zip(D, U):
#     print(f"Processing array {len(Pressure_lst) + 1}/{total_arrays}")
#     try:
#         Pressure_lst.append(PropsSI('P', 'D', Density, 'U', Energy, "HEOS::R1234ze(E)"))
#     except:
#         Pressure_lst.append([np.nan for _ in range(len(Density))])

# for i in range(len(Pressure_lst)):
#     print(D[i])
#     print(Pressure_lst[i])
# for density_arr, pressure_arr in zip(D, Pressure_lst):
#     plt.plot(density_arr, pressure_arr)
# plt.scatter(519.2018, 750077.39)
# plt.show()


