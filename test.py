from CoolProp.CoolProp import PropsSI


print(PropsSI("P", "T", 275, "Q", 1, "R1233zd(E)"))


print(PropsSI("Q", "T", 2.75626562e+02, "P", 3.90919179e+05, "REFPROP::R1233zd(E)"))

print(PropsSI("T", "H", 406000, "S", 1700, "REFPROP::R1233zd(E)"))
print( 3.90919179e+05)