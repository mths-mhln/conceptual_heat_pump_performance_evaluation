###########################################
# Imports
###########################################
import copy

from utils import extract_critical_point
from config import Config
from coolprop_interface import CoolPropAbstractState

import numpy as np





###########################################
# Isoline generation - General
###########################################
def construct_saturation_dome(config: type[Config], AS: type[CoolPropAbstractState]) -> np.ndarray:
    """
    Constructs the saturation dome by evaluating the saturation branches at control points of the independent variable, and combining these with the critical point coordinates. 
    The saturation dome is returned as an array of coordinates.

    Attributes
    ----------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.

    Returns
    -------
    
    """
    # independent variable is the vertical axis variable in the diagram. This is necessary to obtain determinate thdy pairs. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    iv_range[1] = min(iv_range[1], iv_crit)

    # Split range into segment until 5% below critical point. *2 for additional resolution
    if config.thermoplot_settings["diagram_type"] == "TS":
        iv_cp = np.linspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"]*2)
    elif config.thermoplot_settings["diagram_type"] == "PH":
        iv_cp = np.geomspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"]*2)

    # evaluate saturation dome branches left and right of critical point at control points. 
    dome_coords_Q_0 = np.column_stack((AS.PropsSI(dv_type, iv_type, iv_cp, "Q", 0), iv_cp)) # Q = 0 isoline
    dome_coords_Q_1 = np.column_stack((AS.PropsSI(dv_type, iv_type, iv_cp, "Q", 1), iv_cp)) # Q = 1 isoline

    # evaluate critical point coordinates and combine with dome branches
    crit_coords = extract_critical_point(config, AS)
    dome_coords = np.concatenate([dome_coords_Q_0[:-1], crit_coords, dome_coords_Q_1[:-1][::-1]]) # combine into single array of coordinates for saturation dome

    return dome_coords



def construct_quality_isolines(config: type[Config], AS: type[CoolPropAbstractState], n_iq_lines: float) -> np.ndarray:
    """
    Constructs quality isolines by evaluating the quality isoline at control points of the independent variable. The quality isolines are returned as a list of dictionaries 
    containing the isoline value and coordinates.

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.
    n_iq_lines: int
        Number of quality isolines to construct. Quality isolines are constructed for quality values between 0 and 1, so the quality values at which to construct the isolines are 
        determined by dividing the range [0,1] into n_iq_lines evenly spaced intervals.

    Returns
    -------
    quality_isolines_data: list[dict]
        List of dictionaries containing the isoline data, including the "coords" key with the isoline coordinates.
    """
    # independent variable is the vertical axis variable in the diagram. This is necessary to obtain determinate thdy pairs. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    iv_range[1] = min(iv_range[1], iv_crit)

    if config.thermoplot_settings["diagram_type"] == "TS":
        iv_cp = np.linspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"])
    elif config.thermoplot_settings["diagram_type"] == "PH":
        iv_cp = np.geomspace(iv_range[0], iv_range[1], config.thermoplot_settings["n_pts"])

    # instantiate quality isolines list
    quality_isolines_data = []

    # construct quality isolines. 
    for q in np.linspace(0, 1, n_iq_lines):
        # evaluate isoline at control points. 
        quality_arr = np.ones(iv_cp.shape) * q
        dv_vals = AS.PropsSI(dv_type, iv_type, iv_cp, "Q", quality_arr)

        # combine iv and dv into coordinate pairs and return
        quality_isolines_coords = np.column_stack((dv_vals, iv_cp))

        # store quality value and isoline in dictionary
        quality_isolines_data.append({"isoline_val": q, "coords": quality_isolines_coords})

    return quality_isolines_data



def construct_critical_isoline(config: type[Config], AS: type[CoolPropAbstractState], n_pts: int) -> np.ndarray:
    """
    Constructs the critical isoline by evaluating the critical isoline at control points of the independent variable. The critical isoline is returned as an array of coordinates.

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.
    n_pts: int
        Number of control points at which to evaluate the critical isoline. The control points are evenly spaced between the lower bound of the independent variable range and the critical point.

    Returns
    -------
    critical_isoline_coords: np.ndarray
        Array of coordinates of the critical isoline.
    """
    # independent variable is the horizontal axis variable in the diagram. this is necessary as critical isoline reaches 0 slope
    iv_type = config.thermoplot_settings["diagram_type"][-1]
    dv_type = config.thermoplot_settings["diagram_type"][0]

    # Extract plot bounds
    iv_lo, iv_hi = config.thermoplot_settings[f"{iv_type}_range"]
    dv_lo, dv_hi = config.thermoplot_settings[f"{dv_type}_range"]

    # get critical isoline type
    if config.thermoplot_settings["diagram_type"] == "TS":
        critical_isoline_type = "P"
    elif config.thermoplot_settings["diagram_type"] == "PH":
        critical_isoline_type = "T"

    # extract independent variable range and get control points of iv at which to evaluate isoline.
    # dome extends only to critical point, so place that as upper bound.
    iv_range = copy.deepcopy(config.thermoplot_settings[f"{iv_type}_range"])
    iv_cp = np.linspace(iv_range[0], iv_range[1], n_pts)

    # get critical isoline value
    critical_isoline_val = AS.PropsSI(f"{critical_isoline_type}crit")

    # evaluate dv at control points for critical quality of 0.5
    critical_isoline_arr = np.ones(iv_cp.shape) * critical_isoline_val
    dv_vals = AS.PropsSI(dv_type, iv_type, iv_cp, critical_isoline_type, critical_isoline_arr)

    # only keep vals inside of plot range
    v = (np.isfinite(dv_vals) & (dv_vals >= dv_lo) & (dv_vals <= dv_hi) &
         (iv_cp >= iv_lo*0.99) & (iv_cp <= iv_hi*1.01))

    # combine iv and dv into coordinate pairs and return
    critical_isoline_coords = np.column_stack((iv_cp[v], dv_vals[v]))

    return critical_isoline_coords





###########################################
# Isoline generation - TS
###########################################
def isobar_lines_ts(config: type[Config], AS: type[CoolPropAbstractState], n_lines=12):
    """
    Construction of isobars in TS diagram starts at the lowest possible isobar for which the pressure is still larger than the triple point pressure, and extends to 3 times the critical pressure. 
    This decision is arbitrary, but provides a good coverage of the fluid-vapour region of the phase diagram.

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.
    n_lines: int
        Number of isobar lines to construct.

    Returns
    -------
    isobar_lines_data_ts: list[dict]
        List of dictionaries containing the isoline data, including the "coords" key with the isoline coordinates.
    """
    # extract plot limits specified by the user in the config file
    T_lo, T_hi = config.thermoplot_settings["T_range"]
    s_lo, s_hi = config.thermoplot_settings["S_range"]

    # compute critical and triple point pressure. Put lower bound of pressure to higher value than triple point pressure if the value does not exist (and if all else fails to 1e3)
    p_crit = AS.PropsSI("Pcrit")
    T_trip = AS.PropsSI("Ttriple")
    p_trip = AS.PropsSI("P", "T", T_trip*1.001, "Q", 0)
    p_lo_est = max(p_trip*1.1 if np.isfinite(p_trip) else 1e3, 1e3)

    # compute isobar magnitudes for which to calculate the pressure. 
    p_vals = np.geomspace(p_lo_est, p_crit*3.0, n_lines)

    # isobars will be evaluated at control points of temperature. This is not the "best" way to do this. the perfect way would be to construct a rough first draft using this method
    # and to refine it by sampling an interpolant, but that is far above the precision necessary for this tool. Also, this is not a viable method for building isobars inside of the saturation dome. 
    # which is handeled below
    T_sweep = np.linspace(T_lo,T_hi, config.thermoplot_settings["n_pts"])
    
    # Instantiate dictionary to store isobar lines data, which consists of the isobar iso value and the corresponding isoline coordinates.
    isobar_lines_data_ts = []

    # construct isobars. 
    for p in p_vals:

        # if isobar enters the critical dome...
        if p < p_crit:
            # ...we need a method to evaluate the nice horizontal, hence split up the construction of the isobar into three branches: subcooled, two-phase and superheated.
            # get the bounds first
            T_sat = AS.PropsSI("T", "P", p, "Q", 0)
            s_sat_l = AS.PropsSI("S", "P", p, "Q", 0)
            s_sat_v = AS.PropsSI("S", "P", p, "Q", 1)
            if not (np.isfinite(T_sat) and np.isfinite(s_sat_l) and np.isfinite(s_sat_v)):
                continue
            # Subcooled branch (T < T_sat)
            T_liq = T_sweep[T_sweep < T_sat]
            s_liq = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_liq])
            # Superheated branch (T > T_sat)
            T_vap = T_sweep[T_sweep > T_sat]
            s_vap = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_vap])
            # Two-phase horizontal bridge at T_sat
            s_2ph = np.linspace(s_sat_l, s_sat_v, 20)
            T_2ph = np.full(20, T_sat)
            # Stitch in physical order to avoid artificial loops/crossings.
            s_all = np.concatenate([s_liq, s_2ph, s_vap])
            T_all = np.concatenate([T_liq, T_2ph, T_vap])
        else:
            # Ignore separated building and simply use the temperature control points. This has high likelyhood of yielding a sloped isobar inside the saturation dome, however. 
            s_all = np.array([AS.PropsSI("S", "P", p, "T", T) for T in T_sweep])
            T_all = T_sweep

        # filter out any values that are not finite or outside of the plot range. The second should be ensured from the way the isoline is constructed, but it is a failsafe. 
        v = (np.isfinite(s_all) & (s_all >= s_lo) & (s_all <= s_hi) &
             (T_all >= T_lo*0.95) & (T_all <= T_hi*1.05))
        if v.sum() > 3:
            isobar_isoline_coords = np.column_stack((s_all[v], T_all[v]))
            isobar_lines_data_ts.append({"isoline_val": p,"coords": isobar_isoline_coords})
    return isobar_lines_data_ts


def isenthalp_lines_ts(config: type[Config], AS: type[CoolPropAbstractState], n_lines=18):
    """
    Construction of isenthalps in TS diagram is more complicated. First a grid is constructed entropy and temperature pairs. The enthalpy is evaluated at each of these pairs, and the 
    thermpdyanmic calls that yield a value (that are not outside of the thdy library's scope) are recorded. Based on the min and max value that is extractable within the plot range, 
    the isenthalp iso value range is constructed. The isenthalp is then constructed by evaluating the entropy and temperature from the isenthalp iso value and the isobar values that 
    are represetned on the plot, chosen arbitrarily, see isobar_lines_ts. 

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.
    n_lines: int
        Number of isenthalp lines to construct.

    Returns
    -------
    isenthalp_lines_data_ts: list[dict]
        List of dictionaries containing the isoline data, including the "coords" key with the isoline coordinates.
    """
    # Extract plot limits specified by the user in the config file
    T_lo, T_hi = config.thermoplot_settings["T_range"]
    s_lo, s_hi = config.thermoplot_settings["S_range"]

    # Sample enthalpy at a grid of entropy and temperature values within the plot range to find the min and max enthalpy that can be extracted within the plot range. 
    h_samples = []
    for s in np.linspace(s_lo, s_hi, 6):
        for T in np.linspace(T_lo, T_hi, 6):
            try:
                h_samples.append(AS.PropsSI("H", "T", T, "S", s))
            except:
                h_samples.append(np.nan)
    h_samples = [h for h in h_samples if np.isfinite(h)]
    if not h_samples:
        return []
    
    # create isenthalp iso values from the maximum range of enthalpy values visible in the plot ranges specified. 
    h_vals  = np.linspace(min(h_samples)*0.95, max(h_samples)*1.05, n_lines)

    # Compute isobar isoline magnitudes in similar manner to isobar_lines_ts. 
    p_crit  = AS.PropsSI("Pcrit")
    p_arr   = np.geomspace(p_crit*3.0, 1e3, config.thermoplot_settings["n_pts"])

    # instantiate list to store isenthalp lines data, which consists of the isenthalp iso value and the corresponding isoline coordinates.
    isenthalp_lines_data_ts = []

    # for each of the visible isenthalps, compute the T, S pairs at each of the isobar values for that enthalpy value. 
    for h in h_vals:
        s_arr = np.array([AS.PropsSI("S", "P", p, "H", h) for p in p_arr])
        T_arr = np.array([AS.PropsSI("T", "P", p, "H", h) for p in p_arr])
        # filter out any values that are not finite or outside of the plot range. 
        v = (np.isfinite(s_arr) & np.isfinite(T_arr) &
             (s_arr >= s_lo) & (s_arr <= s_hi) &
             (T_arr >= T_lo*0.99) & (T_arr <= T_hi*1.01))
        if v.sum() > 3:
            isenthalp_isoline_coords = np.column_stack((s_arr[v], T_arr[v]))
            isenthalp_lines_data_ts.append({"isoline_val": h,"coords": isenthalp_isoline_coords})

    return isenthalp_lines_data_ts





###########################################
# Isoline generation - PH
###########################################
def isotherm_lines_ph(config: type[Config], AS: type[CoolPropAbstractState], n_lines=18):
    """
    Construction of isotherms in PH diagram is similar to the construction of isobars in TS diagram. Starts at the lowest possible isotherm for which the temperature is still larger 
    than the triple point temperature, and extends to 1.5 times the critical temperature. This decision is arbitrary, but provides a good coverage of the fluid-vapour region of the phase diagram.

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.
    n_lines: int
        Number of isotherm lines to construct.

    Returns
    -------
    isotherm_lines_data_ph: list[dict]
        List of dictionaries containing the isoline data, including the "coords" key with the isoline coordinates.
    """
    # extract plot limits specified by the user in the config file
    p_lo, p_hi = config.thermoplot_settings["P_range"]
    h_lo, h_hi = config.thermoplot_settings["H_range"]

    # compute critical and triple point temperature. Put lower bound of temperature to higher value than triple point temperature if the value does not exist (and if all else fails to 300K)
    T_crit = AS.PropsSI("Tcrit")
    T_trip = AS.PropsSI("Ttriple", "T", 300, "Q", 1)

    # compute isotherm magnitudes for which to calculate the pressure.
    T_vals = np.linspace(T_trip*1.05, T_crit*1.5, n_lines)

    # isotherms will be evaluated at control points of pressure. This is not the "best" way to do this. the perfect way would be to construct a rough first draft using this method
    # and to refine it by sampling an interpolant, but that is far above the precision necessary for this tool. Also, this is not a viable method for building isotherms inside of the saturation dome.
    # which is handeled below.
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, config.thermoplot_settings["n_pts"])

    # Instantiate list to store isotherm lines data, which consists of the isotherm iso value and the corresponding isoline coordinates.
    isotherm_lines_data_ph = []

    # construct isotherms.
    for T in T_vals:
        # if isotherm enters the critical dome...
        if T < T_crit:
            # ...we need a method to evaluate the nice horizontal, hence split up the construction of the isotherm into three branches: subcooled, two-phase and superheated.
            # get the bounds first
            p_sat = AS.PropsSI("P", "T", T, "Q", 0)
            h_sat_l = AS.PropsSI("H", "T", T, "Q", 0)
            h_sat_v = AS.PropsSI("H", "T", T, "Q", 1)
            if not (np.isfinite(p_sat) and np.isfinite(h_sat_l) and np.isfinite(h_sat_v)):
                continue
            # Liquid branch (p > p_sat)
            p_liq = p_arr[p_arr > p_sat]
            h_liq = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_liq])
            # Vapour branch (p < p_sat)
            p_vap = p_arr[p_arr < p_sat]
            h_vap = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_vap])
            # Two-phase horizontal bridge at p_sat
            h_2ph = np.linspace(h_sat_l, h_sat_v, 20)
            p_2ph = np.full(20, p_sat)
            # Stitch: liquid (high→p_sat) → bridge → vapour (p_sat→low)
            h_all = np.concatenate([h_liq[::-1], h_2ph, h_vap[::-1]])
            p_all = np.concatenate([p_liq[::-1], p_2ph, p_vap[::-1]])
        else:
            # Ignore separated building and simply use the pressure control points. This has high likelyhood of yielding a sloped isotherm inside the saturation dome, however.
            p_arr = p_arr[::-1]
            h_all = np.array([AS.PropsSI("H", "P", p, "T", T) for p in p_arr])
            p_all = p_arr

        # filter out any values that are not finite or outside of the plot range. The second should be ensured from the way the isoline is constructed, but it is a failsafe.
        v = (np.isfinite(h_all) &
             (h_all >= h_lo) & (h_all <= h_hi) &
             (p_all >= p_lo*0.9) & (p_all <= p_hi*1.1))
        if v.sum() > 3:
            isotherm_isoline_coords = np.column_stack((h_all[v], p_all[v]))
            isotherm_lines_data_ph.append({"isoline_val": T,"coords": isotherm_isoline_coords})
    return isotherm_lines_data_ph


def isentrop_lines_ph(config: type[Config], AS: type[CoolPropAbstractState], n_lines=12):
    """
    Construction of isentropes in PH diagram is similar to the construction of isenthalps in TS diagram. First a grid is constructed of enthalpy and pressure pairs. The entropy is 
    evaluated at each of these pairs, and the thermpdyanmic calls that yield a value (that are not outside of the thdy library's scope) are recorded. Based on the min and max value
    that is extractable within the plot range, the isentrop iso value range is constructed. The isentrop is then constructed by evaluating the enthalpy and pressure from the isentrop 
    iso value and the isobar values that are represetned on the plot. This part is different from the isenthalps construction.

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py
    n_lines: int
        Number of isentrop lines to construct.

    Returns
    -------
    isentrop_lines_data_ph: list[dict]
        List of dictionaries containing the isoline data, including the "coords" key with the isoline coordinates.
    """
    # Extract plot limits specified by the user in the config file
    p_lo, p_hi = config.thermoplot_settings["P_range"]
    h_lo, h_hi = config.thermoplot_settings["H_range"]

    # Sample entropy at a grid of enthalpy and pressure values within the plot range to find the min and max entropy that can be extracted within the plot range.
    s_samples = []
    for h in np.linspace(h_lo, h_hi, 6):
        for p in np.geomspace(max(p_lo, 1e3), p_hi, 6):
            s_samples.append(AS.PropsSI("S", "P", p, "H", h))
    s_samples = [s for s in s_samples if np.isfinite(s)]
    if not s_samples:
        return []
    
    # create isentrop iso values from the maximum range of entropy values visible in the plot ranges specified.
    s_vals = np.linspace(min(s_samples), max(s_samples), n_lines)

    # Compute isobar isoline magnitudes. Compute a little bit outside of the plot to allow the isolines to extend a little bit outside of the plot, which looks better.
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, config.thermoplot_settings["n_pts"])

    # instantiate list to store isentrop lines data, which consists of the isentrop iso value and the corresponding isoline coordinates.
    isentrop_lines_data_ph = []

    # for each of the visible isentrops, compute the enthalpy and pressure pairs at each of the isobar values for that entropy value.
    for sv in s_vals:
        h_arr = np.array([AS.PropsSI("H", "P", p, "S", sv) for p in p_arr])

        # filter out any values that are not finite or outside of the plot range. 
        v = (np.isfinite(h_arr) &
             (h_arr >= h_lo) & (h_arr <= h_hi) &
             (p_arr >= p_lo*0.9) & (p_arr <= p_hi*1.1))
        if v.sum() > 3:
            isentrop_isoline_coords = np.column_stack((h_arr[v], p_arr[v]))
            isentrop_lines_data_ph.append({"isoline_val": sv,"coords": isentrop_isoline_coords})
    return isentrop_lines_data_ph