###########################################
# Imports
###########################################
import matplotlib.pyplot as plt
import numpy as np

from coolprop_interface import CoolPropAbstractState
from config import Config





###########################################
# utils
###########################################
def configure_matplotlib():
    """
    Simple function to configure matplotlib with the desired settings for the thermoplot. 
    """
    # give matplotlib a LaTeX formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Utopia"],
        "text.latex.preamble": (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"
            r"\usepackage{siunitx}"
            r"\usepackage{bm}"
            r"\sisetup{group-separator={\,},group-minimum-digits=4}"
        ),
    })
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.1
    return None



def extract_critical_point(config: type[Config], AS: type[CoolPropAbstractState]) -> np.ndarray:
    """
    Extract correct critical point coordinates. Depending on diagram type a different approach is necessary. The name of the diagram can be used to create a code that works in general.  

    Arguments
    ---------
    config: Config
        Configuration object containing thermoplot settings, extracted from a .ini file. For more information please refer to config.py
    AS: CoolPropAbstractState
        Abstract state object for property calculations. Created in coolprop_interface.py. See file for more information.

    Returns
    -------
    np.ndarray
        Array containing the critical point coordinates.
    """
    # Only T_crit, P_crit and rho_crit can be used for extracting critical point coordinates. 
    # use rho_crit as second variable for extracting S and H information. 
    iv_type = config.thermoplot_settings["diagram_type"][0]
    dv_type = config.thermoplot_settings["diagram_type"][-1]
    iv_crit = AS.PropsSI(f"{iv_type}crit")
    rho_crit = AS.PropsSI("Dcrit")
    dv_crit = AS.PropsSI(dv_type, "D", rho_crit, iv_type, iv_crit)

    # convert to np.ndarray for easy appending to isoline data
    crit_coords = np.array([[dv_crit, iv_crit]])

    return crit_coords