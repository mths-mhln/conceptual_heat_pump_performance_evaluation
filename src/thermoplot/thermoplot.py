###########################################
# Imports
###########################################
from matplotlib import ticker
import matplotlib.pyplot as plt
import hashlib
import pickle
import tempfile
from pathlib import Path

from config import Config
from utils import configure_matplotlib, extract_critical_point
from coolprop_interface import CoolPropAbstractState
from labelling import draw_isolines_labeled
from isolines import (isobar_lines_ts, isenthalp_lines_ts, isotherm_lines_ph, 
        isentrop_lines_ph, construct_quality_isolines,
        construct_saturation_dome, construct_critical_isoline)





# Create temporary dir for caching of thermoplot
_THERMOPLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "pyshockflow_thermoplot_cache"

def _format_pressure_tick(x, _):
    return f"${x/1000:.0f}$"

def _thermoplot_cache_path(thermoplot_config_file_path: str, thermoplot_overwrite_settings: dict[str, object] | None) -> Path:
    """
    Creates a unique cache path for a given combination of thermoplot config file and thermoplot settings to overwrite. 
    """
    if thermoplot_overwrite_settings is None:
        overwrite_settings_key = None
    else:
        overwrite_settings_key = tuple(sorted(thermoplot_overwrite_settings.items()))
    # save input to cache
    cache_input = (
        thermoplot_config_file_path,
        Path(thermoplot_config_file_path).read_bytes(), # include file contents in cache key so any config change invalidates the cache
        overwrite_settings_key,
    )
    # create cache key by hashing the cache input. Use pickle to serialize the cache input, and then hash the serialized input using sha256 to create a unique 
    # cache key. This ensures that different inputs will result in different cache keys, and that the same input will always result in the same cache key.
    cache_key = hashlib.sha256(pickle.dumps(cache_input, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return _THERMOPLOT_CACHE_DIR / f"{cache_key}.pkl"



def thermoplot_cached(thermoplot_config_file_path: str, thermoplot_overwrite_settings: dict[str, list] = None) -> type[plt.Figure]:
    """
    Disk-cached version of the thermoplot function.
    """
    cache_path = _thermoplot_cache_path(thermoplot_config_file_path, thermoplot_overwrite_settings)
    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            try:
                return pickle.load(cache_file)
            except (EOFError, pickle.UnpicklingError):
                pass


    fig = thermoplot(thermoplot_config_file_path, thermoplot_overwrite_settings)
    _THERMOPLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=_THERMOPLOT_CACHE_DIR, delete=False) as cache_file:
        pickle.dump(fig, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_cache_path = Path(cache_file.name)
    temporary_cache_path.replace(cache_path)
    return fig


def thermoplot(thermoplot_config_file_path: str, thermoplot_overwrite_settings: dict[str, list] = None) -> type[plt.Figure]:
    """
    Integral functionality of the package. Combines functionality by all submodules to produce a thermodynamic diagram figure according
    to user specifications in the config file, and returns the figure object to the user suc that they can further adapt it if they wish.

    Arguments
    ---------
    thermoplot_config_file_path: str
        File path to the config file with user specifications for the thermoplot. See config.py for details on how to specify the config file and which settings are necessary.
    thermoplot_overwrite_settings: dict[str, list], optional
        A dictionary with keys corresponding to thermoplot settings (e.g. "show_isolines") and values corresponding to the value that the user wishes to overwrite this setting with. 
        This allows users to overwrite settings specified in the config file without having to adapt the config file itself. This is useful for example when users want to produce multiple 
        thermoplots with slightly different settings, but do not want to create multiple config files for this. By default, this is set to None, meaning that no settings will be overwritten 
        and the function will use the settings specified in the config file.
    
    Returns
    -------
    plt.Figure
        The figure object containing the thermodynamic diagram.
    """
    # load configuration 
    config = Config(config_file=thermoplot_config_file_path)
    config.get_thermoplot_settings()

    # if thermoplot settings are provided, overwrite any setting that is specified to be overwritten through the thermoplot_settings dictionary.  
    if thermoplot_overwrite_settings is not None:
        for setting_name, setting_value in thermoplot_overwrite_settings.items():
            try:
                config.thermoplot_settings[setting_name] = setting_value
            except KeyError:
                print("Attempting to adapt thermoplot settings with user-provided settings, but the following setting name is not recognized: ", setting_name)

    # apply or clear LaTeX-specific matplotlib settings after all overrides have been resolved.
    if config.thermoplot_settings["latex_formatting"]:
        configure_matplotlib()

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 7))

    # create log axis for pressure in case of PH
    if config.thermoplot_settings['diagram_type'] == "PH":
        ax.set_yscale('log')

    # define color palette
    quality_isoline_color    = '#1a3a6b'   # dark blue
    isoline_2_color          = '#6ab0de'   # light blue  (isobars on TS, isentropes on PH)
    isoline_3_color          = '#e07b20'   # orange      (isenthalps on TS, isotherms on PH)

    # Set plot limits according to user specification.
    iv_lo = config.thermoplot_settings[f"{config.diagram_type[-1]}_range"][0]
    iv_hi = config.thermoplot_settings[f"{config.diagram_type[-1]}_range"][1]
    dv_lo = config.thermoplot_settings[f"{config.diagram_type[0]}_range"][0]
    dv_hi = config.thermoplot_settings[f"{config.diagram_type[0]}_range"][1]
    ax.set_xlim(iv_lo, iv_hi)
    ax.set_ylim(dv_lo, dv_hi)

    # instantiate fluid object
    AS = CoolPropAbstractState("REFPROP", config.thermoplot_settings["fluid_name"])

    # extract critical point, and plot as yellow circle with black edge.
    crit_coords = extract_critical_point(config, AS)
    ax.plot(crit_coords[:, 0], crit_coords[:, 1], marker='o', markerfacecolor='yellow',
        markersize=5, markeredgecolor='black', zorder=9)

    # construct and draw saturation dome. Consists of three parts: isoline for Q = 0, critical point, and isoline for Q = 1. 
    dome_coords = construct_saturation_dome(config, AS)
    ax.plot(dome_coords[:, 0], dome_coords[:, 1], color='black', lw=1.0, zorder=3) # plot saturation dome

    # construct and draw critical isoline as dashed black line
    critical_isoline_coords = construct_critical_isoline(config, AS, n_pts=400)
    ax.plot(critical_isoline_coords[:, 0], critical_isoline_coords[:, 1], color='black', lw=1.0, ls=':', zorder=2)

    # Construct and plot isolines if user specified to do so.
    if config.thermoplot_settings["show_isolines"]:
        # construct and plot quality isolines. 
        n_iq_lines = 10
        quality_isolines_data = construct_quality_isolines(config, AS, n_iq_lines)
        draw_isolines_labeled(ax, quality_isolines_data, 
            color=quality_isoline_color, frac=0.15, 
            lbl_fmt_short = lambda v: rf"${v:.2f}$", 
            lbl_fmt_named=lambda v: rf"$x={v:.2f}$", 
            yscale='linear')

        # construct isolines 2 and 3 metadata. 
        # TS diagram: (2) isobars        (3) isenthalps
        # PH diagram: (2) isentropes     (3) isotherms
        if config.diagram_type == "TS":
            isobar_lines_ts_data = isobar_lines_ts(config, AS)
            draw_isolines_labeled(ax, isobar_lines_ts_data,  
                color=isoline_2_color, frac=0.83,
                lbl_fmt_short=lambda v: rf"${v/1e3:.0f}$",
                lbl_fmt_named=lambda v: rf"$p={v/1e3:.0f}\,\mathrm{{kPa}}$",
                yscale='linear')
            isenthalp_lines_ts_data = isenthalp_lines_ts(config, AS)
            draw_isolines_labeled(ax, isenthalp_lines_ts_data,  
                color=isoline_3_color, frac=0.1,
                lbl_fmt_short=lambda v: rf"${v/1e3:.0f}$",
                lbl_fmt_named=lambda v: rf"$h={v/1e3:.0f}\,\mathrm{{kJ/kg}}$",
                yscale='linear')
            
        elif config.diagram_type == "PH":
            isotherm_lines_ph_data = isotherm_lines_ph(config, AS)
            draw_isolines_labeled(ax, isotherm_lines_ph_data,  
                color=isoline_3_color, frac=0.90,
                lbl_fmt_short=lambda v: rf"${v:.0f}$",
                lbl_fmt_named=lambda v: rf"$T={v:.0f}\,\mathrm{{K}}$",
                yscale='log')
            isentrop_lines_ph_data = isentrop_lines_ph(config, AS)
            draw_isolines_labeled(ax, isentrop_lines_ph_data,
                color=isoline_2_color, frac=0.83,
                lbl_fmt_short=lambda v: rf"${v:.0f}$",
                lbl_fmt_named=lambda v: rf"$s={v:.0f}\,\mathrm{{kJ/kg/K}}$",
                yscale='log')

    # Set plot labels and aesthetics.
    if config.thermoplot_settings["diagram_type"] == "TS":
        ax.set_xlabel(r"$s\ [\mathrm{J/kg/K}]$")
        ax.set_ylabel(r"$T\ [\mathrm{K}]$")
    elif config.thermoplot_settings["diagram_type"] == "PH":
        ax.set_xlabel(r"$h\ [\mathrm{kJ/kg}]$")
        ax.set_ylabel(r"$p\ [\mathrm{Pa}]$")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(_format_pressure_tick))
    fig.tight_layout()

    return fig

 