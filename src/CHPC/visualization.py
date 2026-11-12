import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from scipy.optimize import brentq
from pathlib import Path

from thermoplot import thermoplot_cached, configure_matplotlib
from CHPC.logger import setup_logger
from CHPC.thermodynamics import _get_abstract_state
from CHPC.results import load_results, method_suffix

logger = setup_logger()

# thermoplot config files of the T-s and p-h diagram backgrounds. The fluid and the diagram 
# ranges are overwritten, such that the diagrams are centred around the cycle.
THERMOPLOT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "thermoplot"

# Margins around the cycle in the T-s and p-h diagrams, as a fraction of the span of the cycle 
# (and critical point) in entropy and temperature, or enthalpy and the logarithm of pressure.
TS_MARGINS = {"s_left": 0.2, "s_right": 0.2, "T_bot": 0.25, "T_top": 0.35}
PH_MARGINS = {"h_left": 0.3, "h_right": 0.15, "p_bot": 0.1, "p_top": 0.2}



# Spinodal computation
# ====================
def _dpdrho_T(AS, rho, T):
    """Return dP/dρ|T at a given (ρ, T) using CoolProp's derivative interface."""
    return float(AS.PropsSI("d(P)/d(D)|T", "Dmass", rho, "T", T))


def _find_spinodal_manual(refrigerant, T_min_override=None, T_max_override=None, n_T=100):
    """For each isotherm, find the two densities where dP/dρ|T = 0.
    If T_min_override/T_max_override are provided, use those; otherwise use defaults.
    Returns separate arrays for liquid and vapour spinodal branches in (T, rho, P) space."""
    try:
        AS = _get_abstract_state(f"REFPROP::{refrigerant}")
        T_c = AS.PropsSI("Tcrit")
        rho_c = AS.PropsSI("Dcrit")
        T_min_default = max(AS.PropsSI("Ttriple") + 1.0, 200.0)
        T_max_default = T_c - 0.5
        
        # Use provided bounds or defaults
        T_min = T_min_override if T_min_override is not None else T_min_default
        T_max = T_max_override if T_max_override is not None else T_max_default
        
        # Clamp T_max to below critical point
        T_max = min(T_max, T_c - 0.5)
        # Ensure T_min is valid
        T_min = max(T_min, T_min_default)
        
        if T_max <= T_min:
            return None, None, None, None, None, None

        T_vals = np.linspace(T_min, T_max, n_T)
        liq_T, liq_rho, liq_P = [], [], []
        vap_T, vap_rho, vap_P = [], [], []

        for T in T_vals:
            try:
                # Get saturation densities as bracket bounds
                rho_liq_sat = float(AS.PropsSI("Dmass", "Q", 0.0, "T", T))
                rho_vap_sat = float(AS.PropsSI("Dmass", "Q", 1.0, "T", T))

                # Liquid spinodal: dP/dρ|T = 0 between rho_c and rho_liq_sat
                rho_lo, rho_hi = rho_c * 1.01, rho_liq_sat * 0.999
                dpdrho_lo = _dpdrho_T(AS, rho_lo, T)
                dpdrho_hi = _dpdrho_T(AS, rho_hi, T)
                if np.isfinite(dpdrho_lo) and np.isfinite(dpdrho_hi) and dpdrho_lo * dpdrho_hi < 0:
                    rho_sp = brentq(lambda r: _dpdrho_T(AS, r, T), rho_lo, rho_hi, xtol=1e-6)
                    liq_T.append(T)
                    liq_rho.append(rho_sp)
                    liq_P.append(float(AS.PropsSI("P", "Dmass", rho_sp, "T", T)))

                # Vapour spinodal: dP/dρ|T = 0 between rho_vap_sat and rho_c
                rho_lo, rho_hi = rho_vap_sat * 1.001, rho_c * 0.99
                dpdrho_lo = _dpdrho_T(AS, rho_lo, T)
                dpdrho_hi = _dpdrho_T(AS, rho_hi, T)
                if np.isfinite(dpdrho_lo) and np.isfinite(dpdrho_hi) and dpdrho_lo * dpdrho_hi < 0:
                    rho_sp = brentq(lambda r: _dpdrho_T(AS, r, T), rho_lo, rho_hi, xtol=1e-6)
                    vap_T.append(T)
                    vap_rho.append(rho_sp)
                    vap_P.append(float(AS.PropsSI("P", "Dmass", rho_sp, "T", T)))
            except Exception:
                continue

        return (np.array(liq_T), np.array(liq_rho), np.array(liq_P),
                np.array(vap_T), np.array(vap_rho), np.array(vap_P))
    except Exception:
        return None, None, None, None, None, None


def _get_spinodal_ts(refrigerant, T_lo=None, T_hi=None):
    """Compute spinodal curve and convert to (s, T) coordinates for TS diagram.
    If T_lo/T_hi provided, compute spinodal across that range."""
    liq_T, liq_rho, liq_P, vap_T, vap_rho, vap_P = _find_spinodal_manual(
        refrigerant, T_min_override=T_lo, T_max_override=T_hi, n_T=120
    )
    if liq_T is None or len(liq_T) == 0:
        return None, None, None, None

    AS = _get_abstract_state(f"REFPROP::{refrigerant}")
    liq_s = AS.PropsSI("S", "Dmass", liq_rho, "T", liq_T)
    vap_s = AS.PropsSI("S", "Dmass", vap_rho, "T", vap_T)
    return np.array(liq_s), np.array(liq_T), np.array(vap_s), np.array(vap_T)


def _get_spinodal_ph(refrigerant, p_lo=None, p_hi=None):
    """Compute spinodal curve and convert to (h, p) coordinates for PH diagram.
    If p_lo/p_hi provided, convert pressure range to temperature range for spinodal computation."""
    AS = _get_abstract_state(f"REFPROP::{refrigerant}")

    # Convert pressure bounds to approximate temperature bounds (saturation temperature)
    T_min = float(AS.PropsSI("T", "P", p_lo, "Q", 0.0)) if p_lo is not None else None
    T_max = float(AS.PropsSI("T", "P", p_hi, "Q", 1.0)) if p_hi is not None else None
    T_min = None if T_min is not None and not np.isfinite(T_min) else T_min
    T_max = None if T_max is not None and not np.isfinite(T_max) else T_max
    
    liq_T, liq_rho, liq_P, vap_T, vap_rho, vap_P = _find_spinodal_manual(
        refrigerant, T_min_override=T_min, T_max_override=T_max, n_T=120
    )
    if liq_T is None or len(liq_T) == 0:
        return None, None, None, None

    liq_h = AS.PropsSI("H", "Dmass", liq_rho, "T", liq_T)
    vap_h = AS.PropsSI("H", "Dmass", vap_rho, "T", vap_T)
    return np.array(liq_h), np.array(liq_P), np.array(vap_h), np.array(vap_P)


def _draw_spinodal(ax, x_liq, y_liq, x_vap, y_vap):
    if x_liq is not None and len(x_liq) > 2:
        ax.plot(x_liq, y_liq, color='#8B0000', ls='--', lw=1.2, zorder=2,
                label='Spinodal (liquid)')
    if x_vap is not None and len(x_vap) > 2:
        ax.plot(x_vap, y_vap, color='#DC143C', ls='--', lw=1.2, zorder=2,
                label='Spinodal (vapour)')



# Utility
# =======
def _cop_type_to_latex(cop_key):
    parts = cop_key.split("_", 1)
    if len(parts) == 2:
        return rf"\mathrm{{{parts[0]}}}_{{\mathrm{{{parts[1]}}}}}"
    return rf"\mathrm{{{cop_key}}}"


def _q(out, *args, cycle_config):
    """Property extraction for the diagrams, nan is returned for states that cannot be evaluated."""
    value = _get_abstract_state(f"REFPROP::{cycle_config['refrigerant']}").PropsSI(out, *args)
    return float(value) if np.ndim(value) == 0 else value


def _verification_path_with_isenthalpic_expansion(verification_data, cycle_config, n_pts=80):
    T_arr = np.array(verification_data[0, :], dtype=float)
    p_arr = np.array(verification_data[1, :], dtype=float)
    h_arr = np.array(verification_data[2, :], dtype=float)
    s_arr = np.array(verification_data[3, :], dtype=float)

    if len(T_arr) <= 6:
        return T_arr, p_arr, h_arr, s_arr

    p_start = p_arr[5]
    p_end = p_arr[6]
    h_const = h_arr[5]

    p_seg = np.linspace(p_start, p_end, n_pts)
    T_seg = np.array([_q("T", "P", p, "H", h_const, cycle_config=cycle_config) for p in p_seg])
    s_seg = np.array([_q("S", "P", p, "H", h_const, cycle_config=cycle_config) for p in p_seg])

    v = np.isfinite(T_seg) & np.isfinite(s_seg)
    if v.sum() < 4:
        return T_arr, p_arr, h_arr, s_arr

    p_mid = p_seg[v][1:-1]
    T_mid = T_seg[v][1:-1]
    s_mid = s_seg[v][1:-1]
    h_mid = np.full_like(p_mid, h_const)

    T_new = np.concatenate([T_arr[:6], T_mid, T_arr[6:]])
    p_new = np.concatenate([p_arr[:6], p_mid, p_arr[6:]])
    h_new = np.concatenate([h_arr[:6], h_mid, h_arr[6:]])
    s_new = np.concatenate([s_arr[:6], s_mid, s_arr[6:]])
    return T_new, p_new, h_new, s_new



# Axi bound calculation
# ======================
def _cycle_bounds_ts(cycle_data, cycle_config, margins=None):
    m = TS_MARGINS if margins is None else {**TS_MARGINS, **margins}

    s_vals = [cycle_data[k] for k in ("s_ref_1","s_ref_2","s_ref_3","s_ref_4")]
    T_vals = [cycle_data[k] for k in ("T_ref_1","T_ref_2","T_ref_3","T_ref_4")]

    # Always include the critical point
    p_crit = _q("Pcrit", cycle_config=cycle_config)
    T_crit = _q("Tcrit", cycle_config=cycle_config)
    s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    if np.isfinite(s_crit) and np.isfinite(T_crit):
        s_vals.append(s_crit)
        T_vals.append(T_crit)

    s_lo, s_hi = min(s_vals), max(s_vals)
    T_lo, T_hi = min(T_vals), max(T_vals)
    ds = s_hi - s_lo
    dT = T_hi - T_lo
    return (s_lo - ds*m["s_left"],  s_hi + ds*m["s_right"],
            T_lo - dT*m["T_bot"],   T_hi + dT*m["T_top"])


def _cycle_bounds_ph(cycle_data, cycle_config, margins=None):
    m = PH_MARGINS if margins is None else {**PH_MARGINS, **margins}

    h_vals = [cycle_data[k] for k in ("h_ref_1","h_ref_2","h_ref_3","h_ref_4")]
    p_vals = [cycle_data[k] for k in ("p_ref_1","p_ref_2","p_ref_3","p_ref_4")]

    # Always include the critical point
    p_crit = _q("Pcrit", cycle_config=cycle_config)
    T_crit = _q("Tcrit", cycle_config=cycle_config)
    h_crit = _q("H", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    if np.isfinite(h_crit) and np.isfinite(p_crit):
        h_vals.append(h_crit)
        p_vals.append(p_crit)

    h_lo, h_hi = min(h_vals), max(h_vals)
    p_lo, p_hi = min(p_vals), max(p_vals)
    dh = h_hi - h_lo
    log_p_lo = np.log10(p_lo)
    log_p_hi = np.log10(p_hi)
    log_span = log_p_hi - log_p_lo
    p_lo_plot = max(10 ** (log_p_lo - m["p_bot"] * max(log_span, 0.3)), 1e2)
    p_hi_plot = 10 ** (log_p_hi + m["p_top"] * max(log_span, 0.3))
    return (h_lo - dh*m["h_left"], h_hi + dh*m["h_right"],
            p_lo_plot, p_hi_plot)



# Expansion lines + legend
# ========================
def _add_expansion_lines(ax, cycle_data, diagram_type, cycle_config, include_verification=False):
    s     = cycle_data
    p_exp = np.linspace(s["p_ref_3"], s["p_ref_1"], 150)
    col   = "#02220E"
    alp   = 0.75

    if diagram_type == "PH":
        h_isenth = np.full_like(p_exp, s["h_ref_3"])
        line_isenth, = ax.plot(h_isenth, p_exp, color=col, ls="-.", lw=1.5,
                            alpha=alp, zorder=4,
                            label=r"$\mathrm{Isenthalpic\ expansion}$")
        h_isen = np.array([_q("H", "P", p, "S", s["s_ref_3"], cycle_config=cycle_config) for p in p_exp])
        v = np.isfinite(h_isen)
        line_isen, = ax.plot(h_isen[v], p_exp[v], color=col, ls="--", lw=1.5,
                            alpha=alp, zorder=4,
                            label=r"$\mathrm{Isentropic\ expansion}$")
        # Horizontal connector from isentropic exit → actual cycle point 4
        h_isen_exit = _q("H", "P", s["p_ref_1"], "S", s["s_ref_3"], cycle_config=cycle_config)
        ax.plot([h_isen_exit, s["h_ref_4"]], [s["p_ref_1"], s["p_ref_1"]],
                color=col, ls="--", lw=1.5, alpha=alp, zorder=4)
    else:
        T_exit = _q("T", "P", s["p_ref_1"], "S", s["s_ref_3"], cycle_config=cycle_config)
        T_rng  = np.linspace(s["T_ref_3"], T_exit, 150)
        line_isen, = ax.plot(np.full_like(T_rng, s["s_ref_3"]), T_rng,
                             color=col, ls="--", lw=1.5, alpha=alp, zorder=4,
                             label=r"$\mathrm{Isentropic\ expansion}$")
        # Horizontal connector from isentropic exit → actual cycle point 4
        ax.plot([s["s_ref_3"], s["s_ref_4"]], [T_exit, T_exit],
                color=col, ls="--", lw=1.5, alpha=alp, zorder=4)
        s_is = np.array([_q("S", "P", p, "H", s["h_ref_3"], cycle_config=cycle_config) for p in p_exp])
        T_is = np.array([_q("T", "P", p, "H", s["h_ref_3"], cycle_config=cycle_config) for p in p_exp])
        v = np.isfinite(s_is)
        line_isenth, = ax.plot(s_is[v], T_is[v], color=col, ls="-.", lw=1.5,
                               alpha=alp, zorder=4,
                               label=r"$\mathrm{Isenthalpic\ expansion}$")

    line_turb_label = r"$\mathrm{HP\ cycle\ incl.\ turbine\ expansion}$"
    line_turb = Line2D([0],[0], color='green', lw=1.5, label=line_turb_label)
    handles = [line_turb, line_isen, line_isenth]
    if include_verification:
        line_verif = Line2D(
            [0], [0], color=COL_VERIFICATION, lw=1.5,
            label=r"$\mathrm{Verification\ cycle\ incl.\ isenthalpic\ expansion}$"
        )
        handles.append(line_verif)

    leg = ax.legend(handles=handles,
                    loc="lower right", bbox_to_anchor=(0.9875, 0.015),
                    fontsize=10, framealpha=0.85)
    fr = leg.get_frame()
    fr.set_facecolor((0.96, 0.92, 0.84, 0.72))
    fr.set_edgecolor('#9C7B53')
    fr.set_linewidth(1.2)
    fr.set_boxstyle('round,pad=0.5')
    leg.set_zorder(11)



# Cycle node labels (station numbers)
# ====================================
def _add_node_labels(ax, cycle_data, diagram_type):
    node_positions = [
        (cycle_data["s_ref_1"], cycle_data["T_ref_1"], cycle_data["h_ref_1"], cycle_data["p_ref_1"], "1"),
        (cycle_data["s_ref_2"], cycle_data["T_ref_2"], cycle_data["h_ref_2"], cycle_data["p_ref_2"], "2"),
        (cycle_data["s_ref_3"], cycle_data["T_ref_3"], cycle_data["h_ref_3"], cycle_data["p_ref_3"], "3"),
        (cycle_data["s_ref_4"], cycle_data["T_ref_4"], cycle_data["h_ref_4"], cycle_data["p_ref_4"], "4"),
    ]
    
    # Get axis limits for offset calculation
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    x_range = xh - xl
    
    # Small offset in fraction of axis range (right of the node, close to it)
    x_offset_frac = 0.008
    if diagram_type == "TS":
        y_offset_frac = 0.012
    else:  # PH diagram with log scale
        y_offset_frac = 0.015
    
    if diagram_type == "TS":
        # Use entropy and temperature
        for s, T, _, _, station in node_positions:
            if np.isfinite(s) and np.isfinite(T):
                x_pos = s + x_offset_frac * x_range
                if station in {"2", "3"}:
                    y_pos = T + y_offset_frac * (yh - yl)
                    valign = "bottom"
                else:
                    y_pos = T - y_offset_frac * (yh - yl)
                    valign = "top"
                
                txt = ax.text(x_pos, y_pos, station,
                             fontsize=12, fontweight='extra bold', color='black',
                             ha='left', va=valign,
                             zorder=13)
                # Add white stroke for semi-transparent background effect
                txt.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white', alpha=0.7),
                                     patheffects.Normal()])
    else:  # PH diagram
        # Use enthalpy and pressure
        for _, _, h, p, station in node_positions:
            if np.isfinite(h) and np.isfinite(p):
                x_pos = h + x_offset_frac * x_range
                # For log scale, offset in log space
                log_yl, log_yh = np.log10(yl), np.log10(yh)
                if station in {"2", "3"}:
                    log_y = np.log10(p) + y_offset_frac * (log_yh - log_yl)
                    valign = "bottom"
                else:
                    log_y = np.log10(p) - y_offset_frac * (log_yh - log_yl)
                    valign = "top"
                y_pos = 10 ** log_y
                
                txt = ax.text(x_pos, y_pos, station,
                             fontsize=12, fontweight='bold', color='black',
                             ha='left', va=valign,
                             zorder=13)
                # Add white stroke for semi-transparent background effect
                txt.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white', alpha=0.7),
                                     patheffects.Normal()])




# Arrow / endpoint label helpers (TS)
# ===================================
def _add_mid_arrow(ax, xv, yv, color, frac=0.18):
    if len(xv) < 2: return
    dx, dy = xv[1]-xv[0], yv[1]-yv[0]
    if np.isclose(dx,0) and np.isclose(dy,0): return
    ax.quiver(xv[0]+0.5*dx, yv[0]+0.5*dy, frac*dx, frac*dy,
              angles='xy', scale_units='xy', scale=1, pivot='middle',
              color=color, width=0.002, headwidth=4.5, headlength=6,
              headaxislength=5, zorder=5)


def _add_endpoint_labels(ax, xv, yv, lbl0, lbl1, color, side=1):
    if len(xv) < 2: return
    ps = np.array([xv[0], yv[0]], dtype=float)
    pe = np.array([xv[1], yv[1]], dtype=float)
    # work in normalised axes coords for a stable offset
    xl, xh = ax.get_xlim(); yl, yh = ax.get_ylim()
    ps_n = np.array([(xv[0]-xl)/(xh-xl), (yv[0]-yl)/(yh-yl)])
    pe_n = np.array([(xv[1]-xl)/(xh-xl), (yv[1]-yl)/(yh-yl)])
    vec  = pe_n - ps_n
    norm = np.linalg.norm(vec)
    tan  = vec/norm if norm > 1e-9 else np.array([1.,0.])
    nor  = side * np.array([-tan[1], tan[0]])
    for lbl, pt, sgn in [(lbl0,(xv[0],yv[0]),-1),(lbl1,(xv[1],yv[1]),1)]:
        off = nor*6. + sgn*tan*15.
        ax.annotate(lbl, xy=pt, xytext=(off[0],off[1]),
                    textcoords='offset points', fontsize=10, color=color,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
                    zorder=6, clip_on=True)



# Performance box
# ===============
def _draw_perf_box(ax, perf):
    txt = (
        rf"$\begin{{array}}{{lrl}}"
        rf"\multicolumn{{3}}{{c}}{{\mathrm{{Performance}}}} \\\hline "
        rf"\mathrm{{COP_{{is}}}}     & \num{{{perf['COP_is']:.2f}}}     & [-] \\"
        rf"\mathrm{{COP_{{turb}}}}   & \num{{{perf['COP_turb']:.2f}}}   & [-] \\"
        rf"\mathrm{{COP_{{isenth}}}} & \num{{{perf['COP_isenth']:.2f}}} & [-] \\"
        rf"\dot{{W}}_{{turb}}        & \num{{{perf['Ẇ_turb']:.0f}}}     & [\mathrm{{W}}] \\"
        rf"\dot{{W}}_{{compr}}       & \num{{{perf['Ẇ_comp']:.0f}}}     & [\mathrm{{W}}] \\"
        rf"\dot{{Q}}_{{out}}     & \num{{{perf['Q_out']:.0f}}}       & [\mathrm{{W}}]"
        rf"\end{{array}}$"
    )
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', zorder=10,
            bbox=dict(facecolor=(0.96,0.92,0.84,0.72), edgecolor='#9C7B53',
                      linewidth=1.2, boxstyle='round,pad=0.5'))


def _compute_turbine_expansion_curve(cycle_data, cycle_config, n=220):
    eta_turb = cycle_config.get("η_turb", None)
    if eta_turb is None:
        return None

    p3 = cycle_data["p_ref_3"]
    p4 = cycle_data["p_ref_4"]
    h3 = cycle_data["h_ref_3"]
    s3 = cycle_data["s_ref_3"]

    p_arr = np.linspace(p3, p4, n)
    h_arr = np.full_like(p_arr, np.nan, dtype=float)
    s_arr = np.full_like(p_arr, np.nan, dtype=float)
    T_arr = np.full_like(p_arr, np.nan, dtype=float)

    for i, p in enumerate(p_arr):
        try:
            h_is = _q("H", "P", p, "S", s3, cycle_config=cycle_config)
            if not np.isfinite(h_is):
                continue
            h = h3 - eta_turb * (h3 - h_is)
            h_arr[i] = h
            s_arr[i] = _q("S", "P", p, "H", h, cycle_config=cycle_config)
            T_arr[i] = _q("T", "P", p, "H", h, cycle_config=cycle_config)
        except Exception:
            pass

    valid = np.isfinite(p_arr) & np.isfinite(h_arr) & np.isfinite(s_arr) & np.isfinite(T_arr)
    if valid.sum() < 3:
        return None
    return p_arr[valid], h_arr[valid], s_arr[valid], T_arr[valid]


def _compute_compressor_curve(cycle_data, cycle_config, n=220):
    eta_compr = cycle_config.get("η_compr", None)
    if eta_compr is None:
        return None

    return _compute_compressor_curve_for_eta(cycle_data, cycle_config, eta_compr, n=n)


def _compute_compressor_curve_for_eta(cycle_data, cycle_config, eta_compr, n=220):
    if eta_compr is None:
        return None

    p1 = cycle_data["p_ref_1"]
    p2 = cycle_data["p_ref_2"]
    h1 = cycle_data["h_ref_1"]
    s1 = cycle_data["s_ref_1"]

    p_arr = np.linspace(p1, p2, n)
    h_arr = np.full_like(p_arr, np.nan, dtype=float)
    s_arr = np.full_like(p_arr, np.nan, dtype=float)
    T_arr = np.full_like(p_arr, np.nan, dtype=float)

    for i, p in enumerate(p_arr):
        try:
            h_is = _q("H", "P", p, "S", s1, cycle_config=cycle_config)
            if not np.isfinite(h_is):
                continue
            if np.isclose(eta_compr, 0.0):
                h = h1
            else:
                h = h1 + (h_is - h1) / eta_compr
            h_arr[i] = h
            s_arr[i] = _q("S", "P", p, "H", h, cycle_config=cycle_config)
            T_arr[i] = _q("T", "P", p, "H", h, cycle_config=cycle_config)
        except Exception:
            pass

    valid = np.isfinite(p_arr) & np.isfinite(h_arr) & np.isfinite(s_arr) & np.isfinite(T_arr)
    if valid.sum() < 3:
        return None
    return p_arr[valid], h_arr[valid], s_arr[valid], T_arr[valid]


def _find_expansion_segment_index(x_arr, y_arr, x3, y3, x4, y4):
    x = np.asarray(x_arr)
    y = np.asarray(y_arr)
    if len(x) < 2:
        return None

    tol_x3 = 1e-6 * max(1.0, abs(x3))
    tol_y3 = 1e-6 * max(1.0, abs(y3))
    tol_x4 = 1e-6 * max(1.0, abs(x4))
    tol_y4 = 1e-6 * max(1.0, abs(y4))

    for i in range(len(x) - 1):
        is_3 = (abs(x[i] - x3) <= tol_x3) and (abs(y[i] - y3) <= tol_y3)
        is_4 = (abs(x[i + 1] - x4) <= tol_x4) and (abs(y[i + 1] - y4) <= tol_y4)
        if is_3 and is_4:
            return i
    return None


def _plot_cycle_with_curved_expansion_ts(ax, cycle_data, ts_data, cycle_config):
    x = np.asarray(ts_data["minor"]["s"])
    y = np.asarray(ts_data["minor"]["T"])
    idx_comp = _find_expansion_segment_index(
        x,
        y,
        cycle_data["s_ref_1"],
        cycle_data["T_ref_1"],
        cycle_data["s_ref_2"],
        cycle_data["T_ref_2"],
    )
    idx = _find_expansion_segment_index(
        x,
        y,
        cycle_data["s_ref_3"],
        cycle_data["T_ref_3"],
        cycle_data["s_ref_4"],
        cycle_data["T_ref_4"],
    )
    comp_curve = _compute_compressor_curve(cycle_data, cycle_config)
    curve = _compute_turbine_expansion_curve(cycle_data, cycle_config)

    if idx_comp is None or idx is None or comp_curve is None or curve is None or idx_comp >= idx:
        ax.plot(x, y, color='green', lw=1.5, zorder=7)
        return

    _, _, s_comp, T_comp = comp_curve
    _, _, s_curve, T_curve = curve
    ax.plot(x[:idx_comp + 1], y[:idx_comp + 1], color='green', lw=1.5, zorder=7)
    ax.plot(s_comp, T_comp, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx_comp + 1:idx + 1], y[idx_comp + 1:idx + 1], color='green', lw=1.5, zorder=7)
    ax.plot(s_curve, T_curve, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx + 1:], y[idx + 1:], color='green', lw=1.5, zorder=7)


def _plot_cycle_with_curved_expansion_ph(ax, cycle_data, ph_data, cycle_config):
    x = np.asarray(ph_data["minor"]["h"])
    y = np.asarray(ph_data["minor"]["p"])
    idx_comp = _find_expansion_segment_index(
        x,
        y,
        cycle_data["h_ref_1"],
        cycle_data["p_ref_1"],
        cycle_data["h_ref_2"],
        cycle_data["p_ref_2"],
    )
    idx = _find_expansion_segment_index(
        x,
        y,
        cycle_data["h_ref_3"],
        cycle_data["p_ref_3"],
        cycle_data["h_ref_4"],
        cycle_data["p_ref_4"],
    )
    comp_curve = _compute_compressor_curve(cycle_data, cycle_config)
    curve = _compute_turbine_expansion_curve(cycle_data, cycle_config)

    if idx_comp is None or idx is None or comp_curve is None or curve is None or idx_comp >= idx:
        ax.plot(x, y, color='green', lw=1.5, zorder=7)
        return

    p_comp, h_comp, _, _ = comp_curve
    p_curve, h_curve, _, _ = curve
    ax.plot(x[:idx_comp + 1], y[:idx_comp + 1], color='green', lw=1.5, zorder=7)
    ax.plot(h_comp, p_comp, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx_comp + 1:idx + 1], y[idx_comp + 1:idx + 1], color='green', lw=1.5, zorder=7)
    ax.plot(h_curve, p_curve, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx + 1:], y[idx + 1:], color='green', lw=1.5, zorder=7)



# Colour palette
# ==============
COL_VERIFICATION = '#0d1f3c' # very dark slate blue (for verification/reference cycles)



# TS renderer
# ===========
def _make_plot_ts(cycle_data, perf, ts_data, cycle_config, thermoplot_config_file, verification_data=None, margins=None):
    warnings.filterwarnings("ignore")
    s_lo, s_hi, T_lo, T_hi = _cycle_bounds_ts(cycle_data, cycle_config, margins)

    # T-s diagram background (saturation dome, critical point and isoline, isolines) from thermoplot
    fig = thermoplot_cached(str(thermoplot_config_file), {
        "fluid_name": cycle_config["refrigerant"], "S_range": [s_lo, s_hi], "T_range": [T_lo, T_hi]})
    ax = fig.get_axes()[0]

    # Keep pinch-point clarifying hot-side entropy by default, but if the
    # plotted T_h_out endpoint would leave the TS page, pin it to state 2.
    coolant_flow = {
        "s": list(ts_data["coolant"]["s"]),
        "T": list(ts_data["coolant"]["T"]),
    }
    heating_flow = ts_data["heating"]
    if len(coolant_flow["s"]) >= 2:
        s_h_out = coolant_flow["s"][-1]
        if np.isfinite(s_h_out) and (s_h_out < s_lo or s_h_out > s_hi):
            coolant_flow["s"][-1] = cycle_data["s_ref_2"]

    # Cycle
    ax.scatter(ts_data["major"]["s"], ts_data["major"]["T"],
               color='orange', marker='o', s=5, zorder=8)
    _plot_cycle_with_curved_expansion_ts(ax, cycle_data, ts_data, cycle_config)

    # Verification/reference cycle (if provided)
    if verification_data is not None:
        T_raw = np.array(verification_data[0, :], dtype=float)  # Row 0: temperature
        s_raw = np.array(verification_data[3, :], dtype=float)  # Row 3: entropy
        T_verif, _, _, s_verif = _verification_path_with_isenthalpic_expansion(
            verification_data,
            cycle_config,
        )
        # In his code, he does not account for the fact that if the compression ends in the two-phase regime, there is no reason
        # to include the inflection point (for Q = 1) to the cycle, so I have to remove this point manually... 
        # for the expansion he does account for this cuz it is typical that the expansion ends in the two-phase domain.
        if np.isclose(T_verif[1], T_verif[2]) and s_verif[2] > s_verif[1]:
            T_verif = np.delete(T_verif, 2)
            s_verif = np.delete(s_verif, 2)
            T_raw = np.delete(T_raw, 2)
            s_raw = np.delete(s_raw, 2)
        ax.plot(s_verif, T_verif, color=COL_VERIFICATION, lw=1.5, zorder=6, label='Reference (MTW)')
        ax.scatter(s_raw, T_raw, color=COL_VERIFICATION, marker='o', s=5, zorder=6)

    _add_expansion_lines(
        ax,
        cycle_data,
        "TS",
        cycle_config,
        include_verification=(verification_data is not None),
    )

    for flow, col in [(coolant_flow,"red"),(heating_flow,"blue")]:
        ax.plot(flow["s"], flow["T"], color=col, marker='o', markersize=2, zorder=12)
        _add_mid_arrow(ax, flow["s"], flow["T"], col)
    _add_endpoint_labels(ax, coolant_flow["s"], coolant_flow["T"],
                         r"$T_{h,\mathrm{in}}$", r"$T_{h,\mathrm{out}}$", "red", side=1)
    _add_endpoint_labels(ax, heating_flow["s"], heating_flow["T"],
                         r"$T_{c,\mathrm{in}}$", r"$T_{c,\mathrm{out}}$", "blue",  side=-1)

    # Add cycle node labels (station numbers 1-4)
    _add_node_labels(ax, cycle_data, "TS")

    ax.tick_params(axis='both', which='major', labelsize=12)
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig



# PH renderer
# ===========
def _make_plot_ph(cycle_data, perf, ph_data, cycle_config, thermoplot_config_file, verification_data=None, margins=None):
    warnings.filterwarnings("ignore")
    h_lo, h_hi, p_lo, p_hi = _cycle_bounds_ph(cycle_data, cycle_config, margins)
    p_lo = max(p_lo, 1e2)

    # p-h diagram background (saturation dome, critical point and isoline, isolines) from thermoplot
    fig = thermoplot_cached(str(thermoplot_config_file), {
        "fluid_name": cycle_config["refrigerant"], "H_range": [h_lo, h_hi], "P_range": [p_lo, p_hi]})
    ax = fig.get_axes()[0]

    # Cycle
    ax.scatter(ph_data["major"]["h"], ph_data["major"]["p"],
               color='orange', marker='o', s=5, zorder=8)
    _plot_cycle_with_curved_expansion_ph(ax, cycle_data, ph_data, cycle_config)

    # Verification/reference cycle (if provided)
    if verification_data is not None:
        p_raw = np.array(verification_data[1, :], dtype=float)  # Row 1: pressure
        h_raw = np.array(verification_data[2, :], dtype=float)  # Row 2: enthalpy
        _, p_verif, h_verif, _ = _verification_path_with_isenthalpic_expansion(
            verification_data,
            cycle_config,
        )
        if np.isclose(p_verif[1], p_verif[2]) and h_verif[2] > h_verif[1]:
            p_verif = np.delete(p_verif, 2)
            h_verif = np.delete(h_verif, 2)
            p_raw = np.delete(p_raw, 2)
            h_raw = np.delete(h_raw, 2)
        ax.plot(h_verif, p_verif, color=COL_VERIFICATION, lw=1.5, zorder=6, label='Reference (MTW)')
        ax.scatter(h_raw, p_raw, color=COL_VERIFICATION, marker='o', s=5, zorder=6)

    _add_expansion_lines(
        ax,
        cycle_data,
        "PH",
        cycle_config,
        include_verification=(verification_data is not None),
    )

    # Add cycle node labels (station numbers 1-4)
    _add_node_labels(ax, cycle_data, "PH")

    ax.tick_params(axis='both', which='major', labelsize=12)
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig



# Empty diagrams
# ==============
def _make_empty_plot_ts(refrigerant, thermoplot_config_file, show_spinodal=False):
    warnings.filterwarnings("ignore")
    cycle_config = {"refrigerant": refrigerant}
    T_crit = _q("Tcrit", cycle_config=cycle_config)
    T_trip = _q("Ttriple", cycle_config=cycle_config)
    T_arr = np.linspace(T_trip*1.001, T_crit*0.9999, 300)
    s_dome = np.concatenate([_q("S", "T", T_arr, "Q", 0, cycle_config=cycle_config),
                             _q("S", "T", T_arr, "Q", 1, cycle_config=cycle_config)])
    T_dome = np.concatenate([T_arr, T_arr])
    valid_dome = np.isfinite(s_dome)
    if valid_dome.sum() < 5:
        raise ValueError(f"Unable to build TS saturation dome for {refrigerant}.")

    s_lo_raw = np.min(s_dome[valid_dome])
    s_hi_raw = np.max(s_dome[valid_dome])
    T_lo_raw = np.min(T_dome[valid_dome])
    T_hi_raw = np.max(T_dome[valid_dome])
    ds = max(s_hi_raw - s_lo_raw, 1e-6)
    dT = max(T_hi_raw - T_lo_raw, 1e-6)

    # Expand the visible domain beyond the saturation dome.
    s_lo = s_lo_raw - 0.12 * ds
    s_hi = s_hi_raw + 0.18 * ds
    T_lo = T_lo_raw
    T_hi = max(T_hi_raw + 0.11 * dT, T_crit * 1.175 if np.isfinite(T_crit) else T_hi_raw + 0.11 * dT)

    fig = thermoplot_cached(str(thermoplot_config_file), {
        "fluid_name": refrigerant, "S_range": [s_lo, s_hi], "T_range": [T_lo, T_hi]})
    ax = fig.get_axes()[0]

    # Optional: true spinodal lines computed via dP/drho|T = 0
    if show_spinodal:
        try:
            _draw_spinodal(ax, *_get_spinodal_ts(refrigerant, T_lo=T_lo, T_hi=T_hi))
        except Exception:
            pass

    fig.tight_layout()
    return fig


def _make_empty_plot_ph(refrigerant, thermoplot_config_file, show_spinodal=False):
    warnings.filterwarnings("ignore")
    cycle_config = {"refrigerant": refrigerant}
    p_crit = _q("Pcrit", cycle_config=cycle_config)
    T_trip = _q("Ttriple", cycle_config=cycle_config)
    p_trip = _q("P", "T", T_trip*1.001, "Q", 0, cycle_config=cycle_config)
    p_arr = np.geomspace(p_trip*1.01, p_crit*0.9999, 300)
    h_dome = np.concatenate([_q("H", "P", p_arr, "Q", 0, cycle_config=cycle_config),
                             _q("H", "P", p_arr, "Q", 1, cycle_config=cycle_config)])
    p_dome = np.concatenate([p_arr, p_arr])
    valid_dome = np.isfinite(h_dome)
    if valid_dome.sum() < 5:
        raise ValueError(f"Unable to build PH saturation dome for {refrigerant}.")

    h_lo_raw = np.min(h_dome[valid_dome])
    h_hi_raw = np.max(h_dome[valid_dome])
    p_lo_raw = max(np.min(p_dome[valid_dome]), 1e2)
    p_hi_raw = np.max(p_dome[valid_dome])
    dh = max(h_hi_raw - h_lo_raw, 1e-6)

    # Expand the visible domain beyond the saturation dome.
    h_lo = h_lo_raw - 0.12 * dh
    h_hi = h_hi_raw + 0.18 * dh
    p_lo = p_lo_raw
    p_hi = max(p_hi_raw * 1.6, p_crit * 1.4 if np.isfinite(p_crit) else p_hi_raw * 1.6)

    fig = thermoplot_cached(str(thermoplot_config_file), {
        "fluid_name": refrigerant, "H_range": [h_lo, h_hi], "P_range": [p_lo, p_hi]})
    ax = fig.get_axes()[0]

    # Optional: true spinodal lines computed via dP/drho|T = 0
    if show_spinodal:
        try:
            _draw_spinodal(ax, *_get_spinodal_ph(refrigerant, p_lo=p_lo, p_hi=p_hi))
        except Exception:
            pass

    fig.tight_layout()
    return fig



# Public entry points for TS or PH plotting
# =========================================
def make_thdy_plot(
    cycle_data,
    perf,
    diagram_type,
    cycle_config,
    output_dir,
    ts_data=None,
    ph_data=None,
    verification_data=None,
    file_suffix="",
    margins=None,
    thermoplot_config_file=None,
    verbose=True,
):
    refrigerant = cycle_config["refrigerant"]
    configure_matplotlib()
    if thermoplot_config_file is None:
        thermoplot_config_file = THERMOPLOT_CONFIG_DIR / f"{diagram_type}.ini"
    fig = _make_plot_ts(cycle_data, perf, ts_data, cycle_config, thermoplot_config_file, verification_data=verification_data, margins=margins) if diagram_type == "TS" \
          else _make_plot_ph(cycle_data, perf, ph_data, cycle_config, thermoplot_config_file, verification_data=verification_data, margins=margins)
    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"Conceptual HP Cycle - {refrigerant} - {diagram_type}{file_suffix}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    return fig


def make_empty_thdy_plot(diagram_type, refrigerant, output_dir="results/substance_thermodynamic_diagrams", show_spinodal=False, thermoplot_config_file=None, verbose=True):
    configure_matplotlib()
    if thermoplot_config_file is None:
        thermoplot_config_file = THERMOPLOT_CONFIG_DIR / f"{diagram_type}.ini"
    fig = (_make_empty_plot_ts(refrigerant, thermoplot_config_file, show_spinodal=show_spinodal) if diagram_type == "TS" 
           else _make_empty_plot_ph(refrigerant, thermoplot_config_file, show_spinodal=show_spinodal))
    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"Thermodynamic Diagram - {refrigerant} - {diagram_type}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    return fig


def make_cycle_plots(results_file, ts_margins=None, ph_margins=None, opt_progress_figsize=(10.0, 3.0)):
    """Make the T-s, p-h and optimizer progress diagrams of a CHPC analysis from its results file.

    The diagrams are saved in the folder of the refrigerant, one level above the data folder of the 
    results file, i.e. results/heat_pump_thermodynamic_diagrams/<refrigerant>/.

    Args:
        results_file: Path of the .npz results file written by main.py.
        ts_margins: Optional margins around the cycle in the T-s diagram, e.g. {"T_top": 0.5}, see TS_MARGINS.
        ph_margins: Optional margins around the cycle in the p-h diagram, e.g. {"h_left": 0.5}, see PH_MARGINS.
        opt_progress_figsize: Figure size of the optimizer progress diagram, in inches.

    Returns:
        T-s, p-h and optimizer progress figures. The latter is None for three_temp_fix, which has no optimization trace.
    """
    results = load_results(results_file)
    cycle_data, perf, cycle_config = results["cycle_data"], results["perf"], results["cycle_config"]
    file_suffix = method_suffix(results["general_config"]["calculation_method"], separator=" - ")
    output_dir = Path(results_file).parents[2]
    fig_ts = make_thdy_plot(cycle_data, perf, "TS", cycle_config, output_dir, ts_data=results["ts_data"],
                            file_suffix=file_suffix, margins=ts_margins)
    fig_ph = make_thdy_plot(cycle_data, perf, "PH", cycle_config, output_dir, ph_data=results["ph_data"],
                            file_suffix=file_suffix, margins=ph_margins)
    fig_opt = make_optimizer_progress_plot(cycle_data.get("optimization_trace"), cycle_config, output_dir,
                                           figsize=opt_progress_figsize, file_suffix=file_suffix)
    return fig_ts, fig_ph, fig_opt



# Public entry point for COP_vs_eff_investigation
# ===============================================
def make_COP_vs_eff_plot(X, Y, Z, refrigerant, output_dir="results/COP_investigations", file_suffix=""):
    configure_matplotlib()      
    output_path = Path(output_dir) / refrigerant
    output_path.mkdir(parents=True, exist_ok=True)
    cop_type = "COP_turb"
    cop_type_latex = _cop_type_to_latex(cop_type)

    # 2D heatmap with labeled isolines
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    # Filled heatmap
    cf = ax.contourf(X, Y, Z, levels=30, cmap='viridis', extend='neither')

    # Black contour lines + value labels on the isolines 
    cl = ax.contour(X, Y, Z, levels=12, colors='black', linewidths=0.8, linestyles='-')
    ax.clabel(cl, inline=True, fontsize=9, fmt='%.2f', colors='white',
              manual=False, inline_spacing=3)

    # Colorbar and labels
    cbar = plt.colorbar(cf, ax=ax, label=rf'${cop_type_latex}$')
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r'$\eta_{\mathrm{turb}}$')
    ax.set_ylabel(r'$\eta_{\mathrm{compr}}$')

    # Save 
    fname2d = f"{cop_type}_vs_Efficiencies_{refrigerant}{file_suffix}.pdf"
    fig.savefig(output_path / fname2d, dpi=1000, bbox_inches='tight')
    logger.info(f"Saved: {output_path / fname2d}")

    # 3D surface
    fig3d = plt.figure(figsize=(11, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', linewidth=0.2,
                             antialiased=True, alpha=0.95, rstride=1, cstride=1)

    fig3d.colorbar(surf, ax=ax3d, shrink=0.6, aspect=10, label=rf'${cop_type_latex}$')

    ax3d.set_xlabel(r'$\eta_{\mathrm{turb}}$')
    ax3d.set_ylabel(r'$\eta_{\mathrm{compr}}$')
    ax3d.set_zlabel(rf'${cop_type_latex}$')

    fname3d = f"{cop_type}_vs_Efficiencies_{refrigerant}{file_suffix}_3D.pdf"
    fig3d.savefig(output_path / fname3d, dpi=800, bbox_inches='tight')
    logger.info(f"Saved: {output_path / fname3d}")
    return fig, fig3d


def make_optimizer_progress_plot(
    optimization_trace,
    cycle_config,
    output_dir="results/heat_pump_thermodynamic_diagrams",
    figsize=(10.0, 3.0),
    file_suffix="",
    verbose=True,
):
    if not optimization_trace:
        return None

    objective = np.asarray(optimization_trace.get("objective", []), dtype=float)
    failed = np.asarray(optimization_trace.get("failed", []), dtype=bool)
    if objective.size == 0:
        return None
    eval_idx = np.arange(1, objective.size + 1)
    best_so_far = np.minimum.accumulate(objective)

    # Plot in COP-equivalent space to avoid distorted scales from infeasible values.
    # Valid points use COP = -objective (objective is minimized), infeasible are pinned to 0.
    cop_eval = np.where(failed, 0.0, -objective)
    cop_best_so_far = np.maximum(0.0, -best_so_far)

    n_valid = int(np.sum(~failed))
    n_failed = int(np.sum(failed))

    configure_matplotlib()
    refrigerant = cycle_config["refrigerant"]
    fig, ax = plt.subplots(figsize=figsize)

    # Keep typography consistent with large-format comparison figures.
    label_fontsize = 14
    tick_fontsize = 14
    legend_fontsize = 14

    valid_mask = ~failed
    if np.any(valid_mask):
        ax.scatter(eval_idx[valid_mask], cop_eval[valid_mask], s=10, alpha=0.45,
                  color="#1a3a6b", label=rf"$\mathrm{{Valid\ candidate\ (n={n_valid})}}$")
    if np.any(failed):
        ax.scatter(eval_idx[failed], np.zeros(np.sum(failed)), s=12, alpha=0.8, marker="x",
                  color="#d64545", label=rf"$\mathrm{{Infeasible\ candidate\ (m={n_failed})}}$")

    ax.plot(eval_idx, cop_best_so_far, color="black", lw=1.1,
            label=r"$\mathrm{Best\ so\ far}$", zorder=3)
    ax.axhline(0.0, color="#d64545", ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel(r"$\mathrm{Objective\ evaluation}\ [-]$", fontsize=label_fontsize)
    ax.set_ylabel(r"$\mathrm{COP}_{\mathrm{turb}}\ [-]$", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=legend_fontsize, framealpha=0.85)

    fig.tight_layout()

    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"Optimizer Progress - {refrigerant}{file_suffix}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    return fig
