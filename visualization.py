# visualization.py
# ================
# CoolProp-backed TS / PH diagram rendering.
 
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import CoolProp
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import PropertyPlot
from matplotlib.lines import Line2D
 
from config import refrigerant, resolution
from thermodynamics import isobar_segment
 
 
# LaTeX / font setup
# ==================
def configure_matplotlib():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{siunitx}\sisetup{group-separator={\,},group-minimum-digits=4}"
    })
 
 
# Isoline helpers
# ===============
def format_isoline_label(prop_key, value, include_name=False):
    if prop_key == CoolProp.iP:
        v = f"{value/1e6:.2f}"
        return rf"$p={v}\,\mathrm{{MPa}}$" if include_name else rf"${v}$"
    if prop_key == CoolProp.iQ:
        v = f"{value:.2f}"
        return rf"$x={v}$" if include_name else rf"${v}$"
    if prop_key == CoolProp.iHmass:
        v = f"{value/1e3:.0f}"
        return rf"$h={v}\,\mathrm{{kJ/kg}}$" if include_name else rf"${v}$"
    if prop_key == CoolProp.iSmass:
        v = f"{value/1000:.2f}"
        return rf"$s={v}\,\mathrm{{kJ/kg·K}}$" if include_name else rf"${v}$"
    if prop_key == CoolProp.iT:
        v = f"{value:.0f}"
        return rf"$T={v}\,\mathrm{{K}}$" if include_name else rf"${v}$"
    return rf"${value:.2f}$"
 
 
def find_intersection_with_isotherm(x_data, y_data, T_target=270.0):
    """Return (x, y) where an isoline crosses T_target, or closest point."""
    x = np.array(x_data)
    y = np.array(y_data)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    crossings = np.where(np.diff(np.sign(y - T_target)))[0]
    if len(crossings) > 0:
        idx = crossings[0]
        dy = y[idx + 1] - y[idx]
        frac = 0.5 if abs(dy) < 1e-9 else (T_target - y[idx]) / dy
        return x[idx] + frac * (x[idx + 1] - x[idx]), T_target
    idx = np.argmin(np.abs(y - T_target))
    return x[idx], y[idx]
 
 
def add_isoline_labels(ax_obj, coolplot_obj, prop_key, color, side, diagram_type):
    if prop_key not in coolplot_obj.isolines:
        return
    iso_lines = coolplot_obj.isolines[prop_key]
    if not iso_lines:
        return
 
    for k, iso in enumerate(iso_lines):
        x_data = np.array(iso.x)
        y_data = np.array(iso.y)
        valid = np.isfinite(x_data) & np.isfinite(y_data)
        x_data, y_data = x_data[valid], y_data[valid]
 
        xlim, ylim = ax_obj.get_xlim(), ax_obj.get_ylim()
        inside = ((x_data >= xlim[0]) & (x_data <= xlim[1]) &
                  (y_data >= ylim[0]) & (y_data <= ylim[1]))
        if np.sum(inside) < 3:
            continue
        x_data, y_data = x_data[inside], y_data[inside]
 
        # Label placement
        if prop_key == CoolProp.iQ:
            if diagram_type == "TS":
                inter = find_intersection_with_isotherm(x_data, y_data, T_target=265.0)
                if inter is None:
                    i_label = len(x_data) // 2
                    x_target, y_target = x_data[i_label], y_data[i_label]
                else:
                    x_target, y_target = inter
                sort_idx = np.argsort(x_data)
                x_data, y_data = x_data[sort_idx], y_data[sort_idx]
                i_label = np.argmin(np.abs(x_data - x_target))
            else:
                i_label = len(x_data) // 2
                x_target, y_target = x_data[i_label], y_data[i_label]
        else:
            x_min, x_max = np.min(x_data), np.max(x_data)
            if diagram_type == "TS" and prop_key == CoolProp.iHmass:
                target_k = int(len(iso_lines) * 0.75)
                frac = 0.88 if k == target_k else 0.93
            elif diagram_type == "TS" and prop_key == CoolProp.iP:
                frac = 0.92
            else:
                frac = 0.92 if side == "right" else 0.08
            x_target = x_min + frac * (x_max - x_min)
            i_label = np.argmin(np.abs(x_data - x_target))
            y_target = y_data[i_label]
 
        i_label = max(1, min(len(x_data) - 2, i_label))
        p_prev = ax_obj.transData.transform((x_data[i_label - 1], y_data[i_label - 1]))
        p_next = ax_obj.transData.transform((x_data[i_label + 1], y_data[i_label + 1]))
        angle = np.degrees(np.arctan2(p_next[1] - p_prev[1], p_next[0] - p_prev[0]))
        if angle > 90:  angle -= 180
        if angle < -90: angle += 180
 
        if diagram_type == "TS" and prop_key == CoolProp.iHmass:
            target_k = int(len(iso_lines) * 0.75)
            include_name = (k == target_k)
        else:
            include_name = (k == len(iso_lines) // 2)
 
        label = format_isoline_label(prop_key, iso.value, include_name=include_name)
        ax_obj.text(
            x_target, y_target, label,
            color=color, fontsize=7, rotation=angle, rotation_mode='anchor',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6, clip_on=True
        )
 
 
# Arrow / annotation helpers
# ==========================
def add_mid_arrow(ax_obj, x_vals, y_vals, color, frac=0.18):
    if len(x_vals) < 2 or len(y_vals) < 2:
        return
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
        return
    ax_obj.quiver(
        x_vals[0] + 0.5 * dx, y_vals[0] + 0.5 * dy,
        frac * dx, frac * dy,
        angles='xy', scale_units='xy', scale=1, pivot='middle',
        color=color, width=0.003, headwidth=4.5, headlength=6, headaxislength=5, zorder=5
    )
 
 
def add_endpoint_temperature_labels(ax_obj, x_vals, y_vals, start_label, end_label, color, side=1):
    if len(x_vals) < 2 or len(y_vals) < 2:
        return
    p_start = np.array(ax_obj.transData.transform((x_vals[0], y_vals[0])), dtype=float)
    p_end   = np.array(ax_obj.transData.transform((x_vals[1], y_vals[1])), dtype=float)
    vec  = p_end - p_start
    norm = np.linalg.norm(vec)
    tangent = vec / norm if not np.isclose(norm, 0.0) else np.array([1.0, 0.0])
    normal  = side * np.array([-tangent[1], tangent[0]])
 
    for label, pt, sign in [(start_label, (x_vals[0], y_vals[0]), -1),
                             (end_label,   (x_vals[1], y_vals[1]),  1)]:
        offset = normal * 6.0 + sign * tangent * 15.0
        ax_obj.annotate(
            label, xy=pt, xytext=(offset[0], offset[1]),
            textcoords='offset points', fontsize=8, color=color,
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6, clip_on=True
        )
 
 
# Expansion reference lines
# =========================
def add_expansion_lines(ax, state, perf, diagram_type):
    """Overlay isentropic and isenthalpic expansion lines plus a legend."""
    s = state
    p_exp_range = np.linspace(s["p_ref_3"], s["p_ref_1"], num=100)
    line_color = "#02220E"
    line_alpha = 0.75
 
    if diagram_type == "PH":
        h_isenth = np.full_like(p_exp_range, s["h_ref_3"])
        line_isenth, = ax.plot(h_isenth, p_exp_range, color=line_color, linestyle="-",
                               linewidth=1.5, alpha=line_alpha, zorder=4,
                               label=r"$\mathrm{isenthalpic\ expansion}$")
        h_isen = np.array([
            _safe_props("H", "P", pp, "S", s["s_ref_3"]) for pp in p_exp_range
        ])
        valid = np.isfinite(h_isen)
        line_isen, = ax.plot(h_isen[valid], p_exp_range[valid], color=line_color, linestyle="--",
                             linewidth=1.5, alpha=line_alpha, zorder=4,
                             label=r"$\mathrm{isentropic\ expansion}$")
    else:  # TS
        T_isen_exit = PropsSI("T", "P", s["p_ref_1"], "S", s["s_ref_3"], f"REFPROP::{refrigerant}")
        T_isen_range = np.linspace(s["T_ref_3"], T_isen_exit, num=100)
        line_isen, = ax.plot(np.full_like(T_isen_range, s["s_ref_3"]), T_isen_range,
                             color=line_color, linestyle="--", linewidth=1.5, alpha=line_alpha,
                             zorder=4, label=r"$\mathrm{isentropic\ expansion}$")
 
        s_isenth = np.array([_safe_props("S", "P", pp, "H", s["h_ref_3"]) for pp in p_exp_range])
        T_isenth = np.array([_safe_props("T", "P", pp, "H", s["h_ref_3"]) for pp in p_exp_range])
        valid = np.isfinite(s_isenth)
        line_isenth, = ax.plot(s_isenth[valid], T_isenth[valid],
                               color=line_color, linestyle="-", linewidth=1.5, alpha=line_alpha,
                               zorder=4, label=r"$\mathrm{isenthalpic\ expansion}$")
 
    line_turb = Line2D([0], [0], color='green', linewidth=1.5,
                       label=r"$\mathrm{turbine\ expansion}$")
    leg = ax.legend(handles=[line_isen, line_turb, line_isenth],
                    loc="lower right", bbox_to_anchor=(0.9875, 0.015), fontsize=8, framealpha=0.85)
    frame = leg.get_frame()
    frame.set_facecolor((0.96, 0.92, 0.84, 0.72))
    frame.set_edgecolor('#9C7B53')
    frame.set_linewidth(1.2)
    frame.set_boxstyle('round,pad=0.5')
    leg.set_zorder(11)
 
 
def _safe_props(out, *args):
    try:
        return PropsSI(out, *args, f"REFPROP::{refrigerant}")
    except ValueError:
        return np.nan
 
 
# Main plot function
# ==================
def make_plot(state, perf, diagram_type, ts_data=None, ph_data=None):
    configure_matplotlib()
    warnings.filterwarnings("ignore")
 
    coolplot = PropertyPlot(f"REFPROP::{refrigerant}", diagram_type, unit_system='SI', tp_limits='ACHP')
 
    n_pts_iso = 150 if resolution == "low" else 1500
    n_pts_iso_more = 300 if resolution == "low" else 3000
 
    if diagram_type == "TS":
        coolplot.calc_isolines(CoolProp.iQ,     num=10, points=n_pts_iso)
        coolplot.calc_isolines(CoolProp.iP,     num=12, points=n_pts_iso_more)
        coolplot.calc_isolines(CoolProp.iHmass, num=20, points=n_pts_iso_more)
    else:  # PH
        coolplot.calc_isolines(CoolProp.iQ,     num=10, points=n_pts_iso)
        coolplot.calc_isolines(CoolProp.iSmass, num=12, points=n_pts_iso_more)
        coolplot.calc_isolines(CoolProp.iT,     num=20, points=n_pts_iso_more)
 
    # Extract isolines data
    isolines = {}
    for key, iso_list in coolplot.isolines.items():
        isolines[key] = {"x": [], "y": [], "value": []}
        for iso in iso_list:
            isolines[key]["x"].append(iso.x)
            isolines[key]["y"].append(iso.y)
            isolines[key]["value"].append(iso.value)
 
    fig = coolplot.figure
    ax  = fig.gca()
 
    # Draw isolines
    cmap   = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(isolines))]
    for i, (key, col) in enumerate(zip(isolines.keys(), colors)):
        for xd, yd in zip(isolines[key]["x"], isolines[key]["y"]):
            ax.plot(xd, yd, color=col, zorder=i, linewidth=0.6)
 
    # Critical point
    p_crit = PropsSI("Pcrit", f"REFPROP::{refrigerant}")
    T_crit = PropsSI("Tcrit", f"REFPROP::{refrigerant}")
    if diagram_type == "TS":
        s_crit = PropsSI("S", "P", p_crit, "T", T_crit, f"REFPROP::{refrigerant}")
        ax.plot(s_crit, T_crit, marker="o", markerfacecolor="yellow", markersize=5, markeredgecolor='black')
    else:
        h_crit = PropsSI("H", "T", T_crit, "P", p_crit, f"REFPROP::{refrigerant}")
        ax.plot(h_crit, p_crit, marker="o", markerfacecolor="yellow", markersize=5, markeredgecolor='black')
 
    # Cycle path
    if diagram_type == "TS" and ts_data:
        ax.scatter(ts_data["major"]["s"], ts_data["major"]["T"], color='orange', marker='o', s=5, zorder=8)
        ax.plot(ts_data["minor"]["s"],    ts_data["minor"]["T"], color='green', linewidth=1.5, zorder=7)
    elif diagram_type == "PH" and ph_data:
        ax.scatter(ph_data["major"]["h"], ph_data["major"]["p"], color='orange', marker='o', s=5, zorder=8)
        ax.plot(ph_data["minor"]["h"],    ph_data["minor"]["p"], color='green', linewidth=1.5, zorder=7)
 
        # Force PH axes so the cycle is always the central component.
        # 1. Compute tight bounds around the cycle with 15% padding on each side.
        all_h = [v for v in ph_data["minor"]["h"] if np.isfinite(v)]
        all_p = [v for v in ph_data["minor"]["p"] if np.isfinite(v)]
        h_lo, h_hi = min(all_h), max(all_h)
        p_lo, p_hi = min(all_p), max(all_p)
        h_pad = 0.15 * (h_hi - h_lo)
        p_pad = 0.15 * (p_hi - p_lo)
        h_lo, h_hi = h_lo - h_pad, h_hi + h_pad
        p_lo, p_hi = p_lo - p_pad, p_hi + p_pad
        # 2. Expand to include the full saturation dome visible in the default view.
        default_xlim = ax.get_xlim()
        default_ylim = ax.get_ylim()
        h_lo = min(h_lo, default_xlim[0])
        h_hi = max(h_hi, default_xlim[1])
        p_lo = min(p_lo, default_ylim[0])
        p_hi = max(p_hi, default_ylim[1])
        ax.set_xlim(h_lo, h_hi)
        ax.set_ylim(p_lo, p_hi)
 
    # Expansion reference lines + legend
    add_expansion_lines(ax, state, perf, diagram_type)
 
    # Coolant / heating flow (TS only)
    if diagram_type == "TS" and ts_data:
        for flow, color in [(ts_data["coolant"], "blue"), (ts_data["heating"], "red")]:
            ax.plot(flow["s"], flow["T"], color=color, marker="o", markersize=2, zorder=12)
            add_mid_arrow(ax, flow["s"], flow["T"], color=color)
        add_endpoint_temperature_labels(ax, ts_data["coolant"]["s"], ts_data["coolant"]["T"],
                                        r"$T_{c,\mathrm{in}}$", r"$T_{c,\mathrm{out}}$", "blue",  side=1)
        add_endpoint_temperature_labels(ax, ts_data["heating"]["s"], ts_data["heating"]["T"],
                                        r"$T_{h,\mathrm{in}}$", r"$T_{h,\mathrm{out}}$", "red", side=-1)
 
    # Critical isobar (TS) / critical isotherm (PH)
    if diagram_type == "TS":
        s_range = ax.get_xlim()
        s_c, T_c = isobar_segment(s_range[0], s_range[1], p_crit)
        ax.plot(s_c, T_c, color="black", linestyle=":", linewidth=1.0, zorder=1)
    else:
        h_min, h_max = ax.get_xlim()
        p_min, p_max = ax.get_ylim()
        for p_arr in [np.linspace(max(1e3, p_min * 0.05), p_crit, 500),
                      np.linspace(p_crit, min(10 * p_crit, p_max * 1.1), 500)]:
            h_arr = np.array([_safe_props("H", "P", pp, "T", T_crit) for pp in p_arr])
            valid = np.isfinite(h_arr) & (h_arr >= h_min) & (h_arr <= h_max)
            if np.any(valid):
                ax.plot(h_arr[valid], p_arr[valid], color="black", linestyle=":", linewidth=1.0, zorder=1)
 
    # Isoline labels
    fig.canvas.draw()
    q_col = colors[list(isolines.keys()).index(CoolProp.iQ)]
    add_isoline_labels(ax, coolplot, CoolProp.iQ, q_col, side="left", diagram_type=diagram_type)
 
    if diagram_type == "TS":
        if CoolProp.iP in isolines:
            add_isoline_labels(ax, coolplot, CoolProp.iP,
                               colors[list(isolines.keys()).index(CoolProp.iP)], side="right",
                               diagram_type=diagram_type)
        if CoolProp.iHmass in isolines:
            add_isoline_labels(ax, coolplot, CoolProp.iHmass,
                               colors[list(isolines.keys()).index(CoolProp.iHmass)], side="right",
                               diagram_type=diagram_type)
    else:
        if CoolProp.iSmass in isolines:
            add_isoline_labels(ax, coolplot, CoolProp.iSmass,
                               colors[list(isolines.keys()).index(CoolProp.iSmass)], side="right",
                               diagram_type=diagram_type)
        if CoolProp.iT in isolines:
            add_isoline_labels(ax, coolplot, CoolProp.iT,
                               colors[list(isolines.keys()).index(CoolProp.iT)], side="right",
                               diagram_type=diagram_type)
 
    # Axis labels
    if diagram_type == "TS":
        ax.set_xlabel("$s [J/kg/K]$")
        ax.set_ylabel("$T [K]$")
    else:
        ax.set_xlabel("$h [kJ/kg]$")
        ax.set_ylabel("$p [Pa]$")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}$"))
 
    # Performance box
    textstr = (
        rf"$\begin{{array}}{{lrl}}"
        rf"\multicolumn{{3}}{{c}}{{\mathrm{{Performance}}}} \\\hline "
        rf"\mathrm{{COP_{{is}}}} & \num{{{perf['COP_is']:.2f}}} & [-] \\"
        rf"\mathrm{{COP_{{turb}}}} & \num{{{perf['COP_turb']:.2f}}} & [-] \\"
        rf"\mathrm{{COP_{{isenth}}}} & \num{{{perf['COP_isenth']:.2f}}} & [-] \\"
        rf"\dot{{W}}_{{turb}} & \num{{{perf['Ẇ_turb']:.0f}}} & [\mathrm{{W}}] \\"
        rf"\dot{{W}}_{{compr}} & \num{{{perf['Ẇ_comp']:.0f}}} & [\mathrm{{W}}] \\"
        rf"\dot{{Q}}_{{in,turb}} & \num{{{perf['Q_in']:.0f}}} & [\mathrm{{W}}]"
        rf"\end{{array}}$"
    )
    ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', zorder=10,
            bbox=dict(facecolor=(0.96, 0.92, 0.84, 0.72), edgecolor='#9C7B53',
                      linewidth=1.2, boxstyle='round,pad=0.5'))
 
    ax.set_title(f"$Conceptual$ $Heat$ $Pump$ $Cycle$   $-$   ${refrigerant}$")
    fname = f"Conceptual HP Cycle - {refrigerant} - {diagram_type}.pdf"
    fig.savefig(fname, dpi=1000, bbox_inches="tight")
    print(f"Saved: {fname}")
    return fig
 

