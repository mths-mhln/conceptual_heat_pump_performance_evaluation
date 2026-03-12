# visualization.py
# ================
# All isolines are computed manually via PropsSI over axis bounds that are
# derived purely from the cycle state.  CoolProp's PropertyPlot utility is
# not used, so the view is always centred on the cycle.
#
# Layout convention (same for TS and PH):
#   Performance box  — top-left  → extra left + top padding
#   Expansion legend — bottom-right → extra right + bottom padding

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from CoolProp.CoolProp import PropsSI

from config import refrigerant, resolution
from thermodynamics import isobar_segment   # reused for TS critical-isobar


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def configure_matplotlib():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": (
            r"\usepackage{siunitx}"
            r"\sisetup{group-separator={\,},group-minimum-digits=4}"
        ),
    })


def _q(out, *args):
    """PropsSI call that returns np.nan on failure."""
    try:
        return PropsSI(out, *args, f"REFPROP::{refrigerant}")
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Axis-bound calculation
# ─────────────────────────────────────────────────────────────────────────────

def _cycle_bounds_ts(state, pad=0.35):
    s_vals = [state[k] for k in ("s_ref_1","s_ref_2","s_ref_3","s_ref_4")]
    T_vals = [state[k] for k in ("T_ref_1","T_ref_2","T_ref_3","T_ref_4")]
    s_lo, s_hi = min(s_vals), max(s_vals)
    T_lo, T_hi = min(T_vals), max(T_vals)
    ds = (s_hi - s_lo) * pad
    dT = (T_hi - T_lo) * pad
    # extra left for perf box, extra top for perf box height
    return (s_lo - ds*2.2, s_hi + ds*1.5, T_lo - dT*1.5, T_hi + dT*3.0)


def _cycle_bounds_ph(state, pad=0.35):
    h_vals = [state[k] for k in ("h_ref_1","h_ref_2","h_ref_3","h_ref_4")]
    p_vals = [state[k] for k in ("p_ref_1","p_ref_2","p_ref_3","p_ref_4")]
    h_lo, h_hi = min(h_vals), max(h_vals)
    p_lo, p_hi = min(p_vals), max(p_vals)
    dh = (h_hi - h_lo) * pad
    dp = (p_hi - p_lo) * pad
    # extra left for perf box, extra top for dome + box
    return (h_lo - dh*2.2, h_hi + dh*1.5, p_lo - dp*1.5, p_hi + dp*4.0)


# ─────────────────────────────────────────────────────────────────────────────
# Saturation dome
# ─────────────────────────────────────────────────────────────────────────────

def _saturation_dome_ts(n=400):
    p_crit = _q("Pcrit", "T", 300, "Q", 1)
    T_crit = _q("Tcrit", "T", 300, "Q", 1)
    T_trip = _q("Ttriple", "T", 300, "Q", 1)
    T_arr  = np.linspace(T_trip * 1.001, T_crit * 0.9999, n)
    s_liq  = np.array([_q("S","T",T,"Q",0) for T in T_arr])
    s_vap  = np.array([_q("S","T",T,"Q",1) for T in T_arr])
    s_crit = _q("S","P",p_crit,"T",T_crit)
    s_dome = np.concatenate([s_liq, [s_crit], s_vap[::-1]])
    T_dome = np.concatenate([T_arr, [T_crit], T_arr[::-1]])
    return s_dome, T_dome


def _saturation_dome_ph(n=400):
    p_crit = _q("Pcrit", "T", 300, "Q", 1)
    T_trip = _q("Ttriple", "T", 300, "Q", 1)
    p_trip = _q("P","T",T_trip*1.001,"Q",0)
    p_arr  = np.geomspace(p_trip*1.01, p_crit*0.9999, n)
    h_liq  = np.array([_q("H","P",p,"Q",0) for p in p_arr])
    h_vap  = np.array([_q("H","P",p,"Q",1) for p in p_arr])
    h_crit = _q("H","P",p_crit,"Q",0.5)
    h_dome = np.concatenate([h_liq, [h_crit], h_vap[::-1]])
    p_dome = np.concatenate([p_arr, [p_crit], p_arr[::-1]])
    return h_dome, p_dome


# ─────────────────────────────────────────────────────────────────────────────
# Isoline computation
# ─────────────────────────────────────────────────────────────────────────────

def _quality_isolines_ts(T_lo, T_hi, n_lines=9, n_pts=200):
    T_crit = _q("Tcrit","T",300,"Q",1)
    T_trip = _q("Ttriple","T",300,"Q",1)
    T_arr  = np.linspace(max(T_lo, T_trip*1.001), min(T_hi, T_crit*0.9999), n_pts)
    lines  = []
    for Q in np.linspace(0, 1, n_lines):
        s_arr = np.array([_q("S","T",T,"Q",Q) for T in T_arr])
        v = np.isfinite(s_arr)
        lines.append((s_arr[v], T_arr[v], Q))
    return lines


def _quality_isolines_ph(p_lo, p_hi, n_lines=9, n_pts=200):
    p_crit = _q("Pcrit","T",300,"Q",1)
    T_trip = _q("Ttriple","T",300,"Q",1)
    p_trip = _q("P","T",T_trip*1.001,"Q",0)
    p_arr  = np.geomspace(max(p_lo*0.5, p_trip*1.01), min(p_hi, p_crit*0.9999), n_pts)
    lines  = []
    for Q in np.linspace(0, 1, n_lines):
        h_arr = np.array([_q("H","P",p,"Q",Q) for p in p_arr])
        v = np.isfinite(h_arr)
        lines.append((h_arr[v], p_arr[v], Q))
    return lines


def _isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, n_lines=12, n_pts=200):
    p_crit   = _q("Pcrit","T",300,"Q",1)
    p_lo_est = max(_q("P","T",T_lo*0.98,"Q",0)*0.3, 1e3)
    p_vals   = np.geomspace(p_lo_est, p_crit*3.0, n_lines)
    s_arr    = np.linspace(s_lo, s_hi, n_pts)
    lines    = []
    for p in p_vals:
        T_arr = np.array([_q("T","P",p,"S",s) for s in s_arr])
        v = np.isfinite(T_arr) & (T_arr >= T_lo*0.95) & (T_arr <= T_hi*1.05)
        if v.sum() > 3:
            lines.append((s_arr[v], T_arr[v], p))
    return lines


def _isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, n_lines=18, n_pts=200):
    h_samples = []
    for s in np.linspace(s_lo, s_hi, 6):
        for T in np.linspace(T_lo, T_hi, 6):
            h_samples.append(_q("H","T",T,"S",s))
    h_samples = [h for h in h_samples if np.isfinite(h)]
    if not h_samples:
        return []
    h_vals = np.linspace(min(h_samples)*0.98, max(h_samples)*1.02, n_lines)
    p_crit = _q("Pcrit","T",300,"Q",1)
    p_arr  = np.geomspace(1e3, p_crit*5, n_pts)
    lines  = []
    for h in h_vals:
        s_arr = np.array([_q("S","P",p,"H",h) for p in p_arr])
        T_arr = np.array([_q("T","P",p,"H",h) for p in p_arr])
        v = (np.isfinite(s_arr) & np.isfinite(T_arr) &
             (s_arr >= s_lo) & (s_arr <= s_hi) &
             (T_arr >= T_lo*0.95) & (T_arr <= T_hi*1.05))
        if v.sum() > 3:
            lines.append((s_arr[v], T_arr[v], h))
    return lines


def _isotherm_lines_ph(h_lo, h_hi, p_lo, p_hi, n_lines=18, n_pts=200):
    T_crit = _q("Tcrit","T",300,"Q",1)
    T_trip = _q("Ttriple","T",300,"Q",1)
    T_vals = np.linspace(T_trip*1.05, T_crit*1.5, n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5,1e3), p_hi*1.5, n_pts)
    lines  = []
    for T in T_vals:
        h_arr = np.array([_q("H","P",p,"T",T) for p in p_arr])
        v = (np.isfinite(h_arr) &
             (h_arr >= h_lo) & (h_arr <= h_hi) &
             (p_arr >= p_lo*0.9) & (p_arr <= p_hi*1.1))
        if v.sum() > 3:
            lines.append((h_arr[v], p_arr[v], T))
    return lines


def _isentrop_lines_ph(h_lo, h_hi, p_lo, p_hi, n_lines=12, n_pts=200):
    s_samples = []
    for h in np.linspace(h_lo, h_hi, 6):
        for p in np.geomspace(max(p_lo,1e3), p_hi, 6):
            s_samples.append(_q("S","P",p,"H",h))
    s_samples = [s for s in s_samples if np.isfinite(s)]
    if not s_samples:
        return []
    s_vals = np.linspace(min(s_samples), max(s_samples), n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5,1e3), p_hi*1.5, n_pts)
    lines  = []
    for s in s_vals:
        h_arr = np.array([_q("H","P",p,"S",s) for p in p_arr])
        v = (np.isfinite(h_arr) &
             (h_arr >= h_lo) & (h_arr <= h_hi) &
             (p_arr >= p_lo*0.9) & (p_arr <= p_hi*1.1))
        if v.sum() > 3:
            lines.append((h_arr[v], p_arr[v], s))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Label placement
# ─────────────────────────────────────────────────────────────────────────────

def _label_point(ax, x_data, y_data_for_angle, frac):
    """Return (x_val, y_val, angle_deg, in_bounds) for a label at frac along
    the curve.  y_data_for_angle may be log10(p) for PH to get correct slope.
    x_val / y_val are always in data coordinates (NOT log-transformed)."""
    x = np.array(x_data)
    ya = np.array(y_data_for_angle)   # used only for angle
    if len(x) < 4:
        return None
    idx = int(np.clip(frac * len(x), 1, len(x)-2))
    p0  = ax.transData.transform((x[idx-1], ya[idx-1]))
    p1  = ax.transData.transform((x[idx+1], ya[idx+1]))
    ang = np.degrees(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
    if ang >  90: ang -= 180
    if ang < -90: ang += 180
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    in_bounds = (xl <= x[idx] <= xh) and (yl <= y_data_for_angle[idx] <= yh)
    return x[idx], y_data_for_angle[idx], ang, in_bounds


def _put_label(ax, x_val, y_val, ang, label, color):
    ax.text(x_val, y_val, label,
            color=color, fontsize=7, rotation=ang, rotation_mode='anchor',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6, clip_on=True)


def _draw_isolines_labeled(ax, lines, color, frac, fmt_short, fmt_named,
                           y_for_angle_fn=None):
    """Draw all isolines and place labels.

    The 'named' label (fmt_named, e.g. 'x = 0.50') is placed on the isoline
    whose label point is inside the plot bounds, preferring the middle isoline
    and walking outward symmetrically until one is found.  All other isolines
    get the short label (fmt_short).

    y_for_angle_fn: optional callable applied to y-values before angle calc
                    (pass np.log10 for PH diagram where y-axis is log-scaled).
    """
    if not lines:
        return

    # Build label-point info for every isoline
    infos = []   # (x_val, y_val, ang, in_bounds) or None
    for x_arr, y_arr, val in lines:
        if y_for_angle_fn is not None:
            ya = y_for_angle_fn(np.array(y_arr))
        else:
            ya = np.array(y_arr)
        infos.append(_label_point(ax, x_arr, ya, frac))

    # Find the best isoline for the named label: start at mid, walk outward
    n      = len(lines)
    mid    = n // 2
    named_k = None
    for offset in range(n):
        for k in [mid - offset, mid + offset]:
            if 0 <= k < n and infos[k] is not None and infos[k][3]:
                named_k = k
                break
        if named_k is not None:
            break

    # Draw lines and labels
    for k, ((x_arr, y_arr, val), info) in enumerate(zip(lines, infos)):
        ax.plot(x_arr, y_arr, color=color, lw=0.6, zorder=2)
        if info is None:
            continue
        x_val, y_val, ang, in_bounds = info
        if not in_bounds:
            continue
        lbl = fmt_named(val) if k == named_k else fmt_short(val)
        _put_label(ax, x_val, y_val, ang, lbl, color)


# ─────────────────────────────────────────────────────────────────────────────
# Expansion lines + legend
# ─────────────────────────────────────────────────────────────────────────────

def _add_expansion_lines(ax, state, diagram_type):
    s     = state
    p_exp = np.linspace(s["p_ref_3"], s["p_ref_1"], 150)
    col   = "#02220E"
    alp   = 0.75

    if diagram_type == "PH":
        h_isenth = np.full_like(p_exp, s["h_ref_3"])
        line_isenth, = ax.plot(h_isenth, p_exp, color=col, ls="-", lw=1.5,
                               alpha=alp, zorder=4, label=r"$\mathrm{isenthalpic\ expansion}$")
        h_isen = np.array([_q("H","P",p,"S",s["s_ref_3"]) for p in p_exp])
        v = np.isfinite(h_isen)
        line_isen, = ax.plot(h_isen[v], p_exp[v], color=col, ls="--", lw=1.5,
                             alpha=alp, zorder=4, label=r"$\mathrm{isentropic\ expansion}$")
    else:
        T_exit = _q("T","P",s["p_ref_1"],"S",s["s_ref_3"])
        T_rng  = np.linspace(s["T_ref_3"], T_exit, 150)
        line_isen, = ax.plot(np.full_like(T_rng, s["s_ref_3"]), T_rng,
                             color=col, ls="--", lw=1.5, alpha=alp, zorder=4,
                             label=r"$\mathrm{isentropic\ expansion}$")
        s_is = np.array([_q("S","P",p,"H",s["h_ref_3"]) for p in p_exp])
        T_is = np.array([_q("T","P",p,"H",s["h_ref_3"]) for p in p_exp])
        v = np.isfinite(s_is)
        line_isenth, = ax.plot(s_is[v], T_is[v], color=col, ls="-", lw=1.5,
                               alpha=alp, zorder=4, label=r"$\mathrm{isenthalpic\ expansion}$")

    line_turb = Line2D([0],[0], color='green', lw=1.5, label=r"$\mathrm{turbine\ expansion}$")
    leg = ax.legend(handles=[line_turb, line_isen, line_isenth],
                    loc="lower right", bbox_to_anchor=(0.9875, 0.015),
                    fontsize=8, framealpha=0.85)
    fr = leg.get_frame()
    fr.set_facecolor((0.96, 0.92, 0.84, 0.72))
    fr.set_edgecolor('#9C7B53')
    fr.set_linewidth(1.2)
    fr.set_boxstyle('round,pad=0.5')
    leg.set_zorder(11)


# ─────────────────────────────────────────────────────────────────────────────
# Arrow / endpoint label helpers (TS)
# ─────────────────────────────────────────────────────────────────────────────

def _add_mid_arrow(ax, xv, yv, color, frac=0.18):
    if len(xv) < 2: return
    dx, dy = xv[1]-xv[0], yv[1]-yv[0]
    if np.isclose(dx,0) and np.isclose(dy,0): return
    ax.quiver(xv[0]+0.5*dx, yv[0]+0.5*dy, frac*dx, frac*dy,
              angles='xy', scale_units='xy', scale=1, pivot='middle',
              color=color, width=0.003, headwidth=4.5, headlength=6,
              headaxislength=5, zorder=5)


def _add_endpoint_labels(ax, xv, yv, lbl0, lbl1, color, side=1):
    if len(xv) < 2: return
    ps = np.array(ax.transData.transform((xv[0],yv[0])), dtype=float)
    pe = np.array(ax.transData.transform((xv[1],yv[1])), dtype=float)
    vec = pe - ps; norm = np.linalg.norm(vec)
    tan = vec/norm if norm > 1e-9 else np.array([1.,0.])
    nor = side * np.array([-tan[1], tan[0]])
    for lbl, pt, sgn in [(lbl0,(xv[0],yv[0]),-1),(lbl1,(xv[1],yv[1]),1)]:
        off = nor*6. + sgn*tan*15.
        ax.annotate(lbl, xy=pt, xytext=(off[0],off[1]),
                    textcoords='offset points', fontsize=8, color=color,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
                    zorder=6, clip_on=True)


# ─────────────────────────────────────────────────────────────────────────────
# Performance box
# ─────────────────────────────────────────────────────────────────────────────

def _draw_perf_box(ax, perf):
    txt = (
        rf"$\begin{{array}}{{lrl}}"
        rf"\multicolumn{{3}}{{c}}{{\mathrm{{Performance}}}} \\\hline "
        rf"\mathrm{{COP_{{turb}}}}  & \num{{{perf['COP_turb']:.2f}}}  & [-] \\"
        rf"\mathrm{{COP_{{is}}}}    & \num{{{perf['COP_is']:.2f}}}    & [-] \\"
        rf"\mathrm{{COP_{{isenth}}}}& \num{{{perf['COP_isenth']:.2f}}}& [-] \\"
        rf"\dot{{W}}_{{turb}}       & \num{{{perf['Ẇ_turb']:.0f}}}    & [\mathrm{{W}}] \\"
        rf"\dot{{W}}_{{compr}}      & \num{{{perf['Ẇ_comp']:.0f}}}    & [\mathrm{{W}}] \\"
        rf"\dot{{Q}}_{{in,turb}}    & \num{{{perf['Q_in']:.0f}}}      & [\mathrm{{W}}]"
        rf"\end{{array}}$"
    )
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', zorder=10,
            bbox=dict(facecolor=(0.96,0.92,0.84,0.72), edgecolor='#9C7B53',
                      linewidth=1.2, boxstyle='round,pad=0.5'))


# ─────────────────────────────────────────────────────────────────────────────
# TS renderer
# ─────────────────────────────────────────────────────────────────────────────

def _make_plot_ts(state, perf, ts_data):
    warnings.filterwarnings("ignore")
    n_pts = 150 if resolution == "low" else 600

    s_lo, s_hi, T_lo, T_hi = _cycle_bounds_ts(state)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(s_lo, s_hi)
    ax.set_ylim(T_lo, T_hi)
    cmap = plt.get_cmap('tab20')

    # Saturation dome
    s_dome, T_dome = _saturation_dome_ts(n=n_pts*2)
    ax.plot(s_dome, T_dome, color='black', lw=1.0, zorder=3)

    # Quality isolines
    q_col = cmap(0)
    _draw_isolines_labeled(
        ax, _quality_isolines_ts(T_lo, T_hi, n_pts=n_pts), q_col, 0.25,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named=lambda v: rf"$x={v:.2f}$")

    # Isobar lines
    p_col = cmap(2)
    _draw_isolines_labeled(
        ax, _isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, n_pts=n_pts), p_col, 0.92,
        fmt_short=lambda v: rf"${v/1e6:.2f}$",
        fmt_named=lambda v: rf"$p={v/1e6:.2f}\,\mathrm{{MPa}}$")

    # Isenthalpic lines
    h_col = cmap(4)
    _draw_isolines_labeled(
        ax, _isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, n_pts=n_pts), h_col, 0.93,
        fmt_short=lambda v: rf"${v/1e3:.0f}$",
        fmt_named=lambda v: rf"$h={v/1e3:.0f}\,\mathrm{{kJ/kg}}$")

    # Critical isobar (dotted)
    p_crit = _q("Pcrit","T",300,"Q",1)
    T_crit = _q("Tcrit","T",300,"Q",1)
    sc, Tc = isobar_segment(s_lo, s_hi, p_crit)
    ax.plot(sc, Tc, color='black', ls=':', lw=1.0, zorder=1)

    # Critical point
    s_crit = _q("S","P",p_crit,"T",T_crit)
    ax.plot(s_crit, T_crit, marker='o', markerfacecolor='yellow',
            markersize=5, markeredgecolor='black', zorder=9)

    # Cycle
    ax.scatter(ts_data["major"]["s"], ts_data["major"]["T"],
               color='orange', marker='o', s=5, zorder=8)
    ax.plot(ts_data["minor"]["s"], ts_data["minor"]["T"],
            color='green', lw=1.5, zorder=7)

    _add_expansion_lines(ax, state, "TS")

    for flow, col in [(ts_data["coolant"],"blue"),(ts_data["heating"],"red")]:
        ax.plot(flow["s"], flow["T"], color=col, marker='o', markersize=2, zorder=12)
        _add_mid_arrow(ax, flow["s"], flow["T"], col)
    _add_endpoint_labels(ax, ts_data["coolant"]["s"], ts_data["coolant"]["T"],
                         r"$T_{c,\mathrm{in}}$", r"$T_{c,\mathrm{out}}$", "blue", side=1)
    _add_endpoint_labels(ax, ts_data["heating"]["s"], ts_data["heating"]["T"],
                         r"$T_{h,\mathrm{in}}$", r"$T_{h,\mathrm{out}}$", "red", side=-1)

    ax.set_xlabel(r"$s\ [\mathrm{J/kg/K}]$")
    ax.set_ylabel(r"$T\ [\mathrm{K}]$")
    ax.set_title(rf"$\mathrm{{Conceptual\ Heat\ Pump\ Cycle}}\ -\ \mathrm{{{refrigerant}}}$")
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PH renderer
# ─────────────────────────────────────────────────────────────────────────────

def _make_plot_ph(state, perf, ph_data):
    warnings.filterwarnings("ignore")
    n_pts = 150 if resolution == "low" else 600

    h_lo, h_hi, p_lo, p_hi = _cycle_bounds_ph(state)
    p_lo = max(p_lo, 1e3)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(p_lo, p_hi)
    cmap = plt.get_cmap('tab20')

    # Saturation dome
    h_dome, p_dome = _saturation_dome_ph(n=n_pts*2)
    ax.plot(h_dome, p_dome, color='black', lw=1.0, zorder=3)

    # Quality isolines (label angle computed in log-p space)
    q_col = cmap(0)
    _draw_isolines_labeled(
        ax, _quality_isolines_ph(p_lo, p_hi, n_pts=n_pts), q_col, 0.5,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named=lambda v: rf"$x={v:.2f}$",
        y_for_angle_fn=np.log10)

    # Isotherm lines
    T_col = cmap(4)
    _draw_isolines_labeled(
        ax, _isotherm_lines_ph(h_lo, h_hi, p_lo, p_hi, n_pts=n_pts), T_col, 0.92,
        fmt_short=lambda v: rf"${v:.0f}$",
        fmt_named=lambda v: rf"$T={v:.0f}\,\mathrm{{K}}$",
        y_for_angle_fn=np.log10)

    # Isentropic lines
    s_col = cmap(2)
    _draw_isolines_labeled(
        ax, _isentrop_lines_ph(h_lo, h_hi, p_lo, p_hi, n_pts=n_pts), s_col, 0.92,
        fmt_short=lambda v: rf"${v/1e3:.2f}$",
        fmt_named=lambda v: rf"$s={v/1e3:.2f}\,\mathrm{{kJ/kgK}}$",
        y_for_angle_fn=np.log10)

    # Critical point + critical isotherm (dotted)
    p_crit = _q("Pcrit","T",300,"Q",1)
    T_crit = _q("Tcrit","T",300,"Q",1)
    h_crit = _q("H","P",p_crit,"T",T_crit)
    ax.plot(h_crit, p_crit, marker='o', markerfacecolor='yellow',
            markersize=5, markeredgecolor='black', zorder=9)
    p_arr_iso = np.geomspace(max(p_lo*0.5,1e3), p_hi*1.1, 300)
    h_Tc = np.array([_q("H","P",p,"T",T_crit) for p in p_arr_iso])
    v    = np.isfinite(h_Tc) & (h_Tc >= h_lo) & (h_Tc <= h_hi)
    if v.any():
        ax.plot(h_Tc[v], p_arr_iso[v], color='black', ls=':', lw=1.0, zorder=1)

    # Cycle
    ax.scatter(ph_data["major"]["h"], ph_data["major"]["p"],
               color='orange', marker='o', s=5, zorder=8)
    ax.plot(ph_data["minor"]["h"], ph_data["minor"]["p"],
            color='green', lw=1.5, zorder=7)

    _add_expansion_lines(ax, state, "PH")

    ax.set_xlabel(r"$h\ [\mathrm{kJ/kg}]$")
    ax.set_ylabel(r"$p\ [\mathrm{Pa}]$")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}$"))
    ax.set_title(rf"$\mathrm{{Conceptual\ Heat\ Pump\ Cycle}}\ -\ \mathrm{{{refrigerant}}}$")
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(state, perf, diagram_type, ts_data=None, ph_data=None):
    configure_matplotlib()
    fig = _make_plot_ts(state, perf, ts_data) if diagram_type == "TS" \
          else _make_plot_ph(state, perf, ph_data)
    fname = f"Conceptual HP Cycle - {refrigerant} - {diagram_type}.pdf"
    fig.savefig(fname, dpi=1000, bbox_inches="tight")
    print(f"Saved: {fname}")
    return fig