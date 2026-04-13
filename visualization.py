import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import patheffects
from CoolProp.CoolProp import PropsSI
from logger import setup_logger
from pathlib import Path

from config import general_config
from thermodynamics import _isobar_segment   # TS critical-isobar

logger = setup_logger()



# Utility
# =======
def _configure_matplotlib():
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


def _cop_type_to_latex(cop_key):
    """Convert keys like 'COP_turb' into a LaTeX label body."""
    parts = cop_key.split("_", 1)
    if len(parts) == 2:
        return rf"\mathrm{{{parts[0]}}}_{{\mathrm{{{parts[1]}}}}}"
    return rf"\mathrm{{{cop_key}}}"

def _q(out, *args, cycle_config):
    """PropsSI wrapper — returns np.nan on any failure."""
    refrigerant = cycle_config["refrigerant"]
    try:
        return PropsSI(out, *args, f"REFPROP::{refrigerant}")
    except Exception:
        return np.nan


def _verification_path_with_isenthalpic_expansion(verification_data, cycle_config, n_pts=80):
    """Return verification-cycle arrays with segment 5->6 rebuilt as isenthalpic."""
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



# Axis-bound calculation
# ======================
def _cycle_bounds_ts(cycle_data, cycle_config):
    ts_margin_s_left = general_config.get("ts_margin_s_left", 0.2)
    ts_margin_s_right = general_config.get("ts_margin_s_right", 0.2)
    ts_margin_T_bot = general_config.get("ts_margin_T_bot", 0.2)
    ts_margin_T_top = general_config.get("ts_margin_T_top", 0.15)

    s_vals = [cycle_data[k] for k in ("s_ref_1","s_ref_2","s_ref_3","s_ref_4")]
    T_vals = [cycle_data[k] for k in ("T_ref_1","T_ref_2","T_ref_3","T_ref_4")]

    # Always include the critical point
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    if np.isfinite(s_crit) and np.isfinite(T_crit):
        s_vals.append(s_crit)
        T_vals.append(T_crit)

    s_lo, s_hi = min(s_vals), max(s_vals)
    T_lo, T_hi = min(T_vals), max(T_vals)
    ds = s_hi - s_lo
    dT = T_hi - T_lo
    # left: 1.0×ds  (room for perf box)   right: 0.5×ds
    # bottom: 0.4×dT                       top:   0.6×dT
    return (s_lo - ds*ts_margin_s_left,  s_hi + ds*ts_margin_s_right,
            T_lo - dT*ts_margin_T_bot,   T_hi + dT*ts_margin_T_top)


def _cycle_bounds_ph(cycle_data, cycle_config):
    ph_margin_h_left = general_config.get("ph_margin_h_left", 0.3)
    ph_margin_h_right = general_config.get("ph_margin_h_right", 0.15)
    ph_margin_p_bot = general_config.get("ph_margin_p_bot", 0.1)
    ph_margin_p_top = general_config.get("ph_margin_p_top", 0.2)

    h_vals = [cycle_data[k] for k in ("h_ref_1","h_ref_2","h_ref_3","h_ref_4")]
    p_vals = [cycle_data[k] for k in ("p_ref_1","p_ref_2","p_ref_3","p_ref_4")]

    # Always include the critical point
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
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
    # bottom: 0.3 decades below p_lo   top: 0.5 decades above p_hi
    p_lo_plot = max(10 ** (log_p_lo - ph_margin_p_bot * max(log_span, 0.3)), 1e2)
    p_hi_plot = 10 ** (log_p_hi + ph_margin_p_top * max(log_span, 0.3))
    return (h_lo - dh*ph_margin_h_left, h_hi + dh*ph_margin_h_right,
            p_lo_plot, p_hi_plot)



# Saturation dome
# ===============
def _saturation_dome_ts(cycle_config, n=400):
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_arr  = np.linspace(T_trip*1.001, T_crit*0.9999, n)
    s_liq  = np.array([_q("S", "T", T, "Q", 0, cycle_config=cycle_config) for T in T_arr])
    s_vap  = np.array([_q("S", "T", T, "Q", 1, cycle_config=cycle_config) for T in T_arr])
    s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    return (np.concatenate([s_liq, [s_crit], s_vap[::-1]]),
            np.concatenate([T_arr, [T_crit], T_arr[::-1]]))


def _saturation_dome_ph(cycle_config, n=400):
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    p_trip = _q("P", "T", T_trip*1.001, "Q", 0, cycle_config=cycle_config)
    p_arr  = np.geomspace(p_trip*1.01, p_crit*0.9999, n)
    h_liq  = np.array([_q("H", "P", p, "Q", 0, cycle_config=cycle_config) for p in p_arr])
    h_vap  = np.array([_q("H", "P", p, "Q", 1, cycle_config=cycle_config) for p in p_arr])
    h_crit = _q("H", "P", p_crit, "Q", 0.5, cycle_config=cycle_config)
    return (np.concatenate([h_liq, [h_crit], h_vap[::-1]]),
            np.concatenate([p_arr, [p_crit], p_arr[::-1]]))



# Isoline computation
# ===================
def _quality_isolines_ts(T_lo, T_hi, cycle_config, n_lines=9, n_pts=200):
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_arr  = np.linspace(max(T_lo, T_trip*1.001), min(T_hi, T_crit*0.9999), n_pts)
    lines  = []
    for Q in np.linspace(0, 1, n_lines):
        s_arr = np.array([_q("S", "T", T, "Q", Q, cycle_config=cycle_config) for T in T_arr])
        v = np.isfinite(s_arr)
        if v.sum() > 3:
            lines.append((s_arr[v], T_arr[v], Q))
    return lines


def _quality_isolines_ph(p_lo, p_hi, cycle_config, n_lines=9, n_pts=200):
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    p_trip = _q("P", "T", T_trip*1.001, "Q", 0, cycle_config=cycle_config)
    p_arr  = np.geomspace(max(p_lo*0.5, p_trip*1.01), min(p_hi, p_crit*0.9999), n_pts)
    lines  = []
    for Q in np.linspace(0, 1, n_lines):
        h_arr = np.array([_q("H", "P", p, "Q", Q, cycle_config=cycle_config) for p in p_arr])
        v = np.isfinite(h_arr)
        if v.sum() > 3:
            lines.append((h_arr[v], p_arr[v], Q))
    return lines


def _isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_lines=12, n_pts=200):
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    p_trip = _q("P", "T", T_trip*1.001, "Q", 0, cycle_config=cycle_config)
    p_lo_est = max(p_trip*1.1 if np.isfinite(p_trip) else 1e3, 1e3)
    p_vals = np.geomspace(p_lo_est, p_crit*3.0, n_lines)

    T_sweep = np.linspace(max(T_lo*0.9, T_trip*1.001),
                          min(T_hi*1.1, T_crit*2.0), n_pts)
    lines = []
    for p in p_vals:
        if p < p_crit:
            T_sat = _q("T", "P", p, "Q", 0, cycle_config=cycle_config)
            s_sat_l = _q("S", "P", p, "Q", 0, cycle_config=cycle_config)
            s_sat_v = _q("S", "P", p, "Q", 1, cycle_config=cycle_config)
            if not (np.isfinite(T_sat) and np.isfinite(s_sat_l) and np.isfinite(s_sat_v)):
                continue
            # Subcooled branch (T < T_sat)
            T_liq = T_sweep[T_sweep < T_sat]
            s_liq = np.array([_q("S", "P", p, "T", T, cycle_config=cycle_config) for T in T_liq])
            # Superheated branch (T > T_sat)
            T_vap = T_sweep[T_sweep > T_sat]
            s_vap = np.array([_q("S", "P", p, "T", T, cycle_config=cycle_config) for T in T_vap])
            # Two-phase horizontal bridge at T_sat
            s_2ph = np.linspace(s_sat_l, s_sat_v, 20)
            T_2ph = np.full(20, T_sat)
            # Stitch in physical order to avoid artificial loops/crossings.
            s_all = np.concatenate([s_liq, s_2ph, s_vap])
            T_all = np.concatenate([T_liq, T_2ph, T_vap])
        else:
            s_all = np.array([_q("S", "P", p, "T", T, cycle_config=cycle_config) for T in T_sweep])
            T_all = T_sweep

        v = (np.isfinite(s_all) & (s_all >= s_lo) & (s_all <= s_hi) &
             (T_all >= T_lo*0.95) & (T_all <= T_hi*1.05))
        if v.sum() > 3:
            lines.append((s_all[v], T_all[v], p))
    return lines


def _isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_lines=18, n_pts=200):
    # Sample h range from corners of the visible TS window
    h_samples = []
    for s in np.linspace(s_lo, s_hi, 6):
        for T in np.linspace(T_lo, T_hi, 6):
            h_samples.append(_q("H", "T", T, "S", s, cycle_config=cycle_config))
    h_samples = [h for h in h_samples if np.isfinite(h)]
    if not h_samples:
        return []
    h_vals  = np.linspace(min(h_samples)*0.95, max(h_samples)*1.05, n_lines)
    p_crit  = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    p_arr   = np.geomspace(1e3, p_crit*5, n_pts)
    lines   = []
    for h in h_vals:
        s_arr = np.array([_q("S", "P", p, "H", h, cycle_config=cycle_config) for p in p_arr])
        T_arr = np.array([_q("T", "P", p, "H", h, cycle_config=cycle_config) for p in p_arr])
        v = (np.isfinite(s_arr) & np.isfinite(T_arr) &
             (s_arr >= s_lo) & (s_arr <= s_hi) &
             (T_arr >= T_lo*0.95) & (T_arr <= T_hi*1.05))
        if v.sum() > 3:
            lines.append((s_arr[v], T_arr[v], h))
    return lines


def _isotherm_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_lines=18, n_pts=200):
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_trip = _q("Ttriple", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_vals = np.linspace(T_trip*1.05, T_crit*1.5, n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, n_pts)
    lines  = []
    for T in T_vals:
        if T < T_crit:
            p_sat = _q("P", "T", T, "Q", 0, cycle_config=cycle_config)
            h_sat_l = _q("H", "T", T, "Q", 0, cycle_config=cycle_config)
            h_sat_v = _q("H", "T", T, "Q", 1, cycle_config=cycle_config)
            if not (np.isfinite(p_sat) and np.isfinite(h_sat_l) and np.isfinite(h_sat_v)):
                continue
            # Liquid branch (p > p_sat)
            p_liq = p_arr[p_arr > p_sat]
            h_liq = np.array([_q("H", "P", p, "T", T, cycle_config=cycle_config) for p in p_liq])
            # Vapour branch (p < p_sat)
            p_vap = p_arr[p_arr < p_sat]
            h_vap = np.array([_q("H", "P", p, "T", T, cycle_config=cycle_config) for p in p_vap])
            # Two-phase horizontal bridge at p_sat
            h_2ph = np.linspace(h_sat_l, h_sat_v, 20)
            p_2ph = np.full(20, p_sat)
            # Stitch: liquid (high→p_sat) → bridge → vapour (p_sat→low)
            h_all = np.concatenate([h_liq[::-1], h_2ph, h_vap[::-1]])
            p_all = np.concatenate([p_liq[::-1], p_2ph, p_vap[::-1]])
        else:
            h_all = np.array([_q("H", "P", p, "T", T, cycle_config=cycle_config) for p in p_arr])
            p_all = p_arr

        v = (np.isfinite(h_all) &
             (h_all >= h_lo) & (h_all <= h_hi) &
             (p_all >= p_lo*0.9) & (p_all <= p_hi*1.1))
        if v.sum() > 3:
            lines.append((h_all[v], p_all[v], T))
    return lines


def _isentrop_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_lines=12, n_pts=200):
    s_samples = []
    for h in np.linspace(h_lo, h_hi, 6):
        for p in np.geomspace(max(p_lo, 1e3), p_hi, 6):
            s_samples.append(_q("S", "P", p, "H", h, cycle_config=cycle_config))
    s_samples = [s for s in s_samples if np.isfinite(s)]
    if not s_samples:
        return []
    s_vals = np.linspace(min(s_samples), max(s_samples), n_lines)
    p_arr  = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.5, n_pts)
    lines  = []
    for sv in s_vals:
        h_arr = np.array([_q("H", "P", p, "S", sv, cycle_config=cycle_config) for p in p_arr])
        v = (np.isfinite(h_arr) &
             (h_arr >= h_lo) & (h_arr <= h_hi) &
             (p_arr >= p_lo*0.9) & (p_arr <= p_hi*1.1))
        if v.sum() > 3:
            lines.append((h_arr[v], p_arr[v], sv))
    return lines



# Label placement
# ===============
def _label_angle(x_data, y_data, idx, xl, xh, yl, yh, yscale, ax):
    """Compute rotation angle in degrees for a label at index idx.

    Angle is computed in normalised axes space so that it is correct on screen
    regardless of axis scale (linear or log).  Does NOT require a rendered
    canvas.
    """
    x = np.array(x_data)
    y = np.array(y_data)
    # normalise to [0,1] axes coordinates
    x_norm = (x - xl) / (xh - xl)
    if yscale == 'log':
        log_yl, log_yh = np.log10(yl), np.log10(yh)
        y_norm = (np.log10(np.abs(y) + 1e-300) - log_yl) / (log_yh - log_yl)
    else:
        y_norm = (y - yl) / (yh - yl)

    # Physical aspect of the axes in inches (approximate from figure size)
    figw, figh = ax.get_figure().get_size_inches()
    # axes typically occupies ~78% width, ~78% height of figure
    ax_w_in = figw * 0.78
    ax_h_in = figh * 0.78

    dx = (x_norm[idx+1] - x_norm[idx-1]) * ax_w_in
    dy = (y_norm[idx+1] - y_norm[idx-1]) * ax_h_in

    ang = np.degrees(np.arctan2(dy, dx))
    # Normalise to [-90, 90] so text is always right-reading
    if dx < 0:          # line runs right-to-left → flip 180°
        ang += 180
    ang = ((ang + 90) % 180) - 90
    return ang


def _label_point(ax, x_data, y_data, frac, yscale='linear'):
    """Return (x_val, y_val, angle_deg, in_bounds) or None."""
    x = np.array(x_data)
    y = np.array(y_data)
    if len(x) < 4:
        return None
    idx = int(np.clip(frac * len(x), 1, len(x)-2))
    xl, xh = ax.get_xlim()
    yl, yh = ax.get_ylim()
    in_bounds = (xl <= x[idx] <= xh) and (yl <= y[idx] <= yh)
    ang = _label_angle(x, y, idx, xl, xh, yl, yh, yscale, ax)
    return x[idx], y[idx], ang, in_bounds


def _put_label(ax, x_val, y_val, ang, label, color):
    ax.text(x_val, y_val, label,
            color=color, fontsize=10, rotation=ang, rotation_mode='anchor',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.2),
            zorder=6, clip_on=True)


def _draw_isolines_labeled(ax, lines, color, frac, fmt_short, fmt_named,
                           yscale='linear', flip_q1=False):
    """Draw lines and place labels.

    The 'named' label (fmt_named) goes on the isoline whose label point falls
    inside the plot bounds, preferring the middle line and walking outward.
    All others get fmt_short.  Works before canvas rendering.
    """
    if not lines:
        return

    infos = [_label_point(ax, x, y, frac, yscale) for x, y, _ in lines]

    # Find best candidate for named label (inside bounds, closest to middle)
    n, mid, named_k = len(lines), len(lines) // 2, None
    for offset in range(n):
        for k in [mid - offset, mid + offset]:
            if 0 <= k < n and infos[k] is not None and infos[k][3]:
                named_k = k
                break
        if named_k is not None:
            break

    for k, ((x_arr, y_arr, val), info) in enumerate(zip(lines, infos)):
        ax.plot(x_arr, y_arr, color=color, lw=0.6, zorder=2)
        if info is None:
            continue
        x_v, y_v, ang, in_bounds = info
        if not in_bounds:
            continue
        if abs(val - 1.0) < 1e-9 and flip_q1:
            ang += 180
        lbl = fmt_named(val) if k == named_k else fmt_short(val)
        _put_label(ax, x_v, y_v, ang, lbl, color)



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
    """Add station number labels (1, 2, 3, 4) to the four cycle nodes.
    
    Station numbering (heat pump cycle):
    - Station 1: Evaporator outlet (before compressor)
    - Station 2: Compressor outlet
    - Station 3: Condenser outlet (before expansion valve)
    - Station 4: Expansion valve outlet (back to evaporator)
    
    Labels are positioned at bottom-right of each node with bold black text
    and a semi-transparent white background for clarity.
    """
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
    """Compute a curved expansion path using h(p)=h3-eta*(h3-h_is(p))."""
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


def _find_expansion_segment_index(x_arr, y_arr, x3, y3, x4, y4):
    """Find index i where segment i->i+1 corresponds to 3->4 in minor arrays."""
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
    """Plot cycle minor path but replace straight 3->4 with curved expansion."""
    x = np.asarray(ts_data["minor"]["s"])
    y = np.asarray(ts_data["minor"]["T"])
    idx = _find_expansion_segment_index(
        x,
        y,
        cycle_data["s_ref_3"],
        cycle_data["T_ref_3"],
        cycle_data["s_ref_4"],
        cycle_data["T_ref_4"],
    )
    curve = _compute_turbine_expansion_curve(cycle_data, cycle_config)

    if idx is None or curve is None:
        ax.plot(x, y, color='green', lw=1.5, zorder=7)
        return

    p_arr, h_arr, s_curve, T_curve = curve
    ax.plot(x[:idx + 1], y[:idx + 1], color='green', lw=1.5, zorder=7)
    ax.plot(s_curve, T_curve, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx + 1:], y[idx + 1:], color='green', lw=1.5, zorder=7)


def _plot_cycle_with_curved_expansion_ph(ax, cycle_data, ph_data, cycle_config):
    """Plot cycle minor path but replace straight 3->4 with curved expansion."""
    x = np.asarray(ph_data["minor"]["h"])
    y = np.asarray(ph_data["minor"]["p"])
    idx = _find_expansion_segment_index(
        x,
        y,
        cycle_data["h_ref_3"],
        cycle_data["p_ref_3"],
        cycle_data["h_ref_4"],
        cycle_data["p_ref_4"],
    )
    curve = _compute_turbine_expansion_curve(cycle_data, cycle_config)

    if idx is None or curve is None:
        ax.plot(x, y, color='green', lw=1.5, zorder=7)
        return

    p_curve, h_curve, s_arr, T_arr = curve
    ax.plot(x[:idx + 1], y[:idx + 1], color='green', lw=1.5, zorder=7)
    ax.plot(h_curve, p_curve, color='green', lw=1.5, zorder=7)
    ax.plot(x[idx + 1:], y[idx + 1:], color='green', lw=1.5, zorder=7)



# Colour palette
# ==============
# TS:  quality → dark blue  | isobars → light blue  | isenthalps → orange
# PH:  quality → dark blue  | isentropes → light blue | isotherms → orange
COL_QUALITY    = '#1a3a6b'   # dark blue
COL_ISOBAR_ISO = '#6ab0de'   # light blue  (isobars on TS, isentropes on PH)
COL_ISENTH_ISO = '#e07b20'   # orange      (isenthalps on TS, isotherms on PH)
COL_VERIFICATION = '#0d1f3c' # very dark slate blue (for verification/reference cycles)



# TS renderer
# ===========
def _make_plot_ts(cycle_data, perf, ts_data, cycle_config, verification_data=None):
    warnings.filterwarnings("ignore")
    refrigerant = cycle_config["refrigerant"]
    resolution = general_config.get("resolution", "low")
    n_pts = 150 if resolution == "low" else 600

    s_lo, s_hi, T_lo, T_hi = _cycle_bounds_ts(cycle_data, cycle_config)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(s_lo, s_hi)
    ax.set_ylim(T_lo, T_hi)

    # Saturation dome
    s_dome, T_dome = _saturation_dome_ts(cycle_config, n=n_pts*2)
    ax.plot(s_dome, T_dome, color='black', lw=1.0, zorder=3)

    # Quality isolines — dark blue
    _draw_isolines_labeled(
        ax, _quality_isolines_ts(T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_QUALITY, 0.3,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named =lambda v: rf"$x={v:.2f}$",
        flip_q1=(refrigerant == "R1234ze(Z)"))

    # Isobar lines — light blue
    _draw_isolines_labeled(
        ax, _isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_ISOBAR_ISO, 0.85,
        fmt_short=lambda v: rf"${v/1e3:.0f}$",
        fmt_named =lambda v: rf"$p={v/1e3:.0f}\,\mathrm{{kPa}}$")

    # Isenthalpic lines — orange
    _draw_isolines_labeled(
        ax, _isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_ISENTH_ISO, 0.90,
        fmt_short=lambda v: rf"${v/1e3:.0f}$",
        fmt_named =lambda v: rf"$h={v/1e3:.0f}\,\mathrm{{kJ/kg}}$")

    # Critical isobar (dotted) — filter zeros / invalid T values
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    sc, Tc = _isobar_segment(s_lo, s_hi, p_crit, cycle_config, general_config)
    sc = np.array(sc); Tc = np.array(Tc)
    valid = (Tc > T_lo) & (Tc <= T_hi * 1.05) & np.isfinite(Tc) & (Tc > 1.0)
    if valid.any():
        ax.plot(sc[valid], Tc[valid], color='black', ls=':', lw=1.0, zorder=1)

    # Critical point
    s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    ax.plot(s_crit, T_crit, marker='o', markerfacecolor='yellow',
            markersize=5, markeredgecolor='black', zorder=9)

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

    for flow, col in [(ts_data["coolant"],"red"),(ts_data["heating"],"blue")]:
        ax.plot(flow["s"], flow["T"], color=col, marker='o', markersize=2, zorder=12)
        _add_mid_arrow(ax, flow["s"], flow["T"], col)
    _add_endpoint_labels(ax, ts_data["coolant"]["s"], ts_data["coolant"]["T"],
                         r"$T_{h,\mathrm{in}}$", r"$T_{h,\mathrm{out}}$", "red", side=1)
    _add_endpoint_labels(ax, ts_data["heating"]["s"], ts_data["heating"]["T"],
                         r"$T_{c,\mathrm{in}}$", r"$T_{c,\mathrm{out}}$", "blue",  side=-1)

    # Add cycle node labels (station numbers 1-4)
    _add_node_labels(ax, cycle_data, "TS")

    ax.set_xlabel(r"$s\ [\mathrm{J/kg/K}]$")
    ax.set_ylabel(r"$T\ [\mathrm{K}]$")
    ax.tick_params(axis='both', which='major', labelsize=12)
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig



# PH renderer
# ===========
def _make_plot_ph(cycle_data, perf, ph_data, cycle_config, verification_data=None):
    warnings.filterwarnings("ignore")
    refrigerant = cycle_config["refrigerant"]
    resolution = general_config.get("resolution", "low")
    n_pts = 150 if resolution == "low" else 600

    h_lo, h_hi, p_lo, p_hi = _cycle_bounds_ph(cycle_data, cycle_config)
    p_lo = max(p_lo, 1e2)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(p_lo, p_hi)

    # Saturation dome
    h_dome, p_dome = _saturation_dome_ph(cycle_config, n=n_pts*2)
    ax.plot(h_dome, p_dome, color='black', lw=1.0, zorder=3)

    # Quality isolines — dark blue
    _draw_isolines_labeled(
        ax, _quality_isolines_ph(p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_QUALITY, 0.5,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named =lambda v: rf"$x={v:.2f}$",
        yscale='log')

    # Isotherm lines — orange
    _draw_isolines_labeled(
        ax, _isotherm_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_ISENTH_ISO, 0.88,
        fmt_short=lambda v: rf"${v:.0f}$",
        fmt_named =lambda v: rf"$T={v:.0f}\,\mathrm{{K}}$",
        yscale='log')

    # Isentropic lines — light blue
    _draw_isolines_labeled(
        ax, _isentrop_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_ISOBAR_ISO, 0.88,
        fmt_short=lambda v: rf"${v/1e3:.2f}$",
        fmt_named =lambda v: rf"$s={v/1e3:.2f}\,\mathrm{{kJ/kgK}}$",
        yscale='log')

    # Critical point + critical isotherm (dotted)
    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    h_crit = _q("H", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    ax.plot(h_crit, p_crit, marker='o', markerfacecolor='yellow',
            markersize=5, markeredgecolor='black', zorder=9)
    p_iso = np.geomspace(max(p_lo*0.5, 1e3), p_hi*1.1, 300)
    h_Tc  = np.array([_q("H", "P", p, "T", T_crit, cycle_config=cycle_config) for p in p_iso])
    v     = np.isfinite(h_Tc) & (h_Tc >= h_lo) & (h_Tc <= h_hi)
    if v.any():
        ax.plot(h_Tc[v], p_iso[v], color='black', ls=':', lw=1.0, zorder=1)

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

    ax.set_xlabel(r"$h\ [\mathrm{kJ/kg}]$")
    ax.set_ylabel(r"$p\ [\mathrm{Pa}]$")
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: rf"$\bm{{{x/1000:.0f}}}$"))
    _draw_perf_box(ax, perf)
    fig.tight_layout()
    return fig


def _make_empty_plot_ts(cycle_config):
    warnings.filterwarnings("ignore")
    refrigerant = cycle_config["refrigerant"]
    resolution = general_config.get("resolution", "low")
    n_pts = 150 if resolution == "low" else 600

    s_dome, T_dome = _saturation_dome_ts(cycle_config, n=n_pts*2)
    valid_dome = np.isfinite(s_dome) & np.isfinite(T_dome)
    s_dome = s_dome[valid_dome]
    T_dome = T_dome[valid_dome]
    if len(s_dome) < 5 or len(T_dome) < 5:
        raise ValueError(f"Unable to build TS saturation dome for {refrigerant}.")

    s_lo_raw = np.min(s_dome)
    s_hi_raw = np.max(s_dome)
    T_lo_raw = np.min(T_dome)
    T_hi_raw = np.max(T_dome)
    ds = max(s_hi_raw - s_lo_raw, 1e-6)
    dT = max(T_hi_raw - T_lo_raw, 1e-6)

    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)

    # Expand the computational and visible domain beyond the saturation dome.
    s_lo = s_lo_raw - 0.12 * ds
    s_hi = s_hi_raw + 0.18 * ds
    T_lo = T_lo_raw
    T_hi = max(T_hi_raw + 0.11 * dT, T_crit * 1.175 if np.isfinite(T_crit) else T_hi_raw + 0.11 * dT)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(s_lo, s_hi)
    ax.set_ylim(T_lo, T_hi)
    ax.plot(s_dome, T_dome, color='black', lw=1.0, zorder=3)

    _draw_isolines_labeled(
        ax, _quality_isolines_ts(T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_QUALITY, 0.3,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named =lambda v: rf"$x={v:.2f}$",
        flip_q1=(refrigerant == "R1234ze(Z)"))

    _draw_isolines_labeled(
        ax, _isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_ISOBAR_ISO, 0.85,
        fmt_short=lambda v: rf"${v/1e3:.0f}$",
        fmt_named =lambda v: rf"$p={v/1e3:.0f}\,\mathrm{{kPa}}$")

    _draw_isolines_labeled(
        ax, _isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
        COL_ISENTH_ISO, 0.90,
        fmt_short=lambda v: rf"${v/1e3:.0f}$",
        fmt_named =lambda v: rf"$h={v/1e3:.0f}\,\mathrm{{kJ/kg}}$")

    sc, Tc = _isobar_segment(s_lo, s_hi, p_crit, cycle_config, general_config)
    sc = np.array(sc); Tc = np.array(Tc)
    valid = (Tc > T_lo) & (Tc <= T_hi * 1.05) & np.isfinite(Tc) & (Tc > 1.0)
    if valid.any():
        ax.plot(sc[valid], Tc[valid], color='black', ls=':', lw=1.0, zorder=1)

    s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    if np.isfinite(s_crit) and np.isfinite(T_crit):
        ax.plot(s_crit, T_crit, marker='o', markerfacecolor='yellow',
                markersize=5, markeredgecolor='black', zorder=9)

    ax.set_xlabel(r"$s\ [\mathrm{J/kg/K}]$")
    ax.set_ylabel(r"$T\ [\mathrm{K}]$")
    fig.tight_layout()
    return fig


def _make_empty_plot_ph(cycle_config):
    warnings.filterwarnings("ignore")
    refrigerant = cycle_config["refrigerant"]
    resolution = general_config.get("resolution", "low")
    n_pts = 150 if resolution == "low" else 600

    h_dome, p_dome = _saturation_dome_ph(cycle_config, n=n_pts*2)
    valid_dome = np.isfinite(h_dome) & np.isfinite(p_dome)
    h_dome = h_dome[valid_dome]
    p_dome = p_dome[valid_dome]
    if len(h_dome) < 5 or len(p_dome) < 5:
        raise ValueError(f"Unable to build PH saturation dome for {refrigerant}.")

    h_lo_raw = np.min(h_dome)
    h_hi_raw = np.max(h_dome)
    p_lo_raw = max(np.min(p_dome), 1e2)
    p_hi_raw = np.max(p_dome)
    dh = max(h_hi_raw - h_lo_raw, 1e-6)

    p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)

    # Expand the computational and visible domain beyond the saturation dome.
    h_lo = h_lo_raw - 0.12 * dh
    h_hi = h_hi_raw + 0.18 * dh
    p_lo = p_lo_raw
    p_hi = max(p_hi_raw * 1.6, p_crit * 1.4 if np.isfinite(p_crit) else p_hi_raw * 1.6)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_yscale('log')
    ax.set_xlim(h_lo, h_hi)
    ax.set_ylim(p_lo, p_hi)
    ax.plot(h_dome, p_dome, color='black', lw=1.0, zorder=3)

    _draw_isolines_labeled(
        ax, _quality_isolines_ph(p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_QUALITY, 0.5,
        fmt_short=lambda v: rf"${v:.2f}$",
        fmt_named =lambda v: rf"$x={v:.2f}$",
        yscale='log')

    _draw_isolines_labeled(
        ax, _isotherm_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_ISENTH_ISO, 0.88,
        fmt_short=lambda v: rf"${v:.0f}$",
        fmt_named =lambda v: rf"$T={v:.0f}\,\mathrm{{K}}$",
        yscale='log')

    _draw_isolines_labeled(
        ax, _isentrop_lines_ph(h_lo, h_hi, p_lo, p_hi, cycle_config, n_pts=n_pts),
        COL_ISOBAR_ISO, 0.88,
        fmt_short=lambda v: rf"${v/1e3:.2f}$",
        fmt_named =lambda v: rf"$s={v/1e3:.2f}\,\mathrm{{kJ/kgK}}$",
        yscale='log')

    T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
    h_crit = _q("H", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
    if np.isfinite(h_crit) and np.isfinite(p_crit):
        ax.plot(h_crit, p_crit, marker='o', markerfacecolor='yellow',
                markersize=5, markeredgecolor='black', zorder=9)

    p_iso = np.geomspace(max(p_lo * 0.5, 1e3), p_hi * 1.1, 300)
    h_Tc = np.array([_q("H", "P", p, "T", T_crit, cycle_config=cycle_config) for p in p_iso])
    v = np.isfinite(h_Tc) & (h_Tc >= h_lo) & (h_Tc <= h_hi)
    if v.any():
        ax.plot(h_Tc[v], p_iso[v], color='black', ls=':', lw=1.0, zorder=1)

    ax.set_xlabel(r"$h\ [\mathrm{kJ/kg}]$")
    ax.set_ylabel(r"$p\ [\mathrm{Pa}]$")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}$"))
    fig.tight_layout()
    return fig



# Public entry point for TS or PH plotting
# ========================================
def make_thdy_plot(
    cycle_data,
    perf,
    diagram_type,
    cycle_config,
    output_dir,
    ts_data=None,
    ph_data=None,
    verification_data=None,
    verbose=True,
):
    refrigerant = cycle_config["refrigerant"]
    _configure_matplotlib()
    fig = _make_plot_ts(cycle_data, perf, ts_data, cycle_config, verification_data=verification_data) if diagram_type == "TS" \
          else _make_plot_ph(cycle_data, perf, ph_data, cycle_config, verification_data=verification_data)
    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"Conceptual HP Cycle - {refrigerant} - {diagram_type}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    return fig


def make_empty_thdy_plot(diagram_type, cycle_config, output_dir="substance_thermodynamic_diagrams", verbose=True):
    refrigerant = cycle_config["refrigerant"]
    _configure_matplotlib()
    fig = _make_empty_plot_ts(cycle_config) if diagram_type == "TS" else _make_empty_plot_ph(cycle_config)
    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"Thermodynamic Diagram - {refrigerant} - {diagram_type}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    return fig



# Public entry point for COP_vs_eff_investigation
# ===============================================
def make_COP_vs_eff_plot(X, Y, Z, cycle_config, output_dir="00_obtained_data/COP_investigations"):
    """
    X, Y : 2D meshgrids (η_turb, η_compr)
    Z    : 2D COP values on the same grid
    """
    _configure_matplotlib()      
    refrigerant = cycle_config["refrigerant"]
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
    # ax.grid(True, alpha=0.3)

    # Save 
    fname2d = f"{cop_type}_vs_Efficiencies_{refrigerant}.pdf"
    fig.savefig(output_path / fname2d, dpi=1000, bbox_inches='tight')
    logger.info(f"Saved: {output_path / fname2d}")
    plt.close(fig)

    # 3D surface
    fig3d = plt.figure(figsize=(11, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', linewidth=0.2,
                             antialiased=True, alpha=0.95, rstride=1, cstride=1)

    fig3d.colorbar(surf, ax=ax3d, shrink=0.6, aspect=10, label=rf'${cop_type_latex}$')

    ax3d.set_xlabel(r'$\eta_{\mathrm{turb}}$')
    ax3d.set_ylabel(r'$\eta_{\mathrm{compr}}$')
    ax3d.set_zlabel(rf'${cop_type_latex}$')

    fname3d = f"{cop_type}_vs_Efficiencies_{refrigerant}_3D.pdf"
    fig3d.savefig(output_path / fname3d, dpi=800, bbox_inches='tight')
    logger.info(f"Saved: {output_path / fname3d}")
    plt.close(fig3d)


def make_PR_optimization_plot(
    PR_values,
    COP_values,
    cycle_config,
    output_dir="heat_pump_thermodynamic_diagrams",
    best_PR=None,
    best_COP=None,
    verbose=True,
):
    """Save COP-vs-PR plot for all successful PR evaluations."""
    if PR_values is None or COP_values is None:
        return None

    PR_arr = np.asarray(PR_values, dtype=float)
    COP_arr = np.asarray(COP_values, dtype=float)
    valid = np.isfinite(PR_arr) & np.isfinite(COP_arr)
    if valid.sum() == 0:
        return None

    PR_arr = PR_arr[valid]
    COP_arr = COP_arr[valid]
    order = np.argsort(PR_arr)
    PR_arr = PR_arr[order]
    COP_arr = COP_arr[order]

    _configure_matplotlib()
    refrigerant = cycle_config["refrigerant"]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    ax.plot(PR_arr, COP_arr, color="#1a3a6b", lw=1.2, zorder=2)
    ax.scatter(PR_arr, COP_arr, color="#1a3a6b", s=10, zorder=3)

    if best_PR is not None and best_COP is not None and np.isfinite(best_PR) and np.isfinite(best_COP):
        ax.scatter([best_PR], [best_COP], color="#d64545", s=40, zorder=4,
                   label=rf"$\mathrm{{Optimum}}: \mathrm{{PR}}={best_PR:.3f},\ { _cop_type_to_latex('COP_turb')}={best_COP:.3f}$")
        ax.legend(loc="best", fontsize=9, framealpha=0.85)

    ax.set_xlabel(r"$\mathrm{Pressure\ Ratio}\ [-]$")
    ax.set_ylabel(r"$\mathrm{COP}_{\mathrm{turb}}\ [-]$")
    ax.grid(True, alpha=0.25)

    output_root = Path(output_dir)
    output_path = output_root / refrigerant / f"PR Optimization - {refrigerant}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1000, bbox_inches="tight")
    if verbose:
        logger.info(f"Saved: {output_path}")
    plt.close(fig)
    return fig