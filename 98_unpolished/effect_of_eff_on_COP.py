import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/05_conceptual_heat_pump_performance_evaluation/')
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from config import cycle_config, general_config
from logger import setup_logger
from thermodynamics import _cp_props, _isobar_segment, solve_cycle
from visualization import (
	_configure_matplotlib,
	_draw_isolines_labeled,
	_isobar_lines_ts,
	_isenthalp_lines_ts,
	_q,
	_quality_isolines_ts,
	_saturation_dome_ts,
)


logger = setup_logger()


# Local-only margin controls for this script (does NOT touch config.py)
LOCAL_TS_MARGINS = {
	"s_left": 0.25,
	"s_right": 0.70,
	"T_bot": 0.25,
	"T_top": 0.35,
}


def _cycle_bounds_ts_local(cycle_data, cycle_cfg, margins=None):
	"""Cycle-centered TS bounds using script-local margins only."""
	m = LOCAL_TS_MARGINS if margins is None else margins

	s_vals = [cycle_data[k] for k in ("s_ref_1", "s_ref_2", "s_ref_3", "s_ref_4")]
	T_vals = [cycle_data[k] for k in ("T_ref_1", "T_ref_2", "T_ref_3", "T_ref_4")]

	# Include critical point to keep axis framing consistent with core visualization.
	p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_cfg)
	T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_cfg)
	s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_cfg)
	if np.isfinite(s_crit) and np.isfinite(T_crit):
		s_vals.append(s_crit)
		T_vals.append(T_crit)

	s_lo, s_hi = min(s_vals), max(s_vals)
	T_lo, T_hi = min(T_vals), max(T_vals)
	ds = max(s_hi - s_lo, 1e-12)
	dT = max(T_hi - T_lo, 1e-12)

	return (
		s_lo - ds * m["s_left"],
		s_hi + ds * m["s_right"],
		T_lo - dT * m["T_bot"],
		T_hi + dT * m["T_top"],
	)


def _plot_cycle_numbers(ax, cycle_data, point_overrides=None):
	"""Annotate the four cycle states with orange numbered markers."""
	point_overrides = {} if point_overrides is None else point_overrides
	point_specs = [
		("s_ref_1", "T_ref_1", "1", (10, -9), "top"),
		("s_ref_2", "T_ref_2", "2", (8, 7), "bottom"),
		("s_ref_3", "T_ref_3", "3", (10, 8), "bottom"),
		("s_ref_4", "T_ref_4", "4", (11, -5), "top"),
	]
	for s_key, T_key, label, offset, va in point_specs:
		if label in point_overrides:
			s_val, T_val = point_overrides[label]
		else:
			s_val = cycle_data[s_key]
			T_val = cycle_data[T_key]
		if not (np.isfinite(s_val) and np.isfinite(T_val)):
			continue
		ax.scatter([s_val], [T_val], color="orange", s=16, zorder=12)
		ax.annotate(
			label,
			xy=(s_val, T_val),
			xytext=offset,
			textcoords="offset points",
			color="black",
			fontsize=12,
			fontweight="bold",
			ha="right",
			va=va,
			zorder=13,
		)


def _solve_cycle_at_fixed_pr(base_cfg, pr_value, eta_turb_value):
	"""Solve a cycle using a fixed PR and turbine efficiency."""
	cfg = dict(base_cfg)
	cfg["PR"] = float(pr_value)
	cfg["η_turb"] = float(eta_turb_value)
	return solve_cycle(cfg, general_config, verbose=False)


def _solve_cycle_at_fixed_pr_and_compr(base_cfg, pr_value, eta_compr_value):
	"""Solve a cycle using a fixed PR and compressor efficiency."""
	cfg = dict(base_cfg)
	cfg["PR"] = float(pr_value)
	cfg["η_compr"] = float(eta_compr_value)
	return solve_cycle(cfg, general_config, verbose=False)


def _expansion_curve_ts(cycle_data, cycle_cfg, eta_turb, num_points=220):
	"""Return curved TS expansion path 3->4 using h(p)=h3-eta*(h3-h_is(p))."""
	refrigerant = cycle_cfg["refrigerant"]
	p3 = cycle_data["p_ref_3"]
	p1 = cycle_data["p_ref_1"]
	h3 = cycle_data["h_ref_3"]
	s3 = cycle_data["s_ref_3"]

	p_arr = np.linspace(p3, p1, num_points)
	s_arr = np.full_like(p_arr, np.nan, dtype=float)
	T_arr = np.full_like(p_arr, np.nan, dtype=float)

	for i, p in enumerate(p_arr):
		try:
			h_is = _cp_props("H", "P", p, "S", s3, f"REFPROP::{refrigerant}")
			h = h3 - eta_turb * (h3 - h_is)
			s_arr[i] = _cp_props("S", "P", p, "H", h, f"REFPROP::{refrigerant}")
			T_arr[i] = _cp_props("T", "P", p, "H", h, f"REFPROP::{refrigerant}")
		except Exception:
			pass

	valid = np.isfinite(s_arr) & np.isfinite(T_arr) & (T_arr > 1.0)
	return s_arr[valid], T_arr[valid]

def _plot_boundaryline(ax, s_arr, T_arr, color="black", lw=1.8,
			hatch_length=5.0,      # tune: length of each hatch (in K units)
			hatch_spacing=5.0,     # tune: spacing along the curve (in data units)
			hatch_side=1.0,        # +1 or -1 to flip hatch direction
			angle=0.0,             # hatch angle in degrees (0 = perpendicular, 20 = 20 deg from perpendicular)
			**kwargs):
	"""Mimic MATLAB boundaryline: line + perpendicular hatch marks with optional angle."""
	s = np.asarray(s_arr, dtype=float)
	T = np.asarray(T_arr, dtype=float)
	valid = np.isfinite(s) & np.isfinite(T)
	s = s[valid]
	T = T[valid]

	if len(s) < 2:
		return

	# Plot the main line (ignore any ls passed in kwargs)
	line_kwargs = {k: v for k, v in kwargs.items() if k not in ("ls", "linestyle")}
	ax.plot(s, T, color=color, lw=lw, **line_kwargs)

	# Arc-length parametrization
	ds = np.hypot(np.diff(s), np.diff(T))
	arc = np.concatenate(([0.0], np.cumsum(ds)))
	L = arc[-1]
	if L <= 0:
		return

	# Positions where hatches will be drawn
	num_hatches = max(1, int(L / max(hatch_spacing, 1e-12)))
	if L <= hatch_spacing:
		arc_hatches = np.array([0.5 * L])
	else:
		arc_hatches = np.linspace(hatch_spacing * 0.5, L - hatch_spacing * 0.5, num_hatches)

	# Interpolate position and tangent
	s_h = np.interp(arc_hatches, arc, s)
	T_h = np.interp(arc_hatches, arc, T)

	ds_darc = np.interp(arc_hatches, arc, np.gradient(s, arc))
	dT_darc = np.interp(arc_hatches, arc, np.gradient(T, arc))

	norm = np.hypot(ds_darc, dT_darc)
	mask = norm > 1e-12
	if not np.any(mask):
		return

	s_h = s_h[mask]
	T_h = T_h[mask]
	unit_tx = ds_darc[mask] / norm[mask]
	unit_ty = dT_darc[mask] / norm[mask]

	# Perpendicular vector (rotated 90 degrees from tangent)
	perp_s = -unit_ty * hatch_side
	perp_T = unit_tx * hatch_side

	# Rotate perpendicular vector by angle if specified
	if abs(angle) > 1e-6:
		angle_rad = np.radians(angle)
		cos_a = np.cos(angle_rad)
		sin_a = np.sin(angle_rad)
		rotated_s = perp_s * cos_a - perp_T * sin_a
		rotated_T = perp_s * sin_a + perp_T * cos_a
		s_end = s_h + hatch_length * rotated_s
		T_end = T_h + hatch_length * rotated_T
	else:
		s_end = s_h + hatch_length * perp_s
		T_end = T_h + hatch_length * perp_T

	# Draw each hatch
	for i in range(len(s_h)):
		ax.plot([s_h[i], s_end[i]], [T_h[i], T_end[i]],
				color=color, lw=lw * 0.65, solid_capstyle="butt")

def _plot_compressor_efficiency_overlay(pr_ref, output_dir="98_unpolished/custom_cycle_plots"):
	"""Plot cycles at the same PR for varying compressor efficiencies."""
	eta_compr_values = np.linspace(0.3, 1.0, 4)
	cycles = []
	for eta_compr in eta_compr_values:
		cycle = _solve_cycle_at_fixed_pr_and_compr(cycle_config, pr_ref, eta_compr)
		cycles.append((eta_compr, cycle))

	s_lo, s_hi, T_lo, T_hi = _cycle_bounds_ts_local(cycles[-1][1], cycle_config)
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.set_xlim(s_lo, s_hi)
	ax.set_ylim(T_lo, T_hi)

	resolution = general_config.get("resolution", "low")
	n_pts = 150 if resolution == "low" else 600

	s_dome, T_dome = _saturation_dome_ts(cycle_config, n=n_pts * 2)
	ax.plot(s_dome, T_dome, color="black", lw=1.0, zorder=3)

	# Critical point and critical isobar.
	p_crit = _q("Pcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
	T_crit = _q("Tcrit", "T", 300, "Q", 1, cycle_config=cycle_config)
	sc, Tc = _isobar_segment(s_lo, s_hi, p_crit, cycle_config, general_config)
	sc = np.array(sc)
	Tc = np.array(Tc)
	valid_crit = (Tc > T_lo) & (Tc <= T_hi * 1.05) & np.isfinite(Tc) & (Tc > 1.0)
	if valid_crit.any():
		ax.plot(sc[valid_crit], Tc[valid_crit], color="black", ls=":", lw=1.0, zorder=1)
	s_crit = _q("S", "P", p_crit, "T", T_crit, cycle_config=cycle_config)
	if np.isfinite(s_crit) and np.isfinite(T_crit):
		ax.plot(s_crit, T_crit, marker="o", markerfacecolor="yellow", markersize=5, markeredgecolor="black", zorder=9)

	_draw_isolines_labeled(
		ax,
		_quality_isolines_ts(T_lo, T_hi, cycle_config, n_pts=n_pts),
		"#1a3a6b",
		0.3,
		fmt_short=lambda v: rf"${v:.2f}$",
		fmt_named=lambda v: rf"$x={v:.2f}$",
		flip_q1=(cycle_config["refrigerant"] == "R1234ze(Z)"),
	)
	_draw_isolines_labeled(
		ax,
		_isobar_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
		"#6ab0de",
		0.85,
		fmt_short=lambda v: rf"${v/1e3:.0f}$",
		fmt_named=lambda v: rf"$p={v/1e3:.0f}\,\mathrm{{kPa}}$",
	)
	_draw_isolines_labeled(
		ax,
		_isenthalp_lines_ts(s_lo, s_hi, T_lo, T_hi, cycle_config, n_pts=n_pts),
		"#e07b20",
		0.90,
		fmt_short=lambda v: rf"${v/1e3:.0f}$",
		fmt_named=lambda v: rf"$h={v/1e3:.0f}\,\mathrm{{kJ/kg}}$",
	)

	# Plot compressor-efficiency variants: compression process only (1 -> 2).
	for eta_compr, cycle in cycles:
		if np.isclose(eta_compr, 1.0):
			color = "black"
			lw = 2.0
			alpha = 1.0
		else:
			color = "green"
			lw = 1.4
			alpha = 0.4 + 0.32 * (eta_compr - 0.3) / 0.7
		s1, T1 = cycle["s_ref_1"], cycle["T_ref_1"]
		s2, T2 = cycle["s_ref_2"], cycle["T_ref_2"]
		if np.isfinite(s1) and np.isfinite(T1) and np.isfinite(s2) and np.isfinite(T2):
			ax.plot([s1, s2], [T1, T2], color=color, lw=lw, alpha=alpha, zorder=8)

	# Add turbine-efficiency overlays at the same PR: η_turb in [0, 1], 5 samples, all light green.
	eta_turb_values = np.linspace(0.0, 1.0, 5)
	for eta_turb in eta_turb_values:
		if np.isclose(eta_turb, cycle_config["η_turb"]):
			continue
		try:
			cyc_turb = _solve_cycle_at_fixed_pr(cycle_config, pr_ref, eta_turb)
			alpha_turb = 0.4 + 0.32 * eta_turb
			cfg_turb = dict(cycle_config, PR=pr_ref, η_turb=eta_turb)
			s_turb, T_turb = _expansion_curve_ts(cyc_turb, cfg_turb, eta_turb)
			if len(s_turb) > 1:
				ax.plot(s_turb, T_turb, color="green", lw=1.2, alpha=alpha_turb, zorder=6)
		except Exception as exc:
			logger.warning(f"Skipping turbine overlay at η_turb={eta_turb:.2f}: {exc}")

	# Keep both black expansion paths and add isentropic-cycle evaporation/condensation paths.
	cycle_ref = cycles[-1][1]
	refrigerant = cycle_config["refrigerant"]

	# Isenthalpic expansion (compressor boundary) starting at station 1, going right.
	# This represents the pure isenthalpic (throttling) process η_compr = 0.
	xl, xh = ax.get_xlim()
	xh_right = xh
	p_compr_isenthalpic = np.linspace(cycle_ref["p_ref_1"], 0.01 * cycle_ref["p_ref_1"], 220)
	s_compr_isenth = []
	T_compr_isenth = []
	for p in p_compr_isenthalpic:
		try:
			s_compr_isenth.append(_cp_props("S", "P", p, "H", cycle_ref["h_ref_1"], f"REFPROP::{refrigerant}"))
			T_compr_isenth.append(_cp_props("T", "P", p, "H", cycle_ref["h_ref_1"], f"REFPROP::{refrigerant}"))
		except Exception:
			s_compr_isenth.append(np.nan)
			T_compr_isenth.append(np.nan)
	s_compr_isenth = np.array(s_compr_isenth)
	T_compr_isenth = np.array(T_compr_isenth)
	valid_compr_isenth = np.isfinite(s_compr_isenth) & np.isfinite(T_compr_isenth)
	if np.any(valid_compr_isenth):
		_plot_boundaryline(
        ax,
        s_compr_isenth[valid_compr_isenth],
        T_compr_isenth[valid_compr_isenth],
        color="black",
        lw=2.0,
        hatch_length=7.0,
        hatch_spacing=8.0,
		hatch_side=-1.0,
        angle=-65.0,
        zorder=11
        )

	# Isenthalpic expansion path (3 -> 4_h) in black.
	p_exp = np.linspace(cycle_ref["p_ref_3"], cycle_ref["p_ref_1"], 220)
	s_isenthalpic = []
	T_isenthalpic = []
	for p in p_exp:
		try:
			s_isenthalpic.append(_cp_props("S", "P", p, "H", cycle_ref["h_ref_3"], f"REFPROP::{refrigerant}"))
			T_isenthalpic.append(_cp_props("T", "P", p, "H", cycle_ref["h_ref_3"], f"REFPROP::{refrigerant}"))
		except Exception:
			s_isenthalpic.append(np.nan)
			T_isenthalpic.append(np.nan)
	valid_isenthalpic = np.isfinite(s_isenthalpic) & np.isfinite(T_isenthalpic)
	if np.any(valid_isenthalpic):
		s4_h = float(np.array(s_isenthalpic)[valid_isenthalpic][-1])
		T4_h = float(np.array(T_isenthalpic)[valid_isenthalpic][-1])
	else:
		s4_h, T4_h = np.nan, np.nan
	# ax.plot(
	# 	np.array(s_isenthalpic)[valid_isenthalpic],
	# 	np.array(T_isenthalpic)[valid_isenthalpic],
	# 	color="black",
	# 	lw=1.8,
	# 	ls="-",
	# 	zorder=10,
	# )
	_plot_boundaryline(
    ax,
    np.array(s_isenthalpic)[valid_isenthalpic],
    np.array(T_isenthalpic)[valid_isenthalpic],
    color="black",
    lw=1.8,
    hatch_length=5.0,      # <-- tune these two values to your liking
    hatch_spacing=5,
    zorder=10
    )

	# Isentropic expansion path (3 -> 4_s) in black.
	p_exp = np.linspace(cycle_ref["p_ref_3"], cycle_ref["p_ref_1"], 220)
	s_isen = np.full_like(p_exp, cycle_ref["s_ref_3"], dtype=float)
	T_isen = np.full_like(p_exp, np.nan, dtype=float)
	for i, p in enumerate(p_exp):
		try:
			T_isen[i] = _cp_props("T", "P", p, "S", cycle_ref["s_ref_3"], f"REFPROP::{refrigerant}")
		except Exception:
			pass
	valid_isen = np.isfinite(s_isen) & np.isfinite(T_isen)
	# ax.plot(s_isen[valid_isen], T_isen[valid_isen], color="black", lw=2.0, ls="-", zorder=10)
	_plot_boundaryline(
    ax,
    s_isen[valid_isen],
    T_isen[valid_isen],
    color="black",
    lw=2.0,
    hatch_length=6.0,
    hatch_spacing=4.0,
	hatch_side=-1.0,  # flip hatch direction for isentropic path
	angle = -20.0,
    zorder=10
    )

	# Isentropic expansion state 4_s at p1.
	try:
		h4_s = _cp_props("H", "P", cycle_ref["p_ref_1"], "S", cycle_ref["s_ref_3"], f"REFPROP::{refrigerant}")
		s4_s = _cp_props("S", "P", cycle_ref["p_ref_1"], "H", h4_s, f"REFPROP::{refrigerant}")
		T4_s = _cp_props("T", "P", cycle_ref["p_ref_1"], "H", h4_s, f"REFPROP::{refrigerant}")
	except Exception:
		s4_s, T4_s = np.nan, np.nan

	# Mark isentropic expansion endpoint as station 4'.
	if np.isfinite(s4_s) and np.isfinite(T4_s):
		ax.scatter([s4_s], [T4_s], color="orange", s=16, zorder=14)
		ax.annotate(
			"4'",
			xy=(s4_s, T4_s),
			xytext=(10, -5),
			textcoords="offset points",
			color="black",
			fontsize=12,
			fontweight="bold",
			ha="right",
			va="top",
			zorder=15,
		)

	# Segment between isenthalpic and isentropic state 4 in gray.
	if np.isfinite(s4_h) and np.isfinite(T4_h) and np.isfinite(s4_s) and np.isfinite(T4_s):
		ax.plot([s4_h, s4_s], [T4_h, T4_s], color="#999999", lw=1.8, zorder=10)

	# Isenthalpic-cycle evaporation process (4_h -> 1) in black along p1.
	if np.isfinite(s4_h) and np.isfinite(T4_h):
		s_evap_h, T_evap_h = _isobar_segment(s4_h, cycle_ref["s_ref_1"], cycle_ref["p_ref_1"], cycle_config, general_config)
		s_evap_h = np.array(s_evap_h)
		T_evap_h = np.array(T_evap_h)
		valid_evap_h = np.isfinite(s_evap_h) & np.isfinite(T_evap_h) & (T_evap_h > 1.0)
		ax.plot(s_evap_h[valid_evap_h], T_evap_h[valid_evap_h], color="black", lw=2, zorder=11)

	# Isentropic-cycle evaporation process (4_s -> 1) in gray along p1.
	if np.isfinite(s4_s) and np.isfinite(T4_s):
		s_evap_s, T_evap_s = _isobar_segment(s4_s, cycle_ref["s_ref_1"], cycle_ref["p_ref_1"], cycle_config, general_config)
		s_evap_s = np.array(s_evap_s)
		T_evap_s = np.array(T_evap_s)
		valid_evap_s = np.isfinite(s_evap_s) & np.isfinite(T_evap_s) & (T_evap_s > 1.0)
		ax.plot(s_evap_s[valid_evap_s], T_evap_s[valid_evap_s], color="#999999", lw=2, zorder=9)

	# Isentropic-compression path (1 -> 2_s) and its condensation segment (2_s -> 3) in black.
	try:
		h2_s = _cp_props("H", "P", cycle_ref["p_ref_2"], "S", cycle_ref["s_ref_1"], f"REFPROP::{refrigerant}")
		s2_s = _cp_props("S", "P", cycle_ref["p_ref_2"], "H", h2_s, f"REFPROP::{refrigerant}")
		T2_s = _cp_props("T", "P", cycle_ref["p_ref_2"], "H", h2_s, f"REFPROP::{refrigerant}")
		if np.isfinite(s2_s) and np.isfinite(T2_s):
			# ax.plot([cycle_ref["s_ref_1"], s2_s], [cycle_ref["T_ref_1"], T2_s], color="black", lw=2, zorder=10)
			_plot_boundaryline(
            ax,
            [cycle_ref["s_ref_1"], s2_s],
            [cycle_ref["T_ref_1"], T2_s],
            color="black",
            lw=2.0,
            hatch_length=6.0,
            hatch_spacing=4.0,
            hatch_side=1.0,  # flip hatch direction for isentropic path
            angle=20.0,
            zorder=10
            )
			s_cond_s, T_cond_s = _isobar_segment(s2_s, cycle_ref["s_ref_3"], cycle_ref["p_ref_2"], cycle_config, general_config)
			s_cond_s = np.array(s_cond_s)
			T_cond_s = np.array(T_cond_s)
			valid_cond_s = np.isfinite(s_cond_s) & np.isfinite(T_cond_s) & (T_cond_s > 1.0)
			ax.plot(s_cond_s[valid_cond_s], T_cond_s[valid_cond_s], color="black", lw=2, zorder=10)
	except Exception:
		pass

	# Plot condenser-pressure isobar in dash-dot, but hide the segment between stations 2 and 3.
	s_cond, T_cond = _isobar_segment(s_lo, s_hi, cycle_ref["p_ref_2"], cycle_config, general_config)
	s_cond = np.array(s_cond)
	T_cond = np.array(T_cond)
	valid_cond = np.isfinite(s_cond) & np.isfinite(T_cond) & (T_cond > 1.0)
	if np.any(valid_cond):
		sv = s_cond[valid_cond]
		Tv = T_cond[valid_cond]
		s2 = cycle_ref["s_ref_2"]
		s3 = cycle_ref["s_ref_3"]
		s_left = min(s2, s3)
		s_right = max(s2, s3)

		left_mask = sv < s_left
		right_mask = sv > s_right
		ax.plot(sv[left_mask], Tv[left_mask], color="green", lw=2.1, ls="-.", alpha=0.4, zorder=10)
		ax.plot(sv[right_mask], Tv[right_mask], color="green", lw=2.1, ls="-.", alpha=0.4, zorder=10)

	# Add textbox labels for boundaries
	# Turbine isenthalpic boundary (η_turb = 0)
	if np.any(valid_isenthalpic) and len(s_isenthalpic) > 0:
		s_mid = np.median(np.array(s_isenthalpic)[valid_isenthalpic])
		T_mid = np.median(np.array(T_isenthalpic)[valid_isenthalpic])
		if np.isfinite(s_mid) and np.isfinite(T_mid):
			ax.text(s_mid*1.032, T_mid*0.952, r"$\eta_{\mathrm{turb}} = 0$",
				fontsize=13, color="black", ha="center", va="center",
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
				zorder=16)

	# Turbine isentropic boundary (η_turb = 1)
	if np.any(valid_isen) and len(s_isen) > 0:
		s_mid_isen = np.median(np.array(s_isen)[valid_isen])
		T_mid_isen = np.median(np.array(T_isen)[valid_isen])
		if np.isfinite(s_mid_isen) and np.isfinite(T_mid_isen):
			ax.text(s_mid_isen*0.976, T_mid_isen*0.95, r"$\eta_{\mathrm{turb}} = 1$",
				fontsize=13, color="black", ha="center", va="center",
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
				zorder=16)

	# Compressor isenthalpic boundary (η_compr = 0)
	if np.any(valid_compr_isenth) and len(s_compr_isenth) > 0:
		s_mid_compr = np.median(s_compr_isenth[valid_compr_isenth])
		T_mid_compr = np.median(T_compr_isenth[valid_compr_isenth])
		if np.isfinite(s_mid_compr) and np.isfinite(T_mid_compr):
			ax.text(s_mid_compr*1.03, T_mid_compr*0.96, r"$\eta_{\mathrm{compr}} = 0$",
				fontsize=13, color="black", ha="center", va="center",
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
				zorder=16)

	# Compressor isentropic boundary (η_compr = 1) - the isentropic compression path
	if np.isfinite(s2_s) and np.isfinite(T2_s):
		s_mid_compr_isen = 0.5 * (cycle_ref["s_ref_1"] + s2_s)
		T_mid_compr_isen = 0.5 * (cycle_ref["T_ref_1"] + T2_s)
		ax.text(s_mid_compr_isen*0.976, T_mid_compr_isen*0.992, r"$\eta_{\mathrm{compr}} = 1$",
				fontsize=13, color="black", ha="center", va="center",
				bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
				zorder=16)

	_plot_cycle_numbers(ax, cycle_ref, point_overrides={"4": (s4_h, T4_h)} if np.isfinite(s4_h) and np.isfinite(T4_h) else None)
	ax.set_xlabel(r"$s\ [\mathrm{J/kg/K}]$", fontsize=14)
	ax.set_ylabel(r"$T\ [\mathrm{K}]$", fontsize=14)

	# Manual legend box with a true hatched sample drawn by _plot_boundaryline.
	ds_plot = s_hi - s_lo
	dT_plot = T_hi - T_lo
	box_x0 = s_lo + 0.02 * ds_plot
	box_x1 = s_lo + 0.26 * ds_plot
	box_y0 = T_hi - 0.10 * dT_plot
	box_y1 = T_hi - 0.03 * dT_plot
	ax.plot([box_x0, box_x1], [box_y0, box_y0], color="black", lw=1.0, zorder=18)
	ax.plot([box_x1, box_x1], [box_y0, box_y1], color="black", lw=1.0, zorder=18)
	ax.plot([box_x1, box_x0], [box_y1, box_y1], color="black", lw=1.0, zorder=18)
	ax.plot([box_x0, box_x0], [box_y1, box_y0], color="black", lw=1.0, zorder=18)
	s_leg0 = box_x0 + 0.02 * ds_plot
	s_leg1 = box_x0 + 0.12 * ds_plot
	T_leg = box_y0 + 0.032 * dT_plot
	T_leg_boundaryline = box_y0 + 0.035 * dT_plot
	_plot_boundaryline(
		ax,
		[s_leg0, s_leg1],
		[T_leg_boundaryline, T_leg_boundaryline],
		color="black",
		lw=2,
		hatch_length=4,
		hatch_spacing=6,
		hatch_side=-1.0,
		angle=-55.0,
		zorder=19,
	)
	ax.text(
		box_x0 + 0.14 * ds_plot,
		T_leg,
		"Boundary",
		fontsize=11,
		color="black",
		ha="left",
		va="center",
		zorder=19,
	)

	out_root = Path(output_dir) / cycle_config["refrigerant"]
	out_root.mkdir(parents=True, exist_ok=True)
	out_pdf = out_root / "effect_of_compr_and_turb_eff_on_COP.pdf"
	out_png = out_root / "effect_of_compr_and_turb_eff_on_COP.png"
	fig.savefig(out_pdf, dpi=1000, bbox_inches="tight")
	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	plt.close(fig)
	logger.info(f"Saved: {out_pdf}")
	logger.info(f"Saved: {out_png}")


def plot_single_cycle_custom_ts(output_dir="98_unpolished/custom_cycle_plots"):
	"""Generate only the compressor-efficiency TS overlay plot."""
	_configure_matplotlib()

	cycle = solve_cycle(cycle_config, general_config, verbose=False)
	pr_ref = cycle["p_ref_2"] / cycle["p_ref_1"]
	# Generate only the compressor-efficiency comparison at this pressure ratio.
	_plot_compressor_efficiency_overlay(pr_ref, output_dir=output_dir)


if __name__ == "__main__":
	plot_single_cycle_custom_ts()
