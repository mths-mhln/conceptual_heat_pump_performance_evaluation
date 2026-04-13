import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/05_conceptual_heat_pump_performance_evaluation/')

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from config import cycle_config
from visualization import make_COP_vs_eff_plot
from logger import setup_logger


logger = setup_logger()


def _parse_metadata_from_filename(file_path: Path):
	"""Parse refrigerant and COP key from filename pattern.

	Expected pattern:
	COP_vs_eff_<refrigerant>_<cop_key>.npz
	Example:
	COP_vs_eff_R1234ze(E)_COP_turb.npz
	"""
	stem = file_path.stem
	match = re.match(r"^COP_vs_eff_(.+)_(COP_.+)$", stem)
	if not match:
		raise ValueError(
			f"Unsupported file naming format: {file_path.name}. "
			"Expected COP_vs_eff_<refrigerant>_<cop_key>.npz"
		)
	refrigerant, cop_key = match.group(1), match.group(2)
	return refrigerant, cop_key


def replot_from_saved_data(
	data_dir="COP_investigations/00_obtained_data",
	output_dir="COP_investigations",
):
	"""Recreate COP-vs-eff plots from saved NPZ grids only."""
	data_root = Path(data_dir)
	if not data_root.exists():
		raise FileNotFoundError(f"Data directory not found: {data_root}")

	files = sorted(data_root.glob("COP_vs_eff_*.npz"))
	if not files:
		raise FileNotFoundError(f"No COP sweep files found in: {data_root}")

	logger.info(f"Found {len(files)} saved COP sweep file(s) in {data_root}")

	for file_path in files:
		refrigerant, cop_key = _parse_metadata_from_filename(file_path)
		logger.info(
			f"Loading {file_path.name} (refrigerant={refrigerant}, sweep={cop_key})"
		)

		with np.load(file_path) as data:
			X = data["X"]
			Y = data["Y"]
			Z = data["Z"]

		cfg = dict(cycle_config)
		cfg["refrigerant"] = refrigerant

		# Uses the exact same plotting function/style as the original sweep workflow.
		make_COP_vs_eff_plot(X, Y, Z, cfg, output_dir=output_dir)
		logger.info(f"Replotted COP-vs-eff from saved data: {file_path.name}")

	logger.info("All saved COP sweep plots regenerated successfully.")


def plot_side_by_side_contours(
	data_dir="COP_investigations/00_obtained_data",
	output_dir="COP_investigations/comparison",
	refrigerants=("CO2", "R1234ze(E)"),
):
	"""Plot side-by-side COP contour plots for two refrigerants from saved NPZ data."""
	data_root = Path(data_dir)
	if not data_root.exists():
		raise FileNotFoundError(f"Data directory not found: {data_root}")

	plot_data = []
	for ref in refrigerants:
		pattern = f"COP_vs_eff_{ref}_COP_*.npz"
		matches = sorted(data_root.glob(pattern))
		if not matches:
			raise FileNotFoundError(
				f"No stored COP sweep file found for {ref} in {data_root} "
				f"with pattern {pattern}"
			)

		file_path = matches[0]
		refrigerant, cop_key = _parse_metadata_from_filename(file_path)
		with np.load(file_path) as data:
			X = data["X"]
			Y = data["Y"]
			Z = data["Z"]
		plot_data.append((refrigerant, cop_key, X, Y, Z, file_path.name))

	fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
	cop_label = r"$\mathrm{COP}_{\mathrm{turb}}$"

	for idx, (ax, (refrigerant, cop_key, X, Y, Z, source_name)) in enumerate(zip(axes, plot_data)):
		cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", extend="neither")
		cl = ax.contour(X, Y, Z, levels=12, colors="black", linewidths=1.0, linestyles="-")
		ax.clabel(cl, inline=True, fontsize=20, fmt="%.2f", colors="white", inline_spacing=4)

		cbar = fig.colorbar(cf, ax=ax, pad=0.02)
		cbar.set_label(cop_label, fontsize=24)
		cbar.ax.tick_params(labelsize=22)

		panel_label = f"({chr(ord('a') + idx)})"
		ax.set_title(f"{panel_label} {refrigerant}", fontsize=26, fontweight="bold", pad=12)
		ax.set_xlabel(r"$\eta_{\mathrm{turb}}$", fontsize=24)
		ax.set_ylabel(r"$\eta_{\mathrm{compr}}$", fontsize=24)
		ax.tick_params(axis="both", labelsize=22)
		logger.info(f"Prepared contour subplot for {refrigerant} from {source_name} ({cop_key})")



	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	out_pdf = output_root / "COP_turb_contours_CO2_vs_R1234ze(E).pdf"
	out_png = output_root / "COP_turb_contours_CO2_vs_R1234ze(E).png"

	fig.savefig(out_pdf, dpi=1000, bbox_inches="tight")
	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	plt.close(fig)

	logger.info(f"Saved side-by-side contour comparison: {out_pdf}")
	logger.info(f"Saved side-by-side contour comparison: {out_png}")


if __name__ == "__main__":
	replot_from_saved_data()
	plot_side_by_side_contours()
