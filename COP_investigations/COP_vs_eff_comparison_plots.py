import os
import sys
cwd = os.getcwd()
sys.path.append(f'{cwd}')

import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from logger import setup_logger
from visualization import _configure_matplotlib


logger = setup_logger()


def _parse_metadata_from_filename(file_path: Path):
	"""Parse refrigerant and COP key from COP sweep filename."""
	stem = file_path.stem
	match = re.match(r"^COP_vs_eff_(.+)_(COP_.+)$", stem)
	if not match:
		raise ValueError(
			f"Unsupported file naming format: {file_path.name}. "
			"Expected COP_vs_eff_<refrigerant>_<cop_key>.npz"
		)
	return match.group(1), match.group(2)


def _find_saved_grid(data_root: Path, refrigerant: str, cop_key: str = "COP_turb"):
	"""Find the saved NPZ sweep for a refrigerant, preferring COP_investigations/<refrigerant>/data/."""
	ref_dir = data_root / refrigerant
	if not ref_dir.exists():
		raise FileNotFoundError(
			f"Refrigerant directory not found: {ref_dir}. "
			f"Expected data in COP_investigations/{refrigerant}/"
		)

	data_dir = ref_dir / "data"
	search_dir = data_dir if data_dir.exists() else ref_dir

	exact = search_dir / f"COP_vs_eff_{refrigerant}_{cop_key}.npz"
	if exact.exists():
		return exact

	matches = sorted(search_dir.glob(f"COP_vs_eff_{refrigerant}_COP_*.npz"))
	if not matches:
		raise FileNotFoundError(
			f"No COP sweep files found for {refrigerant} in {search_dir}. "
			f"Expected file like COP_vs_eff_{refrigerant}_{cop_key}.npz"
		)

	if len(matches) > 1:
		logger.warning(
			f"Multiple COP sweep files found for {refrigerant}; using {matches[0].name}"
		)

	return matches[0]


def plot_side_by_side_contours(
	data_dir="COP_investigations",
	output_dir="COP_investigations/00_comparison",
	refrigerants=("CO2", "R1234ze(E)"),
	cop_key="COP_turb",
):
	"""Plot side-by-side COP contour plots for two refrigerants from saved NPZ data folders."""
	_configure_matplotlib()

	if len(refrigerants) != 2:
		raise ValueError("Please provide exactly two refrigerants.")

	data_root = Path(data_dir)
	if not data_root.exists():
		raise FileNotFoundError(f"Data directory not found: {data_root}")

	plot_data = []
	for ref in refrigerants:
		file_path = _find_saved_grid(data_root, ref, cop_key=cop_key)
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
	ref_a, ref_b = refrigerants
	safe_a = ref_a.replace("/", "-")
	safe_b = ref_b.replace("/", "-")
	out_pdf = output_root / f"{cop_key}_contours_{safe_a}_vs_{safe_b}.pdf"
	out_png = output_root / f"{cop_key}_contours_{safe_a}_vs_{safe_b}.png"

	fig.savefig(out_pdf, dpi=1000, bbox_inches="tight")
	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	plt.close(fig)

	logger.info(f"Saved side-by-side contour comparison: {out_pdf}")
	logger.info(f"Saved side-by-side contour comparison: {out_png}")


def _build_arg_parser():
	parser = argparse.ArgumentParser(
		description="Plot side-by-side COP contours for two refrigerants from saved NPZ data."
	)
	parser.add_argument(
		"--refrigerants",
		nargs=2,
		metavar=("REF_A", "REF_B"),
		default=["CO2", "R1234ze(E)"],
		help="Two refrigerants to compare (folders expected under COP_investigations/).",
	)
	parser.add_argument(
		"--data-dir",
		default="COP_investigations",
		help="Root directory that contains refrigerant folders.",
	)
	parser.add_argument(
		"--output-dir",
		default="COP_investigations/comparison",
		help="Directory for comparison figures.",
	)
	parser.add_argument(
		"--cop-key",
		default="COP_turb",
		help="Requested COP key suffix (e.g., COP_turb).",
	)
	return parser


if __name__ == "__main__":
	args = _build_arg_parser().parse_args()
	plot_side_by_side_contours(
		data_dir=args.data_dir,
		output_dir=args.output_dir,
		refrigerants=tuple(args.refrigerants),
		cop_key=args.cop_key,
	)
