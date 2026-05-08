import os
import sys
cwd = os.getcwd()
sys.path.append(f'{cwd}/')

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import cycle_config, general_config
from logger import setup_logger


logger = setup_logger()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
TARGET_REFRIGERANT = "R1234ze(E)"
COP_KEY = "COP_turb"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def _load_xyz(npz_path):
    data = np.load(npz_path)
    required = ("X", "Y", "Z")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"File {npz_path} is missing keys: {missing}")
    return data["X"], data["Y"], data["Z"], (float(data["PR"]) if "PR" in data else np.nan)


def _build_efficiency_grid(base_general_cfg):
    n = 200 if base_general_cfg.get("resolution") == "high" else 20
    eta_turb_arr = np.linspace(0.01, 1.0, n)
    eta_compr_arr = np.linspace(0.01, 1.0, n)
    return np.meshgrid(eta_turb_arr, eta_compr_arr), n


def _save_cop_heatmap(X, Y, Z, out_path, title, cop_key):
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", extend="neither")
    cl = ax.contour(X, Y, Z, levels=12, colors="black", linewidths=0.8, linestyles="-")
    ax.clabel(cl, inline=True, fontsize=9, fmt="%.2f", colors="white", manual=False, inline_spacing=3)
    plt.colorbar(cf, ax=ax, label=cop_key)
    ax.set_xlabel(r"$\eta_{\mathrm{turb}}$")
    ax.set_ylabel(r"$\eta_{\mathrm{compr}}$")
    ax.set_title(title)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def _resolve_data_file(file_name, data_dir):
    candidate = Path(file_name)
    if candidate.exists():
        return candidate

    data_candidate = Path(data_dir) / file_name
    if data_candidate.exists():
        return data_candidate

    raise FileNotFoundError(
        f"Data file not found: {file_name}. Checked {candidate} and {data_candidate}."
    )


def _display_label(file_path):
    return file_path.stem.replace("_", " ")


def _discover_npz_files(data_dir):
    """Discover exactly two NPZ files in the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    npz_files = sorted(data_dir.glob("*.npz"))
    
    if len(npz_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 NPZ files in {data_dir}, but found {len(npz_files)}."
        )
    
    if len(npz_files) > 2:
        logger.warning(
            f"Found {len(npz_files)} NPZ files in {data_dir}. Using the first two: "
            f"{npz_files[0].name} and {npz_files[1].name}"
        )
    
    return npz_files[0], npz_files[1]


# def _compute_fixed_pr_file(refrigerant, cop_key, output_dir):
#     """Compute and save the fixed-PR efficiency sweep for a refrigerant."""
#     cfg_base = dict(cycle_config)
#     cfg_base["refrigerant"] = refrigerant
#     gen_cfg = dict(general_config)

#     # One optimization call to get the reference PR.
#     reference_cycle = solve_cycle(dict(cfg_base), gen_cfg, verbose=False)
#     reference_pr = float(reference_cycle["PR"])

#     (X, Y), n = _build_efficiency_grid(gen_cfg)
#     Z = np.full_like(X, np.nan, dtype=float)
#     total_runs = n * n

#     logger.info(f"Computing fixed-PR sweep for {refrigerant} at PR={reference_pr:.6f}")
#     for k in track(range(total_runs), description="\033[92mINFO    "):
#         i, j = divmod(k, n)
#         cfg = dict(cfg_base)
#         cfg["PR"] = reference_pr
#         cfg["η_turb"] = float(X[i, j])
#         cfg["η_compr"] = float(Y[i, j])
#         try:
#             cycle_data = solve_cycle(cfg, gen_cfg, verbose=False)
#             perf = compute_performance(cycle_data, cfg, gen_cfg)
#             Z[i, j] = perf[cop_key]
#         except Exception:
#             Z[i, j] = np.nan

#     root = Path(output_dir) / refrigerant / "data"
#     root.mkdir(parents=True, exist_ok=True)
#     fixed_path = root / f"COP_vs_eff_{refrigerant}_{cop_key}_fixedPR.npz"
#     np.savez_compressed(fixed_path, X=X, Y=Y, Z=Z, PR=reference_pr)
#     logger.info(f"Saved computed fixed-PR data: {fixed_path}")

#     fig_path = root / f"{cop_key}_vs_Efficiencies_{refrigerant}_fixedPR.pdf"
#     _save_cop_heatmap(
#         X,
#         Y,
#         Z,
#         fig_path,
#         title=rf"{cop_key} at fixed PR={reference_pr:.4f} ({refrigerant})",
#         cop_key=cop_key,
#     )

#     return fixed_path


# def _compute_reoptimized_file(refrigerant, cop_key, output_dir):
#     """Compute and save original implementation sweep (PR optimized at every point)."""
#     cfg_base = dict(cycle_config)
#     cfg_base["refrigerant"] = refrigerant
#     gen_cfg = dict(general_config)

#     (X, Y), n = _build_efficiency_grid(gen_cfg)
#     Z = np.full_like(X, np.nan, dtype=float)
#     total_runs = n * n

#     logger.info(f"Computing re-optimized sweep for {refrigerant}")
#     for k in track(range(total_runs), description="\033[92mINFO    "):
#         i, j = divmod(k, n)
#         cfg = dict(cfg_base)
#         cfg.pop("PR", None)
#         cfg["η_turb"] = float(X[i, j])
#         cfg["η_compr"] = float(Y[i, j])
#         try:
#             cycle_data = solve_cycle(cfg, gen_cfg, verbose=False)
#             perf = compute_performance(cycle_data, cfg, gen_cfg)
#             Z[i, j] = perf[cop_key]
#         except Exception:
#             Z[i, j] = np.nan

#     root = Path(output_dir) / refrigerant / "data"
#     root.mkdir(parents=True, exist_ok=True)
#     reopt_path = root / f"COP_vs_eff_{refrigerant}_{cop_key}_reoptimized.npz"
#     np.savez_compressed(reopt_path, X=X, Y=Y, Z=Z)
#     logger.info(f"Saved computed re-optimized data: {reopt_path}")

#     fig_path = root / f"{cop_key}_vs_Efficiencies_{refrigerant}_reoptimized.pdf"
#     _save_cop_heatmap(
#         X,
#         Y,
#         Z,
#         fig_path,
#         title=rf"{cop_key} with PR optimization per point ({refrigerant})",
#         cop_key=cop_key,
#     )

#     return reopt_path


def run_deviation_study_from_existing(
    data1_file=None,
    data2_file=None,
    data_dir=DEFAULT_DATA_DIR,
    output_dir=OUTPUT_DIR,
):
    """Compare two NPZ sweeps from a local data folder and save the deviation plots next to this script.
    
    Args:
        data1_file: File name of the first NPZ file. If None, auto-discovered from data_dir.
        data2_file: File name of the second NPZ file. If None, auto-discovered from data_dir.
        data_dir: Directory containing the NPZ files.
        output_dir: Directory where the deviation plots will be saved.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    if data1_file is None or data2_file is None:
        discovered_1, discovered_2 = _discover_npz_files(data_dir)
        data1_file = data1_file or discovered_1.name
        data2_file = data2_file or discovered_2.name
        logger.info(f"Auto-discovered data files: {data1_file}, {data2_file}")

    data1_path = _resolve_data_file(data1_file, data_dir)
    data2_path = _resolve_data_file(data2_file, data_dir)

    logger.info(f"Loading data1: {data1_path}")
    logger.info(f"Loading data2: {data2_path}")

    Xf, Yf, Z_data1, pr_data1 = _load_xyz(data1_path)
    Xr, Yr, Z_data2, _ = _load_xyz(data2_path)

    if Xf.shape != Xr.shape or Yf.shape != Yr.shape or Z_data1.shape != Z_data2.shape:
        raise ValueError(
            "Existing files have incompatible shapes: "
            f"data1={Z_data1.shape}, data2={Z_data2.shape}"
        )

    if not (np.allclose(Xf, Xr, equal_nan=True) and np.allclose(Yf, Yr, equal_nan=True)):
        raise ValueError("Efficiency grids in existing files do not match; cannot compute point-wise deviation.")

    X, Y = Xf, Yf
    valid = np.isfinite(Z_data1) & np.isfinite(Z_data2)

    delta = np.full_like(Z_data1, np.nan, dtype=float)
    rel_pct = np.full_like(Z_data1, np.nan, dtype=float)
    delta[valid] = Z_data1[valid] - Z_data2[valid]

    denom_ok = valid & (np.abs(Z_data2) > 1e-12)
    rel_pct[denom_ok] = 100.0 * delta[denom_ok] / np.abs(Z_data2[denom_ok])

    abs_delta = np.abs(delta)
    abs_rel_pct = np.abs(rel_pct)

    mean_abs_delta = float(np.nanmean(abs_delta)) if np.any(valid) else np.nan
    max_abs_delta = float(np.nanmax(abs_delta)) if np.any(valid) else np.nan
    rmse_delta = float(np.sqrt(np.nanmean(delta[valid] ** 2))) if np.any(valid) else np.nan

    mean_abs_rel_pct = float(np.nanmean(abs_rel_pct)) if np.any(denom_ok) else np.nan
    max_abs_rel_pct = float(np.nanmax(abs_rel_pct)) if np.any(denom_ok) else np.nan

    if np.any(np.isfinite(abs_delta)):
        i_max_abs, j_max_abs = np.unravel_index(np.nanargmax(abs_delta), abs_delta.shape)
        eta_turb_at_max_abs = float(X[i_max_abs, j_max_abs])
        eta_compr_at_max_abs = float(Y[i_max_abs, j_max_abs])
    else:
        eta_turb_at_max_abs = np.nan
        eta_compr_at_max_abs = np.nan

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    data1_label = _display_label(data1_path)
    data2_label = _display_label(data2_path)
    comparison_slug = f"{data1_path.stem}_vs_{data2_path.stem}".replace("/", "-")

    dev_data_path = out_root / f"{comparison_slug}_deviation.npz"
    np.savez_compressed(
        dev_data_path,
        X=X,
        Y=Y,
        Z_data1=Z_data1,
        Z_data2=Z_data2,
        delta=delta,
        rel_pct=rel_pct,
        PR_data1=pr_data1,
        data1_source=str(data1_path),
        data2_source=str(data2_path),
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        rmse_delta=rmse_delta,
        mean_abs_rel_pct=mean_abs_rel_pct,
        max_abs_rel_pct=max_abs_rel_pct,
        eta_turb_at_max_abs=eta_turb_at_max_abs,
        eta_compr_at_max_abs=eta_compr_at_max_abs,
    )
    logger.info(f"Saved deviation data: {dev_data_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    c0 = axes[0].contourf(X, Y, delta, levels=30, cmap="coolwarm", extend="both")
    plt.colorbar(c0, ax=axes[0], label=rf"$\Delta$ {COP_KEY} (data1 - data2)")
    axes[0].set_xlabel(r"$\eta_{\mathrm{turb}}$")
    axes[0].set_ylabel(r"$\eta_{\mathrm{compr}}$")
    axes[0].set_title(f"Absolute Deviation\n{data1_label} - {data2_label}")

    c1 = axes[1].contourf(X, Y, abs_rel_pct, levels=30, cmap="magma", extend="max")
    plt.colorbar(c1, ax=axes[1], label=r"Absolute relative deviation [%]")
    axes[1].set_xlabel(r"$\eta_{\mathrm{turb}}$")
    axes[1].set_ylabel(r"$\eta_{\mathrm{compr}}$")
    axes[1].set_title(f"Relative Deviation\n{data1_label} - {data2_label}")

    fig.suptitle(
        rf"Deviation study"
        + (rf" | data1 PR={pr_data1:.5f}" if np.isfinite(pr_data1) else " | data1 PR=n/a")
        + "\n"
        + rf"mean|Δ|={mean_abs_delta:.4f}, max|Δ|={max_abs_delta:.4f}, RMSE={rmse_delta:.4f}, "
        + rf"mean|rel|={mean_abs_rel_pct:.2f}%, max|rel|={max_abs_rel_pct:.2f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    dev_fig_path = out_root / f"{comparison_slug}_deviation.pdf"
    fig.savefig(dev_fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved deviation figure: {dev_fig_path}")

    # Also save each dataset's contour separately next to the comparison figure
    data1_contour_pdf = out_root / f"{data1_path.stem}_contour.pdf"
    data1_contour_png = out_root / f"{data1_path.stem}_contour.png"
    _save_cop_heatmap(X, Y, Z_data1, data1_contour_pdf, title=f"{data1_label}", cop_key=COP_KEY)
    _save_cop_heatmap(X, Y, Z_data1, data1_contour_png, title=f"{data1_label}", cop_key=COP_KEY)

    data2_contour_pdf = out_root / f"{data2_path.stem}_contour.pdf"
    data2_contour_png = out_root / f"{data2_path.stem}_contour.png"
    _save_cop_heatmap(X, Y, Z_data2, data2_contour_pdf, title=f"{data2_label}", cop_key=COP_KEY)
    _save_cop_heatmap(X, Y, Z_data2, data2_contour_png, title=f"{data2_label}", cop_key=COP_KEY)

    logger.info(f"Saved individual contour plots: {data1_contour_pdf}, {data2_contour_pdf}")

    # Save absolute and relative deviation plots separately
    abs_dev_pdf = out_root / f"{comparison_slug}_absolute_deviation.pdf"
    abs_dev_png = out_root / f"{comparison_slug}_absolute_deviation.png"
    _save_cop_heatmap(X, Y, delta, abs_dev_pdf, title=f"Absolute Deviation\n{data1_label} - {data2_label}", cop_key=rf"$\Delta$ {COP_KEY}")
    _save_cop_heatmap(X, Y, delta, abs_dev_png, title=f"Absolute Deviation\n{data1_label} - {data2_label}", cop_key=rf"$\Delta$ {COP_KEY}")

    rel_dev_pdf = out_root / f"{comparison_slug}_relative_deviation.pdf"
    rel_dev_png = out_root / f"{comparison_slug}_relative_deviation.png"
    _save_cop_heatmap(X, Y, abs_rel_pct, rel_dev_pdf, title=f"Relative Deviation\n{data1_label} - {data2_label}", cop_key="Absolute relative deviation [%]")
    _save_cop_heatmap(X, Y, abs_rel_pct, rel_dev_png, title=f"Relative Deviation\n{data1_label} - {data2_label}", cop_key="Absolute relative deviation [%]")

    logger.info(f"Saved separate deviation plots: {abs_dev_pdf}, {rel_dev_pdf}")

    summary = {
        "data1_file": str(data1_path),
        "data2_file": str(data2_path),
        "data1_PR": pr_data1,
        "valid_points": int(np.sum(valid)),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "rmse_delta": rmse_delta,
        "mean_abs_rel_pct": mean_abs_rel_pct,
        "max_abs_rel_pct": max_abs_rel_pct,
        "eta_turb_at_max_abs": eta_turb_at_max_abs,
        "eta_compr_at_max_abs": eta_compr_at_max_abs,
        "deviation_data_file": str(dev_data_path),
        "deviation_figure_file": str(dev_fig_path),
        "absolute_deviation_file": str(abs_dev_pdf),
        "absolute_deviation_png": str(abs_dev_png),
        "relative_deviation_file": str(rel_dev_pdf),
        "relative_deviation_png": str(rel_dev_png),
    }

    logger.info(
        "Summary: "
        + f"mean|Δ|={mean_abs_delta:.5f}, max|Δ|={max_abs_delta:.5f}, RMSE={rmse_delta:.5f}, "
        + f"mean|rel|={mean_abs_rel_pct:.3f}%, max|rel|={max_abs_rel_pct:.3f}%"
    )
    return summary


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare two saved COP sweep files from a local data folder and save the deviation plots next to this script."
    )
    parser.add_argument(
        "--data1-file",
        default=None,
        help="File name of the first NPZ file (optional; auto-discovered if not provided).",
    )
    parser.add_argument(
        "--data2-file",
        default=None,
        help="File name of the second NPZ file (optional; auto-discovered if not provided).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing the NPZ files to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the deviation plots and summary file will be saved.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_deviation_study_from_existing(
        data1_file=args.data1_file,
        data2_file=args.data2_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    logger.info(
        f"Done: valid_points={result['valid_points']}, "
        f"mean|Δ|={result['mean_abs_delta']:.5f}"
    )
