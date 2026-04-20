from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

from config import cycle_config, general_config
from logger import setup_logger
from thermodynamics import compute_performance, solve_cycle


logger = setup_logger()


TARGET_REFRIGERANT = "R1234ze(E)"
COP_KEY = "COP_turb"
OUTPUT_DIR = Path("COP_investigations")


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


def _resolve_existing_files(refrigerant, cop_key, output_dir):
    root = Path(output_dir) / refrigerant
    data_root = root / "data"
    fixed_path = data_root / f"COP_vs_eff_{refrigerant}_{cop_key}_fixedPR.npz"
    fixed_legacy = root / f"COP_vs_eff_{refrigerant}_{cop_key}_fixedPR.npz"

    reopt_candidates = [
        data_root / f"COP_vs_eff_{refrigerant}_{cop_key}_reoptimized.npz",
        data_root / f"COP_vs_eff_{refrigerant}_{cop_key}.npz",  # original implementation filename
        root / f"COP_vs_eff_{refrigerant}_{cop_key}_reoptimized.npz",
        root / f"COP_vs_eff_{refrigerant}_{cop_key}.npz",  # legacy location
    ]
    reopt_path = next((p for p in reopt_candidates if p.exists()), None)

    missing = []
    if not fixed_path.exists() and not fixed_legacy.exists():
        missing.append(str(fixed_path))
    if reopt_path is None:
        missing.append("one of: " + ", ".join(str(p) for p in reopt_candidates))

    if fixed_path.exists():
        fixed_existing = fixed_path
    else:
        fixed_existing = fixed_legacy

    return fixed_existing, reopt_path, missing


def _compute_fixed_pr_file(refrigerant, cop_key, output_dir):
    """Compute and save the fixed-PR efficiency sweep for a refrigerant."""
    cfg_base = dict(cycle_config)
    cfg_base["refrigerant"] = refrigerant
    gen_cfg = dict(general_config)

    # One optimization call to get the reference PR.
    reference_cycle = solve_cycle(dict(cfg_base), gen_cfg, verbose=False)
    reference_pr = float(reference_cycle["PR"])

    (X, Y), n = _build_efficiency_grid(gen_cfg)
    Z = np.full_like(X, np.nan, dtype=float)
    total_runs = n * n

    logger.info(f"Computing fixed-PR sweep for {refrigerant} at PR={reference_pr:.6f}")
    for k in track(range(total_runs), description="\033[92mINFO    "):
        i, j = divmod(k, n)
        cfg = dict(cfg_base)
        cfg["PR"] = reference_pr
        cfg["η_turb"] = float(X[i, j])
        cfg["η_compr"] = float(Y[i, j])
        try:
            cycle_data = solve_cycle(cfg, gen_cfg, verbose=False)
            perf = compute_performance(cycle_data, cfg, gen_cfg)
            Z[i, j] = perf[cop_key]
        except Exception:
            Z[i, j] = np.nan

    root = Path(output_dir) / refrigerant / "data"
    root.mkdir(parents=True, exist_ok=True)
    fixed_path = root / f"COP_vs_eff_{refrigerant}_{cop_key}_fixedPR.npz"
    np.savez_compressed(fixed_path, X=X, Y=Y, Z=Z, PR=reference_pr)
    logger.info(f"Saved computed fixed-PR data: {fixed_path}")

    fig_path = root / f"{cop_key}_vs_Efficiencies_{refrigerant}_fixedPR.pdf"
    _save_cop_heatmap(
        X,
        Y,
        Z,
        fig_path,
        title=rf"{cop_key} at fixed PR={reference_pr:.4f} ({refrigerant})",
        cop_key=cop_key,
    )

    return fixed_path


def _compute_reoptimized_file(refrigerant, cop_key, output_dir):
    """Compute and save original implementation sweep (PR optimized at every point)."""
    cfg_base = dict(cycle_config)
    cfg_base["refrigerant"] = refrigerant
    gen_cfg = dict(general_config)

    (X, Y), n = _build_efficiency_grid(gen_cfg)
    Z = np.full_like(X, np.nan, dtype=float)
    total_runs = n * n

    logger.info(f"Computing re-optimized sweep for {refrigerant}")
    for k in track(range(total_runs), description="\033[92mINFO    "):
        i, j = divmod(k, n)
        cfg = dict(cfg_base)
        cfg.pop("PR", None)
        cfg["η_turb"] = float(X[i, j])
        cfg["η_compr"] = float(Y[i, j])
        try:
            cycle_data = solve_cycle(cfg, gen_cfg, verbose=False)
            perf = compute_performance(cycle_data, cfg, gen_cfg)
            Z[i, j] = perf[cop_key]
        except Exception:
            Z[i, j] = np.nan

    root = Path(output_dir) / refrigerant / "data"
    root.mkdir(parents=True, exist_ok=True)
    reopt_path = root / f"COP_vs_eff_{refrigerant}_{cop_key}_reoptimized.npz"
    np.savez_compressed(reopt_path, X=X, Y=Y, Z=Z)
    logger.info(f"Saved computed re-optimized data: {reopt_path}")

    fig_path = root / f"{cop_key}_vs_Efficiencies_{refrigerant}_reoptimized.pdf"
    _save_cop_heatmap(
        X,
        Y,
        Z,
        fig_path,
        title=rf"{cop_key} with PR optimization per point ({refrigerant})",
        cop_key=cop_key,
    )

    return reopt_path


def _prompt_compute_missing(missing_items):
    logger.warning("Missing required files for extraction-only deviation study:")
    for item in missing_items:
        logger.warning(f"  - {item}")
    answer = input("Missing data files detected. Compute them now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def run_deviation_study_from_existing(
    refrigerant=TARGET_REFRIGERANT,
    cop_key=COP_KEY,
    output_dir=OUTPUT_DIR,
):
    """Compare fixed-PR and re-optimized COP sweeps using existing data, with optional user-confirmed fallback compute."""
    fixed_path, reopt_path, missing = _resolve_existing_files(refrigerant, cop_key, output_dir)

    if missing:
        if _prompt_compute_missing(missing):
            if not Path(fixed_path).exists():
                fixed_path = _compute_fixed_pr_file(refrigerant, cop_key, output_dir)
            if reopt_path is None:
                reopt_path = _compute_reoptimized_file(refrigerant, cop_key, output_dir)
        else:
            raise FileNotFoundError("Required files missing and computation was not approved by user.")

    logger.info(f"Loading fixed-PR data: {fixed_path}")
    logger.info(f"Loading re-optimized data: {reopt_path}")

    Xf, Yf, Z_fixed, pr_fixed = _load_xyz(fixed_path)
    Xr, Yr, Z_reopt, _ = _load_xyz(reopt_path)

    if Xf.shape != Xr.shape or Yf.shape != Yr.shape or Z_fixed.shape != Z_reopt.shape:
        raise ValueError(
            "Existing files have incompatible shapes: "
            f"fixed={Z_fixed.shape}, reoptimized={Z_reopt.shape}"
        )

    if not (np.allclose(Xf, Xr, equal_nan=True) and np.allclose(Yf, Yr, equal_nan=True)):
        raise ValueError("Efficiency grids in existing files do not match; cannot compute point-wise deviation.")

    X, Y = Xf, Yf
    valid = np.isfinite(Z_fixed) & np.isfinite(Z_reopt)

    delta = np.full_like(Z_fixed, np.nan, dtype=float)
    rel_pct = np.full_like(Z_fixed, np.nan, dtype=float)
    delta[valid] = Z_fixed[valid] - Z_reopt[valid]

    denom_ok = valid & (np.abs(Z_reopt) > 1e-12)
    rel_pct[denom_ok] = 100.0 * delta[denom_ok] / np.abs(Z_reopt[denom_ok])

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

    out_root = Path(output_dir) / refrigerant
    out_root.mkdir(parents=True, exist_ok=True)

    dev_data_path = out_root / f"COP_vs_eff_{refrigerant}_{cop_key}_deviation_fixedPR_vs_reoptimized.npz"
    np.savez_compressed(
        dev_data_path,
        X=X,
        Y=Y,
        Z_fixed=Z_fixed,
        Z_reoptimized=Z_reopt,
        delta=delta,
        rel_pct=rel_pct,
        PR_fixed=pr_fixed,
        fixed_source=str(fixed_path),
        reoptimized_source=str(reopt_path),
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
    plt.colorbar(c0, ax=axes[0], label=rf"$\Delta$ {cop_key} (fixed PR - re-optimized)")
    axes[0].set_xlabel(r"$\eta_{\mathrm{turb}}$")
    axes[0].set_ylabel(r"$\eta_{\mathrm{compr}}$")
    axes[0].set_title("Absolute Deviation")

    c1 = axes[1].contourf(X, Y, abs_rel_pct, levels=30, cmap="magma", extend="max")
    plt.colorbar(c1, ax=axes[1], label=r"Absolute relative deviation [%]")
    axes[1].set_xlabel(r"$\eta_{\mathrm{turb}}$")
    axes[1].set_ylabel(r"$\eta_{\mathrm{compr}}$")
    axes[1].set_title("Relative Deviation")

    fig.suptitle(
        rf"Deviation study ({refrigerant})"
        + (rf" | fixed PR={pr_fixed:.5f}" if np.isfinite(pr_fixed) else " | fixed PR=n/a")
        + "\n"
        + rf"mean|Δ|={mean_abs_delta:.4f}, max|Δ|={max_abs_delta:.4f}, RMSE={rmse_delta:.4f}, "
        + rf"mean|rel|={mean_abs_rel_pct:.2f}%, max|rel|={max_abs_rel_pct:.2f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    dev_fig_path = out_root / f"{cop_key}_deviation_fixedPR_vs_reoptimized_{refrigerant}.pdf"
    fig.savefig(dev_fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved deviation figure: {dev_fig_path}")

    summary = {
        "refrigerant": refrigerant,
        "fixed_PR": pr_fixed,
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
    }

    logger.info(
        "Summary: "
        + f"mean|Δ|={mean_abs_delta:.5f}, max|Δ|={max_abs_delta:.5f}, RMSE={rmse_delta:.5f}, "
        + f"mean|rel|={mean_abs_rel_pct:.3f}%, max|rel|={max_abs_rel_pct:.3f}%"
    )
    return summary


if __name__ == "__main__":
    result = run_deviation_study_from_existing(refrigerant=TARGET_REFRIGERANT)
    logger.info(
        f"Done: refrigerant={result['refrigerant']}, valid_points={result['valid_points']}, "
        f"mean|Δ|={result['mean_abs_delta']:.5f}"
    )
