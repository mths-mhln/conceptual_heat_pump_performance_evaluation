from pathlib import Path

import numpy as np
from rich.progress import track

from thermodynamics import solve_cycle, compute_performance
from visualization import make_COP_vs_eff_plot


def _build_efficiency_grid(n):
    eta_turb_arr = np.linspace(0.01, 1.0, n)
    eta_compr_arr = np.linspace(0.01, 1.0, n)
    X, Y = np.meshgrid(eta_turb_arr, eta_compr_arr)
    return eta_turb_arr, eta_compr_arr, X, Y


def _cop_data_paths(refrigerant, cop_sweep_key):
    data_dir = Path("COP_investigations") / refrigerant / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / f"COP_vs_eff_{refrigerant}_{cop_sweep_key}.npz"
    return data_dir, data_file


def _save_sweep_data(data_file, X, Y, Z, attempted, cop_sweep_key, completed_runs, total_runs, status):
    np.savez_compressed(
        data_file,
        X=X,
        Y=Y,
        Z=Z,
        attempted=attempted,
        cop_sweep_key=cop_sweep_key,
        completed_runs=int(completed_runs),
        total_runs=int(total_runs),
        status=status,
    )


def _grid_matches(existing_x, existing_y, current_x, current_y):
    if existing_x.shape != current_x.shape or existing_y.shape != current_y.shape:
        return False
    return np.allclose(existing_x, current_x, equal_nan=True) and np.allclose(existing_y, current_y, equal_nan=True)


def _prompt_choice(prompt, valid_choices, default, logger):
    valid_choices = {choice.lower() for choice in valid_choices}
    default_choice = default.lower()
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            logger.warning(f"No interactive input available. Falling back to default '{default_choice}'.")
            return default_choice
        if answer == "":
            return default_choice
        if answer in valid_choices:
            return answer
        logger.warning(f"Invalid input '{answer}'. Valid options: {sorted(valid_choices)}")


def _initialize_or_resume_sweep(data_file, X, Y, cop_sweep_key, logger, general_config):
    total_runs = X.size
    load_file = data_file
    legacy_file = data_file.parent.parent / data_file.name
    if not load_file.exists() and legacy_file.exists():
        logger.info(f"Found legacy sweep data at {legacy_file}. It will be reused and written to {data_file}.")
        load_file = legacy_file

    if not load_file.exists():
        Z = np.full_like(X, np.nan, dtype=float)
        attempted = np.zeros_like(X, dtype=bool)
        return Z, attempted, True

    logger.info(f"Existing sweep data found at {load_file}")
    with np.load(load_file, allow_pickle=False) as existing:
        if not all(key in existing for key in ("X", "Y", "Z")):
            logger.warning("Existing data file is missing required keys. Starting from scratch.")
            Z = np.full_like(X, np.nan, dtype=float)
            attempted = np.zeros_like(X, dtype=bool)
            return Z, attempted, True

        old_x = existing["X"]
        old_y = existing["Y"]
        old_z = existing["Z"]
        old_attempted = existing["attempted"] if "attempted" in existing else np.isfinite(old_z)

    if not _grid_matches(old_x, old_y, X, Y):
        choice = _prompt_choice(
            "Existing data uses a different efficiency grid. Restart from scratch and overwrite? [Y/n]: ",
            valid_choices={"y", "yes", "n", "no"},
            default="y",
            logger=logger,
        )
        if choice in {"y", "yes"}:
            Z = np.full_like(X, np.nan, dtype=float)
            attempted = np.zeros_like(X, dtype=bool)
            return Z, attempted, True
        logger.info("Keeping existing mismatched file unchanged. Skipping this analysis run.")
        return None, None, False

    Z = old_z.astype(float, copy=True)
    attempted = old_attempted.astype(bool, copy=True)
    pending = int(np.sum(~attempted))

    if pending > 0:
        choice = _prompt_choice(
            f"Found partial progress ({total_runs - pending}/{total_runs}). Continue or restart? [c/r]: ",
            valid_choices={"c", "continue", "r", "restart"},
            default="c",
            logger=logger,
        )
        if choice in {"r", "restart"}:
            Z = np.full_like(X, np.nan, dtype=float)
            attempted = np.zeros_like(X, dtype=bool)
        return Z, attempted, True

    complete_action = str(general_config.get("existing_complete_sweep_action", "prompt")).strip().lower()
    if complete_action not in {"prompt", "abort", "recompute"}:
        logger.warning(
            f"Invalid existing_complete_sweep_action='{complete_action}'. Falling back to 'prompt'."
        )
        complete_action = "prompt"

    if complete_action == "prompt":
        choice = _prompt_choice(
            f"Sweep already complete ({total_runs}/{total_runs}). Abort or recompute? [a/r]: ",
            valid_choices={"a", "abort", "r", "recompute"},
            default="a",
            logger=logger,
        )
    elif complete_action == "abort":
        logger.info("Auto-choice for complete sweep: abort.")
        choice = "abort"
    else:
        logger.info("Auto-choice for complete sweep: recompute.")
        choice = "recompute"

    if choice in {"a", "abort"}:
        logger.info("Sweep run aborted by user choice; existing data left unchanged.")
        return None, None, False

    Z = np.full_like(X, np.nan, dtype=float)
    attempted = np.zeros_like(X, dtype=bool)
    return Z, attempted, True


def run_cop_vs_eff_investigation(cycle_config, general_config, logger):
    cop_sweep_key = "COP_turb"
    if general_config["resolution"] == "high":
        n = 200
    else:
        n = 20

    _, _, X, Y = _build_efficiency_grid(n)
    refrigerant = cycle_config["refrigerant"]
    _, data_file = _cop_data_paths(refrigerant, cop_sweep_key)
    init_result = _initialize_or_resume_sweep(data_file, X, Y, cop_sweep_key, logger, general_config)
    if init_result[0] is None:
        return

    Z, attempted, should_run = init_result
    if not should_run:
        return

    total_runs = n * n
    completed_runs = int(np.sum(attempted))
    pending_indices = np.flatnonzero(~attempted.ravel())
    logger.info(
        f"Running {cop_sweep_key} vs efficiency sweep ({n}x{n} = {total_runs:,} configurations). "
        f"Already completed: {completed_runs:,}; remaining: {pending_indices.size:,}"
    )

    checkpoint_every = 5
    processed_since_save = 0
    sweep_interrupted = False

    try:
        for flat_idx in track(pending_indices, description="\033[92mINFO    "):
            i, j = divmod(int(flat_idx), n)
            cycle_config["η_turb"] = X[i, j]
            cycle_config["η_compr"] = Y[i, j]
            try:
                cycle_data = solve_cycle(cycle_config, general_config, verbose=False)
                perf = compute_performance(cycle_data, cycle_config, general_config)
                Z[i, j] = perf[cop_sweep_key]
            except Exception as e:
                if not general_config.get("ignore_coolprop_warnings", False):
                    logger.warning(f"Failed at η_turb={X[i,j]:.3f}, η_compr={Y[i,j]:.3f}: {e}")
                Z[i, j] = np.nan
            attempted[i, j] = True
            processed_since_save += 1
            completed_runs += 1

            if processed_since_save >= checkpoint_every:
                _save_sweep_data(
                    data_file,
                    X,
                    Y,
                    Z,
                    attempted,
                    cop_sweep_key,
                    completed_runs,
                    total_runs,
                    status="in_progress",
                )
                processed_since_save = 0
    except KeyboardInterrupt:
        sweep_interrupted = True
        logger.warning("Sweep interrupted by user. Saving checkpoint before exiting.")

    _save_sweep_data(
        data_file,
        X,
        Y,
        Z,
        attempted,
        cop_sweep_key,
        completed_runs,
        total_runs,
        status="interrupted" if sweep_interrupted else "complete",
    )
    logger.info(f"Data saved to {data_file}")

    if sweep_interrupted:
        logger.info("You can rerun and choose 'continue' to resume from this checkpoint.")
        return

    if completed_runs < total_runs:
        logger.info("Sweep not fully complete. You can rerun and choose 'continue' to finish remaining points.")
        return

    logger.info("Generating heatmap + 3D surface")
    make_COP_vs_eff_plot(
        X,
        Y,
        Z,
        cycle_config,
        output_dir="COP_investigations",
    )
    logger.info(f"{cop_sweep_key} vs efficiency investigation completed")
