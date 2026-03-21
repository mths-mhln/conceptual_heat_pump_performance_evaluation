from thermodynamics import solve_cycle, compute_performance, build_ts_data, build_ph_data
from visualization import make_thdy_plot, make_COP_vs_eff_plot
from logger import setup_logger
import warnings

from rich.console import Console
from rich.table import Table
from rich.progress import track
import numpy as np
from config import cycle_config, general_config

logger = setup_logger()



def main():
    analysis_type = general_config["analysis_type"]
    if general_config.get("ignore_coolprop_warnings", False):
        warnings.filterwarnings("ignore", module=r"CoolProp(\\.|$)")
        warnings.filterwarnings("ignore", message=r".*CoolProp.*")

    if analysis_type == "single_configuration":
        logger.info("Evaluating conceptual heat pump cycle")
        state = solve_cycle(cycle_config)
        print(state)
        perf = compute_performance(state, cycle_config)
        table = Table()
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_column("Unit", justify="center")
        table.add_column("Description", style="dim")
        rows = [
            ("COP_is",      f"{perf['COP_is']:.2f}",        "—",  "Isentropic coefficient of performance"),
            ("COP_turb",    f"{perf['COP_turb']:.2f}",       "—",  "Turbine-based coefficient of performance"),
            ("COP_isenth",  f"{perf['COP_isenth']:.2f}",     "—",  "Isenthalpic coefficient of performance"),
            ("Ẇ_turb",      f"{perf['Ẇ_turb']:.0f}",         "W",  "Turbine power output"),
            ("Ẇ_comp",      f"{perf['Ẇ_comp']:.0f}",         "W",  "Compressor power input"),
            ("Q_in",        f"{perf['Q_in']:.0f}",           "W",  "Heat absorbed by the cycle (actual)"),
            ("Q_in_isenth", f"{perf['Q_in_isenth']:.0f}",    "W",  "Heat absorbed under isenthalpic process"),
            ("Q_out",       f"{perf['Q_out']:.0f}",          "W",  "Heat rejected by the cycle"),
            ("PR",          f"{perf['PR']:.2f}",             "—",  "Pressure ratio across the compressor"),
        ]
        for symbol, value, unit, description in rows:
            table.add_row(symbol, value, unit, description, style="bright_green")
        Console().print(table)
        logger.info("Rendering T-S diagram")
        make_thdy_plot(state, perf, diagram_type="TS", ts_data=build_ts_data(state, cycle_config, general_config))
        logger.info("Rendering P-H diagram")
        make_thdy_plot(state, perf, diagram_type="PH", ph_data=build_ph_data(state, cycle_config))
        logger.info("Evaluation Completed")

    if analysis_type == "COP_vs_eff_investigation":
        cop_sweep_key = "COP_turb"
        if general_config["resolution"] == "high":
            n = 200
        else:
            n = 20
        η_turb_arr = np.linspace(0.01, 1.0, n)
        η_compr_arr = np.linspace(0.01, 1.0, n)
        # Regular grid for contourf + surface
        X, Y = np.meshgrid(η_turb_arr, η_compr_arr)   # X = η_turb, Y = η_compr
        Z = np.full_like(X, np.nan)                   # COP_turb grid
        total_runs = n * n
        logger.info(f"Running {cop_sweep_key} vs efficiency sweep ({n}×{n} = {total_runs:,} configurations)")
        for k in track(range(total_runs), description="\033[92mINFO\t"):
            i, j = divmod(k, n)
            cycle_config["η_turb"]  = X[i, j]
            cycle_config["η_compr"] = Y[i, j]
            try:
                state = solve_cycle(cycle_config)
                perf  = compute_performance(state, cycle_config)
                Z[i, j] = perf[cop_sweep_key]          # realistic COP with turbine work recovery
            except Exception as e:
                if not general_config.get("ignore_coolprop_warnings", False):
                    logger.warning(f"Failed at η_turb={X[i,j]:.3f}, η_compr={Y[i,j]:.3f}: {e}")
                Z[i, j] = np.nan
        logger.info("Sweep completed. Generating heatmap + 3D surface")
        make_COP_vs_eff_plot(X, Y, Z)  
        logger.info(f"{cop_sweep_key} vs efficiency investigation completed")
    else:
        logger.error(f"Invalid analysis type: {analysis_type}. Please set to 'single_configuration' or 'COP_vs_eff_investigation'.")

if __name__ == "__main__":
    main()

    