import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/')
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/verification/')

from thermodynamics import solve_cycle, compute_performance, build_ts_data, build_ph_data
from visualization import make_thdy_plot, make_COP_vs_eff_plot, make_empty_thdy_plot
from config import cycle_config, general_config
from verification import verification
from logger import setup_logger

from rich.console import Console
from rich.table import Table
from rich.progress import track

import numpy as np
import warnings
import timeit
import os

logger = setup_logger()
start = timeit.default_timer()



def main(perform_verification=True):
    # Perform code verification:
    if perform_verification:
        verification(verification_table=True, generate_thdy_diagrams=False)

    analysis_type = general_config["analysis_type"]
    if general_config.get("ignore_coolprop_warnings", False):
        warnings.filterwarnings("ignore", module=r"CoolProp(\\.|$)")
        warnings.filterwarnings("ignore", message=r".*CoolProp.*")

    if analysis_type == "single_configuration":
        logger.info("Evaluating conceptual heat pump cycle")
        cycle_data = solve_cycle(cycle_config, general_config, verbose=True)
        perf = compute_performance(cycle_data, cycle_config, general_config)
        table = Table(title=f"Conceptual Heat Pump Cycle - {cycle_config['refrigerant']}", show_lines=False)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_column("Unit", justify="center")
        table.add_column("Description", style="dim")
        rows = [
            ("COP_is",      f"{perf['COP_is']:.2f}",        "—",  "Isentropic coefficient of performance"),
            ("COP_turb",    f"{perf['COP_turb']:.2f}",       "—",  "Turbine-based coefficient of performance"),
            ("COP_isenth",  f"{perf['COP_isenth']:.2f}",     "—",  "Isenthalpic coefficient of performance"),
            ("PR",          f"{perf['PR']:.2f}",             "—",  "Pressure ratio across the compressor"),
            ("Ẇ_turb",      f"{perf['Ẇ_turb']:.0f}",         "W",  "Turbine power output"),
            ("Ẇ_comp",      f"{perf['Ẇ_comp']:.0f}",         "W",  "Compressor power input"),
            ("Q_in",        f"{perf['Q_in']:.0f}",           "W",  "Heat absorbed by the cycle (actual)"),
            ("Q_in_isenth", f"{perf['Q_in_isenth']:.0f}",    "W",  "Heat absorbed under isenthalpic process"),
            ("Q_out",       f"{perf['Q_out']:.0f}",          "W",  "Heat rejected by the cycle")
        ]
        for symbol, value, unit, description in rows:
            table.add_row(symbol, value, unit, description)
        Console().print(table)
        logger.info("Rendering T-S diagram")
        make_thdy_plot(
            cycle_data,
            perf,
            diagram_type="TS",
            cycle_config=cycle_config,
            output_dir="heat_pump_thermodynamic_diagrams",
            ts_data=build_ts_data(cycle_data, cycle_config, general_config),
        )
        logger.info("Rendering P-H diagram")
        make_thdy_plot(
            cycle_data,
            perf,
            diagram_type="PH",
            cycle_config=cycle_config,
            output_dir="heat_pump_thermodynamic_diagrams",
            ph_data=build_ph_data(cycle_data, cycle_config),
        )
        logger.info("Evaluation Completed")

    elif analysis_type == "COP_vs_eff_investigation":
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
        for k in track(range(total_runs), description="\033[92mINFO    "):
            i, j = divmod(k, n)
            cycle_config["η_turb"]  = X[i, j]
            cycle_config["η_compr"] = Y[i, j]
            try:
                cycle_data = solve_cycle(cycle_config, general_config, verbose=False)
                perf  = compute_performance(cycle_data, cycle_config, general_config)
                Z[i, j] = perf[cop_sweep_key]          # realistic COP with turbine work recovery
            except Exception as e:
                if not general_config.get("ignore_coolprop_warnings", False):
                    logger.warning(f"Failed at η_turb={X[i,j]:.3f}, η_compr={Y[i,j]:.3f}: {e}")
                Z[i, j] = np.nan
        logger.info("Sweep completed. Saving data to file")
        # Create output directory if it doesn't exist
        output_dir = "COP_investigations"
        os.makedirs(output_dir, exist_ok=True)
        # Save the sweep data for recovery if plotting fails
        refrigerant = cycle_config["refrigerant"]
        data_file = os.path.join(output_dir, f"COP_vs_eff_{refrigerant}_{cop_sweep_key}.npz")
        np.savez_compressed(data_file, X=X, Y=Y, Z=Z)
        logger.info(f"Data saved to {data_file}")
        logger.info("Generating heatmap + 3D surface")
        make_COP_vs_eff_plot(X, Y, Z, cycle_config)  
        logger.info(f"{cop_sweep_key} vs efficiency investigation completed")
        
    elif analysis_type == "substance_thermodynamic_diagrams":
        substances = general_config.get("substances_to_plot", [cycle_config["refrigerant"]])
        logger.info(f"Generating empty thermodynamic diagrams for {len(substances)} substance(s)")
        for substance in substances:
            cycle_config_substance = dict(cycle_config)
            cycle_config_substance["refrigerant"] = substance
            logger.info(f"Rendering empty TS/PH diagrams for {substance}")
            make_empty_thdy_plot(
                diagram_type="TS",
                cycle_config=cycle_config_substance,
                output_dir="substance_thermodynamic_diagrams",
            )
            make_empty_thdy_plot(
                diagram_type="PH",
                cycle_config=cycle_config_substance,
                output_dir="substance_thermodynamic_diagrams",
            )
        logger.info("Substance thermodynamic diagram generation completed")
    else:
        logger.error(
            f"Invalid analysis type: {analysis_type}. Please set to 'single_configuration', "
            "'COP_vs_eff_investigation', or 'substance_thermodynamic_diagrams'."
        )

if __name__ == "__main__":
    main(perform_verification=True)
    end = timeit.default_timer()
    elapsed = end - start
    logger.info(f"Total execution time: {elapsed:.2f} seconds")