import os
import sys
cwd = os.getcwd()
sys.path.append(f'{cwd}/verification/')

from thermodynamics import solve_cycle, compute_performance, build_ts_data, build_ph_data
from visualization import make_thdy_plot, make_empty_thdy_plot, make_optimizer_progress_plot
from cop_sweep import run_cop_vs_eff_investigation
from config import cycle_config, general_config
from verification import verification
from logger import setup_logger

from rich.console import Console
from rich.table import Table

import warnings
import timeit


logger = setup_logger()
start = timeit.default_timer()



def main(perform_verification=True):
    plot_saving_time = 0.0

    # Perform code verification:
    if perform_verification:
        verification(verification_table=True, generate_thdy_diagrams=False)

    analysis_type = general_config["analysis_type"]
    if general_config.get("ignore_coolprop_warnings", False):
        warnings.filterwarnings("ignore", module=r"CoolProp(\\.|$)")
        warnings.filterwarnings("ignore", message=r".*CoolProp.*")

    if analysis_type == "single_configuration":
        logger.info("Evaluating conceptual heat pump cycle")
        cycle_data = solve_cycle(cycle_config, general_config, verbose=False)
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
            ("ṁ_ref",       f"{perf['ṁ_ref']:.3f}",         "kg/s",  "Mass flow rate of the refrigerant"),
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
        t_plot = timeit.default_timer()
        make_thdy_plot(
            cycle_data,
            perf,
            diagram_type="TS",
            cycle_config=cycle_config,
            output_dir="heat_pump_thermodynamic_diagrams",
            ts_data=build_ts_data(cycle_data, cycle_config, general_config),
        )
        plot_saving_time += timeit.default_timer() - t_plot
        logger.info("Rendering P-H diagram")
        t_plot = timeit.default_timer()
        make_thdy_plot(
            cycle_data,
            perf,
            diagram_type="PH",
            cycle_config=cycle_config,
            output_dir="heat_pump_thermodynamic_diagrams",
            ph_data=build_ph_data(cycle_data, cycle_config),
        )
        plot_saving_time += timeit.default_timer() - t_plot
        logger.info("Rendering optimizer progression plot")
        t_plot = timeit.default_timer()
        make_optimizer_progress_plot(
            cycle_data.get("optimization_trace"),
            cycle_config,
            output_dir="heat_pump_thermodynamic_diagrams",
        )
        plot_saving_time += timeit.default_timer() - t_plot
        logger.info("Evaluation Completed")

    elif analysis_type == "COP_vs_eff_investigation":
        run_cop_vs_eff_investigation(cycle_config, general_config, logger)
        
    elif analysis_type == "substance_thermodynamic_diagrams":
        substances = general_config.get("substances_to_plot", [cycle_config["refrigerant"]])
        logger.info(f"Generating empty thermodynamic diagrams for {len(substances)} substance(s)")
        for substance in substances:
            cycle_config_substance = dict(cycle_config)
            cycle_config_substance["refrigerant"] = substance
            logger.info(f"Rendering empty TS/PH diagrams for {substance}")
            t_plot = timeit.default_timer()
            make_empty_thdy_plot(
                diagram_type="TS",
                cycle_config=cycle_config_substance,
                output_dir="substance_thermodynamic_diagrams",
            )
            plot_saving_time += timeit.default_timer() - t_plot
            t_plot = timeit.default_timer()
            make_empty_thdy_plot(
                diagram_type="PH",
                cycle_config=cycle_config_substance,
                output_dir="substance_thermodynamic_diagrams",
            )
            plot_saving_time += timeit.default_timer() - t_plot
        logger.info("Substance thermodynamic diagram generation completed")
    else:
        logger.error(
            f"Invalid analysis type: {analysis_type}. Please set to 'single_configuration', "
            "'COP_vs_eff_investigation', or 'substance_thermodynamic_diagrams'."
        )

    return plot_saving_time

if __name__ == "__main__":
    plot_saving_time = main(perform_verification=False)
    end = timeit.default_timer()
    elapsed = end - start
    eval_time = max(0.0, elapsed - plot_saving_time)
    logger.info(f"Evaluation time (excl. plot rendering): {eval_time:.2f} seconds")