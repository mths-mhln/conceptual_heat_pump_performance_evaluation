import sys
sys.path.append('d:/nexus/02_learning/00_university_education/04_MSc_TUDelft/05_thesis_nexus/07_conceptual_heat_pump_performance_evaluation/')

from thermodynamics import solve_cycle, compute_performance, build_ts_data, build_ph_data
from visualization import make_thdy_plot
from logger import setup_logger
import numpy as np
import warnings

from rich.console import Console
from rich.table import Table
from verification_config import cycle_config_R1233zd_E, cycle_config_n_pentane, cycle_config_MM, general_config, MTW_cycle_performances, MTW_cycle

logger = setup_logger()
console = Console()


def _append_verification_rows(table, refrigerant, perf, mtw_perf):
    max_discrepancy = 0.0
    for metric, mtw_value in mtw_perf.items():
        model_value = perf[metric]
        if mtw_value == 0:
            rel_error = float("nan")
        else:
            rel_error = abs(model_value - mtw_value) / abs(mtw_value)
        max_discrepancy = max(max_discrepancy, 0.0 if np.isnan(rel_error) else rel_error)

        table.add_row(
            refrigerant,
            metric,
            f"{model_value:.4g}",
            f"{mtw_value:.4g}",
            "n/a" if np.isnan(rel_error) else f"{rel_error:.2%}",
        )
    return max_discrepancy



def verification(verification_table = False, generate_thdy_diagrams=False, threshold=0.05):
    logger.info("Code verification against Martin T. White's HP calculation tool")
    analysis_type = general_config["analysis_type"]
    if general_config.get("ignore_coolprop_warnings", False):
        warnings.filterwarnings("ignore", module=r"CoolProp(\\.|$)")
        warnings.filterwarnings("ignore", message=r".*CoolProp.*")

    if analysis_type == "single_configuration":
        table = Table(title="Verification Results", show_lines=False)
        table.add_column("Refrigerant", style="magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Model", justify="right")
        table.add_column("Martin T. White", justify="right")
        table.add_column("Rel. Error", justify="right")

        overall_max_discrepancy = 0.0
        for cycle_config in [cycle_config_R1233zd_E, cycle_config_n_pentane, cycle_config_MM]:  
            logger.info(f"Evaluating conceptual heat pump cycle for refrigerant: {cycle_config['refrigerant']}")
            state = solve_cycle(cycle_config, general_config, verbose = False)
            perf = compute_performance(state, cycle_config, general_config)
            max_discrepancy = _append_verification_rows(
                table=table,
                refrigerant=cycle_config["refrigerant"],
                perf=perf,
                mtw_perf=MTW_cycle_performances[cycle_config["refrigerant"]],
            )
            overall_max_discrepancy = max(overall_max_discrepancy, max_discrepancy)
            
            verification_cycle_data = MTW_cycle[cycle_config["refrigerant"]]
            
            if generate_thdy_diagrams:
                logger.info("Rendering T-S diagram")
                make_thdy_plot(
                    state,
                    perf,
                    diagram_type="TS",
                    cycle_config=cycle_config,
                    output_dir="verification/thermodynamic_diagrams",
                    ts_data=build_ts_data(state, cycle_config, general_config),
                    verification_data=verification_cycle_data,
                    verbose=True,
                )
                
                logger.info("Rendering P-H diagram")
                make_thdy_plot(
                    state,
                    perf,
                    diagram_type="PH",
                    cycle_config=cycle_config,
                    output_dir="verification/thermodynamic_diagrams",
                    ph_data=build_ph_data(state, cycle_config),
                    verification_data=verification_cycle_data,
                    verbose=True,
                )

        table.add_section()
        table.add_row(
            "All",
            "Maximum discrepancy",
            "-",
            "-",
            f"{overall_max_discrepancy:.2%}",
        )
        if verification_table:
            console.print(table)
        if overall_max_discrepancy < threshold:
            logger.info(f"Verification successful: Overall max. discrepancy of {overall_max_discrepancy:.2%} is within the threshold of {threshold:.2%}.\n")
        else:
            logger.warning(f"Verification failed: Overall max. discrepancy of {overall_max_discrepancy:.2%} exceeds the threshold of {threshold:.2%}.\n")
    else:
        logger.error(f"Invalid analysis type for verification: {analysis_type}. Please set to 'single_configuration'.")

if __name__ == "__main__":
    verification()