# main.py
# =======
# Entry point. Solves the cycle, computes performance, and renders both diagrams.

from thermodynamics import solve_cycle, compute_performance, build_ts_data, build_ph_data
from visualization import make_plot


def main():
    print("Solving cycle...")
    state = solve_cycle()

    print("Computing performance...")
    perf = compute_performance(state)
    print(f"  COP_is     = {perf['COP_is']:.2f}")
    print(f"  COP_turb   = {perf['COP_turb']:.2f}")
    print(f"  COP_isenth = {perf['COP_isenth']:.2f}")
    print(f"  W_turb     = {perf['Ẇ_turb']:.0f} W")
    print(f"  W_comp     = {perf['Ẇ_comp']:.0f} W")
    print(f"  Q_in       = {perf['Q_in']:.0f} W")
    print(f"  Q_out      = {perf['Q_out']:.0f} W")

    print("\nRendering T-S diagram...")
    make_plot(state, perf, diagram_type="TS", ts_data=build_ts_data(state))

    print("Rendering P-H diagram...")
    make_plot(state, perf, diagram_type="PH", ph_data=build_ph_data(state))

    print("\nDone.")


if __name__ == "__main__":
    main()