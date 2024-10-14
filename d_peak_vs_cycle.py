import data_analysis
import matplotlib.pyplot as plt
import read_data

if __name__ == "__main__":
    # Load data
    filepath_spectral = "data/Blue laser/10 cycles/operando_blue_laser_15x10_25p.txt"
    filepath_cycles = "data/Blue laser/10 cycles/group_b_operando_2_blue_15s_x10_25p_10_cycles_C01.txt"
    spectral_data = read_data.read_data_from_file(filepath_spectral)
    cycle_data = read_data.read_cycle_data(filepath_cycles)
    # Remove background noise from spectral data
    spectral_data = data_analysis.remove_background_noise(spectral_data)

    # Get defect density
    dd = data_analysis.defect_density(spectral_data)

    # Plot defect density against cycles
    fig, ax_spec = plt.subplots()
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel(r"Defect density $I_D/I_G$", color="tab:red")
    ax_cycle = ax_spec.twinx()
    ax_cycle.set_ylabel("Electric potential (V)", color="tab:blue")
    last_common_ts_idx = data_analysis.find_nearest_index(dd.index, cycle_data["time/s"].iloc[-1])
    ts_common_spec = dd.index[:last_common_ts_idx]

    ax_spec.tick_params(axis="y", color="tab:red")
    ax_cycle.tick_params(axis="y", color="tab:blue")
    ax_spec.plot(ts_common_spec, dd.loc[ts_common_spec], color="tab:red")
    ax_cycle.plot(cycle_data["time/s"], cycle_data["Ewe/V"], color="tab:blue")
    ax_spec.set_title("Defect density vs eletric potential")
    plt.show()

