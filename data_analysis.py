import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from background_red import backcor

def find_nearest_index(array, value):
    # Helper function
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

def gaussian_fit(data: pd.DataFrame, wavenumber_range=np.array([1250., 1400.])) -> pd.DataFrame:
    def gaussian(x, a, mu, sigma, b):
        return a*np.exp(-1*(x - mu)**2/(2*sigma**2)) + b
    wavenumbers = np.array(data.columns)
    timestamps = np.array(data.index)
    low_idx = find_nearest_index(wavenumbers, wavenumber_range[0])
    high_idx = find_nearest_index(wavenumbers, wavenumber_range[1])
    peak_wn = wavenumbers[low_idx:high_idx]
    # Extract datapoints inside wavenumber range
    datapoints = data[peak_wn]
    fitted_data = np.zeros(shape=(len(timestamps), high_idx - low_idx))
    # Fit data for all timesteps
    for i, timestamp in enumerate(timestamps):
        intensities = np.array(datapoints.loc[timestamp])
        min_val = np.min(intensities)
        max_val = np.max(intensities)
        params, cov = curve_fit(gaussian, peak_wn, intensities, p0=[max_val / 2, (peak_wn[-1] - peak_wn[0])/2 + peak_wn[0], 5., min_val / 2])
        fitted_data[i, :] = gaussian(peak_wn, params[0], params[1], params[2], params[3])
    # Create new dataframe and return it
    return pd.DataFrame(index=timestamps, columns=wavenumbers[low_idx:high_idx], data=fitted_data)

def defect_density(data: pd.DataFrame, d_peak_range=np.array([1250., 1400.]), g_peak_range=np.array([1550., 1650.])) -> pd.DataFrame:
    wavenumbers = np.array(data.columns)
    timestamps = np.array(data.index)
    d_peak_low_idx = find_nearest_index(wavenumbers, d_peak_range[0])
    d_peak_high_idx = find_nearest_index(wavenumbers, d_peak_range[1])
    g_peak_low_idx = find_nearest_index(wavenumbers, g_peak_range[0])
    g_peak_high_idx = find_nearest_index(wavenumbers, g_peak_range[1])
    d_peak_wn = wavenumbers[d_peak_low_idx:d_peak_high_idx]
    g_peak_wn = wavenumbers[g_peak_low_idx:g_peak_high_idx]

    defect_densities = np.zeros(shape=(len(timestamps), 1))

    for i, timestamp in enumerate(timestamps):
        d_peak_intensities = np.array(data[d_peak_wn].loc[timestamp])
        g_peak_intensities = np.array(data[g_peak_wn].loc[timestamp])

        defect_densities[i] = np.max(d_peak_intensities) / np.max(g_peak_intensities)

    return pd.DataFrame(data=defect_densities, index=timestamps)

def remove_background_noise(data: pd.DataFrame) -> pd.DataFrame:
    wavenumbers = np.array(data.columns)
    timestamps = np.array(data.index)

    data_out = np.zeros(shape=(len(timestamps), len(wavenumbers)))
    for i, timestamp in enumerate(timestamps):
        intensities = np.array(data[wavenumbers].loc[timestamp])
        filtered_data, pol_coeff, it = backcor(wavenumbers, intensities, 3, 0.001, "ah")
        data_out[i, :] = data[wavenumbers].loc[timestamp] - filtered_data
        data_out[i, :] -= np.min(data_out[i, :])

    return pd.DataFrame(data=data_out, index=timestamps, columns=wavenumbers)



# For testing
if __name__ == "__main__":
    # Get data
    from read_data import read_data_from_file
    loaded_data = read_data_from_file("data/Blue laser/operando_blue_laser_15x10_25p_3_cycles.txt")

    wavenumbers_all = np.array(loaded_data.columns)
    timestamps_all = np.array(loaded_data.index)
    first_timestamp = pd.DataFrame(columns=wavenumbers_all, index=np.array([0.0], dtype=np.float64), data=np.array(loaded_data.loc[0]).reshape((1, len(wavenumbers_all))))

    first_timestamp = remove_background_noise(first_timestamp)

    last_timestamp = pd.DataFrame(columns=wavenumbers_all, index=np.array([timestamps_all[-1]]), data=np.array(loaded_data.loc[timestamps_all[-1]]).reshape(1, len(wavenumbers_all)))
    last_timestamp = remove_background_noise(last_timestamp)

    import matplotlib.pyplot as plt

    wn_low = 1250.
    wn_high = 1400.
    # Remove background
    fitted = gaussian_fit(first_timestamp, wavenumber_range=[wn_low, wn_high])
    # Plot data to compare
    low_idx = find_nearest_index(wavenumbers_all, wn_low)
    high_idx = find_nearest_index(wavenumbers_all, wn_high)
    d_peak_wn = wavenumbers_all[low_idx:high_idx]

    fig, ax = plt.subplots()
    ax.plot(wavenumbers_all, np.array(first_timestamp).flatten())
    plt.show()

    real_data = np.array(first_timestamp[d_peak_wn]).flatten()
    fig, ax = plt.subplots()
    ax.scatter(d_peak_wn, real_data, color="b", label="Real data")
    ax.plot(d_peak_wn, np.array(fitted).flatten(), color="r", label="Gaussian fit")
    ax.legend()
    plt.show()
    loaded_data = remove_background_noise(loaded_data)
    # Test defect densities
    dd = defect_density(loaded_data)
    fig, ax = plt.subplots()
    ax.plot(dd.index, dd)
    plt.show()
