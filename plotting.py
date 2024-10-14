import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import cm
import numpy as np
from read_data import read_data_from_file
import pandas as pd
import data_analysis

def waterfall_plot(data: pd.DataFrame) -> None:
    timestamps = np.array(data.index)
    wavenumbers = np.array(data.columns)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    W, T = np.meshgrid(wavenumbers, timestamps)
    ax.plot_surface(W, T, data[wavenumbers].loc[timestamps], cmap=cm.coolwarm)
    ax.set_xlabel(r"Wavenumbers (cm$^{-1})$")
    ax.set_ylabel(r"Time ($s$)")
    ax.set_zlabel(r"Intensity (Counts)")
    plt.show()

if __name__ == "__main__":
    df = read_data_from_file("data/Blue laser/operando_blue_laser_15x10_25p_3_cycles.txt")
    df = data_analysis.remove_background_noise(df)
    waterfall_plot(df)