import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import cm
import numpy as np
from read_data import read_data_from_file
import pandas as pd

def waterfall_plot(data: pd.DataFrame) -> None:
    timestamps = np.array(data.index)
    print(timestamps.shape)
    wavenumbers = np.array(data.columns)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    verts = []
    #for i, timestamp in enumerate(timestamps):
    #    intensities = np.array(data[wavenumbers].loc[timestamp]).flatten()
    #    verts.append(list(zip(wavenumbers, intensities)))
    #poly = PolyCollection(verts)
    #ax.add_collection3d(poly, zs=timestamps, zdir="y")
    W, T = np.meshgrid(wavenumbers, timestamps)
    ax.plot_surface(W, T, data[wavenumbers].loc[timestamps], cmap=cm.coolwarm)
    ax.set_xlabel(r"Wavenumbers (cm$^{-1})$")
    ax.set_ylabel(r"Time ($s$)")
    ax.set_zlabel(r"Intensity (Counts)")
    plt.show()

if __name__ == "__main__":
    df = read_data_from_file("data/run_1_full_new.txt")
    waterfall_plot(df)