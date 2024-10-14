import read_data
import data_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    cycle_data = read_data.read_cycle_data("data/groupe3b_CV_0.1mV_s_red_laser_C01.txt")
    current = cycle_data["<I>/mA"]
    potential = cycle_data["Ewe/V"]
    time = cycle_data["time/s"]
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.plot(time, current, color="b")
    ax1.plot(time, potential, color="r")
    # Find points where potential shifts sign
    tops = []
    for i in range(1, len(potential) - 1, 1):
        if np.sign(potential[i+1] - potential[i]) != np.sign(potential[i] - potential[i-1]):
            tops.append(i)
    tops.append(len(potential)-1)
    tops = np.array(tops)
    print(tops)
    # Compute integrals
    # Order is discharge -> charge -> discharge
    charge = []
    discharge = []
    dt = time[1] - time[0]
    for i in range(0, len(tops) - 1, 2):
        discharge.append(np.sum(dt*current[tops[i]:tops[i+1]]))
    for i in range(1, len(tops) - 1, 2):
        charge.append(np.sum(dt*current[tops[i]:tops[i+1]]))
    print(np.abs(charge) / np.abs(discharge))
    plt.show()