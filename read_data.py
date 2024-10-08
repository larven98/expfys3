import numpy as np
import pandas as pd

def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except:
        return False

def read_data_from_file(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as file:
        lines = file.readlines()
        # Read header first
        n_header_lines = 0
        n_lines_total = len(lines)
        n_axis = 0
        ax_units = []
        ax_types = []
        for line in lines:
            if line[0] == "#":
                n_header_lines += 1
                if "#AxisUnit" in line:
                    n_axis += 1
                    ax_units.append(line.split("=")[1])
                elif "#AxisType" in line:
                    ax_types.append(line.split("=")[1])
                else:
                    continue
            else:
                break

        n_columns = len(lines[n_header_lines].split("\t")) # remove one since one column will be unit axis
        x_axis = np.zeros(shape=(n_columns-1, ))
        if n_axis > 2:
            # If more than two axis, first row will be unit axis (wavenumber in our case)
            for i, val in enumerate(lines[n_header_lines].split("\t")[1:]):
                if is_float(val):
                    x_axis[i] = np.float64(val)
            n_header_lines += 1
        # First column contains time/wavenumbers
        data = np.zeros(shape=(n_lines_total - n_header_lines, n_columns - 1))
        y_axis = np.zeros(shape=(n_lines_total - n_header_lines)) 
        for i, line in enumerate(lines[n_header_lines:]):
            # Space separated values
            vals = line.split("\t")
            y_axis[i] = np.float64(vals[0])
            for j in range(1, n_columns):
                data[i, j-1] = np.float64(vals[j])
        if n_axis > 2:
            df = pd.DataFrame(index=y_axis, columns=x_axis, data=data)
        else:
            df = pd.DataFrame(index=y_axis, data=data)
        return df

def convert_to_utf8(filepath: str, encoding="iso-8859-15"):
    import codecs
    with codecs.open(filepath, "r", encoding=encoding) as infile:
        lines = infile.readlines()
        with codecs.open(filepath.split(".")[0]+"_utf8.txt", "w", encoding="utf-8") as outfile:
            outfile.writelines(lines)

if __name__ == "__main__":
    filepath = "data/run_1_full_new.txt"
    df = read_data_from_file(filepath)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cols = np.array(df.columns)
    rows = np.array(df.index)
    y = np.array(df[1000.94][320.151]).flatten()
    print(y)
    ax.plot(cols, y)
    fig.savefig("test.png")