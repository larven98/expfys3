import numpy as np
import matplotlib.pyplot as plt

def backcor(n, y, ord=None, s=None, fct=None):
    """
    Background estimation by minimizing a non-quadratic cost function.
    
    Parameters:
    n (numpy.ndarray): Wavelengths or x-values.
    y (numpy.ndarray): Signal or y-values.
    ord (int): Polynomial order.
    s (float): Threshold for the cost function.
    fct (str): Cost function type ('sh', 'ah', 'stq', 'atq').
    
    Returns:
    z (numpy.ndarray): Estimated background.
    a (numpy.ndarray): Polynomial coefficients.
    it (int): Number of iterations.
    """
    
    if ord is None or s is None or fct is None:
        return backcor_gui(n, y)
    
    if fct not in ['sh', 'ah', 'stq', 'atq']:
        raise ValueError("Unknown function type. Choose from 'sh', 'ah', 'stq', 'atq'.")
    
    # Rescaling
    N = len(n)
    n_sorted_indices = np.argsort(n)
    n = np.sort(n)
    y = y[n_sorted_indices]
    maxy = np.max(y)
    dely = (maxy - np.min(y)) / 2
    n = 2 * (n - n[-1]) / (n[-1] - n[0]) + 1
    y = (y - maxy) / dely + 1
    
    # Vandermonde matrix
    p = np.arange(ord + 1)
    T = np.vander(n, ord + 1, increasing=True)
    Tinv = np.linalg.pinv(T.T @ T) @ T.T
    
    # Initial least-squares estimation
    a = Tinv @ y
    z = T @ a
    
    # Variables
    alpha = 0.99 * 0.5  # Scale parameter alpha
    it = 0  # Iteration count
    zp = np.ones(N)  # Previous estimation
    
    # Iterative loop
    while np.sum((z - zp) ** 2) / np.sum(zp ** 2) > 1e-9:
        it += 1
        zp = z.copy()
        res = y - z
        
        # Compute d based on the cost function type
        if fct == 'sh':
            d = (res * (2 * alpha - 1)) * (np.abs(res) < s) + (-alpha * 2 * s - res) * (res <= -s) + (alpha * 2 * s - res) * (res >= s)
        elif fct == 'ah':
            d = (res * (2 * alpha - 1)) * (res < s) + (alpha * 2 * s - res) * (res >= s)
        elif fct == 'stq':
            d = (res * (2 * alpha - 1)) * (np.abs(res) < s) - res * (np.abs(res) >= s)
        elif fct == 'atq':
            d = (res * (2 * alpha - 1)) * (res < s) - res * (res >= s)
        
        # Update z and a
        a = Tinv @ (y + d)
        z = T @ a
    
    # Rescale the result back
    z = (z - 1) * dely + maxy
    a[0] -= 1
    a *= dely
    
    return z, a, it

# GUI Implementation
def backcor_gui(n, y):
    """
    Graphical User Interface for background estimation.
    
    This function sets up a GUI for the user to adjust parameters for background correction.
    """
    ord = 4
    s = 0.01
    fct = 'atq'
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.3)
    
    z, a, it = backcor(n, y, ord, s, fct)
    
    def update(val):
        nonlocal ord, s, fct, z, a, it
        ord = int(slider_order.val)
        s = slider_thresh.val
        fct = dropdown_fct.get()  # Read from dropdown
        z, a, it = backcor(n, y, ord, s, fct)
        ax.cla()
        ax.plot(n, y, 'b-', label='Original Signal')
        ax.plot(n, z, 'r-', label='Estimated Background')
        ax.legend()
        fig.canvas.draw_idle()
    
    # Cost function dropdown
    ax_fct = plt.axes([0.05, 0.7, 0.2, 0.15])
    dropdown_fct = plt.widgets.RadioButtons(ax_fct, ['sh', 'ah', 'stq', 'atq'], active=3)
    dropdown_fct.on_clicked(update)
    
    # Polynomial order slider
    ax_order = plt.axes([0.05, 0.5, 0.2, 0.03], facecolor='lightgray')
    slider_order = plt.widgets.Slider(ax_order, 'Order', 0, 10, valinit=ord, valstep=1)
    slider_order.on_changed(update)
    
    # Threshold slider
    ax_thresh = plt.axes([0.05, 0.4, 0.2, 0.03], facecolor='lightgray')
    slider_thresh = plt.widgets.Slider(ax_thresh, 'Threshold', 0, 1, valinit=s, valstep=0.01)
    slider_thresh.on_changed(update)
    
    ax.plot(n, y, 'b-', label='Original Signal')
    ax.plot(n, z, 'r-', label='Estimated Background')
    ax.legend()
    
    plt.show()

# Example usage (without GUI):
# n = np.linspace(0, 10, 100)
# y = np.sin(n) + np.random.normal(0, 0.1, len(n))
# z, a, it = backcor(n, y, ord=4, s=0.01, fct='atq')
