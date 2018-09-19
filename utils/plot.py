import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

def plot(x, y, x_axis_name=None, y_axis_name=None, title=None, plot_filepath=None):
    """Plots the points given the x-and y-coordinates

    Args:
        x: The horizontal coordinates
        y: The vertical coordinates
        x_axis_name: The horizontal axis name to use in the plot
        y_axis_name: The vertical axis name to use in the plot
        title: The title to use in the plot
        plot_filepath: The optional file path to use when saving the plot
    """
    plt.clf()
    plt.plot(x, y)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_filepath)
