
from pathlib import Path
import numpy as np
import scipy as sp
from matplotlib import cm
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

from validphys.api import API
from validphys.theorycovariance.output import matrix_plot_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_covmat_heatmap(covmat, title):
    """Matrix plot of a covariance matrix."""
    df = covmat

    matrix = df.values
    fig, ax = plt.subplots(figsize=(15, 15))
    matrixplot = ax.matshow(
        matrix,
        cmap=cm.Spectral_r,
        norm=mcolors.SymLogNorm(
            linthresh=0.00001, linscale=1, vmin=-matrix.max(), vmax=matrix.max()
        ),
    )

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = fig.colorbar(matrixplot, cax=cax)
    cbar.set_label(label=r"$\widetilde{P}$", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=25)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(df)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=20)
    ax.xaxis.tick_bottom()
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ticklabels, fontsize=20)
    
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.hlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.margins(x=0, y=0)
    return fig


class PlotHT:
  def __init__(self, preds, uncertainties, x_nodes, color, type, target, show_uncertainty = True):
    self.preds = preds
    self.color = color
    self.type = type
    self.target = target
    self.HT = sp.interpolate.CubicSpline(x_nodes, self.preds)
    self.show_uncertainty = show_uncertainty
    self.x_nodes = x_nodes
    if show_uncertainty:
      self.HT_plus = sp.interpolate.CubicSpline(x_nodes, np.add(self.preds, uncertainties))
      self.HT_minus = sp.interpolate.CubicSpline(x_nodes, np.add(self.preds, -uncertainties))

  def plot_wrapper(self, ax):
    xv = np.logspace(-5, -0.0001, 100)
    legends = []
    legend_label = rf"$H^{self.target}_{self.type} \pm \sigma$"
    legend_name = [legend_label, "knots"]
    knots = ax.plot(self.x_nodes, self.preds, 'o', label='data')
    pl = ax.plot(xv, self.HT(xv), ls = "-", lw = 1, color = self.color)

    pl_lg= ax.fill(np.NaN, np.NaN, color = self.color, alpha = 0.3) # Necessary for fancy legend
    legends.append((pl[0], pl_lg[0]))
    legends.append(knots[0])
    pl_fb = None
    if self.show_uncertainty:
      pl_fb = ax.fill_between(xv, self.HT_plus(xv), self.HT_minus(xv), color = self.color, alpha = 0.3)
    ax.set_xscale("log")
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(rf"$H^{self.target}_{self.type}$", fontsize = 10)
    ax.set_title(rf"$H^{self.target}_{self.type}$", x = 0.15, y=0.85, fontsize=10)
    ax.legend(legends, legend_name, loc=[0.1,0.07], fontsize=10)
    return (pl, pl_lg, pl_fb)

def make_dir(path):
  target_dir = Path(path)
  if not target_dir.is_dir():
      target_dir.mkdir(parents=True, exist_ok=True)