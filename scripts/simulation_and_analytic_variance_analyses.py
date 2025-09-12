#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from labellines import labelLines
from evontogeny import stochastic_model
import pandas as pd
import matplotlib as mpl
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# load simulated data
no_transition_data = pd.read_csv('../data/simulated_data/no_transition_variance_dynamics.csv')
transition_data = pd.read_csv('../data/simulated_data/transition_variance_dynamics.csv')

# define parameters
START = 1
STOP = 50
N_POINTS = 50
N_GENES = 1
LIFETIME = np.linspace(START, STOP, N_POINTS)  # <- life time vector

# NO TRANSITION FITNESS RIDGE
STAGE_TRANSITIONS = [0]
SIGMA_BASE = 0.9
D = [0]
EPSILON = [0]
SIGMA_FUNCT_NO_TRANSITION = stochastic_model.sigma(
    t=LIFETIME, transitions=STAGE_TRANSITIONS, sigma_0=SIGMA_BASE, delta=D, epsilon=EPSILON
)

# SINGLE TRANSITION FITNESS RIDGE
STAGE_TRANSITIONS = [25]
SIGMA_BASE = 0.9
D = [0.6]
EPSILON = [5]
SIGMA_FUNCT_TRANSITION = stochastic_model.sigma(
    t=LIFETIME, transitions=STAGE_TRANSITIONS, sigma_0=SIGMA_BASE, delta=D, epsilon=EPSILON
)

# apply gaussian smoothing to data
smoothed_no_transition = gaussian_filter(no_transition_data.T.iloc[:, :50], sigma=5)
smoothed_transition = gaussian_filter(transition_data.T.iloc[:, :50], sigma=5)

# -------------------------
fig = plt.figure(figsize=(10, 8), constrained_layout=True)

# main grid: 3 rows, 1 column
gs = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[1, 1, 1.25], hspace=0
)

# top 2 rows: 1:4 width ratio
gs_top = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs[:2], width_ratios=[1, 4], hspace=0, wspace=0
)

# bottom row: 1:1 width ratio
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[2], width_ratios=[1, 1], wspace=0.1
)

# assign axes
ax1 = fig.add_subplot(gs_top[0, 0])
ax2 = fig.add_subplot(gs_top[0, 1])
ax3 = fig.add_subplot(gs_top[1, 0])
ax4 = fig.add_subplot(gs_top[1, 1])
ax5 = fig.add_subplot(gs_bottom[0, 0])
ax6 = fig.add_subplot(gs_bottom[0, 1])
# define color scales (shared)
vmin = min(smoothed_no_transition.min().min(), smoothed_transition.min().min())
vmax = max(smoothed_no_transition.max().max(), smoothed_transition.max().max())

# -------------------------
# no transition
# sigma function
ax1.plot(
    list(SIGMA_FUNCT_NO_TRANSITION),
    list(range(len(SIGMA_FUNCT_NO_TRANSITION))),
    color="black",
    linewidth=2
)
ax1.set_xlabel(r'$\sigma(t)$', fontsize=12)
ax1.set_ylabel(r'Ontogenetic time', fontsize=12)
ax1.set_xlim(0.25, 1)
ax1.text(x=0.27, y=47, s='Linear ontogeny', fontsize=8)

hm1 = sns.heatmap(
    data=smoothed_no_transition, cmap='viridis', ax=ax2,
    vmin=vmin, vmax=vmax, cbar=False
)
ax2.invert_yaxis()
ax2.set_ylabel(r'Ontogenetic time', fontsize=12)
ax2.set_xlabel('Evolutionary time (Generations)', fontsize=12)

# -------------------------
# transition
# sigma function
ax3.plot(
    list(SIGMA_FUNCT_TRANSITION),
    list(range(len(SIGMA_FUNCT_TRANSITION))),
    color="black",
    linewidth=2
)
ax3.set_xlabel(r'$\sigma(t)$', fontsize=12)
ax3.set_ylabel(r'Ontogenetic time', fontsize=12)
ax3.set_xlim(0.25, 1)
ax3.axhline(y=10, color='tab:gray', linestyle='--')
ax3.axhline(y=40, color='tab:gray', linestyle='--')
ax3.text(x=0.27, y=47, s='Ontogenetic transition', fontsize=8)

hm2 = sns.heatmap(
    data=smoothed_transition, cmap='viridis', ax=ax4,
    vmin=vmin, vmax=vmax, cbar=False
)
ax4.invert_yaxis()
ax4.set_ylabel(r'Ontogenetic time', fontsize=12)
ax4.set_xlabel('Evolutionary time (Generations)', fontsize=12)

# add contours to ax4
Z = smoothed_transition.values if hasattr(smoothed_transition, "values") else np.array(smoothed_transition)
y = np.arange(Z.shape[0])
x = np.arange(Z.shape[1])
X, Y = np.meshgrid(x, y)
contours = ax4.contour(X, Y, Z, colors='white', linewidths=1, levels=5, alpha=1)
ax4.clabel(contours, inline=True, fontsize=7, fmt="%.3f")

# -------------------------
# Manually add colorbars with height factor
cbar_width = 0.015
cbar_pad = 0.01
cbar_height_factor = 0.925

# colorbar for ax2
im1 = hm1.collections[0]
cb1 = fig.colorbar(im1, ax=ax2, location="right", fraction=0.046, pad=0.01)
cb1.ax.tick_params(labelsize=8)
cb1.set_label("Var(Expression)", fontsize=12)
cb1.set_ticks(np.linspace(vmin, vmax, 8))

# colorbar for ax4
im2 = hm2.collections[0]
cb2 = fig.colorbar(im2, ax=ax4, location="right", fraction=0.046, pad=0.01)
cb2.ax.tick_params(labelsize=8)
cb2.set_label("Var(Expression)", fontsize=12)
cb2.set_ticks(np.linspace(vmin, vmax, 8))

# -------------------------
# Analytic variance analysis
# parameters
M = 0.05  # <- mutational input
sigma_values = [0.1, 0.2, 0.3, 0.4]  # <- range of sigma values
g_span = (0, 10)  # <- generations
V0 = 0  # <- initial variance

# ode function
def dVdg(g, V, M, sigma):
    sigma_sq = sigma**2
    return M - V**2 / (V + sigma_sq)

# variance trajectories
g_points = np.linspace(*g_span, 500)
for sigma in sigma_values:
    sol = solve_ivp(dVdg, g_span, [V0], args=(M, sigma),
                    dense_output=True, max_step=0.05)
    V_points = sol.sol(g_points)[0]
    ax5.plot(g_points, V_points, label=rf'$\sigma={sigma}$', color='black')

# add line labels
labelLines(ax5.get_lines(), align=True, fontsize=12)
ax5.set_xlabel("Generations", fontsize=12)
ax5.set_ylabel("Var (Expr.)", fontsize=12)
ax5.text(-0.35, 0.1125, s=rf'$M$={M}')

# fixed points
sigma_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
fixed_points = [(M + np.sqrt(M**2 + 4*M*(s**2))) / 2 for s in sigma_values]
ax6.plot(sigma_values, fixed_points, 'o-', color='black')
ax6.set_xlabel(r'$\sigma$', fontsize=12)
ax6.set_ylabel(r'Equilibrium variance ($V^*$)', fontsize=12)

# panel labels
ax1.text(-0.25, 0.95, "A", transform=ax1.transAxes,
         fontsize=17, va='bottom', ha='right')
ax3.text(-0.25, 0.95, "B", transform=ax3.transAxes,
         fontsize=17, va='bottom', ha='right')
ax5.text(-0.15, 0.95, "C", transform=ax5.transAxes,
         fontsize=17, va='bottom', ha='right')
ax6.text(-0.15, 0.95, "D", transform=ax6.transAxes,
         fontsize=17, va='bottom', ha='right')

plt.savefig('../figures/variance_analysis.png', dpi=600, bbox_inches='tight')