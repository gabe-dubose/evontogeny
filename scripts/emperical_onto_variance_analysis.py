#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

def simulate_2D_ontogenetic_variation(x, y, sd, N):
    coords = []
    for i in range(N):
        sim_x = np.random.normal(x, sd)
        sim_y = np.random.normal(y, sd)
        coords.append([sim_x, sim_y])
    samples = pd.DataFrame(coords, columns=["x", "y"])
    return samples

# set up figure
fig, [ax1,ax2] = plt.subplots(1,2, figsize=(8,4))

# define colormap
cmap = plt.colormaps["viridis"]
colors = [cmap(i) for i in np.linspace(0, 1, 5)]

# store origin points
origin_points = []
# store distances for each stage
distance_data = []

# define sample size
N=20

# simulate stage 1
origin_x, origin_y = 2, 9.75
origin_points.append((origin_x, origin_y))
stage_population = simulate_2D_ontogenetic_variation(x=origin_x, y=origin_y, sd=0.5, N=N)

# calculate distances
stage_population["distance"] = np.sqrt((stage_population["x"] - origin_x)**2 + 
                                       (stage_population["y"] - origin_y)**2)
stage_population["stage"] = 1
distance_data.append(stage_population[["stage", "distance"]])

ax1.scatter(data=stage_population, x='x', y='y', marker='o', color=colors[0], zorder=2)
for _, row in stage_population.iterrows():
    ax1.plot([origin_x, row['x']], [origin_y, row['y']], color="tab:gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.scatter(x=origin_x, y=origin_y, marker='o', facecolors='none', edgecolors=colors[0], zorder=3, s=100, linewidths=2)

# stage 2
origin_x, origin_y = 3, 10.25
origin_points.append((origin_x, origin_y))
stage_population = simulate_2D_ontogenetic_variation(x=origin_x, y=origin_y, sd=0.5, N=N)

# calcualte distances
stage_population["distance"] = np.sqrt((stage_population["x"] - origin_x)**2 + 
                                       (stage_population["y"] - origin_y)**2)
stage_population["stage"] = 2
distance_data.append(stage_population[["stage", "distance"]])

ax1.scatter(data=stage_population, x='x', y='y', marker='o', color=colors[1], zorder=2)
for _, row in stage_population.iterrows():
    ax1.plot([origin_x, row['x']], [origin_y, row['y']], color="tab:gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.scatter(x=origin_x, y=origin_y, marker='o', facecolors='none', edgecolors=colors[1], zorder=3, s=100, linewidths=2)

# stage 3
origin_x, origin_y = 5.25, 13
origin_points.append((origin_x, origin_y))
stage_population = simulate_2D_ontogenetic_variation(x=origin_x, y=origin_y, sd=0.2, N=N)

# calcualte distances
stage_population["distance"] = np.sqrt((stage_population["x"] - origin_x)**2 + 
                                       (stage_population["y"] - origin_y)**2)
stage_population["stage"] = 3
distance_data.append(stage_population[["stage", "distance"]])

ax1.scatter(data=stage_population, x='x', y='y', marker='o', color=colors[2], zorder=2)
for _, row in stage_population.iterrows():
    ax1.plot([origin_x, row['x']], [origin_y, row['y']], color="tab:gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.scatter(x=origin_x, y=origin_y, marker='o', facecolors='none', edgecolors=colors[2], zorder=3, s=100, linewidths=2)

# stage 4
origin_x, origin_y = 7.5, 10
origin_points.append((origin_x, origin_y))
stage_population = simulate_2D_ontogenetic_variation(x=origin_x, y=origin_y, sd=0.5, N=N)

# calcualte distances
stage_population["distance"] = np.sqrt((stage_population["x"] - origin_x)**2 + 
                                       (stage_population["y"] - origin_y)**2)
stage_population["stage"] = 4
distance_data.append(stage_population[["stage", "distance"]])

ax1.scatter(data=stage_population, x='x', y='y', marker='o', color=colors[3], zorder=2)
for _, row in stage_population.iterrows():
    ax1.plot([origin_x, row['x']], [origin_y, row['y']], color="tab:gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.scatter(x=origin_x, y=origin_y, marker='o', facecolors='none', edgecolors=colors[3], zorder=3, s=100, linewidths=2)

# stage 5
origin_x, origin_y = 8.5, 9.75
origin_points.append((origin_x, origin_y))
stage_population = simulate_2D_ontogenetic_variation(x=origin_x, y=origin_y, sd=0.5, N=N)

# calcualte distances
stage_population["distance"] = np.sqrt((stage_population["x"] - origin_x)**2 + 
                                       (stage_population["y"] - origin_y)**2)
stage_population["stage"] = 5
distance_data.append(stage_population[["stage", "distance"]])

ax1.scatter(data=stage_population, x='x', y='y', marker='o', color=colors[4], zorder=2)
for _, row in stage_population.iterrows():
    ax1.plot([origin_x, row['x']], [origin_y, row['y']], color="tab:gray", linestyle="--", linewidth=0.8, zorder=1)
ax1.scatter(x=origin_x, y=origin_y, marker='o', facecolors='none', edgecolors=colors[4], zorder=3, s=100, linewidths=2)

# connect centroids
origin_points = np.array(origin_points)
x_vals, y_vals = origin_points[:,0], origin_points[:,1]

# smooth spline
x_dense = np.linspace(x_vals.min(), x_vals.max(), 200)
spline = make_interp_spline(x_vals, y_vals, k=2)  # cubic spline
y_dense = spline(x_dense)

# plot curve
ax1.plot(x_dense, y_dense, color="tab:gray", linewidth=2.5, zorder=0, alpha=0.5)

# add arrows to show curve progression
for i in range(0, len(x_dense)-1, 20):
    ax1.annotate(
        '',
        xy=(x_dense[i+1], y_dense[i+1]),
        xytext=(x_dense[i], y_dense[i]),
        arrowprops=dict(
            arrowstyle='->', 
            color='tab:gray', 
            alpha=0.5,
            lw=2
        )
    )

# add text
ax1.text(x=2.6, y=9.4, s='Stage 1', color=colors[0], fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
ax1.text(x=0.85, y=10.65, s='Stage 2', color=colors[1], fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
ax1.text(x=2.75, y=13, s='Stage 3', color=colors[2], fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
ax1.text(x=7.5, y=10.5, s='Stage 4', color=colors[3], fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
ax1.text(x=8, y=9.1, s='Stage 5', color=colors[4], fontsize=8, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))


ax1.set_xlim(0.25, 10)
# adjust plot labels
ax1.set_xticks([]) 
ax1.set_yticks([])
ax1.set_xlabel('Axis 1: X1%', fontsize=12)
ax1.set_ylabel('Axis 2: X2%', fontsize=12)

# plot distances
distance_df = pd.concat(distance_data, ignore_index=True)
sns.pointplot(data=distance_df, x="stage", y="distance", ax=ax2, color="black", errorbar="se")

# plot defined V*
ax2_right = ax2.twinx()
v_stars = [0.5, 0.5, 0.2, 0.5, 0.5]
stages = [0,1,2,3,4]
ax2_right.plot(stages, v_stars, color='tab:gray', linestyle='--', marker='o')
ax2_right.set_ylabel(r'$\sigma(t)$', fontsize=12)
ax2.set_xlabel('Stage', fontsize=12)
ax2.set_ylabel(r'Empirical estimate ($\delta^2_{\mathrm{emp}}(t))$', fontsize=12)

# create ax1 legend
legend_elements = [
    Line2D([0], [0], marker='o', color='black', markerfacecolor='white', label=r'Centroid ($\bar{z}(t)$)', linestyle='None'),
    Line2D([0], [0], marker='o', color='black', markerfacecolor='black', label='Sample ($z_i(t)$)', linestyle='None')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True)

# create ax2 legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$\delta^2_{\mathrm{emp}}(t)$'),
    Line2D([0], [0], color='tab:gray', linestyle='--', label=r'$\sigma(t)$')
]

ax2.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

# add axes labels
# panel labels
ax1.text(-0.025, 1.02, "A", transform=ax1.transAxes,
         fontsize=15, va='bottom', ha='right')
ax2.text(-0.15, 1.02, "B", transform=ax2.transAxes,
         fontsize=15, va='bottom', ha='right')

plt.tight_layout()
plt.savefig('../figures/emperical_selection_estimation_concept.png', dpi=600)