#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix
from pygam import LinearGAM, s, f
import warnings
warnings.filterwarnings("ignore")

# load pearson correlation matrix data
corr_matrix = pd.read_csv('../data/dpl_expression_data/pearson_correlation_matrix.csv', index_col = 0)

# project into PCoA space and calcualte distances to centroids
distances_dict = {}
stages = ['3', '5', 'E', 'L', 'A']

for stage in stages:
    subset = [name for name in corr_matrix.index if len(name) > 5 and name[5] == stage]

    df_subset = corr_matrix.loc[subset, subset]

    dm = DistanceMatrix(df_subset.values, ids=df_subset.index)
    pcoa_results = pcoa(dm, number_of_dimensions=10)

    coords = pcoa_results.samples.values
    explained = pcoa_results.proportion_explained

    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    df_dist = pd.DataFrame({
        'sample': subset,
        'distance_to_centroid': distances
    })

    distances_dict[stage] = df_dist

distances_df = pd.concat(distances_dict.values(), ignore_index=True)
distances_df['stage'] = distances_df['sample'].str[5]
distances_df['plant'] = distances_df['sample'].str[6]
distances_df['infection'] = distances_df['sample'].str[7]
stage_map = {'3': 1, '5': 2, 'E': 3, 'L': 4, 'A': 5}
distances_df['time_point'] = distances_df['stage'].map(stage_map)

# fit GAM
gam = LinearGAM(s(0)).fit(distances_df[['time_point']], distances_df['distance_to_centroid'])

# define predictions
X_grid = np.linspace(distances_df['time_point'].min(), distances_df['time_point'].max(), 100)
pred = gam.predict(X_grid)
conf = gam.confidence_intervals(X_grid, width=0.95)

# calcualte emperical means and error
empirical = (
    distances_df
    .groupby("time_point")['distance_to_centroid']
    .agg(['mean','count','std'])
    .reset_index()
)
empirical['sem'] = empirical['std'] / np.sqrt(empirical['count'])
empirical['ci95'] = 1.96 * empirical['sem']

# set up plot
fig, ax1 = plt.subplots(1, 1, figsize=(5,5))

# plot GAM fit
ax1.plot(X_grid, pred, color='black', lw=2, label="GAM fit")
ax1.fill_between(X_grid, conf[:,0], conf[:,1], color='lightgray', alpha=0.4)

# plot emperical
ax1.errorbar(empirical['time_point'], empirical['mean'],
             yerr=empirical['ci95'], fmt='o',
             color='black', ecolor='black', elinewidth=1.2, capsize=4,
             label="Empirical mean ±95% CI")

ax1.set_ylabel('Transcriptional variation (dispersion)', fontsize=12)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['3rd instar', '5th instar', 'Early pupa', 'Late pupa', 'Adult'], fontsize=12, rotation=45)

y_max = max(empirical['mean'] + empirical['ci95'])
ax1.set_ylim(top=y_max*1.1)
# add images
# fresh adult
fresh_adult_img = mpimg.imread("../images/new_adult.png")
imagebox = OffsetImage(fresh_adult_img, zoom=0.3)
ab = AnnotationBbox(imagebox, (5, y_max*1.1),
                    frameon=False, box_alignment=(0.5, 0), xybox=(0, 10), boxcoords="offset points")
ax1.add_artist(ab)

# late pupa
late_pupa_img = mpimg.imread("../images/late_pupa.png")
imagebox = OffsetImage(late_pupa_img, zoom=0.3)
ab = AnnotationBbox(imagebox, (4, y_max*1.1),
                    frameon=False, box_alignment=(0.5, 0), xybox=(0, 10), boxcoords="offset points")
ax1.add_artist(ab)

# early pupa
early_pupa_img = mpimg.imread("../images/early_pupa.png")
imagebox = OffsetImage(early_pupa_img, zoom=0.31)
ab = AnnotationBbox(imagebox, (3, y_max*1.1),
                    frameon=False, box_alignment=(0.5, 0), xybox=(0, 10), boxcoords="offset points")
ax1.add_artist(ab)

# 5th instar
fifth_instar_img = mpimg.imread("../images/fifth_instar.png")
imagebox = OffsetImage(fifth_instar_img, zoom=0.3)
y_max = max(empirical['mean'] + empirical['ci95'])
ab = AnnotationBbox(imagebox, (2, y_max*1.1),
                    frameon=False, box_alignment=(0.5, 0), xybox=(0, 10), boxcoords="offset points")
ax1.add_artist(ab)

# 3rd instar
third_instar_img = mpimg.imread("../images/third_instar.png")
imagebox = OffsetImage(third_instar_img, zoom=0.3)
ab = AnnotationBbox(imagebox, (1, y_max*1.1),
                    frameon=False, box_alignment=(0.5, 0), xybox=(0, 10), boxcoords="offset points")
ax1.add_artist(ab)
plt.tight_layout()

plt.savefig('../figures/dplex_metamorphosis.png', dpi=600)
print(gam.summary())
