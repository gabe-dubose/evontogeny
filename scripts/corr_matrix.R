#!/usr/bin/env Rscript

# load data

data <- read.csv('../data/dpl_tpm_counts_kallisto.csv', row.names = 1)

# calculate pearson correlation distance
corr.matrix <- cor(t(data), method="pearson")
corr.matrix <- (1 - corr.matrix)/2

write.csv(corr.matrix, '../data/pearson_correlation_matrix.csv')
