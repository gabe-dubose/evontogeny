#!/usr/bin/env python3

from evontogeny import stochastic_model
import numpy as np
import pandas as pd

# ORGANISM
START = 1
STOP = 50
N_POINTS = 50
N_GENES = 1
LIFETIME = np.linspace(START, STOP, N_POINTS) # <- life time vector
DEV_EXPR_MATRIX = stochastic_model.random_developmental_program(n_genes = N_GENES, time = (START, STOP), points = N_POINTS)

# MUTATION
TAU_A = 0.9 # <- Standard deviation in amplitudes of gaussian pulses (magnitude of mutation)
MU_L = 1 # <- mean gaussian pulse width (temporal breadth of mutation)
TAU_L = len(LIFETIME) * 0.05 # <- standard deviation in gaussian pulse width (temporal breadth of mutation)

# POPULATION 
GENERATIONS = 500
POPULATION_SIZE = 1000

# NO TRANSITION FITNESS RIDGE
STAGE_TRANSITIONS = [0] # <- stage transition time points in relative time
SIGMA_BASE = 0.9 # <- base line fitness across non-transition times
D = [0] # <- magnitudes of decrease in fitness ridge width at each stage transition
EPSILON = [0] # <- duration of each stage transition
SIGMA_FUNCT_NO_TRANSITION = stochastic_model.sigma(t = LIFETIME, transitions = STAGE_TRANSITIONS, sigma_0 = SIGMA_BASE, delta = D, epsilon = EPSILON)

# no transition simulation
no_transition_dynamics = stochastic_model.evolve(DEV_EXPR_MATRIX, SIGMA_FUNCT_NO_TRANSITION, TAU_A, MU_L, TAU_L, POPULATION_SIZE, GENERATIONS)
no_transition_dynamics = no_transition_dynamics[0]
# convert to dataframe
no_transition_dynamics_df = pd.concat(no_transition_dynamics, ignore_index=True)
# save to csv
no_transition_dynamics_df.to_csv('../data/simulated_data/no_transition_variance_dynamics.csv', index=False)

# SINGLE TRANSITION FITNESS RIDGE
STAGE_TRANSITIONS = [25] # <- stage transition time points in relative time
SIGMA_BASE = 0.9 # <- base line fitness across non-transition times
D = [0.6] # <- magnitudes of decrease in fitness ridge width at each stage transition
EPSILON = [5] # <- duration of each stage transition
SIGMA_FUNCT_TRANSITION = stochastic_model.sigma(t = LIFETIME, transitions = STAGE_TRANSITIONS, sigma_0 = SIGMA_BASE, delta = D, epsilon = EPSILON)

transition_dynamics = stochastic_model.evolve(DEV_EXPR_MATRIX, SIGMA_FUNCT_TRANSITION, TAU_A, MU_L, TAU_L, POPULATION_SIZE, GENERATIONS)
transition_dynamics = transition_dynamics[0]
# convert to dataframe
transition_dynamics_df = pd.concat(transition_dynamics, ignore_index=True)
# save to csv
transition_dynamics_df.to_csv('../data/simulated_data/transition_variance_dynamics.csv', index=False)