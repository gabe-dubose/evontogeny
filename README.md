[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17107808.svg)](https://doi.org/10.5281/zenodo.17107808)
# EVONTOGENY
EVONTOGENY is a time-dependent geometric model of stabilizing selection acting on an ontogenetic program. Here, genes/traits are considered as expression vectors over time, as opposed to static values as described in standard geometric models. Expression vectors are then mutated by adding Gaussian pulses, and the dynamics of stabilizing selection over time determine the selective relevance of said mutations. 

## Installation
The <i>evontogeny</i> Python module can be install using
```
pip3 install git+https://github.com/gabe-dubose/evontogeny.git
```
## stochastic_model
EVONTOGENY at its core is a stochastic model, meaning evolutioanry dynamics are probabilistically simulated. Therefore, this module contains the functionality for constructing and simulating stochastic EVONTOGENY models. 

### random_developmental_program
The first step in using EVONTOGENY is to define a developmental program. You can of course specify this manually, but for simplicity, there is an option to generate a random developmental program. This is provided by the <i>random_developmental_program</i> function, which randomly generate a set of expression vectors over time.

```
Parameters:
  START           Starting point of time vector
  STOP            Stopping point of time vector
  N_POINTS        Number of time points in time vector
  N_GENES         Number of genes to generate

Returns:
  M_MATRIX        A pandas dataframe of size STOP-START by N_GENES
```
Example:
```
import numpy as np # <-- import numpy to construct time vector
from evontogeny import stochastic_model

START = 1
STOP = 50
N_POINTS = 50
N_GENES = 1
LIFETIME = np.linspace(START, STOP, N_POINTS) # <- life time vector
DEV_EXPR_MATRIX = stochastic_model.random_developmental_program(n_genes = N_GENES, time = (START, STOP), points = N_POINTS)
```


### sigma
The purpose of EVONTOGENY is to explore the relationship between a dynamic fitness ridge across ontogeny. In other words, the strength of stabilizing selection changes throughout an individuals development. The <i>sigma</i> function can be used to specify this fitness ridge. Note that several parameters are specified as vectors to accomodate multiple stage transitions.

```
Parameters:
  STAGE_TRANSITIONS      The time point corresponding to the midpoint of a life stage transition or period of selective constraint
  SIGMA_BASE             The baseline strength of stabilizing selection across non-transition times
  D                      The magnitude of decrease in stabilizing selection during the transition point
  EPSILON                The duration of the developmental transition

Returns:
  SIGMA_T                 A numpy array containing the strength of stabilizing selection at each time point during the life time
```
Example:
```
from evontogeny import stochastic_model

STAGE_TRANSITIONS = [25]
SIGMA_BASE = 0.9
D = [0.6]
EPSILON = [5]

SIGMA_FUNCT = stochastic_model.sigma(t = LIFETIME, transitions = STAGE_TRANSITIONS, sigma_0 = SIGMA_BASE, delta = D, epsilon = EPSILON)
```

### evolve
Finally, the <i>evolve</i> function can be used to run individual-based evolutionary simulations on the previously described developmental program. This is accomplished by mutating expression vectors and using Wright-Fisher style simulations to impose selection on added variation. 

```
Parameters:
  DEV_EXPR_MATRIX      A pandas dataframe containing a developmental expression matrix
  SIGMA_FUNCT          A vector containing the dynamics of stabilizing selection over development
  TAU_A                The standard deviation in amplitudes of gaussian pulses (magnitude of mutation)
  MU_L                 The mean gaussian pulse width (temporal breadth of mutation)
  TAU_L                The standard deviation in gaussian pulse width (temporal breadth of mutation)
  GENERATIONS          Number of generations to run simulations
  POPULATION_SIZE      Number of individuals in the simulated population

Returns:
  dynamics             A list with the following matricies:
                          var_matrix     A matrix of the expression variance within the population for each gene over time
                          mean_matrix    A matrix of the expression mean within the population for each gene over time
```
Example:
```
from evontogeny import stochastic_model

TAU_A = 0.9
MU_L = 1
TAU_L = len(LIFETIME) * 0.05
GENERATIONS = 500
POPULATION_SIZE = 1000
SIGMA_FUNCT = stochastic_model.sigma(t = LIFETIME, transitions = STAGE_TRANSITIONS, sigma_0 = SIGMA_BASE, delta = D, epsilon = EPSILON) # parameters described in function documentation
DEV_EXPR_MATRIX = stochastic_model.random_developmental_program(n_genes = N_GENES, time = (START, STOP), points = N_POINTS) # parameters described in function documentation

dynamics = stochastic_model.evolve(DEV_EXPR_MATRIX, SIGMA_FUNCT, TAU_A, MU_L, TAU_L, POPULATION_SIZE, GENERATIONS)
var_matrix = dynamics[0]
mean_matrix = dynamics[1]
```

