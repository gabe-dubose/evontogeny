import numpy as np
import random 
import pandas as pd

# function to get a random expression pattern
def get_expression_pattern(n_points, x_range):
    x = np.linspace(x_range[0], x_range[1], n_points)
    func_type = random.choice([
        'linear', 'quadratic', 'logistic',
        'exp_increase', 'exp_decay', 'michaelis_menten'
    ])

    if func_type == 'linear':
        a = random.uniform(-5, 5)
        b = random.uniform(-10, 10)
        y = a * x + b

    elif func_type == 'quadratic':
        a = random.uniform(-1, 1)
        b = random.uniform(-5, 5)
        c = random.uniform(-10, 10)
        y = a * x**2 + b * x + c

    elif func_type == 'logistic':
        L = random.uniform(5, 20)
        k = random.uniform(0.5, 2)
        x0 = random.uniform(*x_range)
        y = L / (1 + np.exp(-k * (x - x0)))

    elif func_type == 'exp_increase':
        a = random.uniform(0.5, 2)
        b = random.uniform(0.1, 1)
        y = a * np.exp(b * x)

    elif func_type == 'exp_decay':
        a = random.uniform(0.5, 2)
        b = random.uniform(0.1, 1)
        y = a * np.exp(-b * x)

    elif func_type == 'michaelis_menten':
        Vmax = random.uniform(5, 20)
        Km = random.uniform(0.5, 10)
        y = (Vmax * x) / (Km + x)

    # make all values positive and min-max normalize
    y = np.abs(y)
    y = (y - y.min()) / (y.max() - y.min())

    return y.tolist()

# function to generate random M matrix
def random_developmental_program(n_genes, time, points):
    # initialize matrix
    M_matrix = []
    # generate random expression patterns
    for g in range(n_genes):
        expression = get_expression_pattern(n_points = points, x_range=(time[0], time[1]))
        M_matrix.append(expression)

    # convert to dataframe
    M_matrix = pd.DataFrame(M_matrix)
    return M_matrix
        
# function to generate mutation in a single gene
def mutate(expr, tau_a, mu_l, tau_l):
    """
    Mutate a single gene expression time series (expr: 1D numpy array/list).
    - tau_a: sd for amplitude A ~ N(0, tau_a)
    - mu_l, tau_l: mean and sd for width l ~ |N(mu_l, tau_l)|
    Returns mutated expression (no immediate clipping to >=0).
    """
    expr = np.asarray(expr, dtype=float)
    T = len(expr)
    t_vals = np.arange(T)

    # 1) sample amplitude and width
    A = np.random.normal(0.0, tau_a)
    r = abs(np.random.normal(mu_l, tau_l))
    r = max(r, 1e-6)

    # 2) sample center uniformly across full interval
    s = np.random.randint(0, T)

    # 3) raw Gaussian pulse (discrete)
    p_raw = np.exp(-((t_vals - s) ** 2) / (2.0 * r ** 2))

    # 4) compute reference norm using a centered pulse at midpoint (discrete)
    mid = (T - 1) / 2.0
    p_center = np.exp(-((t_vals - mid) ** 2) / (2.0 * r ** 2))

    # discrete L2 norms
    eps = 1e-12
    norm_raw = np.sqrt(np.sum(p_raw ** 2)) + eps
    norm_center = np.sqrt(np.sum(p_center ** 2)) + eps

    # 5) normalize raw pulse so that its L2 norm equals the centered pulse L2
    p = p_raw * (norm_center / norm_raw)

    # 6) scale by amplitude (A can be negative)
    pulse = A * p

    # 7) add pulse to expression but DO NOT clip here
    mutated = expr + pulse

    # return mutated WITHOUT forced truncation to zero
    return mutated

# function to generate mutations in expression matricies
def mutate_expr_matrix(expr_matrix, tau_a, mu_l, tau_l):
    # initialize mutated matrix
    mutated_expr_matrix = expr_matrix.copy()
    
    # mutate each gene
    for gene in expr_matrix.index:
        expression = list(mutated_expr_matrix.iloc[gene])
        mutated_expression = mutate(expression, tau_a, mu_l, tau_l)
        mutated_expr_matrix.iloc[gene] = mutated_expression
        
    return mutated_expr_matrix
    
# function to define fitness ridge across metamorphosis
def sigma(t, transitions, sigma_0, delta, epsilon):
    t = np.array(t)
    sigma_t = np.full_like(t, fill_value=sigma_0, dtype=float)
    
    for s, d, e in zip(transitions, delta, epsilon):
        dec = d * np.exp(-((t - s) ** 2) / (2 * e ** 2))
        sigma_t -= dec
    return sigma_t

# function to calculate fitness
def calculate_fitness(expr_matrix, opt_expr_matrix, sigma_funct):
    # initialize distnace over life time vector
    distance = 0
    # iterate through each time point and calcualte fitness
    for t in expr_matrix.columns:
        expr = expr_matrix[t]
        opt = opt_expr_matrix[t]
        distance_from_opt = sum([abs(expr[g] - opt[g])**2 for g in range(len(expr))])
        # update distance
        distance += distance_from_opt / (2 * sigma_funct[t] ** 2)
    W = np.exp(-distance)
    return W

# function to evolve expression matrix
# note that the DEV_EXPR_MATRIX_0 parameter is assumed to be the optimum 
def evolve(DEV_EXPR_MATRIX_0, SIGMA_FUNCT, TAU_A, MU_L, TAU_L, POPULATION_SIZE, GENERATIONS):
    
    # initialize a list to store mean and var M matrix over time (list of length GENERATIONS)
    var_matrix = []
    mean_matrix = []

    # initialize population
    population = [DEV_EXPR_MATRIX_0.copy() for i in range(POPULATION_SIZE)]
    # define current generation
    current_generation = population

    # iterate through generations
    for g in range(GENERATIONS):
        population_fitness = []
        mutated_population = []
        
        # mutate each individual and store mutated versions
        for individual in current_generation:
            mutated_individual = mutate_expr_matrix(individual, TAU_A, MU_L, TAU_L)
            mutated_population.append(mutated_individual)
            fitness = calculate_fitness(mutated_individual, DEV_EXPR_MATRIX_0, SIGMA_FUNCT)
            population_fitness.append(fitness)
        
        # select parents from mutated individuals
        parents = random.choices(range(len(population_fitness)), weights=population_fitness, k=POPULATION_SIZE)
        
        # next generation are copies of selected mutated individuals
        next_generation = [mutated_population[parent].copy() for parent in parents]

        # stack individuals for stats
        stacked = pd.concat(next_generation)
        
        # per-gene variance across individuals
        variance_matrix = stacked.groupby(level=0).var()
        var_matrix.append(variance_matrix)
        
        # per-gene mean expression across individuals
        mean_expr_matrix = stacked.groupby(level=0).mean()
        mean_matrix.append(mean_expr_matrix)
        
        # update current generation
        current_generation = next_generation
        
    return [var_matrix, mean_matrix]