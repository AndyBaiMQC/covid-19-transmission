import random

import numpy as np

from ClassicSIR import ClassicSIR
from Constants import COVID_MODELS
from DHSIRModel import DHSIRModel
from HSIRModel import HSIRModel
from plotting_utils import plot_synthetic_results
from utils import random_population, random_SIR_0, barabasi_albert_mobility_matrix, \
    erdos_renyi_graph_mobility_matrix, no_mobility_mobility_matrix, compute_global_seir


def synthetic_exp():
    # timestamp
    T = 100
    # recovery rate
    GAMMA = 0.05
    # number of cities
    N = 12
    # transmission rate
    BETA = np.full(shape=N, fill_value=2.0)
    seeds = range(0, 10)
    results = np.zeros((10, 3, T, 3))
    MODEL_NAME = "DHSIRModel"
    for seed in seeds:
        SIR_global = synthetic_exp_with_seed(seed, BETA, GAMMA, N, T, MODEL_NAME)
        results[seed] = SIR_global
    results = np.average(results, axis=0)
    plot_synthetic_results(T, results, model=MODEL_NAME)


def synthetic_exp_with_seed(seed, beta, gamma, n, t, model):
    random.seed(seed)
    np.random.seed(seed)
    # origin-destination flow matrix => should be replaced by estimation from flight network
    population_0 = random_population(n)
    sir_0 = random_SIR_0(population_0)
    sir = COVID_MODELS[model](population_0, sir_0, t, beta, gamma)
    no_sir, no_pop = sir.run(no_mobility_mobility_matrix(n))
    ba_sir, ba_pop = sir.run(barabasi_albert_mobility_matrix(n, 3))
    er_sir, er_pop = sir.run(erdos_renyi_graph_mobility_matrix(n, 0.25))
    results = np.array([
        compute_global_seir(no_sir, no_pop),
        compute_global_seir(ba_sir, ba_pop),
        compute_global_seir(er_sir, er_pop)])
    return results


synthetic_exp()
