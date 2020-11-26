import networkx as nx
import numpy as np
import math

from scipy import stats

def barabasi_albert_mobility_matrix(n, m):
    G = nx.barabasi_albert_graph(n=n, m=m)
    mobility_matrix = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(mobility_matrix, 0)
    return np.array(mobility_matrix)


def erdos_renyi_graph_mobility_matrix(n, p):
    G = nx.erdos_renyi_graph(n=n, p=p, directed=True)
    mobility_matrix = nx.adjacency_matrix(G).todense()
    np.fill_diagonal(mobility_matrix, 0)
    return np.array(mobility_matrix)


def random_mobility_matrix(n, population, low=0.00002, high=0.0001, international_weight=1e-5):
    scaled_population = population.copy()
    scaled_population[-1] = scaled_population[-1] * international_weight
    mobility_matrix = []
    for pop in scaled_population:
        low_pop = round(pop * low)
        high_pop = max(round(pop * high), low_pop + 1)
        mobility_matrix.append(np.random.randint(low=low_pop, high=high_pop, size=n))
    return np.array(mobility_matrix)


def no_mobility_mobility_matrix(N):
    mobility_matrix = np.zeros((N, N))
    return mobility_matrix


def random_population(N, low=1000, high=1200):
    return np.random.randint(low=low, high=high, size=N)


def random_SIR_0(population, threshold=1100, div=500, alt=0):
    num_cities = len(population)
    # S, I, R for each city
    SIR = np.zeros(shape=(num_cities, 3))
    # initialize S with the population of each city
    SIR[:, 0] = population
    # initialize I randomly
    first_infections = np.where(SIR[:, 0] <= threshold, SIR[:, 0]//div, alt)
    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections
    return SIR


# return (model, t, compartment)
def get_sir_by_scope(exp, sir, population, scope):
    target_indices = exp.get_target_indices()
    if scope == "Canada":
        # exclude international
        return np.sum(sir[:, :, target_indices, :], axis=2) / np.expand_dims(
            np.sum(population[:, :, target_indices], axis=2), 2)
    else:
        idx_prov = target_indices[np.argwhere(exp.provinces == scope)[0][0]]
        return compute_local_seir(sir[:, :, idx_prov, :], population[:, :, idx_prov])


# return (t, compartment)
def compute_global_seir(seir, population):
    return np.sum(seir, axis=1) / np.expand_dims(np.sum(population, axis=1), 1)


# return (t, node, compartment)
def compute_local_seir(seir, population):
    epsilon = 1e-1
    return seir / np.expand_dims(population + epsilon, 2)

def compute_mean_confidence_interval(seir, alpha=0.05):
    # seir.shape = (num_seeds, len(npi_strength), T, len(SEIR_COMPARTMENTS))
    mean = np.average(seir, axis=0)

    ci = np.zeros(mean.shape)

    for i in range(seir.shape[1]):
        for j in range(seir.shape[2]):
            for k in range(seir.shape[3]):
                theta = seir[:,i,j,k]
                n = len(theta)
                se = stats.sem(theta) / math.sqrt(n)
                if seir.shape[0] < 30:
                    ci[i,j,k] = se * stats.t.ppf((1 + (1-alpha)) / 2., n-1)
                else:
                     ci[i,j,k] = se * stats.norm.ppf(1-(alpha/2))

    return mean, ci

def compute_difference(base, seir, cumulative=True):
    diff = np.zeros(seir.shape)
    for strength_idx in range(seir.shape[1]):
        for seed_idx in range(seir.shape[0]):
            diff[seed_idx,strength_idx,:,:] = seir[seed_idx,strength_idx,:,:] - base[seed_idx, :, :]
    if cumulative:
        diff = np.cumsum(diff, axis=2)
    return diff