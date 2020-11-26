import pickle
import random
import sys

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ClassicSEIR import ClassicSEIR
from synthetic_exp1 import construct_network, base_exp, network_exp_with_seed, er_exp, regular_exp, ba_exp, real_exp
from utils import compute_global_seir

def estimate_base(beta_range, n, t, sigma, gamma, tr_sir, symptomatic_rate, e_0_range, i_0_range, r_0_range):
    best_beta, best_e_0, best_i_0, best_r_0, min_error = 1, 1, 1, 1, sys.maxsize
    for beta in beta_range:
        for e_0 in e_0_range:
            for i_0 in i_0_range:
                for r_0 in r_0_range:

                    s_0 = n - e_0 - i_0 - r_0
                    seir_0 = np.array([s_0, e_0, i_0 * symptomatic_rate, i_0 * (1 - symptomatic_rate), r_0])
                    model2 = ClassicSEIR(np.array([n]), seir_0, beta=beta, sigma=sigma,
                                         symptomatic_rate=symptomatic_rate, gamma=gamma, num_days=t)
                    base_seir, base_pop = model2.run()
                    results = compute_global_seir(base_seir, base_pop)

                    error = compute_error(results[:15, 2], tr_sir[:15, 2], t)

                    if error < min_error:
                        min_error = error
                        best_beta = beta
                        best_e_0 = e_0
                        best_i_0 = i_0
                        best_r_0 = r_0

    return best_beta, best_e_0, best_i_0, best_r_0


def estimate_regular(n, k_range, seeds, beta, sigma, symptomatic_rate, gamma, t, base_seir, sir_0):
    best_k, min_error = 0, sys.maxsize
    for k_idx, k in enumerate(k_range):
        transmissibility = beta / k
        error_lst = np.zeros(len(seeds))

        for seed_idx in range(len(seeds)):
            random.seed(seeds[seed_idx])
            np.random.seed(seeds[seed_idx])

            G = nx.random_regular_graph(k, n)
            graph = construct_network(G, n, sir_0)

            seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
            error = compute_error(base_seir[:, 2], seir[:, 2], t)

            error_lst[seed_idx] = error

        avg_error_k = np.average(error_lst)
        print("k:", k,
              " error avg:", avg_error_k,
              " std:", np.std(error_lst),
              " max:", np.max(error_lst),
              " min:", np.min(error_lst),
              " best_seed:", seeds[np.argmin(error_lst)])

        if avg_error_k < min_error:
            min_error = avg_error_k
            best_k = k

    return best_k


def estimate_ER(n, p_range, seeds, transmissibility, sigma, symptomatic_rate, gamma, t, base_seir, sir_0):
    best_p, min_error = 0, sys.maxsize
    for p_idx, p in enumerate(p_range):
        error_lst = np.zeros(len(seeds))

        for seed_idx in range(len(seeds)):
            random.seed(seeds[seed_idx])
            np.random.seed(seeds[seed_idx])

            G = nx.fast_gnp_random_graph(n, p)
            graph = construct_network(G, n, sir_0)

            seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
            error = compute_error(base_seir[:, 2], seir[:, 2], t)

            error_lst[seed_idx] = error

        avg_error_p = np.average(error_lst)
        print("p", p,
              " error avg:", avg_error_p,
              " std:", np.std(error_lst),
              " max:", np.max(error_lst),
              " min:", np.min(error_lst),
              " best_seed:", seeds[np.argmin(error_lst)])

        if avg_error_p < min_error:
            min_error = avg_error_p
            best_p = p

    return best_p


def estimate_BA(n, m_range, p_range, seeds, transmissibility, sigma, symptomatic_rate, gamma, t, base_SEIR, sir_0):
    best_m, best_p, min_error = 0, 0, sys.maxsize

    for idx, m in enumerate(m_range):
        for idx, p in enumerate(p_range):
            error_lst = np.zeros(len(seeds))
            for seed_idx in range(len(seeds)):
                random.seed(seeds[seed_idx])
                np.random.seed(seeds[seed_idx])

                G = nx.dual_barabasi_albert_graph(n, m1=m, m2=2, p=p, seed=seeds[seed_idx])
                graph = construct_network(G, n, sir_0)

                seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
                error = compute_error(base_SEIR[:, 2], seir[:, 2], t)

                error_lst[seed_idx] = error

            avg_error_p = np.average(error_lst)
            print("m", m, " p", p,
                  " error avg:", avg_error_p,
                  " std:", np.std(error_lst),
                  " max:", np.max(error_lst),
                  " min:", np.min(error_lst),
                  " best_seed:", seeds[np.argmin(error_lst)])

            if avg_error_p < min_error:
                min_error = avg_error_p
                best_p = p
                best_m = m

    return best_m, best_p

def compute_error(expected_si, true_si, t):
    err = np.sum((true_si - expected_si) ** 2) / t
    return err


def synthetic_fit():
    tr_sir = pickle.load(open("dataset/stable/contact_network/seir_mtl_100_updated.pkl", "rb"))
    true_n = 1780000
    sim_n = int(true_n / 100)
    t = tr_sir.shape[0]

    beta_range = np.arange(start=0.5, stop=1.01, step=0.01)
    e_0_range = [tr_sir[0, 2] * 2]
    i_0_range = [tr_sir[0, 2]]
    r_0_range = [tr_sir[0, 3]]

    symptomatic_rate = 0.6
    sigma = 1 / 5
    gamma = 1 / 14
    tr_sir = tr_sir / true_n

    best_beta, best_e_0, best_i_0, best_r_0 = estimate_base(beta_range, true_n, t, sigma, gamma, tr_sir, symptomatic_rate,
                                                            e_0_range, i_0_range, r_0_range)
    print("Best beta: ", best_beta)
    print("Best e_0: ", best_e_0)
    print("Best i_0: ", best_i_0)
    print("Best r_0: ", best_r_0)

    base_seir = base_exp(best_e_0, best_i_0, best_r_0, best_beta, true_n, sigma, symptomatic_rate, gamma, t)

    simulate_e = int(np.round(best_e_0 / 100))
    simulate_i = int(np.round(best_i_0 / 100))
    simulate_i_s_0 = int(np.round(simulate_i * symptomatic_rate))
    simulate_i_a_0 = simulate_i - simulate_i_s_0
    sir_0 = np.array([simulate_e, simulate_i_s_0, simulate_i_a_0])
    print("sir 0", sir_0)

    seeds = [3, 6, 13, 20, 21, 26, 43, 64, 97, 100]

    k_range = np.arange(start=10, stop=23, step=1)
    best_k = estimate_regular(sim_n, k_range, seeds, best_beta, sigma, symptomatic_rate, gamma, t, base_seir, sir_0)
    print("Best k: ", best_k)

    transmissibility = best_beta / best_k
    print("transmissibility: ", transmissibility)

    p_range = np.arange(start=0.00130, stop=0.00150, step=0.00001)
    best_p = estimate_ER(sim_n, p_range, seeds, transmissibility, sigma, symptomatic_rate, gamma, t, base_seir, sir_0)
    print("Best p: ", best_p)

    m_range = range(5, 15)
    p_range = [1]
    best_m, best_p = estimate_BA(sim_n, m_range, p_range, seeds, transmissibility, sigma, symptomatic_rate, gamma, t, base_seir, sir_0)
    print("Best_m", best_m, "Best_p", best_p)

if __name__ == "__main__":
    import time

    start = time.time()
    synthetic_fit()
    print("Time: ", time.time() - start)
