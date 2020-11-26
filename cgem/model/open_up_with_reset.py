import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import datetime as dt

from ClassicSEIR import ClassicSEIR
from Constants import SEIR_COMPARTMENTS
from NetworkedSEIR import NetworkedSEIR
from plotting_utils import plot_synthetic_by_compartment, plot_degree_dist, plot_open_up
from utils import compute_global_seir, compute_mean_confidence_interval, compute_difference
from synthetic_exp1 import construct_network

def open_up_with_reset():
    tr_sir = pickle.load(open("dataset/stable/contact_network/seir_mtl_100_updated.pkl", "rb"))
    dates = [dt.datetime.strptime(d, '%Y-%m-%d') for d in pickle.load(open("model/mtl_dates.pkl", 'rb'))]
    start_date = dates[0]

    # wifi1_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2004-08-27_2006-11-30_GCC.edgelist", nodetype=int)
    # wifi2_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2007-07-01_2008-02-26_GCC.edgelist", nodetype=int)
    # wifi3_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2009-12-02_2010-03-08_GCC.edgelist", nodetype=int)

    # timestamp
    # date_open_up = dt.datetime.strptime('2020-05-22', '%Y-%m-%d') # sd
    # date_open_up = dt.datetime.strptime('2020-08-01', '%Y-%m-%d') # q
    # date_open_up = dt.datetime.strptime('2020-06-01', '%Y-%m-%d') # m
    date_open_up = dt.datetime.strptime('2020-07-18', '%Y-%m-%d') # h
    T = (date_open_up-start_date).days + 50
    dates = pd.date_range(start = start_date, periods = T).strftime('%m-%d').tolist()

    base_e_0 = tr_sir[0, 2] * 2
    base_i_0 = tr_sir[0, 2]
    base_r_0 = tr_sir[0, 3]

    # transmission rate
    TRANSMISSIBILITY = 1
    # BETA = TRANSMISSIBILITY * 3
    BETA = 0.78
    # 1 / days latent
    SIGMA = 1 / 5
    # recovery rate = 1 / days infected
    GAMMA = 1 / 14
    # the probability of an exposed become symptomatic infected
    SYMPTOMATIC_RATE = 0.6
    # total population
    n = 1780000
    N = 17800

    hub_percent = [0.01, 0.05, 0.1]
    distance_percent = [0.1, 0.3, 0.5]
    quarantine_percent = [0.5, 0.75, 0.95]

    seeds = [3, 6, 13, 20, 21, 26, 43, 64, 97, 100]
    num_seeds = len(seeds)

    Regular_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    ER_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    BA_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results1 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results2 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results3 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))

    Regular_results_hub = np.zeros((num_seeds, len(hub_percent),T, len(SEIR_COMPARTMENTS)))
    ER_results_hub = np.zeros((num_seeds, len(hub_percent), T, len(SEIR_COMPARTMENTS)))
    BA_results_hub = np.zeros((num_seeds, len(hub_percent), T, len(SEIR_COMPARTMENTS)))
    Real_results_hub1 = np.zeros((num_seeds, len(hub_percent), T, len(SEIR_COMPARTMENTS)))
    Real_results_hub2 = np.zeros((num_seeds, len(hub_percent), T, len(SEIR_COMPARTMENTS)))
    Real_results_hub3 = np.zeros((num_seeds, len(hub_percent), T, len(SEIR_COMPARTMENTS)))


    K = 21 # for regular graph
    ER_P = 0.0014  # for ER graph
    M = 10  # for BA graph
    BA_P = 1 # for BA graph
    REAL_scale = 1
    tr_sir = tr_sir / n
    TRANSMISSIBILITY = BETA/K

    t_open_up =  dates.index('07-18')

    e_0 = int(np.round(base_e_0 / 100))
    i_0 = int(np.round(base_i_0 / 100))
    i_s_0 = int(np.round(i_0 * SYMPTOMATIC_RATE))
    i_a_0 = i_0 - i_s_0
    sir_0 = [e_0, i_s_0, i_a_0]

    for seed_idx, seed in enumerate(seeds):
        seed_time = time.time()
        seir = regular_exp(T, N, K, BETA, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)
        Regular_results[seed_idx] = seir
        ER_results[seed_idx] = er_exp(T, N, ER_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)
        BA_results[seed_idx] = ba_exp(T, N, M, BA_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)
        # Real_results1[seed_idx] = real_exp(T, n, wifi1_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, scale=REAL_scale)
        # Real_results2[seed_idx] = real_exp(T, n, wifi2_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, scale=REAL_scale)
        # Real_results3[seed_idx] = real_exp(T, n, wifi3_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, scale=REAL_scale)
    
    for seed_idx, seed in enumerate(seeds):
        print(seed)
        p_success = 0.8
        for p_idx, p in enumerate(hub_percent):
            if p_idx == 0:
                t_npi = dates.index('03-23')
            else:
                t_npi = dates.index('03-23')
            Regular_results_hub[seed_idx][p_idx] = regular_exp(T, N, K, BETA, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)
            return
            ER_results_hub[seed_idx][p_idx] = er_exp(T, N, ER_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)
            BA_results_hub[seed_idx][p_idx] = ba_exp(T, N, M, BA_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)
            # Real_results_hub1[seed_idx][p_idx] = real_exp(T, n, wifi1_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, REAL_scale, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)
            # Real_results_hub2[seed_idx][p_idx] = real_exp(T, n, wifi2_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, REAL_scale, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)
            # Real_results_hub3[seed_idx][p_idx] = real_exp(T, n, wifi3_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, REAL_scale, npi=["hub"], t_apply_npi=t_npi, t_open_up=t_open_up, hub_p=p, p_success=p_success)

    plot_graphs(np.array([Regular_results, ER_results, BA_results, Real_results1, Real_results2, Real_results3]), 
                             np.array([Regular_results_hub, ER_results_hub, BA_results_hub, Real_results_hub1, Real_results_hub2, Real_results_hub3]),
                             dates, "Remove Hubs 0.8 reset", [dates.index('03-23')], t_open_up, hub_percent)
    
    plot_graphs(np.array([Regular_results, ER_results, BA_results, Real_results1, Real_results2, Real_results3]), 
                             np.array([Regular_results_hub, ER_results_hub, BA_results_hub, Real_results_hub1, Real_results_hub2, Real_results_hub3]),
                             dates, "Remove Hubs 0.8 reset", [dates.index('03-23')], t_open_up, [0.0] + hub_percent, plot_type="result")
    
def plot_graphs(base_seir, npi_seir, dates, npi, t_npi, t_open_up, strength, compartment="Infected", 
                    graph_names=["Regular", "ER", "BA", "wifi 1", "wifi 2", "wifi 3"], with_title=True, plot_type="difference"):

    npi_mean_lst = []
    ci_lst = []

    if plot_type == "difference":
        # base_mean = np.average(base_seir, axis=1)
        for idx in range(npi_seir.shape[0]):
            npi_results = compute_difference(base_seir[idx], npi_seir[idx])
            npi_mean, ci = compute_mean_confidence_interval(npi_results)
            npi_mean_lst.append(npi_mean)
            ci_lst.append(ci)
        npi = npi + "_difference"
    elif plot_type == "result":
        base_seir = np.expand_dims(base_seir, axis=2)
        for idx in range(npi_seir.shape[0]):
            # print(base_seir.shape, npi_seir.shape)
            npi_results = np.concatenate([base_seir[idx], npi_seir[idx]], axis=1)
            npi_mean, ci = compute_mean_confidence_interval(npi_results)
            npi_mean_lst.append(npi_mean)
            ci_lst.append(ci)
        npi = npi + "_result"

    plot_open_up(dates, np.array(npi_mean_lst), compartment, npi, strength, graph_names, t_npi, t_open_up, np.array(ci_lst), with_title)
    
def base_exp(e_0, i_0, r_0, beta, n, sigma, symptomatic_rate, gamma, t):
    s_0 = n - e_0 - i_0 - r_0
    seir_0 = np.array([s_0, e_0, i_0 * symptomatic_rate, i_0 * (1 - symptomatic_rate), r_0])
    model2 = ClassicSEIR(np.array([n]), seir_0, beta=beta, sigma=sigma, symptomatic_rate=symptomatic_rate, gamma=gamma,
                         num_days=t)
    base_seir, base_pop = model2.run()
    base = compute_global_seir(base_seir, base_pop)
    return base

def real_exp(t, n, G, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed, scale=1, npi=None, t_apply_npi=None, t_open_up=None, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    
    graph = G.copy()
    graph = construct_network(graph, n, sir_0)
    
    seir = np.zeros(( t, len(SEIR_COMPARTMENTS)))

    if npi is None:
        seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, scale)
    elif t_open_up is None:
        seir = network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, scale, **kwargs)
    else:
        seir = network_exp_with_open_up(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, t_open_up, scale, **kwargs)
    return seir

def regular_exp(t, n, k, beta, sigma, gamma, symptomatic_rate, sir_0, seed, npi=None, t_apply_npi=None, t_open_up=None, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.random_regular_graph(k, n)
    graph = construct_network(G, n, sir_0)
    
    transmissibility = beta / k

    seir = np.zeros(( t, len(SEIR_COMPARTMENTS)))

    if npi is None:
        seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
    elif t_open_up is None:
        seir = network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, **kwargs)
    else:
        seir = network_exp_with_open_up(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, t_open_up, **kwargs)
    # print(seir.shape)
    return seir


def er_exp(t, n, p, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed, npi=None, t_apply_npi=None, t_open_up=None, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.fast_gnp_random_graph(n, p)
    # print("er number of edges", G.number_of_edges())
    graph = construct_network(G, n, sir_0)

    seir = np.zeros(( t, len(SEIR_COMPARTMENTS)))

    if npi is None:
        seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
    elif t_open_up is None:
        seir = network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, **kwargs)
    else:
        seir = network_exp_with_open_up(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, t_open_up, **kwargs)
    return seir


def ba_exp(t, n, m, p, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed, npi=None, t_apply_npi=None, t_open_up=None, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.dual_barabasi_albert_graph(n, m1=m, m2=1, p=p)
    # print("ba number of edges", G.number_of_edges())
    graph = construct_network(G, n, sir_0)

    seir = np.zeros(( t, len(SEIR_COMPARTMENTS)))

    if npi is None:
        seir = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)
    elif t_open_up is None:
        seir = network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, **kwargs)
    else:
        seir = network_exp_with_open_up(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npi, t_apply_npi, t_open_up, **kwargs)
    return seir

def network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, scale=1):
    model1 = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                           gamma=gamma, num_days=t, scale=scale)
    seir = model1.run(graph)
    return seir

def network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npis, t_apply_npi, scale=1, no_npi_nodes=pickle.load(open("dataset/stable/contact_network/no_npi_nodes.pkl", "rb")), **kwargs):
    print("here")
    model1 = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                           gamma=gamma, num_days=t, scale=scale, no_npi_nodes=no_npi_nodes)
    seir = model1.run_npi(graph, npis, t_apply_npi, **kwargs)
    return seir

def network_exp_with_open_up(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, npis, t_apply_npi, t_open_up, scale=1, no_npi_nodes=pickle.load(open("dataset/stable/contact_network/no_npi_nodes.pkl", "rb")), **kwargs):
    model1 = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                           gamma=gamma, num_days=t, scale=scale, no_npi_nodes=no_npi_nodes)
    seir = model1.run_open_up_with_readjust(graph, npis, t_apply_npi, t_open_up, recovered=277, infected=9, **kwargs)
    return seir

if __name__ == "__main__":
    start = time.time()
    open_up_with_reset()
    print("Time: ", time.time() - start)