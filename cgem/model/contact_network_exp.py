import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd

from ClassicSEIR import ClassicSEIR
from Constants import SEIR_COMPARTMENTS
from NetworkedSEIR import NetworkedSEIR
from plotting_utils import plot_contact_network_by_compartment, plot_degree_dist, plot_npi
from utils import compute_global_seir, compute_mean_confidence_interval
from synthetic_exp1 import construct_network
from npi_exp import base_exp, real_exp, regular_exp, er_exp, ba_exp, plot_npi_graphs

def contact_network_exp():
    tr_sir = pickle.load(open("dataset/stable/contact_network/seir_mtl_100_updated.pkl", "rb"))

    wifi1_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2004-08-27_2006-11-30_GCC.edgelist", nodetype=int)
    wifi2_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2007-07-01_2008-02-26_GCC.edgelist", nodetype=int)
    wifi3_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2009-12-02_2010-03-08_GCC.edgelist", nodetype=int)

    # timestamp
    T = tr_sir.shape[0]

    base_e_0 = tr_sir[0, 2] * 2
    base_i_0 = tr_sir[0, 2]
    base_r_0 = tr_sir[0, 3]

    import datetime as dt
    dates = [dt.datetime.strftime(dt.datetime.strptime(d, '%Y-%m-%d'), "%m-%d") for d in pickle.load(open("model/mtl_dates.pkl", 'rb'))]
    print(dates)

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

    Regular_results_all = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    ER_results_all = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    BA_results_all = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results_all1 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results_all2 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results_all3 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))

    K = 21 # for regular graph
    ER_P = 0.0014  # for ER graph
    M = 10  # for BA graph
    BA_P = 1 # for BA graph
    REAL_scale = 1
    tr_sir = tr_sir / n
    TRANSMISSIBILITY = BETA/K

    e_0 = int(np.round(base_e_0 / 100))
    i_0 = int(np.round(base_i_0 / 100))
    i_s_0 = int(np.round(i_0 * SYMPTOMATIC_RATE))
    i_a_0 = i_0 - i_s_0
    sir_0 = [e_0, i_s_0, i_a_0]

    # [Regular_results_all, ER_results_all, BA_results_all, Real_results_all1, Real_results_all2, Real_results_all3] = pickle.load(open("output/npi/all_0.95_seiir.pkl", "rb"))
    # [Regular_results, ER_results, BA_results, Real_results1, Real_results2, Real_results3] = pickle.load(open("output/npi/no_npi.pkl", "rb"))

    t_npi = [dates.index('03-23'), dates.index('03-23'), 
             dates.index('03-23'), dates.index('04-06')]
    npi = ["quarantine", "distance", "remove hub", "mask"]
    hub_p = 0.1
    p_success=0.8
    distance_p=0.5
    quarantine_p=0.5

    start_time = time.time()
    for seed_idx, seed in enumerate(seeds):
        print(seed)
        Regular_results_all[seed_idx] = regular_exp(T, N, K, BETA, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        ER_results_all[seed_idx] = er_exp(T, N, ER_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        BA_results_all[seed_idx] = ba_exp(T, N, M, BA_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        Real_results_all1[seed_idx] = real_exp(T, N, wifi1_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        Real_results_all2[seed_idx] = real_exp(T, N, wifi2_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        Real_results_all3[seed_idx] = real_exp(T, N, wifi3_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed, npi=npi, t_apply_npi=t_npi, hub_p=hub_p, p_success=p_success, distance_p=distance_p, quarantine_p=quarantine_p)
        pickle.dump([Regular_results_all, ER_results_all, BA_results_all, Real_results_all1, Real_results_all2, Real_results_all3], open("output/npi/all_" + str(quarantine_p) + "_seiir.pkl", "wb"))
    print("all npi time: ", time.time()-start_time)
    pickle.dump([Regular_results_all, ER_results_all, BA_results_all, Real_results_all1, Real_results_all2, Real_results_all3], open("output/npi/all_" + str(quarantine_p) + "_seiir.pkl", "wb"))

    plot_graphs(np.array([Regular_results_all, ER_results_all, BA_results_all, Real_results_all1, Real_results_all2, Real_results_all3]), tr_sir, dates[len(dates) - tr_sir.shape[0]:], t_match=t_npi)

    # plot_npi_graphs(np.array([Regular_results, ER_results, BA_results, Real_results1, Real_results2, Real_results3]), 
    #                          np.array([Regular_results_all, ER_results_all, BA_results_all, Real_results_all1, Real_results_all2, Real_results_all3]),
    #                          dates, "Wearing mask", t_npi, ["all npi"], plot_type="result")


def plot_graphs(seir, real, dates, compartment="Infected", names=["Regular", "ER", "BA", "wifi 1", "wifi 2", "wifi 3", "Real"], t_match=0):
    mean_lst = []
    ci_lst = []
    seir = np.expand_dims(seir, axis=2)
    for idx in range(seir.shape[0]):
        mean, ci = compute_mean_confidence_interval(seir[idx])
        mean = np.squeeze(mean, axis=0)
        ci = np.squeeze(ci, axis=0)
        mean_lst.append(mean)
        ci_lst.append(ci)
    print(real.shape)
    # mean_lst = np.concatenate((np.array(mean_lst), real[None]), axis=0)
    real_i = real[:,2]
    i_s = np.concatenate((np.array(mean_lst)[:,:,2], (real_i/0.6)[None]), axis=0)
    i_a = np.concatenate((np.array(mean_lst)[:,:,3], real_i[None]), axis=0)
    ci_lst = np.concatenate((np.array(ci_lst), np.zeros(ci_lst[0].shape)[None]), axis=0)
    # print(i_s.shape, i_a.shape, ci_lst.shape)
    plot_contact_network_by_compartment(dates, i_s, i_a, compartment, names, ci_lst[:,:,2], t_match)

if __name__ == "__main__":
    contact_network_exp()