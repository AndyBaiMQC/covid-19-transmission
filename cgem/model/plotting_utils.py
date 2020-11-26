import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import collections
import matplotlib.cm as cm

from scipy.ndimage import gaussian_filter1d

from Constants import SIR_COMPARTMENTS, MOBILITY_TYPE_SYN, NETWORK_TYPES, SEIR_COMPARTMENTS, \
    COVID_MODELS, COVID_MODEL_NAMES
from utils import get_sir_by_scope

def plot(x, y, labels, filename, ci, title=None):
    # colors = list(mcd.TABLEAU_COLORS)
    colors = list(mcd.BASE_COLORS)

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y[idx, :], sigma=2), colors[idx % len(colors)], alpha=0.5, lw=2, label=label)
        ax.fill_between(x, (y[idx, :]-ci[idx, :]), (y[idx, :]+ci[idx, :]), color='k', alpha=.1)
    ax.set_xlabel('Dates', fontsize=18)
    ax.set_ylabel('Percentage', fontsize=18)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=30)
    ax.tick_params(labelsize=18)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.grid(b=False)
    legend = ax.legend(fontsize=18)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig(filename)

def plot_for_npi(x, y, labels, filename, t_npi, ci, title=None):
    # colors = list(mcd.BASE_COLORS)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', "tab:brown"]

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y[idx, :], sigma=2), color=colors[idx % len(colors)], alpha=0.5, lw=4, label=label)
        ax.fill_between(x, (y[idx, :]-ci[idx, :]), (y[idx, :]+ci[idx, :]), color='k', alpha=.1)
    if type(t_npi) is list:
        for t in t_npi:
            ax.axvline(x=x[t], linestyle='dashed', color='k')
    else:
        ax.axvline(x=x[t_npi], linestyle='dashed', color='k')
    ax.set_xlabel('Dates', fontsize=40)
    ax.set_ylabel('Infected', fontsize=40)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=90)
    ax.tick_params(labelsize=35)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.grid(b=False)
    legend = ax.legend(fontsize=25)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        ax.set_title(title, fontsize=24)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close('all')

def plot_for_npi2(x, y, labels, filename, t_npi, ci, title=None):
    colors = plt.get_cmap('Blues') (np.linspace(0, 1, 1.5*len(y)))

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y[idx, :], sigma=2), color=colors[-1*(idx+1)], alpha=0.5, lw=4, label=label)
        ax.fill_between(x, (y[idx, :]-ci[idx, :]), (y[idx, :]+ci[idx, :]), color='k', alpha=.1)
    if type(t_npi) is list:
        for t in t_npi:
            ax.axvline(x=x[t], linestyle='dashed', color='k')
    else:
        ax.axvline(x=x[t_npi], linestyle='dashed', color='k')
    ax.set_xlabel('Dates', fontsize=40)
    ax.set_ylabel('Infected', fontsize=40)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=90)
    ax.tick_params(labelsize=40)
    ax.grid(b=False)
    legend = ax.legend(fontsize=35)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        ax.set_title(title, fontsize=24)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close('all')

def plot_for_open_up(x, y, labels, filename, t_npi, t_open_up, ci, title=None):
    # colors = list(mcd.BASE_COLORS)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', "tab:brown"]

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y[idx, :], sigma=2), color=colors[idx % len(colors)], alpha=0.5, lw=4, label=label)
        ax.fill_between(x, (y[idx, :]-ci[idx, :]), (y[idx, :]+ci[idx, :]), color='k', alpha=.1)
    if type(t_npi) is list:
        for t in t_npi:
            ax.axvline(x=x[t], linestyle='dashed', color='k')
    else:
        ax.axvline(x=x[t_npi], linestyle='dashed', color='k')
    
    ax.axvline(x=x[t_open_up], linestyle='dashed', color='k')

    ax.set_xlabel('Dates', fontsize=25)
    ax.set_ylabel('Percentage', fontsize=25)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=90)
    ax.tick_params(labelsize=25)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.grid(b=False)
    legend = ax.legend(fontsize=25)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        ax.set_title(title, fontsize=25)
    plt.tight_layout()
    plt.show()
    # plt.savefig(filename)
    plt.close('all')

def plot_for_open_up2(x, y, labels, filename, t_npi, t_open_up, ci, title=None):
    colors = plt.get_cmap('Blues') (np.linspace(0, 1, 1.5*len(y)))

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y[idx, :], sigma=2), color=colors[-1*(idx+1)], alpha=0.5, lw=4, label=label)
        ax.fill_between(x, (y[idx, :]-ci[idx, :]), (y[idx, :]+ci[idx, :]), color='k', alpha=.1)
    if type(t_npi) is list:
        for t in t_npi:
            ax.axvline(x=x[t], linestyle='dashed', color='k')
    else:
        ax.axvline(x=x[t_npi], linestyle='dashed', color='k')
    
    ax.axvline(x=x[t_open_up], linestyle='dashed', color='k')

    ax.set_xlabel('Dates', fontsize=18)
    ax.set_ylabel('Percentage', fontsize=18)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=90)
    ax.tick_params(labelsize=18)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.grid(b=False)
    legend = ax.legend(fontsize=18)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        ax.set_title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close('all')

def plot_for_contact_network(x, y_all, y_symptomatic, labels, filename, t_npi, ci, title=None):
    # colors = list(mcd.BASE_COLORS)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', "tab:brown"]

    fig = plt.figure(facecolor='w', figsize=(12, 8))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        ax.plot(x, gaussian_filter1d(y_all[idx, :], sigma=2), color=colors[idx % len(colors)], alpha=0.5, lw=4, label=label)
        if idx == len(labels)-1:
            ax.plot(x, gaussian_filter1d(y_symptomatic[idx, :], sigma=2), linestyle='dashed', color=colors[idx % len(colors)], alpha=0.5, lw=4, label=label)
        ax.fill_between(x, (y_all[idx, :]-ci[idx, :]), (y_all[idx, :]+ci[idx, :]), color='k', alpha=.1)
    if type(t_npi) is list:
        for t in t_npi:
            ax.axvline(x=x[t], linestyle='dashed', color='k')
    else:
        ax.axvline(x=x[t_npi], linestyle='dashed', color='k')
    ax.set_xlabel('Dates', fontsize=18)
    ax.set_ylabel('Percentage', fontsize=18)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 7))
    ax.xaxis.set_tick_params(rotation=90)
    ax.tick_params(labelsize=18)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.grid(b=False)
    legend = ax.legend(fontsize=18)
    legend.get_frame().set_alpha(0.5)
    if title is not None:
        ax.set_title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close('all')

def plot_synthetic_results(T, results, model):
    for idx in range(results.shape[0]):
        result = results[idx].T
        graph_name = "graphs/synthetic/%s_mobility_%s.png" % (MOBILITY_TYPE_SYN[idx], model)
        plot(np.arange(0, T), result, SIR_COMPARTMENTS, graph_name)
    for idx in range((results.shape[2])):
        graph_name = "graphs/synthetic/%s.png" % SIR_COMPARTMENTS[idx]
        plot(np.arange(0, T), results[:, :, idx], MOBILITY_TYPE_SYN, graph_name)


def plot_synthetic1_results(T, results, model):
    for network_idx in range(results.shape[0]):
        title = "%s_%s" % (NETWORK_TYPES[network_idx], model)
        graph_name = "graphs/synthetic1/%s.png" % title
        plot(np.arange(0, T), results[network_idx, :, :].T, SEIR_COMPARTMENTS, graph_name, title)
    for compartment_idx in range((results.shape[2])):
        title = SEIR_COMPARTMENTS[compartment_idx]
        graph_name = "graphs/synthetic1/%s.png" % title
        plot(np.arange(0, T), results[:, :, compartment_idx], NETWORK_TYPES, graph_name, title)


def plot_by_model_and_compartment(exp, dates, sir, population, labels, model, compartment):
    model_idx = COVID_MODEL_NAMES.index(model)
    compartment_idx = SIR_COMPARTMENTS.index(compartment)
    target_indices = exp.get_target_indices()
    sir_percentage = sir[:, :, target_indices, :] / np.expand_dims(population[:, :, target_indices], 3)
    graph_name = "graphs/covid/%s_All_Prov_%s.png" % (compartment, model)
    title = "%s %s SIR vs Dates" % (compartment, model)
    plot(dates, sir_percentage[model_idx, :, :, compartment_idx].T, labels, graph_name, title)


def plot_by_model(exp, dates, sir, population, model, scope):
    model_idx = COVID_MODEL_NAMES.index(model)
    sir_percentage = get_sir_by_scope(exp, sir, population, scope)
    graph_name = "graphs/covid/%s_%s.png" % (scope, model)
    title = "%s %s SIR vs Dates" % (scope, model)
    plot(dates, sir_percentage[model_idx, :, 1:].T, SIR_COMPARTMENTS[1:], graph_name, title)


def plot_by_compartment(exp, dates, sir, population, compartment, scope):
    compartment_idx = SIR_COMPARTMENTS.index(compartment)
    sir_percentage = get_sir_by_scope(exp, sir, population, scope)
    graph_name = "graphs/covid/%s_%s.png" % (scope, compartment)
    title = "%s %s vs Dates" % (scope, compartment)
    plot(dates, sir_percentage[:, :, compartment_idx], COVID_MODELS, graph_name, title)

def plot_synthetic_by_compartment(dates, sir, compartment, names, ci, t_match=0):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    graph_name = "graphs/synthetic1/%s.png" % (compartment)
    title = "%s vs Dates" % (compartment)
    if t_match == 0:
        plot(dates, sir[:, :, compartment_idx], names, graph_name, ci[:, :, compartment_idx], title)
    else:
        plot_for_npi(dates, sir[:, :, compartment_idx], names, graph_name, t_match, ci[:, :, compartment_idx], title)

def plot_contact_network_by_compartment(dates, sir_all, sir_symptomatic, compartment, names, ci, t_match=0):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    graph_name = "graphs/synthetic1/npi/all_%s.png" % (compartment)
    title = "%s vs Dates" % (compartment)
    if t_match == 0:
        plot(dates, sir_all, names, graph_name, ci, title)
    else:
        # plot_for_contact_network(dates, sir_all, sir_symptomatic, names, graph_name, t_match, ci, title)
        plot_for_npi(dates, sir_all, names, graph_name, t_match, ci, title)

def plot_npi_by_model(dates, sir, compartment, npi, percent, model, t_npi, ci, with_title=False):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    graph_name = "graphs/synthetic1/npi/%s_%s.png" % (npi, model)
    title = None
    if with_title:
        title = "%s vs Dates (strength of %s on %s)" % (compartment, npi, model)
    names = []
    if "Remove Hubs" in npi:
        names = [str(p*100) + "% H" for p in percent]
    elif "Social distancing" in npi:
        names = [str(p*100) + "% SD" for p in percent]
    elif "Quarantine" in npi:
        names = [str(p*100) + "% Q" for p in percent]
    elif "Wearing mask" in npi:
        names = percent
    plot_for_npi2(dates,sir[:,:, compartment_idx], names, graph_name, t_npi, ci[:,:, compartment_idx], title)

def plot_npi_by_strength(dates, sir, compartment, npi, strengths, model_names, t_npi, ci, with_title=False):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    graph_name = ""
    if type(strengths) == str:
        graph_name = "graphs/synthetic1/npi/%s_%s.png" % (npi, strengths)
    else:
        graph_name = "graphs/synthetic1/npi/%s_%f.png" % (npi, strengths)
    title = None
    if with_title:
        title = "%s vs Dates (All models with %s at %f)" % (compartment, npi, strengths)
    names = model_names # FIXME change naming
    plot_for_npi(dates, sir[:,:, compartment_idx], names, graph_name, t_npi, ci[:,:, compartment_idx], title)

def plot_npi(dates, sir, compartment, npi, strengths, model_names, t_npi, ci, with_title=False):
    for s_idx, s in enumerate(strengths):
        plot_npi_by_strength(dates, sir[:,s_idx,:,:], compartment, npi, s, model_names, t_npi, ci[:,s_idx,:,:], with_title)
    for model_idx, model in enumerate(model_names):
        plot_npi_by_model(dates, sir[model_idx,:,:,:], compartment, npi, strengths, model, t_npi, ci[model_idx,:,:,:], with_title)

def plot_open_up_by_model(dates, sir, compartment, npi, percent, model, t_npi, t_open_up, ci, with_title=False):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    graph_name = "graphs/synthetic1/open_up/%s_%s.png" % (npi, model)
    title = None
    if with_title:
        title = "%s vs Dates (strength of %s on %s)" % (compartment, npi, model)
    names = percent # FIXME change naming
    plot_for_open_up2(dates,sir[:,:, compartment_idx], names, graph_name, t_npi, t_open_up, ci[:,:, compartment_idx], title)

def plot_open_up_by_strength(dates, sir, compartment, npi, strengths, model_names, t_npi, t_open_up, ci, with_title=False):
    compartment_idx = SEIR_COMPARTMENTS.index(compartment)
    if type(strengths) == str:
        graph_name = "graphs/synthetic1/open_up/%s_%s.png" % (npi, strengths)
    else:
        graph_name = "graphs/synthetic1/open_up/%s_%f.png" % (npi, strengths)
    title = None
    if with_title:
        if type(strengths) == str:
            title = "%s vs Dates (All models with %s at %s)" % (compartment, npi, strengths)
        else:
            title = "%s vs Dates (All models with %s at %f)" % (compartment, npi, strengths)
    names = model_names # FIXME change naming
    plot_for_open_up(dates, sir[:,:, compartment_idx], names, graph_name, t_npi, t_open_up, ci[:,:, compartment_idx], title)

def plot_open_up(dates, sir, compartment, npi, strengths, model_names, t_npi, t_open_up, ci, with_title=False):
    for s_idx, s in enumerate(strengths):
        plot_open_up_by_strength(dates, sir[:,s_idx,:,:], compartment, npi, s, model_names, t_npi, t_open_up, ci[:,s_idx,:,:], with_title)
    for model_idx, model in enumerate(model_names):
        plot_open_up_by_model(dates, sir[model_idx,:,:,:], compartment, npi, strengths, model, t_npi, t_open_up, ci[model_idx,:,:,:], with_title)

def plot_degree_dist(graph, log_log=True, title=None, bin=False):
    degree_sequence = sorted([d for n, d in graph.degree()])  # degree sequence

    if bin:
        hist, bin_edges = np.histogram(degree_sequence, bins=1000)
        x = bin_edges[0:-1]
        x_min = np.min(x[np.nonzero(x)])
        x_max = np.max(x)
        y = hist/len(degree_sequence)
        y_min = np.min(y[np.nonzero(y)])

        if log_log:
            plt.yscale('log')
            plt.xscale('log')
        plt.scatter(x, y, s=5)
        if title is not None:
            plt.title('Degree Distribution of ' + title)
        plt.xlabel('Degree k')
        plt.ylabel('$P_k$')

        plt.xlim(x_min/10, x_max*10)
        plt.ylim(y_min/10, np.max(y)*10)
        plt.tight_layout()
        plt.show() # TODO save location
        plt.close()
    else:
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        deg = np.array(deg)
        cnt = np.array(cnt)/len(degree_sequence)

        if log_log:
            plt.yscale('log')
            plt.xscale('log')

        plt.scatter(deg, cnt, s=10)
        if title is not None:
            plt.title('Degree Distribution of ' + title)
        plt.xlabel('Degree k', fontsize=18)
        plt.ylabel('$P_k$', fontsize=18)

        plt.tight_layout()
        plt.show() # TODO save location
        plt.close()