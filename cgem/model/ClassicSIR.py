import numpy as np


class ClassicSIR:

    def __init__(self, population_0, sir_0, num_days, beta_vec, gamma):
        self.population_0 = population_0
        self.sir_0 = sir_0
        self.num_days = num_days
        self.beta_vec = beta_vec
        self.gamma = gamma
        self.num_nodes = len(self.population_0)
        self.sir_all = None
        self.population_all = None
        self.epsilon = 1e-1

    def run(self, mobility_matrix=None, stochastic=False, seed=0):
        self.reset(mobility_matrix, seed)
        for t in range(self.num_days - 1):
            new_infected = self.get_new_infected(t, stochastic)
            new_recovered = self.get_new_recovered(t)
            self.update_sir(new_infected, new_recovered, t, stochastic)
        return self.sir_all, self.population_all

    def get_new_infected(self, t, stochastic=False):
        susceptible = self.sir_all[t, :, 0]
        new_infected = np.round(self.beta_vec * susceptible * self.sir_all[t, :, 1] / (self.population_all[t] + self.epsilon))
        return new_infected

    def get_new_recovered(self, t):
        infected = self.sir_all[t, :, 1]
        new_recovered = np.around(np.random.exponential(scale=self.gamma, size=self.num_nodes) * infected)
        return new_recovered

    def update_sir(self, new_infected, new_recovered, t, stochastic=False):
        self.sir_all[t + 1, :, 0] = np.maximum(np.zeros(shape=self.num_nodes), self.sir_all[t, :, 0] - new_infected)
        self.sir_all[t + 1, :, 1] = np.maximum(np.zeros(shape=self.num_nodes), self.sir_all[t, :, 1] + new_infected - new_recovered)
        self.sir_all[t + 1, :, 2] = self.sir_all[t, :, 2] + new_recovered
        self.population_all[t + 1] = self.population_all[t]

    def reset(self, mobility_matrix=None, seed=0):
        np.random.seed(seed)
        self.sir_all = np.zeros((self.num_days, self.population_0.shape[0], 3))
        self.sir_all[0] = self.sir_0.copy()
        self.population_all = np.zeros((self.num_days, self.population_0.shape[0]))
        self.population_all[0] = self.population_0.copy()
