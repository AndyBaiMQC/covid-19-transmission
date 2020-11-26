import numpy as np

from ClassicSIR import ClassicSIR


class DHSIRModel(ClassicSIR):

    def __init__(self, population_0, sir_0, num_days, beta_vec, gamma):
        super().__init__(population_0, sir_0, num_days, beta_vec, gamma)
        self.mobility_matrix = None

    def update_sir(self, new_infected, new_recovered, t, stochastic=False):
        delta_susceptible, delta_infected, delta_recovered = self.get_mobility_sir(t, stochastic)
        self.sir_all[t + 1, :, 0] = np.maximum(np.zeros(shape=self.num_nodes), self.sir_all[t, :, 0] - new_infected + delta_susceptible)
        self.sir_all[t + 1, :, 1] = np.maximum(np.zeros(shape=self.num_nodes), self.sir_all[t, :, 1] + new_infected - new_recovered + delta_infected)
        self.sir_all[t + 1, :, 2] = self.sir_all[t, :, 2] + new_recovered + delta_recovered
        self.population_all[t + 1] = self.sir_all[t + 1, :, 0] + self.sir_all[t + 1, :, 1] + self.sir_all[t + 1, :, 2]
        if not (self.population_all[t+1] > 0).all():
            raise RuntimeError("invalid mobility matrix")

    def get_mobility_sir(self, t, stochastic=False):
        mobility_matrix = self.mobility_matrix[t] if stochastic else self.mobility_matrix
        frac_susceptible = self.sir_all[t, :, 0] / (self.population_all[t] + self.epsilon)
        frac_infected = self.sir_all[t, :, 1] / (self.population_all[t] + self.epsilon)
        susceptible_matrix = np.round(mobility_matrix.transpose() * frac_susceptible)
        infected_matrix = np.round(mobility_matrix.transpose() * frac_infected)
        recovered_matrix = mobility_matrix.transpose() - infected_matrix - susceptible_matrix
        delta_susceptible = susceptible_matrix.sum(axis=1) - susceptible_matrix.sum(axis=0)
        delta_infected = infected_matrix.sum(axis=1) - infected_matrix.sum(axis=0)
        delta_recovered = recovered_matrix.sum(axis=1) - recovered_matrix.sum(axis=0)
        return delta_susceptible, delta_infected, delta_recovered

    def reset(self, mobility_matrix=None, seed=0):
        super().reset(seed=seed)
        self.mobility_matrix = mobility_matrix


