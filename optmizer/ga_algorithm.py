import numpy as np
import sys


class Individual:
    def __init__(self, cs, selected_clients, mutation_rate=0.01, idGen=0):
        self.rng_individual = np.random.default_rng(np.random.randint(0, int(1e6)))
        self.cs = cs
        self.selected_clients = selected_clients
        self.mutation_rate = mutation_rate
        self.idGen = idGen
        self.chrom = self.rng_individual.permutation(len(selected_clients))

    def evaluate(self):
        value = 0
        k = len(self.cs.tm.user_power) - 1
        for i in range(len(self.chrom)):
            if self.cs.W[self.selected_clients[i], self.chrom[i], k] >= 0:
                value = value + self.cs.W[self.selected_clients[i], self.chrom[i], k]
            else:
                value = value - 0
        return value

    def __evaluate__(self):
        value = 0
        cont = 0
        gama = 1.2

        k = len(self.cs.tm.user_power) - 1
        for i in range(len(self.chrom)):
            if self.cs.final_q[self.selected_clients[i], self.chrom[i], k] != 1:
                value = value + self.cs.tm.q[self.selected_clients[i], self.chrom[i], k]
                cont = cont + 1

        if cont > 0:
            value = value + gama * cont * (-1)
        else:
            value = 100

        return value

    def crossover(self, another):
        cut = self.rng_individual.integers(1, len(self.chrom))

        child_1 = [None] * len(self.chrom)
        child_2 = [None] * len(self.chrom)

        for i in range(cut):
            child_1[i] = self.chrom[i]

        for i in range(cut):
            child_2[i] = another.chrom[i]

        pos = cut
        for gene in another.chrom:
            if gene not in child_1:
                child_1[pos] = gene
                pos += 1

        pos = cut
        for gene in self.chrom:
            if gene not in child_2:
                child_2[pos] = gene
                pos += 1

        child_1 = np.array(child_1)
        child_2 = np.array(child_2)

        children = [Individual(self.cs, self.selected_clients, self.idGen + 1),
                    Individual(self.cs, self.selected_clients, self.idGen + 1)]

        children[0].chrom = child_1
        children[1].chrom = child_2
        return children

    def mutation(self):
        if self.rng_individual.random() < self.mutation_rate:
            i = np.random.randint(0, len(self.chrom))
            j = np.random.randint(0, len(self.chrom))
            self.chrom[i], self.chrom[j] = self.chrom[j], self.chrom[i]
        return self


class GA:
    def __init__(self, cs, selected_clients, nGen, population_size, mutation_rate):
        self.rng_ga = np.random.default_rng(np.random.randint(0, int(1e6)))
        self.cs = cs
        self.population_size = population_size
        self.population_list = []
        self.evaluate_list = []
        self.idGen = 0
        self.best_ind_gen = 0
        self.nGen = nGen
        self.best_individual = None
        self.mutation_rate = mutation_rate
        self.selected_clients = selected_clients

        self.init_population()

    def init_population(self):
        for i in range(self.population_size):
            self.population_list.append(Individual(self.cs, self.selected_clients, self.mutation_rate))

    def sort_population(self):
        self.population_list = sorted(self.population_list, key=lambda ind: ind.evaluate(), reverse=True)
        # self.population_list = sorted(self.population_list, key=lambda ind: ind.evaluate())

    def print_population(self):
        for i in range(self.population_size):
            print(f"{i}: {self.population_list[i].chrom} >>> {self.population_list[i].evaluate()}")

    def sum_evaluation(self):
        value = 0
        for ind in self.population_list:
            value = value + ind.evaluate()
        return value

    def parent_selection(self, sum_evaluation):
        parent = -1
        random_value = self.rng_ga.random() * sum_evaluation
        sum_value = 0
        i = 0
        while i < len(self.population_list) and sum_value < random_value:
            sum_value += self.population_list[i].evaluate()
            parent += 1
            i += 1
        return parent

    def solver(self):
        self.init_population()
        self.sort_population()
        self.best_individual = self.population_list[0]

        for self.idGen in range(self.nGen):
            sum_value = self.sum_evaluation()

            new_population = []
            for j in range(0, self.population_size, 2):
                father_1 = self.parent_selection(sum_value)
                father_2 = self.parent_selection(sum_value)

                children = self.population_list[father_1].crossover(self.population_list[father_2])
                new_population.append(children[0].mutation())
                new_population.append(children[1].mutation())

            self.population_list = list(new_population)
            self.sort_population()
            self.best_ind_gen = self.population_list[0]
            self.evaluate_list.append(self.best_ind_gen.evaluate())

            if self.best_ind_gen.evaluate() > self.best_individual.evaluate():
                self.best_individual = self.best_ind_gen

        return self.best_individual


class GA_Algorithm_Opt:
    def __init__(self, communication_strategy):
        self.cs = communication_strategy

    def opt(self, selected_clients):

        population_size = 100
        mutation_rate = 0.01
        nGen = 1000
        ga = GA(self.cs, selected_clients, nGen, population_size, mutation_rate)
        result = ga.solver()

        print(result.chrom)

        _selected_clients = []
        _rb_allocation = []
        for idx in range(len(result.chrom)):
            _selected_clients.append(selected_clients[idx])
            _rb_allocation.append(result.chrom[idx])

        C = np.zeros((len(_selected_clients), self.cs.tm.rb_number))
        best_k = np.full((len(_selected_clients), self.cs.tm.rb_number), -1, dtype=int)

        for i in range(len(_selected_clients)):
            idx = _selected_clients[i]
            for j in range(self.cs.tm.rb_number):
                # Lists viable powers for (i, j)
                feasible_powers = []
                for k in range(len(self.cs.tm.user_power)):
                    if ((self.cs.tm.user_upload_energy[idx, j, k] <= self.cs.energy_requirement and
                         self.cs.tm.user_delay[idx, j, k] <= self.cs.delay_requirement) and
                            self.cs.tm.q[idx, j, k] <= self.cs.error_rate_requirement):
                        feasible_powers.append(k)

                if not feasible_powers:
                    # No power meets the restrictions
                    C[i, j] = 999.0
                    best_k[i, j] = -1
                else:
                    # Choose the lowest power (Eq. (22)) => min user_p[k]
                    chosen_p = 9999.0
                    chosen_k = -1
                    for k2 in feasible_powers:
                        if self.cs.tm.user_power[k2] < chosen_p:
                            chosen_p = self.cs.tm.user_power[k2]
                            chosen_k = k2

                    C[i, j] = self.cs.tm.q[idx, j, chosen_k]
                    best_k[i, j] = chosen_k

        assigned_users_list = []
        not_assigned_list = []
        rb_allocation_list = []
        rb_power_list = []

        for idx in range(len(_selected_clients)):
            if best_k[idx, _rb_allocation[idx]] >= 0:
                assigned_users_list.append(_selected_clients[idx])
                rb_allocation_list.append(_rb_allocation[idx])

                idx_min_power = best_k[idx, _rb_allocation[idx]]
                indices_list = list(range(idx_min_power, len(self.cs.tm.user_power)))
                idx_power = indices_list[int(round(self.cs.lmbda * (len(indices_list) - 1)))]
                rb_power_list.append(idx_power)
            else:
                not_assigned_list.append(selected_clients[idx])

        return assigned_users_list.copy(), rb_allocation_list.copy(), rb_power_list.copy()
