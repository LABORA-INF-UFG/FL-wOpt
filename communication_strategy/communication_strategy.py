import numpy as np
import sys
import pulp as pl
import re
from optmizer.ga_algorithm import GA_Algorithm_Opt
from optmizer.hungarian_algorithm import Hungarian_Algorithm_Opt
from optmizer.milp_optmizer import Milp_Opt
from optmizer.sinr_optmizer import SINR_Opt


class Communication_Strategy:

    def __init__(self, transmission_model, min_fit_clients, clients_number_data_samples,
                 clients_value_based_emd, emd_mean, clients_emd, optmizer_type,
                 delay_requirement=0.2, energy_requirement=0.0025,
                 error_rate_requirement=0.3, lmbda=260):

        self.tm = transmission_model
        self.min_fit_clients = min_fit_clients
        self.delay_requirement = delay_requirement
        self.energy_requirement = energy_requirement
        self.error_rate_requirement = error_rate_requirement
        self.lmbda = lmbda

        self.clients_number_data_samples = clients_number_data_samples
        self.clients_value_based_emd = clients_value_based_emd
        self.emd_mean = emd_mean
        self.clients_emd = clients_emd

        self.count_selected_clients = 0
        self.selected_clients = np.array([])
        self.rb_allocation = np.array([])
        self.user_power_allocation = np.array([])

        self.success_uploads = []
        self.error_uploads = []

        self.W = np.array([])
        self.final_q = np.array([])
        self.optmizer_type = optmizer_type
        self.optmizer = None

        self.rng_upload_status = np.random.default_rng(np.random.randint(0, int(1e6)))
        self.rng_random_user = np.random.default_rng(np.random.randint(0, int(1e6)))
        self.rng_random_rb = np.random.default_rng(np.random.randint(0, int(1e6)))

        self.round_costs_list = {
            'total_training': [],
            'total_uploads': [],
            'total_error_uploads': [],
            'energy_success': [],
            'energy_error': [],
            'total_energy': [],
            'delay': [],
            'round_delay_upload_success': [],
            'power': []
        }

        self.init()

    def init(self):
        self.compute_transmission_probability_matrix()
        self.compute_final_q()
        if self.optmizer_type == 'MILP':
            self.optmizer = Milp_Opt(self)
        elif self.optmizer_type == 'GA':
            self.optmizer = GA_Algorithm_Opt(self)
        elif self.optmizer_type == 'HUN':
            self.optmizer = Hungarian_Algorithm_Opt(self)
        else:
            self.optmizer = SINR_Opt(self)

    def greater_data_user_selection(self, factor, k):
        selected_clients = self.rng_random_user.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]

        data_samples_list = np.array(self.clients_number_data_samples)[selected_clients]
        pos_list = np.arange(len(data_samples_list))
        print(data_samples_list)

        combined_data = list(zip(data_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        distance_list, pos_list = zip(*sorted_data)
        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def smaller_emd(self, factor, k):
        selected_clients = self.rng_random_user.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        emd_samples_list = np.array(self.clients_value_based_emd)[selected_clients]
        pos_list = np.arange(len(emd_samples_list))

        combined_data = list(zip(emd_samples_list, pos_list))
        sorted_data = sorted(combined_data, key=lambda x: x[0])
        emd_list, pos_list = zip(*sorted_data)

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def smaller_emd_(self, factor, k):
        selected_clients = self.rng_random_user.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        emd_samples_list = np.array(self.clients_value_based_emd)[selected_clients]
        pos_list = np.arange(len(emd_samples_list))

        combined_data = list(zip(emd_samples_list, pos_list))
        sorted_data = sorted(combined_data, key=lambda x: x[0])
        emd_list, pos_list = zip(*sorted_data)

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def greater_loss_user_selection(self, clients_loss_list, factor, k):
        selected_clients = self.rng_random_user.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        loss_samples_list = np.array(clients_loss_list)[selected_clients]
        pos_list = np.arange(len(loss_samples_list))

        combined_data = list(zip(loss_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        loss_list, pos_list = zip(*sorted_data)
        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def random_user_selection(self, k):
        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[self.rng_random_user.permutation(self.tm.user_number)[:k]] = 1
        self.selected_clients = np.where(self.selected_clients > 0)[0]
        self.count_selected_clients = len(self.selected_clients)

    def random_rb_allocation(self):
        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        self.rb_allocation[self.rng_random_rb.permutation(self.tm.rb_number)[:self.min_fit_clients]] = 1
        self.rb_allocation = self.rng_random_rb.permutation(np.where(self.rb_allocation > 0)[0])

    def fixed_user_power_allocation(self):
        self.user_power_allocation = np.zeros(self.count_selected_clients).astype(int)

    def compute_transmission_probability_matrix(self):
        self.W = np.zeros((self.tm.user_number, self.tm.rb_number, len(self.tm.user_power)))
        for i in range(self.tm.user_number):
            for j in range(self.tm.rb_number):
                for k in range(len(self.tm.user_power)):
                    if (self.tm.user_delay[i, j, k] <= self.delay_requirement and
                            self.tm.user_upload_energy[i, j, k] <= self.energy_requirement and
                            self.tm.q[i, j, k] <= self.error_rate_requirement):
                        self.W[i, j, k] = 1 - self.tm.q[i, j, k]

    def compute_final_q(self):
        self.final_q = np.ones((self.tm.user_number, self.tm.rb_number, len(self.tm.user_power)))
        for i in range(self.tm.user_number):
            for j in range(self.tm.rb_number):
                for k in range(len(self.tm.user_power)):
                    if (self.tm.user_delay[i, j, k] <= self.delay_requirement and
                            self.tm.user_upload_energy[i, j, k] <= self.energy_requirement and
                            self.tm.q[i, j, k] <= self.error_rate_requirement):
                        self.final_q[i, j] = self.tm.q[i, j, k]

    def print_values(self):
        print("----------------------------------")
        print(f"selected_clients: {self.selected_clients}")
        if len(self.selected_clients) > 0:
            print(f"rb_allocation: {self.rb_allocation}")
            print(
                f"user_power_allocation: {np.array(self.tm.user_power)[self.user_power_allocation]} - Ind: {self.user_power_allocation}")
        print("----------------------------------")

    def upload_status(self):
        prob = self.rng_upload_status.random(self.count_selected_clients)

        self.success_uploads = []
        self.error_uploads = []

        for i, ue in enumerate(self.selected_clients):
            prob_w = self.W[ue, self.rb_allocation[i], self.user_power_allocation[i]]
            print(
                f"{i} - {ue} --> W: {prob_w:.4f} - P: {prob[i]:.4f} {'' if prob_w > 0 and prob_w >= prob[i] else ' - [X]'}")

            if prob_w > 0 and prob_w >= prob[i]:
                self.success_uploads.append(ue)
            else:
                self.error_uploads.append(ue)

    def round_costs(self):

        total_training = len(self.selected_clients)
        total_uploads = len(self.success_uploads)

        round_energy = 0
        round_energy_success = 0
        round_energy_error = 0
        round_delay = 0
        round_delay_upload_success = 0

        round_power = 0

        for i, ue in enumerate(self.selected_clients):
            round_power = round_power + self.tm.user_power[self.user_power_allocation[i]]
            round_energy = round_energy + self.tm.total_energy[ue, self.rb_allocation[i], self.user_power_allocation[i]]
            round_delay = (round_delay + self.tm.total_delay[ue, self.rb_allocation[i], self.user_power_allocation[i]])

            if ue in self.success_uploads:
                round_energy_success = round_energy_success + self.tm.total_energy[
                    ue, self.rb_allocation[i], self.user_power_allocation[i]]
                round_delay_upload_success = round_delay_upload_success + self.tm.user_delay[
                    ue, self.rb_allocation[i], self.user_power_allocation[i]]
            else:
                round_energy_error = round_energy_error + self.tm.total_energy[
                    ue, self.rb_allocation[i], self.user_power_allocation[i]]

        self.round_costs_list['total_training'].append(total_training)
        self.round_costs_list['total_uploads'].append(total_uploads)
        self.round_costs_list['total_error_uploads'].append(total_training - total_uploads)
        self.round_costs_list['energy_success'].append(round_energy_success)
        self.round_costs_list['energy_error'].append(round_energy_error)
        self.round_costs_list['total_energy'].append(round_energy)
        self.round_costs_list['delay'].append(round_delay)
        self.round_costs_list['round_delay_upload_success'].append(round_delay_upload_success)
        self.round_costs_list['power'].append(round_power)

    def print_round_costs(self):
        print("------------------------------------")
        print(f"total_training: {self.round_costs_list['total_training'][-1]}")
        print(
            f"total_uploads: {self.round_costs_list['total_uploads'][-1]}/{self.round_costs_list['total_training'][-1]}")
        print(
            f"total_error_uploads: {(self.round_costs_list['total_training'][-1] - self.round_costs_list['total_uploads'][-1])}/{self.round_costs_list['total_training'][-1]}")
        print("------------------------------------")

    def optimization(self):
        self.selected_clients, self.rb_allocation, self.user_power_allocation = self.optmizer.opt(self.selected_clients)
