import sys
import numpy as np


class SINR_Opt:
    def __init__(self, communication_strategy):
        self.cs = communication_strategy

    def opt(self, selected_clients):

        C = np.zeros((len(selected_clients), self.cs.tm.rb_number))
        best_k = np.full((len(selected_clients), self.cs.tm.rb_number), -1, dtype=int)

        for i in range(len(selected_clients)):
            idx = selected_clients[i]
            for j in range(self.cs.tm.rb_number):
                feasible_powers = []
                for k in range(len(self.cs.tm.user_power)):
                    if ((self.cs.tm.user_upload_energy[idx, j, k] <= self.cs.energy_requirement and
                         self.cs.tm.user_delay[idx, j, k] <= self.cs.delay_requirement) and
                            self.cs.tm.q[idx, j, k] <= self.cs.error_rate_requirement):
                        feasible_powers.append(k)

                if not feasible_powers:
                    C[i, j] = 999.0
                    best_k[i, j] = -1
                else:
                    chosen_p = 9999.0
                    chosen_k = -1
                    for k2 in feasible_powers:
                        if self.cs.tm.user_power[k2] < chosen_p:
                            chosen_p = self.cs.tm.user_power[k2]
                            chosen_k = k2

                    C[i, j] = self.cs.tm.q[idx, j, chosen_k]
                    best_k[i, j] = chosen_k

        distances = [self.cs.tm.user_distance[key] for key in selected_clients]

        pos_list = np.arange(len(distances))
        combined_data = list(zip(distances, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        _, aux_rb_allocation = zip(*sorted_data)

        assigned_users_list = []
        not_assigned_list = []
        rb_allocation_list = []
        rb_power_list = []

        rb = 0
        for _, idx in enumerate(aux_rb_allocation):

            if best_k[idx, rb] >= 0:
                assigned_users_list.append(selected_clients[idx])
                rb_allocation_list.append(rb)

                indices = list(range(best_k[idx, rb], len(self.cs.tm.user_power)))
                valor_norm = max(0, min((self.cs.tm.user_distance[selected_clients[idx]] - 100) / (500 - 100), 1))
                pos_pw = int(np.round(valor_norm * (len(indices) - 1)))
                #####
                rb_power_list.append(indices[pos_pw])
                rb = rb + 1
            else:
                not_assigned_list.append(selected_clients[idx])

        return assigned_users_list.copy(), rb_allocation_list.copy(), rb_power_list.copy()
