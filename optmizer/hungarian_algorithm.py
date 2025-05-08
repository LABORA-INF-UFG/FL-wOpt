import sys
import numpy as np
from scipy.optimize import linear_sum_assignment


class Hungarian_Algorithm_Opt:
    def __init__(self, communication_strategy):
        self.cs = communication_strategy

    def opt(self, selected_clients):
        C = np.zeros((len(selected_clients), self.cs.tm.rb_number))
        best_k = np.full((len(selected_clients), self.cs.tm.rb_number), -1, dtype=int)

        for i in range(len(selected_clients)):
            idx = selected_clients[i]
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

        # Hungarian algorithm
        assigned_users_list = []
        not_assigned_list = []
        row_ind, col_ind = linear_sum_assignment(C)

        _rb_allocation = []
        _user_power_allocation = []
        total_cost = 0.0
        for idx in range(len(row_ind)):
            i_user = row_ind[idx]
            j_rb = col_ind[idx]
            cost_val = C[i_user, j_rb]
            k_power = best_k[i_user, j_rb]
            total_cost += cost_val

            if k_power < 0:
                not_assigned_list.append(selected_clients[idx])
            else:
                assigned_users_list.append(selected_clients[idx])
                _rb_allocation.append(j_rb)
                _user_power_allocation.append(k_power)

        return assigned_users_list.copy(), _rb_allocation.copy(), _user_power_allocation.copy()
