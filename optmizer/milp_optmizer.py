import numpy as np
import pulp as pl
import sys
import re


class Milp_Opt:
    def __init__(self, communication_strategy):
        self.cs = communication_strategy

    def opt(self, selected_clients):

        model = pl.LpProblem("Max_Prob", pl.LpMaximize)
        x = [[pl.LpVariable(f"x_{i}_{j}", cat=pl.LpBinary) for j in
              range(self.cs.tm.rb_number)] for i in range(len(selected_clients))]

        model += pl.lpSum(
            (self.cs.W[selected_clients[i]][j] * x[i][j])
            for j in range(self.cs.tm.rb_number)
            for i in range(len(selected_clients))), "Max"

        model += pl.lpSum([x[i][j] for j in range(self.cs.tm.rb_number) for i in
                           range(len(selected_clients))]) <= self.cs.min_fit_clients, f"min_fit_clients"

        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][j] for k in range(len(self.cs.tm.user_power)) for j in
                              range(self.cs.tm.rb_number)) >= 0, f"Customer_Channel_Constraints_{i} >= 0"

        for i in range(len(selected_clients)):
            model += pl.lpSum(x[i][j] for j in
                              range(self.cs.tm.rb_number)) <= 1, f"Customer_Channel_Constraints_{i} <= 1"

        for j in range(self.cs.tm.rb_number):
            model += pl.lpSum(x[i][j] for i in
                              range(len(selected_clients))) >= 0, f"Channel_Customer_Constraints_{j} >= 0"

        for j in range(self.cs.tm.rb_number):
            model += pl.lpSum(x[i][j] for i in
                              range(len(selected_clients))) <= 1, f"Channel_Customer_Constraints_{j} <= 1"

        for i in range(len(selected_clients)):
            for j in range(self.cs.tm.rb_number):
                k = len(self.cs.tm.user_power) - 1
                model += x[i][j] * self.cs.tm.user_delay[selected_clients[i]][j][
                    k] <= self.cs.delay_requirement, f"Delay_Constraints_{i}_{j}_{k}"
                model += x[i][j] * self.cs.tm.user_upload_energy[selected_clients[i]][j][
                    k] <= self.cs.energy_requirement, f"Energy_Constraints_{i}_{j}_{k}"
                model += x[i][j] * self.cs.tm.q[selected_clients[i]][j][
                    k] <= self.cs.error_rate_requirement, f"Packet_Error_Rate_Constraints_{i}_{j}_{k}"

        # Solving the problem
        status = model.solve()

        _selected_clients = []
        _rb_allocation = []
        for var in model.variables():
            if pl.value(var) == 1:
                indices = [int(i) for i in re.findall(r'\d+', var.name)]
                _selected_clients.append(selected_clients[indices[0]])
                _rb_allocation.append(indices[1])

        C = np.zeros((len(_selected_clients), self.cs.tm.rb_number))
        best_k = np.full((len(_selected_clients), self.cs.tm.rb_number), -1, dtype=int)

        for i in range(len(_selected_clients)):
            idx = _selected_clients[i]
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

        _user_power_allocation = []
        for i in range(len(_selected_clients)):
            idx_min_power = best_k[i, _rb_allocation[i]]
            indices_list = list(range(idx_min_power, len(self.cs.tm.user_power)))
            idx_power = indices_list[int(round(self.cs.lmbda * (len(indices_list) - 1)))]
            _user_power_allocation.append(idx_power)

        return _selected_clients.copy(), _rb_allocation.copy(), _user_power_allocation.copy()
