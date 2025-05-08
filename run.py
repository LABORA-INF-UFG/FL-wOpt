import sys
import numpy as np
import os
from transmission_model.transmission_model import Transmission_Model
from communication_strategy.communication_strategy import Communication_Strategy
from server.server import Server


class FL(Server):

    def __init__(self, n_rounds, total_number_clients, min_fit_clients, rb_number, load_client_data_constructor,
                 path_server, path_clients, shape, model_type, fixed_user_power, optmizer_type,
                 parallel_processing=False):

        self.tm = Transmission_Model(rb_number=rb_number, user_number=total_number_clients,
                               shape=shape,
                               model_type=model_type,
                               lower_limit_distance=100, upper_limit_distance=500,
                               fixed_user_power=fixed_user_power)

        super().__init__(n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                         path_server, path_clients, shape, model_type, optmizer_type, parallel_processing, tm=self.tm)

        self.strategy = Communication_Strategy(
            transmission_model=self.tm,
            min_fit_clients=min_fit_clients,
            clients_number_data_samples=self.clients_number_data_samples,
            clients_value_based_emd=self.clients_value_based_emd,
            emd_mean=self.emd_mean,
            clients_emd=self.clients_emd,
            optmizer_type=optmizer_type,
            delay_requirement=0.2, energy_requirement=0.0025, error_rate_requirement=0.3, lmbda=0.2)

    def print_result(self):
        print("###############################")
        print(f"centralized_accuracy: ")
        print(self.evaluate_list["centralized"]["accuracy"])

        print(f"centralized_loss: ")
        print(self.evaluate_list["centralized"]["loss"])
        print("###############################")

        for item, value in self.strategy.round_costs_list.items():
            print(item)
            print(value)
            print(np.cumsum(value).tolist())

        print("\ncount_of_client_selected")
        print(self.count_of_client_selected)

        print("\ncount_of_client_uploads")
        print(self.count_of_client_uploads)

    def configure_fit(self):
        print(f"self.optmizer_type: {self.optmizer_type}")
        ############

        if self.optmizer_type == "HUN" or self.optmizer_type == "SINR":
            self.strategy.random_user_selection(k=int(self.min_fit_clients))
            self.strategy.optimization()
        else:
            self.strategy.smaller_emd(factor=2, k=int(self.min_fit_clients))
            self.strategy.optimization()

        ################
        self.strategy.upload_status()
        self.strategy.round_costs()
        self.selected_clients = fl.strategy.success_uploads.copy()


if __name__ == "__main__":

    os.system('clear')
    for i in range(1):
        fl = FL(n_rounds=300,
                min_fit_clients=6,
                rb_number=6,
                total_number_clients=100,
                path_server="../Datasets/mnist/mnist",
                path_clients="../Datasets/mnist/non-iid-0.9-100-rotation-45",
                shape=(28, 28, 1),
                model_type="MLP",
                fixed_user_power=0,  # the allocation is dynamic when the value equals zero
                # fixed_user_power=0.01,

                optmizer_type="GA",  # MILP / HUN / SINR / GA
                load_client_data_constructor=False)

        evaluate_loss, evaluate_accuracy = None, None

        for fl.server_round in range(fl.n_rounds):

            # Select customers who will participate in the next round of communication
            fl.configure_fit()

            fl.strategy.print_values()
            print(f"success_uploads: {fl.strategy.success_uploads} - error_uploads: {fl.strategy.error_uploads}")
            fl.strategy.print_round_costs()

            for cid in fl.strategy.selected_clients:
                fl.count_of_client_selected[cid] = fl.count_of_client_selected[cid] + 1

            if len(fl.selected_clients) > 0:

                for cid in fl.selected_clients:
                    fl.count_of_client_uploads[cid] = fl.count_of_client_uploads[cid] + 1

                weight_list, sample_sizes, info = fl.fit()

                # Aggregation
                fl.aggregate_fit(weight_list, sample_sizes)

            print(f"***************************")
            # Centralized evaluate
            print(f"Centralized evaluate: R: {fl.server_round + 1} ")
            evaluate_loss, evaluate_accuracy = fl.centralized_evaluation()
            print(f"evaluate_accuracy: {evaluate_accuracy}")
            print(f"***************************")

        fl.print_result()
