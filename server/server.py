import numpy as np
import pandas as pd
from ml_model.ml_model import Model
from client.client import Client
import sys


class Server:
    def __init__(self, n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                 path_server, path_clients, shape, model_type, optmizer_type, parallel_processing=False, tm=None):
        self.tm = tm
        self.n_rounds = n_rounds
        self.total_number_clients = total_number_clients
        self.min_fit_clients = min_fit_clients
        self.load_client_data_constructor = load_client_data_constructor
        self.parallel_processing = parallel_processing

        self.path_server = path_server
        self.path_clients = path_clients
        self.shape = shape
        self.model_type = model_type
        self.optmizer_type = optmizer_type

        self.server_round = 0
        self.selected_clients = []

        if self.model_type == "MLP":
            self.model = Model.create_model_mlp()
        else:
            self.model = Model.create_model_cnn(self.shape)

        self.w_global = self.model.get_weights()

        self.clients_model_list = []
        self.clients_number_data_samples = []
        self.clients_value_based_emd = []
        self.clients_emd = []
        self.clients_sinr_avg = []

        self.clients_acc = []
        self.clients_loss = []

        self.count_of_client_selected = []
        self.count_of_client_uploads = []

        self.evaluate_list = {"distributed": {"loss": [], "accuracy": []}, "centralized": {"loss": [], "accuracy": []}}

        self.create_models()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        self.emd_mean = np.mean(self.clients_emd)

    def create_models(self):
        total_data = 0
        for i in range(self.total_number_clients):
            c = Client(i + 1, self.load_client_data_constructor, self.path_clients, self.shape, self.model_type,
                       self.optmizer_type)
            total_data = total_data + c.number_data_samples()

        for i in range(self.total_number_clients):
            self.clients_model_list.append(
                Client(i + 1, self.load_client_data_constructor, self.path_clients, self.shape, self.model_type,
                       self.optmizer_type))
            self.clients_number_data_samples.append(self.clients_model_list[i].number_data_samples())
            self.clients_emd.append(self.clients_model_list[i].compute_emd_aux())

            self.clients_sinr_avg.append(np.mean(self.tm.user_sinr[i]))
            self.clients_acc.append(0)
            self.clients_loss.append(np.inf)
            self.count_of_client_selected.append(0)
            self.count_of_client_uploads.append(0)

        for i in range(self.total_number_clients):
            user_snr_value = self.min_max_normalize(np.min(1 / np.array(self.clients_sinr_avg)),
                                                    np.max(1 / np.array(self.clients_sinr_avg)),
                                                    1 / self.clients_sinr_avg[i])
            data_value = 0
            emd_value = self.min_max_normalize(np.min(self.clients_emd), np.max(self.clients_emd), self.clients_emd[i])
            value_based_emd = emd_value + data_value + user_snr_value
            self.clients_value_based_emd.append(value_based_emd)

    def load_data(self):
        train = pd.read_pickle(f"{self.path_server}/train.pickle")
        test = pd.read_pickle(f"{self.path_server}/test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']
        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.model_type == "CNN":
            x_train = np.array([x.reshape(self.shape) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.shape) for x in x_test.reset_index(drop=True).values])

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def min_max_normalize(min_value, max_value, x):
        return (x - min_value) / (max_value - min_value) + 1e-8

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def aggregate_fit(self, parameters, sample_sizes):

        if self.optmizer_type == "HUN" or self.optmizer_type == "SINR":
            self.aggregate_fit_cmp(parameters, sample_sizes)
        else:
            emd_list = np.array(self.clients_emd)[self.selected_clients]
            sample_sizes = sample_sizes * (1 / emd_list)
            self.aggregate_fit_cmp(parameters, sample_sizes)

    def aggregate_fit_cmp(self, parameters, sample_sizes):
        self.w_global = []
        for weights in zip(*parameters):
            weighted_sum = 0
            total_samples = sum(sample_sizes)
            for i in range(len(weights)):
                weighted_sum += weights[i] * sample_sizes[i]
            self.w_global.append(weighted_sum / total_samples)

    def configure_fit(self):
        self.selected_clients = np.random.permutation(list(range(self.total_number_clients)))[:self.min_fit_clients]

    def fit(self):

        weight_list = []
        sample_sizes_list = []
        info_list = []

        if self.parallel_processing:
            pass
        else:

            for i, pos in enumerate(self.selected_clients):
                print(f"-------> [{i + 1}] (R: {self.server_round + 1}/{self.n_rounds}) CID: {pos}")
                weights, size, info = self.clients_model_list[pos].fit(parameters=self.w_global)
                weight_list.append(weights)
                sample_sizes_list.append(size)
                info_list.append(info)
                self.clients_acc[pos] = info['val_accuracy']
                self.clients_loss[pos] = info['val_loss']

        return weight_list, sample_sizes_list, {
            "acc_loss_local": [(pos + 1, info_list[i]) for i, pos in enumerate(self.selected_clients)]}

    def distributed_evaluation(self):

        loss_list = []
        accuracy_list = []

        if self.parallel_processing:
            pass
        else:
            for i in range(self.total_number_clients):
                loss, accuracy = self.clients_model_list[i].evaluate(parameters=self.w_global)
                loss_list.append(loss)
                accuracy_list.append(accuracy)

        loss = sum(loss_list) / len(loss_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)
        self.evaluate_list["distributed"]["loss"].append(loss)
        self.evaluate_list["distributed"]["accuracy"].append(accuracy)

        return loss, accuracy, {"accuracy_list": [(i + 1, accuracy) for i, accuracy in enumerate(accuracy_list)]}

    def centralized_evaluation(self):
        self.model.set_weights(self.w_global)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)
        self.evaluate_list["centralized"]["loss"].append(loss)
        self.evaluate_list["centralized"]["accuracy"].append(accuracy)
        return loss, accuracy
