import numpy as np
from ml_model.ml_model import Model


class Transmission_Model:

    def __init__(self, rb_number, user_number, shape, model_type, lower_limit_distance=100, upper_limit_distance=500,
                 fixed_user_power=0):

        self.rb_number = rb_number
        self.user_number = user_number
        self.shape = shape
        self.model_type = model_type

        if self.model_type == "MLP":
            self.model = Model.create_model_mlp()
        else:
            self.model = Model.create_model_cnn(self.shape)

        self.total_model_params = self.model.count_params()

        self.fixed_user_power = fixed_user_power
        self.lower_limit_distance = lower_limit_distance
        self.upper_limit_distance = upper_limit_distance

        self.data_size_model = 0

        self.user_bandwidth = 1
        self.N = 10 ** -20
        self.q = np.array([])
        self.h = np.array([])

        self.user_power = np.array([])
        self.user_interference = np.array([])
        self.user_distance = np.array([])
        self.user_angles = np.array([])
        #
        self.user_sinr = np.array([])
        self.user_data_rate = np.array([])
        self.user_delay = np.array([])

        self.base_station_power = 1  # W
        self.base_station_bandwidth = 20  # MHz
        #
        self.base_station_sinr = np.array([])
        self.base_station_data_rate = np.array([])
        self.base_station_delay = np.array([])

        self.total_delay = np.array([])

        self.energy_coeff = 10 ** (-27)
        self.cpu_cycles = 40
        self.cpu_freq = 10 ** 9
        self.user_energy_training = np.array([])
        self.user_upload_energy = np.array([])
        self.total_energy = np.array([])

        self.init()

    def init(self):
        self.init_user_interference()
        self.init_distance()
        self.init_user_power()
        self.init_q()
        self.init_h()
        self.init_user_sinr()
        self.init_user_data_rate()
        self.init_base_station_sinr()
        self.init_base_station_data_rate()
        self.init_data_size_model()
        self.init_user_delay()
        self.init_base_station_delay()
        self.init_totaldelay()
        self.init_user_energy_training()
        self.init_user_upload_energy()
        self.init_total_energy()

    def init_user_interference(self):
        i = np.array([0.05 + i * 0.01 for i in range(self.rb_number)])
        self.user_interference = (i - 0.04) * 0.000001

    def init_distance(self):
        np.random.seed(1)
        self.user_distance, self.user_angles = (self.lower_limit_distance + (
                self.upper_limit_distance - self.lower_limit_distance) *
                                                np.random.rand(self.user_number, 1),
                                                2 * np.pi * np.random.rand(self.user_number))
        np.random.seed()

    def init_user_power(self):
        if self.fixed_user_power == 0:
            inc = 0.00025
            self.user_power = np.arange(0.005, 0.01 + inc, inc)
        else:
            self.user_power = np.arange(self.fixed_user_power, 2 * self.fixed_user_power, self.fixed_user_power)

    def init_q(self):
        self.q = 1 - np.exp(-1.08 * (self.user_interference + self.N * self.user_bandwidth)[:, np.newaxis] /
                            (self.user_power * (self.user_distance ** -2)[:, np.newaxis]))

    def init_h(self):
        o = 1
        self.h = o * (self.user_distance ** (-2))

    def init_user_sinr(self):
        self.user_sinr = ((self.user_power * self.h)[:, np.newaxis] /
                          (self.user_interference + self.user_bandwidth * self.N)[:, np.newaxis])

    def init_user_data_rate(self):
        self.user_data_rate = self.user_bandwidth * np.log2(1 + self.user_sinr)

    def init_base_station_sinr(self):
        base_station_interference = 0.06 * 0.000003  # Interference over downlink
        self.base_station_sinr = (self.base_station_power * self.h /
                                  (base_station_interference + self.N * self.base_station_power))

    def init_base_station_data_rate(self):
        self.base_station_data_rate = self.base_station_bandwidth * np.log2(1 + self.base_station_sinr)

    def init_data_size_model(self):
        # MBytes
        self.data_size_model = self.total_model_params * 4 / (1024 ** 2)

    def init_user_delay(self):
        self.user_delay = self.data_size_model / self.user_data_rate

    def init_base_station_delay(self):
        self.base_station_delay = self.data_size_model / self.base_station_data_rate

    def init_totaldelay(self):
        self.total_delay = self.user_delay + self.base_station_delay[:, np.newaxis]

    def init_user_energy_training(self):
        self.user_energy_training = self.energy_coeff * self.cpu_cycles * (self.cpu_freq ** 2) * self.data_size_model

    def init_user_upload_energy(self):
        self.user_upload_energy = self.user_power * self.user_delay

    def init_total_energy(self):
        self.total_energy = self.user_energy_training + self.user_upload_energy
