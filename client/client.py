import sys
import warnings
import numpy as np
import pandas as pd
from client.emd_compute import EMD_Compute
from ml_model.ml_model import Model
import tensorflow as tf
import ot


class Client:
    def __init__(self, cid, load_data_constructor, path, shape, model_type, optmizer_type):
        self.cid = int(cid)
        self.load_data_constructor = load_data_constructor

        self.path = path
        self.shape = shape
        self.model_type = model_type
        self.fit_type = 'default' if optmizer_type != 'SINR' else 'Moon'
        self.emd_cmp = EMD_Compute()

        if self.model_type == "MLP":
            self.model = Model.create_model_mlp()
            self.previous_model = Model.create_model_mlp()  # Modelo local anterior (Moon)
            self.global_model = Model.create_model_mlp()  # Modelo local anterior (Moon)
        else:
            self.model = Model.create_model_cnn(self.shape)
            self.previous_model = Model.create_model_mlp()
            self.global_model = Model.create_model_mlp()

        self.previous_model.set_weights(self.model.get_weights())
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy()

        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        self.data_size = len(self.y_train)
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        self.batch_size = 128

    def load_data(self):

        train = pd.read_pickle(f"{self.path}/{self.cid}_train.pickle")
        test = pd.read_pickle(f"{self.path}/{self.cid}_test.pickle")

        x_train = train.drop(['label'], axis=1)
        y_train = train['label']

        x_test = test.drop(['label'], axis=1)
        y_test = test['label']

        if self.model_type == "CNN":
            x_train = np.array([x.reshape(self.shape) for x in x_train.reset_index(drop=True).values])
            x_test = np.array([x.reshape(self.shape) for x in x_test.reset_index(drop=True).values])
        return (x_train, y_train), (x_test, y_test)

    def compute_emd_aux(self):
        value_emd = self.emd_cmp.compute_value(self.y_train.values)
        return value_emd

    def number_data_samples(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        return len(self.x_train)

    def fit_default(self, parameters, config=None):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=self.batch_size,
                                 # batch_size=len(self.x_train)
                                 validation_data=(self.x_test, self.y_test), verbose=False)
        sample_size = len(self.x_train)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return self.model.get_weights(), sample_size, {"val_accuracy": history.history['val_accuracy'][-1],
                                                       "val_loss": history.history['val_loss'][-1]}

    def fit_fed_prox(self, parameters, config=None):
        _mu = 0.01

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model.set_weights(parameters)

        batch_size = self.batch_size
        num_batches = len(self.x_train) // batch_size

        for epoch in range(1):  # Fixed number of epochs
            batch_count = 0
            for i in range(num_batches):
                x_batch = self.x_train[i * batch_size:(i + 1) * batch_size]
                y_batch = self.y_train[i * batch_size:(i + 1) * batch_size]

                with tf.GradientTape() as tape:
                    x_batch = tf.reshape(x_batch, (-1, 784))
                    predictions = self.model(x_batch, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)

                    # FedProx term: L2 distance between local and global weights
                    prox_term = sum(
                        tf.reduce_sum(tf.square(w1 - w2)) for w1, w2 in zip(self.model.trainable_weights, parameters))
                    loss = loss + (_mu / 2) * prox_term

                grads = tape.gradient(loss, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                batch_count += 1
                if batch_count >= num_batches:
                    break

            sample_size = len(self.x_train)
            loss, accuracy = self.evaluate(self.model.get_weights())

            if not self.load_data_constructor:
                (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

            return self.model.get_weights(), sample_size, {"val_accuracy": accuracy, "val_loss": loss}

    def fit(self, parameters, config=None):
        if self.fit_type == 'default':
            return self.fit_default(parameters, config)
        else:
            return self.fit_fed_prox(parameters, config)

    def evaluate(self, parameters):
        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)

        if not self.load_data_constructor:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

        return loss, accuracy
