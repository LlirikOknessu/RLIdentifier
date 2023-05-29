import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from src.libs.SimpleWindowGenerator import SimpleWindowGenerator

tf.keras.backend.set_floatx('float64')

EPSILON = 1e-6


class BaseIdentifier:

    def __init__(self, checkpoint_path: Path, params=None):
        if params is None:
            params = {}
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.params = params
        self.model_checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True)
        self.early_stopping = keras.callbacks.EarlyStopping(patience=self.params.get('early_stopping', 50))
        self.optimizer = keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 2))
        self.history = None

    def load_model(self, input_path: Path = None):
        if input_path is None:
            input_path = self.checkpoint_path
        self.model = keras.models.load_model(str(input_path))

class CNNIdentifier(BaseIdentifier):

    def __init__(self, checkpoint_path: Path, params=None):
        super().__init__(checkpoint_path, params)
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.InputLayer(input_shape=[None, 1]))
        for dilation_rate in (1, 2, 4, 8, 16, 32):
            self.model.add(
                keras.layers.Conv1D(filters=self.params.get('filters', 32),
                                    kernel_size=self.params.get('kernel_size', 2),
                                    strides=1,
                                    dilation_rate=dilation_rate,
                                    padding="causal",
                                    activation="relu")
            )
        self.model.add(keras.layers.Conv1D(filters=2, kernel_size=1))
        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=self.optimizer,
                           metrics=["mae"])

    def prepare_dataset(self, x_train: np.ndarray, x_valid: np.ndarray, window_size: int = 128):
        def seq2seq_window_dataset(series: np.ndarray, window_size: int, batch_size: int = 32,
                                   shuffle_buffer: int = 1000):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size + 1))
            ds = ds.shuffle(shuffle_buffer)
            ds = ds.map(lambda w: (w[:-1], w[1:]))
            return ds.batch(batch_size).prefetch(1)

        train_set = seq2seq_window_dataset(x_train, window_size,
                                           batch_size=self.params.get('batch_size', 128))
        valid_set = seq2seq_window_dataset(x_valid, window_size,
                                           batch_size=self.params.get('batch_size', 128))
        return train_set, valid_set

    def train(self, train_set, valid_set, epochs=500):
        self.history = self.model.fit(train_set, epochs=epochs,
                                      validation_data=valid_set,
                                      callbacks=[self.early_stopping, self.model_checkpoint])


class DenseIdentifier(BaseIdentifier):
    def __init__(self, checkpoint_path: Path, window: SimpleWindowGenerator, params=None):
        super().__init__(checkpoint_path, params)
        self.batch_size = self.params.get('batch_size', 128)
        self.train_ds = window.train_dataset
        self.valid_ds = window.test_dataset

        inputs = keras.Input(shape=(window.sequence_length, window.features))
        x = tf.keras.layers.LSTM(units=128, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(units=128, return_sequences=True)(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(activation='relu', units=32)
        )(x)
        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(activation='linear', units=window.labels)
        )(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTMRaw')

        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=self.optimizer,
                           metrics=["mae"])

    def train(self, epochs=500):

        history = self.model.fit(self.train_ds,
                                 validation_data=self.valid_ds,
                                 epochs=epochs,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 callbacks=[self.early_stopping])
        return history


