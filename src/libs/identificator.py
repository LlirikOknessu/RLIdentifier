import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

tf.keras.backend.set_floatx('float64')

EPSILON = 1e-6


class Identifier:

    def __init__(self, checkpoint_path=Path):
        self.checkpoint_path = checkpoint_path
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.InputLayer(input_shape=[None, 1]))
        for dilation_rate in (1, 2, 4, 8, 16, 32):
            self.model.add(
                keras.layers.Conv1D(filters=32,
                                    kernel_size=2,
                                    strides=1,
                                    dilation_rate=dilation_rate,
                                    padding="causal",
                                    activation="relu")
            )
        self.model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
        self.optimizer = keras.optimizers.Adam(lr=3e-4)
        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=self.optimizer,
                           metrics=["mae"])

        self.model_checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True)
        self.early_stopping = keras.callbacks.EarlyStopping(patience=50)
        self.history = None

    @staticmethod
    def prepare_dataset(x_train: np.ndarray, x_valid: np.ndarray, window_size: int = 128):
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
                                           batch_size=128)
        valid_set = seq2seq_window_dataset(x_valid, window_size,
                                           batch_size=128)
        return train_set, valid_set

    def train(self, train_set, valid_set, epochs=500):
        self.history = self.model.fit(train_set, epochs=epochs,
                                      validation_data=valid_set,
                                      callbacks=[self.early_stopping, self.model_checkpoint])

    def load_model(self, input_path: Path = None):
        if input_path is None:
            input_path = self.checkpoint_path
        self.model = keras.models.load_model(str(input_path))
