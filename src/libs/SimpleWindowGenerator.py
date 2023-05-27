import numpy as np
import tensorflow as tf


class SimpleWindowGenerator:
    def __init__(self, x_train_df, y_train_df, x_valid_df, y_valid_df,
                 batch_size: int = 32, sequence_length: int = 100):
        # Store the raw data.
        x_train = x_train_df.to_numpy()
        y_train = y_train_df.to_numpy()
        x_valid = x_valid_df.to_numpy()
        y_valid = y_valid_df.to_numpy()

        self.sequence_length = sequence_length

        if x_train.shape[0] % sequence_length:
            x_valid = x_valid[:-(x_valid.shape[0] % sequence_length)]
            y_valid = y_valid[:-(y_valid.shape[0] % sequence_length)]

        if x_valid.shape[0] % sequence_length:
            x_valid = x_valid[:-(x_valid.shape[0] % sequence_length)]
            y_valid = y_valid[:-(y_valid.shape[0] % sequence_length)]

        train_total_size = int(x_train.shape[0] / sequence_length)
        valid_total_size = int(x_valid.shape[0] / sequence_length)

        self.features = x_train.shape[1]
        self.labels = y_train.shape[1]

        x_train = np.reshape(x_train, (train_total_size, sequence_length, self.features))
        y_train = np.reshape(y_train, (train_total_size, sequence_length, self.labels))
        x_valid = np.reshape(x_valid, (valid_total_size, sequence_length, self.features))
        y_valid = np.reshape(y_valid, (valid_total_size, sequence_length, self.labels))

        train_features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

        test_features_dataset = tf.data.Dataset.from_tensor_slices(x_valid)
        test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_valid)

        train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
        test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

        self.train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = test_dataset.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

    def __repr__(self):
        return '\n'.join([
            f'Train shape: {self.train_dataset}',
            f'Validation shape: {self.test_dataset}'])
