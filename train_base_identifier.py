import neptune
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.libs.identificator import DenseIdentifier
from src.libs.WindowGenerator import WindowGenerator
from pathlib import Path
import pandas as pd


tf.debugging.set_log_device_placement(True)

JIRA = 'IDS-003'
PARAMS = {
    "batch_size": 64,
    "filters": 32,
    "kernel_size": 2,
    "learning_rate": 3e-4,
    "optimizer": "Adam",
    "early_stopping": 50
}

CHECKPOINT_PATH = Path('data/checkpoints') / JIRA
OUTPUT_DATA_FOLDER = Path('data/identification_experiment/')


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == "__main__":
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    # Create a Neptune run object
    # run = neptune.init_run(
    #     project="kirill.ussenko/System-identification",
    #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MDc3N2YyYy0zNGNjLTQwMGQtOTZiOC0zMmMwNTNiMTdmNDEifQ=="
    # )
    # 
    # # Folder
    # run["train/checkpoints"].track_files(str(CHECKPOINT_PATH))
    # 
    # # Track metadata and hyperparameters by assigning them to fields in the run
    # run["JIRA"] = JIRA
    # run["algorithm"] = "ConvNet"
    # run["parameters"] = PARAMS

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    df_train = pd.read_csv('data/identification_experiment/0_train.csv')
    df_valid = pd.read_csv('data/identification_experiment/0_test.csv')

    x_train = df_train[['u', 'y', 'k', 't']]
    x_valid = df_valid[['u', 'y', 'k', 't']]

    wg = WindowGenerator(input_width=63,
                         label_width=1,
                         shift=1,
                         label_columns=['k', 't'],
                         train_df=df_train,
                         val_df=df_valid)
    print(wg)

    t = wg.train
    window_size = 64
    for samples, targets in t:
        print("samples shape:", samples.shape)
        print("targets shape:", targets.shape)
        break

    identifier = DenseIdentifier(checkpoint_path=CHECKPOINT_PATH, params=PARAMS)
    identifier.train(wg, epochs=500)

    # Track the training process by logging your training metrics
    # run["train/mae"].extend(identifier.history.history['mae'])
    # run["train/loss"].extend(identifier.history.history['loss'])
    # run["train/validation_mae"].extend(identifier.history.history['val_mae'])
    # run["train/validation_loss"].extend(identifier.history.history['val_loss'])

    identifier.load_model()

    cnn_forecast = model_forecast(identifier.model, series[..., np.newaxis], window_size)
    cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]

    # Record the final results
    # run["mae"] = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()
    #
    # # Stop the connection and synchronize the data with the Neptune servers
    # run.stop()
    #