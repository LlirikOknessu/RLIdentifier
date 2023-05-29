import neptune
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.libs.identificator import LSTMIdentifier
from src.libs.WindowGenerator import WindowGenerator
from src.libs.SimpleWindowGenerator import SimpleWindowGenerator
from pathlib import Path
import pandas as pd

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
    run = neptune.init_run(
        project="kirill.ussenko/System-identification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MDc3N2YyYy0zNGNjLTQwMGQtOTZiOC0zMmMwNTNiMTdmNDEifQ=="
    )

    # Track metadata and hyperparameters by assigning them to fields in the run
    run["JIRA"] = JIRA
    run["algorithm"] = "ConvNet"
    run["parameters"] = PARAMS

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    df_train = pd.read_csv('data/identification_experiment/0_train.csv')
    df_valid = pd.read_csv('data/identification_experiment/0_test.csv')

    x_train = df_train[['u', 'y']]
    y_train = df_train[['k', 't']]
    x_valid = df_valid[['u', 'y']]
    y_valid = df_valid[['k', 't']]

    swg = SimpleWindowGenerator(x_train_df=x_train,
                                x_valid_df=x_valid,
                                y_train_df=y_train,
                                y_valid_df=y_valid,
                                batch_size=PARAMS.get('batch_size', 64))

    print(swg)

    identifier = LSTMIdentifier(checkpoint_path=CHECKPOINT_PATH, params=PARAMS, window=swg)
    identifier.train(epochs=200)

    # Track the training process by logging your training metrics
    run["train/mae"].extend(identifier.history.history['mae'])
    run["train/loss"].extend(identifier.history.history['loss'])
    run["train/validation_mae"].extend(identifier.history.history['val_mae'])
    run["train/validation_loss"].extend(identifier.history.history['val_loss'])

    identifier.load_model()

    # Folder
    run["train/checkpoints"].track_files(str(CHECKPOINT_PATH))

    # Record the final results
    # run["mae"] = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()

    # Stop the connection and synchronize the data with the Neptune servers
    run.stop()
