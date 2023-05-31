import neptune
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.libs.identificator import Identifier
from pathlib import Path

JIRA = 'IDS-001'

CHECKPOINT_PATH = Path('data/checkpoints') / JIRA
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


time = np.arange(4 * 365 + 1)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

# Create a Neptune run object
run = neptune.init_run(
    project="kirill.ussenko/System-identification",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MDc3N2YyYy0zNGNjLTQwMGQtOTZiOC0zMmMwNTNiMTdmNDEifQ=="
)

# Folder
run["train/checkpoints"].track_files(str(CHECKPOINT_PATH))

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Track metadata and hyperparameters by assigning them to fields in the run
run["JIRA"] = JIRA
run["algorithm"] = "ConvNet"

PARAMS = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam",
}

run["parameters"] = PARAMS

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 64
identifier = Identifier(checkpoint_path=CHECKPOINT_PATH)
train_set, valid_set = identifier.prepare_dataset(x_train=x_train, x_valid=x_valid)
identifier.train(train_set=train_set, valid_set=valid_set, epochs=500)

# Track the training process by logging your training metrics
run["train/mae"].extend(identifier.history.history['mae'])
run["train/loss"].extend(identifier.history.history['loss'])
run["train/validation_mae"].extend(identifier.history.history['val_mae'])
run["train/validation_loss"].extend(identifier.history.history['val_loss'])

identifier.load_model()

cnn_forecast = model_forecast(identifier.model, series[..., np.newaxis], window_size)
cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]

# Record the final results
run["mae"] = keras.metrics.mean_absolute_error(x_valid, cnn_forecast).numpy()

# Stop the connection and synchronize the data with the Neptune servers
run.stop()
